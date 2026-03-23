#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a speech-enhancement model on <input_dir> (wav/mp3/flac),
create clean / noisy / bc / enhanced tracks, and write them under <output_dir>.

Audio layout inside each input file is assumed to be:
  ch-0: clean AC
  ch-1: noisy  AC
  ch-2: BC
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import soundfile as sf
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

from module.datamodule.temporal_transforms import TemporalTransforms
from module.utils.utils import initialize_model_from_dict, load_from_file


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WavDataset(Dataset):
    """Recursively collect all (wav|mp3|flac) files under `input_dir`."""

    AUDIO_EXTS = {".wav", ".mp3", ".flac"}

    def __init__(self, input_dir: str | Path):
        self.files = [str(input_dir)]
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        path = self.files[idx]
        audio, sr = torchaudio.load(path)  # (C, T)

        # TemporalTransforms handles proper copying / slicing for us
        tt = TemporalTransforms(audio, sr, dims={"ac": [0], "noise_ac": [1], "bc": [2]})

        return {
            "sr": sr,
            "clean_ac": tt.audio("ac"),  # (1, T)
            "noise_ac": tt.audio("noise_ac"),  # (1, T)
            "bc": tt.audio("bc"),  # (1, T)
        }


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def chunk_signal(x: torch.Tensor, length: int) -> torch.Tensor:
    """
    Pad (B, 1, T) to a multiple of `length` and reshape to
    (B * n_chunks, 1, length) for batched inference.
    """
    if length <= 0:
        return x
    B, C, T = x.shape
    pad = (length - T % length) % length
    x = F.pad(x, (0, pad))
    return x.view(B, C, -1, length).transpose(1, 2).reshape(-1, C, length)


def unchunk_signal(x: torch.Tensor, B: int, C: int, length: int) -> torch.Tensor:
    """Inverse of `chunk_signal` (assumes the last dimension is `length`)."""
    return x.view(B, -1, C, length).transpose(1, 2).reshape(B, C, -1)


def save_track(wav: torch.Tensor, sr: int, out_path: Path) -> None:
    """Write mono/stereo tensor to a wav file, creating parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav_np = wav.detach().cpu().float().numpy()
    if wav_np.ndim == 2:  # (C, T) -> (T, C) for soundfile
        wav_np = wav_np.T
    sf.write(out_path, wav_np, sr)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def run_inference(
    model_cfg: Path,
    ckpt_path: Path,
    input_dir: Path,
    crop: bool,
) -> None:
    # ---- 1. Load model ----------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_from_file(model_cfg)
    model = initialize_model_from_dict(cfg["model"], cfg.get("metrics"))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    # ---- 2. DataLoader ----------------------------------------------------- #
    loader = DataLoader(
        WavDataset(input_dir),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )

    crop_len = 16000 if crop else 0
    use_amp = device.type == "cuda" and torch.cuda.is_available()

    # ---- 3. Inference loop ------------------------------------------------- #
    for sample in tqdm(loader, total=len(loader.dataset), unit="file"):
        clean_ac = sample["clean_ac"][0]
        noise_ac = sample["noise_ac"][0]
        bc = sample["bc"][0]

        # keep tidy shapes for chunk/un-chunk utility
        clean_ac = clean_ac.unsqueeze(0)  # (B=1, 1, T)
        noise_ac = noise_ac.unsqueeze(0)
        bc = bc.unsqueeze(0)

        # 3-A. Optional chunking
        if crop_len:
            noise_chunks = chunk_signal(noise_ac, crop_len)
            bc_chunks = chunk_signal(bc, crop_len)
        else:
            noise_chunks, bc_chunks = noise_ac, bc

        # 3-B. GPU forward
        noise_chunks = noise_chunks.to(device, non_blocking=True)
        bc_chunks = bc_chunks.to(device, non_blocking=True)

        with torch.autocast(device.type) if use_amp else torch.no_grad():
            enhanced_chunks = model(noise_chunks, bc_chunks)
            if isinstance(enhanced_chunks, (tuple, list)):
                enhanced_chunks = enhanced_chunks[0]

        # 3-C. Stitch back together
        if crop_len:
            enhanced = unchunk_signal(enhanced_chunks, 1, 1, crop_len)
        else:
            enhanced = enhanced_chunks

        # make sure length matches original noisy track
        T = noise_ac.shape[-1]
        enhanced = enhanced[..., :T]
        bc = bc[..., :T]
        clean_ac = clean_ac[..., :T]

        # 3-D. Save all tracks
        mapping = {
            "clean": clean_ac.squeeze(0),
            "ac": noise_ac.squeeze(0),
            "bc": bc.squeeze(0),
        }
        break
    print("✓ All files processed.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run model inference on a folder of wavs")
    p.add_argument("-c", "--config", type=Path, default="config/model_config/DBMIF.yaml")
    p.add_argument("-k", "--ckpt", type=Path, default="logs/experiments/Proposed-2025-07-07_01-27-22/version_0/checkpoints/epoch=199-step=498000.ckpt")
    p.add_argument("-i", "--input", type=Path, default="dataset/gendata/NoiseX-92/singlesnr/snr_-15/Speaker5_C_49.white.wav")
    p.add_argument("--crop", action="store_true", help="Chunk input at 1-s (16k) blocks")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_cfg=args.config,
        ckpt_path=args.ckpt,
        input_dir=args.input,
        crop=args.crop,
    )
