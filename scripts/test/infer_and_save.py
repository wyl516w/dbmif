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
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # noqa: E402
from module.datamodule.temporal_transforms import TemporalTransforms
from module.utils.utils import initialize_model_from_dict, load_from_file


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WavDataset(Dataset):
    """Recursively collect all (wav|mp3|flac) files under `input_dir`."""

    AUDIO_EXTS = {".wav", ".mp3", ".flac"}

    def __init__(self, input_dir: str | Path):
        self.base = Path(input_dir)
        self.files = sorted(p for p in self.base.rglob("*") if p.suffix.lower() in self.AUDIO_EXTS)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        path = self.files[idx]
        rel = path.relative_to(self.base).as_posix()
        audio, sr = torchaudio.load(str(path))  # (C, T)

        # TemporalTransforms handles proper copying / slicing for us
        tt = TemporalTransforms(audio, sr, dims={"ac": [0], "noise_ac": [1], "bc": [2]})

        return {
            "rel": rel,
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
    sf.write(str(out_path), wav_np, sr)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def run_inference(
    model_cfg: Path,
    ckpt_path: Path,
    input_dir: Path,
    output_dir: Path,
    tag_name: str,
    crop: bool,
    num_workers: int,
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
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False,
    )

    crop_len = 16000 if crop else 0
    use_amp = device.type == "cuda" and torch.cuda.is_available()

    # ---- 3. Inference loop ------------------------------------------------- #
    for sample in tqdm(loader, total=len(loader.dataset), unit="file"):
        rel = sample["rel"][0]
        sr = int(sample["sr"][0])
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
            tag_name: enhanced.squeeze(0),
        }
        for tag, wav in mapping.items():
            save_track(wav, sr, output_dir / tag / rel)

    print("✓ All files processed.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    All value-bearing options are required; only the boolean flag --crop is
    optional.  Short aliases are provided for every option.
    """
    parser = argparse.ArgumentParser(
        description="Run a speech-enhancement model on a folder of audio files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="YAML file that defines the model architecture / preprocessing"
    )
    parser.add_argument(
        "-k", "--ckpt",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint (.ckpt or .pth)"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input directory that contains wav/mp3/flac files to enhance"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output root where clean/noisy/bc/enhanced tracks will be written",
    )
    parser.add_argument(
        "-n", "--name",
        required=True,
        help="Sub-folder name for the enhanced track (e.g., model tag)",
    )
    parser.add_argument(
        "--crop", "-p",
        action="store_true",
        help="Chunk input into 1-second (16 000-sample) blocks for inference",
    )
    parser.add_argument(
        "-w", "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for the DataLoader",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_cfg=args.config,
        ckpt_path=args.ckpt,
        input_dir=args.input,
        output_dir=args.output,
        tag_name=args.name,
        crop=args.crop,
        num_workers=args.num_workers,
    )
