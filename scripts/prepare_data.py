#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate AB-Noise fusion test sets with optional SNR control.

Output tree (example)
├── gendata
│   └── QUT-NOISE
│       ├── noise
│       │   ├── CAFE-FOODCOURTB
│       │   │   ├── *.wav
│       │   │   └── groundtruth.txt
│       │   └── ...
│       ├── singlesnr
│       │   ├── snr_-15
│       │   └── ...
│       └── intervalsnr
│           ├── snr_-15_-10
│           └── ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Local imports (add project root to sys.path once)
# -----------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from module.datamodule import AB_Dataset, AB_N_Fusion_Dataset, NoiseDataset  # noqa: E402


# -----------------------------------------------------------------------------
# Dataloader helpers
# -----------------------------------------------------------------------------
def make_test_loader(
    noise_files: Iterable[str],
    snr: list[int] | tuple[int, int],
    data_root: Path,
    noise_root: Path,
    workers: int,
) -> DataLoader:
    """Return a 1-sample loader that fuses AB speech with the given noise."""
    # ↓↓↓ plain AB speech
    ab = AB_Dataset(
        path=data_root,
        folder="Audio/test",
        sr_standard=16000,
        separator="_",
        deterministic=False,
        is_smoothing=True,
        return_file_name=True,
        len_seconds=None,  # 使用全长
    )

    # ↓↓↓ noise
    keys = [Path(f).with_suffix("").as_posix() for f in noise_files]
    key2path = {k: noise_root / f for k, f in zip(keys, noise_files)}

    noise_ds = NoiseDataset(
        keys=keys,
        key_to_path=key2path,
        sr_standard=16000,
        keep_data_in_memory=True,
        return_file_name=True,
    )

    # ↓↓↓ fusion (AB + noise)
    fusion_ds = AB_N_Fusion_Dataset(
        ab_dataset=ab,
        noise_dataset=noise_ds,
        snr=snr,
        ratio=5.0,
        keep_clean_data=False,
        is_smoothing=True,
        return_file_name=True,
    )

    return DataLoader(
        fusion_ds,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=False,
    )


# -----------------------------------------------------------------------------
# Saving utilities
# -----------------------------------------------------------------------------
@torch.no_grad()
def save_dataset(
    loader: DataLoader,
    out_dir: Path,
    max_items: int,
    trans_map: dict[str, str],
    append_gt: bool = False,  # ← 新增：groundtruth 是否追加
) -> None:
    """Write <ac+noise+bc>.wav and optional ground-truth txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "groundtruth.txt"
    # 若选择保留目录且文件已存在 → 追加写入；否则覆盖写入
    gt_mode = "a" if append_gt and gt_path.exists() else "w"
    fout = gt_path.open(gt_mode, encoding="utf-8")

    written, skipped = 0, 0
    bar = tqdm(total=max_items, desc=f"→ {out_dir.name}", ncols=100)

    for ac, nc, bc, fname, nname in loader:
        if written >= max_items:
            break

        wav_path = out_dir / f"{fname[0]}.{nname[0]}.wav"
        if wav_path.exists():
            skipped += 1
            if skipped > max_items:
                tqdm.write(f"[warn] Too many existing files in {out_dir}")
                break
            continue
        skipped = 0

        # save audio
        audio = torch.cat([ac, nc, bc], dim=0).squeeze().t().cpu().numpy()
        sf.write(str(wav_path), audio, 16000)

        # write transcription
        key = fname[0].split(".", 1)[0]
        if key in trans_map:
            rel = wav_path.relative_to(out_dir.parent)
            fout.write(f"{rel.as_posix()} {trans_map[key]}\n")

        written += 1
        bar.update(1)

    bar.close()
    fout.close()


def load_trans_map(script_dir: Path) -> dict[str, str]:
    """Aggregate all utt-id → text mappings under a script folder."""
    mapping: dict[str, str] = {}
    for txt in script_dir.rglob("*.txt"):
        with txt.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if parts and len(parts) == 2:
                    mapping[parts[0]] = parts[1]
    return mapping


# -----------------------------------------------------------------------------
# Noise file helpers
# -----------------------------------------------------------------------------
def gather_noise_files(noise_root: Path, group: str) -> list[str]:
    """Return *.wav list for a given noise group."""
    if group == "Nonspeech":
        return [str(p.relative_to(noise_root)) for p in noise_root.rglob("*.wav")]

    all_wavs = list(noise_root.rglob("*.wav"))
    if group == "QUT-NOISE":
        return [str(p.relative_to(noise_root)) for p in all_wavs if "-" in p.name]

    # default: every wav
    return [str(p.relative_to(noise_root)) for p in all_wavs]


def list_noise_classes(noise_root: Path, group: str) -> list[str]:
    """Extract per-class names (used in 'noise' mode)."""
    if group == "Nonspeech":
        return ["nonspeech"]

    if group == "QUT-NOISE":
        stems = ("-".join(p.stem.split("-")[:2]) for p in noise_root.rglob("*.wav") if "-" in p.name)
        return sorted(set(stems))

    return [p.stem for p in noise_root.rglob("*.wav")]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AB-Noise fusion test data")
    parser.add_argument("--data_name", nargs="+", default=["ABCS_database"])
    parser.add_argument("--noise_name", nargs="+", default=["NoiseX-92", "Nonspeech", "QUT-NOISE"])
    parser.add_argument("--data_dir", default="../dataset/abdata")
    parser.add_argument("--noise_dir", default="../dataset/noisedata")
    parser.add_argument("--output_dir", default="../dataset/gendata")
    parser.add_argument("--max_num", type=int, default=200)
    parser.add_argument("--snr_start", type=int, default=-15)
    parser.add_argument("--snr_end", type=int, default=5)
    parser.add_argument("--snr_step", type=int, default=1)
    parser.add_argument("--interval_width", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=["noise", "singlesnr", "intervalsnr"])
    parser.add_argument("--num_workers", default="0")
    parser.add_argument(
        "--no_clear",
        action="store_true",
        help="Do NOT delete existing output_dir if present; keep files and append groundtruth.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    workers = os.cpu_count() if args.num_workers == "auto" else int(args.num_workers)
    print(f"DataLoader workers = {workers}")

    singles = list(range(args.snr_start, args.snr_end + 1, args.snr_step))
    intervals = [(lo, min(lo + args.interval_width, args.snr_end)) for lo in range(args.snr_start, args.snr_end, args.interval_width)]

    out_root = Path(args.output_dir)
    if out_root.exists():
        if args.no_clear:
            print(f"[keep] output_dir exists → keeping: {out_root}")
        else:
            print(f"[clear] removing existing output_dir: {out_root}")
            shutil.rmtree(out_root)

    for data_name in args.data_name:
        data_root = Path(args.data_dir) / data_name
        trans_map = load_trans_map(data_root)
        print(f"=== Dataset: {data_name} ({len(trans_map):,} transcriptions) ===")

        for noise_grp in args.noise_name:
            noise_root = Path(args.noise_dir) / noise_grp
            print(f"-- Noise source: {noise_grp}")

            all_noise_files = gather_noise_files(noise_root, noise_grp)
            noise_classes = list_noise_classes(noise_root, noise_grp)

            for mode in args.modes:
                print(f"   » Mode: {mode}")

                if mode == "noise":
                    for cls in noise_classes:
                        if noise_grp == "QUT-NOISE":
                            files = [f for f in all_noise_files if f.startswith(f"{cls}-")]
                        elif noise_grp == "Nonspeech":
                            files = all_noise_files
                        else:
                            files = [f for f in all_noise_files if f.startswith(cls)]

                        loader = make_test_loader(files, (args.snr_start, args.snr_end), data_root, noise_root, workers)
                        out_dir = out_root / noise_grp / "noise" / cls
                        save_dataset(loader, out_dir, args.max_num, trans_map, append_gt=args.no_clear)

                if mode == "singlesnr":
                    for snr in singles:
                        loader = make_test_loader(all_noise_files, [snr], data_root, noise_root, workers)
                        out_dir = out_root / noise_grp / "singlesnr" / f"snr_{snr}"
                        save_dataset(loader, out_dir, args.max_num, trans_map, append_gt=args.no_clear)

                if mode == "intervalsnr":
                    for lo, hi in intervals:
                        loader = make_test_loader(all_noise_files, (lo, hi), data_root, noise_root, workers)
                        out_dir = out_root / noise_grp / "intervalsnr" / f"snr_{lo}_{hi}"
                        save_dataset(loader, out_dir, args.max_num, trans_map, append_gt=args.no_clear)

    print("All datasets generated ✔")


if __name__ == "__main__":
    main()
