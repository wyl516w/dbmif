#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute speech-quality metrics for each enhanced .wav file.

You can choose one or more metric groups:
    classic → SI-SDR, PESQ, STOI, eSTOI
    dns     → P.808 MOS, MOS_SIG, MOS_BAK, MOS_OVR
    origin  → classic + MSE

Examples
--------
# Compute both classic and dns metrics and save to all.csv
python compute_metrics.py -c clean/ -e enh/ -o all.csv -m classic dns

# Compute origin metrics only, do not save CSV (prints dataset averages)
python compute_metrics.py -c clean/ -e enh/ -m origin
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Union

import soundfile as sf
import torch
from torchmetrics import MeanSquaredError
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    ShortTimeObjectiveIntelligibility,
    DeepNoiseSuppressionMeanOpinionScore,
)

# Fixed PESQ (torchmetrics implementation still has known issues)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from module.metric.pesq_backward import PerceptualEvaluationSpeechQuality  # type: ignore

from tqdm import tqdm


# --------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------- #
@torch.no_grad()
def load_wav(path: Path) -> torch.Tensor:
    """Return mono (1, T) tensor at 16 kHz; down-mix if stereo."""
    data, sr = sf.read(path)
    if sr != 16000:
        raise ValueError(f"{path} has sample rate {sr} Hz (expected 16 kHz)")
    if data.ndim > 1:  # stereo → mono
        data = data.mean(axis=1)
    return torch.from_numpy(data).unsqueeze(0)


def trim_to_shortest(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim both signals to the shorter length so metric functions match shapes."""
    length = min(x.size(1), y.size(1))
    return x[:, :length], y[:, :length]


# Mapping from metric group name to metric column names
GROUP2COLS: Dict[str, List[str]] = {
    "classic": ["si-sdr", "pesq", "stoi", "estoi"],
    "dns": ["p808_mos", "mos_sig", "mos_bak", "mos_ovr"],
    "origin": ["si-sdr", "pesq", "stoi", "estoi", "mse"],
}


# --------------------------------------------------------------------- #
# Core routine
# --------------------------------------------------------------------- #
@torch.inference_mode()
def compute_metrics_csv(
    clean_root: Path,
    enh_root: Path,
    out_csv: Union[Path, None],
    metric_groups: List[str],
) -> None:
    """
    Compute metrics for all .wav pairs found under `enh_root`.
    If `out_csv` is None, results are not written to disk but averages are printed.
    """

    # Build CSV header: keep order, remove duplicates
    header: List[str] = ["path"]
    for group in metric_groups:
        for col in GROUP2COLS[group]:
            if col not in header:
                header.append(col)

    # Instantiate only the metrics that are required
    sisdr = ScaleInvariantSignalDistortionRatio() if "si-sdr" in header else None
    pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb") if "pesq" in header else None
    stoi = ShortTimeObjectiveIntelligibility(fs=16000) if "stoi" in header else None
    estoi = ShortTimeObjectiveIntelligibility(fs=16000, extended=True) if "estoi" in header else None
    mse = MeanSquaredError() if "mse" in header else None
    dns = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False) if "p808_mos" in header else None

    # Collect all enhanced .wav files
    wav_files = sorted(enh_root.rglob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No .wav files found under {enh_root}")
        return

    # Prepare CSV writer only if saving is requested
    writer = None
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = out_csv.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(header)

    # Running totals for dataset-level averages
    sums: Dict[str, float] = {k: 0.0 for k in header if k != "path"}
    count = 0

    # Iterate over files
    for enh_path in tqdm(wav_files, desc="Files", unit="wav"):
        rel = enh_path.relative_to(enh_root)
        clean_path = (clean_root / rel).with_suffix(".wav")
        if not clean_path.exists():
            tqdm.write(f"[skip] Reference not found: {clean_path}")
            continue

        # Load reference and enhanced audio
        y = load_wav(str(enh_path))  # enhanced
        x = load_wav(str(clean_path))  # clean reference
        x, y = trim_to_shortest(x, y)

        row_dict: Dict[str, Union[str, float]] = {"path": rel.as_posix()}
        row: List[Union[str, float]] = []

        # Classic / origin metrics
        if sisdr is not None:
            v = sisdr(y, x).item()
            row_dict["si-sdr"] = v
            sums["si-sdr"] += v
        if pesq is not None:
            v = pesq(y, x).item()
            row_dict["pesq"] = v
            sums["pesq"] += v
        if stoi is not None:
            v = stoi(y, x).item()
            row_dict["stoi"] = v
            sums["stoi"] += v
        if estoi is not None:
            v = estoi(y, x).item()
            row_dict["estoi"] = v
            sums["estoi"] += v
        if mse is not None:
            v = mse(y, x).item()
            row_dict["mse"] = v
            sums["mse"] += v

        # DNS metrics
        if dns is not None:
            mos = dns(y).squeeze(0).tolist()  # [p808, sig, bak, ovr]
            for key, val in zip(["p808_mos", "mos_sig", "mos_bak", "mos_ovr"], mos):
                sums[key] += val
                row_dict[key] = val

        for key in header:
            row.append(row_dict[key])

        count += 1

        if writer is not None:
            # Format floats with 4 decimal places
            writer.writerow([f"{v:.4f}" if isinstance(v, (float, int)) else v for v in row])

    # Close CSV file if we created it
    if writer is not None:
        csv_file.close()
        print(f"✓ Metrics saved → {out_csv}")

    # Print dataset averages
    if count:
        averages = {k: sums[k] / count for k in sums}
        avg_str = "  ".join(f"{k}={v:.4f}" for k, v in averages.items())
        print("Averages:  ", avg_str)
    else:
        print("[WARN] No valid reference/enhanced pairs evaluated.")


# --------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics for enhanced audio")
    parser.add_argument("-c", "--clean", type=Path, required=True, help="Root of clean reference .wav files")
    parser.add_argument("-e", "--enhanced", type=Path, required=True, help="Root of enhanced .wav files")
    parser.add_argument("-o", "--out_csv", type=Path, default=None, help="Output CSV path (omit to disable saving)")
    parser.add_argument("-m", "--metrics", choices=["classic", "dns", "origin"], nargs="+", default=["classic", "dns", "origin"], help=("Metric group(s) to compute. Multiple groups allowed, e.g. " "-m classic dns"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_metrics_csv(args.clean, args.enhanced, args.out_csv, args.metrics)
