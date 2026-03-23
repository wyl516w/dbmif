#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk every sub-folder of --root that contains:
    metrics.csv, whisper_cer.csv, wenet_cer.csv
Filter rows whose `path` field contains --contains,
then print average metrics and re-computed CERs.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_metrics_csv(csv_path: Path, needle: str) -> Dict[str, float]:
    """Return {metric: mean} over rows whose path contains `needle`."""
    sums: Dict[str, float] = {}
    count = 0

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if needle and needle not in row["path"]:
                continue
            if "snr_10" in row["path"]:
                continue
            for k, v in row.items():
                if k == "path":
                    continue
                sums[k] = sums.get(k, 0.0) + float(v)
            count += 1

    return {k: v / count for k, v in sums.items()} if count else {}


def recalc_cer(csv_path: Path, needle: str) -> float:
    """Return overall CER (%) via formula, after filtering by `needle`."""
    total_N = total_err = 0
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if needle and needle not in row["path"]:
                continue
            N   = int(row["N"])
            sub = int(row["sub"])
            ins = int(row["ins"])
            dele = int(row["del"])
            total_N   += N
            total_err += sub + ins + dele
    return (total_err / total_N * 100) if total_N else 0.0


def process_folder(folder: Path, needle: str) -> None:
    mfile = folder / "metrics.csv"
    wfile = folder / "whisper_cer.csv"
    nfile = folder / "wenet_cer.csv"
    if not (mfile.exists() and wfile.exists() and nfile.exists()):
        return

    print(f"\n=== {folder.relative_to(folder.parent)} ===")

    # metrics.csv
    metrics = read_metrics_csv(mfile, needle)
    if metrics:
        m_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print("metrics:    ", m_str)
    else:
        print("metrics:     (no rows matched)")

    # CERs
    w_cer = recalc_cer(wfile, needle)
    n_cer = recalc_cer(nfile, needle)
    print(f"whisper CER: {w_cer:.2f} %")
    print(f"wenet   CER: {n_cer:.2f} %")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-folder metrics & CER")
    ap.add_argument("--root", type=Path, default="../results",
                    help="root directory that holds experiment folders")
    ap.add_argument("--contains", default="",
                    help="substring that *must* appear in path column (default: all)")
    args = ap.parse_args()

    for sub in sorted(args.root.rglob("*")):
        if sub.is_dir():
            process_folder(sub, args.contains)


if __name__ == "__main__":
    main()
