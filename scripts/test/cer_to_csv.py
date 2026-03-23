#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cer_to_csv.py

Compute per-utterance CER from two-column lists:

    refer.txt      path <TAB> text …
    hyp.txt        path <TAB> text …

If --out is provided, a CSV is generated with columns:
    path, ref, hyp, N, cor, sub, ins, del, cer

If --out is omitted or set to 'None', nothing is saved and only an
aggregate CER is printed.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional


# ───────────────────────── helpers ────────────────────────────
def parse_list(txt: Path) -> Dict[str, str]:
    """Load a <rel path>\t<text> file into a dict."""
    mapping: Dict[str, str] = {}
    if not txt.exists():
        raise FileNotFoundError(txt)
    with txt.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rel, text = line.rstrip("\n").split("\t", maxsplit=1)
            mapping[rel] = text.strip()
    return mapping


def cer_stats(ref: str, hyp: str) -> Tuple[int, int, int]:
    """Return (sub, ins, del) counts at character level."""
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    n, m = len(ref), len(hyp)

    # Edit distance DP
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    op   = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0], op[i][0] = i, "del"
    for j in range(1, m + 1):
        dist[0][j], op[0][j] = j, "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dist[i][j], op[i][j] = dist[i - 1][j - 1], "cor"
            else:
                best, best_op = dist[i - 1][j - 1] + 1, "sub"
                if dist[i - 1][j] + 1 < best:
                    best, best_op = dist[i - 1][j] + 1, "del"
                if dist[i][j - 1] + 1 < best:
                    best, best_op = dist[i][j - 1] + 1, "ins"
                dist[i][j], op[i][j] = best, best_op

    sub = ins = dele = 0
    i, j = n, m
    while i or j:
        o = op[i][j]
        if o == "cor":
            i -= 1; j -= 1
        elif o == "sub":
            sub += 1; i -= 1; j -= 1
        elif o == "del":
            dele += 1; i -= 1
        else:  # ins
            ins += 1; j -= 1
    return sub, ins, dele


def process_lists(
    refer_txt: Path,
    hyp_txt: Path,
    out_csv: Optional[Path] = None,
) -> None:
    ref_map = parse_list(refer_txt)
    hyp_map = parse_list(hyp_txt)

    writer = None
    if out_csv is not None and str(out_csv).lower() != "none":
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        f = out_csv.open("w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["path", "ref", "hyp", "N", "cor", "sub", "ins", "del", "cer"])

    # running totals for aggregate CER
    total_N = total_err = 0

    for path, ref in ref_map.items():
        if path not in hyp_map:
            continue
        hyp = hyp_map[path]
        sub, ins, dele = cer_stats(ref, hyp)
        N = len(ref.replace(" ", ""))
        err = sub + ins + dele
        cer = (err / N * 100) if N else 0.0
        cor = N - sub - dele

        if writer is not None:
            writer.writerow([path, ref, hyp, N, cor, sub, ins, dele, f"{cer:.2f}"])

        total_N   += N
        total_err += err

    if writer is not None:
        f.close()
        print(f"[OK] CSV saved → {out_csv}")
    else:
        print("[INFO] No CSV saved (out=None)")

    # Aggregate CER
    agg_cer = (total_err / total_N * 100) if total_N else 0.0
    print(f"Overall CER = {agg_cer:.2f}%  (tokens={total_N}, errors={total_err})")


# ───────────────────────── CLI ────────────────────────────
def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-utterance CER and (optionally) save to CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r", "--refer",
        type=Path,
        required=True,
        help="Reference list (e.g., refer.txt)"
    )
    parser.add_argument(
        "-p", "--hyp",
        type=Path,
        required=True,
        help="Hypothesis list (e.g., wenet.txt)"
    )
    parser.add_argument(
        "-o", "--out",
        type=Path,
        default=None,
        help="Output CSV path. Omit or set to 'None' to disable saving."
    )
    args = parser.parse_args()
    process_lists(args.refer, args.hyp, args.out)


if __name__ == "__main__":
    cli()
