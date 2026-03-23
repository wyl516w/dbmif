#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run WeNet and Whisper (openai-whisper) on a tree of wav/mp3/flac files.

Outputs in <output_dir> (if given):
    whisper.txt   • <rel/path.wav> <whisper transcript>
    wenet.txt     • <rel/path.wav> <wenet transcript>

If --output_dir is omitted or set to 'None', transcripts are written to input_dir.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
import wenet                    # pip install git+https://github.com/wenet-e2e/wenet.git
import whisper                  # pip install openai-whisper

# ────────── ASR wrappers ────────────────────────────────────────────────
class WhisperASR:
    """openai-whisper model (default: large-v3, Chinese)."""
    def __init__(self, model_name: str = "turbo") -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # load_model handles FP16 automatically on CUDA; keep fp16 flag for transcribe
        self.model = whisper.load_model(model_name, device=device)
        self.fp16 = (device == "cuda")
    def __call__(self, wav: Path) -> str:
        # For Chinese, set language="zh". Task defaults to "transcribe".
        res = self.model.transcribe(str(wav), language="zh", fp16=self.fp16)
        return res.get("text", "").strip()


class WenetASR:
    """WeNet online ASR (Chinese model by default)."""
    def __init__(self, model_type: str = "wenetspeech") -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = wenet.load_model(model_type, device=device)

    def __call__(self, wav: Path) -> str:
        return self.model.transcribe(str(wav)).text.strip()


# ────────── helpers ─────────────────────────────────────────────────────
def load_refer_paths(refer_path: Path) -> List[str]:
    """Return list of relative audio paths extracted from refer.txt."""
    with refer_path.open(encoding="utf-8") as f:
        return [line.split(maxsplit=1)[0] for line in f if line.strip()]


def write_txt(dst: Path, lines: List[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines), encoding="utf-8")


# ────────── main ───────────────────────────────────────────────────────
@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    in_root = Path(args.input_dir)
    out_root: Optional[Path]
    if args.output_dir is not None and str(args.output_dir).lower() != "none":
        out_root = Path(args.output_dir)
    else:
        out_root = in_root

    # 1. Which files?
    rel_paths = load_refer_paths(args.refer_txt)
    audio_files = [in_root / p for p in rel_paths]

    missing = [p for p in audio_files if not p.exists()]
    if missing:
        print(f"[WARN] {len(missing)} files not found; they will be skipped.")
        audio_files = [p for p in audio_files if p.exists()]
    if not audio_files:
        print("[ERR] No valid audio to process — exiting.")
        return

    # 2. ASR engines
    engines: Dict[str, object] = {}
    for name in args.model:
        lname = name.lower()
        if lname == "wenet":
            engines["wenet"] = WenetASR()
        elif lname == "whisper":
            engines["whisper"] = WhisperASR()  # uses openai-whisper
        else:
            print(f"[ERR] Unsupported model name: {name}")

    if not engines:
        print("[ERR] No valid ASR engines selected."); return

    # 3. Run
    for name, engine in engines.items():
        lines: List[str] = []
        for fp in tqdm(audio_files, desc=f"ASR {name}", unit="file"):
            rel = fp.relative_to(in_root).as_posix()
            transcript = engine(fp)
            lines.append(f"{rel}\t{transcript}")

        write_txt(out_root / f"{name}.txt", lines)
        print(f"[OK] {name}.txt written: {len(lines)} lines")


# ────────── CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch ASR with WeNet and openai-whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input_dir",
        type=Path,
        required=True,
        help="Root directory that contains the audio tree"
    )
    parser.add_argument(
        "-r", "--refer_txt",
        type=Path,
        required=True,
        help="Text file listing relative audio paths (text after the first "
             "whitespace is ignored)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=None,
        help="Destination directory for <model>.txt files. "
             "If omitted or 'None', uses input_dir."
    )
    parser.add_argument(
        "-m", "--model",
        nargs="+",
        choices=["wenet", "whisper"],
        default=["wenet", "whisper"],
        help="ASR engines to run (space-separated list)"
    )

    args = parser.parse_args()
    main(args)
