#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate refer.txt, where each line contains:
    <relative/path/to/audio.wav> <reference transcription>
"""

import argparse
import pathlib

AUDIO_EXTS = {".wav", ".mp3", ".flac"}


def load_transcriptions(map_dir: str):
    """
    Read every *.txt file under `map_dir` and build a dictionary
    mapping utt_id → transcription.

    Each line in those files must be:
        <utt_id> <transcription ...>
    """
    trans = {}
    for txt in sorted(pathlib.Path(map_dir).rglob("*.txt")):
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    trans[utt_id] = text
    return trans


def write_references(trans_map, enhanced_dir: str, save_file: str):
    """
    Walk through every audio file under `enhanced_dir` (recursively) and
    write lines of the form

        <relative/path.wav> <reference transcription>

    to `save_file`.

    The key used to look up the transcription is the audio file’s stem
    (everything before the first dot).
    """
    root = pathlib.Path(enhanced_dir).resolve()
    save_path = pathlib.Path(save_file)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as fout:
        for audio in sorted(root.rglob("*")):
            if audio.suffix.lower() not in AUDIO_EXTS:
                continue

            # Relative path such as "Speaker15/file.wav"
            rel_path = audio.relative_to(root).as_posix()

            # e.g. "Speaker15_C_0.white" → "Speaker15_C_0"
            key = audio.stem.split(".", 1)[0]

            ref = trans_map.get(key)
            if ref is None:
                print(f"[Warning] No transcription for {key}; skipped {rel_path}")
                continue

            fout.write(f"{rel_path}\t{ref}\n")

    print(f"[Done] Reference file written to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Create refer.txt (relative path + reference text)")
    parser.add_argument(
        "-m",
        "--map_dir",
        default="../dataset/abdata/ABCS_database/script/test",
        help="Directory of mapping files (each line: utt_id transcription)",
    )
    parser.add_argument(
        "-e",
        "--enhanced_dir",
        default="../results/Proposed0709",
        help="Root directory of enhanced audio files",
    )
    parser.add_argument(
        "-s",
        "--save_file",
        default="../results/Proposed0709/refer.txt",
        help="Path to the output reference file",
    )
    args = parser.parse_args()

    trans_map = load_transcriptions(args.map_dir)
    write_references(trans_map, args.enhanced_dir, args.save_file)


if __name__ == "__main__":
    main()
