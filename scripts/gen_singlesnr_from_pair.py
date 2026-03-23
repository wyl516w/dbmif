#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate singlesnr AB-Noise mixtures from ONE AB wav (stereo: A/B) and ONE noise wav.

Output tree:
<output_dir>/
  └── <noise_tag>/
      └── singlesnr/
          ├── snr_-15/
          │   └── <data_stem>.<noise_stem>.wav   (3-ch: [A, A+N, B])
          ├── snr_-10/
          └── ...

Args:
  --data_path   path to ONE AB wav (stereo: ch0=A, ch1=B)
  --noise_path  path to ONE noise wav
  --output_dir
  --snr_start   e.g., -15
  --snr_end     e.g., 15
  --snr_step    e.g., 5
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (audio[T, C], sr) as float32."""
    audio, sr = sf.read(str(path), always_2d=True)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, sr

def ensure_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    """Tile or center-crop noise to target_len along time axis (T, C)."""
    T = noise.shape[0]
    if T == target_len:
        return noise
    if T > target_len:
        start = (T - target_len) // 2
        return noise[start:start + target_len]
    reps = int(np.ceil(target_len / T))
    tiled = np.tile(noise, (reps, 1))
    return tiled[:target_len]

def to_mono(x: np.ndarray) -> np.ndarray:
    """Average channels -> mono [T]."""
    if x.ndim == 1:
        return x
    return x.mean(axis=1)

def snr_scale(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Scale 'noise' to reach target SNR wrt 'signal'.
    SNR = 10*log10(Psig/Pnoise); returns scaled noise [T].
    """
    eps = 1e-12
    Ps = float(np.mean(signal ** 2)) + eps
    Pn = float(np.mean(noise ** 2)) + eps
    target_ratio = 10.0 ** (-snr_db / 10.0)  # Pn/Ps
    scale = np.sqrt((Ps * target_ratio) / Pn)
    return noise * scale

def peak_normalize(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    """Normalize to avoid clipping for multi-channel [T, C]."""
    mx = np.max(np.abs(x))
    if mx > peak:
        x = x * (peak / (mx + 1e-12))
    return x

def generate_one(data_path: Path, noise_path: Path, out_root: Path,
                 snr_start: int, snr_end: int, snr_step: int) -> None:
    # Read files
    ab, sr_ab = read_wav(data_path)     # expect [T, 2]
    nz, sr_nz = read_wav(noise_path)    # [T, Cn]

    if ab.shape[1] < 2:
        raise ValueError(f"Expected stereo AB file (A/B) at {data_path}, got shape {ab.shape}")
    if sr_ab != sr_nz:
        # raise ValueError(f"Sample rate mismatch: data={sr_ab}, noise={sr_nz}. Please resample first.")
        # resampled noise supported
        print(f"[warn] Sample rate mismatch: data={sr_ab}, noise={sr_nz}. Resampling noise to match data sample rate.")

    A = ab[:, 0].astype(np.float32)
    B = ab[:, 1].astype(np.float32)

    nz_mono = to_mono(nz)
    nz_mono = ensure_length(nz_mono[:, None], len(A))[:, 0]

    noise_tag = noise_path.stem
    data_stem = data_path.stem

    singles = list(range(snr_start, snr_end + 1, snr_step))
    for snr in singles:
        # SNR is referenced to A (since只对A做加噪)
        nz_scaled = snr_scale(A, nz_mono, float(snr))
        A_noisy = A + nz_scaled

        # 3-ch: [A, A+N, B]
        out = np.stack([A, A_noisy, B], axis=1)
        out = peak_normalize(out)

        out_dir = out_root / noise_tag / "singlesnr" 
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"snr_{snr}.{data_stem}.{noise_tag}.wav"
        sf.write(str(out_path), out, sr_ab)
        print(f"[ok] {snr:+d} dB -> {out_path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AB-Noise (singlesnr) from one AB wav and one noise wav")
    p.add_argument("--data_path", required=True, help="ONE AB wav (stereo A/B)")
    p.add_argument("--noise_path", required=True, help="ONE noise wav")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--snr_start", type=int, required=True)
    p.add_argument("--snr_end", type=int, required=True)
    p.add_argument("--snr_step", type=int, required=True)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    generate_one(
        data_path=Path(args.data_path),
        noise_path=Path(args.noise_path),
        out_root=Path(args.output_dir),
        snr_start=args.snr_start,
        snr_end=args.snr_end,
        snr_step=args.snr_step,
    )

if __name__ == "__main__":
    main()
