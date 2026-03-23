#!/usr/bin/env bash
# Unified test pipeline: infer -> metrics -> ASR -> CER.
#
# Usage:
#   ./test.sh MODEL_NAME [--asr "wenet whisper"|wenet|whisper] [--cuda "0[,1]"] \
#                      [--config /path/to/model.yaml] [--ckpt /path/to/model.ckpt] \
#                      [--input ../dataset/gendata/] [--output ../results]
#
# Python helpers live under ./test.

set -euo pipefail

# ---------- locate script & python dir ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYDIR="${SCRIPT_DIR}/test"

# ---------- defaults ----------
MODEL=""
ASR_MODELS="wenet whisper"
CUDA=""
INPUT="../dataset/gendata/"
OUTPUT="../results"
CONFIG=""
CKPT=""

# ---------- helpers ----------
usage() { sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//'; }
need_file() { [[ -f "$1" ]] || { echo "ERROR: missing file: $1"; exit 2; }; }
need_dir()  { [[ -d "$1" ]] || { echo "ERROR: missing dir: $1";  exit 2; }; }
has_wavs()  { find "$1" -type f -name "*.wav" -print -quit | grep -q .; }

autodetect_config() {
  local m="$1"
  local cand1="../config/model_config/${m}.yaml"
  local cand2="./yaml/${m}.yaml"
  [[ -f "$cand1" ]] && { echo "$cand1"; return; }
  [[ -f "$cand2" ]] && { echo "$cand2"; return; }
  echo ""
}

autodetect_ckpt() {
  local m="$1"

  # 1) Prefer ../checkpoints/{MODEL}*.ckpt (pick most recent)
  local ckpt1
  ckpt1=$(ls -t "../checkpoints/${m}"*.ckpt 2>/dev/null | head -n1 || true)
  if [[ -n "${ckpt1:-}" ]]; then
    echo "$ckpt1"
    return
  fi

  # 2) Fallback to logs/ablations layout
  local pattern="../logs/*/${m}-*/version_*/checkpoints/epoch=*-step=*.ckpt"
  ls -t $pattern 2>/dev/null | head -n1 || true
}

ts() { date "+%Y-%m-%d %H:%M:%S"; }

# ---------- args ----------
if [[ $# -lt 1 ]]; then usage; exit 1; fi
MODEL="$1"; shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asr)    ASR_MODELS="$2"; shift 2 ;;
    --cuda)   CUDA="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --ckpt)   CKPT="$2"; shift 2 ;;
    --input)  INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# ---------- resolve paths ----------
ENH_DIR="${OUTPUT%/}/${MODEL}"
CLEAN_DIR="${OUTPUT%/}/clean"
REF_TXT="${ENH_DIR}/refer.txt"
MET_CSV="${ENH_DIR}/metrics.csv"

SKIP_INFER=0
if [[ "$MODEL" == "ac" || "$MODEL" == "bc" ]]; then
  SKIP_INFER=1
fi

# ---------- environment & sanity ----------
[[ -n "$CUDA" ]] && export CUDA_VISIBLE_DEVICES="$CUDA"

if [[ ! -d "$OUTPUT" ]]; then
  mkdir -p "$OUTPUT"
  echo "[mkdir] Created output dir: $OUTPUT"
fi

if (( SKIP_INFER )); then
  echo ">>> MODEL='$MODEL' -> skipping inference (using existing ${ENH_DIR})"
  need_dir "$ENH_DIR"
  if ! has_wavs "$ENH_DIR"; then
    echo "ERROR: No .wav files found under '$ENH_DIR' to evaluate."; exit 2;
  fi
else
  need_dir "$INPUT"
  if [[ -z "$CONFIG" ]]; then CONFIG="$(autodetect_config "$MODEL")"; fi
  if [[ -z "$CKPT"   ]]; then CKPT="$(autodetect_ckpt   "$MODEL")"; fi
  [[ -n "$CONFIG" ]] || { echo "ERROR: cannot auto-detect config for MODEL=$MODEL"; exit 2; }
  [[ -n "$CKPT"   ]] || { echo "ERROR: cannot auto-detect ckpt   for MODEL=$MODEL"; exit 2; }
  need_file "$CONFIG"
  need_file "$CKPT"
fi

read -r -a ASR_ARR <<< "$ASR_MODELS"

echo "============================================================"
echo "test.sh - $(ts)"
echo "Model:    $MODEL   (skip_infer=$SKIP_INFER)"
echo "ASR:      ${ASR_ARR[*]}"
echo "CUDA:     ${CUDA:-<unchanged>}"
if (( ! SKIP_INFER )); then
  echo "Config:   ${CONFIG}"
  echo "Ckpt:     ${CKPT}"
fi
echo "Input:    ${INPUT}"
echo "Output:   ${OUTPUT}"
echo "Enhanced: ${ENH_DIR}"
echo "Clean:    ${CLEAN_DIR}"
echo "PyDir:    ${PYDIR}"
echo "============================================================"

# ---------- 1) Inference (optional) ----------
if (( ! SKIP_INFER )); then
  echo "[1/5] infer_and_save.py"
  python "${PYDIR}/infer_and_save.py" \
    --config "$CONFIG" \
    --ckpt   "$CKPT" \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    -n "$MODEL"
fi

# ---------- 2) Refer list ----------
echo "[2/5] generate_refer.py"
python "${PYDIR}/generate_refer.py" \
  -e "$ENH_DIR/" \
  -s "$REF_TXT"

# ---------- 3) Metrics (classic dns origin) ----------
echo "[3/5] compute_metrics_csv.py"
python "${PYDIR}/compute_metrics_csv.py" \
 -c "$CLEAN_DIR" \
 -e "$ENH_DIR" \
 -o "$MET_CSV" \
 -m classic dns origin

# ---------- 4) ASR ----------
echo "[4/5] batch_asr.py  (models: ${ASR_ARR[*]})"
python "${PYDIR}/batch_asr.py" \
  -i "$ENH_DIR/" \
  -r "$REF_TXT" \
  -m "${ASR_ARR[@]}" \
  -o "$ENH_DIR/"

# ---------- 5) CER ----------
echo "[5/5] cer_to_csv.py (per ASR engine)"
for engine in "${ASR_ARR[@]}"; do
  ASR_TXT="${ENH_DIR}/${engine}.txt"
  CER_CSV="${ENH_DIR}/${engine}_cer.csv"
  if [[ ! -f "$ASR_TXT" ]]; then
    echo "[warn] Missing ASR hypothesis for '$engine': $ASR_TXT (skipping CER)"; continue
  fi
  python "${PYDIR}/cer_to_csv.py" \
    -r "$REF_TXT" \
    -p "$ASR_TXT" \
    -o "$CER_CSV"
  echo "  wrote: $CER_CSV"
done

echo "------------------------------------------------------------"
echo "Done. Outputs:"
echo "  Refer:   $REF_TXT"
echo "  Metrics: $MET_CSV"
for engine in "${ASR_ARR[@]}"; do
  echo "  ASR:     ${ENH_DIR}/${engine}.txt"
  echo "  CER CSV: ${ENH_DIR}/${engine}_cer.csv"
done
echo "============================================================"
