#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
model_cfg_path=./config/model_config
model_cfg=DBMIF.yaml
log_dir=./logs/paper
name=DBMIF
epoch=100
batch_size=16

recommended_gpu_space=200000
time_for_waiting_gpu_space=60

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "model_cfg_path: $model_cfg_path"
echo "model_cfg: $model_cfg"
echo "log_dir: $log_dir"
echo "name: $name"
echo "epoch: $epoch"
echo "batch_size: $batch_size"

# Check whether the selected GPU has enough free memory.
gpu_space=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$CUDA_VISIBLE_DEVICES")
echo "GPU space available: $gpu_space MB"

while [ "$gpu_space" -lt "$recommended_gpu_space" ]; do
    tput cuu1
    echo "GPU space available: $gpu_space MB"
    echo "Waiting for GPU space to be larger than $recommended_gpu_space MB"
    waiting=$time_for_waiting_gpu_space
    while [ "$waiting" -gt 0 ]; do
        echo "$waiting seconds left"
        sleep 1
        waiting=$((waiting-1))
        tput cuu1
    done
    gpu_space=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$CUDA_VISIBLE_DEVICES")
    tput cuu1
done

echo "training $model_cfg with $epoch epochs and logging to $log_dir/$name"
cmd=(
    python main.py
    -model_cfg "$model_cfg_path/$model_cfg"
    -e "$epoch"
    -l "$log_dir"
    -n "$name"
    -b "$batch_size"
)
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
