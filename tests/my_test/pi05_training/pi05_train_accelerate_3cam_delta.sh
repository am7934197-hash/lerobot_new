#!/bin/bash
# ============================================================================
# PI05 Delta Action + 3 Cameras 多卡训练脚本
# ============================================================================
# 前提：conda lerobot 已执行 pip install -e .
#
# 用法:
#   bash pi05_train_accelerate_3cam_delta.sh
# ============================================================================

set -e

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate lerobot

# 显式用 conda 环境的 accelerate（避免 .local/bin 的旧版被优先匹配）
ACCELERATE=/mnt/data1/jizheng/miniconda/envs/lerobot/bin/accelerate

echo "Using accelerate: $ACCELERATE"
echo "Using Python: $(which python)"

export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=lxinhZyaQbBOWbcnFbfdX
export CUDA_VISIBLE_DEVICES=2,3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NUM_PROCESSES=${NUM_PROCESSES:-2}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}

$ACCELERATE launch \
    --num_processes=${NUM_PROCESSES} \
    --multi_gpu \
    --mixed_precision=${MIXED_PRECISION} \
    --num_machines=1 \
    --machine_rank=0 \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    -m lerobot.scripts.lerobot_train \
    --config ./pi05_3cam_delta_config.yaml \
    "$@"
