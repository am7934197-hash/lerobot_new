#!/bin/bash
# 使用 SwanLab 训练 LeRobot 模型的脚本
# 使用 miniconda3 的 lerobot 环境
# 多卡训练配置

# 初始化 conda
eval "$(/mnt/data1/jizheng/miniconda3/bin/conda shell.bash hook)"

# 激活 conda 环境
conda activate lerobot

# 设置 HuggingFace 镜像（加速下载）
export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=lxinhZyaQbBOWbcnFbfdX

# 设置 CUDA 可见设备（GPU 4 和 5）
export CUDA_VISIBLE_DEVICES=4,5

# 运行多卡训练
lerobot-train \
    --config /mnt/data1/jizheng/vla_project/lerobot/tests/swanlab/test_swanlab_config.yaml \
    "$@"
