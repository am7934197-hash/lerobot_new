# ============================================================================
# PI05 多卡训练脚本（accelerate）
# GPU 数量: export CUDA_VISIBLE_DEVICES 控制可见 GPU，--num_processes 控制使用张数
# 用法:
#   sh pi05_train_test.sh                    # 默认 2 卡训练
#   sh pi05_train_test.sh --num_processes=1  # 单卡训练
#   sh pi05_train_test.sh --num_processes=4  # 4 卡训练
# ============================================================================
# lerobot-train \
#     --dataset.repo_id=your_dataset \
#     --policy.type=pi05 \
#     --output_dir=./outputs/pi05_training \
#     --job_name=pi05_training \
#     --policy.repo_id=your_repo_id \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.gradient_checkpointing=true \
#     --wandb.enable=true \
#     --policy.dtype=bfloat16 \
#     --policy.freeze_vision_encoder=false \
#     --policy.train_expert_only=false \
#     --steps=3000 \
#     --policy.device=cuda \
#     --batch_size=32
set -e

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate lerobot

# 显式用 conda 环境的 accelerate（避免 .local/bin 的旧版被优先匹配）
ACCELERATE=/mnt/data1/jizheng/miniconda/envs/lerobot/bin/accelerate

echo "Using accelerate: $ACCELERATE"
echo "Using Python: $(which python)"
# 设置 HuggingFace 镜像（加速下载）
export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=lxinhZyaQbBOWbcnFbfdX

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================================
# accelerate launch 参数说明
# ============================================================================

# --num_processes: 启动的进程总数（= GPU 卡数）。每个进程独占一张 GPU
#    通常与 CUDA_VISIBLE_DEVICES 中列出的 GPU 数量一致
#    例如: CUDA_VISIBLE_DEVICES=0,1,2,3 → NUM_PROCESSES=4
NUM_PROCESSES=${NUM_PROCESSES:-4}

# --mixed_precision: 混合精度训练模式
#    no    = 全精度 float32（显存占用最大，训练最稳定）
#    fp16  = 半精度 float16（省显存 ~40%，需要 loss scaling，NVIDIA V100+ 支持）
#    bf16  = bfloat16（省显存 ~40%，数值范围与 fp32 一致，A100/H100/B200 支持）
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

# --gradient_accumulation_steps: 梯度累积步数
#    每 N 个 batch 才执行一次 optimizer.step()，用于在不增加显存的情况下扩大有效 batch size
#    有效 batch size = YAML batch_size × NUM_PROCESSES × GRADIENT_ACCUMULATION_STEPS
#    例如: 16 × 2 × 2 = 64
#    设为 1 表示不累积（每个 batch 都更新权重）
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}

# 运行 accelerate 多卡训练
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    `# --multi_gpu: 单机多卡模式，pytorch 用 DistributedDataParallel (DDP) 同步梯度` \
    --multi_gpu \
    --mixed_precision=${MIXED_PRECISION} \
    `# --num_machines: 总机器数，单机训练固定 1` \
    --num_machines=1 \
    `# --machine_rank: 当前机器的编号，单机固定 0；多机时每台机器设为 0,1,2...` \
    --machine_rank=0 \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    `# -m: 以 Python 模块方式运行，而非脚本文件路径` \
    -m lerobot.scripts.lerobot_train \
    --config ./pi05_test_swanlab_config.yaml \
    "$@"