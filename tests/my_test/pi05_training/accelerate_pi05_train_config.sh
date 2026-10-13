# ============================================================================
# PI05 多卡训练脚本（accelerate）
#
# 用法:
#   sh pi05_train_accelerate.sh                              # 默认配置
#   MIXED_PRECISION=fp16 sh pi05_train_accelerate.sh         # 半精度
#   NUM_PROCESSES=4 sh pi05_train_accelerate.sh              # 4 卡
#   GRADIENT_ACCUMULATION_STEPS=4 sh pi05_train_accelerate.sh # 梯度累积
#
# accelerate 配置: ./accelerate.yaml（分布式 / 精度 / GPU 等）
# 训练参数配置:     ./pi05_test_swanlab_config.yaml（模型 / 数据集 / 优化器）
#
# 有效 batch_size = YAML batch_size × num_processes × gradient_accumulation_steps
# 当前默认: 16 × 2 × 2 = 64
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
conda init
conda activate lerobot

# 设置 HuggingFace 镜像（加速下载）
export HF_ENDPOINT=https://hf-mirror.com
export SWANLAB_API_KEY=lxinhZyaQbBOWbcnFbfdX

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================================
# 可在命令行通过环境变量覆盖的参数（其余参数在 ./accelerate.yaml 中配置）
# ============================================================================

# 梯度累积步数（accelerate.yaml 中无此字段，必须通过 CLI 传入）
# 每 N 个 batch 才执行一次 optimizer.step()，省显存扩 batch
# 有效 batch = YAML batch_size × num_processes × GRADIENT_ACCUMULATION_STEPS
# 例: 16 × 2 × 2 = 64
# 设为 1 即不累积
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}

# ============================================================================
# 运行训练
# 分布式 / 精度 / GPU 等参数: ./accelerate.yaml
# 模型 / 数据集 / 优化器参数:  ./pi05_test_swanlab_config.yaml
# ============================================================================
accelerate launch \
    --config_file ./accelerate.yaml \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    -m lerobot.scripts.lerobot_train \
    --config ./pi05_test_swanlab_config.yaml \
    "$@"