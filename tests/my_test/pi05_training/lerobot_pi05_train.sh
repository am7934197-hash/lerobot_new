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

# 设置 CUDA 可见设备（GPU 4 和 5）
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 
# 运行多卡训练
lerobot-train \
    --config ./pi05_test_swanlab_config.yaml \
    "$@"