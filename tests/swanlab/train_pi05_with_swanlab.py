#!/usr/bin/env python
"""
使用 SwanLab 训练 PI0.5 策略的示例脚本

这个脚本演示了如何：
1. 使用 SwanLab 替代 WandB 记录训练日志
2. 在小型数据集上训练 PI0.5 模型
3. 配置 API key 并上传到 SwanLab 云端

数据集：lerobot/svla_so101_pickplace (小型 VLA 数据集)
模型：PI0.5 (Physical Intelligence Zero 5)

使用方法:
    python train_pi05_with_swanlab.py --api-key Zu2D1H0CeWLxCZjg7H --epochs 10
"""

import argparse
import os
import time
from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.utils.feature_utils import dataset_to_policy_features
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockEnvConfig:
    type: str = "mock_env"
    fps: int = 30


@dataclass
class MockPolicyConfig:
    type: str = "pi05"


@dataclass 
class TrainConfig:
    """训练配置"""
    output_dir: Path = field(default_factory=lambda: Path("outputs/pi05_swanlab"))
    job_name: str = "pi05_swanlab_training"
    seed: int = 42
    resume: bool = False
    dataset: type = type('obj', (object,), {'repo_id': 'lerobot/svla_so101_pickplace'})
    env: MockEnvConfig = field(default_factory=MockEnvConfig)
    policy: MockPolicyConfig = field(default_factory=MockPolicyConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    is_reward_model_training: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "job_name": self.job_name,
            "seed": self.seed,
            "dataset": self.dataset.repo_id,
            "env": self.env.type,
            "policy": self.policy.type,
            "wandb_backend": self.wandb.backend,
        }


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """创建时间戳增量"""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def setup_swanlab(api_key: str | None, project: str, entity: str | None = None):
    """配置 SwanLab"""
    if api_key:
        # 设置 API key
        os.environ["SWANLAB_API_KEY"] = api_key
        print(f"✓ 已设置 SwanLab API key")
    
    # 配置 WandBConfig 使用 SwanLab
    wandb_config = WandBConfig(
        enable=True,
        backend="swanlab",  # 使用 SwanLab 后端
        project=project,
        entity=entity,
        mode="cloud",  # 云端模式
        add_tags=True,
        notes="PI0.5 policy training with SwanLab",
    )
    
    return wandb_config


def create_model_and_dataset(dataset_id: str, device: torch.device):
    """创建模型和数据集"""
    print(f"\n加载数据集：{dataset_id}")
    # 使用本地缓存路径，避免网络请求
    cache_root = "/mnt/data1/jizheng_home/.cache/huggingface/lerobot/hub"
    try:
        dataset_metadata = LeRobotDatasetMetadata(
            dataset_id,
            root=cache_root,
            revision="v3.0"
        )
    except Exception as e:
        # 如果联网失败，直接使用本地数据
        print(f"  警告：无法连接 HuggingFace Hub ({e})")
        print(f"  尝试直接从本地加载...")
        # 直接使用本地路径，不指定 revision 避免联网验证
        dataset_metadata = LeRobotDatasetMetadata(
            dataset_id,
            root=cache_root
        )
    
    # 获取特征
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"  输入特征：{list(input_features.keys())}")
    print(f"  输出特征：{list(output_features.keys())}")
    
    # 获取状态和动作的维度
    state_dim = input_features["observation.state"].shape[0]
    action_dim = output_features["action"].shape[0]
    
    print(f"  状态维度：{state_dim}")
    print(f"  动作维度：{action_dim}")
    
    # 创建 PI0.5 配置
    cfg = PI05Config(
        max_state_dim=state_dim,
        max_action_dim=action_dim,
        dtype="float32",  # 使用 float32 进行训练
        chunk_size=50,  # 较小的 chunk 用于快速测试
        n_action_steps=50,
        n_obs_steps=1,
        num_inference_steps=10,  # 较少的推理步骤用于快速测试
        freeze_vision_encoder=False,  # 训练整个模型
        gradient_checkpointing=False,  # 小模型不需要梯度检查点
        compile_model=False,  # 测试时不编译模型
    )
    
    # 设置输入输出特征
    cfg.input_features = input_features
    cfg.output_features = output_features
    
    # 创建策略
    print(f"\n创建 PI0.5 策略...")
    policy = PI05Policy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    policy.train()
    policy.to(device) 
    print(f"  ✓ 模型创建完成")
    print(f"  参数量：{sum(p.numel() for p in policy.parameters()):,}")
    
    # 创建时间戳
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    
    # 添加图像特征
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }
    
    # 创建数据集
    print(f"\n加载数据集...")
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    print(f"  数据集大小：{len(dataset):,} 样本")
    print(f"  FPS: {dataset_metadata.fps}")
    
    return policy, preprocessor, postprocessor, dataset, cfg


def train_epoch(policy, preprocessor, dataloader, optimizer, device, epoch, logger, global_step):
    """训练一个 epoch"""
    policy.train()
    total_loss = 0.0
    n_batches = len(dataloader)
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # LeRobotDataset 返回的是字典，包含 observation.* 和 action
        # 将数据移动到设备上
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 预处理
        batch = preprocessor(batch)
        
        # 前向传播
        loss, _ = policy.forward(batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 每 10 个 batch 记录一次
        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            metrics = {
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "batch": batch_idx,
            }
            logger.log_dict(metrics, step=global_step + batch_idx, mode="train")
    
    avg_loss = total_loss / n_batches
    epoch_time = time.time() - start_time
    
    return avg_loss, epoch_time, global_step + n_batches


def evaluate(policy, preprocessor, dataloader, device, epoch, logger, global_step):
    """评估模型"""
    policy.eval()
    total_loss = 0.0
    n_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            # LeRobotDataset 返回的是字典，包含 observation.* 和 action
            # 将数据移动到设备上
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = preprocessor(batch)
            
            # Eval 模式下只计算 reconstruction loss
            loss, _ = policy.forward(batch)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / n_batches
    
    # 记录评估指标
    metrics = {
        "eval_loss": avg_loss,
        "epoch": epoch,
    }
    logger.log_dict(metrics, step=global_step, mode="eval")
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="使用 SwanLab 训练 PI0.5 模型")
    parser.add_argument("--api-key", type=str, default=None, 
                       help="SwanLab API key")
    parser.add_argument("--project", type=str, default="lerobot-pi05-test",
                       help="SwanLab 项目名称")
    parser.add_argument("--entity", type=str, default=None,
                       help="SwanLab 工作空间名称")
    parser.add_argument("--epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="批次大小 (PI0.5 需要更多显存，建议使用较小的 batch size)")
    parser.add_argument("--dataset", type=str, default="lerobot/svla_so101_pickplace",
                       help="数据集 ID")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="outputs/pi05_swanlab",
                       help="输出目录")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="最大训练样本数（用于快速测试，None 表示使用全部数据）")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用设备：{device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置 SwanLab
    print(f"\n{'='*60}")
    print(f"配置 SwanLab")
    print(f"{'='*60}")
    wandb_config = setup_swanlab(
        api_key=args.api_key,
        project=args.project,
        entity=args.entity,
    )
    
    # 创建训练配置
    train_config = TrainConfig(
        output_dir=output_dir,
        job_name=f"{args.project}_run",
        wandb=wandb_config,
    )
    
    # 初始化 logger
    print(f"\n初始化 SwanLab Logger...")
    logger = WandBLogger(train_config)
    print(f"✓ Logger 初始化成功")
    print(f"  后端：{logger.backend}")
    print(f"  Run ID: {train_config.wandb.run_id}")
    
    # 创建模型和数据集
    print(f"\n{'='*60}")
    print(f"创建模型和数据集")
    print(f"{'='*60}")
    policy, preprocessor, postprocessor, dataset, pi05_cfg = create_model_and_dataset(
        args.dataset, device
    )
    
    # 创建数据加载器
    # 如果指定了 max_samples，限制数据集大小
    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
        print(f"  ✓ 限制数据集大小：{len(dataset)} 样本（原始：{len(dataset)}）")
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # 创建优化器（使用 PI0.5 推荐的优化器配置）
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=pi05_cfg.optimizer_lr,
        betas=pi05_cfg.optimizer_betas,
        eps=pi05_cfg.optimizer_eps,
        weight_decay=pi05_cfg.optimizer_weight_decay,
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=pi05_cfg.scheduler_warmup_steps,
        eta_min=pi05_cfg.scheduler_decay_lr,
    )
    
    # 训练循环
    print(f"\n{'='*60}")
    print(f"开始训练")
    print(f"{'='*60}")
    print(f"数据集：{args.dataset}")
    print(f"训练样本：{len(train_dataset):,}")
    print(f"验证样本：{len(val_dataset):,}")
    print(f"批次大小：{args.batch_size}")
    print(f"训练轮数：{args.epochs}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练
        train_loss, epoch_time, global_step = train_epoch(
            policy, preprocessor, train_loader, optimizer, device,
            epoch, logger, global_step
        )
        
        # 评估
        val_loss = evaluate(
            policy, preprocessor, val_loader, device, epoch, logger, global_step
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录 epoch 摘要
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time,
            "samples_per_second": len(train_dataset) / epoch_time,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }
        logger.log_dict(epoch_metrics, step=epoch, mode="train")
        
        # 打印进度
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            checkpoint_dir = output_dir / f"best_model"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(checkpoint_dir)
            preprocessor.save_pretrained(checkpoint_dir)
            postprocessor.save_pretrained(checkpoint_dir)
            
            # 记录 artifact
            logger.log_policy(checkpoint_dir)
            
            print(f"  ✓ 保存最佳模型 (epoch {epoch})")
        
        # 每 2 个 epoch 保存检查点
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(checkpoint_dir)
            preprocessor.save_pretrained(checkpoint_dir)
            postprocessor.save_pretrained(checkpoint_dir)
            
            logger.log_policy(checkpoint_dir)
            print(f"  ✓ 保存检查点：{checkpoint_dir.name}")
    
    # 训练完成
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"{'='*60}")
    print(f"最佳验证损失：{best_val_loss:.4f} (epoch {best_epoch})")
    print(f"模型保存在：{output_dir}")
    
    # 记录最终摘要
    final_metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": args.epochs,
        "final_train_loss": train_loss,
    }
    logger.log_dict(final_metrics, step=args.epochs, mode="eval")
    
    # 结束 run，确保所有数据都同步到云端
    print("\n正在同步数据到云端，请稍候...")
    logger.finish()
    print("✓ 数据同步完成")
    
    print(f"\n查看训练日志:")
    print(f"  本地：swanlab watch {output_dir}/swanlog")
    print(f"  云端：https://swanlab.cn")
    print(f"\n")


if __name__ == "__main__":
    # 设置 HuggingFace 镜像
    os.environ["HF_ENDPOINT"] = "https://hf.mirror.com"
    # 指定使用 GPU (可根据实际情况调整)
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    main()


# python tests/swanlab/train_pi05_with_swanlab.py --api-key Zu2D1H0CeWLxCZjg7H --project lerobot-pi05-test --epochs 10 --batch-size 8 --dataset lerobot/svla_so101_pickplace
