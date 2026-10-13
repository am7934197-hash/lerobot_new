#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mock training script to test WandB/SwanLab logging integration.

This script simulates a real training scenario with:
- Multiple metrics (loss, accuracy, learning rate, etc.)
- Training and evaluation phases
- Custom step tracking (simulating RL training)
- Checkpoint saving
- Progress over epochs

Usage:
    # Test with WandB
    python test_mock_training.py --backend wandb --epochs 5
    
    # Test with SwanLab
    python test_mock_training.py --backend swanlab --epochs 5
    
    # Compare both backends
    python test_mock_training.py --backend both --epochs 3
"""

import argparse
import random
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockDatasetConfig:
    repo_id: str = "mock/dataset"
    root: str | None = None


@dataclass
class MockEnvConfig:
    type: str = "mock_env"
    fps: int = 30


@dataclass
class MockPolicyConfig:
    type: str = "mock_policy"


@dataclass
class MockTrainConfig:
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    job_name: str = "mock_training"
    seed: int = 42
    resume: bool = False
    dataset: MockDatasetConfig = field(default_factory=MockDatasetConfig)
    env: MockEnvConfig = field(default_factory=MockEnvConfig)
    policy: MockPolicyConfig = field(default_factory=MockPolicyConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    is_reward_model_training: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "job_name": self.job_name,
            "seed": self.seed,
            "dataset": self.dataset.repo_id,
            "env": self.env.type,
            "policy": self.policy.type,
            "wandb_backend": self.wandb.backend,
            "batch_size": 32,
            "learning_rate": 1e-3,
        }


class MockPolicy(nn.Module):
    """Simple mock policy for testing."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.network(x)
    
    def save_pretrained(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")


def create_mock_dataset(n_samples: int = 1000, input_dim: int = 10, output_dim: int = 5):
    """Create a mock dataset for training."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, output_dim)
    return TensorDataset(X, y)


def train_epoch(model, dataloader, optimizer, device, epoch, logger, cfg):
    """Train for one epoch and log metrics."""
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        predictions = model(batch_x)
        loss = nn.MSELoss()(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate additional metrics
        current_lr = optimizer.param_groups[0]['lr']
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        
        # Log metrics every 10 batches
        global_step = epoch * n_batches + batch_idx
        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            metrics = {
                "batch_loss": loss.item(),
                "learning_rate": current_lr,
                "gradient_norm": grad_norm,
                "batch": batch_idx,
            }
            logger.log_dict(metrics, step=global_step, mode="train")
    
    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate(model, dataloader, device, epoch, logger, cfg):
    """Evaluate model and log metrics."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            
            loss = nn.MSELoss()(predictions, batch_y)
            mae = torch.abs(predictions - batch_y).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
    
    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    
    # Log evaluation metrics
    metrics = {
        "eval_loss": avg_loss,
        "eval_mae": avg_mae,
        "epoch": epoch,
    }
    logger.log_dict(metrics, step=epoch, mode="eval")
    
    return avg_loss, avg_mae


def save_checkpoint(model, optimizer, epoch, output_dir, logger, cfg):
    """Save model checkpoint and log artifact."""
    checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = checkpoint_dir / "pretrained_model"
    model.save_pretrained(model_path)
    
    # Save optimizer state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_dir / "training_state.pt")
    
    # Log artifact
    logger.log_policy(checkpoint_dir)
    
    return checkpoint_dir


def train_with_logging(cfg, epochs: int = 5, batch_size: int = 32):
    """Main training loop with comprehensive logging."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = MockPolicy(input_dim=10, hidden_dim=64, output_dim=5).to(device)
    
    # Create datasets
    train_dataset = create_mock_dataset(n_samples=1000)
    val_dataset = create_mock_dataset(n_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Initialize logger
    logger = WandBLogger(cfg)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, 
            epoch, logger, cfg
        )
        
        # Evaluate
        val_loss, val_mae = evaluate(
            model, val_loader, device, epoch, logger, cfg
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Log epoch summary
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "epoch_time": epoch_time,
            "samples_per_second": len(train_dataset) / epoch_time,
        }
        logger.log_dict(epoch_metrics, step=epoch, mode="train")
        
        # Print progress
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            checkpoint_dir = save_checkpoint(
                model, optimizer, epoch, cfg.output_dir, logger, cfg
            )
            print(f"  ✓ Saved checkpoint: {checkpoint_dir.name}")
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    
    print(f"{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
    # Log final summary
    final_metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
    }
    logger.log_dict(final_metrics, step=epochs, mode="eval")


def test_rl_style_logging(cfg):
    """Test RL-style logging with custom step keys."""
    print(f"\n{'='*60}")
    print(f"Testing RL-style logging with custom steps")
    print(f"{'='*60}")
    
    logger = WandBLogger(cfg)
    
    # Simulate RL training with different step counters
    for episode in range(1, 11):
        # Simulate environment interaction
        episode_reward = 100 - episode * 2 + random.random() * 10
        episode_length = 50 + random.randint(-10, 10)
        
        # Log interaction metrics
        logger.log_dict({
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "success_rate": min(1.0, episode / 10),
        }, step=None, mode="train", custom_step_key="episode")
        
        # Simulate multiple training steps per episode
        for train_step in range(episode * 10, (episode + 1) * 10):
            policy_loss = 1.0 / (1 + train_step * 0.01)
            value_loss = 0.5 / (1 + train_step * 0.01)
            
            logger.log_dict({
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": 0.5 + random.random() * 0.1,
            }, step=None, mode="train", custom_step_key="training_step")
        
        # Log step counters
        logger.log_dict({
            "episode": episode,
            "training_step": episode * 10,
        }, step=None, mode="train", custom_step_key="episode")
        
        print(f"Episode {episode:3d} | Reward: {episode_reward:6.2f} | "
              f"Length: {episode_length:3d} | Training steps: {episode * 10}")
    
    print(f"✓ RL-style logging test completed")


def main():
    parser = argparse.ArgumentParser(
        description="Mock training script to test WandB/SwanLab logging"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["wandb", "swanlab", "both"],
        default="both",
        help="Which backend to test (default: both)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--test-rl",
        action="store_true",
        help="Also test RL-style logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    backends = ["wandb", "swanlab"] if args.backend == "both" else [args.backend]
    
    for backend in backends:
        print(f"\n{'#'*60}")
        print(f"# Testing {backend.upper()} Backend")
        print(f"{'#'*60}")
        
        # Create output directory
        output_dir = Path(
            args.output_dir if args.output_dir 
            else tempfile.mkdtemp(prefix=f"mock_train_{backend}_")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create config
        wandb_config = WandBConfig(
            enable=True,
            backend=backend,
            project="mock_training_test",
            entity=None,
            mode="offline" if backend == "wandb" else "offline",
            add_tags=True,
        )
        
        train_config = MockTrainConfig(
            output_dir=output_dir,
            job_name=f"mock_training_{backend}",
            wandb=wandb_config,
        )
        
        # Run training
        try:
            train_with_logging(
                train_config, 
                epochs=args.epochs,
                batch_size=32
            )
            
            # Optionally test RL-style logging
            if args.test_rl:
                test_rl_style_logging(train_config)
            
            print(f"\n✅ Successfully completed training with {backend} backend!")
            
        except Exception as e:
            print(f"\n❌ Training failed with {backend} backend!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            if args.backend != "both":
                raise
        
        print(f"\n{'#'*60}")
    
    print(f"\n{'='*60}")
    print(f"✅ All backend tests completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
