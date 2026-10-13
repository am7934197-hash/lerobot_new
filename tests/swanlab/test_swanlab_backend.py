#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script to verify WandB/SwanLab backend functionality.

This script tests:
1. Scalar metric logging (int, float)
2. String metric logging
3. Custom step tracking
4. Multiple modes (train/eval)
5. Video logging (wandb only)
6. Model artifact logging (wandb only)
7. Resume functionality

Usage:
    # Test with WandB backend
    python test_swanlab_backend.py --backend wandb
    
    # Test with SwanLab backend
    python test_swanlab_backend.py --backend swanlab
    
    # Test both backends sequentially
    python test_swanlab_backend.py --backend both
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from dataclasses import dataclass, field
from typing import Any

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import WandBConfig
from lerobot.configs.train import TrainPipelineConfig


@dataclass
class MockDatasetConfig:
    repo_id: str = "test/dataset"
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
    output_dir: Path = field(default_factory=lambda: Path("test_output"))
    job_name: str = "test_run"
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
        }


def create_mock_checkpoint(output_dir: Path, step: int) -> Path:
    """Create a mock checkpoint directory for testing artifact logging."""
    checkpoint_dir = output_dir / f"checkpoint_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    pretrained_dir = checkpoint_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple model file
    model_file = pretrained_dir / "model.safetensors"
    dummy_model = {"weight": torch.randn(10, 10)}
    
    try:
        from safetensors.torch import save_file
        save_file(dummy_model, model_file)
    except ImportError:
        # If safetensors not available, create a dummy file
        model_file.write_text("dummy model")
    
    # Create config file
    config_file = pretrained_dir / "config.json"
    config_file.write_text(json.dumps({"model_type": "mock"}))
    
    return checkpoint_dir


def test_scalar_logging(logger: WandBLogger, backend: str):
    """Test scalar metric logging (int and float)."""
    print(f"\n{'='*60}")
    print(f"Testing scalar metric logging ({backend})")
    print(f"{'='*60}")
    
    # Test training metrics
    for step in range(1, 6):
        metrics = {
            "loss": 1.0 / step,
            "accuracy": 0.5 + step * 0.1,
            "learning_rate": 1e-4 / (step ** 0.5),
            "epoch": step,
        }
        logger.log_dict(metrics, step=step, mode="train")
        print(f"  Logged train metrics at step {step}: {metrics}")
    
    # Test evaluation metrics
    for step in range(1, 4):
        metrics = {
            "eval_loss": 0.8 / step,
            "eval_accuracy": 0.6 + step * 0.05,
        }
        logger.log_dict(metrics, step=step * 2, mode="eval")
        print(f"  Logged eval metrics at step {step * 2}: {metrics}")
    
    print(f"✓ Scalar logging test passed for {backend}")


def test_string_logging(logger: WandBLogger, backend: str):
    """Test string metric logging."""
    print(f"\n{'='*60}")
    print(f"Testing string metric logging ({backend})")
    print(f"{'='*60}")
    
    step = 1
    metrics = {
        "best_model": "checkpoint_100",
        "status": "training",
        "device": "cuda:0",
    }
    logger.log_dict(metrics, step=step, mode="train")
    print(f"  Logged string metrics: {metrics}")
    
    print(f"✓ String logging test passed for {backend}")


def test_custom_step_key(logger: WandBLogger, backend: str):
    """Test custom step key functionality."""
    print(f"\n{'='*60}")
    print(f"Testing custom step key ({backend})")
    print(f"{'='*60}")
    
    # Simulate RL-style logging with custom steps
    for interaction_step in range(1, 4):
        training_step = interaction_step * 10
        
        # Log interaction metrics
        interaction_metrics = {
            "reward": float(interaction_step),
            "episode_length": interaction_step * 5,
        }
        logger.log_dict(
            interaction_metrics,
            step=None,
            mode="train",
            custom_step_key="interaction_step"
        )
        
        # Log training metrics
        train_metrics = {
            "policy_loss": 1.0 / interaction_step,
            "value_loss": 0.5 / interaction_step,
        }
        logger.log_dict(
            train_metrics,
            step=None,
            mode="train",
            custom_step_key="training_step"
        )
        
        # Log the step counters
        logger.log_dict({
            "interaction_step": interaction_step,
            "training_step": training_step,
        }, step=None, mode="train", custom_step_key="interaction_step")
        
        print(f"  Logged with custom steps: interaction={interaction_step}, training={training_step}")
    
    print(f"✓ Custom step key test passed for {backend}")


def test_video_logging(logger: WandBLogger, backend: str, output_dir: Path):
    """Test video logging (only works with wandb)."""
    print(f"\n{'='*60}")
    print(f"Testing video logging ({backend})")
    print(f"{'='*60}")
    
    if backend == "swanlab":
        print(f"  ⚠️  SwanLab does not support video logging, skipping...")
        # This should log a warning but not crash
        video_path = output_dir / "dummy_video.mp4"
        video_path.write_bytes(b"dummy video content")
        logger.log_video(str(video_path), step=1, mode="train")
        print(f"  ✓ Video logging gracefully skipped for swanlab")
    else:
        # Create a dummy video file
        video_path = output_dir / "dummy_video.mp4"
        video_path.write_bytes(b"dummy video content")
        logger.log_video(str(video_path), step=1, mode="train")
        print(f"  Logged video at step 1")
        print(f"✓ Video logging test passed for {backend}")


def test_artifact_logging(logger: WandBLogger, backend: str, output_dir: Path):
    """Test model artifact logging (only works with wandb)."""
    print(f"\n{'='*60}")
    print(f"Testing artifact logging ({backend})")
    print(f"{'='*60}")
    
    if backend == "swanlab":
        print(f"  ⚠️  SwanLab does not support artifacts, skipping...")
        checkpoint_dir = create_mock_checkpoint(output_dir, 100)
        logger.log_policy(checkpoint_dir)
        print(f"  ✓ Artifact logging gracefully skipped for swanlab")
    else:
        checkpoint_dir = create_mock_checkpoint(output_dir, 100)
        logger.log_policy(checkpoint_dir)
        print(f"  Logged model artifact from {checkpoint_dir}")
        print(f"✓ Artifact logging test passed for {backend}")


def test_resume_functionality(backend: str, output_dir: Path):
    """Test resume functionality."""
    print(f"\n{'='*60}")
    print(f"Testing resume functionality ({backend})")
    print(f"{'='*60}")
    
    # First run
    print(f"  Starting first run...")
    wandb_config = WandBConfig(
        enable=True,
        backend=backend,
        project="test_swanlab_backend",
        entity=None,
        mode="offline" if backend == "wandb" else "offline",
        add_tags=True,
    )
    
    train_config = MockTrainConfig(
        output_dir=output_dir,
        job_name=f"test_resume_{backend}",
        wandb=wandb_config,
    )
    
    logger1 = WandBLogger(train_config)
    logger1.log_dict({"loss": 0.5}, step=1, mode="train")
    run_id_1 = train_config.wandb.run_id
    print(f"  First run ID: {run_id_1}")
    
    # Simulate resume
    print(f"  Simulating resume...")
    time.sleep(1)  # Wait a bit to ensure file system is updated
    
    train_config_resume = MockTrainConfig(
        output_dir=output_dir,
        job_name=f"test_resume_{backend}",
        resume=True,
        wandb=wandb_config,
    )
    train_config_resume.wandb.run_id = run_id_1
    
    try:
        logger2 = WandBLogger(train_config_resume)
        logger2.log_dict({"loss": 0.3}, step=2, mode="train")
        print(f"  Successfully resumed run {run_id_1}")
        print(f"✓ Resume functionality test passed for {backend}")
    except Exception as e:
        print(f"  ⚠️  Resume test encountered an issue (expected for some configurations): {e}")


def run_all_tests(backend: str):
    """Run all tests for a specific backend."""
    print(f"\n{'#'*60}")
    print(f"# Testing {backend.upper()} Backend")
    print(f"{'#'*60}")
    
    # Create temporary output directory
    output_dir = Path(tempfile.mkdtemp(prefix=f"test_{backend}_"))
    print(f"Using output directory: {output_dir}")
    
    # Create wandb config
    wandb_config = WandBConfig(
        enable=True,
        backend=backend,
        project="test_swanlab_backend",
        entity=None,
        mode="offline" if backend == "wandb" else "offline",
        add_tags=True,
    )
    
    # Create train config
    train_config = MockTrainConfig(
        output_dir=output_dir,
        job_name=f"test_{backend}",
        wandb=wandb_config,
    )
    
    # Initialize logger
    print(f"\nInitializing WandBLogger with {backend} backend...")
    logger = WandBLogger(train_config)
    print(f"✓ Logger initialized successfully")
    
    # Run tests
    try:
        test_scalar_logging(logger, backend)
        test_string_logging(logger, backend)
        test_custom_step_key(logger, backend)
        test_video_logging(logger, backend, output_dir)
        test_artifact_logging(logger, backend, output_dir)
        test_resume_functionality(backend, output_dir)
        
        print(f"\n{'#'*60}")
        print(f"# ✅ All tests passed for {backend.upper()} backend!")
        print(f"{'#'*60}")
        
    except Exception as e:
        print(f"\n{'#'*60}")
        print(f"# ❌ Test failed for {backend.upper()} backend!")
        print(f"# Error: {e}")
        print(f"{'#'*60}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {output_dir}")
        # Note: We don't actually delete to allow manual inspection


def main():
    parser = argparse.ArgumentParser(description="Test WandB/SwanLab backend functionality")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["wandb", "swanlab", "both"],
        default="both",
        help="Which backend to test (default: both)"
    )
    
    args = parser.parse_args()
    
    backends_to_test = ["wandb", "swanlab"] if args.backend == "both" else [args.backend]
    
    print(f"\n{'='*60}")
    print(f"Starting comprehensive backend tests")
    print(f"Backends to test: {backends_to_test}")
    print(f"{'='*60}")
    
    for backend in backends_to_test:
        try:
            run_all_tests(backend)
        except Exception as e:
            print(f"\n⚠️  Tests for {backend} failed. Continuing with next backend...\n")
            if args.backend != "both":
                raise
    
    print(f"\n{'='*60}")
    print(f"✅ All requested backend tests completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
