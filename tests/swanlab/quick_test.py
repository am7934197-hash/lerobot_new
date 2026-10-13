#!/usr/bin/env python
# Quick test to verify WandB/SwanLab backend switching works correctly

"""
快速测试脚本 - 验证 WandB/SwanLab 后端切换功能

运行方式:
    python quick_test.py wandb    # 测试 WandB
    python quick_test.py swanlab  # 测试 SwanLab
    python quick_test.py both     # 测试两者 (默认)
"""

import sys
from pathlib import Path
import tempfile

# 添加 lerobot 到路径
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import WandBConfig
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimpleConfig:
    """简化的配置用于快速测试"""
    output_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp()))
    job_name: str = "quick_test"
    seed: int = 42
    resume: bool = False
    wandb: WandBConfig = field(default_factory=WandBConfig)
    is_reward_model_training: bool = False
    dataset = type('obj', (object,), {'repo_id': 'test/dataset'})
    env = type('obj', (object,), {'type': 'test_env', 'fps': 30})
    policy = type('obj', (object,), {'type': 'test_policy'})
    
    def to_dict(self) -> dict:
        return {
            "job_name": self.job_name,
            "backend": self.wandb.backend,
        }


def test_backend(backend_name: str):
    """测试指定后端"""
    print(f"\n{'='*60}")
    print(f"测试 {backend_name.upper()} 后端")
    print(f"{'='*60}\n")
    
    # 创建配置
    wandb_config = WandBConfig(
        enable=True,
        backend=backend_name,
        project="backend_quick_test",
        mode="offline",
    )
    
    cfg = SimpleConfig(wandb=wandb_config)
    
    try:
        # 1. 初始化 logger
        print("1. 初始化 WandBLogger...")
        logger = WandBLogger(cfg)
        print(f"   ✓ Logger 初始化成功")
        print(f"   后端：{logger.backend}")
        print(f"   Run ID: {cfg.wandb.run_id}\n")
        
        # 2. 测试标量日志
        print("2. 测试标量指标日志...")
        for step in range(1, 4):
            logger.log_dict({
                "loss": 1.0 / step,
                "accuracy": 0.5 + step * 0.1,
            }, step=step, mode="train")
        print(f"   ✓ 成功记录 {3} 个 step 的指标\n")
        
        # 3. 测试 train/eval 分离
        print("3. 测试 train/eval 分离日志...")
        logger.log_dict({"train_loss": 0.5}, step=1, mode="train")
        logger.log_dict({"eval_loss": 0.4}, step=1, mode="eval")
        print(f"   ✓ 成功记录 train 和 eval 指标\n")
        
        # 4. 测试字符串日志
        print("4. 测试字符串指标日志...")
        logger.log_dict({
            "status": "testing",
            "device": "cpu",
        }, step=1, mode="train")
        print(f"   ✓ 成功记录字符串指标\n")
        
        # 5. 测试自定义步长
        print("5. 测试自定义步长跟踪...")
        logger.log_dict({
            "episode_reward": 100,
            "episode": 1,
        }, step=None, mode="train", custom_step_key="episode")
        print(f"   ✓ 成功记录自定义步长指标\n")
        
        print(f"{'='*60}")
        print(f"✅ {backend_name.upper()} 后端测试通过!")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ {backend_name.upper()} 后端测试失败!")
        print(f"错误：{e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "both"
    
    print(f"\n{'#'*60}")
    print(f"# WandB/SwanLab 后端快速测试")
    print(f"# 测试后端：{backend}")
    print(f"{'#'*60}\n")
    
    backends = ["wandb", "swanlab"] if backend == "both" else [backend]
    
    results = {}
    for b in backends:
        try:
            results[b] = test_backend(b)
        except Exception as e:
            print(f"\n⚠️  {b.upper()} 测试异常：{e}\n")
            results[b] = False
    
    # 打印总结
    print(f"\n{'#'*60}")
    print(f"# 测试总结")
    print(f"{'#'*60}\n")
    
    for b, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {b.upper()}: {status}")
    
    all_passed = all(results.values())
    print(f"\n总体结果：{'✅ 全部通过' if all_passed else '❌ 有失败'}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
