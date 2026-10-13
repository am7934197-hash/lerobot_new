#!/usr/bin/env python
# ============================================================================
# 为 Delta Action（Relative Action）训练重算数据集统计量
# ============================================================================
# 用法:
#   uv run python tests/my_test/pi05_training/recompute_relative_stats.py
#
# 原理:
#   当 use_relative_actions=true 时，训练时模型看到的 action 是 action - state（相对值）。
#   数据集原有的 stats（均值/标准差等）是基于绝对 action 计算的，与模型实际看到的
#   分布不一致，可能导致归一化效果不理想。
#
#   此脚本将数据集复制到新目录，然后在副本上调用 LeRobot 的 recompute_stats()，
#   用 relative = action - state 来重算统计量。原始数据集不受影响。
#
#   ⚠️ 数据文件（parquet/video）使用硬链接复制，不占用额外磁盘空间。
# ============================================================================

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats

# ============================================================================
# 配置区域
# ============================================================================

# 源数据集根目录
SOURCE_ROOT = "/mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3_all_index"

# 输出目录（将在源目录同级创建，末尾追加 _relative_stats）
# 例如: 50003_71684_lerobotv3_relative_stats
OUTPUT_ROOT = Path(SOURCE_ROOT).with_name(Path(SOURCE_ROOT).name + "_relative_stats_50")

# 动作块大小（与训练配置中的 policy.chunk_size 一致）
CHUNK_SIZE = 50

# 在相对动作中保持为绝对值的关节名（与配置中的 relative_exclude_joints 一致）
EXCLUDE_JOINTS = ["gripper"]

# 并行线程数（1 = 单线程，越大越快但占内存）
NUM_WORKERS = 4

# ============================================================================
# 主流程
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def copy_with_hardlinks(src: Path, dst: Path) -> None:
    """递归复制目录，数据文件用硬链接，其余文件正常复制。

    硬链接不占额外磁盘空间，且修改 meta_data/stats.json 不会影响原目录。
    """
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_with_hardlinks(item, target)
        elif item.is_symlink():
            target.symlink_to(os.readlink(item))
        elif item.suffix in (".parquet", ".mp4", ".avi", ".mov", ".webm"):
            # 视频和 parquet 数据文件用硬链接（不复制内容）
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
        else:
            shutil.copy2(item, target)


if __name__ == "__main__":
    source_root = Path(SOURCE_ROOT)

    # 1. 复制数据集到新目录（硬链接数据文件）
    if OUTPUT_ROOT.exists():
        logger.info(f"输出目录已存在，删除旧目录: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    logger.info(f"复制数据集（硬链接模式）:\n  源: {source_root}\n  目标: {OUTPUT_ROOT}")
    copy_with_hardlinks(source_root, OUTPUT_ROOT)
    logger.info("复制完成")

    # 2. 加载副本数据集
    dataset = LeRobotDataset(repo_id=None, root=OUTPUT_ROOT)

    # 3. 重算统计量（基于相对动作）
    logger.info(
        f"开始重算统计量（relative_action=True, chunk_size={CHUNK_SIZE}, "
        f"exclude_joints={EXCLUDE_JOINTS}）..."
    )
    dataset = recompute_stats(
        dataset,
        skip_image_video=True,  # 只重算数值特征（action/state 等）
        relative_action=True,
        relative_exclude_joints=EXCLUDE_JOINTS,
        chunk_size=CHUNK_SIZE,
        num_workers=NUM_WORKERS,
    )
    # recompute_stats 内部已调用 write_stats() 写入 stats.json，无需手动 save

    stats_path = OUTPUT_ROOT / "meta" / "stats.json"
    logger.info(f"重算完成！新的 stats.json: {stats_path}")
    logger.info(f"\n请将训练配置中的 dataset.root 改为:\n  {OUTPUT_ROOT}")
