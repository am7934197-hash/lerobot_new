#!/usr/bin/env python
# ============================================================================
# 数据集维度过滤 + 摄像头过滤脚本
# ============================================================================
# 功能：
#   1. 只保留 arm + gripper 关节（去掉 leg 和 head 维度）
#      action/observation.state 从 23 dims → 16 dims
#   2. 只保留 head_right, left_arm, right_arm 三个摄像头（去掉 head_left）
#   3. 重算统计量（可选）
#
# 用法:
#   uv run python tests/my_test/pi05_training/filter_dataset_arms_only.py
# ============================================================================

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats, remove_feature
from lerobot.datasets.io_utils import write_json

# ============================================================================
# 配置
# ============================================================================

SOURCE_ROOT = Path("/mnt/data2/dataset/data_jizheng/galbot_spray/spray_0708/50003_71684_lerobotv3")

# 输出目录名后面追加 _arms_only
OUTPUT_ROOT = SOURCE_ROOT.with_name(SOURCE_ROOT.name + "_arms_only")

# 要保留的 action/state 维度索引（arms + grippers = 16 dims）
# 0-6: right_arm_joint1-7, 7: right_arm_gripper
# 8-14: left_arm_joint1-7, 15: left_arm_gripper
KEEP_DIM_INDICES = list(range(16))  # [0,1,...,15]

# 要移除的摄像头特征
REMOVE_CAMERAS = ["observation.images.head_left"]

# 重算统计量参数
CHUNK_SIZE = 50
EXCLUDE_JOINTS = ["gripper"]
NUM_WORKERS = 4

# ============================================================================
# 主流程
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def copy_videos_with_hardlinks(src: Path, dst: Path) -> None:
    """硬链接复制 videos 目录（不占额外磁盘空间）"""
    src_videos = src / "videos"
    if not src_videos.exists():
        return
    dst_videos = dst / "videos"
    for cam_dir in src_videos.iterdir():
        if not cam_dir.is_dir():
            continue
        cam_name = cam_dir.name
        if cam_name in REMOVE_CAMERAS:
            logger.info(f"  跳过摄像头: {cam_name}")
            continue
        dst_cam = dst_videos / cam_name
        dst_cam.mkdir(parents=True, exist_ok=True)
        for chunk_dir in cam_dir.iterdir():
            dst_chunk = dst_cam / chunk_dir.name
            dst_chunk.mkdir(exist_ok=True)
            for video_file in chunk_dir.iterdir():
                target = dst_chunk / video_file.name
                try:
                    os.link(video_file, target)
                except OSError:
                    shutil.copy2(video_file, target)


def slice_array_columns(table: pa.Table) -> pa.Table:
    """对 action 和 observation.state 列做维度切片，只保留 KEEP_DIM_INDICES"""
    new_columns = {}
    for col_name in table.column_names:
        col = table.column(col_name)
        if col_name in ("action", "observation.state"):
            # 每行是一个 list/array，取子集
            sliced = []
            for i in range(len(col)):
                val = col[i].as_py()
                if val is not None:
                    sliced.append(np.array(val)[KEEP_DIM_INDICES].tolist())
                else:
                    sliced.append(None)
            new_columns[col_name] = pa.array(sliced, type=pa.list_(pa.float32()))
        else:
            new_columns[col_name] = col
    return pa.table(new_columns)


if __name__ == "__main__":
    # --- 步骤 1: 清理旧输出 ---
    if OUTPUT_ROOT.exists():
        logger.info(f"删除旧目录: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    # --- 步骤 2: 复制并切片 parquet 数据 ---
    src_data = SOURCE_ROOT / "data"
    dst_data = OUTPUT_ROOT / "data"
    logger.info("处理 parquet 数据文件（切片 action/state 维度）...")
    for chunk_dir in sorted(src_data.iterdir()):
        if not chunk_dir.is_dir():
            continue
        dst_chunk = dst_data / chunk_dir.name
        dst_chunk.mkdir(parents=True, exist_ok=True)
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            table = pq.read_table(parquet_file)
            table = slice_array_columns(table)
            pq.write_table(table, dst_chunk / parquet_file.name)
    logger.info("parquet 数据切片完成")

    # --- 步骤 3: 硬链接复制视频（排除 head_left） ---
    logger.info(f"复制视频文件（排除: {REMOVE_CAMERAS}）...")
    copy_videos_with_hardlinks(SOURCE_ROOT, OUTPUT_ROOT)

    # --- 步骤 4: 复制并更新 meta 信息 ---
    src_meta = SOURCE_ROOT / "meta"
    dst_meta = OUTPUT_ROOT / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # 复制 episodes 和 tasks
    for meta_file in src_meta.glob("**/*"):
        rel = meta_file.relative_to(src_meta)
        target = dst_meta / rel
        if meta_file.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif meta_file.suffix == ".parquet":
            shutil.copy2(meta_file, target)

    # 更新 info.json: action/state shape → 16, names 只保留 arms+gripper
    with open(src_meta / "info.json") as f:
        info = json.load(f)

    full_names = info["features"]["action"]["names"]
    info["features"]["action"]["shape"] = [len(KEEP_DIM_INDICES)]
    info["features"]["action"]["names"] = [full_names[i] for i in KEEP_DIM_INDICES]
    info["features"]["observation.state"]["shape"] = [len(KEEP_DIM_INDICES)]
    info["features"]["observation.state"]["names"] = [full_names[i] for i in KEEP_DIM_INDICES]

    # 移除 head_left 特征
    for cam in REMOVE_CAMERAS:
        info["features"].pop(cam, None)

    write_json(info, dst_meta / "info.json")
    logger.info(f"已更新 info.json: action/state → {len(KEEP_DIM_INDICES)} dims")
    logger.info(f"保留的 action 关节: {info['features']['action']['names']}")

    # --- 步骤 5: 拷贝 stats.json ---
    shutil.copy2(src_meta / "stats.json", dst_meta / "stats.json")

    # --- 步骤 6: 使用 LeRobot 加载并重算 stats ---
    logger.info("重算统计量...")
    dataset = LeRobotDataset(repo_id=None, root=OUTPUT_ROOT)
    dataset = recompute_stats(
        dataset,
        skip_image_video=True,
        relative_action=False,  # 先用 absolute stats，后续 delta 训练时再算
        num_workers=NUM_WORKERS,
    )
    logger.info("统计量重算完成！")

    logger.info(f"\n数据集已生成: {OUTPUT_ROOT}")
    logger.info(f"  action/state 维度: 23 → {len(KEEP_DIM_INDICES)} (仅 arms+grippers)")
    logger.info(f"  摄像头: 已排除 {REMOVE_CAMERAS}")
    logger.info(f"\n如需 delta action 训练，再对 {OUTPUT_ROOT} 运行 recompute_relative_stats.py")
