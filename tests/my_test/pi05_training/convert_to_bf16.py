#!/usr/bin/env python3
"""
将 PI05 checkpoint 从 float32 转换为 bfloat16 格式。

用法（在 lerobot conda 环境下）:
    uv run python tests/my_test/pi05_training/convert_to_bf16.py

    # 或者指定自定义路径:
    python convert_to_bf16.py --src ... --dst ...

输出:
    - model.safetensors: 所有 tensor → bfloat16（~50% 体积）
    - config.json: dtype 更新为 bfloat16
    - 其他文件（processor、train_config 等）原样复制
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert_checkpoint_bf16(src_dir: Path, dst_dir: Path) -> None:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source: {src_dir}")
    print(f"Target: {dst_dir}\n")

    # === 1. 转换 model.safetensors ===
    model_path = src_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {src_dir}")

    print(f"Loading model.safetensors ({os.path.getsize(model_path) / 1024**3:.1f} GB)...")
    tensors = load_file(str(model_path))
    converted_count = 0
    for key in tensors:
        if tensors[key].dtype == torch.float32:
            tensors[key] = tensors[key].to(torch.bfloat16)
            converted_count += 1

    dst_model_path = dst_dir / "model.safetensors"
    save_file(tensors, str(dst_model_path))
    print(f"  Converted {converted_count} tensors: float32 → bfloat16")
    print(f"  Saved: {dst_model_path} ({os.path.getsize(dst_model_path) / 1024**3:.1f} GB)\n")

    # === 2. 更新 config.json ===
    config_src = src_dir / "config.json"
    config_dst = dst_dir / "config.json"
    if config_src.exists():
        with open(config_src) as f:
            cfg = json.load(f)
        old_dtype = cfg.get("dtype", "unknown")
        cfg["dtype"] = "bfloat16"
        with open(config_dst, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  config.json: dtype {old_dtype} → bfloat16")

    # === 3. 复制其余文件（processor、train_config 等）===
    for fn in sorted(os.listdir(src_dir)):
        if fn in ("model.safetensors", "config.json"):
            continue
        src_file = src_dir / fn
        dst_file = dst_dir / fn
        if src_file.is_file():
            shutil.copy2(src_file, dst_file)
            print(f"  Copied: {fn}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PI05 checkpoint from float32 to bfloat16")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/mnt/data1/jizheng/vla_project/lerobot/tests/my_test/pi05_training/outputs/train/2026-07-30/14-03-03_pi05_3cam_painting_delta/checkpoints/last/pretrained_model"
        ),
        help="Source checkpoint dir (pretrained_model)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "/mnt/data1/jizheng/vla_project/lerobot/tests/my_test/pi05_training/outputs/train/2026-07-30/14-03-03_pi05_3cam_painting_delta/checkpoints/last/pretrained_model_16"
        ),
        help="Destination dir for bfloat16 checkpoint",
    )
    args = parser.parse_args()
    # 确保 dst 是 pretrained_model_16（用户原意）
    if args.dst.name != "pretrained_model_16":
        print(f"WARNING: dst basename is '{args.dst.name}', expected 'pretrained_model_16'. Adding suffix.")
        args.dst = args.dst.parent / f"{args.dst.name}_16"
    convert_checkpoint_bf16(args.src, args.dst)
