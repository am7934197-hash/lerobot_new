#!/usr/bin/env python3
"""
Fix stats.json for stationary joints: widen q01/q99 to avoid noise amplification
in quantile normalization.

Problem: When q99-q01 is extremely small (e.g., 0.0001 for a joint that barely moves),
the quantile normalization `2.0*(x-q01)/(q99-q01)-1.0` amplifies tiny noise (e.g.,
state->action copy error) into huge values outside [-1, 1], dominating the training loss.

Solution: For joints where range < threshold, replace q01/q99 with mean ± half_range,
using a reasonable minimum range so noise stays within [-1, 1] after normalization.

Usage:
    python fix_stats_stationary_joints.py \
        /mnt/data2/dataset/data_jizheng/galbot_spray/spray_0708/50003_71684_lerobotv3/meta/stats.json

This creates a backup as stats.json.bak and overwrites the original.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Name mapping (index -> joint name), must match your dataset's feature order
JOINT_NAMES = [
    "right_arm_j1", "right_arm_j2", "right_arm_j3", "right_arm_j4",
    "right_arm_j5", "right_arm_j6", "right_arm_j7", "right_arm_grip",
    "left_arm_j1", "left_arm_j2", "left_arm_j3", "left_arm_j4",
    "left_arm_j5", "left_arm_j6", "left_arm_j7", "left_arm_grip",
    "leg_j1", "leg_j2", "leg_j3", "leg_j4", "leg_j5",
    "head_j1", "head_j2",
]

# Indices of joints we know should be stationary during single-arm operation
# STATIONARY_INDICES = list(range(8, 23))  # left_arm(8) + left_grip(1) + legs(5) + head(2) = 15 joints
STATIONARY_INDICES = list(range(0, 23))  # left_arm(8) + left_grip(1) + legs(5) + head(2) = 15 joints
# Minimum q99-q01 range for stationary joints.
# Noise amplitude is typically < 0.001 for state->action copy.
# With range=0.1, max normalized deviation = 2*0.001/0.1 = 0.02 (well within [-1,1]).
MIN_STATIONARY_RANGE = 0.1


def fix_stats(stats_path: str, stationary_indices: list[int], min_range: float, dry_run: bool = False):
    path = Path(stats_path)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        stats = json.load(f)

    features_to_fix = ["action", "observation.state"]

    for feat_name in features_to_fix:
        if feat_name not in stats:
            print(f"  SKIP {feat_name}: not in stats")
            continue

        feat_stats = stats[feat_name]
        q01_list = list(feat_stats["q01"])
        q99_list = list(feat_stats["q99"])
        mean_list = list(feat_stats["mean"])
        std_list = list(feat_stats["std"])

        modified = []
        for idx in stationary_indices:
            jname = JOINT_NAMES[idx]
            rng = q99_list[idx] - q01_list[idx]
            if rng < min_range:
                old_q01 = q01_list[idx]
                old_q99 = q99_list[idx]
                half_rng = min_range / 2.0
                q01_list[idx] = mean_list[idx] - half_rng
                q99_list[idx] = mean_list[idx] + half_rng
                new_rng = q99_list[idx] - q01_list[idx]
                modified.append(
                    f"    [{idx:2d}] {jname:20s}  "
                    f"q01: {old_q01:10.6f} -> {q01_list[idx]:10.6f}  "
                    f"q99: {old_q99:10.6f} -> {q99_list[idx]:10.6f}  "
                    f"range: {rng:.2e} -> {new_rng:.2e}  "
                    f"std: {std_list[idx]:.2e}"
                )

        feat_stats["q01"] = q01_list
        feat_stats["q99"] = q99_list

        print(f"\n[{feat_name}] {len(modified)} stationary joints with range < {min_range}:")
        if modified:
            for line in modified:
                print(line)
        else:
            print("  (none)")

    if dry_run:
        print("\n[Dry run] No changes written.")
        return

    # Backup
    backup_path = path.with_suffix(".json.bak")
    shutil.copy2(path, backup_path)
    print(f"\nBackup: {backup_path}")

    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Written: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Widen q01/q99 for stationary joints in stats.json"
    )
    parser.add_argument("stats_path", nargs="?",
                        default="/mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3_all_index/meta/stats.json",
                        help="Path to stats.json")
    parser.add_argument(
        "--min-range", type=float, default=MIN_STATIONARY_RANGE,
        help=f"Minimum q99-q01 range for stationary joints (default: {MIN_STATIONARY_RANGE})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument(
        "--stationary-indices",
        type=int, nargs="*",
        default=STATIONARY_INDICES,
        help="Indices of stationary joints (0-based, default: 8-22)"
    )
    args = parser.parse_args()

    print(f"Stats path:        {args.stats_path}")
    print(f"Min range:         {args.min_range}")
    print(f"Stationary indices: {args.stationary_indices}")
    print(f"Joint lookup:       {[JOINT_NAMES[i] for i in args.stationary_indices]}")

    fix_stats(args.stats_path, args.stationary_indices, args.min_range, args.dry_run)


if __name__ == "__main__":
    main()
