#!/usr/bin/env python3
"""Convert Galbot spray dataset mcap episodes into a SINGLE LeRobot v3 dataset
containing ALL tasks, with full resume (checkpoint) support.

This is a variant of convert_galbot_dir_to_lerobot_joint.py: instead of
producing one dataset per task, every discovered task is merged into one
dataset. All episodes share a single global episode index, and each episode
carries a ``task_index`` referring to the combined ``meta/tasks.parquet``.

Source layout expected:
    spray/
        51398_76464/          # task_num=51398, recording=76464
            *.FIN.mcap
            *.SYNC.mcap
        51399_76465/          # different task 51399
            ...
        ...

Output layout (single dataset):
    {output_dir}/
        meta/info.json
        meta/stats.json
        meta/tasks.parquet               # ALL tasks; task_index = discovery order
        meta/subtasks.parquet
        meta/episodes/chunk-000/file-000.parquet
        data/chunk-000/file-{episode:03d}.parquet
        videos/{camera}/chunk-000/file-{episode:03d}.mp4
        progress.json                    # resume state, saved after every episode
        conversion_report.json

Resume: ``progress.json`` records every processed/failed episode identifier
(``folder/mcap_stem``) plus per-episode stats, so re-running the script skips
already-completed work and only processes the remaining episodes. Episode
indices and global frame indices continue from where the last run stopped.
New tasks discovered on a later run are appended to ``tasks.parquet`` with the
next available ``task_index``; existing indices never shift.

Usage::

    python tools/convert_galbot_dir_to_lerobot_joint_all_tasks.py \
        --source-root /mnt/data2/dataset/data_jizheng/galbot_spray/spray \
        --output-dir /mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3_all \
        --job-name-yaml tools/job_name.yaml \
        --fps 30

    # Quick test: only first 2 episodes per task
    python tools/convert_galbot_dir_to_lerobot_joint_all_tasks.py \
        --source-root /mnt/data2/dataset/data_jizheng/galbot_spray/spray \
        --output-dir /mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3_all \
        --job-name-yaml tools/job_name.yaml \
        --max-episodes-per-task 2
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse all low-level helpers (mcap extraction, alignment, video encoding,
# stats, metadata builders) from the per-task script.
from convert_galbot_dir_to_lerobot_joint import (
    CAMERA_TOPICS,
    JOINT_GROUPS,
    VideoInfo,
    _serialize_stats,
    align_and_write_episode,
    compute_lerobot_episode_stats,
    dataframe_from_index_mapping,
    discover_task_episodes,
    finalize_lerobot_stats,
    load_task_mapping,
    make_info_payload,
    make_lerobot_features,
    make_safe_dirname,
    mcap_identifier,
    parse_target_resolution,
    write_episode_data,
    write_json,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Progress / resume state
# ---------------------------------------------------------------------------
PROGRESS_DEFAULTS: dict[str, Any] = {
    "processed_identifiers": [],  # list of "{folder_name}/{mcap_stem}"
    "failed_identifiers": [],     # list of skipped/broken episodes
    "episode_rows": [],
    "episode_stats": [],          # serialized per-episode stats
    "total_frames": 0,
    "total_episodes": 0,
    "first_video_infos": None,
    "first_height": 0,
    "first_width": 0,
    "action_dim": 0,
    "state_dim": 0,
    "conversion_entries": [],
    "tasks": [],                  # ordered task names; list index == task_index
    "subtasks": ["complete task"],
}


def load_progress(output_dir: Path) -> dict[str, Any]:
    """Load progress.json from the output directory, or return empty state."""
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        for key, default in PROGRESS_DEFAULTS.items():
            state.setdefault(key, default)
        return state
    return dict(PROGRESS_DEFAULTS)


def save_progress(output_dir: Path, progress: dict[str, Any]) -> None:
    """Save progress state to progress.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "progress.json").open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def _deserialize_episode_stats(serialized_list: list[dict[str, Any]]) -> list[dict[str, dict[str, np.ndarray]]]:
    episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    for serialized in serialized_list:
        deserialized: dict[str, dict[str, np.ndarray]] = {}
        for feature_key, feature_stats in serialized.items():
            deserialized[feature_key] = {
                stat_key: np.array(value, dtype=np.float64)
                for stat_key, value in feature_stats.items()
            }
        episode_stats.append(deserialized)
    return episode_stats


# ===================================================================
# Convert all tasks into a single dataset
# ===================================================================

def convert_all_tasks(
    task_episodes: dict[str, list[tuple[Path, str]]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Convert every task's episodes into one LeRobot v3 dataset.

    Supports resume: already-processed episodes (tracked in progress.json)
    are skipped, and episode/frame indices continue where the last run
    stopped.
    """
    progress = load_progress(output_dir)
    processed_ids: set[str] = set(progress["processed_identifiers"])
    failed_ids: set[str] = set(progress["failed_identifiers"])

    # Task index assignment (stable across resumes: existing indices never move)
    tasks: list[str] = list(progress["tasks"])
    subtasks: list[str] = list(progress["subtasks"])
    task_index_map: dict[str, int] = {name: i for i, name in enumerate(tasks)}

    # folder_name -> task_name, to attribute processed identifiers to tasks
    folder_to_task: dict[str, str] = {
        folder_name: task for task, eps in task_episodes.items() for _, folder_name in eps
    }

    # Per-task already-done counts (for --max-episodes-per-task on resume)
    done_per_task: dict[str, int] = defaultdict(int)
    for ident in processed_ids:
        task = folder_to_task.get(ident.split("/", 1)[0])
        if task is not None:
            done_per_task[task] += 1

    # Build pending list: (task_name, task_index, mcap_path, folder_name)
    pending: list[tuple[str, int, Path, str]] = []
    for task_name, episodes in task_episodes.items():
        if task_name not in task_index_map:
            task_index_map[task_name] = len(tasks)
            tasks.append(task_name)
        task_idx = task_index_map[task_name]
        for mcap_path, folder_name in episodes:
            ident = mcap_identifier(folder_name, mcap_path)
            if ident in processed_ids or ident in failed_ids:
                continue
            if args.max_episodes_per_task is not None and done_per_task[task_name] >= args.max_episodes_per_task:
                continue
            pending.append((task_name, task_idx, mcap_path, folder_name))
            done_per_task[task_name] += 1

    if not pending:
        logging.info(
            "All %d episodes already processed; finalizing dataset.",
            len(processed_ids),
        )
        _finalize_dataset_output(output_dir, args, progress)
        return

    # Restore state from progress
    episode_rows: list[dict[str, Any]] = list(progress["episode_rows"])
    episode_stats_list = _deserialize_episode_stats(progress["episode_stats"])
    conversion_entries: list[dict[str, Any]] = list(progress["conversion_entries"])
    total_frames: int = progress["total_frames"]
    total_episodes: int = progress["total_episodes"]
    first_video_infos: dict[str, VideoInfo] | None = None
    if progress["first_video_infos"] is not None:
        first_video_infos = {
            k: VideoInfo(**v) for k, v in progress["first_video_infos"].items()
        }
    first_height: int = progress["first_height"]
    first_width: int = progress["first_width"]
    action_dim: int = progress["action_dim"]
    state_dim: int = progress["state_dim"]

    subtask_index_map: dict[str, int] = {name: i for i, name in enumerate(subtasks)}

    skipped_count = len(processed_ids)
    logging.info(
        "%d total episodes, %d already done, %d to process.",
        len(processed_ids) + len(pending), skipped_count, len(pending),
    )

    iterator = tqdm(pending, desc="Converting episodes") if tqdm else pending

    for task_name, task_idx, mcap_path, folder_name in iterator:
        ep_index = total_episodes
        ident = mcap_identifier(folder_name, mcap_path)

        logging.info(
            "Processing episode %d [%s]: %s (folder=%s)",
            ep_index, task_name, mcap_path.name, folder_name,
        )

        try:
            state, action, timestamp, frame_index, video_paths, video_infos, num_frames = \
                align_and_write_episode(
                    mcap_path, output_dir, ep_index, JOINT_GROUPS,
                    args.fps, skip_videos=args.skip_videos,
                    ffprobe_path=args.ffprobe_path,
                    target_size=args.target_size, resize_mode=args.resize_mode,
                )

            if state_dim == 0:
                state_dim = state.shape[1]
                action_dim = action.shape[1]
            elif state.shape[1] != state_dim or action.shape[1] != action_dim:
                raise ValueError(
                    f"Feature dimension changed in {mcap_path.name}: "
                    f"state={state.shape[1]}, action={action.shape[1]}"
                )

            if first_video_infos is None:
                first_video_infos = video_infos
                if video_infos:
                    first_height = next(iter(video_infos.values())).height
                    first_width = next(iter(video_infos.values())).width

            subtask_text = "complete task"
            if subtask_text not in subtask_index_map:
                subtask_index_map[subtask_text] = len(subtasks)
                subtasks.append(subtask_text)
            subtask_index = np.full(num_frames, subtask_index_map[subtask_text], dtype=np.int64)

            episode_index_array = np.full(num_frames, ep_index, dtype=np.int64)
            global_index = np.arange(total_frames, total_frames + num_frames, dtype=np.int64)
            task_index_arr = np.full(num_frames, task_idx, dtype=np.int64)

            data_path = output_dir / "data" / "chunk-000" / f"file-{ep_index:03d}.parquet"
            write_episode_data(
                data_path,
                state=state,
                action=action,
                subtask_index=subtask_index,
                timestamp=timestamp,
                frame_index=frame_index,
                episode_index=episode_index_array,
                index=global_index,
                task_index=task_index_arr,
            )

            if first_video_infos is None:
                first_height = 480
                first_width = 640
                first_video_infos = {}
            episode_features = make_lerobot_features(
                height=first_height,
                width=first_width,
                action_dim=action_dim,
                state_dim=state_dim,
                video_infos=first_video_infos,
                fps=args.fps,
            )
            ep_stats = compute_lerobot_episode_stats(
                features=episode_features,
                state=state,
                action=action,
                subtask_index=subtask_index,
                timestamp=timestamp,
                frame_index=frame_index,
                episode_index=episode_index_array,
                index=global_index,
                task_index=task_index_arr,
                video_paths=video_paths,
                num_frames=num_frames,
                ffprobe_path=args.ffprobe_path,
            )
            episode_stats_list.append(ep_stats)

            row: dict[str, Any] = {
                "episode_index": ep_index,
                "tasks": [task_name],
                "length": num_frames,
                "data/chunk_index": 0,
                "data/file_index": ep_index,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + num_frames,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for video_key in CAMERA_TOPICS:
                row[f"videos/{video_key}/chunk_index"] = 0
                row[f"videos/{video_key}/file_index"] = ep_index
                row[f"videos/{video_key}/from_timestamp"] = 0.0
                row[f"videos/{video_key}/to_timestamp"] = num_frames / float(args.fps)
            episode_rows.append(row)

            conversion_entries.append({
                "episode_index": ep_index,
                "source_episode_id": f"{folder_name}/{mcap_path.stem}",
                "frames": num_frames,
                "task": task_name,
                "source_folder": folder_name,
                "subtasks": sorted(set(subtask_index.tolist())),
            })

            total_frames += num_frames
            total_episodes += 1

            processed_ids.add(ident)
        except Exception:
            logging.exception(
                "Skipping broken episode %s (folder=%s)",
                mcap_path.name, folder_name,
            )
            failed_ids.add(ident)

        # Save progress after each episode (successful or failed)
        progress = {
            "processed_identifiers": sorted(processed_ids),
            "failed_identifiers": sorted(failed_ids),
            "episode_rows": episode_rows,
            "episode_stats": [_serialize_stats(s) for s in episode_stats_list],
            "total_frames": total_frames,
            "total_episodes": total_episodes,
            "first_video_infos": {
                k: {
                    "width": v.width, "height": v.height, "fps": v.fps,
                    "frames": v.frames, "duration": v.duration,
                    "codec": v.codec, "pix_fmt": v.pix_fmt, "has_audio": v.has_audio,
                }
                for k, v in (first_video_infos or {}).items()
            },
            "first_height": first_height,
            "first_width": first_width,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "conversion_entries": conversion_entries,
            "tasks": tasks,
            "subtasks": subtasks,
        }
        save_progress(output_dir, progress)

    # Finalize the single dataset
    _finalize_dataset_output(output_dir, args, progress)


def _finalize_dataset_output(
    output_dir: Path,
    args: argparse.Namespace,
    progress: dict[str, Any],
) -> None:
    """Write meta/info.json, meta/stats.json, meta/tasks.parquet, etc."""
    episode_rows = progress["episode_rows"]
    total_episodes = progress["total_episodes"]
    total_frames = progress["total_frames"]
    tasks: list[str] = progress["tasks"]
    subtasks: list[str] = progress["subtasks"]

    if total_episodes == 0:
        logging.info("No episodes converted; skipping finalization.")
        return

    episode_stats_np = _deserialize_episode_stats(progress["episode_stats"])

    first_video_infos: dict[str, VideoInfo] | None = None
    if progress["first_video_infos"] is not None:
        first_video_infos = {
            k: VideoInfo(**v) for k, v in progress["first_video_infos"].items()
        }
    first_height = progress["first_height"]
    first_width = progress["first_width"]
    action_dim = progress["action_dim"]
    state_dim = progress["state_dim"]

    if not first_video_infos:
        first_height = 480
        first_width = 640
        first_video_infos = {}

    features = make_lerobot_features(
        height=first_height,
        width=first_width,
        action_dim=action_dim,
        state_dim=state_dim,
        video_infos=first_video_infos,
        fps=args.fps,
    )

    repo_id = f"{args.repo_id_prefix}/{make_safe_dirname(output_dir.name)}"

    meta_dir = output_dir / "meta"
    write_json(
        meta_dir / "info.json",
        make_info_payload(
            repo_id=repo_id,
            robot_type=args.robot_type,
            fps=args.fps,
            features=features,
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=len(tasks),
            total_subtasks=len(subtasks),
        ),
    )
    write_json(meta_dir / "stats.json", finalize_lerobot_stats(episode_stats_np))
    dataframe_from_index_mapping(
        {name: i for i, name in enumerate(tasks)}, "task", "task_index"
    ).to_parquet(meta_dir / "tasks.parquet")
    dataframe_from_index_mapping(
        {name: i for i, name in enumerate(subtasks)}, "subtask", "subtask_index"
    ).to_parquet(meta_dir / "subtasks.parquet")

    episodes_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

    conversion_entries = progress.get("conversion_entries", [])
    per_task_counts: dict[str, int] = defaultdict(int)
    for entry in conversion_entries:
        per_task_counts[entry["task"]] += 1
    write_json(
        output_dir / "conversion_report.json",
        {
            "source_root": str(args.source_root),
            "output_dir": str(output_dir),
            "repo_id": repo_id,
            "fps": args.fps,
            "total_tasks": len(tasks),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "per_task_episode_counts": dict(per_task_counts),
            "videos": "extracted from mcap",
            "episodes": conversion_entries,
        },
    )

    logging.info(
        "Finalized dataset: %d tasks, %d episodes, %d frames → %s",
        len(tasks), total_episodes, total_frames, output_dir,
    )


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=False,
        default="/mnt/data2/dataset/data_jizheng/galbot_spray/spray",
        help="Root directory containing {task_num}_{recording_num}/ subdirectories with .mcap files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=False,
        default="/mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3_all_index",
        help="Output directory for the single merged dataset (created if missing).",
    )
    parser.add_argument(
        "--job-name-yaml", type=Path, required=False,
        default="/mnt/data1/jizheng/vla_project/galbot-g1/tools/job_name.yaml",
        help="Path to job_name.yaml mapping task numbers to names.",
    )
    parser.add_argument("--repo-id-prefix", default="local/galbot_spray")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="galbot_one_golf")
    parser.add_argument(
        "--max-episodes-per-task", type=int, default=None,
        help="Limit episodes per task (useful for quick tests).",
    )
    parser.add_argument(
        "--task-filter", default=None,
        help="Comma-separated list of task names to process (default: all).",
    )
    parser.add_argument(
        "--skip-videos", action="store_true",
        help="Write metadata/data only; skip video encoding.",
    )
    parser.add_argument(
        "--target-resolution", default="640x480",
        help="目标视频分辨率 'WIDTHxHEIGHT'（默认 '640x480'）。填 'keep'/'0x0' 保持相机原始分辨率。",
    )
    parser.add_argument(
        "--resize-mode", choices=["stretch", "pad"], default="stretch",
        help="缩放模式：stretch=直接拉伸填满目标尺寸（默认）；pad=等比缩放+居中补黑边（不畸变）。",
    )
    parser.add_argument(
        "--no-resize", action="store_true",
        help="不修改分辨率，保持相机原始分辨率（等价于 --target-resolution keep）。",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--ffprobe-path", type=Path, default=Path("ffprobe"),
        help="Path to ffprobe executable.",
    )
    args = parser.parse_args()
    args.target_size = None if args.no_resize else parse_target_resolution(args.target_resolution)
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    source_root = args.source_root
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    task_mapping = load_task_mapping(args.job_name_yaml)
    task_episodes = discover_task_episodes(source_root, task_mapping)

    if not task_episodes:
        raise FileNotFoundError(f"No episodes found under {source_root}")

    # Apply task filter
    if args.task_filter:
        wanted = set(t.strip() for t in args.task_filter.split(","))
        task_episodes = {k: v for k, v in task_episodes.items() if k in wanted}
        logging.info("Task filter applied: %d tasks remaining.", len(task_episodes))

    if not task_episodes:
        raise ValueError("No tasks to process after filtering.")

    # Summary
    total_eps = sum(len(eps) for eps in task_episodes.values())
    logging.info(
        "Discovered %d tasks, %d total episodes.",
        len(task_episodes), total_eps,
    )
    for name, eps in task_episodes.items():
        logging.info("  [%s]: %d episodes", name, len(eps))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_all_tasks(task_episodes, output_dir, args)

    logging.info("Done. Output: %s", output_dir)


if __name__ == "__main__":
    main()
