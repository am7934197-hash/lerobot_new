#!/usr/bin/env python

"""Convert frame-segment JSON labels into FineART-VLA language columns.

The source is expected to be a LeRobot v3 dataset with one JSON file per
episode. Each JSON file contains a continuous ``segments`` timeline with
``f1`` (inclusive), ``f2`` (exclusive), and ``label`` fields.

The converter creates a new dataset and leaves the source untouched. Files are
hard-linked when possible, then the data Parquet files and metadata are replaced
atomically in the destination. This avoids duplicating the video files.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"


def _json_arrow_type() -> pa.DataType:
    # ``datasets.Json`` uses an Arrow JSON extension over string storage, but
    # PyArrow 25 cannot construct that extension inside list<struct<...>> from
    # Python values (ArrowNotImplementedError: extension). Parquet stores the
    # physical string type; Hugging Face Datasets applies its declared Json
    # feature when loading. This also works on older PyArrow releases.
    return pa.string()


PERSISTENT_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("camera", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_json_arrow_type()), nullable=True),
        ]
    )
)
EVENTS_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("camera", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_json_arrow_type()), nullable=True),
        ]
    )
)


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class EpisodeAnnotation:
    episode_index: int
    task: str
    segments: tuple[Segment, ...]
    parquet_path: Path
    timestamps: tuple[float, ...]

    @property
    def length(self) -> int:
        return len(self.timestamps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Source LeRobot v3 dataset root")
    parser.add_argument("output_root", type=Path, nargs="?", help="Destination dataset root")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory containing ep_NNN.json files (auto-detected by default)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate labels against source frames without creating a dataset",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _find_labels_dir(source_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        labels_dir = explicit.resolve()
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")
        return labels_dir

    preferred = source_root / "labels_json" / source_root.name
    if preferred.is_dir() and list(preferred.glob("ep_*.json")):
        return preferred

    label_files = list((source_root / "labels_json").rglob("ep_*.json"))
    parents = {path.parent.resolve() for path in label_files}
    if len(parents) != 1:
        raise ValueError(
            "Could not uniquely detect the labels directory. "
            "Pass --labels-dir explicitly; candidates were: " + ", ".join(map(str, sorted(parents)))
        )
    return parents.pop()


def _episode_data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    relative = template.format(
        chunk_index=episode_index // chunks_size,
        file_index=episode_index % chunks_size,
    )
    return root / relative


def _task_from_label(payload: dict[str, Any], path: Path) -> str:
    for key in ("main_task", "task", "instruction"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"No non-empty main_task/task/instruction in {path}")


def _validate_source(source_root: Path, labels_dir: Path) -> tuple[dict[str, Any], list[EpisodeAnnotation]]:
    info_path = source_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = _load_json(info_path)

    label_paths = sorted(labels_dir.glob("ep_*.json"))
    expected_episodes = int(info["total_episodes"])
    if len(label_paths) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} label files from meta/info.json, "
            f"found {len(label_paths)} in {labels_dir}"
        )

    annotations: list[EpisodeAnnotation] = []
    seen_episodes: set[int] = set()
    total_frames = 0
    for label_path in label_paths:
        payload = _load_json(label_path)
        episode_index = int(payload["ep"])
        if episode_index in seen_episodes:
            raise ValueError(f"Duplicate label for episode {episode_index}: {label_path}")
        seen_episodes.add(episode_index)

        parquet_path = _episode_data_path(source_root, info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Episode {episode_index} data file is missing: {parquet_path}")
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "timestamp"])
        length = table.num_rows
        total_frames += length

        episode_values = set(table["episode_index"].to_pylist())
        if episode_values != {episode_index}:
            raise ValueError(
                f"{parquet_path} has episode_index values {sorted(episode_values)}, "
                f"expected only {episode_index}"
            )
        frame_indices = table["frame_index"].to_pylist()
        if frame_indices != list(range(length)):
            raise ValueError(f"Episode {episode_index} frame_index must be contiguous from 0 to {length - 1}")
        timestamps = tuple(float(value) for value in table["timestamp"].to_pylist())
        if any(right < left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError(f"Episode {episode_index} timestamps are not monotonic")

        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"Episode {episode_index} has no segments")
        segments: list[Segment] = []
        expected_start = 0
        for position, raw in enumerate(raw_segments):
            if not isinstance(raw, dict):
                raise TypeError(f"Episode {episode_index} segment {position} is not an object")
            start = int(raw["f1"])
            end = int(raw["f2"])
            label = str(raw["label"]).strip()
            if start != expected_start or end <= start or end > length or not label:
                raise ValueError(
                    f"Invalid segment in episode {episode_index}: "
                    f"expected start {expected_start}, got [{start}, {end}) label={label!r}, length={length}"
                )
            segments.append(Segment(start=start, end=end, label=label))
            expected_start = end
        if expected_start != length:
            raise ValueError(
                f"Episode {episode_index} labels stop at frame {expected_start}, "
                f"but the episode has {length} frames"
            )

        declared_subtasks = payload.get("subtasks")
        segment_labels = [segment.label for segment in segments]
        if declared_subtasks is not None and list(declared_subtasks) != segment_labels:
            raise ValueError(
                f"Episode {episode_index} subtasks do not match segment labels: "
                f"{declared_subtasks!r} != {segment_labels!r}"
            )

        annotations.append(
            EpisodeAnnotation(
                episode_index=episode_index,
                task=_task_from_label(payload, label_path),
                segments=tuple(segments),
                parquet_path=parquet_path,
                timestamps=timestamps,
            )
        )

    expected_indices = set(range(expected_episodes))
    if seen_episodes != expected_indices:
        raise ValueError(
            f"Labels must cover episode indices 0..{expected_episodes - 1}; "
            f"missing={sorted(expected_indices - seen_episodes)}, "
            f"extra={sorted(seen_episodes - expected_indices)}"
        )
    if total_frames != int(info["total_frames"]):
        raise ValueError(
            f"Data contains {total_frames} frames, but meta/info.json declares {info['total_frames']}"
        )

    annotations.sort(key=lambda item: item.episode_index)
    return info, annotations


def _ordered_values(values: list[str]) -> tuple[list[str], dict[str, int]]:
    ordered = list(dict.fromkeys(values))
    return ordered, {value: index for index, value in enumerate(ordered)}


def _link_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK}:
            raise
        shutil.copy2(source, destination)


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.append_column(name, values) if index < 0 else table.set_column(index, name, values)


def _write_table_atomic(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.fineart.tmp")
    pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
    os.replace(temporary, path)


def _write_json_atomic(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.fineart.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def _convert(
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    annotations: list[EpisodeAnnotation],
) -> dict[str, Any]:
    task_values, task_to_index = _ordered_values([item.task for item in annotations])
    subtask_values, subtask_to_index = _ordered_values(
        [segment.label for item in annotations for segment in item.segments]
    )

    output_parent = output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.fineart-", dir=output_parent))
    staging_root = temporary_parent / output_root.name
    try:
        shutil.copytree(source_root, staging_root, copy_function=_link_or_copy)

        episode_report: list[dict[str, Any]] = []
        for annotation in annotations:
            relative_path = annotation.parquet_path.relative_to(source_root)
            output_path = staging_root / relative_path
            table = pq.read_table(output_path)
            length = table.num_rows

            subtask_indices: list[int] = []
            persistent_rows: list[dict[str, Any]] = []
            for segment in annotation.segments:
                subtask_index = subtask_to_index[segment.label]
                subtask_indices.extend([subtask_index] * (segment.end - segment.start))
                persistent_rows.append(
                    {
                        "role": "assistant",
                        "content": segment.label,
                        "style": "subtask",
                        "timestamp": annotation.timestamps[segment.start],
                        "camera": None,
                        "tool_calls": None,
                    }
                )

            table = _replace_column(
                table,
                "task_index",
                pa.array([task_to_index[annotation.task]] * length, type=pa.int64()),
            )
            table = _replace_column(
                table,
                "subtask_index",
                pa.array(subtask_indices, type=pa.int64()),
            )
            table = _replace_column(
                table,
                LANGUAGE_PERSISTENT,
                pa.array([persistent_rows] * length, type=PERSISTENT_TYPE),
            )
            table = _replace_column(
                table,
                LANGUAGE_EVENTS,
                pa.array([[] for _ in range(length)], type=EVENTS_TYPE),
            )
            _write_table_atomic(table, output_path)
            episode_report.append(
                {
                    "episode_index": annotation.episode_index,
                    "frames": length,
                    "task_index": task_to_index[annotation.task],
                    "segments": len(annotation.segments),
                }
            )

        tasks_table = pa.Table.from_pandas(
            pd.DataFrame(
                {"task_index": range(len(task_values))},
                index=pd.Index(task_values, name="task"),
            ),
            preserve_index=True,
        )
        subtasks_table = pa.Table.from_pandas(
            pd.DataFrame(
                {"subtask_index": range(len(subtask_values))},
                index=pd.Index(subtask_values, name="subtask"),
            ),
            preserve_index=True,
        )
        _write_table_atomic(tasks_table, staging_root / "meta" / "tasks.parquet")
        _write_table_atomic(subtasks_table, staging_root / "meta" / "subtasks.parquet")

        output_info = dict(info)
        output_info["features"] = dict(info["features"])
        output_info["features"][LANGUAGE_PERSISTENT] = {
            "dtype": "language",
            "shape": [1],
            "names": None,
        }
        output_info["features"][LANGUAGE_EVENTS] = {
            "dtype": "language",
            "shape": [1],
            "names": None,
        }
        output_info["total_tasks"] = len(task_values)
        # ``repo_id`` is supplied by DatasetConfig/LeRobotDataset at runtime;
        # it is not a supported DatasetInfo field in LeRobot v3.
        output_info.pop("repo_id", None)
        _write_json_atomic(output_info, staging_root / "meta" / "info.json")

        report = {
            "format": "fineart_vla_language_columns_v1",
            "converted_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "output_root": str(output_root),
            "labels": "segment f1/f2 -> persistent subtask timeline",
            "total_episodes": len(annotations),
            "total_frames": sum(item.length for item in annotations),
            "tasks": task_values,
            "subtasks": subtask_values,
            "episodes": episode_report,
        }
        _write_json_atomic(report, staging_root / "fineart_conversion_report.json")

        staging_root.replace(output_root)
        return report
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _validate_output(output_root: Path, report: dict[str, Any], annotations: list[EpisodeAnnotation]) -> None:
    info = _load_json(output_root / "meta" / "info.json")
    for feature in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS):
        if info["features"].get(feature, {}).get("dtype") != "language":
            raise ValueError(f"Output metadata is missing language feature {feature!r}")

    tasks = pd.read_parquet(output_root / "meta" / "tasks.parquet")
    if list(tasks.index) != report["tasks"] or tasks["task_index"].tolist() != list(range(len(tasks))):
        raise ValueError("Output task table does not match converted labels")
    subtasks = pd.read_parquet(output_root / "meta" / "subtasks.parquet")
    if list(subtasks.index) != report["subtasks"] or subtasks["subtask_index"].tolist() != list(
        range(len(subtasks))
    ):
        raise ValueError("Output subtask table does not match converted labels")

    for annotation in annotations:
        relative_path = annotation.parquet_path.relative_to(Path(report["source_root"]))
        table = pq.read_table(output_root / relative_path)
        for name in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS):
            if name not in table.column_names:
                raise ValueError(f"Output episode {annotation.episode_index} is missing {name}")
        persistent = table[LANGUAGE_PERSISTENT].to_pylist()
        events = table[LANGUAGE_EVENTS].to_pylist()
        if len(persistent) != annotation.length or any(row != persistent[0] for row in persistent):
            raise ValueError(
                f"Episode {annotation.episode_index} persistent timeline is not broadcast to every frame"
            )
        if len(persistent[0]) != len(annotation.segments):
            raise ValueError(
                f"Episode {annotation.episode_index} persistent timeline has the wrong segment count"
            )
        if any(row for row in events):
            raise ValueError(f"Episode {annotation.episode_index} unexpectedly contains language events")


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    labels_dir = _find_labels_dir(source_root, args.labels_dir)
    info, annotations = _validate_source(source_root, labels_dir)

    summary = {
        "source_root": str(source_root),
        "labels_dir": str(labels_dir),
        "total_episodes": len(annotations),
        "total_frames": sum(item.length for item in annotations),
        "tasks": list(dict.fromkeys(item.task for item in annotations)),
        "subtasks": list(
            dict.fromkeys(segment.label for item in annotations for segment in item.segments)
        ),
    }
    if args.validate_only:
        print(json.dumps({"status": "valid", **summary}, indent=2, ensure_ascii=False))
        return

    if args.output_root is None:
        raise ValueError("output_root is required unless --validate-only is used")
    output_root = args.output_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("output_root must be outside source_root so the source remains untouched")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    report = _convert(source_root, output_root, info, annotations)
    _validate_output(output_root, report, annotations)
    print(json.dumps({"status": "converted_and_validated", **report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
