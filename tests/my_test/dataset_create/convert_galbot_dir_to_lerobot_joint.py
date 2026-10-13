#!/usr/bin/env python3
"""Convert Galbot spray dataset mcap episodes into LeRobot v3 format,
grouping episodes by task and supporting resume.

Source layout expected:
    spray/
        51398_76464/          # task_num=51398, recording=76464
            *.FIN.mcap
            *.SYNC.mcap
        51398_88429/          # same task 51398, different recording
            *.FIN.mcap
            ...
        51399_76465/          # different task 51399
            ...

Task mapping: job_name.yaml maps the first number (e.g. 51398) to a
human-readable task name. Folders sharing the same prefix belong to the
same task and are merged into a single LeRobot dataset.

Resume: each task maintains a progress.json inside its output directory.
Re-running the script skips already-processed episodes and only appends
new ones. Stats are re-aggregated from all episodes at the end.

Usage::

    python tools/convert_galbot_dir_to_lerbot_joint.py \
        --source-root /mnt/data2/dataset/data_jizheng/galbot_spray/spray \
        --output-base /mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3 \
        --job-name-yaml tools/job_name.yaml \
        --fps 30

    # Quick test: only first 2 episodes per task
    python tools/convert_galbot_dir_to_lerbot_joint.py \
        --source-root /mnt/data2/dataset/data_jizheng/galbot_spray/spray \
        --output-base /mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3 \
        --job-name-yaml tools/job_name.yaml \
        --max-episodes-per-task 2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.reader import SeekingReader

from lerobot.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Constants (shared with convert_galbot_mcap_to_lerobot_joint.py)
# ---------------------------------------------------------------------------
CODEBASE_VERSION = "v3.0"
SKIP_EMPTY_ACTIONS = True
OBS_STATE = "observation.state"
ACTION = "action"

CAMERA_TOPICS = {
    "observation.images.head_left": "/front_head_camera/left_color/image_raw",
    "observation.images.head_right": "/front_head_camera/right_color/image_raw",
    "observation.images.left_arm": "/left_arm_camera/color/image_raw",
    "observation.images.right_arm": "/right_arm_camera/color/image_raw",
}

JOINT_GROUPS = ["right_arm", "right_gripper", "left_arm", "left_gripper", "leg", "head"]
DIM_MAPS = {
    "right_arm": 7,
    "right_gripper": 1,
    "left_arm": 7,
    "left_gripper": 1,
    "leg": 5,
    "head": 2,
}
SENSOR_TOPICS = ["singorix/wbcs/sensor"]
ACTION_TOPICS = ["singorix/wbcs/target"]

ARM_STATE_ACTION_NAMES = [
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
    "right_arm_gripper",
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
    "left_arm_gripper",
    "leg_joint1",
    "leg_joint2",
    "leg_joint3",
    "leg_joint4",
    "leg_joint5",
    "head_joint1",
    "head_joint2",
]

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


# ---------------------------------------------------------------------------
# VideoInfo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    codec: str
    pix_fmt: str
    has_audio: bool


# ---------------------------------------------------------------------------
# ffprobe helper
# ---------------------------------------------------------------------------
def parse_rate(rate: str) -> float:
    if "/" not in rate:
        return float(rate)
    numerator, denominator = rate.split("/", maxsplit=1)
    denominator_value = float(denominator)
    if denominator_value == 0:
        return 0.0
    return float(numerator) / denominator_value


def ffprobe_video(path: Path, ffprobe_path: Path = Path("ffprobe")) -> VideoInfo:
    command = [
        str(ffprobe_path), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {path}")

    stream = streams[0]
    fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    frames = int(stream.get("nb_frames") or round(duration * fps))

    audio_command = [
        str(ffprobe_path), "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index",
        "-of", "json",
        str(path),
    ]
    audio_payload = json.loads(subprocess.check_output(audio_command, text=True))

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frames=frames,
        duration=duration,
        codec=str(stream.get("codec_name") or ""),
        pix_fmt=str(stream.get("pix_fmt") or ""),
        has_audio=bool(audio_payload.get("streams")),
    )


def video_feature_info(info: VideoInfo, fps: int) -> dict[str, Any]:
    return {
        "video.height": info.height,
        "video.width": info.width,
        "video.codec": info.codec,
        "video.pix_fmt": info.pix_fmt,
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": info.has_audio,
    }


# ---------------------------------------------------------------------------
# Protobuf helpers
# ---------------------------------------------------------------------------
def _build_descriptor_pool(reader) -> descriptor_pool.DescriptorPool:
    summary = reader.get_summary()
    pool = descriptor_pool.DescriptorPool()
    all_fds: list[descriptor_pb2.FileDescriptorProto] = []
    for sid, schema in summary.schemas.items():
        if schema.encoding == "protobuf":
            fds = descriptor_pb2.FileDescriptorSet()
            fds.ParseFromString(schema.data)
            all_fds.extend(fds.file)

    added: set[str] = set()

    def _add_fd(fd: descriptor_pb2.FileDescriptorProto) -> None:
        if fd.name in added:
            return
        for dep in fd.dependency:
            if dep not in added:
                for other in all_fds:
                    if other.name == dep:
                        _add_fd(other)
                        break
        pool.Add(fd)
        added.add(fd.name)

    for fd in all_fds:
        _add_fd(fd)
    return pool


def _decode_message(
    message_data: bytes,
    schema_name: str,
    pool: descriptor_pool.DescriptorPool,
) -> Any:
    desc = pool.FindMessageTypeByName(schema_name)
    cls = message_factory.GetMessageClass(desc)
    msg = cls()
    msg.ParseFromString(message_data)
    return msg


def _msg_ts_ns(msg) -> int:
    ts = msg.header.timestamp
    return int(ts.sec) * 1_000_000_000 + int(ts.nanosec)


def _decode_image_bgr_from_message(msg) -> np.ndarray | None:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


# ---------------------------------------------------------------------------
# Sensor / action extraction
# ---------------------------------------------------------------------------
def _extract_sensor_vector(msg, groups: list[str]) -> tuple[int, np.ndarray]:
    ts = _msg_ts_ns(msg)
    source_map = msg.joint_sensor_map
    parts: list[list[float]] = []
    for group_name in groups:
        if group_name not in source_map:
            parts.append([0.0] * DIM_MAPS.get(group_name, 0))
            continue
        group = source_map[group_name]
        parts.append(list(group.position))
    flat = [v for sub in parts for v in sub]
    return ts, np.asarray(flat, dtype=np.float32)


def _extract_target_group_vector(msg) -> list[tuple[int, str, np.ndarray]]:
    ts = _msg_ts_ns(msg)
    source_map = msg.target_group_trajectory_map
    if not source_map:
        if SKIP_EMPTY_ACTIONS:
            return []
        raise ValueError("Target message has empty target_group_trajectory_map")
    results: list[tuple[int, str, np.ndarray]] = []
    for group_name in source_map:
        group = source_map[group_name]
        if not group.group_commands:
            raise ValueError(f"Target group {group_name} has no group_commands")
        cmd = group.group_commands[0]
        positions = [jc.position for jc in cmd.joint_commands]
        results.append((ts, group_name, np.asarray(positions, dtype=np.float32)))
    return results


def extract_all_from_mcap(
    mcap_path: Path,
    joint_groups: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, list[tuple[int, np.ndarray]]],
    dict[str, list[tuple[int, np.ndarray]]],
    dict[str, tuple[int, int]],
]:
    sensor_ts_list: list[int] = []
    sensor_pos_list: list[np.ndarray] = []
    target_group_data: dict[str, list[tuple[int, np.ndarray]]] = {g: [] for g in joint_groups}
    camera_data: dict[str, list[tuple[int, np.ndarray]]] = {}
    camera_shapes: dict[str, tuple[int, int]] = {}

    with open(mcap_path, "rb") as f:
        reader = SeekingReader(f, record_size_limit=None)
        pool = _build_descriptor_pool(reader)
        schemas_by_id = {sid: s for sid, s in reader.get_summary().schemas.items()}

        for schema_obj, channel, message in reader.iter_messages(topics=SENSOR_TOPICS):
            schema = schemas_by_id[channel.schema_id]
            msg = _decode_message(message.data, schema.name, pool)
            ts, pos = _extract_sensor_vector(msg, joint_groups)
            sensor_ts_list.append(ts)
            sensor_pos_list.append(pos)

        for schema_obj, channel, message in reader.iter_messages(topics=ACTION_TOPICS):
            schema = schemas_by_id[channel.schema_id]
            msg = _decode_message(message.data, schema.name, pool)
            for ts, group_name, pos in _extract_target_group_vector(msg):
                if group_name in target_group_data:
                    target_group_data[group_name].append((ts, pos))

        for topic in CAMERA_TOPICS.values():
            camera_data[topic] = []
            for schema_obj, channel, message in reader.iter_messages(topics=[topic]):
                schema = schemas_by_id[channel.schema_id]
                msg = _decode_message(message.data, schema.name, pool)
                img = _decode_image_bgr_from_message(msg)
                if img is None:
                    logging.warning("Failed to decode frame for %s", topic)
                    continue
                if topic not in camera_shapes:
                    camera_shapes[topic] = (int(img.shape[0]), int(img.shape[1]))
                try:
                    ts = _msg_ts_ns(msg)
                except Exception:
                    ts = message.log_time
                camera_data[topic].append((ts, img))

    if not sensor_ts_list:
        raise ValueError(f"No sensor messages found in {mcap_path}")

    sensor_ts = np.asarray(sensor_ts_list, dtype=np.int64)
    sensor_arr = np.stack(sensor_pos_list)

    return sensor_ts, sensor_arr, target_group_data, camera_data, camera_shapes


# ---------------------------------------------------------------------------
# Timestamp alignment
# ---------------------------------------------------------------------------
def nearest_indices_by_timestamp(target_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
    target_ts = np.asarray(target_ts, dtype=np.int64)
    source_ts = np.asarray(source_ts, dtype=np.int64)
    if source_ts.ndim != 1 or target_ts.ndim != 1:
        raise ValueError("Timestamp arrays must be one-dimensional.")
    if len(source_ts) == 0:
        raise ValueError("Cannot align against an empty timestamp array.")

    order = np.argsort(source_ts, kind="stable")
    sorted_ts = source_ts[order]
    insertion = np.searchsorted(sorted_ts, target_ts, side="left")
    right = np.clip(insertion, 0, len(sorted_ts) - 1)
    left = np.clip(insertion - 1, 0, len(sorted_ts) - 1)

    left_delta = np.abs(target_ts - sorted_ts[left])
    right_delta = np.abs(sorted_ts[right] - target_ts)
    sorted_indices = np.where(left_delta <= right_delta, left, right)
    return order[sorted_indices]


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------
def parse_target_resolution(value: str) -> tuple[int, int] | None:
    """解析分辨率参数 'WIDTHxHEIGHT' → (width, height)。

    'keep' / 'none' / '0x0' / 空 → None（保持相机原始分辨率，不做修改）。
    """
    text = str(value or "").strip().lower()
    if text in ("keep", "none", "0x0", ""):
        return None
    match = re.fullmatch(r"(\d+)\s*[x×]\s*(\d+)", text)
    if not match:
        raise ValueError(
            f"Invalid target resolution: {value!r} "
            f"(expected 'WIDTHxHEIGHT' e.g. '640x480', or 'keep' to disable)"
        )
    return int(match.group(1)), int(match.group(2))


def resize_bgr_frame(img: np.ndarray, target_size: tuple[int, int], resize_mode: str) -> np.ndarray:
    """把一帧 BGR 图像缩放到 target_size=(width, height)。

    resize_mode:
      - "stretch": 直接拉伸填满目标尺寸（宽高比不同时会畸变）
      - "pad": 等比缩放 fit-within + 居中补黑边，不畸变、不裁切
    """
    target_w, target_h = target_size
    src_h, src_w = img.shape[:2]
    if (src_w, src_h) == (target_w, target_h):
        return img

    if resize_mode == "pad":
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0, y0 = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    interpolation = (
        cv2.INTER_AREA if (target_w < src_w and target_h < src_h) else cv2.INTER_LINEAR
    )
    return cv2.resize(img, (target_w, target_h), interpolation=interpolation)


def encode_bgr_frames_h264(
    frames_bgr: list[np.ndarray],
    dst: Path,
    fps: float,
    crf: int = 30,
    g: int = 2,
    target_size: tuple[int, int] | None = None,
    resize_mode: str = "stretch",
) -> None:
    """Encode a list of BGR frames to an h264 mp4 using PyAV.

    If ``target_size`` is not None, every frame is resized to (width, height)
    before encoding (see :func:`resize_bgr_frame` for the resize strategy).
    """
    import av

    dst.parent.mkdir(parents=True, exist_ok=True)
    if target_size is not None:
        width, height = target_size
    else:
        height, width = frames_bgr[0].shape[:2]
    options = {"crf": str(crf), "g": str(g)}

    logging.getLogger("libav").setLevel(av.logging.WARNING)
    with av.open(str(dst), "w") as output:
        stream = output.add_stream("h264", int(round(fps)), options=options)
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        for img in frames_bgr:
            if target_size is not None:
                img = resize_bgr_frame(img, target_size, resize_mode)
            rgb = img[:, :, ::-1]
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    if not dst.exists():
        raise OSError(f"Video encoding failed, file not found: {dst}")


# ---------------------------------------------------------------------------
# Episode extraction + video writing
# ---------------------------------------------------------------------------
def align_and_write_episode(
    mcap_path: Path,
    output_root: Path,
    ep_index: int,
    joint_groups: list[str],
    fps: float,
    skip_videos: bool = False,
    ffprobe_path: Path = Path("ffprobe"),
    target_size: tuple[int, int] | None = None,
    resize_mode: str = "stretch",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Path], dict[str, VideoInfo], int]:
    sensor_ts, sensor_arr, target_group_data, camera_data, camera_shapes = extract_all_from_mcap(
        mcap_path, joint_groups
    )

    # Determine master camera (most frames)
    master_topic = max(camera_data, key=lambda t: len(camera_data[t]))
    master_items = camera_data[master_topic]
    num_frames = len(master_items)
    master_ts = np.asarray([ts for ts, _ in master_items], dtype=np.int64)

    logging.info(
        "Episode %s: master=%s (%d frames), sensor=%d, targets=%s",
        mcap_path.name, master_topic, num_frames,
        len(sensor_ts), {k: len(v) for k, v in target_group_data.items()},
    )

    # Align sensor → master camera
    sensor_nearest = nearest_indices_by_timestamp(master_ts, sensor_ts)
    state = sensor_arr[sensor_nearest]

    # Build action by aligning each target group independently
    group_slices: dict[str, slice] = {}
    idx = 0
    for group_name in joint_groups:
        d = DIM_MAPS[group_name]
        group_slices[group_name] = slice(idx, idx + d)
        idx += d

    action = state.copy()  # hold-position fallback
    for group_name in joint_groups:
        group_items = target_group_data[group_name]
        if not group_items:
            logging.info("No target data for group %s in %s", group_name, mcap_path.name)
            continue
        group_ts = np.asarray([ts for ts, _ in group_items], dtype=np.int64)
        group_pos = np.stack([pos for _, pos in group_items])
        nearest = nearest_indices_by_timestamp(master_ts, group_ts)
        aligned = group_pos[nearest]
        action[:, group_slices[group_name]] = aligned

    # Align cameras and write videos
    video_paths: dict[str, Path] = {}
    video_infos: dict[str, VideoInfo] = {}
    topic_to_key = {v: k for k, v in CAMERA_TOPICS.items()}

    for topic, key in topic_to_key.items():
        items = camera_data[topic]
        if not items:
            raise ValueError(f"No frames for camera topic {topic}")
        topic_ts = np.asarray([ts for ts, _ in items], dtype=np.int64)
        frames = [img for _, img in items]

        if topic == master_topic:
            aligned_frames = frames
        else:
            nearest = nearest_indices_by_timestamp(master_ts, topic_ts)
            aligned_frames = [frames[i] for i in nearest]

        if not skip_videos:
            dst = output_root / "videos" / key / "chunk-000" / f"file-{ep_index:03d}.mp4"
            if len(aligned_frames) == 0:
                raise ValueError(f"No frames written for {topic}")
            encode_bgr_frames_h264(
                aligned_frames, dst, fps, target_size=target_size, resize_mode=resize_mode
            )

            info = ffprobe_video(dst, ffprobe_path=ffprobe_path)
            video_paths[key] = dst
            video_infos[key] = info

    # Timestamps on constant-fps grid
    frame_index = np.arange(num_frames, dtype=np.int64)
    timestamp_sec = (frame_index / float(fps)).astype(np.float32)

    return state, action, timestamp_sec, frame_index, video_paths, video_infos, num_frames


# ---------------------------------------------------------------------------
# LeRobot metadata helpers
# ---------------------------------------------------------------------------
def vector_names(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(dim)]


def state_action_names(dim: int) -> list[str]:
    if dim == len(ARM_STATE_ACTION_NAMES):
        return ARM_STATE_ACTION_NAMES
    return vector_names("value", dim)


def make_lerobot_features(
    height: int,
    width: int,
    action_dim: int,
    state_dim: int,
    video_infos: dict[str, VideoInfo] | None = None,
    fps: int = 30,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_action_names(state_dim),
        },
        ACTION: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": state_action_names(action_dim),
        },
        "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
    }

    for video_key in CAMERA_TOPICS:
        info = video_infos.get(video_key) if video_infos else None
        camera_height = info.height if info else height
        camera_width = info.width if info else width
        features[video_key] = {
            "dtype": "video",
            "shape": (camera_height, camera_width, 3),
            "names": ["height", "width", "channels"],
            "info": video_feature_info(info, fps) if info else None,
        }

    features.update(DEFAULT_FEATURES)
    return features


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def fixed_size_list(values: np.ndarray, dtype: pa.DataType) -> pa.FixedSizeListArray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D array for fixed size list column, got {array.shape}")
    flat = pa.array(array.reshape(-1), type=dtype)
    return pa.FixedSizeListArray.from_arrays(flat, list_size=array.shape[1])


def write_episode_data(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    subtask_index: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    index: np.ndarray,
    task_index: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            OBS_STATE: fixed_size_list(state.astype(np.float32, copy=False), pa.float32()),
            ACTION: fixed_size_list(action.astype(np.float32, copy=False), pa.float32()),
            "subtask_index": pa.array(subtask_index.astype(np.int64, copy=False), type=pa.int64()),
            "timestamp": pa.array(timestamp.astype(np.float32, copy=False), type=pa.float32()),
            "frame_index": pa.array(frame_index.astype(np.int64, copy=False), type=pa.int64()),
            "episode_index": pa.array(episode_index.astype(np.int64, copy=False), type=pa.int64()),
            "index": pa.array(index.astype(np.int64, copy=False), type=pa.int64()),
            "task_index": pa.array(task_index.astype(np.int64, copy=False), type=pa.int64()),
        }
    )
    pq.write_table(table, path, compression="snappy", use_dictionary=True)


def _read_next_video_frame_rgb(cap: Any, video_path: Path, frame_index: int) -> np.ndarray:
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def compute_video_stats_from_path(video_path: Path, num_frames: int) -> dict[str, np.ndarray]:
    if num_frames <= 0:
        raise ValueError(f"Cannot compute video stats for {num_frames} frames: {video_path}")
    sampled_indices = sample_indices(num_frames)
    sample_positions: dict[int, list[int]] = {}
    for output_index, frame_index in enumerate(sampled_indices):
        sample_positions.setdefault(int(frame_index), []).append(output_index)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    images = None
    try:
        for frame_index in range(max(sample_positions) + 1):
            frame_rgb = _read_next_video_frame_rgb(cap, video_path, frame_index)
            if frame_index not in sample_positions:
                continue
            image = np.transpose(frame_rgb, (2, 0, 1))
            image = auto_downsample_height_width(image)
            if images is None:
                images = np.empty((len(sampled_indices), *image.shape), dtype=np.uint8)
            for output_index in sample_positions[frame_index]:
                images[output_index] = image
    finally:
        cap.release()

    if images is None:
        raise ValueError(f"Cannot compute video stats for empty video: {video_path}")

    stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
    return {
        key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
        for key, value in stats.items()
    }


def compute_lerobot_episode_stats(
    *,
    features: dict[str, dict[str, Any]],
    state: np.ndarray,
    action: np.ndarray,
    subtask_index: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    index: np.ndarray,
    task_index: np.ndarray,
    video_paths: dict[str, Path],
    num_frames: int,
    ffprobe_path: Path = Path("ffprobe"),
) -> dict[str, dict[str, np.ndarray]]:
    episode_data = {
        OBS_STATE: state,
        ACTION: action,
        "subtask_index": subtask_index,
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_index,
        "index": index,
        "task_index": task_index,
    }
    numeric_features = {key: features[key] for key in episode_data}
    episode_stats = compute_episode_stats(episode_data, numeric_features)

    for video_key, video_path in video_paths.items():
        video_info = ffprobe_video(video_path, ffprobe_path=ffprobe_path)
        episode_stats[video_key] = compute_video_stats_from_path(video_path, video_info.frames)

    return episode_stats


def _serialize_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[Any]]]:
    serialized: dict[str, dict[str, list[Any]]] = {}
    for feature_key, feature_stats in stats.items():
        serialized[feature_key] = {
            stat_key: np.asarray(value).tolist() for stat_key, value in feature_stats.items()
        }
    return serialized


def finalize_lerobot_stats(
    episode_stats: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, list[Any]]]:
    if not episode_stats:
        return {}
    return _serialize_stats(aggregate_stats(episode_stats))


def dataframe_from_index_mapping(mapping: dict[str, int], index_name: str, column_name: str) -> pd.DataFrame:
    rows = sorted(mapping.items(), key=lambda item: item[1])
    df = pd.DataFrame({column_name: [idx for _, idx in rows]}, index=[text for text, _ in rows])
    df.index.name = index_name
    return df


def make_info_payload(
    *,
    repo_id: str,
    robot_type: str,
    fps: int,
    features: dict[str, dict[str, Any]],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_subtasks: int,
) -> dict[str, Any]:
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_subtasks": total_subtasks,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "repo_id": repo_id,
    }


# ---------------------------------------------------------------------------
# Task name mapping from YAML
# ---------------------------------------------------------------------------
def load_task_mapping(yaml_path: Path) -> dict[int, str]:
    """Load job_name.yaml, returning {task_num: task_name}.

    The YAML file has lines like:
        左侧前车门上下铰链喷涂: 51398
        右侧后车门上铰链喷涂操作: 50003
    """
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    mapping: dict[int, str] = {}
    for task_name, task_num in raw.items():
        mapping[int(task_num)] = str(task_name)
    logging.info("Loaded %d task mappings from %s", len(mapping), yaml_path)
    return mapping


def make_safe_dirname(task_name: str) -> str:
    """Convert a task name to a filesystem-safe directory name."""
    safe = re.sub(r'[^\w\u4e00-\u9fff]+', '_', task_name).strip('_')
    return safe if safe else f"task_{hash(task_name) & 0xFFFF:04x}"


# ===================================================================
# Source discovery: group folders by task prefix
# ===================================================================

def discover_task_episodes(
    source_root: Path,
    task_mapping: dict[int, str],
) -> dict[str, list[tuple[Path, str]]]:
    """Scan source_root and group mcap episodes by task name.

    Each folder is named ``{task_num}_{recording_num}`` (e.g. ``51398_76464``).
    The first number before ``_`` is matched against ``task_mapping`` to get
    the human-readable task name. All folders with the same task_num are merged
    into one task group.

    Returns:
        {task_name: [(mcap_path, source_folder_name), ...]}
        Sorted deterministically (by folder name, then by mcap stem).
    """
    # Group folders by task_num prefix
    task_folders: dict[int, list[Path]] = defaultdict(list)
    unknown_prefixes: set[str] = set()

    for folder in sorted(source_root.iterdir()):
        if not folder.is_dir():
            continue
        # Extract task_num: first sequence of digits before underscore
        match = re.match(r'^(\d+)_', folder.name)
        if not match:
            logging.warning("Skipping folder with unrecognized name format: %s", folder.name)
            continue
        task_num = int(match.group(1))
        if task_num not in task_mapping:
            unknown_prefixes.add(str(task_num))
            continue
        task_folders[task_num].append(folder)

    if unknown_prefixes:
        logging.warning(
            "Folders with unknown task prefixes (not in job_name.yaml): %s",
            ", ".join(sorted(unknown_prefixes)),
        )

    # Collect mcap episodes per task_name, preserving sorted order
    result: dict[str, list[tuple[Path, str]]] = {}

    for task_num in sorted(task_folders):
        task_name = task_mapping[task_num]
        episodes: list[tuple[Path, str]] = []
        for folder in task_folders[task_num]:
            # Collect .FIN.mcap; fall back to .SYNC.mcap if no .FIN
            fins = sorted(folder.glob("*.FIN.mcap"))
            fin_stems = {p.name[: -len(".FIN.mcap")] for p in fins}
            for mcap_path in fins:
                episodes.append((mcap_path, folder.name))
            for sync_path in sorted(folder.glob("*.SYNC.mcap")):
                stem = sync_path.name[: -len(".SYNC.mcap")]
                if stem not in fin_stems:
                    episodes.append((sync_path, folder.name))

        if episodes:
            result[task_name] = episodes
            logging.info(
                "Task [%s] (num=%d): %d episodes from %d folders",
                task_name, task_num, len(episodes), len(task_folders[task_num]),
            )
        else:
            logging.warning("Task [%s] (num=%d): no mcap files found", task_name, task_num)

    return result


# ===================================================================
# Progress / resume
# ===================================================================

def load_progress(output_dir: Path) -> dict[str, Any]:
    """Load progress.json from an output directory, or return empty state."""
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
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
    }


def save_progress(output_dir: Path, progress: dict[str, Any]) -> None:
    """Save progress state to progress.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "progress.json").open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def mcap_identifier(folder_name: str, mcap_path: Path) -> str:
    """Unique identifier for an mcap episode: 'folder_name/mcap_stem'."""
    return f"{folder_name}/{mcap_path.stem}"


# ===================================================================
# Convert a single task
# ===================================================================

def convert_task(
    task_name: str,
    episodes: list[tuple[Path, str]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Convert all episodes for one task into a LeRobot v3 dataset.

    Supports resume: if output_dir already exists and has a progress.json,
    already-processed episodes are skipped.
    """
    progress = load_progress(output_dir)
    processed_ids: set[str] = set(progress["processed_identifiers"])
    failed_ids: set[str] = set(progress.get("failed_identifiers", []))

    # Determine which episodes still need processing
    pending: list[tuple[int, Path, str]] = []  # (sorted_index, mcap_path, folder_name)
    for idx, (mcap_path, folder_name) in enumerate(episodes):
        ident = mcap_identifier(folder_name, mcap_path)
        if ident not in processed_ids and ident not in failed_ids:
            pending.append((idx, mcap_path, folder_name))

    if not pending:
        logging.info("Task [%s]: all %d episodes already processed, skipping.", task_name, len(episodes))
        _finalize_task_output(output_dir, task_name, progress, args)
        return

    # Apply --max-episodes-per-task to pending only
    if args.max_episodes_per_task is not None:
        already_done = len(processed_ids)
        remaining = args.max_episodes_per_task - already_done
        if remaining <= 0:
            logging.info(
                "Task [%s]: already have %d episodes (max=%d), skipping.",
                task_name, already_done, args.max_episodes_per_task,
            )
            _finalize_task_output(output_dir, task_name, progress, args)
            return
        pending = pending[:remaining]

    # Restore state from progress
    episode_rows: list[dict[str, Any]] = list(progress["episode_rows"])
    episode_stats_list: list[dict[str, dict[str, np.ndarray]]] = []
    # Deserialize saved stats back to numpy
    for serialized in progress["episode_stats"]:
        deserialized: dict[str, dict[str, np.ndarray]] = {}
        for feature_key, feature_stats in serialized.items():
            deserialized[feature_key] = {
                stat_key: np.array(value, dtype=np.float64)
                for stat_key, value in feature_stats.items()
            }
        episode_stats_list.append(deserialized)

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

    tasks: dict[str, int] = {task_name: 0}
    subtasks: dict[str, int] = {"complete task": 0}

    skipped_count = len(processed_ids)
    to_process = len(pending)
    logging.info(
        "Task [%s]: %d total, %d already done, %d to process.",
        task_name, len(episodes), skipped_count, to_process,
    )

    iterator = tqdm(pending, desc=f"Task: {task_name}") if tqdm else pending

    for _, mcap_path, folder_name in iterator:
        ep_index = total_episodes
        ident = mcap_identifier(folder_name, mcap_path)

        logging.info(
            "Processing episode %d: %s (folder=%s)",
            ep_index, mcap_path.name, folder_name,
        )

        try:
            state, action, timestamp, frame_index, video_paths, video_infos, num_frames = \
                align_and_write_episode(
                    mcap_path, output_dir, ep_index, JOINT_GROUPS,
                    args.fps, skip_videos=args.skip_videos,
                    ffprobe_path=args.ffprobe_path,
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
                first_height = next(iter(video_infos.values())).height
                first_width = next(iter(video_infos.values())).width

            subtask_text = "complete task"
            subtasks.setdefault(subtask_text, len(subtasks))
            subtask_index = np.full(num_frames, subtasks[subtask_text], dtype=np.int64)

            episode_index_array = np.full(num_frames, ep_index, dtype=np.int64)
            global_index = np.arange(total_frames, total_frames + num_frames, dtype=np.int64)
            task_index_arr = np.full(num_frames, 0, dtype=np.int64)

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
        }
        save_progress(output_dir, progress)

    # Finalize output
    _finalize_task_output(output_dir, task_name, progress, args)


def _finalize_task_output(
    output_dir: Path,
    task_name: str,
    progress: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write meta/info.json, meta/stats.json, meta/episodes, etc."""
    episode_rows = progress["episode_rows"]
    episode_stats_list = progress["episode_stats"]
    total_episodes = progress["total_episodes"]
    total_frames = progress["total_frames"]

    if total_episodes == 0:
        logging.info("Task [%s]: no episodes, skipping finalization.", task_name)
        return

    # Deserialize episode stats
    episode_stats_np: list[dict[str, dict[str, np.ndarray]]] = []
    for serialized in episode_stats_list:
        deserialized: dict[str, dict[str, np.ndarray]] = {}
        for feature_key, feature_stats in serialized.items():
            deserialized[feature_key] = {
                stat_key: np.array(value, dtype=np.float64)
                for stat_key, value in feature_stats.items()
            }
        episode_stats_np.append(deserialized)

    # Build VideoInfo from saved dict
    first_video_infos: dict[str, VideoInfo] | None = None
    if progress["first_video_infos"] is not None:
        first_video_infos = {
            k: VideoInfo(**v) for k, v in progress["first_video_infos"].items()
        }
    first_height = progress["first_height"]
    first_width = progress["first_width"]
    action_dim = progress["action_dim"]
    state_dim = progress["state_dim"]

    if first_video_infos is None:
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

    tasks: dict[str, int] = {task_name: 0}
    subtasks: dict[str, int] = {"complete task": 0}

    repo_id = args.repo_id_prefix + "/" + make_safe_dirname(task_name)

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
    dataframe_from_index_mapping(tasks, "task", "task_index").to_parquet(meta_dir / "tasks.parquet")
    dataframe_from_index_mapping(subtasks, "subtask", "subtask_index").to_parquet(
        meta_dir / "subtasks.parquet"
    )

    episodes_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

    # Write conversion report
    conversion_entries = progress.get("conversion_entries", [])
    write_json(
        output_dir / "conversion_report.json",
        {
            "source_root": str(args.source_root),
            "output_dir": str(output_dir),
            "repo_id": repo_id,
            "fps": args.fps,
            "task_name": task_name,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "videos": "extracted from mcap",
            "episodes": conversion_entries,
        },
    )

    logging.info(
        "Task [%s]: finalized %d episodes, %d frames → %s",
        task_name, total_episodes, total_frames, output_dir,
    )


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=False,default="/mnt/data2/dataset/data_jizheng/galbot_spray/spray",
        help="Root directory containing {task_num}_{recording_num}/ subdirectories with .mcap files.",
    )
    parser.add_argument(
        "--output-base", type=Path, required=False,default="/mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3_all",
        help="Base output directory. Each task gets its own subdirectory here.",
    )
    parser.add_argument(
        "--job-name-yaml", type=Path, required=False,default="/mnt/data1/jizheng/vla_project/galbot-g1/tools/job_name.yaml",
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

    output_base = args.output_base
    output_base.mkdir(parents=True, exist_ok=True)

    for task_name, episodes in task_episodes.items():
        safe_name = make_safe_dirname(task_name)
        task_output_dir = output_base / safe_name
        logging.info("=" * 60)
        logging.info("Processing task: %s → %s", task_name, task_output_dir)
        logging.info("=" * 60)

        try:
            convert_task(task_name, episodes, task_output_dir, args)
        except Exception:
            logging.exception("Failed to convert task [%s]", task_name)
            continue

    logging.info("All tasks processed. Output base: %s", output_base)


if __name__ == "__main__":
    main()
