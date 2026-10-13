#!/usr/bin/env python3
"""Convert Galbot mcap episodes into LeRobot v3 format.

Directory layout expected:
    root_dir/
        Pick CocaCola from the fourth shelf/
            *.mcap          (both .FIN.mcap and .SYNC.mcap; .FIN preferred)

The converter reads robot state from `singorix/wbcs/sensor` and actions from
`singorix/wbcs/target`, aligns them on the sensor timestamp axis, extracts
compressed camera images from the mcap and encodes them into mp4 videos, then
writes the LeRobot v3 directory layout directly.

Example source root::

    /media/ljw/Extreme SSD/vla_data/银河数据/rosbagdata/data_original/Pick CocaCola from the fourth shelf

Usage::

    # Convert all episodes under a task directory (writes videos and parquet)
    python tools/dataset/convert_galbot_mcap_to_lerobot.py \
        --source-root "/media/ljw/Extreme SSD/vla_data/银河数据/rosbagdata/data_original" \
        --output-root /home/ljw/projects/lerobotpiper/asset/data/galbot_mcap_lerobot \
        --fps 30 \
        --force

    # Convert only a few episodes for a quick test
    python tools/dataset/convert_galbot_mcap_to_lerobot.py \
        --source-root "/media/ljw/Extreme SSD/vla_data/银河数据/a" \
        --output-root /home/ljw/projects/lerobotpiper/asset/data/galbot_mcap_test \
        --max-episodes 2 \
        --force

    # Data/parquet only -- skip video encoding for faster debugging
    python tools/dataset/convert_galbot_mcap_to_lerobot.py \
        --source-root "/media/ljw/Extreme SSD/vla_data/银河数据/a" \
        --output-root /home/ljw/projects/lerobotpiper/asset/data/galbot_mcap_test \
        --max-episodes 2 \
        --skip-videos \
        --force
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
# from mcap.reader import make_reader
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
except ImportError:  # pragma: no cover - cosmetic fallback
    tqdm = None


# SOURCE_ROOT = Path("./origin_mcap")
# SOURCE_ROOT = Path("D:\\work\\robot\\galbot\\全身数据数采\\80702d84d7914b65bf17a2b2f32b67e3\\80702d84d7914b65bf17a2b2f32b67e3")
SOURCE_ROOT = Path("/mnt/data2/dataset/data_jizheng/galbot_spray/spray_0708/50003_71684")
# DEFAULT_OUTPUT_ROOT = Path("./galbot_mcap_lerobot/data")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data2/dataset/data_jizheng/galbot_spray/spray_0708/50003_71684_lerobotv3")
DEFAULT_REPO_ID = "local/galbot_mcap"
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

# Body-part groups that make up state / action vectors.
# Order must be consistent across episodes.
JOINT_GROUPS = [ "right_arm", "right_gripper","left_arm", "left_gripper", "leg", "head"]#
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
# ODOM_TOPICS = ["/odom/base_link"]

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
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
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
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
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
    """Build a protobuf descriptor pool from the mcap's embedded schemas."""
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
    """Return timestamp in nanoseconds from a galbot Header message."""
    ts = msg.header.timestamp
    return int(ts.sec) * 1_000_000_000 + int(ts.nanosec)


def _decode_image_bgr_from_message(msg) -> np.ndarray | None:
    """Decode a galbot CompressedImage protobuf to BGR numpy array."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


# ---------------------------------------------------------------------------
# Sensor / action extraction
# ---------------------------------------------------------------------------

def _extract_sensor_vector(
    msg,
    groups: list[str],
) -> tuple[int, np.ndarray]:
    """Extract a flat state vector from a *sensor* message.

    Returns (ts_ns, positions ndarray[float32]).
    """
    ts = _msg_ts_ns(msg)
    source_map = msg.joint_sensor_map

    parts: list[list[float]] = []
    for group_name in groups:
        if group_name not in source_map:
            # dim_map = {"left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1}
            dim_map = DIM_MAPS
            parts.append([0.0] * dim_map.get(group_name, 0))
            continue
        group = source_map[group_name]
        parts.append(list(group.position))

    flat = [v for sub in parts for v in sub]
    return ts, np.asarray(flat, dtype=np.float32)


def _extract_target_group_vector(msg) -> list[tuple[int, str, np.ndarray]]:
    """Extract every group's target from a *target* message.

    A single target message can contain trajectories for multiple joint groups.
    Returns a list of (ts_ns, group_name, positions ndarray[float32]) so that
    every group is emitted deterministically, regardless of protobuf map
    iteration order.
    """
    ts = _msg_ts_ns(msg)
    source_map = msg.target_group_trajectory_map
    if not source_map:
        if SKIP_EMPTY_ACTIONS:
            return []
        raise ValueError("Target message has empty target_group_trajectory_map")

    results: list[tuple[int, str, np.ndarray]] = []
    # Iterate source_map directly; group lookup by name is deterministic.
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
    """Read an mcap and return raw (sensor_ts, sensor_arr, target_group_data, camera_data, camera_shapes).

    Camera frames are decoded to BGR and stored with their timestamps (nanoseconds).
    """
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
                # Prefer protobuf header timestamp; fallback to mcap log_time
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
# Video extraction from mcap
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

    Mirrors LeRobot's ``encode_video_frames`` defaults (h264, yuv420p, g=2,
    crf=30) so the output matches datasets produced by the native writer.

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
    """Extract, align to master camera, write videos, and return aligned data.

    Returns (state, action, timestamp_sec, frame_index, video_paths, video_infos, num_frames).
    """
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
        mcap_path.name,
        master_topic,
        num_frames,
        len(sensor_ts),
        {k: len(v) for k, v in target_group_data.items()},
    )

    # Align sensor to master camera timestamps
    sensor_nearest = nearest_indices_by_timestamp(master_ts, sensor_ts)
    state = sensor_arr[sensor_nearest]

    # Build action by aligning each target group independently
    dim_map = DIM_MAPS
    group_slices: dict[str, slice] = {}
    idx = 0
    for group_name in joint_groups:
        d = dim_map[group_name]
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

    # Timestamps must lie on the constant-fps grid of the encoded video, not the
    # raw (irregular) camera timestamps. Otherwise the LeRobot loader's nearest
    # video frame drifts past its decode tolerance. This matches the native
    # writer's convention: timestamp = frame_index / fps.
    frame_index = np.arange(num_frames, dtype=np.int64)
    timestamp_sec = (frame_index / float(fps)).astype(np.float32)

    return state, action, timestamp_sec, frame_index, video_paths, video_infos, num_frames


# ---------------------------------------------------------------------------
# LeRobot helpers (re-used from convert_galbot_to_lerobot.py)
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
    """Compute LeRobot-compatible per-channel stats from sampled video frames."""
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


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def mcap_episodes(source_root: Path) -> list[tuple[Path, str]]:
    """Return one (mcap_path, task_name) per episode found under source_root.

    Each ``.FIN.mcap`` is a distinct episode. Two layouts are supported:
        - source_root / TaskName / *.FIN.mcap            (one episode per mcap)
        - source_root / TaskName / episode_dir / *.FIN.mcap

    A ``.SYNC.mcap`` is only used as a fallback when no ``.FIN.mcap`` exists for
    the same recording. The recording stem (filename without the .FIN/.SYNC
    suffix) is used to deduplicate, so an episode is never counted twice.
    """
    results: list[tuple[Path, str]] = []

    def _collect(directory: Path, task_name: str) -> None:
        fins = sorted(directory.glob("*.FIN.mcap"))
        for mcap_path in fins:
            results.append((mcap_path, task_name))
        # Recording stems already covered by a .FIN file.
        fin_stems = {p.name[: -len(".FIN.mcap")] for p in fins}
        for sync_path in sorted(directory.glob("*.SYNC.mcap")):
            stem = sync_path.name[: -len(".SYNC.mcap")]
            if stem in fin_stems:
                continue
            results.append((sync_path, task_name))

    if list(source_root.glob("*.mcap")):
        _collect(source_root, source_root.name)
    for task_dir in sorted(source_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if list(task_dir.glob("*.mcap")):
            _collect(task_dir, task_name)
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            if list(ep_dir.glob("*.mcap")):
                _collect(ep_dir, task_name)
    return results


def missing_required_video_reason(mcap_path: Path) -> str | None:
    # Video topics are inside the mcap; we just need to verify the mcap exists.
    if not mcap_path.exists():
        return f"Missing mcap file: {mcap_path}"
    return None


def filter_episodes_with_required_videos(episodes: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    filtered = []
    for mcap_path, task_name in episodes:
        reason = missing_required_video_reason(mcap_path)
        if reason is not None:
            logging.warning("Skipping episode %s: %s", mcap_path.name, reason)
            continue
        filtered.append((mcap_path, task_name))
    return filtered


def copy_episode_videos(ep_dir: Path, output_root: Path, episode_index: int) -> None:
    """Videos are already extracted into output_root/videos/... by extract_video_from_mcap."""
    pass


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
# Main conversion
# ---------------------------------------------------------------------------

def _empty_progress() -> dict[str, Any]:
    return {
        "processed_identifiers": [],
        "failed_identifiers": [],
        "episode_rows": [],
        "episode_stats": [],
        "conversion_entries": [],
        "total_frames": 0,
        "total_episodes": 0,
        "first_video_infos": None,
        "first_height": 0,
        "first_width": 0,
        "action_dim": 0,
        "state_dim": 0,
    }


def load_progress(output_root: Path) -> dict[str, Any]:
    """Load progress.json from the output dir, or return empty state."""
    path = output_root / "progress.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return _empty_progress()


def save_progress(output_root: Path, progress: dict[str, Any]) -> None:
    """Persist progress state so a re-run can resume from where it stopped."""
    with (output_root / "progress.json").open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def mcap_identifier(mcap_path: Path, source_root: Path) -> str:
    """Unique identifier for an mcap episode (stable across resumes)."""
    try:
        return str(mcap_path.relative_to(source_root))
    except ValueError:
        return mcap_path.stem


def _deserialize_episode_stats(
    serialized_list: list,
) -> list[dict[str, dict[str, np.ndarray]]]:
    """Deserialize saved per-episode stats back to numpy arrays."""
    result: list[dict[str, dict[str, np.ndarray]]] = []
    for serialized in serialized_list:
        episode: dict[str, dict[str, np.ndarray]] = {}
        for feature_key, feature_stats in serialized.items():
            episode[feature_key] = {
                stat_key: np.array(value, dtype=np.float64)
                for stat_key, value in feature_stats.items()
            }
        result.append(episode)
    return result


def convert_dataset(args: argparse.Namespace) -> Path:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    # Resume support: reuse progress.json if present; --force wipes and restarts.
    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        elif (output_root / "progress.json").exists():
            logging.info("Resuming from existing progress in %s", output_root)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                f"Use --force to overwrite and restart."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    episodes = mcap_episodes(source_root)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    episodes = filter_episodes_with_required_videos(episodes)
    if not episodes:
        raise FileNotFoundError(f"No episodes with mcap files found under {source_root}")

    # Restore progress state (empty on first run)
    progress = load_progress(output_root)
    processed_ids: set[str] = set(progress["processed_identifiers"])
    failed_ids: set[str] = set(progress.get("failed_identifiers", []))

    episode_rows: list[dict[str, Any]] = list(progress["episode_rows"])
    episode_stats_list: list[dict[str, dict[str, np.ndarray]]] = _deserialize_episode_stats(
        progress["episode_stats"]
    )
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

    # Only process episodes not yet done and not previously failed
    pending = [
        (mcap_path, task_name)
        for mcap_path, task_name in episodes
        if mcap_identifier(mcap_path, source_root) not in processed_ids
        and mcap_identifier(mcap_path, source_root) not in failed_ids
    ]
    logging.info(
        "Total=%d, processed=%d, failed=%d, to process=%d",
        len(episodes), len(processed_ids), len(failed_ids), len(pending),
    )

    # Rebuild task/subtask registries from already-processed rows
    tasks: dict[str, int] = {}
    subtasks: dict[str, int] = {}
    for row in episode_rows:
        for t in row["tasks"]:
            tasks.setdefault(t, len(tasks))
    subtasks.setdefault("complete task", 0)

    iterator = tqdm(pending, desc="Converting episodes") if tqdm else pending

    for mcap_path, task_name in iterator:
        if args.task_name:
            task_name = args.task_name
        ident = mcap_identifier(mcap_path, source_root)
        ep_index = total_episodes

        tasks.setdefault(task_name, len(tasks))
        task_idx = tasks[task_name]

        logging.info("Processing episode %s (task=%s)", mcap_path.name, task_name)

        try:
            state, action, timestamp, frame_index, video_paths, video_infos, num_frames = align_and_write_episode(
                mcap_path, output_root, ep_index, JOINT_GROUPS, args.fps, skip_videos=args.skip_videos, ffprobe_path=args.ffprobe_path,
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
                first_height = next(iter(video_infos.values())).height
                first_width = next(iter(video_infos.values())).width

            # Subtask: single subtask per episode for now
            subtask_text = "complete task"
            subtasks.setdefault(subtask_text, len(subtasks))
            subtask_index = np.full(num_frames, subtasks[subtask_text], dtype=np.int64)

            episode_index_array = np.full(num_frames, ep_index, dtype=np.int64)
            global_index = np.arange(total_frames, total_frames + num_frames, dtype=np.int64)
            task_index = np.full(num_frames, task_idx, dtype=np.int64)

            data_path = output_root / "data" / "chunk-000" / f"file-{ep_index:03d}.parquet"
            write_episode_data(
                data_path,
                state=state,
                action=action,
                subtask_index=subtask_index,
                timestamp=timestamp,
                frame_index=frame_index,
                episode_index=episode_index_array,
                index=global_index,
                task_index=task_index,
            )

            if first_video_infos is None:
                # Videos were skipped; use conservative defaults for feature metadata.
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
            episode_stats_list.append(
                compute_lerobot_episode_stats(
                    features=episode_features,
                    state=state,
                    action=action,
                    subtask_index=subtask_index,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    episode_index=episode_index_array,
                    index=global_index,
                    task_index=task_index,
                    video_paths=video_paths,
                    num_frames=num_frames,
                    ffprobe_path=args.ffprobe_path,
                )
            )

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

            conversion_entries.append(
                {
                    "episode_index": ep_index,
                    "source_episode_id": mcap_path.stem,
                    "frames": num_frames,
                    "task": task_name,
                    "subtasks": sorted(set(subtask_index.tolist())),
                }
            )
            total_frames += num_frames
            total_episodes += 1
            processed_ids.add(ident)
        except Exception:
            logging.exception("Skipping broken episode %s", mcap_path.name)
            failed_ids.add(ident)

        # Persist progress after every episode (success or failure)
        progress = {
            "processed_identifiers": sorted(processed_ids),
            "failed_identifiers": sorted(failed_ids),
            "episode_rows": episode_rows,
            "episode_stats": [_serialize_stats(s) for s in episode_stats_list],
            "conversion_entries": conversion_entries,
            "total_frames": total_frames,
            "total_episodes": total_episodes,
            "first_video_infos": (
                {
                    k: {
                        "width": v.width, "height": v.height, "fps": v.fps,
                        "frames": v.frames, "duration": v.duration,
                        "codec": v.codec, "pix_fmt": v.pix_fmt, "has_audio": v.has_audio,
                    }
                    for k, v in first_video_infos.items()
                }
                if first_video_infos else None
            ),
            "first_height": first_height,
            "first_width": first_width,
            "action_dim": action_dim,
            "state_dim": state_dim,
        }
        save_progress(output_root, progress)

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

    meta_dir = output_root / "meta"
    write_json(
        meta_dir / "info.json",
        make_info_payload(
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            fps=args.fps,
            features=features,
            total_episodes=len(episode_rows),
            total_frames=total_frames,
            total_tasks=len(tasks),
            total_subtasks=len(subtasks),
        ),
    )
    write_json(meta_dir / "stats.json", finalize_lerobot_stats(episode_stats_list))
    dataframe_from_index_mapping(tasks, "task", "task_index").to_parquet(meta_dir / "tasks.parquet")
    dataframe_from_index_mapping(subtasks, "subtask", "subtask_index").to_parquet(
        meta_dir / "subtasks.parquet"
    )

    episodes_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

    write_json(
        output_root / "conversion_report.json",
        {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "repo_id": args.repo_id,
            "fps": args.fps,
            "total_episodes": len(episode_rows),
            "total_frames": total_frames,
            "total_failed": len(failed_ids),
            "videos": "extracted from mcap",
            "episodes": conversion_entries,
        },
    )
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="galbot_one_golf")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Write metadata/data only; useful for tests.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--ffprobe-path", type=Path, default=Path("ffprobe"), help="Path to ffprobe executable")
    parser.add_argument("--task-name", default=None, help="覆盖所有 episode 的 task name（不指定则用目录名）")
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
    args = parser.parse_args()
    args.target_size = None if args.no_resize else parse_target_resolution(args.target_resolution)
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s:%(message)s")
    output_root = convert_dataset(args)
    logging.info("Wrote LeRobot dataset to %s", output_root)


if __name__ == "__main__":
    main()
