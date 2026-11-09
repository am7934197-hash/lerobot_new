# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import hashlib
import math
import os
import pickle  # nosec
import threading
import time
import uuid
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import LatencyTracker, RTCConfig, RTCProcessor
from lerobot.policies.rtc.configuration_rtc import validate_trained_rtc_horizon
from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.lerobot_types import PolicyAction
from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


def _resolve_local_text_tokenizer(tokenizer_name: str | None) -> str | None:
    """Resolve a tokenizer without requiring Hub access on an inference host."""
    override = os.environ.get("LEROBOT_TEXT_TOKENIZER_PATH")
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_dir():
            raise FileNotFoundError(
                f"LEROBOT_TEXT_TOKENIZER_PATH is not a directory: {override_path}"
            )
        return str(override_path)

    if not tokenizer_name:
        return None
    tokenizer_path = Path(tokenizer_name).expanduser()
    if tokenizer_path.is_dir():
        return str(tokenizer_path)
    if "/" not in tokenizer_name:
        return tokenizer_name

    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        )
    ).expanduser()
    snapshots_dir = cache_root / f"models--{tokenizer_name.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.is_dir():
        return tokenizer_name

    candidates = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: (path.name != "local", path.name),
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate)
    return tokenizer_name


def _conditioned_rtc_delay_frames(
    predicted_delay_frames: int,
    safety_margin_frames: int,
    training_max_delay: int,
    prefix_frames_used: int,
) -> int:
    """Return a conservative trained-RTC delay within checkpoint/prefix capacity."""
    return min(
        predicted_delay_frames + safety_margin_frames,
        training_max_delay,
        prefix_frames_used,
    )


def _rtc_delay_rejection_reason(
    *,
    trained_with_prefix: bool,
    actual_delay_frames: int,
    conditioned_delay_frames: int,
) -> str | None:
    if trained_with_prefix and actual_delay_frames > conditioned_delay_frames:
        return "actual_delay_exceeded_conditioning"
    return None


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.server_instance_id = uuid.uuid4().hex
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.rename_map: dict[str, str] = {}
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.language_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self.rtc_config = config.rtc
        self.rtc_latency_tracker = LatencyTracker()
        self._relative_step: RelativeActionsProcessorStep | None = None
        self._normalizer_step: NormalizerProcessorStep | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    @property
    def transport_image_features(self):
        """Return model shapes keyed by the camera names present on the wire."""
        result = {}
        for raw_key in self.lerobot_features or {}:
            if not raw_key.startswith("observation.images."):
                continue
            policy_key = self.rename_map.get(raw_key, raw_key)
            if policy_key not in self.policy_image_features:
                raise KeyError(
                    f"Camera mapping {raw_key!r} -> {policy_key!r} is not present in checkpoint"
                )
            result[raw_key] = self.policy_image_features[policy_key]
        return result

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        self.rtc_latency_tracker.reset()

    @staticmethod
    def _trace_id(context, fallback: str = "?") -> str:
        metadata = dict(context.invocation_metadata() or ())
        return str(metadata.get("x-lerobot-trace-id", fallback))

    def _configure_rtc(self, raw_config: dict[str, Any] | None) -> None:
        values = asdict(self.config.rtc)
        if raw_config:
            for key in (
                "enabled",
                "mode",
                "execution_horizon",
                "delay_safety_margin_frames",
                "max_guidance_weight",
                "prefix_attention_schedule",
                "debug",
                "debug_maxlen",
            ):
                if key in raw_config:
                    values[key] = raw_config[key]
        schedule = values["prefix_attention_schedule"]
        if isinstance(schedule, str):
            values["prefix_attention_schedule"] = RTCAttentionSchedule(schedule.upper())
        self.rtc_config = RTCConfig(**values)
        self.rtc_latency_tracker.reset()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        context.set_trailing_metadata(
            (
                ("x-lerobot-protocol-version", str(self.config.protocol_version)),
                ("x-lerobot-server-instance-id", self.server_instance_id),
                ("x-lerobot-profile", self.config.expected_policy_type),
                (
                    "x-lerobot-checkpoint-fingerprint",
                    self.config.expected_checkpoint_fingerprint,
                ),
            )
        )
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )
        if int(getattr(policy_specs, "protocol_version", 0)) != self.config.protocol_version:
            raise ValueError("Client protocol_version does not match the server")
        if self.config.expected_policy_type and (
            policy_specs.policy_type != self.config.expected_policy_type
            or getattr(policy_specs, "profile", "") != self.config.expected_policy_type
        ):
            raise ValueError("Server is pinned to a different policy Profile")
        if self.config.expected_checkpoint_path and Path(
            policy_specs.pretrained_name_or_path
        ).resolve() != Path(self.config.expected_checkpoint_path).resolve():
            raise ValueError("Server is pinned to a different checkpoint path")
        config_path = Path(policy_specs.pretrained_name_or_path) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Checkpoint config does not exist: {config_path}")
        checkpoint_fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
        requested_fingerprint = str(getattr(policy_specs, "checkpoint_fingerprint", ""))
        if requested_fingerprint and checkpoint_fingerprint != requested_fingerprint:
            raise ValueError("Checkpoint fingerprint does not match the client request")
        if (
            self.config.expected_checkpoint_fingerprint
            and checkpoint_fingerprint != self.config.expected_checkpoint_fingerprint
        ):
            raise ValueError("Checkpoint fingerprint does not match the pinned server config")

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.rename_map = dict(policy_specs.rename_map)
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self._configure_rtc(getattr(policy_specs, "rtc_config", None))

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        # max-autotune torch.compile 在推理时容易 CUDA illegal memory access。
        # 默认关掉；需要编译时 export LEROBOT_POLICY_COMPILE=1
        pretrained_path = policy_specs.pretrained_name_or_path
        load_kwargs: dict[str, Any] = {}
        if os.environ.get("LEROBOT_POLICY_COMPILE", "0") != "1":
            policy_cfg = PreTrainedConfig.from_pretrained(
                pretrained_path,
                fallback_policy_type=self.policy_type,
            )
            if getattr(policy_cfg, "compile_model", False):
                policy_cfg.compile_model = False
                self.logger.info(
                    "Disabled torch.compile for inference "
                    "(checkpoint had compile_model=true). Set LEROBOT_POLICY_COMPILE=1 to keep it."
                )
            load_kwargs["config"] = policy_cfg

        self.policy = policy_class.from_pretrained(pretrained_path, **load_kwargs)
        if self.policy_type in {"pi05", "fineart_vla"}:
            self.policy.config.rtc_config = self.rtc_config
            self.policy.init_rtc_processor()
            if self.rtc_config.enabled and not isinstance(self.policy.rtc_processor, RTCProcessor):
                raise RuntimeError("Pi0.5 RTCProcessor was not initialized")
            if self.rtc_config.enabled and self.rtc_config.mode == "trained":
                validate_trained_rtc_horizon(
                    self.rtc_config.execution_horizon,
                    self.policy.config.chunk_size,
                    self.policy.config.rtc_training_max_delay,
                )
            self.logger.info(
                "Pi0.5 RTC configured: enabled=%s mode=%s execution_horizon=%d "
                "max_guidance_weight=%.3f prefix_attention_schedule=%s processor=%s",
                self.rtc_config.enabled,
                self.rtc_config.mode,
                self.rtc_config.execution_horizon,
                self.rtc_config.max_guidance_weight,
                self.rtc_config.prefix_attention_schedule.value,
                type(self.policy.rtc_processor).__name__,
            )
        elif self.rtc_config.enabled:
            self.logger.warning(
                "RTC is enabled but policy type %s is not pi05; RTC input is ignored",
                self.policy_type,
            )
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        preprocessor_overrides = {
            "device_processor": device_override,
            "rename_observations_processor": {"rename_map": policy_specs.rename_map},
        }
        local_text_tokenizer = None
        if self.policy_type == "fineart_vla":
            local_text_tokenizer = _resolve_local_text_tokenizer(
                getattr(self.policy.config, "text_tokenizer_name", None)
            )
        if local_text_tokenizer is not None:
            preprocessor_overrides["action_tokenizer_processor"] = {
                "paligemma_tokenizer_name": local_text_tokenizer
            }
            if local_text_tokenizer != getattr(self.policy.config, "text_tokenizer_name", None):
                self.logger.info("Using local text tokenizer: %s", local_text_tokenizer)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": device_override},
        )
        if self.policy_type == "fineart_vla":
            self.language_preprocessor, _unused_postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=policy_specs.pretrained_name_or_path,
                preprocessor_overrides=preprocessor_overrides,
                postprocessor_overrides={"device_processor": device_override},
            )
        abs_steps = [
            s
            for s in self.postprocessor.steps
            if getattr(s, "_registry_name", None) == "absolute_actions_processor"
        ]
        if abs_steps and any(getattr(s, "relative_step", None) is None for s in abs_steps):
            raise RuntimeError(
                "Failed to pair AbsoluteActionsProcessorStep with RelativeActionsProcessorStep. "
                "Relative actions cannot be converted back to joint targets."
            )
        if abs_steps:
            self.logger.info("Relative/absolute action processors are paired")
        self._relative_step = next(
            (
                step
                for step in self.preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        self._normalizer_step = next(
            (step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")
        handler_start = time.perf_counter()
        trace_id = self._trace_id(context)
        self.logger.info(
            "服务端追踪：收到请求 trace_id=%s event=SERVER_REQUEST_RECEIVED perf_s=%.6f",
            trace_id,
            handler_start,
        )

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        if not isinstance(timed_observation, TimedObservation):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Expected TimedObservation")
        if int(getattr(timed_observation, "protocol_version", 0)) != self.config.protocol_version:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Protocol version mismatch")
        if getattr(timed_observation, "profile", "") != self.policy_type:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Profile mismatch")
        if getattr(timed_observation, "request_kind", "") not in {"action", "next_subtask"}:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unknown request_kind")
        deserialize_time = time.perf_counter() - start_deserialize
        decoded_at = time.perf_counter()
        trace_id = str(getattr(timed_observation, "trace_id", "") or trace_id)
        timed_observation.trace_id = trace_id
        timed_observation.server_request_start_perf = handler_start
        timed_observation.server_receive_ms = (decoded_at - handler_start) * 1000
        timed_observation.server_queue_enter_perf = decoded_at
        self.logger.info(
            "服务端追踪：观测数据解码完成 trace_id=%s event=SERVER_OBSERVATION_DECODED perf_s=%.6f "
            "server_receive_ms=%.3f",
            trace_id,
            decoded_at,
            timed_observation.server_receive_ms,
        )

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            if self.policy is None:
                raise RuntimeError(
                    "Policy is not loaded. Client must SendPolicyInstructions (setup_policy) before GetActions."
                )
            getactions_starts = time.perf_counter()
            request_trace_id = self._trace_id(context)
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            queue_done = time.perf_counter()
            trace_id = str(getattr(obs, "trace_id", "") or request_trace_id)
            queue_enter = float(getattr(obs, "server_queue_enter_perf", getactions_starts))
            server_queue_ms = max(0.0, (queue_done - queue_enter) * 1000)
            self.logger.info(
                "SERVER TRACE trace_id=%s event=SERVER_QUEUE_DONE perf_s=%.6f server_queue_ms=%.3f",
                trace_id,
                queue_done,
                server_queue_ms,
            )
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            if getattr(obs, "request_kind", "action") == "next_subtask":
                payload = self._predict_subtask_response(obs)
                context.set_trailing_metadata(
                    (("x-lerobot-trace-id", str(getattr(obs, "request_id", trace_id))),)
                )
                return services_pb2.Actions(data=pickle.dumps(payload))  # nosec

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            infer_at = time.perf_counter()
            self.logger.info(
                "SERVER TRACE trace_id=%s event=SERVER_INFER_START perf_s=%.6f",
                trace_id,
                infer_at,
            )
            action_chunk, stage_timing = self._predict_action_chunk(obs)
            action_chunk.update(self._response_context(obs, "action_chunk"))
            infer_done = time.perf_counter()
            inference_time = infer_done - infer_at
            self.logger.info(
                "服务端追踪：本次推理完成 trace_id=%s event=SERVER_INFER_END perf_s=%.6f "
                "本次模型推理耗时=%.3f毫秒 server_model_infer_ms=%.3f "
                "prefix_frames_used=%d inference_delay_frames=%d",
                trace_id,
                infer_done,
                stage_timing["server_model_infer_ms"],
                stage_timing["server_model_infer_ms"],
                action_chunk["prefix_frames_used"],
                action_chunk["inference_delay_frames"],
            )

            start_time = time.perf_counter()
            # Instrumented RTC clients receive the correlated response envelope.
            # Legacy LeRobot clients (no trace ID) keep the original list payload.
            wire_payload = action_chunk if action_chunk["trace_id"] else action_chunk["actions"]
            actions_bytes = pickle.dumps(wire_payload)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)
            response_at = time.perf_counter()
            metadata = (
                ("x-lerobot-trace-id", trace_id),
                ("x-lerobot-server-receive-ms", str(getattr(obs, "server_receive_ms", 0.0))),
                ("x-lerobot-server-queue-ms", str(server_queue_ms)),
                ("x-lerobot-server-preprocess-ms", str(stage_timing["server_preprocess_ms"])),
                ("x-lerobot-server-model-ms", str(stage_timing["server_model_infer_ms"])),
                ("x-lerobot-server-postprocess-ms", str(stage_timing["server_postprocess_ms"])),
                ("x-lerobot-server-serialize-ms", str(serialize_time * 1000)),
                ("x-lerobot-server-get-actions-ms", str((response_at - getactions_starts) * 1000)),
                ("x-lerobot-inference-delay-frames", str(action_chunk["inference_delay_frames"])),
                ("x-lerobot-prefix-frames-used", str(action_chunk["prefix_frames_used"])),
            )
            context.set_trailing_metadata(metadata)
            self.logger.info(
                "SERVER TRACE trace_id=%s event=SERVER_RESPONSE_SEND perf_s=%.6f "
                "server_receive_ms=%.3f server_queue_ms=%.3f server_preprocess_ms=%.3f "
                "server_model_infer_ms=%.3f server_postprocess_ms=%.3f server_serialize_ms=%.3f",
                trace_id,
                response_at,
                getattr(obs, "server_receive_ms", 0.0),
                server_queue_ms,
                stage_timing["server_preprocess_ms"],
                stage_timing["server_model_infer_ms"],
                stage_timing["server_postprocess_ms"],
                serialize_time * 1000,
            )

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(
        self,
        observation: dict[str, torch.Tensor],
        *,
        prev_chunk_left_over: torch.Tensor | None = None,
        inference_delay: int = 0,
    ) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        if (
            self.policy_type in {"pi05", "fineart_vla"}
            and self.rtc_config.enabled
            and prev_chunk_left_over is not None
        ):
            chunk = self.policy.predict_action_chunk(
                observation,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=self.rtc_config.execution_horizon,
            )
        else:
            chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    @staticmethod
    def _normalize_prefix_length(prefix: torch.Tensor, target_steps: int) -> torch.Tensor:
        if prefix.ndim != 2:
            raise ValueError(f"previous_actions must have shape [T, A], got {tuple(prefix.shape)}")
        if prefix.shape[0] >= target_steps:
            return prefix[:target_steps]
        if prefix.shape[0] == 0:
            raise ValueError("Cannot pad an empty RTC prefix")
        hold = prefix[-1:].expand(target_steps - prefix.shape[0], -1)
        return torch.cat([prefix, hold], dim=0)

    def _prepare_rtc_prefix(
        self,
        observation_t: TimedObservation,
    ) -> tuple[torch.Tensor | None, int]:
        """Convert the observation-aligned client action prefix to Pi0.5 model space."""
        previous_actions = getattr(observation_t, "previous_actions", None)
        if (
            previous_actions is None
            or self.policy_type not in {"pi05", "fineart_vla"}
            or not self.rtc_config.enabled
        ):
            return None, 0

        prefix_absolute = torch.as_tensor(previous_actions, dtype=torch.float32)
        if prefix_absolute.numel() == 0:
            return None, 0
        if prefix_absolute.ndim != 2:
            raise ValueError(
                f"previous_actions must have shape [T, A], got {tuple(prefix_absolute.shape)}"
            )

        expected_action_dim = self.policy.config.output_features["action"].shape[0]
        if prefix_absolute.shape[1] != expected_action_dim:
            raise ValueError(
                "previous_actions action dimension mismatch: "
                f"got {prefix_absolute.shape[1]}, expected {expected_action_dim}"
            )

        if self._relative_step is not None:
            current_state = self._relative_step.get_cached_state()
            if current_state is None:
                raise RuntimeError("RelativeActionsProcessorStep has no cached observation state")
            prefix_model = reanchor_relative_rtc_prefix(
                prev_actions_absolute=prefix_absolute,
                current_state=current_state,
                relative_step=self._relative_step,
                normalizer_step=self._normalizer_step,
                policy_device=self.device,
            )
        else:
            prefix_model = prefix_absolute
            if self._normalizer_step is not None:
                transition = self._normalizer_step(create_transition(action=prefix_model))
                prefix_model = transition[TransitionKey.ACTION]
            prefix_model = prefix_model.to(self.device)

        prefix_frames_used = min(len(prefix_model), self.rtc_config.execution_horizon)
        prefix_model = self._normalize_prefix_length(
            prefix_model, self.rtc_config.execution_horizon
        )
        return prefix_model, prefix_frames_used

    def _response_context(self, observation_t: TimedObservation, response_kind: str) -> dict[str, Any]:
        return {
            "protocol_version": self.config.protocol_version,
            "response_kind": response_kind,
            "session_id": str(getattr(observation_t, "session_id", "")),
            "request_id": str(getattr(observation_t, "request_id", "")),
            "query_id": str(getattr(observation_t, "query_id", "")),
            "subtask_epoch": int(getattr(observation_t, "subtask_epoch", 0)),
            "profile": str(getattr(observation_t, "profile", "")),
            "checkpoint_fingerprint": str(
                getattr(observation_t, "checkpoint_fingerprint", "")
            ),
            "server_instance_id": self.server_instance_id,
            "control_step": int(getattr(observation_t, "control_step", 0)),
            "trajectory_step": int(getattr(observation_t, "trajectory_step", 0)),
            "stage_step": int(getattr(observation_t, "stage_step", 0)),
            "observation_timestamp_ns": int(
                getattr(observation_t, "observation_timestamp_ns", 0)
            ),
        }

    def _predict_subtask_response(self, observation_t: TimedObservation) -> dict[str, Any]:
        if self.policy_type != "fineart_vla" or self.language_preprocessor is None:
            raise RuntimeError("Text queries require a loaded fineart_vla checkpoint")
        if not self.policy.supports_text_generation():
            raise RuntimeError("The loaded checkpoint does not support text generation")
        raw = observation_t.get_observation()
        observation = raw_observation_to_observation(
            raw,
            self.lerobot_features,
            self.transport_image_features,
        )
        observation[QUERY_KIND] = "next_subtask"
        observation[QUERY_TEXT] = str(
            getattr(observation_t, "query_text", "Describe the next subtask.")
        )
        allowed_subtasks = list(getattr(observation_t, "allowed_subtasks", []))
        if not allowed_subtasks or any(
            not isinstance(candidate, str) or not candidate.strip()
            for candidate in allowed_subtasks
        ):
            raise ValueError("next_subtask requires a non-empty allowed_subtasks list")
        processed = self.language_preprocessor(observation)
        with torch.inference_mode():
            generated_text = self.policy.generate_text(
                processed,
                allowed_subtasks=allowed_subtasks,
            )
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise ValueError("FineART text head returned an empty response")
        if len(generated_text) > 4096:
            raise ValueError("FineART text response exceeds 4096 characters")
        if generated_text not in allowed_subtasks:
            raise ValueError("FineART constrained text head returned a subtask outside the allowlist")
        response = self._response_context(observation_t, "subtask_query")
        response.update(
            {
                "generated_text": generated_text,
                "allowed_subtasks": allowed_subtasks,
            }
        )
        return response

    def _predict_action_chunk(self, observation_t: TimedObservation) -> tuple[dict[str, Any], dict[str, float]]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.transport_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        start_prefix_preprocess = time.perf_counter()
        rtc_prefix, prefix_frames_used = self._prepare_rtc_prefix(observation_t)
        preprocessing_time += time.perf_counter() - start_prefix_preprocess

        # Keep observation, prefix, generated chunk, and client-side trimming on
        # the same timeline.  The prefix therefore remains anchored at observation
        # capture instead of being shifted forward before denoising.
        request_start_perf = float(
            getattr(
                observation_t,
                "server_request_start_perf",
                getattr(observation_t, "server_queue_enter_perf", start_prepare),
            )
        )
        elapsed_before_inference = max(0.0, time.perf_counter() - request_start_perf)
        predicted_remaining_latency = self.rtc_latency_tracker.max() or 0.0
        request_fps = float(getattr(observation_t, "fps", 0.0) or self.config.fps)
        predicted_delay_frames = math.ceil(
            (elapsed_before_inference + predicted_remaining_latency) * request_fps
        )
        guidance_delay_frames = predicted_delay_frames
        conditioned_delay_frames = predicted_delay_frames
        if self.rtc_config.mode == "trained" and rtc_prefix is not None:
            training_max_delay = int(self.policy.config.rtc_training_max_delay)
            conditioned_delay_frames = _conditioned_rtc_delay_frames(
                predicted_delay_frames,
                int(self.rtc_config.delay_safety_margin_frames),
                training_max_delay,
                prefix_frames_used,
            )
            guidance_delay_frames = conditioned_delay_frames
            if conditioned_delay_frames < (
                predicted_delay_frames + int(self.rtc_config.delay_safety_margin_frames)
            ):
                self.logger.warning(
                    "Clamped trained RTC delay from %d to %d frames "
                    "(margin=%d, checkpoint max=%d, available=%d)",
                    predicted_delay_frames,
                    conditioned_delay_frames,
                    self.rtc_config.delay_safety_margin_frames,
                    training_max_delay,
                    prefix_frames_used,
                )
            self.logger.info(
                "Trained RTC delay prediction: predicted=%d margin=%d "
                "conditioned=%d prefix=%d checkpoint_max=%d",
                predicted_delay_frames,
                self.rtc_config.delay_safety_margin_frames,
                conditioned_delay_frames,
                prefix_frames_used,
                training_max_delay,
            )

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(
            observation,
            prev_chunk_left_over=rtc_prefix,
            inference_delay=conditioned_delay_frames,
        )
        # torch.compile CUDA graph replay is async — synchronize so inference timing
        # captures real GPU execution time and postprocessing timing is not inflated.
        if action_tensor.device.type == "cuda":
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            "本次模型推理耗时：%.3f 毫秒，动作张量形状：%s",
            inference_time * 1000,
            tuple(action_tensor.shape),
        )

        """4. Apply postprocessor"""
        # Postprocessor handles (batch, action_dim) tensors. Instead of looping 50 times
        # per timestep of the chunk, squeeze the batch dim and process all timesteps
        # in a single batch call: (1, T, D) → (T, D) → postprocess → (T, D).
        start_postprocess = time.perf_counter()

        # Squeeze batch dim: (1, chunk_size, action_dim) → (chunk_size, action_dim)
        action_tensor = self.postprocessor(action_tensor.squeeze(0))
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess
        self.rtc_latency_tracker.add(postprocess_stops - start_inference)

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        actual_delay_frames = math.ceil(
            max(0.0, postprocess_stops - request_start_perf) * request_fps
        )
        rejection_reason = _rtc_delay_rejection_reason(
            trained_with_prefix=(
                self.rtc_config.mode == "trained" and rtc_prefix is not None
            ),
            actual_delay_frames=actual_delay_frames,
            conditioned_delay_frames=conditioned_delay_frames,
        )
        response_status = "rtc_rejected" if rejection_reason else "ok"
        if rejection_reason:
            action_chunk = []
            self.logger.error(
                "Rejecting trained RTC chunk without breaking the client stream: "
                "actual_delay=%d conditioned_delay=%d predicted_delay=%d "
                "prefix=%d checkpoint_max=%d",
                actual_delay_frames,
                conditioned_delay_frames,
                predicted_delay_frames,
                prefix_frames_used,
                self.policy.config.rtc_training_max_delay,
            )
        response = {
            "response_status": response_status,
            "rejection_reason": rejection_reason,
            "trace_id": str(getattr(observation_t, "trace_id", "")),
            "observation_timestep": observation_t.get_timestep(),
            "inference_delay_frames": actual_delay_frames,
            "predicted_delay_frames": predicted_delay_frames,
            "conditioned_delay_frames": conditioned_delay_frames,
            "guidance_delay_frames": conditioned_delay_frames,
            "prefix_frames_used": prefix_frames_used,
            "rtc_training_max_delay": int(
                getattr(self.policy.config, "rtc_training_max_delay", 0)
            ),
            "expired_prefix_frames": 0,
            "previous_action_start_index": int(
                getattr(observation_t, "previous_action_start_index", 0)
            ),
            "actions": action_chunk,
        }
        timing = {
            "server_preprocess_ms": 1000 * (prepare_time + preprocessing_time),
            "server_model_infer_ms": 1000 * inference_time,
            "server_postprocess_ms": 1000 * postprocessing_time,
        }
        return response, timing

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        compression=grpc.Compression.Gzip,
        options=[
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
        ],
    )
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    bind_address = f"{cfg.host}:{cfg.port}"
    if cfg.require_mtls:
        server_credentials = grpc.ssl_server_credentials(
            private_key_certificate_chain_pairs=(
                (
                    Path(cfg.tls_server_key_path).read_bytes(),
                    Path(cfg.tls_server_cert_path).read_bytes(),
                ),
            ),
            root_certificates=Path(cfg.tls_ca_cert_path).read_bytes(),
            require_client_auth=True,
        )
        bound_port = server.add_secure_port(bind_address, server_credentials)
    else:
        bound_port = server.add_insecure_port(bind_address)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind PolicyServer to {bind_address}")

    policy_server.logger.info(
        "PolicyServer started on %s transport=%s instance=%s",
        bind_address,
        "mTLS" if cfg.require_mtls else "insecure-loopback",
        policy_server.server_instance_id,
    )
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
