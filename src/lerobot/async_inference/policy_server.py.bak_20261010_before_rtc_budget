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
import math
import os
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
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
from lerobot.types import PolicyAction

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


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
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
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
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
                "execution_horizon",
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
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self._configure_rtc(getattr(policy_specs, "rtc_config", None))

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        # max-autotune torch.compile 在推理时容易 CUDA illegal memory access。
        # 默认关掉；需要编译时 export LEROBOT_POLICY_COMPILE=1
        pretrained_path = policy_specs.pretrained_name_or_path
        load_kwargs: dict[str, Any] = {}
        if os.environ.get("LEROBOT_POLICY_COMPILE", "0") != "1":
            policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
            if getattr(policy_cfg, "compile_model", False):
                policy_cfg.compile_model = False
                self.logger.info(
                    "Disabled torch.compile for inference "
                    "(checkpoint had compile_model=true). Set LEROBOT_POLICY_COMPILE=1 to keep it."
                )
            load_kwargs["config"] = policy_cfg

        self.policy = policy_class.from_pretrained(pretrained_path, **load_kwargs)
        if self.policy_type == "pi05":
            self.policy.config.rtc_config = self.rtc_config
            self.policy.init_rtc_processor()
            if self.rtc_config.enabled and not isinstance(self.policy.rtc_processor, RTCProcessor):
                raise RuntimeError("Pi0.5 RTCProcessor was not initialized")
            self.logger.info(
                "Pi0.5 RTC configured: enabled=%s mode=guided execution_horizon=%d "
                "max_guidance_weight=%.3f prefix_attention_schedule=%s processor=%s",
                self.rtc_config.enabled,
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
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
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
            "SERVER TRACE trace_id=%s event=SERVER_REQUEST_RECEIVED perf_s=%.6f",
            trace_id,
            handler_start,
        )

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize
        decoded_at = time.perf_counter()
        trace_id = str(getattr(timed_observation, "trace_id", "") or trace_id)
        timed_observation.trace_id = trace_id
        timed_observation.server_request_start_perf = handler_start
        timed_observation.server_receive_ms = (decoded_at - handler_start) * 1000
        timed_observation.server_queue_enter_perf = decoded_at
        self.logger.info(
            "SERVER TRACE trace_id=%s event=SERVER_OBSERVATION_DECODED perf_s=%.6f "
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

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            infer_at = time.perf_counter()
            self.logger.info(
                "SERVER TRACE trace_id=%s event=SERVER_INFER_START perf_s=%.6f",
                trace_id,
                infer_at,
            )
            action_chunk, stage_timing = self._predict_action_chunk(obs)
            infer_done = time.perf_counter()
            inference_time = infer_done - infer_at
            self.logger.info(
                "SERVER TRACE trace_id=%s event=SERVER_INFER_END perf_s=%.6f "
                "server_model_infer_ms=%.3f prefix_frames_used=%d inference_delay_frames=%d",
                trace_id,
                infer_done,
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
        if self.policy_type == "pi05" and self.rtc_config.enabled and prev_chunk_left_over is not None:
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
        if previous_actions is None or self.policy_type != "pi05" or not self.rtc_config.enabled:
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
            self.policy_image_features,
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
        guidance_delay_frames = math.ceil(
            (elapsed_before_inference + predicted_remaining_latency) * request_fps
        )

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(
            observation,
            prev_chunk_left_over=rtc_prefix,
            inference_delay=guidance_delay_frames,
        )
        # torch.compile CUDA graph replay is async — synchronize so inference timing
        # captures real GPU execution time and postprocessing timing is not inflated.
        if action_tensor.device.type == "cuda":
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
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
        response = {
            "trace_id": str(getattr(observation_t, "trace_id", "")),
            "observation_timestep": observation_t.get_timestep(),
            "inference_delay_frames": actual_delay_frames,
            "guidance_delay_frames": guidance_delay_frames,
            "prefix_frames_used": prefix_frames_used,
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
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
