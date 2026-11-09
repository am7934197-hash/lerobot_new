#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass, field
from numbers import Real

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224
RIGHT_ARM_IMAGE_KEY = f"{OBS_IMAGES}.right_arm"


@dataclass
class RightArmImageDropoutConfig:
    """Training-only dropout for the right-arm camera."""

    enabled: bool = False
    probability: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "right_arm_image_dropout.probability must be between 0 and 1, "
                f"got {self.probability}"
            )


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: torch.dtype | None = torch.float32

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # MEM short-horizon observation memory (https://arxiv.org/abs/2603.03596).
    # Historical image tokens are fused inside SigLIP and dropped before the
    # language backbone. Historical proprioceptive states become one continuous
    # backbone token per frame. Both paths are opt-in and independent.
    #
    # MEM pre-trains on six observations spaced one second apart. `memory_stride` is
    # counted in dataset frames, so the default matches that spacing only at 30 fps,
    # the usual LeRobot recording rate. Scale it with the dataset: a 10 fps dataset
    # such as `lerobot/robomme` needs `memory_stride=10` for the same one second.
    use_visual_memory: bool = False
    use_proprioceptive_memory: bool = False
    memory_frames: int = 6
    memory_stride: int = 30
    memory_temporal_attention_every: int = 4

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Existing PI0.5 configs may hide state tokens for supervised examples.
    # Inference always keeps state tokens when proprioceptive memory is disabled.
    state_dropout_prob: float = 0.0

    # During training, independently mask the right-arm camera per sample.
    right_arm_image_dropout: RightArmImageDropoutConfig = field(
        default_factory=RightArmImageDropoutConfig
    )

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None
    # Maximum clean action-prefix length sampled during training. Zero disables trained RTC.
    rtc_training_max_delay: int = 0
    # Prefix lengths are sampled uniformly by default for backward compatibility.
    # In weighted mode, one weight is required for every d in [0, max_delay].
    rtc_training_sampling_mode: str = "uniform"
    rtc_training_sampling_weights: list[float] | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    # Loaded after the full PI0.5/FineART checkpoint so these weights win.
    vision_tower_pretrained_path: str | None = None
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    optimizer_fused: bool | None = None

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError(
                "rtc_training_max_delay must satisfy "
                f"0 <= delay < chunk_size ({self.chunk_size}), got {self.rtc_training_max_delay}"
            )
        if self.rtc_training_sampling_mode not in {"uniform", "weighted"}:
            raise ValueError(
                "rtc_training_sampling_mode must be 'uniform' or 'weighted', "
                f"got {self.rtc_training_sampling_mode!r}"
            )
        if self.rtc_training_sampling_mode == "weighted":
            weights = self.rtc_training_sampling_weights
            if weights is None:
                raise ValueError(
                    "rtc_training_sampling_weights must be provided when "
                    "rtc_training_sampling_mode='weighted'"
                )
            expected_length = self.rtc_training_max_delay + 1
            if len(weights) != expected_length:
                raise ValueError(
                    "rtc_training_sampling_weights must contain exactly "
                    f"rtc_training_max_delay + 1 ({expected_length}) values, got {len(weights)}"
                )
            for index, weight in enumerate(weights):
                if isinstance(weight, bool) or not isinstance(weight, Real):
                    raise ValueError(
                        "rtc_training_sampling_weights must contain finite, non-negative numbers; "
                        f"weight for d={index} is {weight!r}"
                    )
                if not math.isfinite(float(weight)) or weight < 0:
                    raise ValueError(
                        "rtc_training_sampling_weights must contain finite, non-negative numbers; "
                        f"weight for d={index} is {weight!r}"
                    )
            if math.fsum(float(weight) for weight in weights) <= 0:
                raise ValueError("rtc_training_sampling_weights must have a sum greater than 0")

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in [torch.bfloat16, torch.float32]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.memory_frames < 1:
            raise ValueError("memory_frames must be at least 1")
        if self.memory_stride < 1:
            raise ValueError("memory_stride must be at least 1")
        if self.memory_temporal_attention_every < 1:
            raise ValueError("memory_temporal_attention_every must be at least 1")

        if not 0.0 <= self.state_dropout_prob <= 1.0:
            raise ValueError(
                f"state_dropout_prob must be between 0 and 1, got {self.state_dropout_prob}"
            )

    def get_rtc_training_sampling_probabilities(self) -> list[float]:
        """Return normalized probabilities for d=0..rtc_training_max_delay."""
        num_delays = self.rtc_training_max_delay + 1
        if self.rtc_training_sampling_mode == "uniform":
            return [1.0 / num_delays] * num_delays

        weights = self.rtc_training_sampling_weights
        if weights is None:  # pragma: no cover - defensive guard for mutated configs
            raise ValueError("rtc_training_sampling_weights is required in weighted mode")
        total = math.fsum(float(weight) for weight in weights)
        return [float(weight) / total for weight in weights]

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        if self.input_features is None or self.output_features is None:
            raise ValueError(
                "input_features and output_features must be resolved (e.g. from the dataset) "
                "before calling validate_features()"
            )
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

        if self.right_arm_image_dropout.enabled and RIGHT_ARM_IMAGE_KEY not in self.image_features:
            raise ValueError(
                "right_arm_image_dropout is enabled, but the policy has no "
                f"{RIGHT_ARM_IMAGE_KEY!r} image feature"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            fused=self.optimizer_fused,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def image_observation_delta_indices(self) -> list[int] | None:
        if not self.use_visual_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def state_observation_delta_indices(self) -> list[int] | None:
        if not self.use_proprioceptive_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
