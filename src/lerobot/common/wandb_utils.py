#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Literal, cast

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


def cfg_to_group(
    cfg: TrainPipelineConfig, return_list: bool = False, truncate_tags: bool = False, max_tag_length: int = 64
) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""

    def _maybe_truncate(tag: str) -> str:
        """Truncate tag to max_tag_length characters if required.

        wandb rejects tags longer than 64 characters.
        See: https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py
        """
        if len(tag) <= max_tag_length:
            return tag
        return tag[:max_tag_length]

    trainable_kind = "reward_model" if cfg.is_reward_model_training else "policy"
    trainable_tag = f"{trainable_kind}:{cfg.trainable_config.type}"
    lst = [
        trainable_tag,
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    if truncate_tags:
        lst = [_maybe_truncate(tag) for tag in lst]
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    return match.group(1)


def get_swanlab_run_id_from_filesystem(log_dir: Path) -> str:
    paths = glob(str(log_dir / "swanlog" / "**" / "swan-*" / "logs"), recursive=True)
    if not paths:
        raise RuntimeError("Couldn't get the previous SwanLab run ID for run resumption.")
    return Path(paths[0]).parent.name


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """A logging adapter that preserves the historical WandB/SwanLab YAML interface."""

    def __init__(self, cfg: TrainPipelineConfig) -> None:
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)
        self.backend = self.cfg.backend
        self._wandb_custom_step_key: set[str] | None = None

        if self.backend == "swanlab":
            self._init_swanlab(cfg)
            return

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id: str | None = None
        if cfg.wandb.run_id:
            wandb_run_id = cfg.wandb.run_id
        elif cfg.resume:
            # `validate()` always resolves `output_dir` on resume; None here means it was skipped.
            if self.log_dir is None:
                raise ValueError("Resuming a WandB run requires `cfg.output_dir` to locate the previous run.")
            wandb_run_id = get_wandb_run_id_from_filesystem(self.log_dir)
        resume = self.cfg.resume or ("must" if cfg.resume else None)
        if resume is not None and resume not in {"allow", "must", "never", "auto"}:
            raise ValueError(f"Unsupported WandB resume mode: {resume!r}")
        mode: Literal["online", "offline", "disabled"] = "online"
        if self.cfg.mode == "offline":
            mode = "offline"
        elif self.cfg.mode == "disabled":
            mode = "disabled"
        run = wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True) if self.cfg.add_tags else None,
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume=cast(Literal["allow", "must", "never", "auto"] | None, resume),
            mode=mode,
            settings=wandb.Settings(
                console=self.cfg.console,
                console_multipart=self.cfg.console_multipart,
                console_chunk_max_seconds=self.cfg.console_chunk_max_seconds,
            ),
        )
        run_id = run.id
        # NOTE: We will override the cfg.wandb.run_id with the wandb run id.
        # This is because we want to be able to resume the run from the wandb run id.
        cfg.wandb.run_id = run_id
        # Handle custom step key for rl asynchronous training.
        self._wandb_custom_step_key: set[str] | None = None
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def _init_swanlab(self, cfg: TrainPipelineConfig) -> None:
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError("SwanLab is not installed. Install it with `pip install swanlab`.") from exc

        if self.log_dir is None:
            raise ValueError("SwanLab logging requires cfg.output_dir")
        run_id = cfg.wandb.run_id
        if run_id is None and cfg.resume:
            run_id = get_swanlab_run_id_from_filesystem(self.log_dir)
        mode_map = {
            "online": "cloud",
            "cloud": "cloud",
            "offline": "offline",
            "disabled": "disabled",
            "local": "local",
        }
        mode = mode_map.get(self.cfg.mode or "online", "cloud")
        swanlab.init(
            id=run_id,
            project=self.cfg.project,
            workspace=self.cfg.entity,
            experiment_name=self.job_name,
            description=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True) if self.cfg.add_tags else None,
            logdir=str(self.log_dir / "swanlog"),
            config=cfg.to_dict(),
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=mode,
            api_key=os.environ.get("SWANLAB_API_KEY"),
        )
        run = getattr(swanlab, "run", None)
        cfg.wandb.run_id = getattr(run, "name", None) or run_id
        logging.info(colored("Logs will be synced with swanlab.", "blue", attrs=["bold"]))
        run_url = getattr(run, "url", None)
        if run_url:
            logging.info(f"Track this run --> {colored(run_url, 'yellow', attrs=['bold'])}")
        self._wandb = swanlab

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoint the policy to the active tracking backend."""
        if self.cfg.disable_artifact:
            return
        if self.backend == "swanlab":
            logging.info("SwanLab artifact upload is disabled for checkpoint %s", checkpoint_dir)
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        # Check if this is a PEFT model (has adapter files instead of model.safetensors)
        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            # PEFT model: add adapter files and configs
            artifact.add_file(adapter_model_file)
            adapter_config_file = pretrained_model_dir / "adapter_config.json"
            if adapter_config_file.exists():
                artifact.add_file(adapter_config_file)
            # Also add the policy config which is needed for loading
            config_file = pretrained_model_dir / "config.json"
            if config_file.exists():
                artifact.add_file(config_file)
        elif standard_model_file.exists():
            # Standard model: add the single safetensors file
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to WandB."
            )
            return

        self._wandb.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval", "replay"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        if self.backend == "swanlab":
            batch_data = {}
            for key, value in d.items():
                if custom_step_key is not None and key == custom_step_key:
                    continue
                if hasattr(value, "numel") and value.numel() == 1:
                    value = value.detach().cpu().item()
                if not isinstance(value, (int | float | str)):
                    logging.warning(
                        'SwanLab logging of key "%s" was ignored because type %s is unsupported.',
                        key,
                        type(value),
                    )
                    continue
                batch_data[f"{mode}/{key}"] = value
            if custom_step_key is not None:
                batch_data[f"{mode}/{custom_step_key}"] = d[custom_step_key]
            if batch_data:
                self._wandb.log(batch_data, step=step)
            return

        # NOTE: This is not simple. Wandb step must always monotonically increase and it
        # increases with each wandb.log call, but in the case of asynchronous RL for example,
        # multiple time steps is possible. For example, the interaction step with the environment,
        # the training step, the evaluation step, etc. So we need to define a custom step key
        # to log the correct step for each metric.
        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                self._wandb.define_metric(new_custom_key, hidden=True)

        batch_data = {}
        for k, v in d.items():
            # Skip the custom step key here, it's added to the batch below.
            if custom_step_key is not None and k == custom_step_key:
                continue

            if not isinstance(v, (int | float | str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue

            batch_data[f"{mode}/{k}"] = v

        if batch_data:
            if custom_step_key is not None:
                batch_data[f"{mode}/{custom_step_key}"] = d[custom_step_key]
                self._wandb.log(batch_data)
            else:
                self._wandb.log(data=batch_data, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        if self.backend == "swanlab":
            if not str(video_path).lower().endswith(".gif"):
                logging.warning("SwanLab video logging currently requires GIF; skipping %s", video_path)
                return
            video = self._wandb.Video(str(video_path), caption=f"{mode}_video")
        else:
            video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": video}, step=step)

    def log_image(self, image_path: str | Path, step: int, mode: str = "train", key: str = "image"):
        if mode not in {"train", "eval", "replay"}:
            raise ValueError(mode)
        image = self._wandb.Image(str(image_path))
        self._wandb.log({f"{mode}/{key}": image}, step=step)

    def finish(self) -> None:
        self._wandb.finish()
