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
import time
from glob import glob
from pathlib import Path

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

# Support both WandB and SwanLab
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


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

    if cfg.is_reward_model_training:
        trainable_tag = f"reward_model:{cfg.reward_model.type}"
    else:
        trainable_tag = f"policy:{cfg.policy.type}"
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


def get_wandb_run_id_from_filesystem(log_dir: Path, backend: str = "wandb") -> str:
    # Get the run ID from filesystem.
    if backend == "swanlab":
        paths = glob(str(log_dir / "swanlog" / "**/swan-*/logs"), recursive=True)
        if len(paths) < 1:
            raise RuntimeError("Couldn't get the previous SwanLab run ID for run resumption.")
        exp_folder = paths[0].split("/")[-2]
        return exp_folder
    else:
        paths = glob(str(log_dir / "wandb/latest-run/run-*"))
        if len(paths) != 1:
            raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
        match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
        if match is None:
            raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
        wandb_run_id = match.groups(0)[0]
        return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """A helper class to log object using wandb or swanlab."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)
        
        # Determine which backend to use
        self.backend = getattr(self.cfg, 'backend', 'wandb')

        if self.backend == "swanlab":
            self._init_swanlab(cfg)
        else:
            self._init_wandb(cfg)

    def _init_wandb(self, cfg: TrainPipelineConfig):
        """Initialize WandB backend."""
        if not WANDB_AVAILABLE:
            raise ImportError("WandB is not installed. Please install it with: pip install wandb")

        os.environ["WANDB_SILENT"] = "True"

        wandb_run_id = (
            cfg.wandb.run_id
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir, backend="wandb")
            if cfg.resume
            else None
        )
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True) if self.cfg.add_tags else None,
            dir=self.log_dir,
            config=cfg.to_dict(),
            save_code=False,
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        run_id = wandb.run.id
        cfg.wandb.run_id = run_id
        self._wandb_custom_step_key: set[str] | None = None
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb
        self._backend_lib = wandb

    def _init_swanlab(self, cfg: TrainPipelineConfig):
        """Initialize SwanLab backend."""
        if not SWANLAB_AVAILABLE:
            raise ImportError("SwanLab is not installed. Please install it with: pip install swanlab")

        swanlab_run_id = (
            cfg.wandb.run_id
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir, backend="swanlab")
            if cfg.resume
            else None
        )
        
        # Map wandb mode to swanlab mode
        mode_map = {
            "online": "cloud",
            "cloud": "cloud",  # Direct mapping for swanlab native mode
            "offline": "offline",
            "disabled": "disabled",
            "local": "local",
        }
        swanlab_mode = mode_map.get(self.cfg.mode, "cloud") if self.cfg.mode else "cloud"
        
        # Get API key from environment
        api_key = os.environ.get("SWANLAB_API_KEY")
        
        logging.info(colored(f"SwanLab initializing with: project={self.cfg.project}, workspace={self.cfg.entity}, mode={swanlab_mode}, api_key={'***' if api_key else None}", "cyan"))
        
        swanlab.init(
            id=swanlab_run_id,
            project=self.cfg.project,
            workspace=self.cfg.entity,
            experiment_name=self.job_name,
            description=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True) if self.cfg.add_tags else None,
            logdir=str(self.log_dir / "swanlog"),
            config=cfg.to_dict(),
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=swanlab_mode,
            api_key=api_key,
        )
        
        logging.info(colored(f"SwanLab init() called, checking run object...", "cyan"))
        # SwanLab uses 'name' for the run identifier, not 'id'
        run_id = swanlab.run.name if hasattr(swanlab.run, 'name') else str(time.time())
        cfg.wandb.run_id = run_id
        self._wandb_custom_step_key: set[str] | None = None
        logging.info(colored("Logs will be synced with swanlab.", "blue", attrs=["bold"]))
        # SwanLab uses 'url' attribute
        if hasattr(swanlab.run, 'url'):
            logging.info(f"Track this run --> {colored(swanlab.run.url, 'yellow', attrs=['bold'])}")
        else:
            logging.warning(colored("SwanLab run URL not available", "yellow"))
        self._wandb = swanlab
        self._backend_lib = swanlab
        logging.info(colored(f"SwanLab backend initialized: {self._backend_lib}", "green", attrs=["bold"]))
        
        # 测试 log 是否工作
        try:
            logging.info(colored("Testing SwanLab log...", "cyan"))
            self._backend_lib.log({"test_init": 1}, step=0)
            logging.info(colored("✓ SwanLab log test passed", "green"))
        except Exception as e:
            logging.error(colored(f"✗ SwanLab log test failed: {e}", "red", attrs=["bold"]))

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb/swanlab."""
        if self.cfg.disable_artifact:
            return

        if self.backend == "swanlab":
            # SwanLab doesn't support artifacts in the same way as WandB
            logging.info(
                f"SwanLab does not support model artifacts like WandB. "
                f"Skipping model artifact logging for checkpoint: {checkpoint_dir}"
            )
            return

        # WandB backend
        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._backend_lib.Artifact(artifact_name, type="model")
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

        self._backend_lib.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval", "replay"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # 检查 backend 是否已初始化
        if not hasattr(self, '_backend_lib') or self._backend_lib is None:
            logging.warning(
                colored(f"Backend not initialized! Cannot log metrics. Backend: {self.backend}", "red", attrs=["bold"])
            )
            return

        if self.backend == "swanlab":
            # SwanLab implementation
            for k, v in d.items():
                # Policy output dictionaries commonly expose detached scalar
                # tensors. SwanLab accepts Python scalars, so preserve these
                # metrics instead of dropping them at the wrapper boundary.
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    v = v.detach().cpu().item()
                if not isinstance(v, (int | float | str)):
                    logging.warning(
                        f'SwanLab logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                    )
                    continue

                if custom_step_key is not None and k == custom_step_key:
                    continue

                if custom_step_key is not None:
                    value_custom_step = d[custom_step_key]
                    data = {f"{mode}/{k}": v, f"{mode}/{custom_step_key}": value_custom_step}
                    logging.debug(f"SwanLab logging: {data}")
                    try:
                        self._backend_lib.log(data)
                    except Exception as e:
                        logging.error(colored(f"SwanLab log error: {e}", "red"))
                    continue

                logging.debug(f"SwanLab logging: {{'{mode}/{k}': {v}, step: {step}}}")
                try:
                    self._backend_lib.log(data={f"{mode}/{k}": v}, step=step)
                except Exception as e:
                    logging.error(colored(f"SwanLab log error: {e}", "red"))
        else:
            # WandB implementation
            if custom_step_key is not None:
                if self._wandb_custom_step_key is None:
                    self._wandb_custom_step_key = set()
                new_custom_key = f"{mode}/{custom_step_key}"
                if new_custom_key not in self._wandb_custom_step_key:
                    self._wandb_custom_step_key.add(new_custom_key)
                    self._backend_lib.define_metric(new_custom_key, hidden=True)

            for k, v in d.items():
                if not isinstance(v, (int | float | str)):
                    logging.warning(
                        f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                    )
                    continue

                if self._wandb_custom_step_key is not None and k in self._wandb_custom_step_key:
                    continue

                if custom_step_key is not None:
                    value_custom_step = d[custom_step_key]
                    data = {f"{mode}/{k}": v, f"{mode}/{custom_step_key}": value_custom_step}
                    self._backend_lib.log(data)
                    continue

                self._backend_lib.log(data={f"{mode}/{k}": v}, step=step)

    def log_image(self, image_path: str | Path, step: int, mode: str = "train", key: str = "heatmap"):
        """Log an image without making callers depend on the selected backend."""
        if mode not in {"train", "eval", "replay"}:
            raise ValueError(mode)
        try:
            image = self._backend_lib.Image(str(image_path))
            self._backend_lib.log({f"{mode}/{key}": image}, step=step)
        except Exception as error:
            logging.error(colored(f"{self.backend} image log error: {error}", "red"))

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        if self.backend == "swanlab":
            # SwanLab supports Video but only GIF format
            # Check if the video is a GIF file
            if not str(video_path).lower().endswith(".gif"):
                logging.warning(
                    f"SwanLab only supports GIF format for video logging. "
                    f"Skipping video: {video_path} (expected .gif file)"
                )
                return
            
            # SwanLab Video API
            try:
                video = self._backend_lib.Video(str(video_path), caption=f"{mode}_video")
                self._backend_lib.log({f"{mode}/video": video}, step=step)
                logging.debug(f"SwanLab logged video: {video_path}")
            except Exception as e:
                logging.error(colored(f"SwanLab video log error: {e}", "red"))
        else:
            # WandB implementation
            wandb_video = self._backend_lib.Video(video_path, fps=self.env_fps, format="mp4")
            self._backend_lib.log({f"{mode}/video": wandb_video}, step=step)

    def finish(self):
        """Finish the run and ensure all data is synced."""
        if self.backend == "swanlab":
            # SwanLab needs to call finish() to sync data to cloud
            self._backend_lib.finish()
        else:
            # WandB
            self._backend_lib.finish()
