#!/usr/bin/env python

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

from pathlib import Path
from unittest.mock import Mock, patch

from lerobot.common.train_utils import (
    get_best_checkpoint_dir,
    get_best_checkpoint_window_start,
    get_step_checkpoint_dir,
    get_step_identifier,
    is_best_checkpoint_candidate_step,
    load_best_checkpoint_metadata,
    load_training_state,
    load_training_step,
    save_best_checkpoint_metadata,
    save_checkpoint,
    save_training_state,
    save_training_step,
    update_last_checkpoint,
)
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_get_step_checkpoint_dir():
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def test_best_checkpoint_window_and_candidate_steps():
    assert get_best_checkpoint_window_start(50_000, 5_000) == 45_000
    assert is_best_checkpoint_candidate_step(45_000, 50_000, 5_000, 50)
    assert is_best_checkpoint_candidate_step(45_050, 50_000, 5_000, 50)
    assert is_best_checkpoint_candidate_step(50_000, 50_000, 5_000, 50)
    assert not is_best_checkpoint_candidate_step(44_999, 50_000, 5_000, 50)
    assert not is_best_checkpoint_candidate_step(45_001, 50_000, 5_000, 50)

    # The final step is always considered even when it is off the regular cadence.
    assert is_best_checkpoint_candidate_step(1_000, 1_000, 333, 50)


def test_save_load_best_checkpoint_metadata(tmp_path):
    checkpoint_dir = get_best_checkpoint_dir(tmp_path)
    checkpoint_dir.mkdir(parents=True)
    save_best_checkpoint_metadata(
        checkpoint_dir,
        step=47_350,
        loss=0.0042,
        total_steps=50_000,
        window_start=45_000,
        every_n_steps=50,
    )

    assert load_best_checkpoint_metadata(checkpoint_dir) == {
        "step": 47_350,
        "loss": 0.0042,
        "total_steps": 50_000,
        "window_start": 45_000,
        "every_n_steps": 50,
    }


def test_save_load_training_step(tmp_path):
    save_training_step(5000, tmp_path)
    assert (tmp_path / TRAINING_STEP).is_file()


def test_load_training_step(tmp_path):
    step = 5000
    save_training_step(step, tmp_path)
    loaded_step = load_training_step(tmp_path)
    assert loaded_step == step


def test_update_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


@patch("lerobot.common.train_utils.save_training_state")
def test_save_checkpoint(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    cfg = Mock()
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


@patch("lerobot.common.train_utils.save_training_state")
def test_save_checkpoint_peft(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    policy.config = Mock()
    policy.config.save_pretrained = Mock()
    cfg = Mock()
    cfg.use_peft = True
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    policy.config.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


def test_save_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert (tmp_path / TRAINING_STATE_DIR).is_dir()
    assert (tmp_path / TRAINING_STATE_DIR / TRAINING_STEP).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / RNG_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_PARAM_GROUPS).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / SCHEDULER_STATE).is_file()


def test_save_load_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    loaded_step, loaded_optimizer, loaded_scheduler = load_training_state(tmp_path, optimizer, scheduler)
    assert loaded_step == 10
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler
