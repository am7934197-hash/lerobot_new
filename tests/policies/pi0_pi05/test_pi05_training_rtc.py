#!/usr/bin/env python

import pytest
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    _build_flow_matching_inputs,
    _prepare_trained_rtc_prefix,
    _reduce_training_rtc_loss,
    _sample_training_rtc_prefix_lengths,
    _sample_training_rtc_prefix_mask,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from lerobot.policies.rtc.configuration_rtc import RTCConfig, validate_trained_rtc_horizon


def test_pi05_training_rtc_config_defaults_off_and_validates_range():
    assert PI05Config().rtc_training_max_delay == 0
    assert PI05Config().rtc_training_sampling_mode == "uniform"
    assert PI05Config().rtc_training_sampling_weights is None
    assert PI05Config(chunk_size=50, rtc_training_max_delay=10).rtc_training_max_delay == 10

    with pytest.raises(ValueError, match="rtc_training_max_delay"):
        PI05Config(chunk_size=10, n_action_steps=10, rtc_training_max_delay=10)


def test_training_rtc_weighted_config_normalizes_weights_and_allows_zero():
    config = PI05Config(
        chunk_size=50,
        rtc_training_max_delay=3,
        rtc_training_sampling_mode="weighted",
        rtc_training_sampling_weights=[1, 0, 2, 1],
    )

    assert config.get_rtc_training_sampling_probabilities() == [0.25, 0.0, 0.5, 0.25]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rtc_training_sampling_mode": "other"}, "sampling_mode"),
        ({"rtc_training_sampling_mode": "weighted"}, "must be provided"),
        (
            {
                "rtc_training_max_delay": 2,
                "rtc_training_sampling_mode": "weighted",
                "rtc_training_sampling_weights": [1, 2],
            },
            "exactly",
        ),
        (
            {
                "rtc_training_max_delay": 1,
                "rtc_training_sampling_mode": "weighted",
                "rtc_training_sampling_weights": [1, -1],
            },
            "non-negative",
        ),
        (
            {
                "rtc_training_max_delay": 1,
                "rtc_training_sampling_mode": "weighted",
                "rtc_training_sampling_weights": [1, float("inf")],
            },
            "finite",
        ),
        (
            {
                "rtc_training_max_delay": 1,
                "rtc_training_sampling_mode": "weighted",
                "rtc_training_sampling_weights": [0, 0],
            },
            "sum greater than 0",
        ),
    ],
)
def test_training_rtc_rejects_invalid_sampling_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PI05Config(**kwargs)


def test_training_rtc_uniform_mode_ignores_custom_weights():
    config = PI05Config(
        rtc_training_max_delay=2,
        rtc_training_sampling_mode="uniform",
        rtc_training_sampling_weights=[float("nan")],
    )

    assert config.get_rtc_training_sampling_probabilities() == pytest.approx([1 / 3] * 3)


def test_training_rtc_builds_clean_prefix_with_per_token_time():
    actions = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    noise = actions + 10
    time = torch.tensor([0.25, 0.75])
    prefix_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    x_t, velocity, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask)

    assert model_time.shape == (2, 4)
    assert torch.equal(model_time[prefix_mask], torch.zeros(3))
    assert torch.equal(x_t[prefix_mask], actions[prefix_mask])
    assert torch.equal(velocity, noise - actions)
    expected = time[0] * noise[0, 2] + (1 - time[0]) * actions[0, 2]
    assert torch.allclose(x_t[0, 2], expected)


def test_training_rtc_loss_excludes_prefix_for_both_reductions():
    losses = torch.tensor(
        [
            [[100.0, 100.0], [2.0, 4.0], [6.0, 8.0]],
            [[100.0, 100.0], [10.0, 12.0], [14.0, 16.0]],
        ]
    )
    prefix_mask = torch.tensor([[True, False, False], [True, False, False]])

    per_sample = _reduce_training_rtc_loss(losses, prefix_mask, reduction="none")
    mean = _reduce_training_rtc_loss(losses, prefix_mask, reduction="mean")

    assert torch.allclose(per_sample, torch.tensor([5.0, 13.0]))
    assert torch.allclose(mean, torch.tensor(9.0))


def test_training_rtc_prefix_mask_is_prefix_shaped_and_bounded():
    torch.manual_seed(7)
    mask = _sample_training_rtc_prefix_mask(128, 50, 10, torch.device("cpu"))

    assert mask is not None
    lengths = mask.sum(dim=1)
    assert torch.all(lengths <= 10)
    assert torch.all(mask[:, 1:].to(torch.int8) <= mask[:, :-1].to(torch.int8))
    assert _sample_training_rtc_prefix_mask(2, 50, 0, torch.device("cpu")) is None


def test_training_rtc_uniform_sampling_keeps_original_randint_rng_path():
    torch.manual_seed(123)
    sampled = _sample_training_rtc_prefix_lengths(64, 10, torch.device("cpu"))
    torch.manual_seed(123)
    expected = torch.randint(0, 11, (64,), device=torch.device("cpu"))

    assert torch.equal(sampled, expected)


def test_training_rtc_weighted_sampling_is_per_sample_and_respects_zero_weights():
    torch.manual_seed(9)
    sampled = _sample_training_rtc_prefix_lengths(
        512,
        3,
        torch.device("cpu"),
        sampling_mode="weighted",
        sampling_weights=[1, 0, 2, 1],
    )
    torch.manual_seed(9)
    repeated = _sample_training_rtc_prefix_lengths(
        512,
        3,
        torch.device("cpu"),
        sampling_mode="weighted",
        sampling_weights=[1, 0, 2, 1],
    )

    assert sampled is not None
    assert torch.equal(sampled, repeated)
    assert sampled.device.type == "cpu"
    assert torch.unique(sampled).numel() > 1
    assert not torch.any(sampled == 1)


def test_training_rtc_weighted_sampling_matches_configured_distribution():
    torch.manual_seed(17)
    sampled = _sample_training_rtc_prefix_lengths(
        50_000,
        2,
        torch.device("cpu"),
        sampling_mode="weighted",
        sampling_weights=[1, 3, 6],
    )

    assert sampled is not None
    actual = torch.bincount(sampled, minlength=3).double() / sampled.numel()
    assert torch.allclose(actual, torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64), atol=0.01)


def test_training_rtc_sampling_stats_track_interval_and_cumulative_counts():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = PI05Config(
        rtc_training_max_delay=2,
        rtc_training_sampling_mode="weighted",
        rtc_training_sampling_weights=[1, 0, 3],
    )
    policy._rtc_training_interval_counts = None
    policy._rtc_training_cumulative_counts = None

    policy._record_rtc_training_prefix_lengths(torch.tensor([0, 2, 2, 2]))
    policy._record_rtc_training_prefix_lengths(torch.tensor([0, 0, 2]))
    interval, cumulative = policy.get_rtc_training_sampling_counts()
    metrics = policy.get_rtc_training_sampling_log_metrics(interval, cumulative)

    assert torch.equal(interval, torch.tensor([3, 0, 4]))
    assert torch.equal(cumulative, torch.tensor([3, 0, 4]))
    assert metrics["rtc_sampling/configured_weight/d_1"] == 0
    assert metrics["rtc_sampling/configured_probability/d_2"] == 0.75
    assert metrics["rtc_sampling/interval/count_d_0"] == 3
    assert metrics["rtc_sampling/cumulative/ratio_d_2"] == pytest.approx(4 / 7)

    policy.reset_rtc_training_interval_counts()
    interval, cumulative = policy.get_rtc_training_sampling_counts()
    assert torch.equal(interval, torch.zeros(3, dtype=torch.int64))
    assert torch.equal(cumulative, torch.tensor([3, 0, 4]))


def test_prepare_trained_rtc_prefix_validates_and_pads_action_dim():
    latent = torch.randn(1, 6, 4)
    previous = torch.randn(3, 2)

    prefix, mask = _prepare_trained_rtc_prefix(latent, previous, inference_delay=3, training_max_delay=4)

    assert prefix is not None and mask is not None
    assert prefix.shape == latent.shape
    assert mask.shape == latent.shape
    assert torch.allclose(prefix[0, :3, :2], previous)
    assert torch.count_nonzero(prefix[0, :3, 2:]) == 0
    assert mask[0, :3].all()
    assert not mask[0, 3:].any()

    with pytest.raises(ValueError, match="exceeds"):
        _prepare_trained_rtc_prefix(latent, previous, inference_delay=5, training_max_delay=4)


def test_pi05_time_embedding_and_adarms_accept_per_token_times():
    time = torch.tensor([[0.0, 0.5, 1.0], [0.25, 0.75, 1.0]])
    embedding = create_sinusoidal_pos_embedding(time, 8, 0.004, 4.0, time.device)
    assert embedding.shape == (2, 3, 8)

    norm = PiGemmaRMSNorm(dim=4, cond_dim=8)
    output, gate = norm(torch.randn(2, 3, 4), embedding.float())
    assert output.shape == (2, 3, 4)
    assert gate is not None and gate.shape == (2, 3, 4)


def test_trained_rtc_inference_config_and_horizon_contract():
    assert RTCConfig(mode="trained").mode == "trained"
    validate_trained_rtc_horizon(execution_horizon=10, prediction_steps=50, training_max_delay=10)

    with pytest.raises(ValueError, match="at least"):
        validate_trained_rtc_horizon(execution_horizon=9, prediction_steps=50, training_max_delay=10)
    with pytest.raises(ValueError, match="mode"):
        RTCConfig(mode="unknown")
