import pytest
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, _sample_training_rtc_prefix_lengths


def test_weighted_rtc_sampling_matches_configured_distribution():
    config = PI05Config(
        rtc_training_max_delay=2,
        rtc_training_sampling_mode="weighted",
        rtc_training_sampling_weights=[1, 3, 6],
    )
    assert config.get_rtc_training_sampling_probabilities() == pytest.approx([0.1, 0.3, 0.6])

    torch.manual_seed(17)
    sampled = _sample_training_rtc_prefix_lengths(
        50_000,
        2,
        torch.device("cpu"),
        sampling_mode="weighted",
        sampling_weights=[1, 3, 6],
    )
    actual = torch.bincount(sampled, minlength=3).double() / sampled.numel()
    assert torch.allclose(actual, torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64), atol=0.01)


def test_rtc_sampling_stats_track_interval_and_cumulative_counts():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = PI05Config(
        rtc_training_max_delay=2,
        rtc_training_sampling_mode="weighted",
        rtc_training_sampling_weights=[1, 0, 3],
    )
    # Also exercise the lazy initialization used by native FineART checkpoints.
    policy._record_rtc_training_prefix_lengths(torch.tensor([0, 2, 2, 2]))
    interval, cumulative = policy.get_rtc_training_sampling_counts()
    metrics = policy.get_rtc_training_sampling_log_metrics(interval, cumulative)

    assert torch.equal(interval, torch.tensor([1, 0, 3]))
    assert torch.equal(cumulative, torch.tensor([1, 0, 3]))
    assert metrics["rtc_sampling/configured_probability/d_2"] == 0.75

    policy.reset_rtc_training_interval_counts()
    interval, cumulative = policy.get_rtc_training_sampling_counts()
    assert torch.equal(interval, torch.zeros(3, dtype=torch.int64))
    assert torch.equal(cumulative, torch.tensor([1, 0, 3]))

