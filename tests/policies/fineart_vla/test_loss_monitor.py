from __future__ import annotations

import json

import torch

from lerobot.policies.fineart_vla.modeling_fineart_vla import _compute_flow_monitor_statistics
from lerobot.utils.fineart_loss_monitor import FINEART_LOSS_MONITOR_KEY, FineARTLossMonitor


ACTION_NAMES = [
    *[f"right_arm_joint{i}" for i in range(1, 8)],
    "right_arm_gripper",
    *[f"left_arm_joint{i}" for i in range(1, 8)],
    "left_arm_gripper",
]


def _original_flow_loss(flow, predict_actions, prefix_mask):
    postfix = (~prefix_mask).unsqueeze(-1).expand_as(flow)
    per_sample = (flow * postfix).sum(dim=(1, 2, 3)) / postfix.sum(dim=(1, 2, 3)).clamp_min(1)
    return (per_sample * predict_actions).sum() / predict_actions.sum().clamp_min(1)


def test_per_dimension_statistics_reconstruct_flow_with_rtc_and_padding():
    generator = torch.Generator().manual_seed(7)
    flow = torch.rand(3, 2, 6, 16, generator=generator)
    delays = torch.tensor([[0, 2], [4, 1], [3, 5]])
    positions = torch.arange(flow.shape[2])[None, None, :]
    prefix_mask = positions < delays[:, :, None]
    predict_actions = torch.tensor([True, False, True])
    action_is_pad = torch.tensor(
        [
            [False, False, False, False, True, True],
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
        ]
    )

    expected = _original_flow_loss(flow, predict_actions, prefix_mask)
    stats = _compute_flow_monitor_statistics(flow, predict_actions, prefix_mask, action_is_pad)
    dim_values = stats["dim_sum"] / stats["sample_count"]

    torch.testing.assert_close(dim_values.mean(), expected)
    assert stats["sample_count"].item() == 2
    # The third sample's real frames are entirely inside the RTC clean prefix,
    # so padding-aware diagnostics correctly omit it instead of reporting zero.
    assert stats["real_sample_count"].item() == 1
    assert stats["real_timestep_count"].item() < stats["supervised_timestep_count"].item()


def test_monitoring_statistics_do_not_change_loss_or_gradient():
    source = torch.randn(2, 1, 5, 16, generator=torch.Generator().manual_seed(11))
    prefix_mask = torch.tensor([[[False, False, False, False, False]], [[True, True, False, False, False]]])
    predict_actions = torch.tensor([True, True])

    with_monitor = source.clone().requires_grad_()
    loss_with = _original_flow_loss(with_monitor, predict_actions, prefix_mask)
    stats = _compute_flow_monitor_statistics(with_monitor, predict_actions, prefix_mask, None)
    assert all(not value.requires_grad for value in stats.values())
    loss_with.backward()

    without_monitor = source.clone().requires_grad_()
    loss_without = _original_flow_loss(without_monitor, predict_actions, prefix_mask)
    loss_without.backward()

    torch.testing.assert_close(loss_with, loss_without, rtol=0, atol=0)
    torch.testing.assert_close(with_monitor.grad, without_monitor.grad, rtol=0, atol=0)


def _payload(*, flow_raw, text_raw, fast_raw, flow_stats=None, text_tokens=0, fast_tokens=0):
    flow_present = flow_stats is not None
    total = 10 * flow_raw + text_raw + fast_raw
    return {
        FINEART_LOSS_MONITOR_KEY: {
            "objective": {
                "total": torch.tensor(total),
                "flow_raw": torch.tensor(flow_raw),
                "text_raw": torch.tensor(text_raw),
                "fast_raw": torch.tensor(fast_raw),
                "flow_weighted": torch.tensor(10 * flow_raw),
                "text_weighted": torch.tensor(text_raw),
                "fast_weighted": torch.tensor(fast_raw),
                "flow_present": torch.tensor(flow_present),
                "text_present": torch.tensor(text_tokens > 0),
                "fast_present": torch.tensor(fast_tokens > 0),
            },
            "counts": {
                "text_valid_tokens": torch.tensor(text_tokens),
                "text_valid_samples": torch.tensor(int(text_tokens > 0)),
                "fast_valid_tokens": torch.tensor(fast_tokens),
                "fast_valid_samples": torch.tensor(int(fast_tokens > 0)),
            },
            "flow": flow_stats,
        }
    }


def _monitor(tmp_path):
    return FineARTLossMonitor(
        output_dir=tmp_path,
        action_names=ACTION_NAMES,
        loss_weights={"flow": 10, "text": 1, "fast": 1},
        gradient_accumulation_steps=8,
        initial_optimizer_step=0,
        heatmap_every_n_optimizer_steps=100,
        save_jsonl=True,
        save_heatmap=False,
        is_main_process=True,
    )


def test_window_aggregation_handles_language_action_mixed_and_padding(tmp_path):
    monitor = _monitor(tmp_path)
    flow = torch.arange(2 * 1 * 5 * 16, dtype=torch.float32).reshape(2, 1, 5, 16) / 100
    prefix = torch.tensor([[[False, False, False, False, False]], [[True, True, False, False, False]]])
    predict = torch.tensor([True, True])
    padding = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
    stats = _compute_flow_monitor_statistics(flow, predict, prefix, padding)
    flow_raw = float((stats["dim_sum"] / stats["sample_count"]).mean())

    # One mixed/action microbatch and one pure-language microbatch. The overview
    # keeps the latter's zero flow contribution, while active diagnostics do not.
    monitor.update(_payload(flow_raw=flow_raw, text_raw=2.0, fast_raw=3.0, flow_stats=stats, text_tokens=4, fast_tokens=8))
    monitor.update(_payload(flow_raw=0.0, text_raw=5.0, fast_raw=0.0, text_tokens=6))
    metrics, heatmap = monitor.flush(micro_step=10, optimizer_step=1)

    assert heatmap is None
    assert abs(metrics["loss_check/total_minus_weighted_sum"]) < 1e-6
    assert abs(metrics["loss_check/flow_dim_mean_minus_active_raw"]) < 1e-6
    assert metrics["counts/window_rank_microbatches"] == 2
    assert metrics["counts/window_flow_rank_microbatches"] == 1
    assert metrics["counts/window_text_valid_tokens"] == 10
    assert metrics["steps/micro_step"] == 10
    assert metrics["steps/optimizer_step"] == 1
    assert len([key for key in metrics if key.startswith("flow_dim/")]) == 16
    assert len([key for key in metrics if key.startswith("flow_real_dim/")]) == 16

    records = (tmp_path / "loss_monitor/loss_metrics.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["flow_dim"] is not None
    assert (tmp_path / "loss_monitor/loss_metrics.csv").is_file()


def test_pure_language_window_omits_action_diagnostics(tmp_path):
    monitor = _monitor(tmp_path)
    monitor.update(_payload(flow_raw=0.0, text_raw=1.5, fast_raw=0.0, text_tokens=3))
    metrics, _ = monitor.flush(micro_step=8, optimizer_step=1)

    assert metrics["counts/window_flow_rank_microbatches"] == 0
    assert not any(key.startswith("flow_dim/") for key in metrics)
    assert "diagnostics/flow_active_raw" not in metrics


def test_low_frequency_heatmap_uses_optimizer_step_axis(tmp_path):
    monitor = FineARTLossMonitor(
        output_dir=tmp_path,
        action_names=ACTION_NAMES,
        loss_weights={"flow": 10, "text": 1, "fast": 1},
        gradient_accumulation_steps=8,
        initial_optimizer_step=0,
        heatmap_every_n_optimizer_steps=1,
        save_jsonl=False,
        save_heatmap=True,
        is_main_process=True,
    )
    flow = torch.ones(1, 1, 4, 16)
    stats = _compute_flow_monitor_statistics(
        flow,
        torch.tensor([True]),
        torch.zeros(1, 1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.bool),
    )
    monitor.update(_payload(flow_raw=1.0, text_raw=0.0, fast_raw=0.0, flow_stats=stats))
    _, heatmap = monitor.flush(micro_step=8, optimizer_step=1)

    assert heatmap is not None and heatmap.is_file()
