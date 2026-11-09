"""Windowed, distributed FineART loss diagnostics.

All inputs are detached tensors emitted by the policy's existing forward pass.
This module never performs another model forward and never participates in the
training loss or backward graph.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


FINEART_LOSS_MONITOR_KEY = "__fineart_loss_monitor__"

_OBJECTIVE_KEYS = (
    "total",
    "flow_raw",
    "text_raw",
    "fast_raw",
    "flow_weighted",
    "text_weighted",
    "fast_weighted",
)
_PRESENCE_KEYS = ("flow_present", "text_present", "fast_present")
_COUNT_KEYS = (
    "text_valid_tokens",
    "text_valid_samples",
    "fast_valid_tokens",
    "fast_valid_samples",
)


def _joint_group(name: str) -> str:
    if name.startswith("right_arm_joint"):
        return "right_arm"
    if name.startswith("left_arm_joint"):
        return "left_arm"
    if name in {"right_arm_gripper", "left_arm_gripper"}:
        return "grippers"
    return "other"


class FineARTLossMonitor:
    """Accumulate all microbatches, reduce sums/counts, and format diagnostics."""

    def __init__(
        self,
        *,
        output_dir: Path,
        action_names: list[str],
        loss_weights: dict[str, float],
        gradient_accumulation_steps: int,
        initial_optimizer_step: int,
        heatmap_every_n_optimizer_steps: int,
        save_jsonl: bool,
        save_heatmap: bool,
        is_main_process: bool,
    ) -> None:
        if len(action_names) != 16:
            raise ValueError(f"S1 FineART loss monitoring requires 16 action names, got {action_names}")
        self.action_names = list(action_names)
        self.groups = [_joint_group(name) for name in action_names]
        group_counts = {group: self.groups.count(group) for group in set(self.groups)}
        if group_counts != {"right_arm": 7, "left_arm": 7, "grippers": 2}:
            raise ValueError(
                "Action metadata does not match the verified S1 layout "
                f"(right=7, left=7, grippers=2): {list(enumerate(action_names))}"
            )

        self.loss_weights = {key: float(value) for key, value in loss_weights.items()}
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.heatmap_every = int(heatmap_every_n_optimizer_steps)
        self.save_jsonl = bool(save_jsonl)
        self.save_heatmap = bool(save_heatmap)
        self.is_main_process = bool(is_main_process)
        self.output_dir = Path(output_dir) / "loss_monitor"
        self.jsonl_path = self.output_dir / "loss_metrics.jsonl"
        self.csv_path = self.output_dir / "loss_metrics.csv"
        self.heatmap_path = self.output_dir / "flow_dim_heatmap.png"
        self._csv_fields: list[str] | None = None
        self._history_steps: list[int] = []
        self._history_values: list[list[float]] = []
        self._next_heatmap_step = (
            ((int(initial_optimizer_step) // self.heatmap_every) + 1) * self.heatmap_every
            if self.heatmap_every > 0
            else None
        )
        self._acc: dict[str, Tensor] | None = None

        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "definition": "Flow Matching MSE in normalized training space; not radians, mm, or action MAE.",
                "flow_dim_reduction": (
                    "RTC postfix time -> per-sample mean; active action samples -> local mean; "
                    "rank/microbatch objectives -> sum/count across the logging window."
                ),
                "flow_real_dim_reduction": "Same as flow_dim, additionally excluding action_is_pad.",
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "loss_weights": self.loss_weights,
                "action_dimensions": [
                    {"index": index, "joint_name": name, "group": group}
                    for index, (name, group) in enumerate(zip(self.action_names, self.groups, strict=True))
                ],
            }
            (self.output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            )
            logging.info(
                "FineART loss monitor: normalized-space Flow Matching MSE; weights flow=%g text=%g fast=%g",
                self.loss_weights["flow"],
                self.loss_weights["text"],
                self.loss_weights["fast"],
            )
            for index, (name, group) in enumerate(zip(self.action_names, self.groups, strict=True)):
                logging.info("FineART action dimension %02d -> %-24s -> %s", index, name, group)

    @staticmethod
    def _as_float64(value: Tensor, device: torch.device) -> Tensor:
        return value.detach().to(device=device, dtype=torch.float64)

    def _ensure_accumulator(self, device: torch.device) -> dict[str, Tensor]:
        if self._acc is None:
            dim = len(self.action_names)
            self._acc = {
                "rank_microbatches": torch.zeros((), dtype=torch.float64, device=device),
                "objective": torch.zeros(len(_OBJECTIVE_KEYS), dtype=torch.float64, device=device),
                "presence": torch.zeros(len(_PRESENCE_KEYS), dtype=torch.float64, device=device),
                "active_raw": torch.zeros(3, dtype=torch.float64, device=device),
                "counts": torch.zeros(len(_COUNT_KEYS), dtype=torch.float64, device=device),
                "flow_dim_local_mean_sum": torch.zeros(dim, dtype=torch.float64, device=device),
                "flow_valid_rank_microbatches": torch.zeros((), dtype=torch.float64, device=device),
                "flow_dim_sample_sum": torch.zeros(dim, dtype=torch.float64, device=device),
                "flow_sample_count": torch.zeros((), dtype=torch.float64, device=device),
                "flow_timestep_count": torch.zeros((), dtype=torch.float64, device=device),
                "real_dim_local_mean_sum": torch.zeros(dim, dtype=torch.float64, device=device),
                "real_valid_rank_microbatches": torch.zeros((), dtype=torch.float64, device=device),
                "real_dim_sample_sum": torch.zeros(dim, dtype=torch.float64, device=device),
                "real_sample_count": torch.zeros((), dtype=torch.float64, device=device),
                "real_timestep_count": torch.zeros((), dtype=torch.float64, device=device),
            }
        return self._acc

    def update(self, output_dict: dict[str, Any] | None) -> None:
        """Consume one local microbatch without synchronizing or retaining a graph."""
        if not output_dict:
            return
        payload = output_dict.get(FINEART_LOSS_MONITOR_KEY)
        if payload is None:
            return
        objective = payload["objective"]
        device = objective["total"].device
        acc = self._ensure_accumulator(device)
        acc["rank_microbatches"] += 1

        objective_values = torch.stack(
            [self._as_float64(objective[key], device) for key in _OBJECTIVE_KEYS]
        )
        presence_values = torch.stack(
            [self._as_float64(objective[key], device) for key in _PRESENCE_KEYS]
        )
        acc["objective"] += objective_values
        acc["presence"] += presence_values
        acc["active_raw"] += objective_values[1:4] * presence_values

        counts = payload["counts"]
        acc["counts"] += torch.stack(
            [self._as_float64(counts[key], device) for key in _COUNT_KEYS]
        )

        flow = payload.get("flow")
        if flow is None:
            return
        sample_count = self._as_float64(flow["sample_count"], device)
        dim_sum = self._as_float64(flow["dim_sum"], device)
        flow_present = sample_count.gt(0).to(torch.float64)
        acc["flow_dim_local_mean_sum"] += dim_sum / sample_count.clamp_min(1) * flow_present
        acc["flow_valid_rank_microbatches"] += flow_present
        acc["flow_dim_sample_sum"] += dim_sum
        acc["flow_sample_count"] += sample_count
        acc["flow_timestep_count"] += self._as_float64(
            flow["supervised_timestep_count"], device
        )

        real_sample_count = flow.get("real_sample_count")
        if real_sample_count is not None:
            real_sample_count = self._as_float64(real_sample_count, device)
            real_present = real_sample_count.gt(0).to(torch.float64)
            real_dim_sum = self._as_float64(flow["real_dim_sum"], device)
            acc["real_dim_local_mean_sum"] += (
                real_dim_sum / real_sample_count.clamp_min(1) * real_present
            )
            acc["real_valid_rank_microbatches"] += real_present
            acc["real_dim_sample_sum"] += real_dim_sum
            acc["real_sample_count"] += real_sample_count
            acc["real_timestep_count"] += self._as_float64(flow["real_timestep_count"], device)

    @staticmethod
    def _distributed_sum(values: dict[str, Tensor]) -> dict[str, Tensor]:
        names = list(values)
        sizes = [values[name].numel() for name in names]
        packed = torch.cat([values[name].reshape(-1) for name in names])
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        reduced: dict[str, Tensor] = {}
        offset = 0
        for name, size in zip(names, sizes, strict=True):
            reduced[name] = packed[offset : offset + size].reshape(values[name].shape)
            offset += size
        return reduced

    def flush(self, *, micro_step: int, optimizer_step: int) -> tuple[dict[str, float], Path | None]:
        """Reduce the complete log window on every rank and reset local accumulators."""
        if self._acc is None:
            return {}, None
        reduced = self._distributed_sum(self._acc)
        for tensor in self._acc.values():
            tensor.zero_()

        rank_microbatches = float(reduced["rank_microbatches"].item())
        if rank_microbatches == 0:
            return {}, None
        objective = (reduced["objective"] / rank_microbatches).cpu().tolist()
        objective_by_name = dict(zip(_OBJECTIVE_KEYS, objective, strict=True))
        presence = reduced["presence"].cpu().tolist()
        presence_by_name = dict(zip(_PRESENCE_KEYS, presence, strict=True))
        weighted_sum = (
            objective_by_name["flow_weighted"]
            + objective_by_name["text_weighted"]
            + objective_by_name["fast_weighted"]
        )

        metrics: dict[str, float] = {
            "steps/micro_step": float(micro_step),
            "steps/optimizer_step": float(optimizer_step),
            "steps/gradient_accumulation": float(self.gradient_accumulation_steps),
            "loss_overview/total": objective_by_name["total"],
            "loss_overview/flow_weighted": objective_by_name["flow_weighted"],
            "loss_overview/text_weighted": objective_by_name["text_weighted"],
            "loss_overview/fast_weighted": objective_by_name["fast_weighted"],
            "loss_raw/flow": objective_by_name["flow_raw"],
            "loss_raw/text": objective_by_name["text_raw"],
            "loss_raw/fast": objective_by_name["fast_raw"],
            "loss_check/weighted_sum": weighted_sum,
            "loss_check/total_minus_weighted_sum": objective_by_name["total"] - weighted_sum,
            "counts/window_rank_microbatches": rank_microbatches,
        }
        for key, value in zip(_COUNT_KEYS, reduced["counts"].cpu().tolist(), strict=True):
            metrics[f"counts/window_{key}"] = float(value)
        for branch_index, branch in enumerate(("flow", "text", "fast")):
            valid = float(presence[branch_index])
            metrics[f"counts/window_{branch}_rank_microbatches"] = valid
            if valid > 0:
                metrics[f"diagnostics/{branch}_active_raw"] = float(
                    reduced["active_raw"][branch_index].item() / valid
                )

        flow_valid = float(reduced["flow_valid_rank_microbatches"].item())
        flow_dim_values: list[float] | None = None
        flow_dim_sample_values: list[float] | None = None
        if flow_valid > 0:
            flow_dims = reduced["flow_dim_local_mean_sum"] / flow_valid
            flow_dim_values = flow_dims.cpu().tolist()
            flow_dim_mean = float(flow_dims.mean().item())
            metrics["diagnostics/flow_dim_mean"] = flow_dim_mean
            metrics["loss_check/flow_dim_mean_minus_active_raw"] = (
                flow_dim_mean - metrics["diagnostics/flow_active_raw"]
            )
            flow_samples = float(reduced["flow_sample_count"].item())
            metrics["counts/window_flow_action_samples"] = flow_samples
            metrics["counts/window_flow_supervised_timesteps"] = float(
                reduced["flow_timestep_count"].item()
            )
            if flow_samples > 0:
                flow_dim_sample_values = (
                    reduced["flow_dim_sample_sum"] / flow_samples
                ).cpu().tolist()
            self._add_dimension_metrics(metrics, "flow_dim", flow_dim_values)
            self._add_group_metrics(metrics, "flow_group", flow_dim_values)

        real_valid = float(reduced["real_valid_rank_microbatches"].item())
        real_dim_values: list[float] | None = None
        if real_valid > 0:
            real_dims = reduced["real_dim_local_mean_sum"] / real_valid
            real_dim_values = real_dims.cpu().tolist()
            metrics["counts/window_flow_real_action_samples"] = float(
                reduced["real_sample_count"].item()
            )
            metrics["counts/window_flow_real_timesteps"] = float(
                reduced["real_timestep_count"].item()
            )
            self._add_dimension_metrics(metrics, "flow_real_dim", real_dim_values)

        record = {
            "micro_step": int(micro_step),
            "optimizer_step": int(optimizer_step),
            "metrics": metrics,
            "flow_dim": flow_dim_values,
            "flow_dim_sample_weighted": flow_dim_sample_values,
            "flow_real_dim": real_dim_values,
        }
        heatmap_path = None
        if self.is_main_process:
            if self.save_jsonl:
                self._write_record(record)
            if flow_dim_values is not None:
                self._history_steps.append(int(optimizer_step))
                self._history_values.append(flow_dim_values)
                if (
                    self.save_heatmap
                    and self._next_heatmap_step is not None
                    and optimizer_step >= self._next_heatmap_step
                ):
                    heatmap_path = self._render_heatmap()
                    while self._next_heatmap_step <= optimizer_step:
                        self._next_heatmap_step += self.heatmap_every
        return metrics, heatmap_path

    def _add_dimension_metrics(self, metrics: dict[str, float], prefix: str, values: list[float]) -> None:
        for name, group, value in zip(self.action_names, self.groups, values, strict=True):
            metrics[f"{prefix}/{group}/{name}"] = float(value)

    def _add_group_metrics(self, metrics: dict[str, float], prefix: str, values: list[float]) -> None:
        for group in ("right_arm", "left_arm", "grippers"):
            selected = [value for value, item_group in zip(values, self.groups, strict=True) if item_group == group]
            metrics[f"{prefix}/{group}"] = float(sum(selected) / len(selected))

    def _write_record(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        flat = {"micro_step": record["micro_step"], "optimizer_step": record["optimizer_step"]}
        flat.update(record["metrics"])
        if self._csv_fields is None:
            self._csv_fields = list(flat)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(flat)

    def _render_heatmap(self) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        values = np.asarray(self._history_values, dtype=np.float64).T
        fig_width = min(20.0, max(10.0, 0.035 * values.shape[1]))
        fig, axis = plt.subplots(figsize=(fig_width, 7.5))
        image = axis.imshow(values, aspect="auto", interpolation="nearest", origin="upper")
        axis.set_yticks(range(len(self.action_names)), labels=self.action_names)
        tick_count = min(12, len(self._history_steps))
        tick_indices = np.linspace(0, len(self._history_steps) - 1, tick_count, dtype=int)
        axis.set_xticks(tick_indices, labels=[self._history_steps[index] for index in tick_indices])
        axis.set_xlabel("optimizer_step")
        axis.set_ylabel("real action dimension")
        axis.set_title("Windowed per-dimension Flow Matching MSE (normalized training space)")
        colorbar = fig.colorbar(image, ax=axis)
        colorbar.set_label("Flow Matching MSE")
        fig.tight_layout()
        fig.savefig(self.heatmap_path, dpi=160)
        plt.close(fig)
        return self.heatmap_path
