"""Streaming action-chunk error metrics and lightweight report writers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class ActionErrorAccumulator:
    """Accumulate MAE/MSE over batches shaped ``[batch, horizon, action_dim]``.

    Counts are kept per horizon step so padded action targets at the end of an
    episode can be excluded without biasing later horizons.
    """

    def __init__(self) -> None:
        self.abs_sum: Tensor | None = None
        self.sq_sum: Tensor | None = None
        self.count: Tensor | None = None
        self.num_batches = 0

    def update(self, prediction: Tensor, target: Tensor, is_pad: Tensor | None = None) -> None:
        if prediction.ndim != 3 or target.ndim != 3:
            raise ValueError(
                f"prediction and target must be [B, H, D], got {prediction.shape} and {target.shape}"
            )
        horizon = min(prediction.shape[1], target.shape[1])
        action_dim = min(prediction.shape[2], target.shape[2])
        error = prediction[:, :horizon, :action_dim].float() - target[:, :horizon, :action_dim].float()

        if is_pad is None:
            valid = torch.ones(error.shape[:2], dtype=torch.bool, device=error.device)
        else:
            valid = ~is_pad[:, :horizon].to(device=error.device, dtype=torch.bool)
            if valid.ndim == 3:
                valid = valid.all(dim=-1)
            if valid.ndim != 2:
                raise ValueError(f"is_pad must be [B, H] or [B, H, 1], got {is_pad.shape}")

        weight = valid.unsqueeze(-1)
        abs_sum = (error.abs() * weight).sum(dim=0).transpose(0, 1).cpu().double()
        sq_sum = (error.square() * weight).sum(dim=0).transpose(0, 1).cpu().double()
        count = valid.sum(dim=0).cpu().double().unsqueeze(0).expand(action_dim, -1)

        if self.abs_sum is None:
            self.abs_sum = torch.zeros_like(abs_sum)
            self.sq_sum = torch.zeros_like(sq_sum)
            self.count = torch.zeros_like(count)
        if self.abs_sum.shape != abs_sum.shape:
            raise ValueError(f"Action shape changed from {self.abs_sum.shape} to {abs_sum.shape}")
        self.abs_sum += abs_sum
        self.sq_sum += sq_sum
        self.count += count
        self.num_batches += 1

    def compute(self) -> dict[str, np.ndarray | float]:
        if self.abs_sum is None or self.sq_sum is None or self.count is None:
            raise RuntimeError("No action errors have been accumulated")
        count = self.count.clamp_min(1)
        mae_matrix = (self.abs_sum / count).numpy()
        mse_matrix = (self.sq_sum / count).numpy()
        dim_count = self.count.sum(dim=1).clamp_min(1)
        mae_per_dim = (self.abs_sum.sum(dim=1) / dim_count).numpy()
        mse_per_dim = (self.sq_sum.sum(dim=1) / dim_count).numpy()
        total_count = self.count.sum().clamp_min(1)
        return {
            "mae": float(self.abs_sum.sum() / total_count),
            "mse": float(self.sq_sum.sum() / total_count),
            "mae_per_dim": mae_per_dim,
            "mse_per_dim": mse_per_dim,
            "mae_matrix": mae_matrix,
            "mse_matrix": mse_matrix,
        }

    def scalar_dict(self) -> dict[str, float]:
        metrics = self.compute()
        result = {"mae": float(metrics["mae"]), "mse": float(metrics["mse"])}
        for dim, value in enumerate(metrics["mae_per_dim"]):
            result[f"mae_dim_{dim}"] = float(value)
        for dim, value in enumerate(metrics["mse_per_dim"]):
            result[f"mse_dim_{dim}"] = float(value)
        return result

    def save(
        self,
        output_dir: str | Path,
        prefix: str = "action_error",
        action_names: list[str] | None = None,
    ) -> dict[str, Path]:
        """Save per-dimension CSV plus ``[action_dim, horizon]`` CSV/NPY matrices."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = self.compute()
        paths = {
            "per_dim_csv": output_dir / f"{prefix}_per_dim.csv",
            "mae_csv": output_dir / f"{prefix}_mae_matrix.csv",
            "mse_csv": output_dir / f"{prefix}_mse_matrix.csv",
            "mae_npy": output_dir / f"{prefix}_mae_matrix.npy",
            "mse_npy": output_dir / f"{prefix}_mse_matrix.npy",
        }
        with paths["per_dim_csv"].open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["action_dim", "action_name", "mae", "mse"])
            for dim, (mae, mse) in enumerate(zip(metrics["mae_per_dim"], metrics["mse_per_dim"], strict=True)):
                name = action_names[dim] if action_names is not None and dim < len(action_names) else ""
                writer.writerow([dim, name, float(mae), float(mse)])
        header = ",".join(f"horizon_{index}" for index in range(metrics["mae_matrix"].shape[1]))
        np.savetxt(paths["mae_csv"], metrics["mae_matrix"], delimiter=",", header=header, comments="")
        np.savetxt(paths["mse_csv"], metrics["mse_matrix"], delimiter=",", header=header, comments="")
        np.save(paths["mae_npy"], metrics["mae_matrix"])
        np.save(paths["mse_npy"], metrics["mse_matrix"])
        return paths

    def save_heatmap(
        self,
        path: str | Path,
        title: str = "Action MAE by dimension and horizon",
        action_names: list[str] | None = None,
    ) -> Path:
        """Save a MAE heatmap. Matplotlib is imported only when this is requested."""
        import matplotlib.pyplot as plt

        matrix = self.compute()["mae_matrix"]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        width = max(10.0, matrix.shape[1] * 0.22)
        height = max(5.0, matrix.shape[0] * 0.28)
        figure, axis = plt.subplots(figsize=(width, height))
        image = axis.imshow(matrix, aspect="auto", origin="upper", interpolation="nearest", cmap="magma")
        axis.set_xlabel("Horizon step")
        axis.set_ylabel("Action dimension")
        axis.set_title(title)
        axis.set_xticks(np.arange(matrix.shape[1]))
        axis.set_yticks(np.arange(matrix.shape[0]))
        if action_names:
            labels = [
                f"{dim}: {action_names[dim]}" if dim < len(action_names) else str(dim)
                for dim in range(matrix.shape[0])
            ]
            axis.set_yticklabels(labels)
        figure.colorbar(image, ax=axis, label="MAE")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path
