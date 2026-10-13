import csv

import numpy as np
import torch

from lerobot.utils.action_error_metrics import ActionErrorAccumulator


def test_action_error_metrics_are_per_dim_and_horizon_and_ignore_padding(tmp_path):
    target = torch.zeros(2, 3, 2)
    prediction = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 4.0], [100.0, 100.0]],
            [[3.0, 6.0], [4.0, 8.0], [5.0, 10.0]],
        ]
    )
    is_pad = torch.tensor([[False, False, True], [False, False, False]])

    accumulator = ActionErrorAccumulator()
    accumulator.update(prediction, target, is_pad)
    metrics = accumulator.compute()

    np.testing.assert_allclose(metrics["mae_matrix"], [[2.0, 3.0, 5.0], [4.0, 6.0, 10.0]])
    np.testing.assert_allclose(metrics["mse_matrix"], [[5.0, 10.0, 25.0], [20.0, 40.0, 100.0]])
    np.testing.assert_allclose(metrics["mae_per_dim"], [3.0, 6.0])
    np.testing.assert_allclose(metrics["mse_per_dim"], [11.0, 44.0])

    paths = accumulator.save(tmp_path, prefix="replay", action_names=["left_joint", "right_joint"])
    assert np.load(paths["mae_npy"]).shape == (2, 3)
    with paths["per_dim_csv"].open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["action_name"] == "left_joint"


def test_action_error_scalar_keys_match_replay_logging_contract():
    accumulator = ActionErrorAccumulator()
    accumulator.update(torch.ones(1, 2, 2), torch.zeros(1, 2, 2))

    scalars = accumulator.scalar_dict()

    assert scalars == {
        "mae": 1.0,
        "mse": 1.0,
        "mae_dim_0": 1.0,
        "mae_dim_1": 1.0,
        "mse_dim_0": 1.0,
        "mse_dim_1": 1.0,
    }
