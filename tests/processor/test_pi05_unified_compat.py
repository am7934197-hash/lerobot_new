import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.utils.constants import OBS_STATE


def test_state_dropout_only_applies_to_supervised_training_samples():
    step = Pi05PrepareStateTokenizerProcessorStep(state_dropout_prob=1.0)
    observation = {OBS_STATE: torch.zeros(2, 4)}
    tasks = {"task": ["pick_object", "place_object"]}

    training_output = step(
        {
            TransitionKey.OBSERVATION: observation,
            TransitionKey.ACTION: torch.zeros(2, 3, 2),
            TransitionKey.COMPLEMENTARY_DATA: tasks.copy(),
        }
    )
    assert training_output[TransitionKey.COMPLEMENTARY_DATA]["task"] == [
        "Task: pick object;\nAction: ",
        "Task: place object;\nAction: ",
    ]

    inference_output = step(
        {
            TransitionKey.OBSERVATION: observation,
            TransitionKey.ACTION: None,
            TransitionKey.COMPLEMENTARY_DATA: tasks.copy(),
        }
    )
    assert all(
        ", State: " in prompt
        for prompt in inference_output[TransitionKey.COMPLEMENTARY_DATA]["task"]
    )

