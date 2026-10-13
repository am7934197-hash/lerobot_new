#!/usr/bin/env python

import os
import unittest
from unittest import mock

import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import (
    RIGHT_ARM_IMAGE_KEY,
    PI05Config,
    RightArmImageDropoutConfig,
)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_STATE


HEAD_IMAGE_KEY = "observation.images.head_right"
LEFT_ARM_IMAGE_KEY = "observation.images.left_arm"


def _visual_feature(size: int = 4) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.VISUAL, shape=(3, size, size))


def _make_bare_policy(
    *, enabled: bool, probability: float, training: bool, image_size: int = 4
) -> PI05Policy:
    config = PI05Config(
        device="cpu",
        image_resolution=(image_size, image_size),
        right_arm_image_dropout=RightArmImageDropoutConfig(
            enabled=enabled,
            probability=probability,
        ),
    )
    config.input_features = {
        HEAD_IMAGE_KEY: _visual_feature(image_size),
        LEFT_ARM_IMAGE_KEY: _visual_feature(image_size),
        RIGHT_ARM_IMAGE_KEY: _visual_feature(image_size),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }

    policy = object.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.register_parameter("_device_anchor", nn.Parameter(torch.empty(0)))
    policy.train(training)
    return policy


def _batch(batch_size: int = 4, image_size: int = 4) -> dict[str, torch.Tensor]:
    shape = (batch_size, 3, image_size, image_size)
    return {
        HEAD_IMAGE_KEY: torch.full(shape, 0.25),
        LEFT_ARM_IMAGE_KEY: torch.full(shape, 0.50),
        RIGHT_ARM_IMAGE_KEY: torch.full(shape, 0.75),
    }


class TestRightArmImageDropout(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dump_patch = mock.patch.dict(os.environ, {"PI05_DUMP_MODEL_INPUTS": "0"})
        self.input_dump_patch.start()
        self.addCleanup(self.input_dump_patch.stop)

    def test_masks_only_right_arm_during_training(self) -> None:
        policy = _make_bare_policy(enabled=True, probability=1.0, training=True)

        images, masks = policy._preprocess_images(_batch())

        self.assertTrue(torch.allclose(images[0], torch.full_like(images[0], -0.5)))
        self.assertTrue(torch.allclose(images[1], torch.zeros_like(images[1])))
        self.assertTrue(torch.equal(images[2], torch.full_like(images[2], -1.0)))
        self.assertTrue(masks[0].all())
        self.assertTrue(masks[1].all())
        self.assertFalse(masks[2].any())

    def test_can_be_disabled(self) -> None:
        cases = [
            (False, 1.0, True),
            (True, 0.0, True),
            (True, 1.0, False),
        ]
        for enabled, probability, training in cases:
            with self.subTest(enabled=enabled, probability=probability, training=training):
                policy = _make_bare_policy(
                    enabled=enabled,
                    probability=probability,
                    training=training,
                )

                images, masks = policy._preprocess_images(_batch())

                self.assertTrue(torch.allclose(images[2], torch.full_like(images[2], 0.5)))
                self.assertTrue(all(mask.all() for mask in masks))

    def test_disabled_dropout_does_not_consume_random_numbers(self) -> None:
        policy = _make_bare_policy(enabled=False, probability=1.0, training=True)
        torch.manual_seed(42)
        expected = torch.rand(4)
        torch.manual_seed(42)

        policy._preprocess_images(_batch())
        actual = torch.rand(4)

        self.assertTrue(torch.equal(actual, expected))

    def test_probability_is_per_sample(self) -> None:
        batch_size = 10_000
        policy = _make_bare_policy(
            enabled=True,
            probability=0.2,
            training=True,
            image_size=1,
        )
        torch.manual_seed(42)

        _, masks = policy._preprocess_images(_batch(batch_size=batch_size, image_size=1))
        dropped_ratio = (~masks[2]).float().mean().item()

        self.assertAlmostEqual(dropped_ratio, 0.2, delta=0.02)

    def test_rejects_invalid_probability(self) -> None:
        for probability in (-0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "probability must be between 0 and 1"):
                    RightArmImageDropoutConfig(probability=probability)

    def test_enabled_dropout_requires_camera_feature(self) -> None:
        config = PI05Config(
            device="cpu",
            right_arm_image_dropout=RightArmImageDropoutConfig(enabled=True),
        )
        config.input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            HEAD_IMAGE_KEY: _visual_feature(),
        }
        config.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        }

        with self.assertRaisesRegex(
            ValueError,
            "has no 'observation.images.right_arm' image feature",
        ):
            config.validate_features()


if __name__ == "__main__":
    unittest.main()
