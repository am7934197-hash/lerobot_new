from __future__ import annotations

import unittest

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.async_inference.policy_server import _conditioned_rtc_delay_frames
from lerobot.async_inference.policy_server import _rtc_delay_rejection_reason
from lerobot.policies.rtc import RTCConfig


class ProtocolContractTest(unittest.TestCase):
    def test_trained_rtc_delay_uses_margin_and_capacity_bounds(self):
        self.assertEqual(_conditioned_rtc_delay_frames(7, 2, 15, 15), 9)
        self.assertEqual(_conditioned_rtc_delay_frames(14, 2, 15, 15), 15)
        self.assertEqual(_conditioned_rtc_delay_frames(7, 2, 15, 8), 8)

    def test_trained_rtc_delay_overrun_is_recoverable_rejection(self):
        self.assertEqual(
            _rtc_delay_rejection_reason(
                trained_with_prefix=True,
                actual_delay_frames=16,
                conditioned_delay_frames=15,
            ),
            "actual_delay_exceeded_conditioning",
        )
        self.assertIsNone(
            _rtc_delay_rejection_reason(
                trained_with_prefix=True,
                actual_delay_frames=8,
                conditioned_delay_frames=9,
            )
        )

    def test_rtc_delay_margin_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "delay_safety_margin_frames"):
            RTCConfig(delay_safety_margin_frames=-1)

    def test_public_bind_requires_mtls(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            PolicyServerConfig(host="0.0.0.0", port=18003, require_mtls=False)

    def test_response_echoes_all_race_identifiers(self):
        server = PolicyServer(PolicyServerConfig(host="127.0.0.1"))
        observation = TimedObservation(
            timestamp=1.0,
            timestep=2,
            observation={},
            session_id="session-a",
            request_id="request-b",
            query_id="query-c",
            subtask_epoch=3,
            profile="fineart_vla",
            checkpoint_fingerprint="fingerprint",
            control_step=4,
            trajectory_step=5,
            stage_step=6,
            observation_timestamp_ns=7,
        )
        response = server._response_context(observation, "subtask_query")
        self.assertEqual(response["session_id"], "session-a")
        self.assertEqual(response["request_id"], "request-b")
        self.assertEqual(response["query_id"], "query-c")
        self.assertEqual(response["subtask_epoch"], 3)
        self.assertEqual(response["profile"], "fineart_vla")
        self.assertEqual(response["checkpoint_fingerprint"], "fingerprint")
        self.assertEqual(response["server_instance_id"], server.server_instance_id)


if __name__ == "__main__":
    unittest.main()
