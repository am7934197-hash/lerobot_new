import json
from pathlib import Path

import pytest

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.async_inference.policy_server import _conditioned_rtc_delay_frames
from lerobot.async_inference.policy_server import _resolve_local_text_tokenizer
from lerobot.async_inference.policy_server import _rtc_delay_rejection_reason
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc import RTCConfig


def test_fineart_vla_is_supported_by_legacy_policy_server():
    assert "fineart_vla" in SUPPORTED_POLICIES


def test_text_tokenizer_resolves_from_local_hf_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "hub"
    snapshot = cache_root / "models--google--paligemma-3b-pt-224" / "snapshots" / "local"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.delenv("LEROBOT_TEXT_TOKENIZER_PATH", raising=False)

    assert _resolve_local_text_tokenizer("google/paligemma-3b-pt-224") == str(snapshot)


def test_explicit_text_tokenizer_override_has_priority(tmp_path, monkeypatch):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    monkeypatch.setenv("LEROBOT_TEXT_TOKENIZER_PATH", str(tokenizer_path))

    assert _resolve_local_text_tokenizer("google/paligemma-3b-pt-224") == str(
        Path(tokenizer_path)
    )


def test_legacy_checkpoint_type_can_be_supplied_by_policy_server(tmp_path):
    config = PI05Config(rtc_training_max_delay=15)
    config.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.pop("type")
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = PreTrainedConfig.from_pretrained(tmp_path, fallback_policy_type="pi05")

    assert loaded.type == "pi05"
    assert loaded.rtc_training_max_delay == 15


def test_public_bind_requires_mtls():
    with pytest.raises(ValueError, match="loopback"):
        PolicyServerConfig(host="0.0.0.0", port=18003, require_mtls=False)


def test_response_context_echoes_race_identifiers():
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
    assert response["session_id"] == "session-a"
    assert response["request_id"] == "request-b"
    assert response["query_id"] == "query-c"
    assert response["subtask_epoch"] == 3
    assert response["server_instance_id"] == server.server_instance_id


def test_trained_rtc_delay_adds_margin_before_conditioning():
    assert _conditioned_rtc_delay_frames(7, 2, 15, 15) == 9


def test_trained_rtc_delay_is_bounded_by_checkpoint_and_prefix():
    assert _conditioned_rtc_delay_frames(14, 2, 15, 15) == 15
    assert _conditioned_rtc_delay_frames(7, 2, 15, 8) == 8


def test_rtc_delay_margin_must_be_non_negative():
    with pytest.raises(ValueError, match="delay_safety_margin_frames"):
        RTCConfig(delay_safety_margin_frames=-1)


def test_trained_rtc_delay_overrun_returns_rejection_reason():
    assert _rtc_delay_rejection_reason(
        trained_with_prefix=True,
        actual_delay_frames=16,
        conditioned_delay_frames=15,
    ) == "actual_delay_exceeded_conditioning"
    assert _rtc_delay_rejection_reason(
        trained_with_prefix=True,
        actual_delay_frames=8,
        conditioned_delay_frames=9,
    ) is None
