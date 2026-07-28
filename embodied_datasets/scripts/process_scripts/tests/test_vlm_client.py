import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")

from common.vlm_client import OpenAIVLMClient, _encode_frame_to_data_url, _sample_frames


def _view(num_frames: int) -> np.ndarray:
    return np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)


def test_sample_frames_returns_all_when_total_below_five():
    result = _sample_frames(_view(3))
    assert len(result) == 3


def test_sample_frames_returns_all_when_total_equals_five():
    result = _sample_frames(_view(5))
    assert len(result) == 5


def test_sample_frames_returns_empty_when_total_zero():
    result = _sample_frames(_view(0))
    assert result == []


def test_sample_frames_evenly_spaced_when_total_above_five():
    view = _view(20)
    result = _sample_frames(view)
    assert len(result) == 5
    assert np.array_equal(result[0], view[0])
    assert np.array_equal(result[-1], view[19])


def test_encode_frame_to_data_url_produces_valid_jpeg_data_url():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    url = _encode_frame_to_data_url(frame)
    assert url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic bytes


def _fake_response(payload: dict):
    message = SimpleNamespace(content=json.dumps(payload))
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _fake_response_raw(content: str):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _client() -> OpenAIVLMClient:
    return OpenAIVLMClient(base_url="http://localhost:9000/v1", model="qwen2.5-vl-7b-instruct", api_key="test-key")


def test_check_returns_no_frames_available_when_frames_list_empty():
    verdict = _client().check("pick up the cup", [])
    assert verdict.consistent is True
    assert verdict.reason == "no_frames_available"


def test_check_returns_no_frames_available_when_view_has_zero_timesteps():
    verdict = _client().check("pick up the cup", [_view(0)])
    assert verdict.consistent is True
    assert verdict.reason == "no_frames_available"


def test_check_parses_consistent_verdict_from_valid_json():
    client = _client()
    client._client.chat.completions.create = MagicMock(
        return_value=_fake_response(
            {
                "scene_description": "a table with a cup",
                "action_description": "picking up the cup",
                "consistent": True,
                "reason": "matches",
            }
        )
    )
    verdict = client.check("pick up the cup", [_view(3)])
    assert verdict.consistent is True
    assert verdict.reason == "matches"
    client._client.chat.completions.create.assert_called_once()


def test_check_parses_inconsistent_verdict_from_valid_json():
    client = _client()
    client._client.chat.completions.create = MagicMock(
        return_value=_fake_response(
            {
                "scene_description": "a table with a cup",
                "action_description": "pouring water",
                "consistent": False,
                "reason": "instruction says pick up, robot is pouring",
            }
        )
    )
    verdict = client.check("pick up the cup", [_view(3)])
    assert verdict.consistent is False
    assert verdict.reason == "instruction says pick up, robot is pouring"


def test_check_fails_safe_on_malformed_json(monkeypatch):
    monkeypatch.setattr("common.vlm_client.time.sleep", lambda seconds: None)
    client = _client()
    client._client.chat.completions.create = MagicMock(return_value=_fake_response_raw("not valid json"))
    verdict = client.check("pick up the cup", [_view(3)])
    assert verdict.consistent is True
    assert verdict.reason == "vlm_call_failed"


def test_check_fails_safe_on_missing_field(monkeypatch):
    monkeypatch.setattr("common.vlm_client.time.sleep", lambda seconds: None)
    client = _client()
    client._client.chat.completions.create = MagicMock(
        return_value=_fake_response({"scene_description": "x", "action_description": "y"})
    )
    verdict = client.check("pick up the cup", [_view(3)])
    assert verdict.consistent is True
    assert verdict.reason == "vlm_call_failed"


def test_check_retries_three_times_then_fails_safe_on_repeated_network_error(monkeypatch):
    monkeypatch.setattr("common.vlm_client.time.sleep", lambda seconds: None)
    client = _client()
    mock_create = MagicMock(side_effect=ConnectionError("boom"))
    client._client.chat.completions.create = mock_create
    verdict = client.check("pick up the cup", [_view(3)])
    assert verdict.consistent is True
    assert verdict.reason == "vlm_call_failed"
    assert mock_create.call_count == 3
