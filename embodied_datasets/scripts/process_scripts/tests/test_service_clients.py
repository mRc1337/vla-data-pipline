import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from common.service_clients import (
    NullClient,
    get_instruction_consistency_client,
    get_video_state_consistency_client,
)


def test_get_instruction_consistency_client_returns_null_client_when_unconfigured():
    client = get_instruction_consistency_client(None, None, None)
    assert isinstance(client, NullClient)


def test_get_instruction_consistency_client_raises_when_api_key_env_not_set():
    with pytest.raises(RuntimeError):
        get_instruction_consistency_client("http://localhost:9000", "qwen2.5-vl-7b-instruct", None)


def test_get_instruction_consistency_client_raises_when_named_env_var_missing(monkeypatch):
    monkeypatch.delenv("MISSING_VLM_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_instruction_consistency_client("http://localhost:9000", "qwen2.5-vl-7b-instruct", "MISSING_VLM_KEY")


def test_get_instruction_consistency_client_returns_openai_client_when_fully_configured(monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "secret")
    client = get_instruction_consistency_client("http://localhost:9000", "qwen2.5-vl-7b-instruct", "TEST_VLM_KEY")
    from common.vlm_client import OpenAIVLMClient

    assert isinstance(client, OpenAIVLMClient)


def test_get_video_state_consistency_client_returns_null_client_when_unconfigured():
    client = get_video_state_consistency_client(None)
    assert isinstance(client, NullClient)


def test_get_video_state_consistency_client_returns_local_sam3_client_when_configured():
    from common.sam3_client import LocalSam3Client

    client = get_video_state_consistency_client("/fake/checkpoint.pt")
    assert isinstance(client, LocalSam3Client)


def test_null_client_check_reports_not_configured():
    client = NullClient()
    verdict = client.check("pick up the cup", [np.zeros((4, 4, 3), dtype=np.uint8)])
    assert verdict.consistent is True
    assert verdict.reason == "vlm_service_not_configured"


def test_null_client_segment_raises():
    client = NullClient()
    with pytest.raises(RuntimeError):
        client.segment(np.zeros((4, 4, 3), dtype=np.uint8), (2.0, 2.0))
