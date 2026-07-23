import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from common.service_clients import (
    NullClient,
    get_instruction_consistency_client,
    get_video_state_consistency_client,
)


def test_get_instruction_consistency_client_returns_null_client_when_unconfigured():
    client = get_instruction_consistency_client(None)
    assert isinstance(client, NullClient)


def test_get_instruction_consistency_client_raises_for_real_url():
    with pytest.raises(NotImplementedError):
        get_instruction_consistency_client("http://localhost:9000")


def test_get_video_state_consistency_client_returns_null_client_when_unconfigured():
    client = get_video_state_consistency_client(None)
    assert isinstance(client, NullClient)


def test_get_video_state_consistency_client_raises_for_real_url():
    with pytest.raises(NotImplementedError):
        get_video_state_consistency_client("http://localhost:9001")


def test_null_client_check_reports_not_configured():
    client = NullClient()
    verdict = client.check("pick up the cup", [np.zeros((4, 4, 3), dtype=np.uint8)])
    assert verdict.consistent is True
    assert verdict.reason == "vlm_service_not_configured"


def test_null_client_segment_raises():
    client = NullClient()
    with pytest.raises(RuntimeError):
        client.segment(np.zeros((4, 4, 3), dtype=np.uint8))
