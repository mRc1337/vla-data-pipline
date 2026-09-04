from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

from common.sam3_client import LocalSam3Client


def _client() -> LocalSam3Client:
    return LocalSam3Client(model_id="facebook/sam3", text_prompt="robot gripper", hf_token="fake-token")


def test_construction_does_not_load_predictor(monkeypatch):
    load_calls = []
    monkeypatch.setattr(LocalSam3Client, "_load_predictor", lambda self: load_calls.append(1) or "fake-predictor")

    _client()

    assert load_calls == []


def test_segment_loads_predictor_once_and_caches(monkeypatch):
    load_calls = []

    def fake_load(self):
        load_calls.append(1)
        return "fake-predictor"

    def fake_run(self, predictor, frame, box_prompt):
        assert predictor == "fake-predictor"
        return np.zeros(frame.shape[:2], dtype=bool)

    monkeypatch.setattr(LocalSam3Client, "_load_predictor", fake_load)
    monkeypatch.setattr(LocalSam3Client, "_run_box_prompt", fake_run)

    client = _client()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    client.segment(frame, (2.0, 2.0, 6.0, 6.0))
    client.segment(frame, (1.0, 1.0, 5.0, 5.0))

    assert len(load_calls) == 1


def _fake_instance_segmentation_result(boxes, masks_hw, scores=None):
    """boxes: list of (x0,y0,x1,y1). masks_hw: list of (H,W) 0/1 arrays."""
    num = len(boxes)
    return [
        {
            "scores": torch.tensor(scores if scores is not None else [0.9] * num),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "masks": torch.tensor(np.stack(masks_hw) if masks_hw else np.zeros((0, 8, 8)), dtype=torch.long),
        }
    ]


def test_run_box_prompt_calls_processor_with_text_and_box(monkeypatch):
    client = _client()

    mask = np.zeros((8, 8), dtype=np.int64)
    mask[2:6, 2:6] = 1
    fake_processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 8, 8)})
    fake_processor.post_process_instance_segmentation.return_value = _fake_instance_segmentation_result(
        [[2.0, 2.0, 6.0, 6.0]], [mask]
    )
    fake_model = MagicMock(return_value=SimpleNamespace())

    monkeypatch.setattr(client, "_load_predictor", lambda: (fake_model, fake_processor))

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result_mask = client.segment(frame, (2.0, 2.0, 6.0, 6.0))

    assert result_mask.shape == (8, 8)
    assert result_mask.dtype == bool
    assert result_mask[3, 3] == True  # noqa: E712 -- numpy bool, `is True` would fail
    assert result_mask[0, 0] == False  # noqa: E712

    _, call_kwargs = fake_processor.call_args
    assert call_kwargs["text"] == "robot gripper"
    assert call_kwargs["input_boxes"] == [[[2.0, 2.0, 6.0, 6.0]]]


def test_run_box_prompt_returns_zero_mask_when_no_predictions(monkeypatch):
    client = _client()

    fake_processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 8, 8)})
    fake_processor.post_process_instance_segmentation.return_value = _fake_instance_segmentation_result([], [])
    fake_model = MagicMock(return_value=SimpleNamespace())

    monkeypatch.setattr(client, "_load_predictor", lambda: (fake_model, fake_processor))

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result_mask = client.segment(frame, (2.0, 2.0, 6.0, 6.0))

    assert result_mask.shape == (8, 8)
    assert not result_mask.any()


def test_run_box_prompt_picks_prediction_closest_to_query_box(monkeypatch):
    client = _client()

    near_mask = np.zeros((8, 8), dtype=np.int64)
    near_mask[2:6, 2:6] = 1
    far_mask = np.zeros((8, 8), dtype=np.int64)
    far_mask[0:2, 0:2] = 1

    # Query box is centered at (4, 4). The far prediction's box is centered
    # near (1, 1); the near prediction's box is centered at (4, 4) exactly.
    fake_processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 8, 8)})
    fake_processor.post_process_instance_segmentation.return_value = _fake_instance_segmentation_result(
        [[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 6.0, 6.0]], [far_mask, near_mask]
    )
    fake_model = MagicMock(return_value=SimpleNamespace())

    monkeypatch.setattr(client, "_load_predictor", lambda: (fake_model, fake_processor))

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result_mask = client.segment(frame, (2.0, 2.0, 6.0, 6.0))

    assert np.array_equal(result_mask, near_mask.astype(bool))
