import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")

from common.sam3_client import LocalSam3Client


def test_construction_does_not_load_predictor(monkeypatch):
    load_calls = []
    monkeypatch.setattr(LocalSam3Client, "_load_predictor", lambda self: load_calls.append(1) or "fake-predictor")

    LocalSam3Client(checkpoint_path="/fake/path.pt")

    assert load_calls == []


def test_segment_loads_predictor_once_and_caches(monkeypatch):
    load_calls = []

    def fake_load(self):
        load_calls.append(1)
        return "fake-predictor"

    def fake_run(self, predictor, frame, point_prompt):
        assert predictor == "fake-predictor"
        return np.zeros(frame.shape[:2], dtype=bool)

    monkeypatch.setattr(LocalSam3Client, "_load_predictor", fake_load)
    monkeypatch.setattr(LocalSam3Client, "_run_point_prompt", fake_run)

    client = LocalSam3Client(checkpoint_path="/fake/path.pt")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    client.segment(frame, (4.0, 4.0))
    client.segment(frame, (5.0, 5.0))

    assert len(load_calls) == 1


def test_segment_returns_bool_mask_matching_frame_shape(monkeypatch):
    monkeypatch.setattr(LocalSam3Client, "_load_predictor", lambda self: "fake-predictor")

    def fake_run(self, predictor, frame, point_prompt):
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[0, 0] = True
        return mask

    monkeypatch.setattr(LocalSam3Client, "_run_point_prompt", fake_run)

    client = LocalSam3Client(checkpoint_path="/fake/path.pt")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = client.segment(frame, (4.0, 4.0))

    assert mask.shape == (8, 8)
    assert mask.dtype == bool


def test_segment_raises_not_implemented_by_default():
    client = LocalSam3Client(checkpoint_path="/fake/path.pt")
    with pytest.raises(NotImplementedError):
        client.segment(np.zeros((4, 4, 3), dtype=np.uint8), (2.0, 2.0))
