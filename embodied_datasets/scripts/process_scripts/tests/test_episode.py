import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from episode import CameraCalibration, Episode, StageResult


def test_episode_construction():
    ep = Episode(
        episode_index=0,
        timestamps=np.arange(5, dtype=np.float64),
        state=np.zeros((5, 3)),
        action=np.zeros((5, 2)),
    )
    assert ep.episode_index == 0
    assert ep.frames == {}
    assert ep.language_instruction is None
    assert ep.state.shape == (5, 3)


def test_episode_with_frames_and_instruction():
    ep = Episode(
        episode_index=1,
        timestamps=np.arange(3, dtype=np.float64),
        state=np.zeros((3, 2)),
        action=np.zeros((3, 2)),
        frames={"head": np.zeros((3, 4, 4, 3), dtype=np.uint8)},
        language_instruction="pick up the cup",
    )
    assert ep.frames["head"].shape == (3, 4, 4, 3)
    assert ep.language_instruction == "pick up the cup"


def test_stage_result_defaults():
    ep = Episode(
        episode_index=0,
        timestamps=np.arange(2, dtype=np.float64),
        state=np.zeros((2, 1)),
        action=np.zeros((2, 1)),
    )
    result = StageResult(episode=ep)
    assert result.dropped_frame_indices == []
    assert result.rejected is False
    assert result.skip_reason is None
    assert result.stats == {}


def test_episode_defaults_camera_calibration_to_empty_dict():
    ep = Episode(
        episode_index=0,
        timestamps=np.arange(5, dtype=np.float64),
        state=np.zeros((5, 3)),
        action=np.zeros((5, 2)),
    )
    assert ep.camera_calibration == {}


def test_episode_with_camera_calibration():
    calibration = CameraCalibration(fx=100.0, fy=100.0, cx=16.0, cy=16.0, extrinsics=np.eye(4))
    ep = Episode(
        episode_index=0,
        timestamps=np.arange(3, dtype=np.float64),
        state=np.zeros((3, 2)),
        action=np.zeros((3, 2)),
        camera_calibration={"observation.image": calibration},
    )
    assert ep.camera_calibration["observation.image"].fx == 100.0
    assert ep.camera_calibration["observation.image"].extrinsics.shape == (4, 4)
