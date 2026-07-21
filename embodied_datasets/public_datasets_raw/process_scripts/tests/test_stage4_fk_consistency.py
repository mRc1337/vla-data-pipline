import pytest

lerobot = pytest.importorskip("lerobot")

import math
from pathlib import Path

import numpy as np

from common.episode import Episode
from common.schema import ProcessConfig
from stage4_fk_consistency import apply

URDF_PATH = str(Path(__file__).parent / "fixtures" / "simple_arm.urdf")


def _episode_with_offset(offset_x: float, num_frames=20):
    joints = np.zeros((num_frames, 2))  # both joints at 0 -> FK position = [1.5, 0, 0]
    state = np.zeros((num_frames, 3 + 4 + 2))  # [pos(3) + quat(4) + gripper(2)]... but joints go first per contract
    # Layout per module contract: [joint(dof_per_arm) | eef_pos(3) + eef_quat(4) | gripper]
    full_state = np.zeros((num_frames, 2 + 3 + 4))
    full_state[:, :2] = joints
    full_state[:, 2:5] = [1.5 + offset_x, 0.0, 0.0]
    full_state[:, 5:9] = [0.0, 0.0, 0.0, 1.0]
    return Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=full_state,
        action=full_state.copy(),
    )


def test_skips_when_not_feasible():
    episode = _episode_with_offset(0.0)
    config = ProcessConfig(id="x", fk_check_feasible=False)
    result = apply(episode, config)
    assert result.skip_reason == "fk_check_not_feasible"


def test_systematic_offset_is_corrected():
    episode = _episode_with_offset(0.1)  # 10cm systematic offset, all frames identical
    config = ProcessConfig(id="x", fk_check_feasible=True, urdf_path=URDF_PATH, dof_per_arm=2, tcp_offset_tolerance=0.02)
    result = apply(episode, config)
    assert result.stats["corrected"] is True
    assert np.allclose(result.episode.state[:, 2], 1.5, atol=1e-6)


def test_small_offset_within_tolerance_is_untouched():
    episode = _episode_with_offset(0.001)
    config = ProcessConfig(id="x", fk_check_feasible=True, urdf_path=URDF_PATH, dof_per_arm=2, tcp_offset_tolerance=0.02)
    result = apply(episode, config)
    assert result.stats.get("corrected", False) is False
    assert np.allclose(result.episode.state[:, 2], episode.state[:, 2])


def test_missing_dof_per_arm_is_skipped_gracefully():
    episode = _episode_with_offset(0.0)
    config = ProcessConfig(id="x", fk_check_feasible=True, urdf_path=URDF_PATH, dof_per_arm=None, tcp_offset_tolerance=0.02)
    result = apply(episode, config)
    assert result.skip_reason == "fk_check_not_feasible"


def test_dof_per_arm_exceeding_state_width_is_skipped_not_crashed():
    episode = _episode_with_offset(0.0)
    config = ProcessConfig(id="x", fk_check_feasible=True, urdf_path=URDF_PATH, dof_per_arm=100, tcp_offset_tolerance=0.02)
    result = apply(episode, config)
    assert result.skip_reason == "fk_check_not_feasible"


def test_dof_per_arm_matches_urdf_but_state_too_narrow_for_eef_slice_is_skipped():
    # dof_per_arm (2) correctly matches the URDF's active-joint count, but
    # the episode itself doesn't have enough columns to hold both the
    # joint values and the EEF_POS_WIDTH-wide reported-position slice
    # (e.g. a corrupted/truncated episode). Must skip, not raise a
    # broadcast ValueError while assigning into a too-narrow slice.
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(5, dtype=np.float64),
        state=np.zeros((5, 3)),
        action=np.zeros((5, 3)),
    )
    config = ProcessConfig(id="x", fk_check_feasible=True, urdf_path=URDF_PATH, dof_per_arm=2, tcp_offset_tolerance=0.02)
    result = apply(episode, config)
    assert result.skip_reason == "fk_check_not_feasible"


def test_zero_frame_episode_is_skipped_not_nan():
    episode = Episode(
        episode_index=0,
        timestamps=np.zeros((0,), dtype=np.float64),
        state=np.zeros((0, 9)),
        action=np.zeros((0, 9)),
    )
    config = ProcessConfig(id="x", fk_check_feasible=True, urdf_path=URDF_PATH, dof_per_arm=2, tcp_offset_tolerance=0.02)
    result = apply(episode, config)
    assert result.skip_reason == "fk_check_not_feasible"


def test_dof_per_arm_mismatched_with_urdf_active_joints_does_not_corrupt_eef_data():
    # simple_arm.urdf has exactly 2 active (non-fixed) joints. If
    # config.dof_per_arm disagrees with that count, FkChain.forward()'s
    # zip()-based angle assignment would silently zero-pad or truncate the
    # joint vector (see common/fk_backend.py), producing a wrong-but-not-
    # crashing FK position. That wrong position can look like a constant
    # systematic sensor offset relative to the (actually correct) reported
    # eef position, and get "corrected" -- overwriting good data with a
    # wrong value derived from a mismatched joint count. This must be
    # refused instead.
    num_frames = 10
    state = np.zeros((num_frames, 1 + 3 + 4))  # dof_per_arm=1 but urdf needs 2
    true_eef_pos = np.array([1.0, 0.5, 0.0])  # reachable only with joint2 != 0
    state[:, 1:4] = true_eef_pos
    state[:, 4:8] = [0.0, 0.0, 0.0, 1.0]
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=state,
        action=state.copy(),
    )
    config = ProcessConfig(id="x", fk_check_feasible=True, urdf_path=URDF_PATH, dof_per_arm=1, tcp_offset_tolerance=0.02)
    result = apply(episode, config)
    assert result.skip_reason == "fk_check_not_feasible"
    assert np.allclose(result.episode.state[:, 1:4], true_eef_pos)
