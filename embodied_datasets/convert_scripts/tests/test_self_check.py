import pytest

pytest.importorskip("lerobot")

from pathlib import Path

import numpy as np

from shared.episode import Episode
from common_convert.report import ConversionReport
from common_convert.self_check import check_fk_consistency, check_scale, check_shape, run_self_check

URDF_PATH = str(Path(__file__).parent / "fixtures" / "simple_arm.urdf")


def _episode(state, action=None):
    return Episode(
        episode_index=0,
        timestamps=np.arange(state.shape[0], dtype=np.float64),
        state=state,
        action=action if action is not None else state.copy(),
    )


def test_check_shape_passes_when_dims_match_declared_config():
    episode = _episode(np.zeros((5, 9)), np.zeros((5, 4)))
    result = check_shape([episode], state_dim=9, action_dim=4)
    assert result.passed


def test_check_shape_fails_when_state_width_mismatches():
    episode = _episode(np.zeros((5, 9)), np.zeros((5, 4)))
    result = check_shape([episode], state_dim=10, action_dim=4)
    assert not result.passed
    assert "state width 9" in result.reasons[0]


def test_check_shape_skips_dims_not_declared():
    episode = _episode(np.zeros((5, 9)), np.zeros((5, 4)))
    result = check_shape([episode], state_dim=None, action_dim=None)
    assert result.passed


def test_check_scale_passes_within_tolerance():
    report = ConversionReport(num_episodes=95, num_frames=1000)
    result = check_scale(report, expected_num_episodes=100)
    assert result.passed


def test_check_scale_fails_when_most_episodes_missing():
    report = ConversionReport(num_episodes=10, num_frames=100)
    result = check_scale(report, expected_num_episodes=100)
    assert not result.passed


def test_check_scale_skips_when_expected_not_declared():
    report = ConversionReport(num_episodes=10, num_frames=100)
    result = check_scale(report, expected_num_episodes=None)
    assert result.passed


def test_check_fk_consistency_passes_when_eef_pos_matches_fk():
    state = np.zeros((3, 2 + 3 + 4))
    state[:, 2:5] = [1.5, 0.0, 0.0]
    state[:, 5:9] = [0.0, 0.0, 0.0, 1.0]
    episode = _episode(state)
    result = check_fk_consistency([episode], urdf_path=URDF_PATH, dof_per_arm=2)
    assert result.passed


def test_check_fk_consistency_fails_on_large_offset():
    state = np.zeros((3, 2 + 3 + 4))
    state[:, 2:5] = [1.5 + 0.5, 0.0, 0.0]  # 50cm off -- way past the 10cm tolerance
    state[:, 5:9] = [0.0, 0.0, 0.0, 1.0]
    episode = _episode(state)
    result = check_fk_consistency([episode], urdf_path=URDF_PATH, dof_per_arm=2)
    assert not result.passed


def test_check_fk_consistency_skips_when_urdf_not_given():
    episode = _episode(np.zeros((3, 9)))
    result = check_fk_consistency([episode], urdf_path=None, dof_per_arm=2)
    assert result.passed


def test_run_self_check_combines_all_three_checks():
    state = np.zeros((3, 2 + 3 + 4))
    state[:, 2:5] = [1.5, 0.0, 0.0]
    state[:, 5:9] = [0.0, 0.0, 0.0, 1.0]
    episode = _episode(state)
    report = ConversionReport(num_episodes=1, num_frames=3)
    result = run_self_check(
        [episode], report, state_dim=9, action_dim=9, expected_num_episodes=1, urdf_path=URDF_PATH, dof_per_arm=2
    )
    assert result.passed
