import math

import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np
from scipy.spatial.transform import Rotation

from episode import Episode
from common.schema import ProcessConfig
from stage5_orientation_alignment import apply


def _episode(num_frames=5):
    full_state = np.zeros((num_frames, 2 + 3 + 4))
    full_state[:, 2:5] = [1.0, 0.0, 0.0]
    full_state[:, 5:9] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion
    return Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=full_state,
        action=full_state.copy(),
    )


def test_skipped_without_transform():
    episode = _episode()
    config = ProcessConfig(id="x", dof_per_arm=2)
    result = apply(episode, config)
    assert result.skip_reason == "no_base_to_world_transform_configured"


def test_translation_is_applied():
    episode = _episode()
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert np.allclose(result.episode.state[:, 2:5], [1.0, 0.0, 1.0])


def test_rotation_is_applied_to_position_and_quaternion():
    episode = _episode()
    rotation = Rotation.from_euler("z", 90, degrees=True)
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert np.allclose(result.episode.state[0, 2:5], [0.0, 1.0, 0.0], atol=1e-6)
    new_quat = Rotation.from_quat(result.episode.state[0, 5:9])
    assert np.allclose(new_quat.as_euler("xyz", degrees=True)[2], 90.0, atol=1e-4)


def test_transform_with_wrong_element_count_is_skipped_not_crashed():
    episode = _episode()
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=[1.0, 2.0, 3.0])
    result = apply(episode, config)
    assert result.skip_reason == "invalid_base_to_world_transform"
    assert np.array_equal(result.episode.state, episode.state)


def test_non_rotation_matrix_is_skipped_not_crashed():
    episode = _episode()
    transform = np.eye(4)
    transform[:3, :3] = np.diag([-1.0, 1.0, 1.0])  # reflection: det = -1, not a valid rotation
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert result.skip_reason == "invalid_base_to_world_transform"
    assert np.array_equal(result.episode.state, episode.state)


def test_nan_transform_is_skipped_not_crashed():
    episode = _episode()
    transform = np.full((4, 4), math.nan)
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert result.skip_reason == "invalid_base_to_world_transform"
    assert np.array_equal(result.episode.state, episode.state)


def test_inf_transform_is_skipped_not_hung():
    episode = _episode()
    transform = np.eye(4)
    transform[0, 0] = math.inf
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert result.skip_reason == "invalid_base_to_world_transform"
    assert np.array_equal(result.episode.state, episode.state)


def test_dof_per_arm_leaving_insufficient_columns_for_position_is_skipped_not_crashed():
    episode = _episode()  # 9 columns total: 2 joint + 3 pos + 4 quat
    transform = np.eye(4)
    config = ProcessConfig(id="x", dof_per_arm=8, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert result.skip_reason == "state_too_narrow_for_eef_slice"
    assert np.array_equal(result.episode.state, episode.state)


def test_dof_per_arm_larger_than_state_width_is_skipped_not_crashed():
    episode = _episode()
    transform = np.eye(4)
    config = ProcessConfig(id="x", dof_per_arm=100, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert result.skip_reason == "state_too_narrow_for_eef_slice"
    assert np.array_equal(result.episode.state, episode.state)


def test_negative_dof_per_arm_is_clamped_not_corrupted():
    episode = _episode()
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]
    config = ProcessConfig(id="x", dof_per_arm=-2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    # Negative dof_per_arm is clamped to 0, so this behaves like
    # dof_per_arm=0: the eef position slice starts at column 0
    # (_episode()'s columns 0:3 are [0.0, 0.0, 1.0]), and the translation
    # [0, 0, 1] is added on top -> [0.0, 0.0, 2.0]. What matters here is
    # that this does not raise; the exact (clamped) column choice is
    # covered by test_missing_dof_per_arm_defaults_to_zero_joint_columns.
    assert np.allclose(result.episode.state[:, 0:3], [0.0, 0.0, 2.0])
    # The quat_slice this clamping selects (columns 3:7) is not a valid
    # quaternion ([0.0, 0.0, 0.0, 0.0], zero norm) -- it must be left
    # untouched rather than crashing Rotation.from_quat.
    assert np.allclose(result.episode.state[:, 3:7], [0.0, 0.0, 0.0, 0.0])


def test_zero_frames_does_not_crash():
    episode = _episode(num_frames=0)
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert result.episode.state.shape == (0, 9)


def test_zero_norm_quaternion_row_is_left_untouched_not_crashed():
    # Some frames may carry an all-zero (unfilled/invalid) quaternion --
    # e.g. from upstream zero-padding rather than a dof_per_arm mismatch.
    # Rotation.from_quat raises ValueError on a zero-norm row; the
    # position transform must still apply, and the degenerate quaternion
    # must be left as-is rather than crashing the whole episode.
    episode = _episode()
    episode.state[2, 5:9] = [0.0, 0.0, 0.0, 0.0]  # frame 2's quat is degenerate
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert np.allclose(result.episode.state[:, 2:5], [1.0, 0.0, 1.0])  # position still transformed
    assert np.allclose(result.episode.state[2, 5:9], [0.0, 0.0, 0.0, 0.0])  # degenerate quat untouched
    assert np.allclose(result.episode.state[0, 5:9], [0.0, 0.0, 0.0, 1.0])  # other frames' quats untouched too


def test_inf_quaternion_row_is_left_untouched_not_crashed():
    # A literal Inf in a quaternion column passes a bare `norm > 1e-8`
    # check (inf > 1e-8 is True), and Rotation.from_quat would silently
    # produce NaN, crashing later at the `.as_quat()` composition. The
    # guard must reject non-finite rows outright.
    episode = _episode()
    episode.state[2, 5:9] = [math.inf, 0.0, 0.0, 1.0]  # frame 2's quat has a literal Inf
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert np.allclose(result.episode.state[:, 2:5], [1.0, 0.0, 1.0])  # position still transformed
    assert np.array_equal(result.episode.state[2, 5:9], [math.inf, 0.0, 0.0, 1.0])  # degenerate quat untouched
    assert np.allclose(result.episode.state[0, 5:9], [0.0, 0.0, 0.0, 1.0])  # other frames' quats untouched too


def test_overflowing_norm_quaternion_row_is_left_untouched_not_crashed():
    # Finite-but-huge components whose L2 norm overflows to inf also pass
    # a bare `norm > 1e-8` check, since the norm computation itself
    # overflows to inf before the comparison runs.
    episode = _episode()
    episode.state[2, 5:9] = [1e200, 1e200, 1e200, 1e200]  # frame 2's quat overflows to inf norm
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]
    config = ProcessConfig(id="x", dof_per_arm=2, base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert np.allclose(result.episode.state[:, 2:5], [1.0, 0.0, 1.0])  # position still transformed
    assert np.array_equal(result.episode.state[2, 5:9], [1e200, 1e200, 1e200, 1e200])  # degenerate quat untouched
    assert np.allclose(result.episode.state[0, 5:9], [0.0, 0.0, 0.0, 1.0])  # other frames' quats untouched too


def test_missing_dof_per_arm_defaults_to_zero_joint_columns():
    num_frames = 3
    full_state = np.zeros((num_frames, 3 + 4))
    full_state[:, 0:3] = [1.0, 0.0, 0.0]
    full_state[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=full_state,
        action=full_state.copy(),
    )
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]
    config = ProcessConfig(id="x", base_to_world_transform=transform.flatten().tolist())
    result = apply(episode, config)
    assert np.allclose(result.episode.state[:, 0:3], [1.0, 0.0, 1.0])
