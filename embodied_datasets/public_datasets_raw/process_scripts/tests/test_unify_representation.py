import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from shared.episode import Episode
from common.schema import ProcessConfig
from unify_representation import apply


def _single_arm_episode(num_frames=4):
    # columns: [6 joint | 3 eef_pos + 4 eef_quat | 1 gripper] = 14
    state = np.zeros((num_frames, 14))
    state[:, :6] = np.arange(6)
    state[:, 6:9] = [1.0, 2.0, 3.0]
    state[:, 9:13] = [0.0, 0.0, 0.0, 1.0]
    state[:, 13] = 0.5
    return Episode(
        episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=state.copy()
    )


def test_skipped_for_non_robot_embodiment():
    episode = _single_arm_episode()
    config = ProcessConfig(id="x", embodiment_class="human_hand")
    result = apply(episode, config)
    assert result.skip_reason == "embodiment_not_robot_collected"


def test_single_arm_parallel_jaw_packs_into_first_35_dims():
    episode = _single_arm_episode()
    config = ProcessConfig(id="x", embodiment_class="single_arm", num_arms=1, dof_per_arm=6, gripper_type="parallel_jaw")
    result = apply(episode, config)
    canonical = result.stats["canonical_state"]
    mask = result.stats["canonical_mask"]
    assert canonical.shape == (4, 80)
    assert np.allclose(canonical[:, :6], np.arange(6))
    assert np.allclose(canonical[:, 7:14], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(canonical[:, 14], 0.5)
    assert np.all(canonical[:, 15:35] == 0)
    assert not np.any(mask[35:70])  # second arm slot untouched for single-arm dataset


def test_dexterous_hand_with_15_dof_leaves_gripper_tail_unmasked():
    # A dexterous hand with fewer DOF than GRIPPER_SLOT (21) still packs
    # correctly and leaves the unused tail of the gripper slot zero/unmasked.
    num_frames = 2
    state = np.zeros((num_frames, 7 + 7 + 15))
    state[:, 7 + 7:] = np.arange(15)
    episode = Episode(episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=state.copy())
    config = ProcessConfig(id="x", embodiment_class="single_arm", num_arms=1, dof_per_arm=7, gripper_type="dexterous_hand")
    result = apply(episode, config)
    canonical = result.stats["canonical_state"]
    mask = result.stats["canonical_mask"]
    assert np.allclose(canonical[:, 14:29], np.arange(15))
    assert np.all(canonical[:, 29:35] == 0)
    assert not np.any(mask[29:35])


def test_dexterous_hand_with_21_dof_packs_without_truncation():
    # Regression for the GRIPPER_SLOT widen: humanoidbench's Shadow Hand
    # reports dof_per_hand=21, which the OLD GRIPPER_SLOT=15 would have
    # silently truncated (dropping the last 6 DOF). With GRIPPER_SLOT=21
    # (ARM_BLOCK_DIM=35), all 21 values must land in canonical[:, 14:35]
    # with mask=True, and nothing should spill into arm2's block at [35:].
    num_frames = 2
    state = np.zeros((num_frames, 7 + 7 + 21))
    state[:, 7 + 7:] = np.arange(21)
    episode = Episode(episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=state.copy())
    config = ProcessConfig(id="x", embodiment_class="humanoid", num_arms=1, dof_per_arm=7, gripper_type="dexterous_hand")
    result = apply(episode, config)
    canonical = result.stats["canonical_state"]
    mask = result.stats["canonical_mask"]
    assert canonical.shape == (num_frames, 80)
    assert np.allclose(canonical[:, 14:35], np.arange(21))
    assert np.all(mask[14:35])
    assert not np.any(mask[35:])


def test_mobile_base_velocity_is_packed_after_arm_columns():
    # 14 arm cols (single arm) + 3 trailing mobile-base vx/vy/yaw cols = 17.
    # Regression for a bug where cols_per_arm was computed by dividing the
    # FULL column count (including the trailing mobile-base columns) by
    # num_arms, leaving no room for the mobile-base check to ever succeed --
    # the has_mobile_base branch was permanently dead code.
    num_frames = 3
    state = np.zeros((num_frames, 17))
    state[:, 14:17] = [0.1, 0.2, 0.3]
    episode = Episode(episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=state.copy())
    config = ProcessConfig(
        id="x", embodiment_class="mobile_manipulator", num_arms=1, dof_per_arm=6,
        gripper_type="parallel_jaw", has_mobile_base=True,
    )
    result = apply(episode, config)
    canonical = result.stats["canonical_state"]
    mask = result.stats["canonical_mask"]
    assert np.allclose(canonical[:, 70:73], [0.1, 0.2, 0.3])
    assert np.all(mask[70:73])


def test_dof_per_arm_exceeding_available_columns_does_not_crash():
    # config.dof_per_arm says 6 joints, but this episode's state only has 3
    # columns total (malformed/misconfigured data). Regression for a crash:
    # canonical[:, offset:offset+joint_width] used a fixed width while the
    # source slice arm_cols[:, :joint_width] was narrower, causing a
    # ValueError broadcast mismatch.
    num_frames = 3
    state = np.zeros((num_frames, 3))
    state[:, :3] = [1.0, 2.0, 3.0]
    episode = Episode(episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=state.copy())
    config = ProcessConfig(id="x", embodiment_class="single_arm", num_arms=1, dof_per_arm=6, gripper_type="parallel_jaw")
    result = apply(episode, config)
    canonical = result.stats["canonical_state"]
    mask = result.stats["canonical_mask"]
    assert np.allclose(canonical[:, :3], [1.0, 2.0, 3.0])
    assert np.all(mask[:3])
    assert not np.any(mask[3:])


def test_negative_dof_per_arm_is_clamped_not_corrupted():
    # dof_per_arm is a field shared across stage4/stage5/unify_representation
    # (common/schema.py leaves it an unconstrained Optional[int], matching
    # convert_scripts' DatasetConfig). stage5_orientation_alignment.py
    # already established the convention of clamping negative values to 0
    # at the call site rather than rejecting them in the schema (see its
    # test_negative_dof_per_arm_is_clamped_not_corrupted). Regression for a
    # crash: `config.dof_per_arm or 0` does NOT catch negative values (a
    # negative int is truthy), so dof_per_arm stayed -2, turning
    # arm_cols[:, :joint_width] / arm_cols[:, eef_start:...] into numpy's
    # negative-index slicing semantics and crashing with a width mismatch.
    episode = _single_arm_episode()
    config = ProcessConfig(id="x", embodiment_class="single_arm", num_arms=1, dof_per_arm=-2, gripper_type="parallel_jaw")
    result = apply(episode, config)
    canonical = result.stats["canonical_state"]
    mask = result.stats["canonical_mask"]
    # Behaves like dof_per_arm=0: eef position slice starts at column 0.
    assert np.allclose(canonical[:, 7:10], [0.0, 1.0, 2.0])
    assert not np.any(mask[0:7])


def test_zero_frames_episode_does_not_crash():
    state = np.zeros((0, 14))
    episode = Episode(episode_index=0, timestamps=np.arange(0, dtype=np.float64), state=state, action=state.copy())
    config = ProcessConfig(id="x", embodiment_class="single_arm", num_arms=1, dof_per_arm=6, gripper_type="parallel_jaw")
    result = apply(episode, config)
    assert result.stats["canonical_state"].shape == (0, 80)


def test_dual_arm_packs_arm1_and_arm2_into_correct_blocks():
    # 2 arms x 14 cols each (6 joint | 3 eef_pos + 4 eef_quat | 1 gripper) = 28
    # total columns, no mobile base. Use distinct, easily-checkable values per
    # arm so a mix-up between arm1[0:35]/arm2[35:70] is caught.
    num_frames = 2
    state = np.zeros((num_frames, 28))
    # arm1: joints 0..5, eef_pos (1,2,3), eef_quat (0,0,0,1), gripper 0.5
    state[:, 0:6] = np.arange(6)
    state[:, 6:9] = [1.0, 2.0, 3.0]
    state[:, 9:13] = [0.0, 0.0, 0.0, 1.0]
    state[:, 13] = 0.5
    # arm2: joints 10..15, eef_pos (4,5,6), eef_quat (1,0,0,0), gripper 0.9
    state[:, 14:20] = np.arange(10, 16)
    state[:, 20:23] = [4.0, 5.0, 6.0]
    state[:, 23:27] = [1.0, 0.0, 0.0, 0.0]
    state[:, 27] = 0.9
    episode = Episode(
        episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=state.copy()
    )
    config = ProcessConfig(id="x", embodiment_class="dual_arm", num_arms=2, dof_per_arm=6, gripper_type="parallel_jaw")
    result = apply(episode, config)
    canonical = result.stats["canonical_state"]
    mask = result.stats["canonical_mask"]
    assert canonical.shape == (num_frames, 80)

    # arm1 -> canonical[:, 0:35]
    assert np.allclose(canonical[:, 0:6], np.arange(6))
    assert np.allclose(canonical[:, 7:14], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(canonical[:, 14], 0.5)
    assert np.all(mask[0:6])
    assert np.all(mask[7:14])
    assert mask[14]
    assert np.all(canonical[:, 15:35] == 0)
    assert not np.any(mask[15:35])

    # arm2 -> canonical[:, 35:70]
    assert np.allclose(canonical[:, 35:41], np.arange(10, 16))
    assert np.allclose(canonical[:, 42:49], [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(canonical[:, 49], 0.9)
    assert np.all(mask[35:41])
    assert np.all(mask[42:49])
    assert mask[49]
    assert np.all(canonical[:, 50:70] == 0)
    assert not np.any(mask[50:70])

    # no mobile base configured -> [70:80] stays zero/unmasked
    assert np.all(canonical[:, 70:80] == 0)
    assert not np.any(mask[70:80])


def test_zero_or_negative_num_arms_clamped_to_one():
    episode = _single_arm_episode()
    for num_arms in (0, -3):
        config = ProcessConfig(id="x", embodiment_class="single_arm", num_arms=num_arms, dof_per_arm=6, gripper_type="parallel_jaw")
        result = apply(episode, config)
        canonical = result.stats["canonical_state"]
        assert np.allclose(canonical[:, :6], np.arange(6))
