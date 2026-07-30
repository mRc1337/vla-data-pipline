import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from episode import Episode
from common.schema import ProcessConfig
from unify_representation import apply

from pathlib import Path

from unify_representation import apply_action

URDF_PATH = str(Path(__file__).parent / "fixtures" / "simple_arm.urdf")


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
    assert canonical.shape == (4, 128)
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
    assert canonical.shape == (num_frames, 128)
    assert np.allclose(canonical[:, 14:35], np.arange(21))
    assert np.all(mask[14:35])
    assert not np.any(mask[35:])


def test_mobile_base_columns_excluded_from_arm_packing_but_not_captured():
    # 14 arm cols (single arm) + 3 trailing mobile-base vx/vy/yaw cols = 17.
    # Mobile-base velocity has no slot in the canonical layout (folded into
    # the [70:128] reserve, not yet implemented) -- but the trailing 3
    # columns must still be excluded from the arm-column split, or they'd
    # shift cols_per_arm and corrupt the arm1 packing checked below.
    num_frames = 3
    state = np.zeros((num_frames, 17))
    state[:, :6] = np.arange(6)
    state[:, 6:9] = [1.0, 2.0, 3.0]
    state[:, 9:13] = [0.0, 0.0, 0.0, 1.0]
    state[:, 13] = 0.5
    state[:, 14:17] = [0.1, 0.2, 0.3]
    episode = Episode(episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=state.copy())
    config = ProcessConfig(
        id="x", embodiment_class="mobile_manipulator", num_arms=1, dof_per_arm=6,
        gripper_type="parallel_jaw", has_mobile_base=True,
    )
    result = apply(episode, config)
    canonical = result.stats["canonical_state"]
    mask = result.stats["canonical_mask"]
    assert np.allclose(canonical[:, :6], np.arange(6))
    assert np.allclose(canonical[:, 7:14], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(canonical[:, 14], 0.5)
    assert np.all(canonical[:, 70:128] == 0)
    assert not np.any(mask[70:128])


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
    # (common/schema.py leaves it an unconstrained Optional[int]).
    # stage5_orientation_alignment.py
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
    assert result.stats["canonical_state"].shape == (0, 128)


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
    assert canonical.shape == (num_frames, 128)

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

    # no mobile base configured -> [70:128] stays zero/unmasked (reserved)
    assert np.all(canonical[:, 70:128] == 0)
    assert not np.any(mask[70:128])


def test_zero_or_negative_num_arms_clamped_to_one():
    episode = _single_arm_episode()
    for num_arms in (0, -3):
        config = ProcessConfig(id="x", embodiment_class="single_arm", num_arms=num_arms, dof_per_arm=6, gripper_type="parallel_jaw")
        result = apply(episode, config)
        canonical = result.stats["canonical_state"]
        assert np.allclose(canonical[:, :6], np.arange(6))


def _single_arm_episode_with_action(action, num_frames=3):
    state = np.zeros((num_frames, 14))  # [6 joint | 3 eef_pos + 4 eef_quat | 1 gripper]
    state[:, 6:9] = [1.0, 2.0, 3.0]
    state[:, 9:13] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion
    action_arr = np.tile(action, (num_frames, 1)).astype(float)
    return Episode(episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=action_arr)


def test_apply_action_skips_unsupported_action_space():
    episode = _single_arm_episode_with_action(np.zeros(8))
    config = ProcessConfig(id="x", action_space="joint_velocity", action_frame="delta")
    result = apply_action(episode, config)
    assert result.skip_reason == "action_space_not_supported"


def test_apply_action_skips_joint_position_without_urdf_available():
    episode = _single_arm_episode_with_action(np.zeros(8))
    config = ProcessConfig(id="x", action_space="joint_position", action_frame="delta", urdf_available=False)
    result = apply_action(episode, config)
    assert result.skip_reason == "fk_not_available_for_joint_action"


def test_apply_action_skips_unsupported_action_frame():
    episode = _single_arm_episode_with_action(np.zeros(8))
    config = ProcessConfig(id="x", action_space="eef_pose", action_frame="both")
    result = apply_action(episode, config)
    assert result.skip_reason == "action_frame_not_supported"


def test_apply_action_eef_pose_delta_frame_passes_through_position_and_axis_angle():
    # action columns: [eef_pos(3), eef_quat(4), gripper(1)] = 8, single arm.
    # Identity quaternion -> zero rotation delta (axis-angle of identity is [0,0,0]).
    action = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.7]
    episode = _single_arm_episode_with_action(action)
    config = ProcessConfig(id="x", action_space="eef_pose", action_frame="delta", num_arms=1, gripper_type="parallel_jaw")
    result = apply_action(episode, config)
    assert result.skip_reason is None
    canonical = result.stats["action_canonical"]
    mask = result.stats["action_canonical_mask"]
    assert canonical.shape == (3, 54)
    assert np.allclose(canonical[:, 0:3], [0.1, 0.2, 0.3])
    assert np.allclose(canonical[:, 3:6], [0.0, 0.0, 0.0])
    assert np.allclose(canonical[:, 6], 0.7)
    assert np.all(mask[0:7])
    assert not np.any(mask[27:54])  # single arm: arm2 block untouched


def test_apply_action_eef_pose_absolute_frame_subtracts_current_state_eef():
    # episode.state's eef_pos is (1,2,3) with identity quat (set by the
    # _single_arm_episode_with_action helper). action targets eef_pos (4,5,6)
    # with identity quat -> delta position = (3,3,3), delta rotation = 0.
    action = [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.5]
    episode = _single_arm_episode_with_action(action)
    config = ProcessConfig(id="x", action_space="eef_pose", action_frame="absolute", num_arms=1, dof_per_arm=6, gripper_type="parallel_jaw")
    result = apply_action(episode, config)
    assert result.skip_reason is None
    canonical = result.stats["action_canonical"]
    assert np.allclose(canonical[:, 0:3], [3.0, 3.0, 3.0])
    assert np.allclose(canonical[:, 3:6], [0.0, 0.0, 0.0])


def test_apply_action_eef_pose_absolute_frame_rotation_delta_with_nonidentity_quats():
    # state (current) rotation = 90 deg about z; target rotation = 90 deg about x.
    # Chosen specifically so the delta discriminates a correct
    # target * current^-1 composition from a swapped-order or missing-.inv()
    # bug -- all three produce different results for this pair (independently
    # verified with scipy.spatial.transform.Rotation in this repo's venv):
    #   correct (target * current^-1):     [ 1.2092,  1.2092, -1.2092]
    #   swapped (current * target^-1):     [-1.2092, -1.2092,  1.2092]
    #   missing .inv() (target * current): [ 1.2092, -1.2092,  1.2092]
    num_frames = 2
    state = np.zeros((num_frames, 14))
    state[:, 6:9] = [1.0, 2.0, 3.0]
    state[:, 9:13] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # 90 deg about z
    action = np.zeros((num_frames, 8))
    action[:, 0:3] = [1.0, 2.0, 3.0]  # same position as state -> position delta is zero, isolates the rotation check
    action[:, 3:7] = [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]  # 90 deg about x
    action[:, 7] = 0.5
    episode = Episode(episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=action)
    config = ProcessConfig(id="x", action_space="eef_pose", action_frame="absolute", num_arms=1, dof_per_arm=6, gripper_type="parallel_jaw")
    result = apply_action(episode, config)
    assert result.skip_reason is None
    canonical = result.stats["action_canonical"]
    assert np.allclose(canonical[:, 0:3], [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(canonical[:, 3:6], [1.2091995761561456, 1.2091995761561456, -1.2091995761561456])


def test_apply_action_joint_position_uses_fk_when_urdf_available():
    # simple_arm.urdf: 2 revolute joints, tool0 at (1.5,0,0) when both joints
    # are 0 (same fixture stage4_fk_consistency.py's tests use). action
    # columns: [joint(2), gripper(1)] = 3, single arm.
    action = [0.0, 0.0, 0.6]
    episode = _single_arm_episode_with_action(action)
    config = ProcessConfig(
        id="x", action_space="joint_position", action_frame="delta",
        urdf_available=True, urdf_path=URDF_PATH, dof_per_arm=2, num_arms=1, gripper_type="parallel_jaw",
    )
    result = apply_action(episode, config)
    assert result.skip_reason is None
    canonical = result.stats["action_canonical"]
    assert np.allclose(canonical[:, 0:3], [1.5, 0.0, 0.0])
    assert np.allclose(canonical[:, 6], 0.6)


def test_apply_action_joint_position_fk_rotation_output_is_correct():
    action = [np.pi / 2, 0.0, 0.6]  # joint1=pi/2, joint2=0, gripper=0.6
    episode = _single_arm_episode_with_action(action)
    config = ProcessConfig(
        id="x", action_space="joint_position", action_frame="delta",
        urdf_available=True, urdf_path=URDF_PATH, dof_per_arm=2, num_arms=1, gripper_type="parallel_jaw",
    )
    result = apply_action(episode, config)
    assert result.skip_reason is None
    canonical = result.stats["action_canonical"]
    assert np.allclose(canonical[:, 0:3], [0.0, 1.5, 0.0], atol=1e-9)
    assert np.allclose(canonical[:, 3:6], [0.0, 0.0, np.pi / 2])


def test_apply_action_joint_position_skips_on_dof_mismatch():
    # simple_arm.urdf has exactly 2 active joints; dof_per_arm=5 mismatches it,
    # mirroring stage4_fk_consistency.py's identical guard.
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.6]
    episode = _single_arm_episode_with_action(action)
    config = ProcessConfig(
        id="x", action_space="joint_position", action_frame="delta",
        urdf_available=True, urdf_path=URDF_PATH, dof_per_arm=5, num_arms=1, gripper_type="parallel_jaw",
    )
    result = apply_action(episode, config)
    assert result.skip_reason == "fk_not_available_for_joint_action"


def test_apply_action_dual_arm_packs_into_correct_blocks():
    # 2 arms x 8 action cols each (eef_pos(3)+eef_quat(4)+gripper(1)) = 16.
    num_frames = 2
    action = np.zeros((num_frames, 16))
    action[:, 0:3] = [1.0, 2.0, 3.0]
    action[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    action[:, 7] = 0.5
    action[:, 8:11] = [4.0, 5.0, 6.0]
    action[:, 11:15] = [0.0, 0.0, 0.0, 1.0]
    action[:, 15] = 0.9
    state = np.zeros((num_frames, 28))  # 2 arms x 14 state cols each, unused by delta frame
    episode = Episode(episode_index=0, timestamps=np.arange(num_frames, dtype=np.float64), state=state, action=action)
    config = ProcessConfig(id="x", action_space="eef_pose", action_frame="delta", num_arms=2, gripper_type="parallel_jaw")
    result = apply_action(episode, config)
    assert result.skip_reason is None
    canonical = result.stats["action_canonical"]
    mask = result.stats["action_canonical_mask"]
    assert canonical.shape == (num_frames, 54)
    assert np.allclose(canonical[:, 0:3], [1.0, 2.0, 3.0])
    assert np.allclose(canonical[:, 6], 0.5)
    assert np.allclose(canonical[:, 27:30], [4.0, 5.0, 6.0])
    assert np.allclose(canonical[:, 33], 0.9)
    assert np.all(mask[0:7])
    assert np.all(mask[27:34])


def test_apply_action_zero_frames_episode_does_not_crash():
    state = np.zeros((0, 14))
    action = np.zeros((0, 8))
    episode = Episode(episode_index=0, timestamps=np.arange(0, dtype=np.float64), state=state, action=action)
    config = ProcessConfig(id="x", action_space="eef_pose", action_frame="delta", num_arms=1, gripper_type="parallel_jaw")
    result = apply_action(episode, config)
    assert result.skip_reason is None
    assert result.stats["action_canonical"].shape == (0, 54)
