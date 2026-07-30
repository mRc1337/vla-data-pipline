"""Cross-embodiment canonical 128-dim state projection (design doc
section 8). Per-arm 35-dim block = [joint(<=7) | eef_pos(3)+eef_quat(4) |
gripper_or_hand_slot(<=21)]; dual-arm datasets concatenate arm1[0:35] +
arm2[35:70]; [70:128] is reserved for future whole-body control / other
sensor modalities (left zero) -- mobile-base velocity (vx/vy/yaw) is
carved out of the arm-column split when has_mobile_base is set (so it
doesn't corrupt arm/gripper packing) but is not itself captured in any
canonical slot; the raw values live only in staging, not this layout.

Assumes episode.state columns are ALREADY ordered per-arm as [joint |
eef_pos+eef_quat | gripper_or_hand_slot], concatenated arm-by-arm -- this
module only pads/truncates/repositions into the fixed 128-dim layout, it
does not reorder raw per-dataset columns. That reordering is whatever
upstream process produced the staging data's responsibility.

Known limitations:
- This module trusts `config.dof_per_arm` as the authoritative
  joint/eef boundary for each arm; it has no independent ground truth
  (e.g. a URDF) to cross-check that value against the arm's true joint
  count. Unlike stage4, which can detect a `dof_per_arm` mismatch via its
  URDF and skip the episode, this module CANNOT detect the analogous
  mismatch: if `dof_per_arm` is misconfigured (e.g. set larger than the
  arm's actual joint count) but the arm's total column width still
  happens to be >= the assumed joint+eef+gripper width, `_pack_arm` will
  silently pack real data into the wrong canonical slot -- e.g. real
  eef_pos/eef_quat values landing in the joint slot, and the real
  gripper reading landing in what downstream consumers read as an eef
  dim -- with `mask=True` set on those wrong dims and no error raised.
  Correctness here depends entirely on `dof_per_arm` being accurate for
  the arm it describes; getting that right is `run_pipeline.py`'s /
  the config author's responsibility, not something this module verifies
  or can verify.
"""
from __future__ import annotations

from typing import Set

import numpy as np

from episode import Episode, StageResult
from common.schema import ProcessConfig
from fk_backend import FkChain
from scipy.spatial.transform import Rotation

JOINT_SLOT = 7
EEF_SLOT = 7
GRIPPER_SLOT = 21
ARM_BLOCK_DIM = JOINT_SLOT + EEF_SLOT + GRIPPER_SLOT  # 35
CANONICAL_DIM = 128

ROBOT_EMBODIMENT_CLASSES: Set[str] = {
    "single_arm",
    "dual_arm",
    "half_humanoid",
    "humanoid",
    "mobile_manipulator",
    "quadruped",
}
SIMPLE_GRIPPER_TYPES: Set[str] = {"parallel_jaw", "three_jaw", "cage_pinch", "suction"}

ACTION_JOINT_SLOT = 7
ACTION_EEF_POS_SLOT = 3
ACTION_EEF_ROT_SLOT = 3
ACTION_ARM_BLOCK_DIM = ACTION_JOINT_SLOT + ACTION_EEF_POS_SLOT + ACTION_EEF_ROT_SLOT + 1 + GRIPPER_SLOT  # 35
ACTION_CANONICAL_DIM = 128

SUPPORTED_ACTION_SPACES: Set[str] = {"eef_pose", "joint_position"}
SUPPORTED_ACTION_FRAMES: Set[str] = {"delta", "absolute"}


def _pack_gripper(canonical: np.ndarray, mask: np.ndarray, offset: int, gripper_cols: np.ndarray, gripper_type: str) -> None:
    """Packs up to GRIPPER_SLOT columns of gripper/hand data at
    canonical[:, offset:offset+width]. Shared by the state layer's
    _pack_arm and the action layer's apply_action()."""
    if gripper_type in SIMPLE_GRIPPER_TYPES:
        width = min(gripper_cols.shape[1], 1)
    elif gripper_type == "dexterous_hand":
        width = min(gripper_cols.shape[1], GRIPPER_SLOT)
    else:
        width = 0
    if width > 0:
        canonical[:, offset:offset + width] = gripper_cols[:, :width]
        mask[offset:offset + width] = True


def _pack_arm(canonical: np.ndarray, mask: np.ndarray, offset: int, arm_cols: np.ndarray, dof_per_arm: int, gripper_type: str) -> None:
    # Slice first, then derive width from the *actual* slice (not from
    # dof_per_arm/JOINT_SLOT/EEF_SLOT directly): if arm_cols has fewer
    # columns than the assumed joint+eef layout expects (e.g. dof_per_arm
    # misconfigured larger than this arm's real column count, or a
    # malformed episode with too few state columns), numpy slicing beyond
    # the array's width silently truncates rather than raising, so
    # joint_cols/eef_cols end up narrower than JOINT_SLOT/EEF_SLOT. Using a
    # fixed width on the destination side while the source is narrower is
    # what caused a ValueError broadcast crash; deriving both sides from
    # the same slice keeps them in sync and degrades to a smaller packed
    # region instead of crashing.
    joint_cols = arm_cols[:, :min(dof_per_arm, JOINT_SLOT)]
    joint_width = joint_cols.shape[1]
    if joint_width > 0:
        canonical[:, offset:offset + joint_width] = joint_cols
        mask[offset:offset + joint_width] = True

    eef_start = dof_per_arm
    eef_offset = offset + JOINT_SLOT
    eef_cols = arm_cols[:, eef_start:eef_start + EEF_SLOT]
    eef_width = eef_cols.shape[1]
    if eef_width > 0:
        canonical[:, eef_offset:eef_offset + eef_width] = eef_cols
        mask[eef_offset:eef_offset + eef_width] = True

    gripper_start = eef_start + EEF_SLOT
    gripper_cols = arm_cols[:, gripper_start:]
    gripper_offset = eef_offset + EEF_SLOT
    _pack_gripper(canonical, mask, gripper_offset, gripper_cols, gripper_type)


def _state_arm_slice(episode: Episode, config: ProcessConfig, arm_idx: int, num_arms: int) -> np.ndarray:
    """Returns arm `arm_idx`'s raw per-dataset state columns, using the same
    mobile-base carve-out + equal-division convention apply() uses. Shared
    by apply() and apply_action() -- the latter needs each arm's *original*
    reported eef pose from episode.state to compute action_frame="absolute"
    deltas, which must use this exact same column-slicing convention.

    The mobile-base vx/vy/yaw columns (when present) are appended AFTER all
    arm columns in episode.state, and must be carved out of the total width
    BEFORE dividing the remainder among arms -- otherwise they'd shift
    cols_per_arm and corrupt the arm/gripper packing. Mobile-base velocity
    itself has no slot in the canonical layout (folded into the [70:128]
    reserve, not yet implemented), so the carved-out values are discarded
    rather than written anywhere.
    """
    mobile_base_width = 3 if config.has_mobile_base else 0
    arm_cols_total = max(episode.state.shape[1] - mobile_base_width, 0)
    cols_per_arm = arm_cols_total // num_arms
    start = arm_idx * cols_per_arm
    return episode.state[:, start:start + cols_per_arm]


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if config.embodiment_class not in ROBOT_EMBODIMENT_CLASSES:
        return StageResult(episode=episode, skip_reason="embodiment_not_robot_collected")

    num_frames = episode.state.shape[0]
    canonical = np.zeros((num_frames, CANONICAL_DIM), dtype=np.float64)
    mask = np.zeros(CANONICAL_DIM, dtype=bool)

    # dof_per_arm may legitimately be None/0 (no joint columns). Clamp
    # negative values (not a sensible column offset -- `x or 0` alone does
    # NOT catch these since a negative int is truthy) down to 0, matching
    # the established convention in stage5_orientation_alignment.py for
    # this same shared ProcessConfig field. Without this, a negative
    # dof_per_arm turns joint_cols/eef_start slicing into numpy's
    # negative-index semantics (e.g. arr[:, :-2] means "all but the last
    # 2", not "0 columns"), producing a width mismatch crash in _pack_arm.
    dof_per_arm = max(config.dof_per_arm or 0, 0)
    num_arms = max(1, min(config.num_arms, 2))

    for arm_idx in range(num_arms):
        arm_cols = _state_arm_slice(episode, config, arm_idx, num_arms)
        _pack_arm(canonical, mask, arm_idx * ARM_BLOCK_DIM, arm_cols, dof_per_arm, config.gripper_type)

    return StageResult(
        episode=episode,
        stats={"canonical_state": canonical, "canonical_mask": mask, "canonical_dim": CANONICAL_DIM},
    )


def apply_action(episode: Episode, config: ProcessConfig) -> StageResult:
    """Cross-embodiment canonical 128-dim action projection (design doc
    docs/superpowers/specs/2026-07-30-action-canonicalization-128dim-design.md).
    Independent numbering from apply()'s 128-dim state layout -- offsets are
    not shared, only some constant widths coincide numerically (both layers
    are joint(7)+eef(7)+gripper(21)=35 per arm). Assumes action/state
    rotation columns are already quaternion (same upstream-staging contract
    apply()'s module docstring documents for state).

    Unlike the state layer, the joint slot and eef slot populate somewhat
    independently: the joint slot never needs forward kinematics (it's a
    direct or state-subtracted copy of the raw joint action), so an
    unavailable/mismatched URDF only leaves the eef slot at zero/mask=False
    (recorded via stats["eef_slot_skip_reason"]) rather than skipping the
    whole episode's action canonicalization.

    Must be called with the SAME episode apply() was just called with,
    before episode.state is replaced with canonical_state -- this function
    reads episode.state's original per-dataset eef/joint values (via
    _state_arm_slice) to compute action_frame="absolute" deltas.
    """
    if config.action_space not in SUPPORTED_ACTION_SPACES:
        return StageResult(episode=episode, skip_reason="action_space_not_supported")
    if config.action_frame not in SUPPORTED_ACTION_FRAMES:
        return StageResult(episode=episode, skip_reason="action_frame_not_supported")

    dof_per_arm = max(config.dof_per_arm or 0, 0)
    num_arms = max(1, min(config.num_arms, 2))

    chain = None
    fk_available = True
    if config.action_space == "joint_position":
        if not config.urdf_available or not config.urdf_path:
            fk_available = False
        else:
            try:
                chain = FkChain(config.urdf_path)
            except ValueError:
                # See stage4_fk_consistency.py's identical except-clause: a
                # URDF whose structure doesn't match ikpy's assumptions is a
                # property of the dataset's URDF, not a bug in this pipeline.
                fk_available = False
            else:
                if dof_per_arm != len(chain._active_link_indices):
                    fk_available = False
                    chain = None

    num_frames = episode.action.shape[0]
    action_canonical = np.zeros((num_frames, ACTION_CANONICAL_DIM), dtype=np.float64)
    mask = np.zeros(ACTION_CANONICAL_DIM, dtype=bool)
    action_cols_per_arm = episode.action.shape[1] // num_arms

    stats = {
        "action_canonical": action_canonical,
        "action_canonical_mask": mask,
        "action_canonical_dim": ACTION_CANONICAL_DIM,
    }

    for arm_idx in range(num_arms):
        start = arm_idx * action_cols_per_arm
        arm_action_cols = episode.action[:, start:start + action_cols_per_arm]
        offset = arm_idx * ACTION_ARM_BLOCK_DIM

        if config.action_space == "joint_position":
            joint_action_cols = arm_action_cols[:, :dof_per_arm]
            gripper_action_cols = arm_action_cols[:, dof_per_arm:]
        else:  # eef_pose
            joint_action_cols = None
            gripper_action_cols = arm_action_cols[:, 7:]

        # --- joint slot: no FK needed, populates whenever raw joint columns exist ---
        if joint_action_cols is not None:
            joint_width = min(joint_action_cols.shape[1], ACTION_JOINT_SLOT)
            if joint_width > 0:
                if config.action_frame == "delta":
                    joint_values = joint_action_cols[:, :joint_width]
                else:  # absolute
                    state_arm_cols = _state_arm_slice(episode, config, arm_idx, num_arms)
                    joint_values = joint_action_cols[:, :joint_width] - state_arm_cols[:, :joint_width]
                action_canonical[:, offset:offset + joint_width] = joint_values
                mask[offset:offset + joint_width] = True

        # --- eef slot: needs FK when action_space is joint_position ---
        eef_offset = offset + ACTION_JOINT_SLOT
        if config.action_space == "joint_position" and not fk_available:
            stats["eef_slot_skip_reason"] = "fk_not_available_for_joint_action"
        else:
            if config.action_space == "joint_position":
                target_pos = np.zeros((num_frames, 3))
                target_quat = np.zeros((num_frames, 4))
                target_quat[:, 3] = 1.0
                for t in range(num_frames):
                    pos, quat = chain.forward(joint_action_cols[t])
                    target_pos[t] = pos
                    target_quat[t] = quat
            else:  # eef_pose
                target_pos = arm_action_cols[:, :3]
                target_quat = arm_action_cols[:, 3:7]

            target_rotvec = Rotation.from_quat(target_quat).as_rotvec()

            if config.action_frame == "delta":
                pos_delta = target_pos
                rot_delta = target_rotvec
            else:  # absolute
                state_arm_cols = _state_arm_slice(episode, config, arm_idx, num_arms)
                state_eef_pos = state_arm_cols[:, dof_per_arm:dof_per_arm + 3]
                state_eef_quat = state_arm_cols[:, dof_per_arm + 3:dof_per_arm + 7]
                pos_delta = target_pos - state_eef_pos
                rot_delta = (Rotation.from_quat(target_quat) * Rotation.from_quat(state_eef_quat).inv()).as_rotvec()

            pos_width = min(pos_delta.shape[1], ACTION_EEF_POS_SLOT)
            if pos_width > 0:
                action_canonical[:, eef_offset:eef_offset + pos_width] = pos_delta[:, :pos_width]
                mask[eef_offset:eef_offset + pos_width] = True

            rot_offset = eef_offset + ACTION_EEF_POS_SLOT
            rot_width = min(rot_delta.shape[1], ACTION_EEF_ROT_SLOT)
            if rot_width > 0:
                action_canonical[:, rot_offset:rot_offset + rot_width] = rot_delta[:, :rot_width]
                mask[rot_offset:rot_offset + rot_width] = True

        # --- gripper slot: never depends on FK ---
        gripper_offset = eef_offset + ACTION_EEF_POS_SLOT + ACTION_EEF_ROT_SLOT + 1
        _pack_gripper(action_canonical, mask, gripper_offset, gripper_action_cols, config.gripper_type)

    return StageResult(episode=episode, stats=stats)
