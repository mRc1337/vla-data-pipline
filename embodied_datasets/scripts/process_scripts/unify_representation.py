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
