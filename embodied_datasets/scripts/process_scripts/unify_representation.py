"""Cross-embodiment canonical 80-dim state projection (design doc
section 8). Per-arm 35-dim block = [joint(<=7) | eef_pos(3)+eef_quat(4) |
gripper_or_hand_slot(<=21)]; dual-arm datasets concatenate arm1[0:35] +
arm2[35:70]; [70:73] holds mobile-base vx/vy/yaw when has_mobile_base;
[73:80] is reserved (left zero).

Assumes episode.state columns are ALREADY ordered per-arm as [joint |
eef_pos+eef_quat | gripper_or_hand_slot], concatenated arm-by-arm -- this
module only pads/truncates/repositions into the fixed 80-dim layout, it
does not reorder raw per-dataset columns. That reordering is
convert_scripts' responsibility.

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
  the upstream `DatasetConfig`'s responsibility, not something this
  module verifies or can verify.
"""
from __future__ import annotations

from typing import Set

import numpy as np

from shared.episode import Episode, StageResult
from common.schema import ProcessConfig

JOINT_SLOT = 7
EEF_SLOT = 7
GRIPPER_SLOT = 21
ARM_BLOCK_DIM = JOINT_SLOT + EEF_SLOT + GRIPPER_SLOT  # 35
CANONICAL_DIM = 80
MOBILE_BASE_SLOT = slice(2 * ARM_BLOCK_DIM, 2 * ARM_BLOCK_DIM + 3)

ROBOT_EMBODIMENT_CLASSES: Set[str] = {
    "single_arm",
    "dual_arm",
    "half_humanoid",
    "humanoid",
    "mobile_manipulator",
    "quadruped",
}
SIMPLE_GRIPPER_TYPES: Set[str] = {"parallel_jaw", "three_jaw", "cage_pinch", "suction"}


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
    if gripper_type in SIMPLE_GRIPPER_TYPES:
        width = min(gripper_cols.shape[1], 1)
    elif gripper_type == "dexterous_hand":
        width = min(gripper_cols.shape[1], GRIPPER_SLOT)
    else:
        width = 0
    if width > 0:
        canonical[:, gripper_offset:gripper_offset + width] = gripper_cols[:, :width]
        mask[gripper_offset:gripper_offset + width] = True


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
    # The mobile-base vx/vy/yaw columns (when present) are appended AFTER
    # all arm columns in episode.state -- that's what mobile_cols_start =
    # num_arms * cols_per_arm below assumes. So those trailing 3 columns
    # must be carved out of the total width BEFORE dividing the remainder
    # among arms; dividing the full width (arm cols + mobile-base cols) by
    # num_arms, as before, always leaves a remainder < num_arms <= 2,
    # which can never satisfy the ">= mobile_cols_start + 3" check below --
    # making the has_mobile_base branch permanently dead code (mobile-base
    # velocities silently never captured, mask always False for
    # MOBILE_BASE_SLOT).
    mobile_base_width = 3 if config.has_mobile_base else 0
    arm_cols_total = max(episode.state.shape[1] - mobile_base_width, 0)
    cols_per_arm = arm_cols_total // num_arms

    for arm_idx in range(num_arms):
        start = arm_idx * cols_per_arm
        arm_cols = episode.state[:, start:start + cols_per_arm]
        _pack_arm(canonical, mask, arm_idx * ARM_BLOCK_DIM, arm_cols, dof_per_arm, config.gripper_type)

    if config.has_mobile_base:
        mobile_cols_start = num_arms * cols_per_arm
        if episode.state.shape[1] - mobile_cols_start >= 3:
            canonical[:, MOBILE_BASE_SLOT] = episode.state[:, mobile_cols_start:mobile_cols_start + 3]
            mask[MOBILE_BASE_SLOT] = True

    return StageResult(
        episode=episode,
        stats={"canonical_state": canonical, "canonical_mask": mask, "canonical_dim": CANONICAL_DIM},
    )
