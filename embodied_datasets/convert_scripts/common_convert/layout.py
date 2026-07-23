"""Column-order assembly shared by every convert_scripts/<dataset_id>.py:
packs raw joint/gripper columns into this project's canonical staging
layout [joint | eef_pos(3) + eef_quat(4) | gripper], computing eef_pos/
eef_quat via forward kinematics from the real joint angles rather than
trusting whatever (if any) eef reading the raw format separately provides
-- see design doc section 7. This is deliberate even for raw formats that
do expose their own eef reading (e.g. simulator ground truth): it keeps
every dataset's eef_pos/eef_quat columns derived the same, single way, so
process_scripts' Stage4 FK-consistency check later has a meaningful,
uniform signal to check against, and this module's own FK self-check
(self_check.py) has one less axis of per-dataset variation to reason
about.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from shared.fk_backend import FkChain

EEF_POS_WIDTH = 3
EEF_QUAT_WIDTH = 4


def assemble_state(
    joints: np.ndarray, gripper: np.ndarray, urdf_path: Optional[str], dof_per_arm: int
) -> np.ndarray:
    """joints: (T, dof_per_arm) radians. gripper: (T, gripper_dim), any
    width/unit -- passed through unchanged, this stage does not normalize
    gripper representation (that happens later, in process_scripts'
    unify_representation.py). Returns
    (T, dof_per_arm + EEF_POS_WIDTH + EEF_QUAT_WIDTH + gripper_dim).

    When `urdf_path` is falsy, eef_pos/eef_quat are zero-filled rather
    than computed -- matching process_scripts' mask semantics for "this
    dimension doesn't apply to this dataset" (see
    docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
    section 8).
    """
    num_frames = joints.shape[0]
    eef_pos = np.zeros((num_frames, EEF_POS_WIDTH))
    eef_quat = np.zeros((num_frames, EEF_QUAT_WIDTH))
    if urdf_path:
        chain = FkChain(urdf_path)
        for t in range(num_frames):
            position, quaternion = chain.forward(joints[t, :dof_per_arm])
            eef_pos[t] = position
            eef_quat[t] = quaternion
    return np.concatenate([joints, eef_pos, eef_quat, gripper], axis=1)
