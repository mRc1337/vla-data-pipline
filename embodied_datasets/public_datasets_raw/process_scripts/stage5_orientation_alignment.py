"""Stage5: applies the configured base-to-world 4x4 transform to every
frame's eef position + orientation, unifying the world-frame convention
across datasets. Always runs; skips (with reason recorded) only when no
transform is configured. See design doc section 7 row 5.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from common.episode import Episode, StageResult
from common.schema import ProcessConfig

EEF_POS_WIDTH = 3
EEF_QUAT_WIDTH = 4
TRANSFORM_ELEMENT_COUNT = 16


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if not config.base_to_world_transform:
        return StageResult(episode=episode, skip_reason="no_base_to_world_transform_configured")

    if len(config.base_to_world_transform) != TRANSFORM_ELEMENT_COUNT:
        # Can't reshape into a 4x4 matrix -- a hand-authored config value
        # with the wrong element count. Nothing safe to do with it.
        return StageResult(episode=episode, skip_reason="invalid_base_to_world_transform")

    transform = np.array(config.base_to_world_transform, dtype=np.float64).reshape(4, 4)

    if not np.all(np.isfinite(transform)):
        # pydantic's default float validation allows NaN/Inf, so a
        # non-finite transform is a legally-reachable config value.
        # Rotation.from_matrix's SVD-based orthonormalization raises
        # LinAlgError on NaN input but can hang indefinitely on Inf input
        # -- worse than a crash. Reject before any linear algebra runs.
        return StageResult(episode=episode, skip_reason="invalid_base_to_world_transform")

    rotation_matrix = transform[:3, :3]
    translation = transform[:3, 3]

    try:
        world_rotation = Rotation.from_matrix(rotation_matrix)
    except ValueError:
        # rotation_matrix isn't a valid rotation (e.g. a reflection or a
        # degenerate/non-orthonormal block with non-positive determinant).
        # A hand-authored or miscalibrated transform can plausibly have
        # this, and Rotation.from_matrix raises rather than silently
        # coercing it, so bail out instead of crashing the whole pipeline
        # run over one dataset's bad config.
        return StageResult(episode=episode, skip_reason="invalid_base_to_world_transform")

    # dof_per_arm may legitimately be None/0 (state has no joint columns,
    # eef position starts at column 0). Clamp negative values (not a
    # sensible column offset) down to 0, and clamp values wider than the
    # state itself so the slice below stays a valid, in-range placeholder
    # rather than wrapping via numpy's negative-index semantics.
    joint_dim = max(0, min(config.dof_per_arm or 0, episode.state.shape[1]))

    if episode.state.shape[1] < joint_dim + EEF_POS_WIDTH:
        # Not enough columns to hold even the eef position slice this
        # module's contract assumes -- nothing safe to transform.
        return StageResult(episode=episode, skip_reason="state_too_narrow_for_eef_slice")

    pos_slice = slice(joint_dim, joint_dim + EEF_POS_WIDTH)
    quat_slice = slice(joint_dim + EEF_POS_WIDTH, joint_dim + EEF_POS_WIDTH + EEF_QUAT_WIDTH)

    new_state = episode.state.copy()
    positions = episode.state[:, pos_slice]
    new_state[:, pos_slice] = positions @ rotation_matrix.T + translation

    quats = episode.state[:, quat_slice]
    # A dof_per_arm/state-layout mismatch (or genuinely zero-padded
    # orientation data upstream) can hand us an all-zero "quaternion" row.
    # Rotation.from_quat raises ValueError on any zero-norm row rather than
    # normalizing it, so guard for that in addition to the width check --
    # otherwise a single degenerate row crashes the whole episode. A row
    # with a literal Inf, or with finite-but-huge components whose L2 norm
    # itself overflows to inf, passes a bare `norm > 1e-8` check (inf >
    # 1e-8 is True) and Rotation.from_quat silently produces NaN -- the
    # crash then surfaces later, uncaught, at the `.as_quat()` composition
    # below. So require the *computed norm* to be finite too -- checking
    # isfinite(quats) alone isn't enough, since finite-but-huge components
    # (e.g. 1e200) are individually finite yet their squared sum overflows
    # to inf during the norm computation itself. Overflow during that
    # computation is expected/handled here, not a bug, so suppress the
    # warning it would otherwise raise.
    with np.errstate(over="ignore"):
        quat_norms = np.linalg.norm(quats, axis=1)
    quat_norms_ok = (
        quats.shape[1] == EEF_QUAT_WIDTH
        and bool(np.all(np.isfinite(quat_norms)))
        and bool(np.all(quat_norms > 1e-8))
    )
    if quat_norms_ok:
        frame_rotation = Rotation.from_quat(quats)
        new_state[:, quat_slice] = (world_rotation * frame_rotation).as_quat()

    new_episode = replace(episode, state=new_state)
    return StageResult(episode=new_episode, stats={"world_frame_convention": config.world_frame_convention})
