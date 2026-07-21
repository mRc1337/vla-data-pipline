"""Stage4: joint-to-eef forward-kinematics consistency check. Gated by
fk_check_feasible = urdf_available AND action_space in
{joint_position, eef_pose} (computed by run_pipeline.py and written into
config before calling apply()). Systematic median TCP offset is corrected;
high-variance (non-systematic) offset is left alone and flagged for manual
review. See design doc section 7 row 4.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from common.episode import Episode, StageResult
from common.fk_backend import FkChain
from common.schema import ProcessConfig

EEF_POS_WIDTH = 3


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if not config.fk_check_feasible or not config.urdf_path:
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")

    joint_dim = config.dof_per_arm
    if not joint_dim or joint_dim <= 0:
        # dof_per_arm is required to know where the joint columns end and
        # the reported eef-position slice begins. Without it we cannot
        # locate either safely, so the check isn't feasible.
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")

    try:
        chain = FkChain(config.urdf_path)
    except ValueError:
        # ikpy's Chain.from_urdf_file raises ValueError when the URDF's
        # structure doesn't match its assumptions -- e.g. it hard-codes an
        # expectation that the root link is named "base_link"; a URDF
        # whose root link is named anything else (confirmed with
        # TheRobotStudio/SO-ARM100's so100.urdf, whose root link is
        # "base") makes ikpy's internal _find_next_link raise
        # ValueError("Error: link base_link given but not found in the
        # URDF"). That is a property of the dataset's URDF, not a bug in
        # this pipeline, so it belongs in the same "not feasible" bucket
        # as every other infeasibility check above/below. Deliberately
        # narrow (not a bare `except Exception`): FileNotFoundError (bad
        # path) and xml.etree.ElementTree.ParseError (malformed XML) are
        # different failure classes and are intentionally left to
        # propagate, as would a genuine programming error inside
        # FkChain.__init__ unrelated to URDF parsing.
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")

    # config.dof_per_arm is expected to equal the URDF's number of active
    # (non-fixed) joints -- run_pipeline.py is responsible for keeping the
    # two in sync when it builds this config. If they disagree,
    # FkChain.forward()'s zip()-based angle assignment (Task 5) silently
    # truncates extra joint values or zero-pads missing ones instead of
    # raising, which produces a wrong-but-not-crashing FK position. That
    # wrong position can differ from the (actually correct) reported eef
    # position by a constant amount across frames, indistinguishable from
    # a genuine systematic sensor offset -- which this function would then
    # "correct" by overwriting good eef data with the wrong FK-derived
    # value. Bail out instead of trusting a mismatched pairing.
    if joint_dim != len(chain._active_link_indices):
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")

    num_frames = episode.state.shape[0]
    if num_frames == 0 or episode.state.shape[1] < joint_dim + EEF_POS_WIDTH:
        # No frames to check, or the episode doesn't have enough columns to
        # hold both the joint_dim joint values and the EEF_POS_WIDTH
        # reported-position slice this module's contract assumes.
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")

    reported_positions = episode.state[:, joint_dim:joint_dim + EEF_POS_WIDTH]
    fk_positions = np.zeros((num_frames, EEF_POS_WIDTH))
    for t in range(num_frames):
        position, _quat = chain.forward(episode.state[t, :joint_dim])
        fk_positions[t] = position

    offsets = reported_positions - fk_positions
    median_offset = np.median(offsets, axis=0)
    offset_magnitude = float(np.linalg.norm(median_offset))
    residual_variance = float(np.var(offsets - median_offset, axis=0).sum())

    stats = {
        "median_offset": median_offset.tolist(),
        "offset_magnitude": offset_magnitude,
        "residual_variance": residual_variance,
    }

    if offset_magnitude > config.tcp_offset_tolerance:
        corrected_state = episode.state.copy()
        corrected_state[:, joint_dim:joint_dim + EEF_POS_WIDTH] = reported_positions - median_offset
        new_episode = replace(episode, state=corrected_state)
        stats["corrected"] = True
        return StageResult(episode=new_episode, stats=stats)

    if residual_variance > config.tcp_offset_tolerance ** 2:
        stats["corrected"] = False
        stats["flagged_for_manual_review"] = True
        return StageResult(episode=episode, stats=stats)

    stats["corrected"] = False
    return StageResult(episode=episode, stats=stats)
