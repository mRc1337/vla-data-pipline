"""Stage1: sudden-change detection via Savitzky-Golay smoothing +
residual/acceleration/jerk thresholds. Flagged frames are linearly
interpolated; if the flagged fraction exceeds episode_reject_threshold the
whole episode is rejected. See design doc section 7 row 1.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import savgol_filter

from episode import Episode, StageResult
from common.schema import ProcessConfig


def _threshold_for_dim(config: ProcessConfig, kind: str, dim: int, default: float) -> float:
    overrides = config.per_dim_thresholds.get("state", {})
    dim_overrides = overrides.get(str(dim))
    if dim_overrides and kind in dim_overrides:
        return dim_overrides[kind]
    return default


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    state = episode.state
    num_frames, num_dims = state.shape

    if num_frames < 2:
        return StageResult(episode=episode, skip_reason="episode_too_short_for_savgol")

    max_window = num_frames if num_frames % 2 == 1 else num_frames - 1
    window = min(config.savgol_window, max_window)
    if window % 2 == 0:
        window -= 1
    if window <= config.savgol_polyorder or window < 1:
        return StageResult(episode=episode, skip_reason="episode_too_short_for_savgol")

    smoothed = savgol_filter(state, window_length=window, polyorder=config.savgol_polyorder, axis=0)
    residual = state - smoothed
    accel = np.gradient(state, axis=0)
    jerk = np.gradient(accel, axis=0)

    flagged = np.zeros(num_frames, dtype=bool)
    for dim in range(num_dims):
        residual_thr = _threshold_for_dim(config, "residual", dim, config.residual_threshold)
        accel_thr = _threshold_for_dim(config, "accel", dim, config.accel_threshold)
        jerk_thr = _threshold_for_dim(config, "jerk", dim, config.jerk_threshold)
        flagged |= np.abs(residual[:, dim]) > residual_thr
        flagged |= np.abs(accel[:, dim]) > accel_thr
        flagged |= np.abs(jerk[:, dim]) > jerk_thr

    flagged_fraction = float(flagged.sum()) / num_frames
    if flagged_fraction > config.episode_reject_threshold:
        return StageResult(episode=episode, rejected=True, stats={"flagged_fraction": flagged_fraction})

    flagged_idx = np.where(flagged)[0]
    valid_idx = np.where(~flagged)[0]
    fixed_state = state.copy()
    stats = {"flagged_fraction": flagged_fraction, "num_flagged": int(flagged.sum())}
    if len(valid_idx) >= 2 and len(flagged_idx) > 0:
        for dim in range(num_dims):
            fixed_state[flagged_idx, dim] = np.interp(flagged_idx, valid_idx, state[valid_idx, dim])
    elif len(flagged_idx) > 0:
        stats["interpolation_skipped"] = True

    new_episode = replace(episode, state=fixed_state)
    return StageResult(
        episode=new_episode,
        dropped_frame_indices=flagged_idx.tolist(),
        stats=stats,
    )
