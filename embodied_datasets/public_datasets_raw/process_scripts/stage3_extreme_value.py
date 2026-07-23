"""Stage3: two-pass extreme-value filtering. Pass 1 (compute_bounds) computes
q(quantile_low)/q(quantile_high) per dimension across ALL episodes of a
dataset (gripper dims exempted, per design doc section 7 row 3, since they
are bimodal open/closed rather than continuous). Pass 2 (apply) drops
per-episode frames outside those bounds. run_pipeline.py is responsible for
calling compute_bounds once and writing the result into
config.extreme_value_bounds before running apply() per episode -- apply()
itself takes no dataset-wide state so it stays a pure function of
(episode, config).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np

from shared.episode import Episode, StageResult
from common.schema import ProcessConfig


def _quantile_bounds(values: np.ndarray, quantile_low: float, quantile_high: float, exempt_dims: List[int]) -> Dict[int, List[float]]:
    bounds: Dict[int, List[float]] = {}
    for dim in range(values.shape[1]):
        if dim in exempt_dims:
            continue
        low = float(np.quantile(values[:, dim], quantile_low))
        high = float(np.quantile(values[:, dim], quantile_high))
        bounds[dim] = [low, high]
    return bounds


def compute_bounds(episodes: List[Episode], config: ProcessConfig) -> Dict[str, Dict[int, List[float]]]:
    all_state = np.concatenate([ep.state for ep in episodes], axis=0)
    all_action = np.concatenate([ep.action for ep in episodes], axis=0)
    return {
        "state": _quantile_bounds(all_state, config.quantile_low, config.quantile_high, config.gripper_dims_state),
        "action": _quantile_bounds(all_action, config.quantile_low, config.quantile_high, config.gripper_dims_action),
    }


def _out_of_bounds_mask(values: np.ndarray, bounds: Dict[int, List[float]]) -> np.ndarray:
    mask = np.zeros(values.shape[0], dtype=bool)
    for dim_key, (low, high) in bounds.items():
        dim = int(dim_key)
        if dim >= values.shape[1]:
            # Bound references a column that doesn't exist on this episode's
            # array (e.g. a malformed/corrupted episode with a different
            # column count than the rest of its dataset). Skip rather than
            # raise IndexError -- this dim simply can't be checked here.
            continue
        mask |= (values[:, dim] < low) | (values[:, dim] > high)
    return mask


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if not config.extreme_value_bounds:
        return StageResult(episode=episode, skip_reason="extreme_value_bounds_not_computed")

    state_mask = _out_of_bounds_mask(episode.state, config.extreme_value_bounds.get("state", {}))
    action_mask = _out_of_bounds_mask(episode.action, config.extreme_value_bounds.get("action", {}))
    dropped_mask = state_mask | action_mask
    dropped_idx = np.where(dropped_mask)[0].tolist()

    if not dropped_idx:
        return StageResult(episode=episode)

    keep_mask = ~dropped_mask
    new_episode = replace(
        episode,
        state=episode.state[keep_mask],
        action=episode.action[keep_mask],
        timestamps=episode.timestamps[keep_mask],
        frames={view: frames[keep_mask] for view, frames in episode.frames.items()},
    )
    return StageResult(episode=new_episode, dropped_frame_indices=dropped_idx)
