"""Stage2: state-action trend alignment via per-dimension cross-correlation
lag estimation + directional agreement. See design doc section 7 row 2.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import correlate

from common.episode import Episode, StageResult
from common.schema import ProcessConfig


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    state = episode.state
    action = episode.action
    num_frames = state.shape[0]
    if num_frames < 2:
        return StageResult(episode=episode, skip_reason="insufficient_frames_for_trend_alignment")

    state_delta = np.diff(state, axis=0, prepend=state[:1])
    num_dims = min(state_delta.shape[1], action.shape[1])
    if num_dims == 0:
        return StageResult(episode=episode, skip_reason="no_common_state_action_dims")

    lags = []
    directional_agreements = []
    for dim in range(num_dims):
        a = action[:, dim]
        s = state_delta[:, dim]
        a_centered = a - a.mean()
        s_centered = s - s.mean()
        corr = correlate(s_centered, a_centered, mode="full")
        lag = int(np.argmax(corr) - (len(a_centered) - 1))
        lags.append(lag)
        shifted_a = np.roll(a, lag)
        directional_agreements.append(float(np.mean(np.sign(shifted_a) == np.sign(s))))

    lag = int(np.median(lags))
    directional_agreement = float(np.mean(directional_agreements))

    if abs(lag) > config.max_lag_frames or directional_agreement < config.da_threshold:
        return StageResult(
            episode=episode,
            skip_reason="trend_misaligned",
            stats={"lag": lag, "directional_agreement": directional_agreement},
        )

    if lag == 0:
        aligned_state, aligned_action = state, action
        aligned_frames = episode.frames
    elif lag > 0:
        aligned_state, aligned_action = state[lag:], action[:-lag]
        # Mirror state's trim (not action's -- state and frames are the two
        # arrays being shifted forward by `lag`; action is trimmed from the
        # opposite end instead). Matches the keep_mask-based frame rebuild
        # stage3/check3 do when they drop frames.
        aligned_frames = {view: frames[lag:] for view, frames in episode.frames.items()}
    else:
        aligned_state, aligned_action = state[:lag], action[-lag:]
        aligned_frames = {view: frames[:lag] for view, frames in episode.frames.items()}

    new_length = aligned_state.shape[0]
    new_episode = replace(
        episode,
        state=aligned_state,
        action=aligned_action,
        timestamps=episode.timestamps[:new_length],
        frames=aligned_frames,
    )
    return StageResult(episode=new_episode, stats={"lag": lag, "directional_agreement": directional_agreement})
