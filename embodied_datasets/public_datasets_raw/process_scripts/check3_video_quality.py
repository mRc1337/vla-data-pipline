"""Check3: video quality filtering (black / blurry / still-run frames) via
plain OpenCV -- no external service dependency, always runs. See design
doc section 7 row 8.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from shared.episode import Episode, StageResult
from common.schema import ProcessConfig


def _consecutive_run_flags(flags: np.ndarray, min_run: int) -> np.ndarray:
    result = np.zeros_like(flags)
    run_start = None
    for i, flagged in enumerate(flags):
        if flagged:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                result[run_start:i] = True
            run_start = None
    if run_start is not None and len(flags) - run_start >= min_run:
        result[run_start:len(flags)] = True
    return result


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if not episode.frames:
        return StageResult(episode=episode, skip_reason="no_video_frames")

    num_frames = episode.state.shape[0]
    flagged = np.zeros(num_frames, dtype=bool)
    stats = {}

    for view, frames in episode.frames.items():
        gray = frames.astype(np.float64).mean(axis=-1)  # (T, H, W)
        brightness = gray.mean(axis=(1, 2))
        black = brightness < config.black_threshold

        # `gray` is already float64 -- feed it to cv2.Laplacian directly.
        # Casting to uint8 here would truncate frames whose native range is
        # e.g. normalized [0, 1] floats down to all-zero, making every frame
        # spuriously read as maximally blurry (and maximally black). Since
        # Episode.frames carries no documented dtype/range contract,
        # cv2.Laplacian's own support for float64 input sidesteps the risk
        # entirely rather than guessing at a rescale.
        blur_scores = np.array([cv2.Laplacian(f, cv2.CV_64F).var() for f in gray])
        blurry = blur_scores < config.blur_threshold

        frame_diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
        frame_diffs = np.concatenate([[np.inf], frame_diffs])
        still = frame_diffs < config.still_threshold
        # `still[i]` means frame i is ~identical to frame i-1, so a run of
        # consecutive True values at [run_start, run_end] actually spans
        # frames [run_start-1, run_end] -- one frame more than the run's own
        # length, since the run's anchor frame (run_start-1) is identical to
        # everything after it but was never itself compared against an
        # earlier held frame. Compare against min_run-1 so "N consecutive
        # identical frames" (the intuitive config meaning) lines up with the
        # diff-run length, then extend the flags left by one frame so the
        # anchor is dropped along with the rest of the still run.
        still_run_threshold = max(config.still_min_consecutive_frames - 1, 0)
        still_diff_run = _consecutive_run_flags(still, still_run_threshold)
        still_run = still_diff_run.copy()
        still_run[:-1] |= still_diff_run[1:]

        view_flagged = black | blurry | still_run
        flagged |= view_flagged
        stats[view] = {
            "num_black": int(black.sum()),
            "num_blurry": int(blurry.sum()),
            "num_still": int(still_run.sum()),
        }

    dropped_idx = np.where(flagged)[0].tolist()
    if not dropped_idx:
        return StageResult(episode=episode, stats=stats)

    keep_mask = ~flagged
    new_episode = replace(
        episode,
        state=episode.state[keep_mask],
        action=episode.action[keep_mask],
        timestamps=episode.timestamps[keep_mask],
        frames={view: frames[keep_mask] for view, frames in episode.frames.items()},
    )
    return StageResult(episode=new_episode, dropped_frame_indices=dropped_idx, stats=stats)
