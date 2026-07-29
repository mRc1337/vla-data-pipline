"""Orchestrates Stage1-5 + Check1-3 + unify_representation for one lerobot
dataset and writes the cleaned/aligned output. See
docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
section 10.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np

from episode import Episode  # noqa: E402
from common.io import load_process_config  # noqa: E402
from lerobot_io import load_lerobot_episodes, write_lerobot_episodes  # noqa: E402

import stage1_sudden_change  # noqa: E402
import stage2_trend_alignment  # noqa: E402
import stage3_extreme_value  # noqa: E402
import stage4_fk_consistency  # noqa: E402
import stage5_orientation_alignment  # noqa: E402
import check1_instruction_consistency  # noqa: E402
import check2_video_state_consistency  # noqa: E402
import check3_video_quality  # noqa: E402
import unify_representation  # noqa: E402


def run_dataset(input_path: Path, output_path: Path, process_config_path: Path) -> dict:
    config = load_process_config(process_config_path)
    fps = config.fps or 1.0

    episodes = load_lerobot_episodes(input_path)
    log: List[tuple] = []

    survivors = []
    for episode in episodes:
        result = stage1_sudden_change.apply(episode, config)
        log.append(("stage1_sudden_change", episode.episode_index, result.skip_reason, result.rejected))
        if result.rejected:
            continue
        episode = result.episode

        result = stage2_trend_alignment.apply(episode, config)
        log.append(("stage2_trend_alignment", episode.episode_index, result.skip_reason, result.rejected))
        if result.skip_reason:
            continue
        survivors.append(result.episode)

    config.extreme_value_bounds = stage3_extreme_value.compute_bounds(survivors, config) if survivors else None

    final_episodes: List[Episode] = []
    # unify_representation's canonical_mask is provably dataset-constant (it
    # depends only on config.dof_per_arm/num_arms/gripper_type/
    # has_mobile_base, all fixed for the whole run, not per-episode data) --
    # so it's captured once from whichever episode's result first produces
    # it (non-skipped) rather than threaded through every Episode object.
    canonical_mask: Optional[np.ndarray] = None
    for episode in survivors:
        result = stage3_extreme_value.apply(episode, config)
        log.append(("stage3_extreme_value", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        if episode.state.shape[0] == 0:
            log.append(("run_pipeline", episode.episode_index, "all_frames_dropped", True))
            continue

        result = stage4_fk_consistency.apply(episode, config)
        log.append(("stage4_fk_consistency", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        result = stage5_orientation_alignment.apply(episode, config)
        log.append(("stage5_orientation_alignment", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        result = check1_instruction_consistency.apply(episode, config)
        log.append(("check1_instruction_consistency", episode.episode_index, result.skip_reason, result.rejected))

        result = check2_video_state_consistency.apply(episode, config)
        log.append(("check2_video_state_consistency", episode.episode_index, result.skip_reason, result.rejected))

        result = check3_video_quality.apply(episode, config)
        log.append(("check3_video_quality", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        if episode.state.shape[0] == 0:
            log.append(("run_pipeline", episode.episode_index, "all_frames_dropped", True))
            continue

        result = unify_representation.apply(episode, config)
        log.append(("unify_representation", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        # For robot-collected embodiment classes (result.skip_reason is None),
        # unify_representation computes a cross-embodiment canonical 128-dim
        # projection of episode.state but does NOT itself replace
        # episode.state (see unify_representation.py's apply() docstring/
        # tests -- it returns the original episode object with the
        # canonical vector only in result.stats). Replacing it here, rather
        # than in unify_representation.apply, keeps that module's contract
        # (compute, don't mutate) and makes this the single place that
        # decides what actually gets written to the output dataset.
        # episode.action is explicitly left untouched -- Task 15 only
        # canonicalizes state, by design.
        if result.skip_reason is None:
            episode = replace(episode, state=result.stats["canonical_state"])
            if canonical_mask is None:
                canonical_mask = result.stats["canonical_mask"]

        final_episodes.append(episode)

    if final_episodes:
        write_lerobot_episodes(final_episodes, output_path, fps=fps, robot_type=config.id, canonical_mask=canonical_mask)

    total_frames = sum(ep.state.shape[0] for ep in final_episodes)
    return {
        "input_episodes": len(episodes),
        "output_episodes": len(final_episodes),
        "output_frames": total_frames,
        "fps": fps,
        "log": log,
    }


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the process_scripts cleaning/alignment pipeline on a lerobot dataset.")
    parser.add_argument("--input", required=True, help="Path to the input lerobot dataset.")
    parser.add_argument("--output", required=True, help="Path to write the cleaned/aligned lerobot dataset to.")
    parser.add_argument("--config", required=True, help="Path to the ProcessConfig yaml (cleaning/alignment parameters).")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    process_config_path = Path(args.config)

    if not process_config_path.exists():
        print(f"error: config file not found: {process_config_path}", file=sys.stderr)
        return 1

    stats = run_dataset(input_path, output_path, process_config_path)

    if stats["output_episodes"] == 0:
        # run_dataset() never called write_lerobot_episodes (it early-returns
        # on an empty episode list), so output_path was never created on
        # disk -- there is nothing new persisted.
        print(
            f"warning: produced 0 output episodes ({stats['input_episodes']} input) -- "
            f"nothing written to {output_path}",
            file=sys.stderr,
        )
        print(f"processed: {stats['input_episodes']} -> 0 episodes (FAILED)")
        return 1

    print(f"processed: {stats['input_episodes']} -> {stats['output_episodes']} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
