"""Instrumented mirror of process_scripts/run_pipeline.py::run_dataset() --
same stage1-5/check1-3/unify_representation call sequence, imported
directly from process_scripts, but records every stage's full StageResult
(skip_reason/rejected/dropped_frame_indices/stats) into a JSON sidecar
(_inspect_metadata.json) instead of only run_pipeline.py's compact 4-tuple
log. Does not import or modify run_pipeline.py itself.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np

_PROCESS_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "process_scripts"
if str(_PROCESS_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESS_SCRIPTS_DIR))

from episode import Episode, StageResult  # noqa: E402
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

from metadata_io import save_metadata, stage_result_to_record


def run_dataset_instrumented(input_path: Path, output_path: Path, process_config_path: Path) -> dict:
    """Runs the same pipeline run_pipeline.py::run_dataset() runs, but
    additionally writes `_inspect_metadata.json` into output_path recording
    every stage's full StageResult for every episode. Returns the metadata
    dict (identical to what gets written to disk)."""
    config = load_process_config(process_config_path)
    fps = config.fps or 1.0

    episodes = load_lerobot_episodes(input_path)
    episode_records = {
        ep.episode_index: {
            "episode_index": ep.episode_index,
            "input_frame_count": ep.state.shape[0],
            "output_frame_count": 0,
            "survived": False,
            "stages": [],
        }
        for ep in episodes
    }

    def record(stage_name: str, episode_index: int, result: StageResult) -> None:
        episode_records[episode_index]["stages"].append(stage_result_to_record(stage_name, result))

    survivors = []
    for episode in episodes:
        result = stage1_sudden_change.apply(episode, config)
        record("stage1_sudden_change", episode.episode_index, result)
        if result.rejected:
            continue
        episode = result.episode

        result = stage2_trend_alignment.apply(episode, config)
        record("stage2_trend_alignment", episode.episode_index, result)
        if result.rejected:
            continue
        survivors.append(result.episode)

    config.extreme_value_bounds = stage3_extreme_value.compute_bounds(survivors, config) if survivors else None

    final_episodes: List[Episode] = []
    canonical_mask: Optional[np.ndarray] = None
    action_canonical_mask: Optional[np.ndarray] = None
    for episode in survivors:
        result = stage3_extreme_value.apply(episode, config)
        record("stage3_extreme_value", episode.episode_index, result)
        episode = result.episode

        if episode.state.shape[0] == 0:
            record("run_pipeline", episode.episode_index, StageResult(episode=episode, rejected=True, skip_reason="all_frames_dropped"))
            continue

        result = stage4_fk_consistency.apply(episode, config)
        record("stage4_fk_consistency", episode.episode_index, result)
        episode = result.episode

        result = stage5_orientation_alignment.apply(episode, config)
        record("stage5_orientation_alignment", episode.episode_index, result)
        episode = result.episode

        result = check1_instruction_consistency.apply(episode, config)
        record("check1_instruction_consistency", episode.episode_index, result)

        result = check2_video_state_consistency.apply(episode, config)
        record("check2_video_state_consistency", episode.episode_index, result)

        result = check3_video_quality.apply(episode, config)
        record("check3_video_quality", episode.episode_index, result)
        episode = result.episode

        if episode.state.shape[0] == 0:
            record("run_pipeline", episode.episode_index, StageResult(episode=episode, rejected=True, skip_reason="all_frames_dropped"))
            continue

        result = unify_representation.apply(episode, config)
        record("unify_representation", episode.episode_index, result)
        episode = result.episode

        action_result = unify_representation.apply_action(episode, config)
        record("unify_representation_action", episode.episode_index, action_result)

        if result.skip_reason is None:
            episode = replace(episode, state=result.stats["canonical_state"])
            if canonical_mask is None:
                canonical_mask = result.stats["canonical_mask"]

        if action_result.skip_reason is None:
            episode = replace(episode, action=action_result.stats["action_canonical"])
            if action_canonical_mask is None:
                action_canonical_mask = action_result.stats["action_canonical_mask"]

        episode_records[episode.episode_index]["output_frame_count"] = episode.state.shape[0]
        episode_records[episode.episode_index]["survived"] = True
        final_episodes.append(episode)

    if final_episodes:
        write_lerobot_episodes(
            final_episodes, output_path, fps=fps, robot_type=config.id,
            canonical_mask=canonical_mask, action_canonical_mask=action_canonical_mask,
        )

    metadata = {"episodes": list(episode_records.values())}
    save_metadata(metadata, output_path)
    return metadata
