"""Reference convert_scripts implementation for a raw dataset that is
already in LeRobot format (raw_format: LeRobot) -- see
docs/superpowers/specs/2026-07-21-convert-scripts-verify-scripts-design.md
section 7. Structural template for other already-LeRobot-format datasets:
the conversion work here is relayout (raw column order -> this project's
canonical [joint | eef_pos + eef_quat | gripper] order), not format
translation.

Documented assumption about this dataset's raw `observation.state` layout
([joint(dof_per_arm) | gripper(rest)]) -- no raw data has been downloaded
to confirm this against the actual dataset; verify the real column layout
against `meta/info.json`/the dataset's README before reusing this
structure for a real onboarding run (see the implementation plan's
Self-Review).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from common.paths import urdf_assets_dir
from common.schema import DatasetConfig
from common_convert.layout import assemble_state
from common_convert.report import ConversionReport
from shared.episode import Episode
from shared.lerobot_io import load_lerobot_episodes, write_lerobot_episodes


def _resolve_urdf_path(raw_path: Path, config: DatasetConfig) -> Optional[str]:
    """See mimicgen.py's identical helper for why `data_root` is derived
    from `raw_path` rather than added as a new convert() parameter."""
    if not config.urdf_available or not config.robot_platform:
        return None
    robot_platform = getattr(config.robot_platform, "value", config.robot_platform)
    data_root = raw_path.parents[1]
    candidate = urdf_assets_dir(data_root, robot_platform) / f"{robot_platform}.urdf"
    return str(candidate) if candidate.exists() else None


def convert(raw_path: Path, output_path: Path, config: DatasetConfig) -> ConversionReport:
    dof_per_arm = config.dof_per_arm or 0
    urdf_path = _resolve_urdf_path(raw_path, config)
    warnings: List[str] = []

    raw_episodes = load_lerobot_episodes(raw_path)
    out_episodes: List[Episode] = []
    for raw_episode in raw_episodes:
        if raw_episode.state.shape[1] < dof_per_arm:
            warnings.append(
                f"episode {raw_episode.episode_index}: raw state width {raw_episode.state.shape[1]} "
                f"< declared dof_per_arm {dof_per_arm}, skipping"
            )
            continue
        joints = raw_episode.state[:, :dof_per_arm]
        gripper = raw_episode.state[:, dof_per_arm:]
        state = assemble_state(joints, gripper, urdf_path, dof_per_arm)
        out_episodes.append(
            Episode(
                episode_index=raw_episode.episode_index,
                timestamps=raw_episode.timestamps,
                state=state.astype(np.float32),
                action=raw_episode.action,
                frames=raw_episode.frames,
                language_instruction=raw_episode.language_instruction,
            )
        )

    write_lerobot_episodes(out_episodes, output_path, fps=config.fps or 1.0, robot_type=config.id)

    total_frames = sum(episode.state.shape[0] for episode in out_episodes)
    return ConversionReport(
        num_episodes=len(out_episodes), num_frames=total_frames, warnings=warnings, urdf_path=urdf_path
    )
