"""Reference convert_scripts implementation for an HDF5 (robomimic-format)
raw dataset -- see
docs/superpowers/specs/2026-07-21-convert-scripts-verify-scripts-design.md
section 7. Structural template: copy this file's shape for other HDF5
datasets, swapping the actual HDF5 group/key names for the target
dataset's real schema (this one matches robomimic's documented layout,
per configs/mimicgen.yaml's raw_format field_sources note).

Expected raw_path layout (robomimic HDF5 -- this reference reads the
first *.hdf5 file found directly under raw_path):

    raw_path/*.hdf5
      data/
        demo_0/
          obs/
            robot0_joint_pos    (T, dof_per_arm)  -- radians (robosuite/MuJoCo convention)
            robot0_gripper_qpos (T, gripper_dim)
          actions               (T, action_dim)
        demo_1/
          ...

No real raw data has been downloaded to verify this against -- see the
implementation plan's Self-Review for what to check before relying on
this against an actual download.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from common.paths import urdf_assets_dir
from common.schema import DatasetConfig
from common_convert.layout import assemble_state
from common_convert.report import ConversionReport
from shared.episode import Episode
from shared.lerobot_io import write_lerobot_episodes


def _resolve_urdf_path(raw_path: Path, config: DatasetConfig) -> Optional[str]:
    """`raw_path` is always `<data_root>/raw/<dataset_id>` (see
    run_convert.py) -- data_root is derived from it (`raw_path.parents[1]`)
    rather than added as a new convert() parameter, since the per-dataset
    contract (design doc section 7) fixes convert()'s signature to exactly
    (raw_path, output_path, config). URDF file naming convention
    (`<robot_platform>.urdf` inside `urdf_assets_dir(data_root,
    robot_platform)`) is this plan's own choice -- not mandated by the
    design doc, which doesn't specify a filename convention.
    """
    if not config.urdf_available or not config.robot_platform:
        return None
    robot_platform = getattr(config.robot_platform, "value", config.robot_platform)
    data_root = raw_path.parents[1]
    candidate = urdf_assets_dir(data_root, robot_platform) / f"{robot_platform}.urdf"
    return str(candidate) if candidate.exists() else None


def convert(raw_path: Path, output_path: Path, config: DatasetConfig) -> ConversionReport:
    hdf5_files = sorted(raw_path.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"no *.hdf5 file found under {raw_path}")

    dof_per_arm = config.dof_per_arm or 0
    urdf_path = _resolve_urdf_path(raw_path, config)
    warnings: List[str] = []
    episodes: List[Episode] = []

    with h5py.File(hdf5_files[0], "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for episode_index, demo_key in enumerate(demo_keys):
            demo = f["data"][demo_key]
            joints = np.asarray(demo["obs"]["robot0_joint_pos"])
            gripper = np.asarray(demo["obs"]["robot0_gripper_qpos"])
            actions = np.asarray(demo["actions"])
            num_frames = joints.shape[0]

            if joints.shape[1] != dof_per_arm:
                warnings.append(
                    f"{demo_key}: robot0_joint_pos width {joints.shape[1]} != declared dof_per_arm {dof_per_arm}"
                )

            state = assemble_state(joints, gripper, urdf_path, dof_per_arm)
            episodes.append(
                Episode(
                    episode_index=episode_index,
                    timestamps=np.arange(num_frames, dtype=np.float64) / (config.fps or 1.0),
                    state=state.astype(np.float32),
                    action=actions.astype(np.float32),
                )
            )

    write_lerobot_episodes(episodes, output_path, fps=config.fps or 1.0, robot_type=config.id)

    total_frames = sum(episode.state.shape[0] for episode in episodes)
    return ConversionReport(
        num_episodes=len(episodes), num_frames=total_frames, warnings=warnings, urdf_path=urdf_path
    )
