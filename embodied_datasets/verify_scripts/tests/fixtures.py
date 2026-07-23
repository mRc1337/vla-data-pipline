"""Synthetic LeRobotDataset builder for verify_scripts' own tests (a "raw
dataset that's already in LeRobot format" fixture for check_lerobot's
tests). See shared/lerobot_io.py's docstring for the underlying
lerobot==0.4.4 API notes this mirrors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def make_synthetic_dataset(
    root: Path, repo_id: str, num_episodes: int, num_frames: int, state_dim: int = 4, action_dim: int = 4, fps: float = 10.0
) -> Path:
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root, features=features, use_videos=False)
    rng = np.random.RandomState(0)
    for _episode in range(num_episodes):
        for _frame in range(num_frames):
            dataset.add_frame(
                {
                    "observation.state": rng.uniform(-1, 1, size=state_dim).astype(np.float32),
                    "action": rng.uniform(-1, 1, size=action_dim).astype(np.float32),
                    "task": "synthetic",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root
