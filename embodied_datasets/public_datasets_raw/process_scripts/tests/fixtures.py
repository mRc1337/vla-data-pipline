"""Synthetic LeRobotDataset builders for tests that need real lerobot I/O
(as opposed to the stage/check unit tests, which construct Episode numpy
arrays directly and never touch lerobot).

Verified against the installed lerobot==0.4.4 API (lerobot.datasets.lerobot_dataset.
LeRobotDataset). Notable differences from a naive v2.1-era guess:
- `add_frame()` requires a "task" key in the frame dict (raises ValueError:
  "Missing features: {'task'}" otherwise).
- `LeRobotDataset.create(...)`/`.save_episode()` buffer episode metadata and only
  flush it to disk deterministically when `.finalize()` is called (relying on
  `__del__` for cleanup leaves the metadata parquet writer potentially unflushed
  and triggers a spurious "Exception ignored in __del__" at interpreter shutdown).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def make_synthetic_dataset(
    root: Path,
    repo_id: str,
    num_episodes: int,
    num_frames: int,
    state_dim: int = 4,
    action_dim: int = 4,
    fps: float = 10.0,
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
