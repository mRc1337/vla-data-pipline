"""Load/save process_scripts/configs/<id>.yaml as ProcessConfig, plus
thin wrappers over the official lerobot dataset read/write API.

The lerobot wrappers below were verified against the installed lerobot==0.4.4
API (lerobot.datasets.lerobot_dataset.LeRobotDataset), which differs from a
naive v2.1-era guess in a few load-bearing ways:

- `add_frame()` requires a "task" key in the frame dict in addition to the
  declared features (raises ValueError otherwise: "Missing features: {'task'}").
- There is no `dataset.episode_data_index` attribute in this version. Episode
  boundaries live in `dataset.meta.episodes`, a `datasets.Dataset` where row
  `i` has `dataset_from_index`/`dataset_to_index` giving the half-open range
  of global frame indices for episode `i` (indices align 1:1 with
  `dataset[j]`).
- `LeRobotDataset.create(...)` + `.add_frame()` + `.save_episode()` buffer
  episode metadata in memory; it is only flushed to a parquet file
  deterministically by calling `.finalize()`. Relying on `__del__`/GC instead
  leaves the metadata parquet writer in an inconsistent state and can emit a
  spurious "Exception ignored in __del__" at interpreter shutdown.
- `fps` is written to metadata as-is (float is accepted at runtime despite
  the type hint `fps: int`); frame timestamps are derived from `frame_index /
  fps`, not from any timestamp we pass in — round-tripping does not preserve
  the exact original `Episode.timestamps` values, only a regular fps grid.
- A second, differently-shaped feature (e.g. a `(80,)` bool mask alongside
  a `(state_dim,)` float32 state) needs no special handling: `features` is
  just a dict, each entry independently declares its own `dtype`/`shape`,
  and `add_frame()` takes one dict with a key per declared feature (plus
  "task") -- there is no per-feature file or separate dataset object.
  `dtype: "bool"` round-trips correctly as a `torch.bool` tensor on read;
  the only side effect observed is a cosmetic `RuntimeWarning` from
  lerobot's own stats computation ("Converting input from bool to
  numpy.uint8 for compatibility") when it histograms the bool column --
  harmless, not something this wrapper can or should suppress.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .episode import Episode
from .schema import ProcessConfig


def load_process_config(path: Path) -> ProcessConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProcessConfig(**raw)


def save_process_config(config: ProcessConfig, path: Path) -> None:
    data = config.model_dump(mode="json", exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def load_lerobot_episodes(dataset_path: Path) -> List[Episode]:
    """Read every episode out of a lerobot dataset stored at `dataset_path`.

    Populates `Episode.language_instruction` from each frame's `"task"` key
    (verified against the installed lerobot==0.4.4 API: `dataset[i]["task"]`
    is a plain `str`, identical across every frame of one episode -- this
    project's own `write_lerobot_episodes` and `tests/fixtures.py`'s
    `make_synthetic_dataset` both write exactly one task string per frame).
    That "identical across every frame" assumption is verified, not just
    trusted: if any frame's task text disagrees with frame 0's, this raises
    `ValueError` naming the episode and the mismatched frame index rather
    than silently picking frame 0's text for the whole episode.
    An episode whose task is the empty string (the fallback
    `write_lerobot_episodes`/`run_pipeline.py` use when the original
    `Episode.language_instruction` was `None`) maps back to `None`, not
    `""`, so callers can use a plain truthiness check.

    Populates `Episode.frames[<feature_key>]` for every feature declared
    `dtype: "video"` in `dataset.meta.features`. Verified empirically:
    `dataset[i][<video_key>]` is a decoded `torch.Tensor` of shape `(C, H,
    W)`, dtype `float32`, values in `[0, 1]` -- not the `(H, W, 3)` uint8
    `0-255` layout `check3_video_quality.py` operates on, so this function
    transposes to channel-last and rescales to `uint8` `0-255` before
    stacking into a `(T, H, W, 3)` array.
    """
    dataset = LeRobotDataset(repo_id=dataset_path.name, root=dataset_path)
    video_keys = [key for key, feature in dataset.meta.features.items() if feature.get("dtype") == "video"]
    episodes: List[Episode] = []
    for episode_index in range(dataset.num_episodes):
        episode_meta = dataset.meta.episodes[episode_index]
        from_index = episode_meta["dataset_from_index"]
        to_index = episode_meta["dataset_to_index"]
        rows = [dataset[i] for i in range(from_index, to_index)]
        state = np.stack([row["observation.state"].numpy() for row in rows])
        action = np.stack([row["action"].numpy() for row in rows])
        timestamps = np.array([row["timestamp"].item() for row in rows], dtype=np.float64)

        task = rows[0]["task"] if rows else ""
        for i, row in enumerate(rows):
            if row["task"] != task:
                raise ValueError(
                    f"episode {episode_index} has inconsistent per-frame task text: "
                    f"frame 0 has {task!r}, frame {i} has {row['task']!r}"
                )
        language_instruction = task if task else None

        frames = {}
        for video_key in video_keys:
            # (T, C, H, W) float32 in [0, 1] -> (T, H, W, C) uint8 in [0, 255].
            stacked_chw = np.stack([row[video_key].numpy() for row in rows])
            stacked_hwc = np.transpose(stacked_chw, (0, 2, 3, 1))
            frames[video_key] = np.clip(np.round(stacked_hwc * 255.0), 0, 255).astype(np.uint8)

        episodes.append(
            Episode(
                episode_index=episode_index,
                timestamps=timestamps,
                state=state,
                action=action,
                frames=frames,
                language_instruction=language_instruction,
            )
        )
    return episodes


def write_lerobot_episodes(
    episodes: List[Episode],
    output_path: Path,
    fps: float,
    robot_type: str,
    canonical_mask: Optional[np.ndarray] = None,
) -> None:
    """Write `episodes` out as a new lerobot dataset rooted at `output_path`.

    `canonical_mask`, when given, is a dataset-constant 1-D bool array (see
    unify_representation.py's `StageResult.stats["canonical_mask"]`) broadcast
    as an extra per-frame feature ("observation.state_canonical_mask") to
    every frame of every episode -- lerobot features are inherently per-frame,
    there is no dataset-level (non-per-frame) metadata slot to store a
    constant value in instead. When `canonical_mask` is None (the default),
    no such feature is added, preserving the pre-existing single-feature
    schema for every caller that doesn't pass it.
    """
    if not episodes:
        return
    state_dim = episodes[0].state.shape[1]
    action_dim = episodes[0].action.shape[1]
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    if canonical_mask is not None:
        canonical_mask = np.asarray(canonical_mask, dtype=bool)
        features["observation.state_canonical_mask"] = {
            "dtype": "bool",
            "shape": (canonical_mask.shape[0],),
            "names": None,
        }
    dataset = LeRobotDataset.create(
        repo_id=output_path.name,
        fps=fps,
        root=output_path,
        features=features,
        robot_type=robot_type,
        use_videos=False,
    )
    for episode in episodes:
        for t in range(episode.state.shape[0]):
            frame = {
                "observation.state": episode.state[t].astype(np.float32),
                "action": episode.action[t].astype(np.float32),
                "task": episode.language_instruction or "",
            }
            if canonical_mask is not None:
                frame["observation.state_canonical_mask"] = canonical_mask
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
