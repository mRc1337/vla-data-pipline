"""Thin wrappers over the official lerobot dataset read/write API.

Verified against the installed lerobot==0.4.4 API (lerobot.datasets.lerobot_dataset.
LeRobotDataset), which differs from a naive v2.1-era guess in a few load-bearing ways:

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
  fps`, not from any timestamp we pass in -- round-tripping does not preserve
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
from typing import Dict, List, Optional

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from episode import CameraCalibration, Episode


def _load_camera_calibration(dataset, video_keys, rows, episode_index: int) -> Dict[str, CameraCalibration]:
    """See docs/superpowers/specs/2026-07-28-sam3-check2-camera-calibration-design.md
    section 2.2: for each video key, `<key>_intrinsics` (shape (4,),
    [fx, fy, cx, cy]) and `<key>_extrinsics` (shape (16,), flattened
    row-major camera_from_base 4x4) are optional per-frame features. Both
    present -> validate identical across every frame of this episode (same
    "must be episode-constant" contract as `task`/language_instruction
    above) and build a CameraCalibration from frame 0. Only one present ->
    the data is self-contradictory, raise. Neither present -> this view has
    no calibration, no entry in the returned dict.
    """
    calibration: Dict[str, CameraCalibration] = {}
    for video_key in video_keys:
        intrinsics_key = f"{video_key}_intrinsics"
        extrinsics_key = f"{video_key}_extrinsics"
        has_intrinsics = intrinsics_key in dataset.meta.features
        has_extrinsics = extrinsics_key in dataset.meta.features
        if has_intrinsics != has_extrinsics:
            raise ValueError(
                f"episode {episode_index} view {video_key!r} has only one of "
                f"{intrinsics_key!r}/{extrinsics_key!r} -- both or neither must be present"
            )
        if not has_intrinsics:
            continue

        first_intrinsics = rows[0][intrinsics_key].numpy()
        first_extrinsics = rows[0][extrinsics_key].numpy()
        for i, row in enumerate(rows):
            if not np.array_equal(row[intrinsics_key].numpy(), first_intrinsics):
                raise ValueError(
                    f"episode {episode_index} view {video_key!r} has inconsistent per-frame "
                    f"{intrinsics_key!r}: frame 0 has {first_intrinsics.tolist()}, "
                    f"frame {i} has {row[intrinsics_key].numpy().tolist()}"
                )
            if not np.array_equal(row[extrinsics_key].numpy(), first_extrinsics):
                raise ValueError(
                    f"episode {episode_index} view {video_key!r} has inconsistent per-frame "
                    f"{extrinsics_key!r}: frame 0 has {first_extrinsics.tolist()}, "
                    f"frame {i} has {row[extrinsics_key].numpy().tolist()}"
                )

        fx, fy, cx, cy = [float(v) for v in first_intrinsics]
        calibration[video_key] = CameraCalibration(
            fx=fx, fy=fy, cx=cx, cy=cy, extrinsics=first_extrinsics.reshape(4, 4).astype(float)
        )
    return calibration


def _rows_for_episode(dataset, from_index: int, to_index: int, load_video_frames: bool) -> list:
    """Returns the per-frame row dicts for one episode's absolute frame
    range `[from_index, to_index)`.

    `dataset[i]` (LeRobotDataset.__getitem__) unconditionally decodes every
    declared video feature for frame `i` as part of building the row --
    verified against lerobot==0.4.4's DatasetReader.get_item, which merges
    decoded video frames into the row whenever `dataset.meta.video_keys` is
    non-empty, before any feature is read out of it. So when
    `load_video_frames` is False, this reads `dataset.get_raw_item(i)`
    instead (the underlying HF-dataset row with no video decoding, no
    delta-timestamp expansion, no image transforms) to actually skip that
    decode cost rather than just discarding the result afterward.

    The raw row has no `"task"` key -- `get_item` only resolves
    `task_index` -> task string via `dataset.meta.tasks` for the decoded
    path -- so this manually replicates that one step (the only other
    per-row transform `get_item` does besides video decoding) so every
    caller downstream can keep reading `row["task"]` the same way
    regardless of `load_video_frames`.
    """
    if load_video_frames:
        return [dataset[i] for i in range(from_index, to_index)]
    rows = [dataset.get_raw_item(i) for i in range(from_index, to_index)]
    for row in rows:
        row["task"] = dataset.meta.tasks.iloc[row["task_index"].item()].name
    return rows


def load_lerobot_episodes(dataset_path: Path, load_video_frames: bool = True) -> List[Episode]:
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

    `load_video_frames`, when False, skips decoding video content entirely
    -- `Episode.frames[<video_key>]` is still populated for every view key
    (an empty `(0, 0, 0, 0)` placeholder array, not real pixel data), so
    callers that only need to know WHICH view keys exist (`.frames.keys()`)
    keep working unchanged. Every other field (state/action/timestamps/
    language_instruction/camera_calibration) is populated exactly as when
    True. Use this when a caller never reads `.frames` values for real
    pixels (e.g. inspect_tool/app.py's `_load_final_episodes`, which now
    reads video pixels from the local HTTP video server instead) to avoid
    paying video-decode cost and memory for arrays nothing looks at.
    """
    dataset = LeRobotDataset(repo_id=dataset_path.name, root=dataset_path)
    video_keys = [key for key, feature in dataset.meta.features.items() if feature.get("dtype") == "video"]
    episodes: List[Episode] = []
    for episode_index in range(dataset.num_episodes):
        episode_meta = dataset.meta.episodes[episode_index]
        from_index = episode_meta["dataset_from_index"]
        to_index = episode_meta["dataset_to_index"]
        rows = _rows_for_episode(dataset, from_index, to_index, load_video_frames)
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
            if load_video_frames:
                # (T, C, H, W) float32 in [0, 1] -> (T, H, W, C) uint8 in [0, 255].
                stacked_chw = np.stack([row[video_key].numpy() for row in rows])
                stacked_hwc = np.transpose(stacked_chw, (0, 2, 3, 1))
                frames[video_key] = np.clip(np.round(stacked_hwc * 255.0), 0, 255).astype(np.uint8)
            else:
                frames[video_key] = np.empty((0, 0, 0, 0), dtype=np.uint8)

        camera_calibration = _load_camera_calibration(dataset, video_keys, rows, episode_index)

        episodes.append(
            Episode(
                episode_index=episode_index,
                timestamps=timestamps,
                state=state,
                action=action,
                frames=frames,
                language_instruction=language_instruction,
                camera_calibration=camera_calibration,
            )
        )
    return episodes


def write_lerobot_episodes(
    episodes: List[Episode],
    output_path: Path,
    fps: float,
    robot_type: str,
    canonical_mask: Optional[np.ndarray] = None,
    action_canonical_mask: Optional[np.ndarray] = None,
    write_videos: bool = False,
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

    `action_canonical_mask` is the analogous dataset-constant 1-D bool array
    for the action layer (see unify_representation.py's
    `StageResult.stats["action_canonical_mask"]` from `apply_action()`),
    written as its own extra per-frame feature ("action_canonical_mask").
    Independent of `canonical_mask` -- either, both, or neither may be
    passed.

    `write_videos`, when True, additionally declares a `dtype: "video"`
    feature for every view key present in `episodes[0].frames` and writes
    each frame's per-view image alongside state/action -- used only by
    inspect_tool's instrumented_pipeline.py (run_pipeline.py never passes
    this), so production dataset output is unaffected by this parameter's
    existence. Relies on `episode.frames` already being frame-count-aligned
    with `episode.state`/`episode.action` -- true for every Episode this
    pipeline produces, since stage1/stage2/stage3/check3 all slice `frames`
    in lockstep with their state/action drops.
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
    if action_canonical_mask is not None:
        action_canonical_mask = np.asarray(action_canonical_mask, dtype=bool)
        features["action_canonical_mask"] = {
            "dtype": "bool",
            "shape": (action_canonical_mask.shape[0],),
            "names": None,
        }
    view_keys = list(episodes[0].frames.keys()) if write_videos else []
    for view_key in view_keys:
        height, width = episodes[0].frames[view_key].shape[1:3]
        features[view_key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    # PyAV's add_stream(vcodec, fps, ...) needs an fps with a `.numerator`
    # attribute -- a plain float (even a whole number) raises AttributeError
    # ("'float' object has no attribute 'numerator'"). Only the
    # video-encoding path is affected; the pre-existing non-video path keeps
    # passing `fps` through as-is.
    create_fps = int(fps) if write_videos else fps
    dataset = LeRobotDataset.create(
        repo_id=output_path.name,
        fps=create_fps,
        root=output_path,
        features=features,
        robot_type=robot_type,
        use_videos=write_videos,
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
            if action_canonical_mask is not None:
                frame["action_canonical_mask"] = action_canonical_mask
            for view_key in view_keys:
                frame[view_key] = episode.frames[view_key][t]
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
