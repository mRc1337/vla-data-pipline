"""Config-driven RLDS/TFDS reader (Open-X-Embodiment convention: each
example is a dict with a ``"steps"`` sequence of per-timestep dicts under
``observation``/``action``/...).

Uses ``tfds.builder_from_directory`` -- the same entry point
``dump_dataset_schema.py``'s RLDS inspector already uses to report typed
feature specs -- so a dataset that ``dump_dataset_schema.py`` recognized as
``rlds`` is exactly what this reader expects. ``tensorflow_datasets`` is an
optional dependency (see ``requirements.txt``'s comment on it); this raises a
clear ``RuntimeError`` with the install hint if it's missing, same pattern as
``convert_core.hdf5_common.require_h5py``.

Known cost, by design, not yet solved: unlike HDF5 (where a header read is
enough to know each episode's frame count) or the count reported cheaply by
``builder.info.splits[...].num_examples``, RLDS's per-episode step count is
only knowable by decoding the ``"steps"`` sequence. ``build_plan`` therefore
decodes the *entire* requested split once and caches each episode's decoded
steps in ``EpisodePlan.extra["steps"]`` for ``iter_frames`` to replay without
re-decoding. This is fine for the small-to-medium open-source RLDS datasets
this was written against, but will not scale to a multi-hundred-GB shard
without changes -- before running this against one of those, switch
``iter_frames`` to re-open a fresh per-episode sub-iterator instead of
holding every episode's decoded steps in memory at once. Flagged again in
``PIPELINE_STATUS.md``.

Untested against a real ``tensorflow_datasets`` install as of this writing --
it isn't installed even in this project's own ``vla_data_pipline`` conda env
(only ``dump_dataset_schema.py``'s soft-import path has ever run for real
here). Validate against one real downloaded RLDS dataset on the server
before trusting this on anything larger.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from convert_core.dataset_config import DatasetConversionConfig
from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError


def _require_tfds():
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow_datasets is required for RLDS datasets; "
            "pip install tensorflow-cpu==2.15.0 tensorflow-datasets==4.9.9 "
            "(pins matched to the sibling openpi repo, per dump_dataset_schema.py)"
        ) from exc
    return tfds


def _find_version_dirs(dataset_root: Path) -> list[Path]:
    return sorted({info_path.parent for info_path in dataset_root.rglob("dataset_info.json")})


def _get_by_path(mapping: Any, path: str) -> Any:
    node = mapping
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _materialize_steps(steps_obj: Any, tfds: Any) -> list[dict[str, Any]]:
    if hasattr(steps_obj, "as_numpy_iterator"):
        return list(tfds.as_numpy(steps_obj))
    return list(steps_obj)


def _resolve_instruction(config: DatasetConversionConfig, first_step: dict[str, Any]) -> str:
    if config.instruction_source == "path_parent":
        raise ConversionError(
            "format=rlds does not support instruction_source=path_parent (no per-episode "
            "filesystem path exists); use instruction_source=field or constant"
        )
    if config.instruction_source == "constant":
        if not config.instruction_constant:
            raise ConversionError("instruction_source=constant requires instruction_constant to be set")
        return config.instruction_constant

    field_path = config.instruction_field or "language_instruction"
    try:
        value = _get_by_path(first_step, field_path)
    except KeyError:
        raise ConversionError(
            f"missing instruction field {field_path!r} on the first step; set instruction_field "
            "to the correct RLDS step field path, or use instruction_source=constant"
        ) from None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        item = value.reshape(-1)[0]
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)
    return str(value)


def _validate_vector_step(step: dict[str, Any], field, *, episode_index: int) -> None:
    try:
        value = _get_by_path(step, field.source_key)
    except KeyError:
        raise ConversionError(f"episode {episode_index}: missing field {field.source_key!r}") from None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != field.dim:
        raise ConversionError(
            f"episode {episode_index}: field {field.source_key!r} has size {array.size}, expected dim={field.dim}"
        )


def _camera_shape(step: dict[str, Any], source_key: str, feature_key: str, *, episode_index: int) -> tuple[int, int]:
    try:
        value = _get_by_path(step, source_key)
    except KeyError:
        raise ConversionError(f"episode {episode_index}: missing camera field {source_key!r}") from None
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ConversionError(
            f"episode {episode_index}: camera {feature_key!r} at {source_key!r} is not a decoded "
            f"(H, W, 3) uint8 image, got shape={image.shape} dtype={image.dtype} -- if this RLDS "
            "dataset stores encoded image bytes instead of auto-decoded arrays, this reader does "
            "not yet decode them; add that support before using this dataset"
        )
    return int(image.shape[0]), int(image.shape[1])


class RldsReader:
    def build_plan(
        self, config: DatasetConversionConfig, raw_root: Path, staging_root: Path
    ) -> DatasetConversionPlan:
        tfds = _require_tfds()
        if not config.vector_fields:
            raise ConversionError("format=rlds requires at least one entry in vector_fields")
        if config.fps is None:
            raise ConversionError(
                "format=rlds requires an explicit fps in the dataset config; no generic "
                "per-episode FPS metadata is read from RLDS yet"
            )

        dataset_root = raw_root / config.dataset_uid
        if not dataset_root.is_dir():
            raise ConversionError(f"raw dataset directory does not exist: {dataset_root}")
        version_dirs = _find_version_dirs(dataset_root)
        if not version_dirs:
            raise ConversionError(f"no dataset_info.json found below {dataset_root}")
        if len(version_dirs) > 1:
            raise ConversionError(
                f"multiple TFDS version directories found below {dataset_root}: "
                f"{[str(path) for path in version_dirs]}; point raw_root/dataset_uid at a single one"
            )
        version_dir = version_dirs[0]

        builder = tfds.builder_from_directory(str(version_dir))
        available_splits = list(builder.info.splits.keys())
        split = "+".join(available_splits) if config.split == "all" else config.split
        raw_dataset = builder.as_dataset(split=split)

        episodes: list[EpisodePlan] = []
        camera_dims: dict[str, tuple[int, int]] | None = None
        for episode_index, episode in enumerate(tfds.as_numpy(raw_dataset)):
            steps = _materialize_steps(episode.get("steps"), tfds)
            if not steps:
                raise ConversionError(f"episode {episode_index} of {config.dataset_uid} has zero steps")
            first_step = steps[0]

            for field in config.vector_fields:
                _validate_vector_step(first_step, field, episode_index=episode_index)

            frame_cameras = {
                camera.feature_key: _camera_shape(
                    first_step, camera.source_key, camera.feature_key, episode_index=episode_index
                )
                for camera in config.cameras
            }
            if camera_dims is None:
                camera_dims = frame_cameras
            elif frame_cameras != camera_dims:
                raise ConversionError(
                    f"episode {episode_index}: camera shapes {frame_cameras} differ from episode 0's {camera_dims}"
                )

            episodes.append(
                EpisodePlan(
                    episode_uid=f"episode_{episode_index}",
                    source_relative_path=f"{split}[{episode_index}]",
                    instruction=_resolve_instruction(config, first_step),
                    num_frames=len(steps),
                    extra={"steps": steps},
                )
            )

        if not episodes:
            raise ConversionError(f"{config.dataset_uid}: RLDS split {split!r} contains zero episodes")

        vector_features = tuple(
            VectorFeatureSpec(feature_key=field.feature_key, dim=field.dim, names=tuple(field.names) if field.names else None)
            for field in config.vector_fields
        )
        camera_features = tuple(
            CameraFeatureSpec(feature_key=key, height=height, width=width)
            for key, (height, width) in (camera_dims or {}).items()
        )
        integer_fps = int(round(config.fps))

        return DatasetConversionPlan(
            dataset_uid=config.dataset_uid,
            output_path=staging_root / "lerobot_v3_0" / config.dataset_uid,
            fps=integer_fps,
            measured_fps=float(config.fps),
            robot_type=config.robot_type,
            vector_features=vector_features,
            camera_features=camera_features,
            episodes=tuple(episodes),
            extra={
                "vector_fields": [
                    {"feature_key": field.feature_key, "source_key": field.source_key} for field in config.vector_fields
                ],
                "cameras": [
                    {"feature_key": camera.feature_key, "source_key": camera.source_key} for camera in config.cameras
                ],
            },
        )

    def iter_frames(self, plan: DatasetConversionPlan, episode: EpisodePlan) -> Iterator[dict[str, Any]]:
        steps: list[dict[str, Any]] = episode.extra["steps"]
        vector_fields: list[dict[str, str]] = plan.extra["vector_fields"]
        cameras: list[dict[str, str]] = plan.extra["cameras"]
        for step in steps:
            frame: dict[str, Any] = {
                field["feature_key"]: np.asarray(_get_by_path(step, field["source_key"]), dtype=np.float32).reshape(-1)
                for field in vector_fields
            }
            for camera in cameras:
                frame[camera["feature_key"]] = np.asarray(_get_by_path(step, camera["source_key"]), dtype=np.uint8)
            frame["task"] = episode.instruction
            yield frame
