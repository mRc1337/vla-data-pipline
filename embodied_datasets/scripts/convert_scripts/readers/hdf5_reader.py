"""Config-driven HDF5 reader for the general case: episodes are single HDF5
files with a fixed set of named numeric vector fields and an auto-discovered
(or explicitly configured) set of RGB cameras.

Deliberately does *not* handle Mobile ALOHA's dual-arm + mobile-base 14+2
action-layout ambiguity (``separate_14_plus_2`` vs. ``combined_16`` cross-
validated against ``/base_action``) -- that is genuine per-embodiment domain
logic, not a generic HDF5 concern, and stays in
``convert_mobile_aloha_to_lerobot.py`` unmodified. Any HDF5 dataset whose
action is a single unambiguous vector (the common case for single-arm or
already-flattened action spaces) belongs here instead, driven entirely by
``configs/<uid>.yaml``'s ``vector_fields``/``cameras`` -- no new Python file
per dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from convert_core.dataset_config import CameraFieldConfig, DatasetConversionConfig
from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError
from convert_core.hdf5_common import (
    FPS_RELATIVE_TOLERANCE,
    CameraSpec,
    camera_fps,
    classify_camera_dataset,
    discover_cameras,
    natural_sort_key,
    normalize_hdf5_key,
    read_rgb_frame,
    relative_difference,
    require_dataset,
    require_h5py,
    validate_numeric_matrix,
)


@dataclass(frozen=True)
class _EpisodeHdf5Info:
    source_path: Path
    instruction: str
    num_frames: int
    cameras: tuple[CameraSpec, ...]
    fps_values: tuple[float, ...]


def _discover_episode_paths(raw_dataset_root: Path, episode_glob: str) -> list[Path]:
    patterns = {episode_glob}
    if episode_glob.endswith(".h5"):
        patterns.add(episode_glob[: -len(".h5")] + ".hdf5")
    elif episode_glob.endswith(".hdf5"):
        patterns.add(episode_glob[: -len(".hdf5")] + ".h5")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in raw_dataset_root.glob(pattern) if path.is_file())
    return sorted(paths, key=natural_sort_key)


def _resolve_instruction(
    config: DatasetConversionConfig, raw_dataset_root: Path, episode_path: Path, h5_file: Any
) -> str:
    if config.instruction_source == "path_parent":
        relative_parent = episode_path.parent.relative_to(raw_dataset_root)
        if relative_parent == Path("."):
            raise ConversionError(
                f"{episode_path}: episode is directly below the dataset root and has no instruction subdirectory"
            )
        return relative_parent.as_posix()
    if config.instruction_source == "constant":
        if not config.instruction_constant:
            raise ConversionError("instruction_source=constant requires instruction_constant to be set")
        return config.instruction_constant
    if not config.instruction_field:
        raise ConversionError("instruction_source=field requires instruction_field to be set")
    value = h5_file[normalize_hdf5_key(config.instruction_field)] if normalize_hdf5_key(
        config.instruction_field
    ) in h5_file else None
    if value is None:
        raise ConversionError(f"{episode_path}: missing instruction field {config.instruction_field!r}")
    raw = value[()]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, np.ndarray):
        item = raw.reshape(-1)[0]
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)
    return str(raw)


def _camera_spec_from_config(h5_file: Any, camera_cfg: CameraFieldConfig, episode_path: Path) -> CameraSpec:
    dataset = require_dataset(h5_file, camera_cfg.source_key, episode_path)
    source_key = normalize_hdf5_key(camera_cfg.source_key)
    storage, layout, height, width = classify_camera_dataset(dataset, source_path=episode_path, source_key=source_key)
    name = camera_cfg.feature_key.rsplit(".", 1)[-1]
    return CameraSpec(
        name=name,
        source_key=source_key,
        feature_key=camera_cfg.feature_key,
        height=height,
        width=width,
        storage=storage,
        source_layout=layout,
    )


def _inspect_episode(episode_path: Path, *, raw_dataset_root: Path, config: DatasetConversionConfig) -> _EpisodeHdf5Info:
    h5py = require_h5py()
    if not config.vector_fields:
        raise ConversionError("format=hdf5 requires at least one entry in vector_fields")

    with h5py.File(episode_path, "r") as h5_file:
        instruction = _resolve_instruction(config, raw_dataset_root, episode_path, h5_file)

        num_frames: int | None = None
        for field in config.vector_fields:
            dataset = require_dataset(h5_file, field.source_key, episode_path)
            frames, _width = validate_numeric_matrix(
                dataset,
                source_path=episode_path,
                key=field.source_key,
                expected_width=field.dim,
                expected_frames=num_frames,
            )
            if num_frames is None:
                num_frames = frames
        assert num_frames is not None

        if config.cameras:
            cameras = tuple(
                _camera_spec_from_config(h5_file, camera_cfg, episode_path) for camera_cfg in config.cameras
            )
        else:
            cameras, _skipped_depth = discover_cameras(h5_file, config.images_key, episode_path)

        for camera in cameras:
            camera_dataset = h5_file[camera.source_key]
            if camera_dataset.shape[0] != num_frames:
                raise ConversionError(
                    f"{episode_path}: {camera.source_key} has {camera_dataset.shape[0]} frames, "
                    f"expected {num_frames}"
                )

        fps_values = tuple(
            camera_fps(
                h5_file,
                camera,
                images_key=config.images_key,
                num_frames=num_frames,
                source_path=episode_path,
                raw_dataset_root=raw_dataset_root,
                fallback_fps=config.fps,
            ).value
            for camera in cameras
        )

    return _EpisodeHdf5Info(
        source_path=episode_path,
        instruction=instruction,
        num_frames=num_frames,
        cameras=cameras,
        fps_values=fps_values,
    )


class Hdf5Reader:
    """Generic single-file-per-episode HDF5 reader, config-driven."""

    def build_plan(
        self, config: DatasetConversionConfig, raw_root: Path, staging_root: Path
    ) -> DatasetConversionPlan:
        if Path(config.dataset_uid).name != config.dataset_uid or config.dataset_uid in {"", ".", ".."}:
            raise ConversionError(f"dataset UID must be one path component, got {config.dataset_uid!r}")
        raw_dataset_root = raw_root / config.dataset_uid
        if not raw_dataset_root.is_dir():
            raise ConversionError(f"raw dataset directory does not exist: {raw_dataset_root}")

        episode_paths = _discover_episode_paths(raw_dataset_root, config.episode_glob)
        if not episode_paths:
            raise ConversionError(f"no episodes matched {config.episode_glob!r} below {raw_dataset_root}")

        infos = [
            _inspect_episode(path, raw_dataset_root=raw_dataset_root, config=config) for path in episode_paths
        ]

        reference_cameras = tuple((camera.feature_key, camera.height, camera.width) for camera in infos[0].cameras)
        for info in infos[1:]:
            cameras = tuple((camera.feature_key, camera.height, camera.width) for camera in info.cameras)
            if cameras != reference_cameras:
                raise ConversionError(
                    f"{info.source_path}: camera schema {cameras} differs from first episode schema {reference_cameras}"
                )

        fps_values = [value for info in infos for value in info.fps_values]
        if not fps_values:
            raise ConversionError("no cameras found; cannot measure FPS")
        measured_fps = float(np.median(fps_values))
        inconsistent = [
            value for value in fps_values if relative_difference(value, measured_fps) > FPS_RELATIVE_TOLERANCE
        ]
        if inconsistent:
            details = ", ".join(f"{value:.6g}" for value in fps_values)
            raise ConversionError(f"camera/episode FPS values differ by more than 2%: {details}")
        integer_fps = int(round(measured_fps))
        if integer_fps <= 0 or relative_difference(measured_fps, integer_fps) > FPS_RELATIVE_TOLERANCE:
            raise ConversionError(
                f"measured FPS {measured_fps:.6g} is not within 2% of an integer; "
                "LeRobot video encoding requires an integer FPS and this reader does not resample"
            )

        vector_features = tuple(
            VectorFeatureSpec(feature_key=field.feature_key, dim=field.dim, names=tuple(field.names) if field.names else None)
            for field in config.vector_fields
        )
        camera_features = tuple(
            CameraFeatureSpec(feature_key=camera.feature_key, height=camera.height, width=camera.width)
            for camera in infos[0].cameras
        )
        episodes = tuple(
            EpisodePlan(
                episode_uid=f"episode_{index}",
                source_relative_path=info.source_path.relative_to(raw_dataset_root).as_posix(),
                instruction=info.instruction,
                num_frames=info.num_frames,
                extra={"source_path": info.source_path, "cameras": info.cameras},
            )
            for index, info in enumerate(infos)
        )
        return DatasetConversionPlan(
            dataset_uid=config.dataset_uid,
            output_path=staging_root / "lerobot_v3_0" / config.dataset_uid,
            fps=integer_fps,
            measured_fps=measured_fps,
            robot_type=config.robot_type,
            vector_features=vector_features,
            camera_features=camera_features,
            episodes=episodes,
            extra={
                "uncompressed_color_order": config.uncompressed_color_order,
                "vector_fields": [
                    {"feature_key": field.feature_key, "source_key": field.source_key, "dim": field.dim}
                    for field in config.vector_fields
                ],
            },
        )

    def iter_frames(self, plan: DatasetConversionPlan, episode: EpisodePlan) -> Iterator[dict[str, Any]]:
        h5py = require_h5py()
        source_path: Path = episode.extra["source_path"]
        cameras: tuple[CameraSpec, ...] = episode.extra["cameras"]
        uncompressed_color_order: str = plan.extra["uncompressed_color_order"]
        vector_fields: list[dict[str, Any]] = plan.extra["vector_fields"]

        with h5py.File(source_path, "r") as h5_file:
            arrays = {
                field["feature_key"]: np.asarray(h5_file[normalize_hdf5_key(field["source_key"])][:], dtype=np.float32)
                for field in vector_fields
            }
            for frame_index in range(episode.num_frames):
                frame: dict[str, Any] = {key: values[frame_index] for key, values in arrays.items()}
                frame["task"] = episode.instruction
                for camera in cameras:
                    frame[camera.feature_key] = read_rgb_frame(
                        h5_file[camera.source_key],
                        frame_index,
                        camera,
                        source_path=source_path,
                        uncompressed_color_order=uncompressed_color_order,
                    )
                yield frame
