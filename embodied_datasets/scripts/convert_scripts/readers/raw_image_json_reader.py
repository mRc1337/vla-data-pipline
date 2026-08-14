"""Config-driven reader for a new, greenfield source convention: one folder
per episode holding image files plus a JSON metadata sidecar. There is no
existing precedent for this layout anywhere else in the repo (unlike hdf5 and
rlds, which ``dump_dataset_schema.py`` already recognized) -- the convention
below is this reader's own invention, chosen to match the shape most small
open-source robot-learning dumps already use:

    <raw-root>/<dataset-uid>/.../<episode-dir>/
        <metadata_filename>              (default "metadata.json")
        <image files matching image_glob, naturally sorted>

``<metadata_filename>`` is expected to look like::

    {
      "language_instruction": "pick up the red block",
      "fps": 10,
      "frames": [
        {"state": [...], "action": [...], "image": "frame_0000.jpg"},
        ...
      ]
    }

``vector_fields[].source_key``/``cameras[].source_key`` are dotted paths
looked up inside each element of ``frames`` (e.g. ``"state"``, or
``"obs.state"`` for a nested field). A camera's ``source_key`` may instead be
the literal sentinel ``"$glob"``, meaning: don't look up a filename in the
JSON at all, just use the ``image_glob``-matched files directly, in natural
sort order, one per frame -- for the common case where episodes don't record
per-frame filenames because the numbering is already unambiguous.

This convention has not been checked against any real downloaded dataset
yet. Treat it as a starting point: inspect a couple of the actual
raw_image_json-shaped datasets once they're identified, and adjust this
docstring/parsing (most likely the ``frames`` key name, or whether metadata
is one file per episode vs. one file for the whole dataset) rather than
bending every dataset's config to fit an untested guess. See
``PIPELINE_STATUS.md``.
"""
from __future__ import annotations

import json
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
from convert_core.path_utils import natural_sort_key

GLOB_SENTINEL = "$glob"


def _get_dotted(mapping: Any, path: str) -> Any:
    node = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _discover_episode_dirs(dataset_root: Path, metadata_filename: str) -> list[Path]:
    return sorted({path.parent for path in dataset_root.rglob(metadata_filename)}, key=natural_sort_key)


def _load_metadata(episode_dir: Path, metadata_filename: str) -> dict[str, Any]:
    metadata_path = episode_dir / metadata_filename
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConversionError(f"{metadata_path}: cannot read/parse metadata JSON: {exc}") from exc


def _resolve_instruction(config: DatasetConversionConfig, episode_dir: Path, dataset_root: Path, metadata: dict[str, Any]) -> str:
    if config.instruction_source == "path_parent":
        relative = episode_dir.relative_to(dataset_root)
        if relative == Path("."):
            raise ConversionError(f"{episode_dir}: episode is the dataset root itself, no instruction subdirectory")
        return relative.as_posix()
    if config.instruction_source == "constant":
        if not config.instruction_constant:
            raise ConversionError("instruction_source=constant requires instruction_constant to be set")
        return config.instruction_constant
    field_path = config.instruction_field or "language_instruction"
    try:
        value = _get_dotted(metadata, field_path)
    except KeyError:
        raise ConversionError(
            f"{episode_dir}: missing instruction field {field_path!r} in {config.metadata_filename}"
        ) from None
    return str(value)


def _image_files(episode_dir: Path, image_glob: str) -> list[Path]:
    return sorted((path for path in episode_dir.glob(image_glob) if path.is_file()), key=natural_sort_key)


def _load_image(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("opencv-python-headless is required to decode raw_image_json frames") from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ConversionError(f"{path}: failed to decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class RawImageJsonReader:
    def build_plan(
        self, config: DatasetConversionConfig, raw_root: Path, staging_root: Path
    ) -> DatasetConversionPlan:
        if not config.vector_fields:
            raise ConversionError("format=raw_image_json requires at least one entry in vector_fields")
        if not config.cameras:
            raise ConversionError("format=raw_image_json requires at least one entry in cameras")

        dataset_root = raw_root / config.dataset_uid
        if not dataset_root.is_dir():
            raise ConversionError(f"raw dataset directory does not exist: {dataset_root}")
        episode_dirs = _discover_episode_dirs(dataset_root, config.metadata_filename)
        if not episode_dirs:
            raise ConversionError(f"no {config.metadata_filename!r} found below {dataset_root}")

        episodes: list[EpisodePlan] = []
        camera_dims: dict[str, tuple[int, int]] | None = None
        fps_values: list[float] = []

        for episode_dir in episode_dirs:
            metadata = _load_metadata(episode_dir, config.metadata_filename)
            frames = metadata.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ConversionError(f"{episode_dir}: {config.metadata_filename} has no non-empty 'frames' list")

            for field in config.vector_fields:
                try:
                    value = _get_dotted(frames[0], field.source_key)
                except KeyError:
                    raise ConversionError(f"{episode_dir}: frame 0 is missing field {field.source_key!r}") from None
                size = np.asarray(value, dtype=np.float64).reshape(-1).size
                if size != field.dim:
                    raise ConversionError(
                        f"{episode_dir}: field {field.source_key!r} has size {size}, expected dim={field.dim}"
                    )

            glob_files = None
            frame_cameras: dict[str, tuple[int, int]] = {}
            for camera in config.cameras:
                if camera.source_key == GLOB_SENTINEL:
                    if glob_files is None:
                        glob_files = _image_files(episode_dir, config.image_glob)
                        if len(glob_files) != len(frames):
                            raise ConversionError(
                                f"{episode_dir}: {len(glob_files)} files match {config.image_glob!r}, "
                                f"expected {len(frames)} (one per frame in 'frames')"
                            )
                    sample_path = glob_files[0]
                else:
                    try:
                        image_name = _get_dotted(frames[0], camera.source_key)
                    except KeyError:
                        raise ConversionError(
                            f"{episode_dir}: frame 0 is missing camera field {camera.source_key!r}"
                        ) from None
                    sample_path = episode_dir / str(image_name)
                    if not sample_path.is_file():
                        raise ConversionError(f"{episode_dir}: image file does not exist: {sample_path}")
                height, width = _load_image(sample_path).shape[:2]
                frame_cameras[camera.feature_key] = (height, width)

            if camera_dims is None:
                camera_dims = frame_cameras
            elif frame_cameras != camera_dims:
                raise ConversionError(
                    f"{episode_dir}: camera shapes {frame_cameras} differ from the first episode's {camera_dims}"
                )

            episode_fps = metadata.get("fps", config.fps)
            if episode_fps is None:
                raise ConversionError(
                    f"{episode_dir}: {config.metadata_filename} has no 'fps' field, and the dataset "
                    "config has no fallback fps either"
                )
            fps_values.append(float(episode_fps))

            episodes.append(
                EpisodePlan(
                    episode_uid=episode_dir.name,
                    source_relative_path=episode_dir.relative_to(dataset_root).as_posix(),
                    instruction=_resolve_instruction(config, episode_dir, dataset_root, metadata),
                    num_frames=len(frames),
                    extra={"episode_dir": episode_dir, "frames": frames},
                )
            )

        measured_fps = float(np.median(fps_values))
        integer_fps = int(round(measured_fps))
        if integer_fps <= 0:
            raise ConversionError(f"measured/fallback FPS {measured_fps!r} is not a positive number")

        vector_features = tuple(
            VectorFeatureSpec(feature_key=field.feature_key, dim=field.dim, names=tuple(field.names) if field.names else None)
            for field in config.vector_fields
        )
        camera_features = tuple(
            CameraFeatureSpec(feature_key=key, height=height, width=width)
            for key, (height, width) in (camera_dims or {}).items()
        )

        return DatasetConversionPlan(
            dataset_uid=config.dataset_uid,
            output_path=staging_root / "lerobot_v3_0" / config.dataset_uid,
            fps=integer_fps,
            measured_fps=measured_fps,
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
                "image_glob": config.image_glob,
            },
        )

    def iter_frames(self, plan: DatasetConversionPlan, episode: EpisodePlan) -> Iterator[dict[str, Any]]:
        episode_dir: Path = episode.extra["episode_dir"]
        frames: list[dict[str, Any]] = episode.extra["frames"]
        vector_fields: list[dict[str, str]] = plan.extra["vector_fields"]
        cameras: list[dict[str, str]] = plan.extra["cameras"]

        glob_files: list[Path] | None = None
        for frame_index, frame_data in enumerate(frames):
            frame: dict[str, Any] = {
                field["feature_key"]: np.asarray(_get_dotted(frame_data, field["source_key"]), dtype=np.float32).reshape(-1)
                for field in vector_fields
            }
            for camera in cameras:
                if camera["source_key"] == GLOB_SENTINEL:
                    if glob_files is None:
                        glob_files = _image_files(episode_dir, plan.extra["image_glob"])
                    image_path = glob_files[frame_index]
                else:
                    image_path = episode_dir / str(_get_dotted(frame_data, camera["source_key"]))
                frame[camera["feature_key"]] = _load_image(image_path)
            frame["task"] = episode.instruction
            yield frame
