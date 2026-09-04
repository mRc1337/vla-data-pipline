"""Generic HDF5 helpers shared by every HDF5-backed reader.

Extracted from ``convert_mobile_aloha_to_lerobot.py`` (the original, and still
only battle-tested, HDF5 converter in this repo). Everything here is format
detail that has nothing to do with Mobile ALOHA specifically -- camera
discovery under an images group, RGB frame decoding, FPS inference from
timestamps/attrs/sidecar files, and generic numeric-matrix validation. What
stayed behind in ``convert_mobile_aloha_to_lerobot.py`` is genuinely
ALOHA-specific: the dual-arm + mobile-base 14+2 action layout ambiguity
(``separate_14_plus_2`` vs. ``combined_16``) and its cross-validation against
``/base_action``. That's robot-embodiment domain logic, not a generic HDF5
concern, so it was not generalized away -- see ``readers/hdf5_reader.py`` for
the config-driven reader that this module backs instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np

from convert_core.errors import ConversionError
from convert_core.path_utils import natural_sort_key

FPS_ATTRIBUTE_NAMES = {"fps", "frame_rate", "framerate", "camera_fps"}
TIMESTAMP_JITTER_TOLERANCE = 0.10
FPS_RELATIVE_TOLERANCE = 0.02


@dataclass(frozen=True)
class CameraSpec:
    name: str
    source_key: str
    feature_key: str
    height: int
    width: int
    storage: str
    source_layout: str


@dataclass(frozen=True)
class FpsEvidence:
    camera: str
    value: float
    source: str


def require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("h5py is required; install the project's requirements.txt") from exc
    return h5py


def decode_hdf5_text(value: Any, *, description: str) -> str:
    """Decode one scalar HDF5 text attr without accepting lossy coercions."""

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConversionError(f"{description} is not valid UTF-8") from exc
    if isinstance(value, str):
        return value
    raise ConversionError(f"{description} must be a UTF-8 string, got {type(value).__name__}")


def robomimic_demo_sort_key(name: str) -> int:
    """Validate and numerically sort canonical ``data/demo_N`` groups."""

    match = re.fullmatch(r"demo_(\d+)", name)
    if match is None:
        raise ConversionError(
            f"unexpected robomimic episode name {name!r}; expected demo_<integer>"
        )
    return int(match.group(1))


def hdf5_leaf_schema(group: Any) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Return sorted time-tail shapes and exact dtypes for every dataset leaf."""

    h5py = require_h5py()
    leaves: list[tuple[str, tuple[int, ...], str]] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            if not obj.shape:
                raise ConversionError(f"HDF5 leaf {name!r} is scalar, expected a time axis")
            leaves.append(
                (name, tuple(int(value) for value in obj.shape[1:]), str(obj.dtype))
            )

    group.visititems(visit)
    return tuple(sorted(leaves))


def validate_time_major_group(
    group: Any,
    schema: Sequence[tuple[str, tuple[int, ...], str]],
    *,
    expected_frames: int,
    description: str,
) -> None:
    """Require every planned leaf to keep its full schema and first dimension."""

    actual = hdf5_leaf_schema(group)
    if tuple(schema) != actual:
        raise ConversionError(f"{description}: schema differs from the partition reference")
    for key, _shape, _dtype in schema:
        dataset = group[key]
        if int(dataset.shape[0]) != expected_frames:
            raise ConversionError(
                f"{description}/{key}: first dimension {dataset.shape[0]} does not "
                f"match num_samples={expected_frames}"
            )


def normalize_hdf5_key(key: str) -> str:
    return "/" + key.strip("/")


def get_hdf5_object(h5_file: Any, key: str) -> Any | None:
    normalized = normalize_hdf5_key(key)
    return h5_file[normalized] if normalized in h5_file else None


def require_dataset(h5_file: Any, key: str, source_path: Path) -> Any:
    h5py = require_h5py()
    value = get_hdf5_object(h5_file, key)
    if value is None:
        raise ConversionError(f"{source_path}: missing required HDF5 key {normalize_hdf5_key(key)!r}")
    if not isinstance(value, h5py.Dataset):
        raise ConversionError(f"{source_path}: HDF5 key {normalize_hdf5_key(key)!r} is not a dataset")
    return value


def validate_numeric_matrix(
    dataset: Any,
    *,
    source_path: Path,
    key: str,
    expected_width: int | Sequence[int],
    expected_frames: int | None = None,
) -> tuple[int, int]:
    widths = {expected_width} if isinstance(expected_width, int) else set(expected_width)
    if dataset.ndim != 2 or dataset.shape[1] not in widths:
        expected = "/".join(str(width) for width in sorted(widths))
        raise ConversionError(
            f"{source_path}: {normalize_hdf5_key(key)} must have shape (T, {expected}), got {dataset.shape}"
        )
    if expected_frames is not None and dataset.shape[0] != expected_frames:
        raise ConversionError(
            f"{source_path}: {normalize_hdf5_key(key)} has {dataset.shape[0]} frames, "
            f"expected {expected_frames}"
        )
    if not np.issubdtype(dataset.dtype, np.number):
        raise ConversionError(f"{source_path}: {normalize_hdf5_key(key)} must be numeric, got {dataset.dtype}")
    values = np.asarray(dataset[:])
    if not np.isfinite(values).all():
        raise ConversionError(f"{source_path}: {normalize_hdf5_key(key)} contains NaN or infinity")
    return int(dataset.shape[0]), int(dataset.shape[1])


def camera_name(relative_key: str) -> str:
    parts = [part.strip().replace(" ", "_") for part in relative_key.strip("/").split("/") if part.strip()]
    if not parts:
        raise ConversionError("an RGB camera has an empty name")
    return ".".join(parts)


def compressed_image_bytes(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.uint8)
    array = np.asarray(value)
    if array.dtype.kind in {"S", "V"}:
        return np.frombuffer(array.tobytes(), dtype=np.uint8)
    return np.asarray(array, dtype=np.uint8).reshape(-1)


def decode_compressed_image(dataset: Any, frame_index: int, *, source_path: Path, source_key: str) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("opencv-python-headless is required to decode compressed RGB frames") from exc

    encoded = compressed_image_bytes(dataset[frame_index])
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ConversionError(f"{source_path}: failed to decode frame {frame_index} from {source_key}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def classify_camera_dataset(dataset: Any, *, source_path: Path, source_key: str) -> tuple[str, str, int, int]:
    shape = tuple(int(value) for value in dataset.shape)
    if dataset.ndim == 4 and shape[-1] == 3:
        if dataset.dtype != np.uint8:
            raise ConversionError(f"{source_path}: RGB camera {source_key} must be uint8, got {dataset.dtype}")
        return "uncompressed", "THWC", shape[1], shape[2]
    if dataset.ndim == 4 and shape[1] == 3:
        if dataset.dtype != np.uint8:
            raise ConversionError(f"{source_path}: RGB camera {source_key} must be uint8, got {dataset.dtype}")
        return "uncompressed", "TCHW", shape[2], shape[3]
    if dataset.ndim in {1, 2} and shape[0] > 0:
        sample = decode_compressed_image(dataset, 0, source_path=source_path, source_key=source_key)
        return "compressed", "encoded", int(sample.shape[0]), int(sample.shape[1])
    raise ConversionError(
        f"{source_path}: unsupported RGB camera layout at {source_key}: shape={shape}, dtype={dataset.dtype}"
    )


def _visit_image_object(
    value: Any,
    relative_key: str,
    images_group: Any,
    source_path: Path,
    camera_specs: list[CameraSpec],
    skipped_depth: list[str],
) -> None:
    h5py = require_h5py()
    if not isinstance(value, h5py.Dataset):
        return
    source_key = normalize_hdf5_key(f"{images_group.name}/{relative_key}")
    lowered = relative_key.casefold()
    if "depth" in lowered:
        skipped_depth.append(source_key)
        return
    if "timestamp" in lowered:
        return
    storage, layout, height, width = classify_camera_dataset(
        value, source_path=source_path, source_key=source_key
    )
    name = camera_name(relative_key)
    camera_specs.append(
        CameraSpec(
            name=name,
            source_key=source_key,
            feature_key=f"observation.images.{name}",
            height=height,
            width=width,
            storage=storage,
            source_layout=layout,
        )
    )


def discover_cameras(h5_file: Any, images_key: str, source_path: Path) -> tuple[tuple[CameraSpec, ...], tuple[str, ...]]:
    h5py = require_h5py()
    images = get_hdf5_object(h5_file, images_key)
    if images is None or not isinstance(images, h5py.Group):
        raise ConversionError(f"{source_path}: missing image group {normalize_hdf5_key(images_key)!r}")

    camera_specs: list[CameraSpec] = []
    skipped_depth: list[str] = []
    images.visititems(
        lambda relative_key, value: _visit_image_object(
            value, relative_key, images, source_path, camera_specs, skipped_depth
        )
    )
    if not camera_specs:
        raise ConversionError(f"{source_path}: no RGB cameras found below {normalize_hdf5_key(images_key)}")

    feature_keys = [camera.feature_key for camera in camera_specs]
    if len(feature_keys) != len(set(feature_keys)):
        raise ConversionError(f"{source_path}: camera names collide after feature-key normalization: {feature_keys}")
    return tuple(sorted(camera_specs, key=lambda item: item.feature_key)), tuple(sorted(skipped_depth))


def read_rgb_frame(
    dataset: Any,
    frame_index: int,
    camera: CameraSpec,
    *,
    source_path: Path,
    uncompressed_color_order: str,
) -> np.ndarray:
    if camera.storage == "compressed":
        return decode_compressed_image(
            dataset, frame_index, source_path=source_path, source_key=camera.source_key
        )
    image = np.asarray(dataset[frame_index])
    if camera.source_layout == "TCHW":
        image = np.transpose(image, (1, 2, 0))
    if uncompressed_color_order == "bgr":
        image = image[..., ::-1]
    image = np.ascontiguousarray(image, dtype=np.uint8)
    expected = (camera.height, camera.width, 3)
    if image.shape != expected:
        raise ConversionError(
            f"{source_path}: frame {frame_index} from {camera.source_key} has shape {image.shape}, expected {expected}"
        )
    return image


def as_finite_positive_float(value: Any) -> float | None:
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        result = float(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def fps_from_attrs(objects: Iterable[Any]) -> tuple[float, str] | None:
    for obj in objects:
        for key, value in obj.attrs.items():
            if str(key).casefold().replace("-", "_") not in FPS_ATTRIBUTE_NAMES:
                continue
            fps = as_finite_positive_float(value)
            if fps is not None:
                return fps, f"HDF5 attribute {obj.name}:{key}"
    return None


def timestamp_candidates(camera: CameraSpec, images_key: str) -> tuple[str, ...]:
    image_path = Path(camera.source_key.strip("/"))
    leaf = image_path.name
    parent = "/" + image_path.parent.as_posix()
    return tuple(
        dict.fromkeys(
            [
                f"{camera.source_key}_timestamps",
                f"{parent}/{leaf}_timestamps",
                f"{parent}/timestamps/{leaf}",
                f"{normalize_hdf5_key(images_key)}/timestamps/{leaf}",
                f"/observations/timestamps/{leaf}",
                f"/camera_timestamps/{leaf}",
                f"/timestamps/{leaf}",
                "/timestamps",
            ]
        )
    )


def fps_from_timestamps(values: np.ndarray, *, source_path: Path, source_key: str) -> tuple[float, str]:
    timestamps = np.asarray(values, dtype=np.float64).reshape(-1)
    if timestamps.size < 2 or not np.isfinite(timestamps).all():
        raise ConversionError(f"{source_path}: {source_key} must contain at least two finite timestamps")
    diffs = np.diff(timestamps)
    if np.any(diffs <= 0):
        raise ConversionError(f"{source_path}: {source_key} timestamps are not strictly increasing")
    median_delta = float(np.median(diffs))

    units = ((1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns"))
    plausible = [
        (1.0 / (median_delta * scale), label)
        for scale, label in units
        if 1.0 <= 1.0 / (median_delta * scale) <= 240.0
    ]
    if not plausible:
        raise ConversionError(f"{source_path}: cannot infer timestamp unit/FPS from {source_key}")
    fps, unit = plausible[0]
    normalized_diffs = diffs * next(scale for scale, label in units if label == unit)
    jitter = float(np.percentile(np.abs(normalized_diffs - np.median(normalized_diffs)), 95))
    if jitter / float(np.median(normalized_diffs)) > TIMESTAMP_JITTER_TOLERANCE:
        raise ConversionError(
            f"{source_path}: {source_key} has unstable intervals (95th percentile jitter exceeds "
            f"{TIMESTAMP_JITTER_TOLERANCE:.0%})"
        )
    return fps, f"HDF5 timestamps {source_key} ({unit})"


def walk_for_fps(value: Any, camera_name_: str) -> float | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in FPS_ATTRIBUTE_NAMES:
                fps = as_finite_positive_float(nested)
                if fps is not None:
                    return fps
        for key, nested in value.items():
            if str(key).casefold() == camera_name_.casefold():
                fps = walk_for_fps(nested, camera_name_)
                if fps is not None:
                    return fps
        for nested in value.values():
            fps = walk_for_fps(nested, camera_name_)
            if fps is not None:
                return fps
    elif isinstance(value, list):
        for nested in value:
            fps = walk_for_fps(nested, camera_name_)
            if fps is not None:
                return fps
    return None


def fps_from_sidecar(raw_dataset_root: Path, episode_path: Path, camera_name_: str) -> tuple[float, str] | None:
    directories = tuple(dict.fromkeys([episode_path.parent, raw_dataset_root]))
    candidates: list[Path] = []
    for directory in directories:
        for suffix in ("*.json", "*.yaml", "*.yml"):
            candidates.extend(directory.glob(suffix))
    for path in sorted(set(candidates), key=natural_sort_key):
        if path.stat().st_size > 1_000_000:
            continue
        try:
            if path.suffix.casefold() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                import yaml

                data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        fps = walk_for_fps(data, camera_name_)
        if fps is not None:
            return fps, f"sidecar {path.relative_to(raw_dataset_root).as_posix()}"
    return None


def camera_fps(
    h5_file: Any,
    camera: CameraSpec,
    *,
    images_key: str,
    num_frames: int,
    source_path: Path,
    raw_dataset_root: Path,
    fallback_fps: float | None,
) -> FpsEvidence:
    h5py = require_h5py()
    for key in timestamp_candidates(camera, images_key):
        value = get_hdf5_object(h5_file, key)
        if value is None or not isinstance(value, h5py.Dataset):
            continue
        if value.ndim != 1 or value.shape[0] != num_frames:
            continue
        fps, source = fps_from_timestamps(value[:], source_path=source_path, source_key=key)
        return FpsEvidence(camera=camera.name, value=fps, source=source)

    camera_dataset = h5_file[camera.source_key]
    images_group = get_hdf5_object(h5_file, images_key)
    attr_evidence = fps_from_attrs([camera_dataset, camera_dataset.parent, images_group, h5_file["/"]])
    if attr_evidence is not None:
        fps, source = attr_evidence
        return FpsEvidence(camera=camera.name, value=fps, source=source)

    sidecar_evidence = fps_from_sidecar(raw_dataset_root, source_path, camera.name)
    if sidecar_evidence is not None:
        fps, source = sidecar_evidence
        return FpsEvidence(camera=camera.name, value=fps, source=source)
    if fallback_fps is not None:
        return FpsEvidence(camera=camera.name, value=fallback_fps, source="explicit --fps fallback")
    raise ConversionError(
        f"{source_path}: no FPS metadata found for camera {camera.name!r}; provide camera timestamps, "
        "an fps/frame_rate attribute, a JSON/YAML sidecar, or --fps"
    )


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)
