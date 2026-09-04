"""Evidence-preserving reader for the released RoboVerse v2 trajectory collection.

The Hugging Face release is not one table format.  It is a collection of
pickle/gzip-pickle files whose root maps robot names to episode lists.  Each
episode contains an action stream, an initial scene state, and optionally a
post-step state stream and source-specific metadata.  This reader deliberately
does not import RoboVerse/MetaSim: the official on-disk v2 contract is simple
Python containers and NumPy/Torch numeric values.

No timing or camera samples are stored in these trajectory files.  Timing is
therefore described as an ordinal step index by the conversion plan; callers
must opt in before writing the LeRobot-required integer ``fps`` field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from convert_core.errors import ConversionError


SOURCE_DATASET = "RoboVerseOrg/roboverse_data"
SOURCE_REVISION = "fab63ccaaed54f413901f86edc3fa1ab77a96500"
POSITION_NAMES = ("x", "y", "z")
QUATERNION_NAMES = ("w", "x", "y", "z")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class FeatureColumn:
    feature_key: str
    source_path: tuple[str, ...]
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] | None
    source_components: tuple[str, ...] | None = None
    source_component_indices: tuple[int, ...] | None = None
    split_named_mapping: bool = False
    source_component_shapes: tuple[tuple[int, ...], ...] | None = None
    split_reason: str | None = None
    source_dtype_options: tuple[tuple[str, ...], ...] | None = None

    def signature(self) -> tuple[Any, ...]:
        return (
            self.feature_key,
            self.source_path,
            self.dtype,
            self.shape,
            self.names,
            self.source_components,
            self.source_component_indices,
            self.split_named_mapping,
            self.source_component_shapes,
            self.split_reason,
            self.source_dtype_options,
        )


@dataclass(frozen=True)
class DtypeRun:
    start: int
    end_exclusive: int
    dtype: str


@dataclass(frozen=True)
class DtypePromotion:
    source_path: tuple[str, ...]
    source_component: str | None
    source_component_index: int | None
    source_shape: tuple[int, ...]
    target_dtype: str
    source_dtypes: tuple[str, ...]
    runs: tuple[DtypeRun, ...]


@dataclass(frozen=True)
class NumericStatistics:
    """Component-wise source statistics gathered without changing values."""

    frames: int
    finite_min: tuple[int | float | bool | None, ...]
    finite_max: tuple[int | float | bool | None, ...]
    nan_count: tuple[int, ...]
    positive_infinity_count: tuple[int, ...]
    negative_infinity_count: tuple[int, ...]


@dataclass(frozen=True)
class SourceEpisode:
    source_file: SourceFile
    robot_name: str
    source_episode_index: int
    episode_uid: str
    source_suite: str
    source_task: str
    task_origin: str
    source_split: str
    task_text: str
    action_count: int
    state_count: int
    action_columns: tuple[FeatureColumn, ...]
    state_columns: tuple[FeatureColumn, ...]
    state_empty_fields: tuple[tuple[str, str, str], ...]
    static_payload: dict[str, Any]
    action_dtype_promotions: tuple[DtypePromotion, ...] = ()
    state_dtype_promotions: tuple[DtypePromotion, ...] = ()


@dataclass(frozen=True)
class _ParsedEpisode:
    episode: SourceEpisode
    action_statistics: tuple[NumericStatistics, ...]
    state_statistics: tuple[NumericStatistics, ...]


@dataclass(frozen=True)
class PartPlan:
    part_id: str
    source_suite: str
    robot_name: str
    stream_kind: str
    feature_columns: tuple[FeatureColumn, ...]
    feature_statistics: tuple[NumericStatistics, ...]
    empty_state_fields: tuple[tuple[str, str, str], ...]
    episodes: tuple[SourceEpisode, ...]

    @property
    def num_frames(self) -> int:
        if self.stream_kind == "state":
            return sum(episode.state_count for episode in self.episodes)
        return sum(episode.action_count for episode in self.episodes)

    def episode_frames(self, episode: SourceEpisode) -> int:
        return episode.state_count if self.stream_kind == "state" else episode.action_count


@dataclass(frozen=True)
class CollectionPlan:
    source_root: Path
    source_revision: str
    files: tuple[SourceFile, ...]
    parts: tuple[PartPlan, ...]
    sidecars: tuple[str, ...]
    duplicate_aliases: tuple[tuple[str, str], ...]
    calvin_instruction_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_issues: tuple[dict[str, Any], ...] = ()
    sidecar_payloads: tuple[dict[str, Any], ...] = ()
    auxiliary_files: tuple[dict[str, Any], ...] = ()

    @property
    def num_source_episodes(self) -> int:
        return len({episode.episode_uid for part in self.parts for episode in part.episodes})

    @property
    def num_output_episodes(self) -> int:
        return sum(len(part.episodes) for part in self.parts)

    @property
    def num_output_frames(self) -> int:
        return sum(part.num_frames for part in self.parts)


def _load_file(path: Path) -> Any:
    try:
        if path.name.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as handle:
                return pickle.load(handle)
        if path.suffix == ".pkl":
            with path.open("rb") as handle:
                return pickle.load(handle)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConversionError(f"cannot read RoboVerse source {path}: {exc}") from exc
    raise ConversionError(f"unsupported RoboVerse trajectory file: {path}")


def load_source_file(path: Path) -> Any:
    """Public loading hook used by the writer and validation tests."""

    return _load_file(path)


def _candidate_paths(trajs_root: Path) -> tuple[list[Path], list[Path]]:
    candidates: list[Path] = []
    auxiliary: list[Path] = []
    # ``Path.rglob().is_file()`` performs a separate remote stat for every
    # entry on OSSFS.  os.walk consumes directory listings and only the chosen
    # trajectory candidates are stat'ed later for fingerprinting.
    for directory, _, filenames in os.walk(trajs_root):
        for filename in filenames:
            if filename.endswith("_v2.pkl.gz") or filename.endswith("_v2.pkl") or filename.endswith("_v2.json"):
                candidates.append(Path(directory) / filename)
            else:
                auxiliary.append(Path(directory) / filename)
    return (
        sorted(candidates, key=lambda item: item.as_posix().casefold()),
        sorted(auxiliary, key=lambda item: item.as_posix().casefold()),
    )


def _decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_compressed_aliases(paths: list[Path], root: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    by_uncompressed_name: dict[str, Path] = {}
    for path in paths:
        key = path.as_posix()[:-3] if path.name.endswith(".pkl.gz") else path.as_posix()
        if path.suffix == ".pkl":
            by_uncompressed_name[key] = path
    skipped: set[Path] = set()
    aliases: list[tuple[str, str]] = []
    for path in paths:
        if not path.name.endswith(".pkl.gz"):
            continue
        plain = by_uncompressed_name.get(path.as_posix()[:-3])
        if plain is None:
            continue
        if _decompressed_sha256(path) != _decompressed_sha256(plain):
            raise ConversionError(f"compressed/uncompressed RoboVerse pair differs: {plain} vs {path}")
        skipped.add(path)
        aliases.append((path.relative_to(root).as_posix(), plain.relative_to(root).as_posix()))
    return [path for path in paths if path not in skipped], aliases


def _raw_numpy_value(value: Any, description: str) -> np.ndarray:
    try:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
    except Exception as exc:
        raise ConversionError(f"cannot materialize {description} as a NumPy value: {exc}") from exc
    if array.dtype.kind not in "biuf":
        raise ConversionError(f"{description} is not a numeric scalar/array: dtype={array.dtype}")
    if array.size == 0:
        raise ConversionError(f"{description} is an empty numeric array with shape {array.shape}")
    return array


def _numpy_value(value: Any, description: str) -> np.ndarray:
    array = _raw_numpy_value(value, description)
    return array.reshape(1) if array.ndim == 0 else array


def _component_names(field_name: str, size: int) -> tuple[str, ...] | None:
    if field_name in {"pos", "vel", "ang_vel"} and size == 3:
        return POSITION_NAMES
    if field_name in {"rot", "quat", "quaternion"} and size == 4:
        return QUATERNION_NAMES
    # ``names=None`` is truthful for source arrays whose component semantics
    # are not declared. Numeric placeholders would look authoritative while
    # adding no source evidence.
    return None


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    if not normalized:
        raise ConversionError(f"cannot encode empty feature component from {value!r}")
    return normalized


@dataclass(frozen=True)
class _AtomicNumeric:
    source_path: tuple[str, ...]
    source_component: str | None
    source_component_index: int | None
    shape: tuple[int, ...]
    dtype: str
    array: np.ndarray = field(compare=False, repr=False)

    def structural_signature(self) -> tuple[Any, ...]:
        return (
            self.source_path,
            self.source_component,
            self.source_component_index,
            self.shape,
        )


PromotionKey = tuple[tuple[str, ...], str | None, int | None]
PromotionMap = dict[PromotionKey, DtypePromotion]


def _atomic_value(
    value: Any,
    *,
    source_path: tuple[str, ...],
    source_component: str | None,
    source_component_index: int | None,
    description: str,
) -> _AtomicNumeric:
    array = _raw_numpy_value(value, description)
    if source_component is None:
        array = array.reshape(1) if array.ndim == 0 else array
    elif array.size != 1:
        raise ConversionError(f"{description} named mapping must contain scalar values")
    return _AtomicNumeric(
        source_path,
        source_component,
        source_component_index,
        tuple(array.shape),
        str(array.dtype),
        array,
    )


def _mapping_atomic_values(
    value: Any,
    *,
    source_path: tuple[str, ...],
    description: str,
) -> list[_AtomicNumeric]:
    if isinstance(value, dict):
        if not value:
            raise ConversionError(f"{description} is empty")
        return [
            _atomic_value(
                item,
                source_path=source_path,
                source_component=str(name),
                source_component_index=index,
                description=f"{description}/{name}",
            )
            for index, (name, item) in enumerate(value.items())
        ]
    return [
        _atomic_value(
            value,
            source_path=source_path,
            source_component=None,
            source_component_index=None,
            description=description,
        )
    ]


def _action_atomic_values(step: Any) -> tuple[_AtomicNumeric, ...]:
    if not isinstance(step, dict):
        return tuple(
            _mapping_atomic_values(step, source_path=(), description="action")
        )
    result: list[_AtomicNumeric] = []
    for control_name, value in step.items():
        result.extend(
            _mapping_atomic_values(
                value,
                source_path=(str(control_name),),
                description=f"action/{control_name}",
            )
        )
    return tuple(result)


def _state_atomic_values(step: Any) -> tuple[_AtomicNumeric, ...]:
    if not isinstance(step, dict):
        raise ConversionError(f"state step must be a mapping, got {type(step).__name__}")
    result: list[_AtomicNumeric] = []
    for entity_name, entity in step.items():
        if not isinstance(entity, dict):
            raise ConversionError(f"state entity {entity_name!r} must be a mapping")
        for field_name, value in entity.items():
            if value is None or (isinstance(value, dict) and not value):
                continue
            result.extend(
                _mapping_atomic_values(
                    value,
                    source_path=(str(entity_name), str(field_name)),
                    description=f"state/{entity_name}/{field_name}",
                )
            )
    if not result:
        raise ConversionError("state step has no numeric fields")
    return tuple(result)


def _lossless_cast(array: np.ndarray, target_dtype: str, description: str) -> np.ndarray:
    source_dtype = str(array.dtype)
    try:
        cast = array.astype(np.dtype(target_dtype), casting="unsafe", copy=False)
        restored = cast.astype(array.dtype, casting="unsafe", copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConversionError(
            f"{description} cannot be promoted losslessly from {source_dtype} to {target_dtype}: {exc}"
        ) from exc
    if not np.array_equal(restored, array):
        raise ConversionError(
            f"{description} cannot be promoted losslessly from {source_dtype} to {target_dtype}"
        )
    return cast


def _infer_dtype_promotions(
    values: list[Any],
    atomic_builder: Callable[[Any], tuple[_AtomicNumeric, ...]],
    description: str,
    *,
    enabled: bool,
) -> tuple[DtypePromotion, ...]:
    if not enabled or not values:
        return ()
    first_atoms = atomic_builder(values[0])
    expected_structure = tuple(item.structural_signature() for item in first_atoms)
    dtype_order = [[atom.dtype] for atom in first_atoms]
    runs = [[DtypeRun(0, 1, atom.dtype)] for atom in first_atoms]
    first_changes: list[int | None] = [None] * len(first_atoms)
    for frame_index, value in enumerate(values[1:], start=1):
        atoms = atomic_builder(value)
        if tuple(item.structural_signature() for item in atoms) != expected_structure:
            raise ConversionError(f"{description} schema changes at frame {frame_index}")
        for component_index, atom in enumerate(atoms):
            if atom.dtype not in dtype_order[component_index]:
                dtype_order[component_index].append(atom.dtype)
            previous = runs[component_index][-1]
            if atom.dtype == previous.dtype:
                runs[component_index][-1] = DtypeRun(
                    previous.start,
                    frame_index + 1,
                    previous.dtype,
                )
            else:
                if first_changes[component_index] is None:
                    first_changes[component_index] = frame_index
                runs[component_index].append(
                    DtypeRun(frame_index, frame_index + 1, atom.dtype)
                )

    promotions: list[DtypePromotion] = []
    for component_index, first in enumerate(first_atoms):
        source_dtypes = tuple(dtype_order[component_index])
        if len(source_dtypes) == 1:
            continue
        floating = tuple(dtype for dtype in source_dtypes if np.dtype(dtype).kind == "f")
        nonfloating = tuple(dtype for dtype in source_dtypes if np.dtype(dtype).kind in "biu")
        first_change = first_changes[component_index]
        if len(floating) != 1 or len(floating) + len(nonfloating) != len(source_dtypes):
            raise ConversionError(
                f"{description} schema changes at frame {first_change}; dynamic dtype set "
                f"{list(source_dtypes)} has no single existing floating target"
            )
        target_dtype = floating[0]
        for frame_index, value in enumerate(values):
            atom = atomic_builder(value)[component_index]
            if atom.dtype != target_dtype:
                _lossless_cast(
                    atom.array,
                    target_dtype,
                    f"{description} frame {frame_index} component "
                    f"{atom.source_path}/{atom.source_component}",
                )
        promotions.append(
            DtypePromotion(
                first.source_path,
                first.source_component,
                first.source_component_index,
                first.shape,
                target_dtype,
                source_dtypes,
                tuple(runs[component_index]),
            )
        )
    return tuple(promotions)


def _promotion_map(promotions: tuple[DtypePromotion, ...]) -> PromotionMap:
    return {
        (item.source_path, item.source_component, item.source_component_index): item
        for item in promotions
    }


def dtype_promotion_dict(promotion: DtypePromotion) -> dict[str, Any]:
    return {
        "source_path": list(promotion.source_path),
        "source_component": promotion.source_component,
        "source_component_index": promotion.source_component_index,
        "source_shape": list(promotion.source_shape),
        "source_dtypes": list(promotion.source_dtypes),
        "target_dtype": promotion.target_dtype,
        "runs": [
            {
                "start": run.start,
                "end_exclusive": run.end_exclusive,
                "dtype": run.dtype,
            }
            for run in promotion.runs
        ],
        "numeric_values_exact": True,
        "episode_boundary_preserved": True,
        "lossy": False,
    }


def _mapping_columns(
    *,
    feature_key: str,
    source_path: tuple[str, ...],
    value: Any,
    description: str,
    dtype_promotions: PromotionMap | None = None,
) -> tuple[FeatureColumn, ...]:
    dtype_promotions = dtype_promotions or {}
    if isinstance(value, dict):
        names = tuple(str(name) for name in value)
        if not names:
            raise ConversionError(f"{description} is empty")
        arrays = [_raw_numpy_value(item, f"{description}/{name}") for name, item in value.items()]
        if any(array.size != 1 for array in arrays):
            raise ConversionError(f"{description} named mapping must contain scalar values")
        source_dtypes = tuple(str(array.dtype) for array in arrays)
        component_promotions = tuple(
            dtype_promotions.get((source_path, name, index))
            for index, name in enumerate(names)
        )
        dtypes = tuple(
            promotion.target_dtype if promotion is not None else dtype
            for dtype, promotion in zip(source_dtypes, component_promotions, strict=True)
        )
        dtype_options = tuple(
            promotion.source_dtypes if promotion is not None else (dtype,)
            for dtype, promotion in zip(source_dtypes, component_promotions, strict=True)
        )
        shapes = tuple(tuple(array.shape) for array in arrays)
        if len(set(dtypes)) != 1 or len(set(shapes)) != 1:
            if len(set(dtypes)) != 1 and len(set(shapes)) != 1:
                split_reason = "component_dtype_and_shape"
            elif len(set(dtypes)) != 1:
                split_reason = "component_dtype"
            else:
                split_reason = "component_shape"
            columns: list[FeatureColumn] = []
            used_keys: dict[str, str] = {}
            for index, (name, dtype, shape) in enumerate(
                zip(names, dtypes, shapes, strict=True)
            ):
                split_key = f"{feature_key}.{_safe_component(name)}"
                previous = used_keys.setdefault(split_key, name)
                if previous != name:
                    raise ConversionError(
                        f"mixed-dtype component collision: {previous!r} and {name!r} both map to {split_key}"
                    )
                columns.append(
                    FeatureColumn(
                        split_key,
                        source_path,
                        dtype,
                        (1, *shape),
                        (name,),
                        (name,),
                        (index,),
                        True,
                        (shape,),
                        split_reason,
                        (dtype_options[index],) if component_promotions[index] is not None else None,
                    )
                )
            return tuple(columns)
        component_shape = shapes[0]
        return (
            FeatureColumn(
                feature_key,
                source_path,
                dtypes[0],
                (len(arrays), *component_shape),
                names,
                names,
                tuple(range(len(names))),
                False,
                shapes,
                None,
                dtype_options if any(component_promotions) else None,
            ),
        )
    else:
        array = _numpy_value(value, description)
        if tuple(array.shape) == (0,):
            raise ConversionError(f"{description} is empty")
        promotion = dtype_promotions.get((source_path, None, None))
        return (
            FeatureColumn(
                feature_key,
                source_path,
                promotion.target_dtype if promotion is not None else str(array.dtype),
                tuple(array.shape),
                _component_names(source_path[-1], array.size) if array.ndim == 1 else None,
                source_dtype_options=(promotion.source_dtypes,) if promotion is not None else None,
            ),
        )


def _derive_array_action_names(episode: dict[str, Any], robot_name: str, width: int) -> tuple[str, ...]:
    for container_key in ("reset_state", "init_state"):
        container = episode.get(container_key)
        if isinstance(container, list) and container:
            container = container[0]
        if not isinstance(container, dict):
            continue
        if "robots" in container and isinstance(container["robots"], dict):
            robot = container["robots"].get(robot_name)
        else:
            robot = container.get(robot_name)
        if isinstance(robot, dict) and isinstance(robot.get("dof_pos"), dict):
            names = tuple(str(name) for name in robot["dof_pos"])
            if len(names) == width:
                return names
    raise ConversionError(
        f"array action for robot {robot_name!r} has width {width}, but no equally sized named dof_pos order exists"
    )


def action_columns(
    step: Any,
    episode: dict[str, Any],
    robot_name: str,
    dtype_promotions: PromotionMap | None = None,
) -> tuple[FeatureColumn, ...]:
    if isinstance(step, dict):
        columns: list[FeatureColumn] = []
        used_keys: dict[str, str] = {}
        for control_name, value in step.items():
            feature_key = "action" if len(step) == 1 else f"action.{_safe_component(str(control_name))}"
            mapped = _mapping_columns(
                feature_key=feature_key,
                source_path=(str(control_name),),
                value=value,
                description=f"action/{control_name}",
                dtype_promotions=dtype_promotions,
            )
            for column in mapped:
                previous = used_keys.setdefault(column.feature_key, str(control_name))
                if previous != str(control_name):
                    raise ConversionError(
                        f"action feature-key collision: {previous!r} and {control_name!r} "
                        f"both map to {column.feature_key}"
                    )
                columns.append(column)
        return tuple(columns)
    array = _numpy_value(step, "action")
    promotion = (dtype_promotions or {}).get(((), None, None))
    return (
        FeatureColumn(
            "action",
            (),
            promotion.target_dtype if promotion is not None else str(array.dtype),
            tuple(array.shape),
            _derive_array_action_names(episode, robot_name, array.size),
            source_dtype_options=(promotion.source_dtypes,) if promotion is not None else None,
        ),
    )


def state_columns(
    step: Any,
    dtype_promotions: PromotionMap | None = None,
) -> tuple[FeatureColumn, ...]:
    if not isinstance(step, dict):
        raise ConversionError(f"state step must be a mapping, got {type(step).__name__}")
    columns: list[FeatureColumn] = []
    used_keys: dict[str, tuple[str, str]] = {}
    for entity_name, entity in step.items():
        if not isinstance(entity, dict):
            raise ConversionError(f"state entity {entity_name!r} must be a mapping")
        for field_name, value in entity.items():
            # Empty per-entity DOF mappings are an explicit source value (for a
            # rigid object), not a numeric vector.  They remain in the static
            # source payload but do not create an impossible zero-width
            # LeRobot feature.
            if value is None or (isinstance(value, dict) and not value):
                continue
            encoded = f"observation.state.{_safe_component(str(entity_name))}.{_safe_component(str(field_name))}"
            source = (str(entity_name), str(field_name))
            previous = used_keys.setdefault(encoded, source)
            if previous != source:
                raise ConversionError(f"feature-key collision: {previous} and {source} both map to {encoded}")
            mapped = _mapping_columns(
                feature_key=encoded,
                source_path=source,
                value=value,
                description=f"state/{entity_name}/{field_name}",
                dtype_promotions=dtype_promotions,
            )
            for column in mapped:
                previous = used_keys.setdefault(column.feature_key, source)
                if previous != source:
                    raise ConversionError(
                        f"feature-key collision: {previous} and {source} both map to {column.feature_key}"
                    )
                columns.append(column)
    if not columns:
        raise ConversionError("state step has no numeric fields")
    return tuple(columns)


def state_empty_fields(step: Any) -> tuple[tuple[str, str, str], ...]:
    """Return explicitly present state fields that carry no numeric samples."""

    if not isinstance(step, dict):
        raise ConversionError(f"state step must be a mapping, got {type(step).__name__}")
    result: list[tuple[str, str, str]] = []
    for entity_name, entity in step.items():
        if not isinstance(entity, dict):
            raise ConversionError(f"state entity {entity_name!r} must be a mapping")
        for field_name, value in entity.items():
            if value is None:
                result.append((str(entity_name), str(field_name), "null"))
            elif isinstance(value, dict) and not value:
                result.append((str(entity_name), str(field_name), "empty_mapping"))
    return tuple(result)


def _schema_signature(columns: tuple[FeatureColumn, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(column.signature() for column in columns)


def merge_numeric_statistics(
    left: NumericStatistics,
    right: NumericStatistics,
) -> NumericStatistics:
    """Merge exact component-wise scan statistics for the same feature."""

    if len(left.finite_min) != len(right.finite_min):
        raise ConversionError("cannot merge numeric statistics with different widths")

    def merged_min(a: int | float | bool | None, b: int | float | bool | None):
        if a is None:
            return b
        if b is None:
            return a
        return a if a <= b else b

    def merged_max(a: int | float | bool | None, b: int | float | bool | None):
        if a is None:
            return b
        if b is None:
            return a
        return a if a >= b else b

    return NumericStatistics(
        frames=left.frames + right.frames,
        finite_min=tuple(
            merged_min(a, b)
            for a, b in zip(left.finite_min, right.finite_min, strict=True)
        ),
        finite_max=tuple(
            merged_max(a, b)
            for a, b in zip(left.finite_max, right.finite_max, strict=True)
        ),
        nan_count=tuple(a + b for a, b in zip(left.nan_count, right.nan_count, strict=True)),
        positive_infinity_count=tuple(
            a + b
            for a, b in zip(
                left.positive_infinity_count,
                right.positive_infinity_count,
                strict=True,
            )
        ),
        negative_infinity_count=tuple(
            a + b
            for a, b in zip(
                left.negative_infinity_count,
                right.negative_infinity_count,
                strict=True,
            )
        ),
    )


def numeric_statistics_dict(statistics: NumericStatistics) -> dict[str, Any]:
    return {
        "frames": statistics.frames,
        "finite_min": list(statistics.finite_min),
        "finite_max": list(statistics.finite_max),
        "nan_count": list(statistics.nan_count),
        "positive_infinity_count": list(statistics.positive_infinity_count),
        "negative_infinity_count": list(statistics.negative_infinity_count),
    }


def _validate_stream_schema(
    values: list[Any],
    expected: tuple[FeatureColumn, ...],
    builder: Any,
    description: str,
    *builder_args: Any,
) -> tuple[NumericStatistics, ...]:
    signature = _schema_signature(expected)
    widths = [int(math.prod(column.shape)) for column in expected]
    minimums: list[list[int | float | bool | None]] = [[None] * width for width in widths]
    maximums: list[list[int | float | bool | None]] = [[None] * width for width in widths]
    nan_counts = [[0] * width for width in widths]
    positive_infinity_counts = [[0] * width for width in widths]
    negative_infinity_counts = [[0] * width for width in widths]
    for index, value in enumerate(values):
        actual = builder(value, *builder_args)
        if _schema_signature(actual) != signature:
            raise ConversionError(f"{description} schema changes at frame {index}")
        for column_index, column in enumerate(expected):
            array = column_array(value, column).reshape(-1)
            for component_index, scalar in enumerate(array):
                if array.dtype.kind == "f":
                    if bool(np.isnan(scalar)):
                        nan_counts[column_index][component_index] += 1
                        continue
                    if bool(np.isposinf(scalar)):
                        positive_infinity_counts[column_index][component_index] += 1
                        continue
                    if bool(np.isneginf(scalar)):
                        negative_infinity_counts[column_index][component_index] += 1
                        continue
                item = scalar.item()
                current_min = minimums[column_index][component_index]
                current_max = maximums[column_index][component_index]
                if current_min is None or item < current_min:
                    minimums[column_index][component_index] = item
                if current_max is None or item > current_max:
                    maximums[column_index][component_index] = item
    return tuple(
        NumericStatistics(
            frames=len(values),
            finite_min=tuple(minimums[index]),
            finite_max=tuple(maximums[index]),
            nan_count=tuple(nan_counts[index]),
            positive_infinity_count=tuple(positive_infinity_counts[index]),
            negative_infinity_count=tuple(negative_infinity_counts[index]),
        )
        for index in range(len(expected))
    )


def _jsonable(value: Any) -> Any:
    """Encode static source metadata without losing numeric type/shape."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            array = value.detach().cpu().numpy()
        except Exception as exc:
            raise ConversionError(f"cannot preserve tensor metadata: {exc}") from exc
        return {"__array__": array.tolist(), "dtype": f"torch.{str(array.dtype)}", "shape": list(array.shape)}
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist(), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return {"__scalar__": _jsonable(value.item()), "dtype": str(value.dtype)}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ConversionError(f"unsupported static metadata type {type(value).__name__}")


def jsonable_static_payload(value: Any) -> Any:
    return _jsonable(value)


def _source_suite(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) < 2 or parts[0] != "trajs":
        raise ConversionError(f"unexpected RoboVerse source path: {relative_path}")
    return parts[1]


def _task_and_split(
    relative_path: str,
    root_metadata: dict[str, Any],
    episode: dict[str, Any],
) -> tuple[str, str, str]:
    parts = Path(relative_path).parts
    suite = parts[1]
    if isinstance(episode.get("extra"), dict) and episode["extra"].get("task_name"):
        task = str(episode["extra"]["task_name"])
        task_origin = "episode.extra.task_name"
    elif root_metadata.get("task_name"):
        task = str(root_metadata["task_name"])
        task_origin = "source_file.metadata.task_name"
    elif suite == "calvin" and len(parts) > 3 and parts[2] == "calvin_traj_ann":
        task = f"calvin/{parts[3]}/{Path(relative_path).name.removesuffix('.pkl')}"
        task_origin = "official_release_path_identifier_not_natural_language"
    elif len(parts) > 2:
        task = f"{suite}/{parts[2]}"
        task_origin = "official_release_path_identifier_not_natural_language"
    else:
        task = suite
        task_origin = "official_release_path_identifier_not_natural_language"
    if suite == "calvin" and "env_D_val_out" in parts:
        split = "validation"
    elif suite == "calvin" and any(name in parts for name in ("env_A_out", "env_B_out", "env_C_out", "env_D_out")):
        split = "train"
    else:
        split = "unspecified"
    return task, split, task_origin


def _episode_static_payload(episode: dict[str, Any], source_file_payload: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in episode.items() if key not in {"actions", "states"}}
    if source_file_payload:
        payload["source_file_payload"] = source_file_payload
    return _jsonable(payload)


def _source_file(path: Path, root: Path) -> SourceFile:
    stat = path.stat()
    return SourceFile(path, path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns)


def _episodes_from_file(
    source: SourceFile,
    data: Any,
    *,
    max_episodes: int | None = None,
    allow_lossless_dtype_promotion: bool = False,
) -> tuple[list[_ParsedEpisode], bool]:
    if not isinstance(data, dict):
        if source.path.suffix == ".json":
            return [], True
        raise ConversionError(f"{source.relative_path} root must be a mapping")
    root_metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    robot_entries = [
        (key, value)
        for key, value in data.items()
        if isinstance(key, str)
        and key != "metadata"
        and isinstance(value, list)
        and value
        and all(isinstance(item, dict) for item in value)
    ]
    if not robot_entries:
        if source.path.suffix == ".json":
            return [], True
        raise ConversionError(f"{source.relative_path} has no non-empty robot episode lists")
    if source.path.suffix == ".json" and all(
        not (isinstance(episode.get("actions"), list) and episode["actions"])
        and not (isinstance(episode.get("states"), list) and episode["states"])
        for _, episodes in robot_entries
        for episode in episodes
    ):
        # Several official initial-state payloads use the same robot -> list
        # envelope as trajectory files, and one debug payload is named
        # ``franka_v2.json`` rather than ``initial_state_v2.json``.  Their
        # robot-shaped envelope must not make static source data look like an
        # invalid zero-frame trajectory.
        return [], True
    robot_keys = {name for name, _ in robot_entries}
    source_file_payload = {
        key: value for key, value in data.items() if str(key) not in robot_keys
    }
    result: list[_ParsedEpisode] = []
    suite = _source_suite(source.relative_path)
    for robot_name, episodes in robot_entries:
        for episode_index, episode in enumerate(episodes):
            if max_episodes is not None and len(result) >= max_episodes:
                return result, False
            actions = episode.get("actions")
            states = episode.get("states")
            action_values = actions if isinstance(actions, list) else []
            state_values = states if isinstance(states, list) else []
            if not action_values and not state_values:
                raise ConversionError(
                    f"{source.relative_path} robot={robot_name} episode={episode_index} has no action or state frames"
                )
            promotion_description = (
                f"{source.relative_path} robot={robot_name} episode={episode_index}"
            )
            action_promotions: tuple[DtypePromotion, ...] = ()
            state_promotions: tuple[DtypePromotion, ...] = ()
            action_spec = action_columns(action_values[0], episode, robot_name) if action_values else ()
            state_spec = state_columns(state_values[0]) if state_values else ()
            empty_state_spec = state_empty_fields(state_values[0]) if state_values else ()
            action_statistics: tuple[NumericStatistics, ...] = ()
            if action_values:
                try:
                    action_statistics = _validate_stream_schema(
                        action_values,
                        action_spec,
                        action_columns,
                        f"{promotion_description} action",
                        episode,
                        robot_name,
                    )
                except ConversionError:
                    if not allow_lossless_dtype_promotion:
                        raise
                    action_promotions = _infer_dtype_promotions(
                        action_values,
                        _action_atomic_values,
                        f"{promotion_description} action",
                        enabled=True,
                    )
                    if not action_promotions:
                        raise
                    action_promotion_map = _promotion_map(action_promotions)
                    action_spec = action_columns(
                        action_values[0],
                        episode,
                        robot_name,
                        action_promotion_map,
                    )
                    action_statistics = _validate_stream_schema(
                        action_values,
                        action_spec,
                        action_columns,
                        f"{promotion_description} action",
                        episode,
                        robot_name,
                        action_promotion_map,
                    )
            state_statistics: tuple[NumericStatistics, ...] = ()
            if state_values:
                try:
                    state_statistics = _validate_stream_schema(
                        state_values,
                        state_spec,
                        state_columns,
                        f"{promotion_description} state",
                    )
                except ConversionError:
                    if not allow_lossless_dtype_promotion:
                        raise
                    state_promotions = _infer_dtype_promotions(
                        state_values,
                        _state_atomic_values,
                        f"{promotion_description} state",
                        enabled=True,
                    )
                    if not state_promotions:
                        raise
                    state_promotion_map = _promotion_map(state_promotions)
                    state_spec = state_columns(state_values[0], state_promotion_map)
                    state_statistics = _validate_stream_schema(
                        state_values,
                        state_spec,
                        state_columns,
                        f"{promotion_description} state",
                        state_promotion_map,
                    )
            if state_values:
                for frame_index, state in enumerate(state_values):
                    if state_empty_fields(state) != empty_state_spec:
                        raise ConversionError(
                            f"{source.relative_path} robot={robot_name} episode={episode_index} "
                            f"empty state-field schema changes at frame {frame_index}"
                        )
            task, split, task_origin = _task_and_split(source.relative_path, root_metadata, episode)
            uid_seed = f"{source.relative_path}\0{robot_name}\0{episode_index}".encode()
            uid = hashlib.sha256(uid_seed).hexdigest()[:20]
            result.append(
                _ParsedEpisode(
                    SourceEpisode(
                        source,
                        robot_name,
                        episode_index,
                        uid,
                        suite,
                        task,
                        task_origin,
                        split,
                        task,
                        len(action_values),
                        len(state_values),
                        action_spec,
                        state_spec,
                        empty_state_spec,
                        _episode_static_payload(episode, source_file_payload),
                        action_promotions,
                        state_promotions,
                    ),
                    action_statistics,
                    state_statistics,
                )
            )
    return result, False


def _part_id(source: SourceFile, robot_name: str, stream_kind: str, signature: tuple[Any, ...]) -> str:
    suite = _source_suite(source.relative_path)
    stem = Path(source.relative_path).name
    for suffix in (".pkl.gz", ".pkl", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    human = "-".join((_safe_component(suite), _safe_component(robot_name), _safe_component(stream_kind), _safe_component(stem)))
    human = human[:100].rstrip("-")
    digest = hashlib.sha256(
        repr((source.relative_path, robot_name, stream_kind, signature)).encode()
    ).hexdigest()[:12]
    return f"{human}-{digest}"


def _part_group_key(
    episode: SourceEpisode,
    stream_kind: str,
    columns: tuple[FeatureColumn, ...],
    empty_fields: tuple[tuple[str, str, str], ...],
) -> tuple[Any, ...]:
    return (
        episode.source_file.relative_path,
        episode.robot_name,
        stream_kind,
        _schema_signature(columns),
        empty_fields,
    )


def _parsed_streams(parsed: _ParsedEpisode) -> tuple[
    tuple[
        str,
        tuple[FeatureColumn, ...],
        tuple[NumericStatistics, ...],
        tuple[tuple[str, str, str], ...],
    ],
    ...,
]:
    episode = parsed.episode
    if episode.action_count and episode.state_count and episode.action_count == episode.state_count:
        return (
            (
                "aligned",
                episode.action_columns + episode.state_columns,
                parsed.action_statistics + parsed.state_statistics,
                episode.state_empty_fields,
            ),
        )
    streams = []
    if episode.action_count:
        streams.append(("action", episode.action_columns, parsed.action_statistics, ()))
    if episode.state_count:
        streams.append(
            (
                "state",
                episode.state_columns,
                parsed.state_statistics,
                episode.state_empty_fields,
            )
        )
    return tuple(streams)


def _make_parts(
    episodes: Iterable[SourceEpisode],
    statistics_by_key: dict[tuple[Any, ...], tuple[NumericStatistics, ...]],
) -> list[PartPlan]:
    grouped: dict[tuple[Any, ...], list[SourceEpisode]] = {}
    columns_by_key: dict[tuple[Any, ...], tuple[FeatureColumn, ...]] = {}
    empty_fields_by_key: dict[tuple[Any, ...], tuple[tuple[str, str, str], ...]] = {}
    for episode in episodes:
        if episode.action_count and episode.state_count and episode.action_count == episode.state_count:
            streams = (("aligned", episode.action_columns + episode.state_columns, episode.state_empty_fields),)
        else:
            streams_list: list[
                tuple[str, tuple[FeatureColumn, ...], tuple[tuple[str, str, str], ...]]
            ] = []
            if episode.action_count:
                streams_list.append(("action", episode.action_columns, ()))
            if episode.state_count:
                streams_list.append(("state", episode.state_columns, episode.state_empty_fields))
            streams = tuple(streams_list)
        for stream_kind, columns, empty_fields in streams:
            key = _part_group_key(episode, stream_kind, columns, empty_fields)
            grouped.setdefault(key, []).append(episode)
            columns_by_key[key] = columns
            empty_fields_by_key[key] = empty_fields
    parts: list[PartPlan] = []
    seen_ids: set[str] = set()
    for key in sorted(grouped, key=lambda item: repr(item).casefold()):
        source_path, robot_name, stream_kind, signature, empty_fields = key
        source = grouped[key][0].source_file
        part_id = _part_id(source, robot_name, stream_kind, (signature, empty_fields))
        if part_id in seen_ids:
            raise ConversionError(f"RoboVerse part id collision: {part_id}")
        seen_ids.add(part_id)
        parts.append(
            PartPlan(
                part_id,
                grouped[key][0].source_suite,
                robot_name,
                stream_kind,
                columns_by_key[key],
                statistics_by_key[key],
                empty_fields_by_key[key],
                tuple(grouped[key]),
            )
        )
    return parts


def _calvin_aliases(root: Path) -> dict[str, tuple[str, ...]]:
    path = root / "trajs" / "calvin" / "calvin_traj_ann" / "ann_dict.npy"
    if not path.is_file():
        return {}
    try:
        raw = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise ConversionError(f"cannot read CALVIN annotation dictionary {path}: {exc}") from exc
    grouped: dict[str, list[str]] = {}
    for instruction, task_index in raw.items():
        grouped.setdefault(str(int(task_index)), []).append(str(instruction))
    return {key: tuple(values) for key, values in sorted(grouped.items(), key=lambda item: int(item[0]))}


def inspect_collection(
    source_root: Path,
    *,
    suites: set[str] | None = None,
    source_paths: set[str] | None = None,
    tasks: set[str] | None = None,
    max_source_files: int | None = None,
    max_episodes: int | None = None,
    allow_lossless_dtype_promotion: bool = False,
    progress_callback: Callable[[int, int, str, int], None] | None = None,
) -> CollectionPlan:
    source_root = source_root.expanduser().resolve()
    trajs_root = source_root / "trajs"
    if not trajs_root.is_dir():
        raise ConversionError(f"RoboVerse trajs directory does not exist: {trajs_root}")
    if source_paths:
        normalized = {Path(value).as_posix() for value in source_paths}
        candidates = []
        missing: list[str] = []
        for relative in sorted(normalized, key=str.casefold):
            candidate = (source_root / relative).resolve()
            if not candidate.is_relative_to(source_root):
                raise ConversionError(f"requested source path escapes the RoboVerse root: {relative}")
            if not candidate.is_file():
                missing.append(relative)
            elif not (
                candidate.name.endswith("_v2.pkl.gz")
                or candidate.name.endswith("_v2.pkl")
                or candidate.name.endswith("_v2.json")
            ):
                raise ConversionError(f"requested path is not a RoboVerse v2 source file: {relative}")
            else:
                candidates.append(candidate)
        auxiliary_paths: list[Path] = []
        if missing:
            raise ConversionError(f"requested RoboVerse source files were not found: {missing}")
    else:
        candidates, auxiliary_paths = _candidate_paths(trajs_root)
    if suites:
        candidates = [path for path in candidates if _source_suite(path.relative_to(source_root).as_posix()) in suites]
    candidates, duplicate_aliases = _deduplicate_compressed_aliases(candidates, source_root)
    if max_source_files is not None:
        if max_source_files <= 0:
            raise ConversionError("max_source_files must be positive")
        candidates = candidates[:max_source_files]
    source_files: list[SourceFile] = []
    auxiliary_files: list[dict[str, Any]] = []
    for path in auxiliary_paths:
        stat = path.stat()
        relative = path.relative_to(source_root).as_posix()
        auxiliary_files.append(
            {
                "relative_path": relative,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "role": (
                    "calvin_instruction_aliases"
                    if relative == "trajs/calvin/calvin_traj_ann/ann_dict.npy"
                    else "not_consumed_by_trajectory_converter"
                ),
            }
        )
    episodes: list[SourceEpisode] = []
    statistics_by_key: dict[tuple[Any, ...], tuple[NumericStatistics, ...]] = {}
    sidecars: list[str] = []
    sidecar_payloads: list[dict[str, Any]] = []
    source_issues: list[dict[str, Any]] = []
    for file_index, path in enumerate(candidates):
        source = _source_file(path, source_root)
        source_files.append(source)
        if source.size == 0 and path.name.startswith("initial_state_v2") and path.suffix == ".json":
            sidecars.append(source.relative_path)
            sidecar_payloads.append({"relative_path": source.relative_path, "payload": None})
            source_issues.append(
                {
                    "relative_path": source.relative_path,
                    "kind": "empty_source_sidecar",
                    "detail": "official initial_state_v2.json is zero bytes and is not a trajectory episode",
                    "blocking": False,
                    "local_size": source.size,
                }
            )
            if progress_callback is not None:
                progress_callback(file_index + 1, len(candidates), source.relative_path, len(episodes))
            continue
        try:
            loaded = _load_file(path)
            # A task may come from file/episode metadata, so task selection is
            # applied after parsing.  Do not stop early inside a source file in
            # that case or later matching episodes could be missed.
            remaining = (
                None
                if tasks or max_episodes is None
                else max_episodes - len(episodes)
            )
            parsed, is_sidecar = _episodes_from_file(
                source,
                loaded,
                max_episodes=remaining,
                allow_lossless_dtype_promotion=allow_lossless_dtype_promotion,
            )
        except ConversionError as exc:
            is_declared_sidecar = path.name.startswith("initial_state_v2") and path.suffix == ".json"
            if is_declared_sidecar:
                sidecars.append(source.relative_path)
                sidecar_payloads.append({"relative_path": source.relative_path, "payload": None})
            source_issues.append(
                {
                    "relative_path": source.relative_path,
                    "kind": (
                        "invalid_source_sidecar"
                        if is_declared_sidecar
                        else "unreadable_or_invalid_trajectory"
                    ),
                    "detail": str(exc),
                    "blocking": not is_declared_sidecar,
                    "local_size": source.size,
                }
            )
            if progress_callback is not None:
                progress_callback(file_index + 1, len(candidates), source.relative_path, len(episodes))
            continue
        if is_sidecar:
            sidecars.append(source.relative_path)
            sidecar_payloads.append(
                {"relative_path": source.relative_path, "payload": _jsonable(loaded)}
            )
            if progress_callback is not None:
                progress_callback(file_index + 1, len(candidates), source.relative_path, len(episodes))
            continue
        for parsed_episode in parsed:
            if tasks and (
                parsed_episode.episode.source_task not in tasks
                and parsed_episode.episode.task_text not in tasks
            ):
                continue
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            episode = parsed_episode.episode
            episodes.append(episode)
            for stream_kind, columns, statistics, empty_fields in _parsed_streams(parsed_episode):
                key = _part_group_key(episode, stream_kind, columns, empty_fields)
                existing = statistics_by_key.get(key)
                if existing is None:
                    statistics_by_key[key] = statistics
                else:
                    statistics_by_key[key] = tuple(
                        merge_numeric_statistics(left, right)
                        for left, right in zip(existing, statistics, strict=True)
                    )
        if progress_callback is not None:
            progress_callback(file_index + 1, len(candidates), source.relative_path, len(episodes))
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        blocking_details = [
            str(issue["detail"]) for issue in source_issues if issue.get("blocking")
        ]
        if blocking_details:
            raise ConversionError(
                "no valid RoboVerse trajectory episodes matched the selection; "
                + "; ".join(blocking_details)
            )
        raise ConversionError("no RoboVerse trajectory episodes matched the selection")
    include_calvin = any(episode.source_suite == "calvin" for episode in episodes)
    return CollectionPlan(
        source_root,
        SOURCE_REVISION,
        tuple(source_files),
        tuple(_make_parts(episodes, statistics_by_key)),
        tuple(sidecars),
        tuple(duplicate_aliases),
        _calvin_aliases(source_root) if include_calvin else {},
        tuple(source_issues),
        tuple(sidecar_payloads),
        tuple(auxiliary_files),
    )


def _get_source_value(container: Any, source_path: tuple[str, ...], description: str) -> Any:
    value = container
    for component in source_path:
        if not isinstance(value, dict) or component not in value:
            raise ConversionError(f"missing {description} component {component!r}")
        value = value[component]
    return value


def column_array(container: Any, column: FeatureColumn) -> np.ndarray:
    value = _get_source_value(container, column.source_path, column.feature_key) if column.source_path else container
    if isinstance(value, dict):
        if column.source_components is None:
            raise ConversionError(f"{column.feature_key} has no recorded source component order")
        if column.source_dtype_options is None:
            value = [value[name] for name in column.source_components]
            array = _numpy_value(value, column.feature_key)
        else:
            if len(column.source_dtype_options) != len(column.source_components):
                raise ConversionError(
                    f"{column.feature_key} dtype provenance does not match its source components"
                )
            arrays: list[np.ndarray] = []
            for index, (name, allowed_dtypes) in enumerate(
                zip(
                    column.source_components,
                    column.source_dtype_options,
                    strict=True,
                )
            ):
                item = _raw_numpy_value(value[name], f"{column.feature_key}/{name}")
                expected_shape = (
                    column.source_component_shapes[index]
                    if column.source_component_shapes is not None
                    else tuple(item.shape)
                )
                if tuple(item.shape) != expected_shape or str(item.dtype) not in allowed_dtypes:
                    raise ConversionError(
                        f"{column.feature_key}/{name} changed outside recorded dtype/shape provenance: "
                        f"{item.dtype}{item.shape}"
                    )
                arrays.append(
                    item
                    if str(item.dtype) == column.dtype
                    else _lossless_cast(item, column.dtype, f"{column.feature_key}/{name}")
                )
            array = np.stack(arrays, axis=0)
    else:
        array = _numpy_value(value, column.feature_key)
        if column.source_dtype_options is not None:
            if len(column.source_dtype_options) != 1 or str(array.dtype) not in column.source_dtype_options[0]:
                raise ConversionError(
                    f"{column.feature_key} changed outside recorded dtype provenance: {array.dtype}"
                )
            if str(array.dtype) != column.dtype:
                array = _lossless_cast(array, column.dtype, column.feature_key)
    if str(array.dtype) != column.dtype or tuple(array.shape) != column.shape:
        raise ConversionError(
            f"{column.feature_key} changed from {column.dtype}{column.shape} to {array.dtype}{array.shape}"
        )
    return array


def iter_part_frames(part: PartPlan, episode: SourceEpisode) -> Iterator[dict[str, Any]]:
    data = _load_file(episode.source_file.path)
    source_episode = data[episode.robot_name][episode.source_episode_index]
    if part.stream_kind == "state":
        stream = source_episode["states"]
    else:
        stream = source_episode["actions"]
    for index, primary in enumerate(stream):
        frame: dict[str, Any] = {"task": episode.task_text}
        if part.stream_kind == "aligned":
            state = source_episode["states"][index]
            for column in part.feature_columns:
                container = primary if column.feature_key.startswith("action") else state
                frame[column.feature_key] = column_array(container, column)
        else:
            for column in part.feature_columns:
                frame[column.feature_key] = column_array(primary, column)
        yield frame
