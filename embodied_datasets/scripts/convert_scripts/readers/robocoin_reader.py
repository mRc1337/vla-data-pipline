"""Lazy, task-scoped inspection for the RoboCOIN LeRobot-v2.1 release.

The top-level RoboCOIN directories are the publisher's stable categories.
Catalog construction reads only their small ``meta/info.json`` and
``meta/tasks.jsonl`` files.  Expensive inventory/footer/media inspection is
deferred until the corresponding task is about to be converted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from convert_core.checkpoint import canonical_fingerprint
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError
from readers.robogene_reader import (
    RobogeneCatalog,
    RobogeneEpisodeSource,
    RobogenePartition,
    _EPISODE_RE,
    _VIDEO_RE,
    _episode_index,
    _feature_specs,
    _file_inventory,
    _inspect_payload_samples,
    _read_json,
    _read_jsonl,
    _task_rows,
)


ROBOCOIN_CONVERSION_POLICY_VERSION = 2
_AGILEX_ROBOT_TYPE = "aloha"
_AGILEX_COMMON_FLOAT64_FIELDS = (
    "action",
    "observation.state",
    "gripper_open_scale_state",
    "gripper_open_scale_action",
)
_PUBLISHER_VIDEO_ALIASES = {
    "observation.images.cam_high_rgb": "observation.images.cam_head_rgb",
    "observation.images.cam_third_view": "observation.images.cam_front_rgb",
}


def _bytes_metadata_payload(metadata: dict[bytes, bytes] | None) -> list[list[str]]:
    return [
        [key.hex(), value.hex()]
        for key, value in sorted((metadata or {}).items())
    ]


def _arrow_type_payload(data_type: Any, *, float_scalar_override: str | None = None) -> dict[str, Any]:
    """Describe nested Arrow types without volatile schema-level metadata."""

    import pyarrow as pa

    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return {
            "kind": "large_list" if pa.types.is_large_list(data_type) else "list",
            "value_field": _arrow_field_payload(
                data_type.value_field,
                float_scalar_override=float_scalar_override,
            ),
        }
    if pa.types.is_fixed_size_list(data_type):
        return {
            "kind": "fixed_size_list",
            "list_size": data_type.list_size,
            "value_field": _arrow_field_payload(
                data_type.value_field,
                float_scalar_override=float_scalar_override,
            ),
        }
    if float_scalar_override is not None and (
        pa.types.is_float32(data_type) or pa.types.is_float64(data_type)
    ):
        return {"kind": float_scalar_override}
    return {"kind": str(data_type)}


def _arrow_field_payload(field: Any, *, float_scalar_override: str | None = None) -> dict[str, Any]:
    return {
        "name": field.name,
        "nullable": field.nullable,
        "metadata": _bytes_metadata_payload(field.metadata),
        "type": _arrow_type_payload(
            field.type,
            float_scalar_override=float_scalar_override,
        ),
    }


def _semantic_schema_payload(
    schema: Any,
    *,
    float64_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return the compatibility schema, excluding top-level writer metadata.

    RoboCOIN's sole top-level Arrow metadata key is ``pandas``.  Its
    ``index_columns[].stop`` varies (and is sometimes stale) per episode, so it
    is audit evidence rather than a feature-schema compatibility constraint.
    Field order, recursive types, nullability, and field metadata remain part
    of this signature.
    """

    import pyarrow as pa

    promoted = set(float64_fields)
    fields = []
    for field in schema:
        override = None
        if field.name in promoted:
            if not (pa.types.is_list(field.type) or pa.types.is_large_list(field.type)):
                raise ConversionError(
                    f"RoboCOIN promoted field {field.name!r} is not a variable list: {field.type}"
                )
            value_type = field.type.value_type
            if not (pa.types.is_float32(value_type) or pa.types.is_float64(value_type)):
                raise ConversionError(
                    f"RoboCOIN promoted field {field.name!r} is not float32/float64: {field.type}"
                )
            override = "double"
        fields.append(_arrow_field_payload(field, float_scalar_override=override))
    missing = promoted - set(schema.names)
    if missing:
        raise ConversionError(f"RoboCOIN promoted fields are absent from Parquet: {sorted(missing)}")
    return {"fields": fields}


def _episode_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ordered = sorted(indices)
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def _conversion_schema_policy(
    task_key: str,
    info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return effective output metadata plus an auditable conversion policy."""

    effective = json.loads(json.dumps(info))
    fields: dict[str, Any] = {}
    if str(info.get("robot_type")) == _AGILEX_ROBOT_TYPE:
        features = effective.get("features")
        if not isinstance(features, dict):
            raise ConversionError(f"RoboCOIN Agilex task has no feature metadata: {task_key}")
        for key in _AGILEX_COMMON_FLOAT64_FIELDS:
            feature = features.get(key)
            if not isinstance(feature, dict) or feature.get("dtype") != "float32":
                raise ConversionError(
                    f"RoboCOIN Agilex field policy expected declared float32 for {task_key}:{key}"
                )
            feature["dtype"] = "float64"
            fields[key] = {
                "declared_dtype": "float32",
                "accepted_source_physical_dtypes": ["float32", "float64"],
                "output_dtype": "float64",
                "conversion": "lossless common-supertype promotion",
                "lossy": False,
            }
    return effective, {
        "version": ROBOCOIN_CONVERSION_POLICY_VERSION,
        "semantic_schema_signature": {
            "includes": [
                "field order",
                "recursive Arrow types",
                "nullability",
                "field metadata",
            ],
            "ignores": ["top-level pandas schema metadata"],
        },
        "dtype_resolution": fields,
        "rationale": (
            "publisher declares float32 while source Parquet stores selected fields as "
            "float64 (with mixed float32/float64 action and observation.state); output "
            "uses the lossless common supertype float64"
            if fields
            else "no dtype promotion required"
        ),
    }


def _sample_identity(path: Path, sample_bytes: int = 64 * 1024) -> tuple[int, tuple[str, ...]]:
    """Return size plus first/middle/last digests without hashing a whole video."""

    size = path.stat().st_size
    length = min(size, sample_bytes)
    offsets = sorted({0, max(0, (size - length) // 2), max(0, size - length)})
    digests = []
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            digests.append(hashlib.sha256(stream.read(length)).hexdigest())
    return size, tuple(digests)


@dataclass(frozen=True)
class RobocoinTaskCatalogEntry:
    key: str
    root: Path
    info: dict[str, Any]
    instructions: tuple[str, ...]
    info_schema_fingerprint: str
    partition: str
    partition_unit_index: int
    partition_episode_start: int
    partition_frame_start: int
    root_episode_start: int
    root_frame_start: int
    info_record: dict[str, Any]
    tasks_record: dict[str, Any]
    episodes_record: dict[str, Any] | None
    selected_episodes: int
    selected_frames: int

    @property
    def total_episodes(self) -> int:
        return self.selected_episodes

    @property
    def total_frames(self) -> int:
        return self.selected_frames


@dataclass(frozen=True)
class RobocoinCatalog:
    tasks: tuple[RobocoinTaskCatalogEntry, ...]
    fingerprint_payload: dict[str, Any]


def _small_file_record(path: Path, raw_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(raw_root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _validate_task_name(name: str) -> str:
    path = Path(name)
    if not name or path.is_absolute() or path.name != name or ".." in path.parts:
        raise ConversionError(f"invalid exact RoboCOIN task key: {name!r}")
    return name


def build_robocoin_catalog(
    raw_root: Path,
    *,
    task_names: set[str] | None = None,
    start_task: str | None = None,
    max_tasks: int | None = None,
    limit_episodes_per_task: int | None = None,
) -> RobocoinCatalog:
    """Build a stable catalog without walking task payload directories."""

    if not raw_root.is_dir():
        raise ConversionError(f"RoboCOIN raw root is missing: {raw_root}")
    requested = {_validate_task_name(name) for name in task_names or set()}
    if start_task is not None:
        _validate_task_name(start_task)
    roots = sorted(
        (path for path in raw_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    available = {path.name for path in roots}
    missing = requested - available
    if missing:
        raise ConversionError(f"requested RoboCOIN tasks are absent: {sorted(missing)}")
    if start_task is not None and start_task not in available:
        raise ConversionError(f"--start-task is absent: {start_task}")
    if start_task is not None:
        start_index = next(index for index, path in enumerate(roots) if path.name == start_task)
        roots = roots[start_index:]
    if requested:
        roots = [path for path in roots if path.name in requested]
    if max_tasks is not None:
        roots = roots[:max_tasks]
    if not roots:
        raise ConversionError("RoboCOIN task selection is empty")

    preliminary: list[dict[str, Any]] = []
    for root in roots:
        info_path = root / "meta" / "info.json"
        tasks_path = root / "meta" / "tasks.jsonl"
        info = _read_json(info_path)
        task_rows = _read_jsonl(tasks_path)
        try:
            total_episodes = int(info["total_episodes"])
            total_frames = int(info["total_frames"])
            fps = float(info["fps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionError(f"invalid RoboCOIN catalog metadata in {root}") from exc
        if total_episodes <= 0 or total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ConversionError(f"invalid RoboCOIN catalog totals/FPS in {root}")
        if not fps.is_integer():
            raise ConversionError(f"fractional RoboCOIN FPS is unsupported in {root}: {fps}")
        declared_instructions = tuple(str(row["task"]) for row in task_rows)
        if not declared_instructions:
            raise ConversionError(f"RoboCOIN task metadata is empty in {tasks_path}")
        effective_info, _schema_policy = _conversion_schema_policy(root.name, info)
        schema_payload = {
            "robot_type": effective_info.get("robot_type"),
            "fps": effective_info.get("fps"),
            "features": effective_info.get("features"),
        }
        schema_fingerprint = canonical_fingerprint(schema_payload)
        selected_episodes = total_episodes
        selected_frames = total_frames
        episodes_path = root / "meta" / "episodes.jsonl"
        episode_rows = _read_jsonl(episodes_path)
        selected = (
            episode_rows[:limit_episodes_per_task]
            if limit_episodes_per_task is not None
            else episode_rows
        )
        selected_episodes = len(selected)
        selected_frames = sum(int(row["length"]) for row in selected)
        if selected_episodes <= 0 or selected_frames <= 0:
            raise ConversionError(f"RoboCOIN episode selection is empty in {root}")
        if limit_episodes_per_task is None and (
            selected_episodes != total_episodes or selected_frames != total_frames
        ):
            raise ConversionError(f"RoboCOIN catalog episode totals disagree in {root}")
        selected_declared = [
            str(row["tasks"][0]) for row in selected if len(row.get("tasks", [])) == 1
        ]
        instructions = tuple(
            dict.fromkeys(selected_declared or declared_instructions)
        )
        preliminary.append(
            {
                "key": root.name,
                "root": root,
                "info": info,
                "instructions": instructions,
                "schema": schema_fingerprint,
                "partition": f"schema-{schema_fingerprint[:12]}",
                "info_record": _small_file_record(info_path, raw_root),
                "tasks_record": _small_file_record(tasks_path, raw_root),
                "episodes_record": _small_file_record(episodes_path, raw_root),
                "selected_episodes": selected_episodes,
                "selected_frames": selected_frames,
            }
        )

    partition_episode: dict[str, int] = {}
    partition_frame: dict[str, int] = {}
    partition_unit: dict[str, int] = {}
    root_episode = 0
    root_frame = 0
    entries: list[RobocoinTaskCatalogEntry] = []
    for row in preliminary:
        partition = row["partition"]
        entry = RobocoinTaskCatalogEntry(
            key=row["key"],
            root=row["root"],
            info=row["info"],
            instructions=row["instructions"],
            info_schema_fingerprint=row["schema"],
            partition=partition,
            partition_unit_index=partition_unit.get(partition, 0),
            partition_episode_start=partition_episode.get(partition, 0),
            partition_frame_start=partition_frame.get(partition, 0),
            root_episode_start=root_episode,
            root_frame_start=root_frame,
            info_record=row["info_record"],
            tasks_record=row["tasks_record"],
            episodes_record=row["episodes_record"],
            selected_episodes=int(row["selected_episodes"]),
            selected_frames=int(row["selected_frames"]),
        )
        entries.append(entry)
        partition_unit[partition] = entry.partition_unit_index + 1
        partition_episode[partition] = entry.partition_episode_start + entry.total_episodes
        partition_frame[partition] = entry.partition_frame_start + entry.total_frames
        root_episode += entry.total_episodes
        root_frame += entry.total_frames

    payload = {
        "schema_version": 2,
        "conversion_policy_version": ROBOCOIN_CONVERSION_POLICY_VERSION,
        "source_root": str(raw_root),
        "tasks": [
            {
                "key": entry.key,
                "partition": entry.partition,
                "partition_unit_index": entry.partition_unit_index,
                "partition_episode_range": [
                    entry.partition_episode_start,
                    entry.partition_episode_start + entry.total_episodes,
                ],
                "partition_frame_range": [
                    entry.partition_frame_start,
                    entry.partition_frame_start + entry.total_frames,
                ],
                "root_episode_range": [
                    entry.root_episode_start,
                    entry.root_episode_start + entry.total_episodes,
                ],
                "root_frame_range": [
                    entry.root_frame_start,
                    entry.root_frame_start + entry.total_frames,
                ],
                "instructions": list(entry.instructions),
                "info": entry.info,
                "info_schema_fingerprint": entry.info_schema_fingerprint,
                "info_record": entry.info_record,
                "tasks_record": entry.tasks_record,
                "episodes_record": entry.episodes_record,
                "selected_episodes": entry.selected_episodes,
                "selected_frames": entry.selected_frames,
            }
            for entry in entries
        ],
    }
    return RobocoinCatalog(tuple(entries), payload)


def catalog_to_payload(catalog: RobocoinCatalog) -> dict[str, Any]:
    return catalog.fingerprint_payload


def catalog_from_payload(payload: dict[str, Any]) -> RobocoinCatalog:
    if (
        payload.get("schema_version") != 2
        or payload.get("conversion_policy_version") != ROBOCOIN_CONVERSION_POLICY_VERSION
        or not isinstance(payload.get("tasks"), list)
    ):
        raise ConversionError("unsupported RoboCOIN catalog state")
    raw_root = Path(str(payload.get("source_root", "")))
    entries: list[RobocoinTaskCatalogEntry] = []
    for row in payload["tasks"]:
        try:
            key = _validate_task_name(str(row["key"]))
            info = dict(row["info"])
            entries.append(
                RobocoinTaskCatalogEntry(
                    key=key,
                    root=raw_root / key,
                    info=info,
                    instructions=tuple(str(value) for value in row["instructions"]),
                    info_schema_fingerprint=str(row["info_schema_fingerprint"]),
                    partition=str(row["partition"]),
                    partition_unit_index=int(row["partition_unit_index"]),
                    partition_episode_start=int(row["partition_episode_range"][0]),
                    partition_frame_start=int(row["partition_frame_range"][0]),
                    root_episode_start=int(row["root_episode_range"][0]),
                    root_frame_start=int(row["root_frame_range"][0]),
                    info_record=dict(row["info_record"]),
                    tasks_record=dict(row["tasks_record"]),
                    episodes_record=(
                        dict(row["episodes_record"])
                        if row.get("episodes_record") is not None
                        else None
                    ),
                    selected_episodes=int(row["selected_episodes"]),
                    selected_frames=int(row["selected_frames"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionError(f"invalid RoboCOIN catalog state: {exc}") from exc
    return RobocoinCatalog(tuple(entries), payload)


def validate_catalog_small_files(catalog: RobocoinCatalog) -> None:
    """Validate only the two small files used to build each catalog row."""

    raw_root = Path(str(catalog.fingerprint_payload["source_root"]))
    for entry in catalog.tasks:
        for record in (entry.info_record, entry.tasks_record, entry.episodes_record):
            if record is None:
                continue
            path = raw_root / str(record["path"])
            stat = path.stat()
            if stat.st_size != int(record["size"]) or stat.st_mtime_ns != int(record["mtime_ns"]):
                raise ConversionError(f"RoboCOIN catalog source changed: {path}")


def inspect_robocoin_task(
    entry: RobocoinTaskCatalogEntry,
    *,
    output_uid: str,
    output_root: Path,
    limit_episodes: int | None = None,
) -> tuple[RobogenePartition, str, dict[str, Any]]:
    """Perform the one allowed full metadata/inventory pass for one task."""

    info = _read_json(entry.root / "meta" / "info.json")
    task_rows = _read_jsonl(entry.root / "meta" / "tasks.jsonl")
    raw_episode_rows = _read_jsonl(entry.root / "meta" / "episodes.jsonl")
    raw_stats_rows = _read_jsonl(entry.root / "meta" / "episodes_stats.jsonl")
    tasks_by_index = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    episode_rows = {int(row["episode_index"]): row for row in raw_episode_rows}
    stats_rows = {int(row["episode_index"]): row for row in raw_stats_rows}
    if set(episode_rows) != set(stats_rows):
        raise ConversionError(f"RoboCOIN episode/stats index mismatch in {entry.root}")
    if len(episode_rows) != int(info.get("total_episodes", -1)):
        raise ConversionError(f"RoboCOIN episode count mismatch in {entry.root}")
    for index, row in episode_rows.items():
        values = row.get("tasks")
        if not isinstance(values, list) or len(values) > 1:
            raise ConversionError(
                f"RoboCOIN episode {entry.key}:{index} has unsupported task metadata {values!r}"
            )
    effective_info, schema_policy = _conversion_schema_policy(entry.key, info)
    vectors, cameras = _feature_specs(effective_info)
    inventory, files = _file_inventory(entry.root)
    data_paths = sorted(
        path for path in files
        if path.relative_to(entry.root).parts[0] == "data" and path.suffix == ".parquet"
    )
    data_by_episode = {_episode_index(path, _EPISODE_RE, "data"): path for path in data_paths}
    if len(data_by_episode) != len(data_paths) or set(data_by_episode) != set(episode_rows):
        raise ConversionError(f"RoboCOIN data inventory mismatch in {entry.root}")
    video_keys = tuple(camera.feature_key for camera in cameras)
    videos_by_key: dict[str, dict[int, Path]] = {key: {} for key in video_keys}
    alias_paths: dict[str, dict[int, Path]] = {}
    for path in sorted(
        path for path in files
        if path.relative_to(entry.root).parts[0] == "videos" and path.suffix == ".mp4"
    ):
        relative = path.relative_to(entry.root / "videos")
        if len(relative.parts) < 2:
            raise ConversionError(f"invalid RoboCOIN video path: {path}")
        key = relative.parts[1] if relative.parts[0].startswith("chunk-") else relative.parts[0]
        episode_index = _episode_index(path, _VIDEO_RE, "video")
        if key not in videos_by_key:
            if key not in _PUBLISHER_VIDEO_ALIASES:
                raise ConversionError(f"undeclared RoboCOIN video key {key!r} in {entry.root}")
            alias_paths.setdefault(key, {})[episode_index] = path
            continue
        if episode_index in videos_by_key[key]:
            raise ConversionError(f"duplicate RoboCOIN video episode {episode_index}: {path}")
        videos_by_key[key][episode_index] = path
    if any(set(paths) != set(episode_rows) for paths in videos_by_key.values()):
        raise ConversionError(f"RoboCOIN video inventory mismatch in {entry.root}")
    duplicate_alias_evidence: list[dict[str, Any]] = []
    for alias, paths in alias_paths.items():
        target = _PUBLISHER_VIDEO_ALIASES[alias]
        if target not in videos_by_key or set(paths) != set(episode_rows):
            raise ConversionError(
                f"undeclared RoboCOIN video alias {alias!r} has no complete declared target {target!r}"
            )
        for episode_index, alias_path in paths.items():
            target_path = videos_by_key[target][episode_index]
            if _sample_identity(alias_path) != _sample_identity(target_path):
                raise ConversionError(
                    f"undeclared RoboCOIN video {alias_path} is not an exact sampled duplicate of {target_path}"
                )
        duplicate_alias_evidence.append(
            {
                "undeclared_path_key": alias,
                "declared_field": target,
                "episodes": len(paths),
                "validation": "equal size and equal first/middle/last 64KiB digests for every episode",
                "handling": "not recopied because it is a redundant publisher alias, not a declared dataset field",
            }
        )

    selected_indices = sorted(episode_rows)
    if limit_episodes is not None:
        selected_indices = selected_indices[:limit_episodes]
    if not selected_indices:
        raise ConversionError(f"RoboCOIN task contains no selected episodes: {entry.key}")

    # Footer-only schema audit is cheap and catches semantic changes without
    # treating per-episode pandas RangeIndex metadata as a feature schema.
    source_schema_variants: dict[str, dict[str, Any]] = {}
    compatible_schemas: dict[str, list[int]] = {}
    instructions_by_episode: dict[int, str] = {}
    episode_task_metadata_mismatches: list[dict[str, Any]] = []
    import pyarrow.parquet as pq

    for index in selected_indices:
        path = data_by_episode[index]
        schema = pq.ParquetFile(path).schema_arrow
        raw_schema = _semantic_schema_payload(schema)
        raw_fingerprint = canonical_fingerprint(raw_schema)
        variant = source_schema_variants.setdefault(
            raw_fingerprint,
            {
                "fingerprint": raw_fingerprint,
                "episodes": [],
                "field_types": {field.name: str(field.type) for field in schema},
            },
        )
        variant["episodes"].append(index)
        promoted_fields = tuple(schema_policy["dtype_resolution"])
        compatible_schema = _semantic_schema_payload(
            schema,
            float64_fields=promoted_fields,
        )
        compatible_fingerprint = canonical_fingerprint(compatible_schema)
        compatible_schemas.setdefault(compatible_fingerprint, []).append(index)
        declared = episode_rows[index].get("tasks", [])
        # The frame-level task_index plus tasks.jsonl is the relationship the
        # source LeRobot loader actually uses.  Several RoboCOIN exports have
        # empty or punctuation-drifted episode.tasks metadata, so audit and
        # report that discrepancy instead of guessing from the directory.
        values = pq.read_table(path, columns=["task_index"])["task_index"].unique().to_pylist()
        if len(values) != 1 or int(values[0]) not in tasks_by_index:
            raise ConversionError(
                f"cannot resolve one stable task from {path}: task_index={values!r}"
            )
        instruction = tasks_by_index[int(values[0])]
        if declared != [instruction]:
            episode_task_metadata_mismatches.append(
                {
                    "episode_index": index,
                    "episodes_jsonl_tasks": list(declared),
                    "parquet_task_index": int(values[0]),
                    "resolved_task": instruction,
                }
            )
        instructions_by_episode[index] = instruction
    if len(compatible_schemas) != 1:
        changes = {
            fingerprint[:12]: _episode_ranges(indices)
            for fingerprint, indices in compatible_schemas.items()
        }
        raise ConversionError(f"RoboCOIN task {entry.key} changes Parquet schema: {changes}")
    physical_schema_fingerprint = next(iter(compatible_schemas))
    for field, resolution in schema_policy["dtype_resolution"].items():
        resolution["observed_source_physical_types"] = sorted(
            {
                str(variant["field_types"][field])
                for variant in source_schema_variants.values()
            }
        )
    source_schema_evidence = []
    for fingerprint, variant in sorted(source_schema_variants.items()):
        indices = list(variant.pop("episodes"))
        source_schema_evidence.append(
            {
                **variant,
                "fingerprint": fingerprint,
                "episode_count": len(indices),
                "episode_ranges": _episode_ranges(indices),
            }
        )

    sources: list[RobogeneEpisodeSource] = []
    for episode_index in selected_indices:
        episode = episode_rows[episode_index]
        instruction = instructions_by_episode[episode_index]
        data_path = data_by_episode[episode_index]
        videos = tuple((key, videos_by_key[key][episode_index]) for key in video_keys)
        sources.append(
            RobogeneEpisodeSource(
                source_id=f"{entry.key}/episode-{episode_index:06d}",
                task_root=entry.root,
                split="robocoin",
                task_name=entry.key,
                source_episode_index=episode_index,
                instruction=instruction,
                stats=stats_rows[episode_index],
                length=int(episode["length"]),
                data_path=data_path,
                video_paths=videos,
                data_bytes=data_path.stat().st_size,
                video_bytes=sum(path.stat().st_size for _, path in videos),
            )
        )
    episodes = tuple(
        EpisodePlan(
            source.source_id,
            source.source_id,
            source.instruction,
            source.length,
            {
                "checkpoint_unit": entry.key,
                "source_task": entry.key,
                "source_episode_index": source.source_episode_index,
            },
        )
        for source in sources
    )
    selected_info_schema = canonical_fingerprint(
        {
            "robot_type": effective_info.get("robot_type"),
            "fps": effective_info.get("fps"),
            "features": effective_info.get("features"),
        }
    )
    if selected_info_schema != entry.info_schema_fingerprint:
        raise ConversionError(f"RoboCOIN task schema metadata changed after catalog: {entry.key}")
    plan = DatasetConversionPlan(
        dataset_uid=f"{output_uid}-{entry.partition}",
        output_path=output_root / output_uid / entry.partition,
        fps=int(info["fps"]),
        measured_fps=float(info["fps"]),
        robot_type=str(info.get("robot_type") or "unknown"),
        vector_features=vectors,
        camera_features=cameras,
        episodes=episodes,
        extra={
            "source_dataset": "RoboCOIN",
            "source_task": entry.key,
            "source_root": str(entry.root.parent),
            "source_files": list(inventory),
            "info_schema_fingerprint": entry.info_schema_fingerprint,
            "parquet_schema_fingerprint": physical_schema_fingerprint,
            "source_parquet_schema_variants": source_schema_evidence,
            "schema_policy": schema_policy,
            "video_encoding": {"mode": "copy", "no_resampling": True},
        },
    )
    samples = _inspect_payload_samples(tuple(sources), cameras, float(info["fps"]))
    partition = RobogenePartition(
        entry.partition,
        "robocoin",
        entry.info_schema_fingerprint,
        plan,
        effective_info,
        tuple(sources),
        tuple(
            {
                "path": f"{entry.key}/{row['path']}",
                "size": row["size"],
                "mtime_ns": row["mtime_ns"],
            }
            for row in inventory
        ),
        (),
        samples,
    )
    mapping = []
    output_features = plan.feature_schema()
    for key, feature in info["features"].items():
        output_dtype = str(output_features[key]["dtype"])
        promoted = output_dtype != str(feature["dtype"])
        mapping.append(
            {
                "source_field": key,
                "shape_dtype": f"{feature['shape']}/{feature['dtype']}",
                "semantics": "RoboCOIN publisher-declared LeRobot field and units; values preserved",
                "lerobot_field": key,
                "conversion": (
                    "rebase generated index"
                    if key in {"episode_index", "index", "task_index"}
                    else "lossless float32-to-float64 common-supertype promotion"
                    if promoted
                    else "copy"
                ),
                "output_dtype": output_dtype,
                "evidence": "meta/info.json, README.md, and Parquet footer",
                "lossy": False,
            }
        )
    peak = sum(source.data_bytes + source.video_bytes for source in sources)
    peak += max((source.data_bytes for source in sources), default=0) + 64 * 1024 * 1024
    fingerprint_payload = {
        "schema_version": 2,
        "conversion_policy_version": ROBOCOIN_CONVERSION_POLICY_VERSION,
        "task": entry.key,
        "output_uid": output_uid,
        "partition": entry.partition,
        "partition_unit_index": entry.partition_unit_index,
        "partition_episode_start": entry.partition_episode_start,
        "partition_frame_start": entry.partition_frame_start,
        "source_files": list(partition.source_files),
        "info_schema_fingerprint": entry.info_schema_fingerprint,
        "parquet_schema_fingerprint": physical_schema_fingerprint,
        "source_parquet_schema_variants": source_schema_evidence,
        "schema_policy": schema_policy,
        "fps": plan.fps,
        "features": plan.feature_schema(),
        "video_encoding": plan.extra["video_encoding"],
        "episodes": [
            {
                "source_id": source.source_id,
                "length": source.length,
                "instruction": source.instruction,
                "data": source.data_path.relative_to(entry.root.parent).as_posix(),
                "videos": [path.relative_to(entry.root.parent).as_posix() for _, path in source.video_paths],
            }
            for source in sources
        ],
    }
    evidence = {
        "estimated_peak_bytes": peak,
        "selected_episodes": len(sources),
        "selected_frames": sum(source.length for source in sources),
        "source_file_count": len(inventory),
        "payload_samples": list(samples),
        "mapping": mapping,
        "duplicate_video_aliases": duplicate_alias_evidence,
        "episode_task_metadata_mismatches": episode_task_metadata_mismatches,
        "source_parquet_schema_variants": source_schema_evidence,
        "schema_policy": schema_policy,
    }
    return partition, canonical_fingerprint(fingerprint_payload), evidence


def task_plan_to_payload(
    partition: RobogenePartition,
    *,
    task_fingerprint: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    catalog = RobogeneCatalog(
        (partition,),
        {"task_fingerprint": task_fingerprint, "evidence": evidence},
        tuple(evidence["mapping"]),
    )
    from readers.robogene_reader import catalog_to_payload as robogene_catalog_to_payload

    return {
        "schema_version": 2,
        "conversion_policy_version": ROBOCOIN_CONVERSION_POLICY_VERSION,
        "task_fingerprint": task_fingerprint,
        "evidence": evidence,
        "partition": robogene_catalog_to_payload(catalog)["partitions"][0],
    }


def task_plan_from_payload(payload: dict[str, Any]) -> tuple[RobogenePartition, str, dict[str, Any]]:
    if (
        payload.get("schema_version") != 2
        or payload.get("conversion_policy_version") != ROBOCOIN_CONVERSION_POLICY_VERSION
        or not isinstance(payload.get("partition"), dict)
    ):
        raise ConversionError("unsupported RoboCOIN task plan")
    from readers.robogene_reader import catalog_from_payload as robogene_catalog_from_payload

    wrapped = {
        "schema_version": 1,
        "fingerprint_payload": {},
        "mapping_table": [],
        "partitions": [payload["partition"]],
    }
    partition = robogene_catalog_from_payload(wrapped).partitions[0]
    fingerprint = payload.get("task_fingerprint")
    evidence = payload.get("evidence")
    if not isinstance(fingerprint, str) or not isinstance(evidence, dict):
        raise ConversionError("invalid RoboCOIN task plan identity")
    return partition, fingerprint, evidence
