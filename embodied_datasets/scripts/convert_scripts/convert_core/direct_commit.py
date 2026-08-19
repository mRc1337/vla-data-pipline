"""Zero-copy bulk commit for parallel LeRobot work units.

Workers still use the verified generic LeRobot writer, but each unit is given a
globally preassigned chunk.  Only the small generated index columns are
rewritten; MP4 files are moved into their final names on the same OSSFS mount.
Metadata is streamed and batched after all bulk units are verified, avoiding
LeRobot's full-dataset data/video concatenation pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from convert_core.checkpoint import atomic_write_json, read_json_object
from convert_core.episode_spec import DatasetConversionPlan
from convert_core.errors import ConversionError
from convert_core.parallel import (
    ParallelWorkUnit,
    validate_verified_unit_marker,
    validate_work_units,
)
from convert_core.staging import sha256_file


DIRECT_COMMIT_SCHEMA_VERSION = 1
DEFAULT_METADATA_BATCH_BYTES = 128 * 1024 * 1024
DEFAULT_METADATA_BATCH_EPISODES = 1000


@dataclass(frozen=True)
class DirectCommitPreparation:
    committed: tuple[ParallelWorkUnit, ...]
    uncommitted: tuple[ParallelWorkUnit, ...]
    discarded_corrupt: tuple[str, ...]


def committed_marker_path(
    resume_root: Path, partition_name: str, unit: ParallelWorkUnit
) -> Path:
    return resume_root / "committed" / partition_name / f"unit-{unit.index:06d}.json"


def _header(unit: ParallelWorkUnit, partition_name: str) -> dict[str, Any]:
    return {
        "direct_commit_schema_version": DIRECT_COMMIT_SCHEMA_VERSION,
        "partition": partition_name,
        "unit_index": unit.index,
        "unit_key": unit.key,
        "fingerprint": unit.fingerprint,
        "episode_start": unit.episode_start,
        "episode_end": unit.episode_end,
        "frame_start": unit.frame_start,
        "frame_end": unit.frame_end,
        "task_indices": list(unit.task_indices),
    }


def _record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def _validate_record(root: Path, record: Mapping[str, Any], description: str) -> None:
    value = record.get("relative_path")
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ConversionError(f"invalid {description} path in direct commit marker")
    path = root / value
    if not path.is_file():
        raise ConversionError(f"missing {description}: {path}")
    if path.stat().st_size != int(record.get("size", -1)):
        raise ConversionError(f"changed {description} size: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise ConversionError(f"changed {description} digest: {path}")


def _parse_chunk_file(relative_path: str) -> tuple[int, int]:
    path = Path(relative_path)
    try:
        chunk = int(path.parent.name.removeprefix("chunk-"))
        file_index = int(path.stem.removeprefix("file-"))
    except ValueError as exc:
        raise ConversionError(f"invalid LeRobot chunk path {relative_path!r}") from exc
    return chunk, file_index


def _bulk_source_paths(unit: ParallelWorkUnit) -> list[Path]:
    root = Path(unit.target_path)
    paths: list[Path] = []
    for directory in (root / "data", root / "videos"):
        if directory.is_dir():
            paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    if not paths:
        raise ConversionError(f"work unit has no bulk files: {root}")
    return paths


def _destination_for_source(
    unit: ParallelWorkUnit,
    source: Path,
    *,
    data_ordinal: int,
    video_ordinals: dict[str, int],
) -> tuple[Path, int, dict[str, int]]:
    root = Path(unit.target_path)
    relative = source.relative_to(root)
    if relative.parts[0] == "data":
        destination = Path("data") / f"chunk-{unit.index:03d}" / f"file-{data_ordinal:03d}.parquet"
        return destination, data_ordinal + 1, video_ordinals
    if len(relative.parts) < 4 or relative.parts[0] != "videos":
        raise ConversionError(f"unexpected work unit bulk path: {relative}")
    key = relative.parts[1]
    ordinal = video_ordinals.get(key, 0)
    destination = (
        Path("videos")
        / key
        / f"chunk-{unit.index:03d}"
        / f"file-{ordinal:03d}.mp4"
    )
    updated = dict(video_ordinals)
    updated[key] = ordinal + 1
    return destination, data_ordinal, updated


def _set_column(table: Any, name: str, values: Any) -> Any:
    import pyarrow as pa

    index = table.schema.get_field_index(name)
    if index < 0:
        raise ConversionError(f"Parquet table is missing generated column {name!r}")
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def _globalize_data_file(path: Path, unit: ParallelWorkUnit) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    local_episode = table["episode_index"].combine_chunks()
    if table.num_rows != unit.weight and len(list((Path(unit.target_path) / "data").rglob("*.parquet"))) == 1:
        raise ConversionError(
            f"unit {unit.key!r} data rows changed: {table.num_rows} != {unit.weight}"
        )
    local_min = int(pc.min(local_episode).as_py())
    local_max = int(pc.max(local_episode).as_py())
    local_count = unit.episode_end - unit.episode_start
    if local_min < 0 or local_max >= local_count:
        raise ConversionError(f"unit {unit.key!r} has invalid local episode indices")
    global_episode = pc.add(local_episode, pa.scalar(unit.episode_start, type=local_episode.type))
    global_index = pc.add(
        table["index"].combine_chunks(),
        pa.scalar(unit.frame_start, type=table.schema.field("index").type),
    )
    task_lookup = pa.array(unit.task_indices, type=table.schema.field("task_index").type)
    global_task = pc.take(task_lookup, local_episode)
    table = _set_column(table, "episode_index", global_episode)
    table = _set_column(table, "index", global_index)
    table = _set_column(table, "task_index", global_task)
    temporary = path.with_name(f".{path.name}.global-{uuid.uuid4().hex}")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata_inventory(unit: ParallelWorkUnit) -> list[dict[str, Any]]:
    root = Path(unit.target_path)
    paths = []
    for relative in (Path("meta"), Path("conversion_manifest.json")):
        path = root / relative
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(item for item in path.rglob("*") if item.is_file()))
    if not any(record.relative_to(root).as_posix() == "meta/info.json" for record in paths):
        raise ConversionError(f"work unit metadata is incomplete: {root}")
    return [_record(path, relative_to=root) for path in paths]


def _make_commit_intent(
    unit: ParallelWorkUnit,
    partition_name: str,
    partition_root: Path,
) -> dict[str, Any]:
    root = Path(unit.target_path)
    for path in sorted((root / "data").rglob("*.parquet")):
        _globalize_data_file(path, unit)
    data_ordinal = 0
    video_ordinals: dict[str, int] = {}
    bulk: list[dict[str, Any]] = []
    for source in _bulk_source_paths(unit):
        destination, data_ordinal, video_ordinals = _destination_for_source(
            unit,
            source,
            data_ordinal=data_ordinal,
            video_ordinals=video_ordinals,
        )
        if data_ordinal > 1000 or any(value > 1000 for value in video_ordinals.values()):
            raise ConversionError(
                f"unit {unit.key!r} exceeds LeRobot's 1000-files-per-chunk limit"
            )
        record = _record(source, relative_to=root)
        record["destination"] = destination.as_posix()
        bulk.append(record)
    destinations = [record["destination"] for record in bulk]
    if len(destinations) != len(set(destinations)):
        raise ConversionError(f"unit {unit.key!r} generated duplicate final paths")
    return {
        **_header(unit, partition_name),
        "status": "committing",
        "partition_root": str(partition_root),
        "bulk": bulk,
        "work_metadata": _metadata_inventory(unit),
    }


def _finish_intent(
    unit: ParallelWorkUnit,
    marker_path: Path,
    marker: dict[str, Any],
    partition_root: Path,
) -> dict[str, Any]:
    root = Path(unit.target_path)
    for record in marker.get("bulk", []):
        source = root / str(record["relative_path"])
        destination = partition_root / str(record["destination"])
        if source.exists() and destination.exists():
            raise ConversionError(
                f"both work and final paths exist during direct commit: {source}, {destination}"
            )
        if source.exists():
            _validate_record(root, record, "work unit bulk file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        _validate_record(
            partition_root,
            {
                **record,
                "relative_path": record["destination"],
            },
            "committed bulk file",
        )
    marker = {**marker, "status": "verified"}
    atomic_write_json(marker_path, marker)
    return marker


def commit_verified_unit(
    unit: ParallelWorkUnit,
    *,
    partition_name: str,
    partition_root: Path,
    resume_root: Path,
) -> dict[str, Any]:
    """Patch generated indices and move one verified unit's bulk files in place."""

    marker_path = committed_marker_path(resume_root, partition_name, unit)
    if marker_path.exists():
        marker = read_json_object(marker_path, "direct commit marker")
        for key, value in _header(unit, partition_name).items():
            if marker.get(key) != value:
                raise ConversionError(f"direct commit marker identity changed at {marker_path}: {key}")
        if marker.get("status") == "committing":
            return _finish_intent(unit, marker_path, marker, partition_root)
        validate_committed_unit(
            unit,
            partition_name=partition_name,
            partition_root=partition_root,
            resume_root=resume_root,
        )
        return marker
    validate_verified_unit_marker(unit)
    marker = _make_commit_intent(unit, partition_name, partition_root)
    atomic_write_json(marker_path, marker)
    return _finish_intent(unit, marker_path, marker, partition_root)


def validate_committed_unit(
    unit: ParallelWorkUnit,
    *,
    partition_name: str,
    partition_root: Path,
    resume_root: Path,
) -> dict[str, Any]:
    marker_path = committed_marker_path(resume_root, partition_name, unit)
    marker = read_json_object(marker_path, "direct commit marker")
    for key, value in _header(unit, partition_name).items():
        if marker.get(key) != value:
            raise ConversionError(f"direct commit marker is stale at {marker_path}: {key}")
    if marker.get("status") != "verified":
        raise ConversionError(f"direct commit is unfinished at {marker_path}")
    for record in marker.get("bulk", []):
        _validate_record(
            partition_root,
            {**record, "relative_path": record["destination"]},
            "committed bulk file",
        )
    root = Path(unit.target_path)
    for record in marker.get("work_metadata", []):
        _validate_record(root, record, "committed work metadata")
    return marker


def _discard_corrupt_commit(
    unit: ParallelWorkUnit,
    marker: Mapping[str, Any],
    *,
    partition_root: Path,
    marker_path: Path,
) -> None:
    for record in marker.get("bulk", []):
        destination = record.get("destination")
        if isinstance(destination, str) and destination and not Path(destination).is_absolute():
            (partition_root / destination).unlink(missing_ok=True)
    target = Path(unit.target_path)
    if target.exists():
        shutil.rmtree(target)
    marker_path.unlink(missing_ok=True)


def _discard_preassigned_chunk(unit: ParallelWorkUnit, partition_root: Path) -> None:
    data_chunk = partition_root / "data" / f"chunk-{unit.index:03d}"
    if data_chunk.exists():
        shutil.rmtree(data_chunk)
    videos = partition_root / "videos"
    if videos.is_dir():
        for key_root in videos.iterdir():
            chunk = key_root / f"chunk-{unit.index:03d}"
            if chunk.exists():
                shutil.rmtree(chunk)


def prepare_direct_commits(
    units: Sequence[ParallelWorkUnit],
    *,
    partition_name: str,
    partition_root: Path,
    resume_root: Path,
) -> DirectCommitPreparation:
    """Revalidate every committed chunk and isolate only corrupt units for rebuilding."""

    validate_work_units(units)
    committed: list[ParallelWorkUnit] = []
    uncommitted: list[ParallelWorkUnit] = []
    discarded: list[str] = []
    for unit in units:
        marker_path = committed_marker_path(resume_root, partition_name, unit)
        if not marker_path.exists():
            # An absent marker cannot authorize reuse. Deterministic per-unit
            # chunk assignment lets us remove any orphan left by external
            # marker deletion without touching another unit.
            _discard_preassigned_chunk(unit, partition_root)
            uncommitted.append(unit)
            continue
        try:
            marker = read_json_object(marker_path, "direct commit marker")
        except ConversionError:
            _discard_preassigned_chunk(unit, partition_root)
            target = Path(unit.target_path)
            if target.exists():
                shutil.rmtree(target)
            marker_path.unlink(missing_ok=True)
            discarded.append(unit.key)
            uncommitted.append(unit)
            continue
        for key, value in _header(unit, partition_name).items():
            if marker.get(key) != value:
                raise ConversionError(f"direct commit identity changed at {marker_path}: {key}")
        try:
            if marker.get("status") == "committing":
                _finish_intent(unit, marker_path, marker, partition_root)
            validate_committed_unit(
                unit,
                partition_name=partition_name,
                partition_root=partition_root,
                resume_root=resume_root,
            )
        except (ConversionError, OSError, ValueError):
            _discard_corrupt_commit(
                unit,
                marker,
                partition_root=partition_root,
                marker_path=marker_path,
            )
            discarded.append(unit.key)
            uncommitted.append(unit)
        else:
            committed.append(unit)
    return DirectCommitPreparation(
        tuple(committed), tuple(uncommitted), tuple(discarded)
    )


def _replace_episode_stat(
    values: dict[str, list[Any]], row: int, feature: str, data: np.ndarray
) -> None:
    from lerobot.datasets.compute_stats import get_feature_stats

    stats = get_feature_stats(data, axis=0, keepdims=True)
    for stat, result in stats.items():
        column = f"stats/{feature}/{stat}"
        if column not in values:
            raise ConversionError(f"episode metadata is missing {column}")
        values[column][row] = np.asarray(result).tolist()


def _mapping_by_local_path(marker: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(record["relative_path"]): str(record["destination"])
        for record in marker.get("bulk", [])
    }


def _patch_episode_table(
    table: Any,
    unit: ParallelWorkUnit,
    marker: Mapping[str, Any],
) -> Any:
    import pyarrow as pa

    values = {name: table[name].to_pylist() for name in table.column_names}
    mapping = _mapping_by_local_path(marker)
    camera_keys = sorted(
        name.removeprefix("videos/").removesuffix("/chunk_index")
        for name in table.column_names
        if name.startswith("videos/") and name.endswith("/chunk_index")
    )
    for row in range(table.num_rows):
        local_episode = int(values["episode_index"][row])
        if local_episode < 0 or local_episode >= len(unit.task_indices):
            raise ConversionError(f"invalid local episode metadata in {unit.key!r}")
        length = int(values["length"][row])
        global_episode = unit.episode_start + local_episode
        global_from = unit.frame_start + int(values["dataset_from_index"][row])
        global_to = unit.frame_start + int(values["dataset_to_index"][row])
        local_data = (
            f"data/chunk-{int(values['data/chunk_index'][row]):03d}/"
            f"file-{int(values['data/file_index'][row]):03d}.parquet"
        )
        if local_data not in mapping:
            raise ConversionError(f"missing data mapping for {local_data}")
        data_chunk, data_file = _parse_chunk_file(mapping[local_data])
        values["episode_index"][row] = global_episode
        values["dataset_from_index"][row] = global_from
        values["dataset_to_index"][row] = global_to
        values["data/chunk_index"][row] = data_chunk
        values["data/file_index"][row] = data_file
        for key in camera_keys:
            local_video = (
                f"videos/{key}/chunk-"
                f"{int(values[f'videos/{key}/chunk_index'][row]):03d}/"
                f"file-{int(values[f'videos/{key}/file_index'][row]):03d}.mp4"
            )
            if local_video not in mapping:
                raise ConversionError(f"missing video mapping for {local_video}")
            video_chunk, video_file = _parse_chunk_file(mapping[local_video])
            values[f"videos/{key}/chunk_index"][row] = video_chunk
            values[f"videos/{key}/file_index"][row] = video_file
        _replace_episode_stat(
            values,
            row,
            "episode_index",
            np.full(length, global_episode, dtype=np.int64),
        )
        _replace_episode_stat(
            values,
            row,
            "index",
            np.arange(global_from, global_to, dtype=np.int64),
        )
        _replace_episode_stat(
            values,
            row,
            "task_index",
            np.full(length, unit.task_indices[local_episode], dtype=np.int64),
        )
    arrays = [pa.array(values[field.name], type=field.type) for field in table.schema]
    return pa.Table.from_arrays(arrays, schema=table.schema)


def _episode_stats(table: Any) -> Iterable[dict[str, dict[str, np.ndarray]]]:
    columns = [name for name in table.column_names if name.startswith("stats/")]
    values = {name: table[name].to_pylist() for name in columns}
    for row in range(table.num_rows):
        stats: dict[str, dict[str, np.ndarray]] = {}
        for column in columns:
            feature, stat = column[len("stats/") :].rsplit("/", 1)
            stats.setdefault(feature, {})[stat] = np.asarray(values[column][row])
        yield stats


def _set_metadata_location(table: Any, chunk_index: int, file_index: int) -> Any:
    return _set_column(
        _set_column(
            table,
            "meta/episodes/chunk_index",
            np.full(table.num_rows, chunk_index, dtype=np.int64),
        ),
        "meta/episodes/file_index",
        np.full(table.num_rows, file_index, dtype=np.int64),
    )


def finalize_direct_partition(
    plan: DatasetConversionPlan,
    units: Sequence[ParallelWorkUnit],
    partition_root: Path,
    *,
    resume_root: Path,
    reader_format: str,
    parallel_evidence: Mapping[str, Any],
    metadata_batch_bytes: int = DEFAULT_METADATA_BATCH_BYTES,
    metadata_batch_episodes: int = DEFAULT_METADATA_BATCH_EPISODES,
) -> Path:
    """Stream final metadata/stats after all bulk chunks are committed."""

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    from convert_core.lerobot_writer import (
        build_manifest,
        validate_video_files,
        validate_written_dataset,
    )

    validate_work_units(units)
    if metadata_batch_bytes <= 0 or metadata_batch_episodes <= 0:
        raise ValueError("metadata batch limits must be positive")
    markers = [
        validate_committed_unit(
            unit,
            partition_name=plan.output_path.name,
            partition_root=partition_root,
            resume_root=resume_root,
        )
        for unit in units
    ]
    meta_root = partition_root / "meta"
    if meta_root.exists():
        shutil.rmtree(meta_root)
    meta_root.mkdir(parents=True)

    first_info = json.loads(
        (Path(units[0].target_path) / "meta" / "info.json").read_text(encoding="utf-8")
    )
    tasks = list(dict.fromkeys(episode.instruction for episode in plan.episodes))
    first_info.update(
        total_episodes=len(plan.episodes),
        total_frames=plan.num_frames,
        total_tasks=len(tasks),
        splits={"train": f"0:{len(plan.episodes)}"},
    )
    atomic_write_json(meta_root / "info.json", first_info)
    task_frame = pd.DataFrame(
        {"task_index": range(len(tasks))},
        index=pd.Index(tasks, name="task"),
    )
    task_frame.to_parquet(meta_root / "tasks.parquet")

    chunks_size = int(first_info["chunks_size"])
    buffered: list[Any] = []
    buffered_rows = 0
    buffered_bytes = 0
    metadata_ordinal = 0
    global_stats: dict[str, dict[str, np.ndarray]] | None = None

    def flush() -> None:
        nonlocal buffered, buffered_rows, buffered_bytes, metadata_ordinal
        if not buffered:
            return
        chunk_index, file_index = divmod(metadata_ordinal, chunks_size)
        table = pa.concat_tables(buffered) if len(buffered) > 1 else buffered[0]
        table = _set_metadata_location(table, chunk_index, file_index)
        path = (
            meta_root
            / "episodes"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="snappy", use_dictionary=True)
        metadata_ordinal += 1
        buffered = []
        buffered_rows = 0
        buffered_bytes = 0

    for unit, marker in zip(units, markers, strict=True):
        episode_paths = sorted((Path(unit.target_path) / "meta" / "episodes").rglob("*.parquet"))
        if not episode_paths:
            raise ConversionError(f"unit metadata has no episode parquet: {unit.target_path}")
        unit_tables = [_patch_episode_table(pq.read_table(path), unit, marker) for path in episode_paths]
        unit_table = pa.concat_tables(unit_tables) if len(unit_tables) > 1 else unit_tables[0]
        if unit_table.num_rows != unit.episode_end - unit.episode_start:
            raise ConversionError(f"unit {unit.key!r} episode metadata count changed")
        for stats in _episode_stats(unit_table):
            global_stats = stats if global_stats is None else aggregate_stats([global_stats, stats])
        if buffered and (
            buffered_rows + unit_table.num_rows > metadata_batch_episodes
            or buffered_bytes + unit_table.nbytes > metadata_batch_bytes
        ):
            flush()
        buffered.append(unit_table)
        buffered_rows += unit_table.num_rows
        buffered_bytes += unit_table.nbytes
    flush()
    if global_stats is None:
        raise ConversionError("direct commit produced no episode statistics")
    write_stats(global_stats, partition_root)

    validate_written_dataset(plan, partition_root)
    video_evidence = validate_video_files(
        plan, partition_root, expected_frames=plan.num_frames
    )
    manifest = build_manifest(plan, reader_format=reader_format)
    manifest.update(
        {
            "num_video_files": sum(len(rows) for rows in video_evidence.values()),
            "video_validation": video_evidence,
            "parallel": {
                "schema_version": DIRECT_COMMIT_SCHEMA_VERSION,
                "direct_final_chunks": True,
                "bulk_aggregation_copy": False,
                "aggregation_order": [unit.key for unit in units],
                "checkpoint_granularity": "reader-defined work unit",
                "metadata_batch_files": metadata_ordinal,
                **dict(parallel_evidence),
            },
        }
    )
    atomic_write_json(partition_root / "conversion_manifest.json", manifest)
    return partition_root
