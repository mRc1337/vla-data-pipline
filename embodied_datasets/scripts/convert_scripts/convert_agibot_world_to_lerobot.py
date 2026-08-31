#!/usr/bin/env python3
"""Lossless, task-transactional AgiBot World 2026 -> LeRobot v3.0 conversion.

Source MP4 containers are copied byte-for-byte.  Source Parquet payload columns
are retained exactly; only LeRobot-generated episode/global/task indices are
rebased.  Archive scans are task scoped and frozen in immutable plans so a
resume reads only the catalog, markers, and current unfinished task.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping, Sequence

sys.dont_write_bytecode = True

from convert_core.checkpoint import atomic_write_json, canonical_fingerprint, read_json_object
from convert_core.direct_commit import commit_verified_unit, committed_marker_path, finalize_direct_partition, globalize_unit_data_files, read_committed_unit_marker
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import _video_frame_count, validate_parquet_feature_schema
from convert_core.parallel import ParallelWorkUnit, PreparedUnits, prepare_work_units, run_parallel_work_units, write_verified_unit_marker
from convert_core.staging import DEFAULT_MIN_LOCAL_FREE_BYTES, StagingCapacityGuard, configure_runtime_environment, create_incomplete_output, exclusive_staging_lock, make_staging_layout, publish_success, validate_source_and_output_roots
from readers.agibot_world_reader import AgibotCatalog, CatalogTask, EpisodeSource, SourceShard, build_lightweight_catalog, catalog_from_payload, catalog_payload, dataset_plan, episode_source, extract_members, preflight_task, shard_from_payload, validate_source_file_records


DEFAULT_RAW_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw/agibot_world")
DEFAULT_OUTPUT = Path("/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/agibot_world")
DEFAULT_LOCAL = Path("/home/pai/zxw/agibot_world_staging")
REMOTE_WORK = Path("/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/agibot_world")
DEFAULT_MAX_LOCAL_INFLIGHT_BYTES = 128 * 1024**3
TASK_MARKER_VERSION = 3


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _slug(value: str) -> str:
    return value.replace("/", "--")


def _write_immutable(path: Path, payload: Mapping[str, Any], description: str) -> None:
    value = dict(payload)
    if path.is_file():
        if read_json_object(path, description) != value:
            raise ConversionError(f"immutable {description} changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _fingerprint_without(payload: Mapping[str, Any], key: str = "fingerprint") -> str:
    value = dict(payload)
    value.pop(key, None)
    return canonical_fingerprint(value)


def _output_parent(path: Path, uid: str) -> Path:
    return path.parent if path.name in {uid, "agibot_world"} else path


def _selection(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tasks": sorted(args.task), "start_task": args.start_task, "max_tasks": args.max_tasks,
        "max_shards_per_task": args.max_shards_per_task, "max_episodes_per_task": args.max_episodes_per_task,
    }


def _select_tasks(catalog: AgibotCatalog, args: argparse.Namespace) -> tuple[CatalogTask, ...]:
    tasks = list(catalog.tasks)
    if args.task:
        requested = set(args.task)
        known = {task.task_key for task in tasks}
        missing = sorted(requested - known)
        if missing:
            raise ConversionError(f"unknown AgiBot task(s): {missing}")
        tasks = [task for task in tasks if task.task_key in requested]
    if args.start_task is not None:
        if args.start_task.isdigit():
            start = int(args.start_task)
            tasks = [task for task in tasks if task.task_index >= start]
        else:
            indices = [index for index, task in enumerate(tasks) if task.task_key == args.start_task]
            if not indices:
                raise ConversionError(f"unknown --start-task: {args.start_task}")
            tasks = tasks[indices[0]:]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise ConversionError("AgiBot selection contains no tasks")
    return tuple(tasks)


def _load_or_create_catalog(args: argparse.Namespace, layout: Any) -> tuple[dict[str, Any], AgibotCatalog]:
    path = layout.resume / "task_catalog.json"
    selection = _selection(args)
    if args.resume and path.is_file():
        payload = read_json_object(path, "AgiBot task catalog")
        if payload.get("selection") != selection or payload.get("output_dataset_uid") != args.output_dataset_uid:
            raise ConversionError("AgiBot resume selection/output UID changed")
        if payload.get("catalog_fingerprint") != _fingerprint_without(payload, "catalog_fingerprint"):
            raise ConversionError("AgiBot task catalog fingerprint is corrupt")
        return payload, catalog_from_payload(payload)
    if path.exists():
        raise ConversionError(f"AgiBot task catalog exists; rerun with --resume: {path}")
    full = build_lightweight_catalog(args.raw_root)
    selected = _select_tasks(full, args)
    catalog = AgibotCatalog(full.source_root, selected)
    payload = catalog_payload(catalog, output_uid=args.output_dataset_uid, selection=selection)
    _write_immutable(path, payload, "AgiBot task catalog")
    return payload, catalog


def _plan_path(layout: Any, task_key: str) -> Path:
    return layout.resume / "tasks" / _slug(task_key) / "task_plan.json"


def _task_marker_path(layout: Any, task_key: str) -> Path:
    return layout.resume / "tasks" / _slug(task_key) / "commit.json"


def _load_plan(path: Path) -> dict[str, Any]:
    plan = read_json_object(path, "AgiBot task plan")
    if plan.get("fingerprint") != _fingerprint_without(plan):
        raise ConversionError(f"AgiBot task plan fingerprint changed: {path}")
    return plan


def _read_task_marker(path: Path, *, task_key: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    marker = read_json_object(path, "AgiBot task commit marker")
    if marker.get("schema_version") != TASK_MARKER_VERSION or marker.get("status") != "committed" or marker.get("task_key") != task_key:
        raise ConversionError(f"invalid AgiBot task commit marker: {path}")
    if marker.get("marker_fingerprint") != _fingerprint_without(marker, "marker_fingerprint"):
        raise ConversionError(f"AgiBot task commit marker fingerprint changed: {path}")
    plan = marker.get("task_plan")
    if not isinstance(plan, Mapping) or plan.get("fingerprint") != _fingerprint_without(plan):
        raise ConversionError(f"invalid embedded AgiBot task plan in marker: {path}")
    if marker.get("fingerprint") != plan.get("fingerprint"):
        raise ConversionError(f"AgiBot task marker fingerprint changed: {path}")
    return marker


def _validate_plan_catalog_identity(plan: Mapping[str, Any], task: CatalogTask, output_uid: str) -> None:
    planned_shards = plan.get("shards")
    if not isinstance(planned_shards, list) or not planned_shards:
        raise ConversionError(f"AgiBot task plan has no shards: {task.task_key}")
    expected_shards = task.shards[: len(planned_shards)]
    if [item.get("shard_id") for item in planned_shards] != [item.shard_id for item in expected_shards]:
        raise ConversionError(f"AgiBot task plan shard selection changed: {task.task_key}")
    expected_files = [record for shard in expected_shards for record in shard.source_files]
    expected = {
        "task_key": task.task_key,
        "partition_name": task.partition_name,
        "catalog_task_index": task.task_index,
        "output_dataset_uid": output_uid,
        "source_files": expected_files,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ConversionError(f"AgiBot task plan no longer matches catalog: {task.task_key}: {key}")


def _validate_marker_catalog_identity(marker: Mapping[str, Any], task: CatalogTask, output_uid: str) -> None:
    _validate_plan_catalog_identity(marker["task_plan"], task, output_uid)


def _copy_stream(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        while block := incoming.read(64 * 1024 * 1024):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if destination.stat().st_size != source.stat().st_size:
        raise ConversionError(f"short AgiBot payload copy: {source} -> {destination}")


def _set_column(table: Any, name: str, values: Any) -> Any:
    import pyarrow as pa
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ConversionError(f"AgiBot Parquet is missing generated column {name!r}")
    return table.set_column(index, table.schema.field(index), pa.array(values, type=table.schema.field(index).type))


def _rewrite_generated_columns(path: Path, source: EpisodeSource) -> None:
    import numpy as np
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    if table.num_rows != source.length:
        raise ConversionError(f"AgiBot Parquet row count changed: {path}")
    if [int(value) for value in table["frame_index"].to_pylist()] != list(range(source.length)):
        raise ConversionError(f"AgiBot source frame_index is not contiguous: {path}")
    preserved_names = [name for name in table.column_names if name not in {"episode_index", "index", "task_index"}]
    preserved = table.select(preserved_names)
    # A direct-commit unit is a one-episode LeRobot mini-dataset.  Keep its
    # generated columns local here; direct_commit globalizes them exactly once
    # while assigning the preplanned final chunk.
    table = _set_column(table, "episode_index", np.zeros(source.length, dtype=np.int64))
    table = _set_column(table, "index", np.arange(source.length, dtype=np.int64))
    table = _set_column(table, "task_index", np.zeros(source.length, dtype=np.int64))
    temporary = path.with_name(f".{path.name}.rewrite")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if not pq.read_table(path, columns=preserved_names).equals(preserved):
        raise ConversionError(f"AgiBot payload columns changed while rebasing {source.source_id}")


def _episode_metadata(source: EpisodeSource, *, fps: int) -> Any:
    import pyarrow as pa
    row: dict[str, Any] = {
        "episode_index": 0, "tasks": [source.instruction], "length": source.length,
        "data/chunk_index": 0, "data/file_index": 0, "dataset_from_index": 0,
        "dataset_to_index": source.length, "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
    }
    for key, _member in source.video_members:
        row[f"videos/{key}/chunk_index"] = 0
        row[f"videos/{key}/file_index"] = 0
        row[f"videos/{key}/from_timestamp"] = 0.0
        row[f"videos/{key}/to_timestamp"] = source.length / float(fps)
    stats = source.stats.get("stats")
    if not isinstance(stats, Mapping):
        raise ConversionError(f"AgiBot episode has no statistics: {source.source_id}")
    for feature, values in stats.items():
        if not isinstance(values, Mapping):
            raise ConversionError(f"invalid AgiBot episode stats: {source.source_id}:{feature}")
        for statistic, value in values.items():
            row[f"stats/{feature}/{statistic}"] = value
    return pa.Table.from_pylist([row])


def _unit_info(task_plan: Mapping[str, Any], source: EpisodeSource) -> dict[str, Any]:
    return {
        "codebase_version": "v3.0", "robot_type": task_plan["robot_type"], "fps": int(task_plan["fps"]),
        "features": task_plan["features"], "total_episodes": 1, "total_frames": source.length, "total_tasks": 1,
        "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 500,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "splits": {"train": "0:1"},
    }


@dataclass(frozen=True)
class UnitPayload:
    task_plan: dict[str, Any]
    source: EpisodeSource
    materialized_root: Path


def _build_unit(unit: ParallelWorkUnit) -> dict[str, Any]:
    import pyarrow.parquet as pq
    payload = unit.payload
    if not isinstance(payload, UnitPayload):
        raise ConversionError(f"invalid AgiBot unit payload: {unit.key}")
    source, root = payload.source, Path(unit.target_path)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    source_data = payload.materialized_root / source.data_member
    destination_data = root / "data" / "chunk-000" / "file-000.parquet"
    _copy_stream(source_data, destination_data)
    _rewrite_generated_columns(destination_data, source)
    for key, member in source.video_members:
        destination = root / "videos" / key / "chunk-000" / "file-000.mp4"
        _copy_stream(payload.materialized_root / member, destination)
    metadata = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_episode_metadata(source, fps=int(payload.task_plan["fps"])), metadata, compression="snappy", use_dictionary=True)
    atomic_write_json(root / "meta" / "info.json", _unit_info(payload.task_plan, source))
    atomic_write_json(root / "conversion_manifest.json", {"source_format": "agibot_world_lerobot_v21_archive", "unit": unit.key, "source": source.source_id, "video_operation": "byte-for-byte copy"})
    _validate_local_unit(unit, globalized=False)
    globalize_unit_data_files(unit)
    _validate_local_unit(unit, globalized=True)
    write_verified_unit_marker(unit)
    return {"unit": unit.key, "frames": source.length}


def _validate_local_unit(unit: ParallelWorkUnit, *, globalized: bool = True) -> None:
    import pyarrow.parquet as pq
    payload = unit.payload
    if not isinstance(payload, UnitPayload):
        raise ConversionError(f"invalid AgiBot unit payload: {unit.key}")
    root, source = Path(unit.target_path), payload.source
    data = root / "data" / "chunk-000" / "file-000.parquet"
    if not data.is_file() or pq.ParquetFile(data).metadata.num_rows != source.length:
        raise ConversionError(f"AgiBot unit Parquet is incomplete: {unit.key}")
    indices = pq.read_table(data, columns=["frame_index", "episode_index", "index", "task_index"])
    expected_episode = source.global_episode_index if globalized else 0
    expected_frame_start = source.global_frame_start if globalized else 0
    expected_task = source.partition_task_index if globalized else 0
    if (
        indices["frame_index"].to_pylist() != list(range(source.length))
        or indices["episode_index"].to_pylist() != [expected_episode] * source.length
        or indices["index"].to_pylist() != list(range(expected_frame_start, expected_frame_start + source.length))
        or indices["task_index"].to_pylist() != [expected_task] * source.length
    ):
        state = "global" if globalized else "unit-local"
        raise ConversionError(f"AgiBot unit has invalid {state} generated indices: {unit.key}")
    plan = dataset_plan(payload.task_plan, (source,))
    validate_parquet_feature_schema(plan, root)
    meta = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not meta.is_file() or pq.ParquetFile(meta).metadata.num_rows != 1:
        raise ConversionError(f"AgiBot unit metadata is incomplete: {unit.key}")
    cameras = {camera.feature_key: camera for camera in plan.camera_features}
    for key, _member in source.video_members:
        path = root / "videos" / key / "chunk-000" / "file-000.mp4"
        frames, height, width, fps, codec, pix_fmt = _video_frame_count(path)
        expected_info = payload.task_plan["features"][key]["info"]
        expected_codec = expected_info.get("video.codec", expected_info.get("source_video.codec"))
        expected_pix_fmt = expected_info.get("video.pix_fmt", expected_info.get("source_video.pix_fmt"))
        if frames != source.length or (height, width) != (cameras[key].height, cameras[key].width) or fps is None or abs(fps - plan.fps) > 1e-6 or codec != expected_codec or pix_fmt != expected_pix_fmt:
            raise ConversionError(f"AgiBot unit video validation failed: {path}")


def _unit(task_plan: Mapping[str, Any], source: EpisodeSource, *, layout: Any, materialized_root: Path) -> ParallelWorkUnit:
    index = int(task_plan["unit_start"]) + (source.global_episode_index - int(task_plan["episode_start"]))
    estimate = _unit_local_peak(source)
    return ParallelWorkUnit(
        index=index, key=source.source_id, dataset_uid=str(task_plan["output_dataset_uid"]),
        target_path=str(layout.work / "tasks" / _slug(str(task_plan["task_key"])) / "units" / f"unit-{index:06d}"),
        episode_start=source.global_episode_index, episode_end=source.global_episode_index + 1,
        frame_start=source.global_frame_start, frame_end=source.global_frame_start + source.length,
        task_indices=(source.partition_task_index,), weight=source.length,
        estimated_memory_bytes=max(source.data_bytes, 64 * 1024 * 1024), estimated_temp_bytes=estimate,
        fingerprint=str(task_plan["fingerprint"]), payload=UnitPayload(dict(task_plan), source, materialized_root),
    )


def _unit_local_peak(source: EpisodeSource) -> int:
    # Final copied payload plus a second Parquet-sized rewrite file and bounded
    # Arrow/writer overhead. Source extraction is accounted separately.
    return 2 * source.data_bytes + source.video_bytes + 64 * 1024 * 1024


def _bounded_worker_count(
    sources: Sequence[EpisodeSource], requested_workers: int, byte_budget: int
) -> tuple[int, int]:
    """Return a safe worker count and its conservative in-flight byte peak."""

    if requested_workers <= 0 or byte_budget <= 0:
        raise ValueError("worker count and local in-flight byte budget must be positive")
    estimates = sorted((_unit_local_peak(source) for source in sources), reverse=True)
    if not estimates:
        return 0, 0
    if estimates[0] > byte_budget:
        raise ConversionError(
            "AgiBot unit does not fit --max-local-inflight-bytes: "
            f"required={estimates[0]}, limit={byte_budget}"
        )
    workers = 0
    peak = 0
    for estimate in estimates[:requested_workers]:
        if peak + estimate > byte_budget:
            break
        workers += 1
        peak += estimate
    return workers, peak


def _materialize(
    shard: SourceShard,
    sources: Sequence[EpisodeSource],
    root: Path,
    *,
    progress_check: Callable[[], None] | None = None,
) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    groups: dict[tuple[str, ...], tuple[Any, dict[str, Path]]] = {}
    for source in sources:
        for member in (source.data_member, *(path for _key, path in source.video_members)):
            parts = shard.data if member == source.data_member else shard.videos
            key = tuple(str(path) for path in parts.paths)
            if key not in groups:
                groups[key] = (parts, {})
            groups[key][1][member] = root / member
    for parts, wanted in groups.values():
        sizes = extract_members(parts, wanted, progress_check=progress_check)
        for source in sources:
            for member in (source.data_member, *(path for _key, path in source.video_members)):
                if member in sizes and sizes[member] != source.member_sizes[member]:
                    raise ConversionError(f"AgiBot archive member size changed: {member}")
    atomic_write_json(root / "extracted.json", {"members": sorted(member for _parts, wanted in groups.values() for member in wanted)})


def _warmup(video: Path, *, fps: int, threads: int, marker_path: Path, task_fingerprint: str) -> None:
    output = marker_path.parent / "warmup.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    import av
    count = 0
    try:
        with av.open(str(video), "r") as incoming, av.open(str(output), "w") as outgoing:
            source = incoming.streams.video[0]
            stream = outgoing.add_stream("h264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = int(source.width), int(source.height), "yuv420p"
            stream.thread_count = threads
            for frame in incoming.decode(source):
                for packet in stream.encode(frame):
                    outgoing.mux(packet)
                count += 1
                if count == 30:
                    break
            for packet in stream.encode():
                outgoing.mux(packet)
        frames, _height, _width, actual_fps, codec, _pix = _video_frame_count(output)
        if count != 30 or frames != 30 or actual_fps is None or abs(actual_fps - fps) > 1e-6 or codec != "h264":
            raise ConversionError("AgiBot CPU encoder warmup validation failed")
    finally:
        output.unlink(missing_ok=True)
    _write_immutable(marker_path, {"schema_version": 1, "status": "passed", "frames": 30, "codec": "h264", "fps": fps, "encoder_threads_per_worker": threads, "task_fingerprint": task_fingerprint}, "AgiBot warmup marker")


def _validate_warmup_marker(path: Path, *, threads: int) -> None:
    marker = read_json_object(path, "AgiBot warmup marker")
    expected = {
        "schema_version": 1,
        "status": "passed",
        "frames": 30,
        "codec": "h264",
        "encoder_threads_per_worker": threads,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ConversionError(f"AgiBot warmup marker is stale or corrupt at {path}: {key}")


def _shard_for_id(task_plan: Mapping[str, Any], shard_id: str) -> SourceShard:
    matches = [shard_from_payload(item) for item in task_plan["shards"] if item["shard_id"] == shard_id]
    if len(matches) != 1:
        raise ConversionError(f"AgiBot task plan has no unique shard {shard_id!r}")
    return matches[0]


def _run_task(
    task_plan: dict[str, Any],
    *,
    layout: Any,
    args: argparse.Namespace,
    capacity: StagingCapacityGuard,
    previously_committed_units: int,
) -> tuple[dict[str, Any], int]:
    sources = tuple(episode_source(row) for row in task_plan["episodes"])
    partition = str(task_plan["partition_name"])
    uncommitted_sources: list[EpisodeSource] = []
    for source in sources:
        unit = _unit(task_plan, source, layout=layout, materialized_root=Path("/unused"))
        marker_path = committed_marker_path(layout.resume, partition, unit)
        if not marker_path.is_file():
            # Existing local output may be upload-ready and should be allowed
            # to drain even when free space is already below the normal
            # dispatch threshold. Corrupt/incomplete output is re-budgeted
            # after prepare_work_units classifies it below.
            if not Path(unit.target_path).exists():
                uncommitted_sources.append(source)
            continue
        try:
            read_committed_unit_marker(
                unit, partition_name=partition, resume_root=layout.resume
            )
        except ConversionError as exc:
            marker = read_json_object(marker_path, "AgiBot direct commit marker")
            if marker.get("status") != "committing":
                raise exc
    if uncommitted_sources:
        _task_workers, task_inflight_peak = _bounded_worker_count(
            uncommitted_sources, args.workers, args.max_local_inflight_bytes
        )
        capacity.check(
            f"task in-flight peak {task_plan['task_key']}",
            required_additional_bytes=task_inflight_peak,
        )
    partition_root = layout.final / partition
    partition_root.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list[EpisodeSource]] = {}
    for source in sources:
        by_shard.setdefault(source.shard_id, []).append(source)
    results, committed_count = [], 0
    warmup_marker = layout.resume / "warmup.json"
    for shard_id, shard_sources in by_shard.items():
        shard = _shard_for_id(task_plan, shard_id)
        provisional = tuple(_unit(task_plan, source, layout=layout, materialized_root=Path("/unused")) for source in shard_sources)
        candidates: list[ParallelWorkUnit] = []
        for source, unit in zip(shard_sources, provisional, strict=True):
            marker_path = committed_marker_path(layout.resume, partition, unit)
            if marker_path.is_file():
                try:
                    read_committed_unit_marker(unit, partition_name=partition, resume_root=layout.resume)
                except ConversionError as exc:
                    marker = read_json_object(marker_path, "AgiBot direct commit marker")
                    if marker.get("status") != "committing":
                        raise exc
                    candidates.append(unit)
                else:
                    committed_count += 1
            else:
                candidates.append(unit)
        if not candidates:
            results.append({"shard_id": shard_id, "skipped_from_markers": len(shard_sources), "upload_seconds": 0.0})
            continue

        # Classify verified local units before touching source archives. Upload
        # and delete them first, so an interrupted upload remains upload-only
        # and never forces extraction merely to release existing local state.
        prepared: PreparedUnits = prepare_work_units(candidates, _validate_local_unit, require_complete_plan=False)
        submitted_uploads = 0
        completed_uploads = 0
        upload_elapsed_seconds = 0.0

        def commit_one(unit: ParallelWorkUnit, *, trusted: bool) -> None:
            nonlocal submitted_uploads, completed_uploads, upload_elapsed_seconds
            submitted_uploads += 1
            upload_started = time.monotonic()
            try:
                commit_verified_unit(
                    unit,
                    partition_name=partition,
                    partition_root=partition_root,
                    resume_root=layout.resume,
                    trust_verified_marker=trusted,
                    retain_local_after_commit=False,
                )
            finally:
                upload_elapsed_seconds += time.monotonic() - upload_started
            completed_uploads += 1

        for unit in prepared.reusable:
            commit_one(unit, trusted=False)

        pending_indices = {unit.index for unit in prepared.pending}
        pending_sources = [source for source, unit in zip(shard_sources, provisional, strict=True) if unit.index in pending_indices]
        materialized_bytes = sum(source.data_bytes + source.video_bytes for source in pending_sources)
        effective_workers = 0
        inflight_peak = 0
        local_materialized = False
        if pending_sources:
            local_budget = args.max_local_inflight_bytes - materialized_bytes
            if local_budget > 0:
                try:
                    local_workers, local_inflight_peak = _bounded_worker_count(
                        pending_sources, args.workers, local_budget
                    )
                except ConversionError:
                    pass
                else:
                    local_required = materialized_bytes + local_inflight_peak
                    local_available = shutil.disk_usage(layout.local_root).free
                    local_materialized = (
                        local_available - local_required >= args.min_local_free_bytes
                    )
                    if local_materialized:
                        effective_workers, inflight_peak = local_workers, local_inflight_peak
            if local_materialized:
                materialized_root = layout.work / "tasks" / _slug(str(task_plan["task_key"])) / "materialized" / shard_id
                capacity.check(
                    f"materialize {task_plan['task_key']}/{shard_id}",
                    required_additional_bytes=materialized_bytes + inflight_peak,
                )
            else:
                materialized_root = REMOTE_WORK / str(task_plan["output_dataset_uid"]) / _slug(str(task_plan["task_key"])) / shard_id
                remote_available = shutil.disk_usage(REMOTE_WORK.parent).free
                if remote_available < materialized_bytes:
                    raise ConversionError(
                        f"AgiBot task materialization does not fit local or approved OSS scratch: "
                        f"required={materialized_bytes}, remote_available={remote_available}"
                    )
                effective_workers, inflight_peak = _bounded_worker_count(
                    pending_sources, args.workers, args.max_local_inflight_bytes
                )
                capacity.check(
                    f"task output {task_plan['task_key']}/{shard_id}",
                    required_additional_bytes=inflight_peak,
                )
        else:
            materialized_root = Path("/unused")
        if pending_sources:
            _materialize(
                shard,
                pending_sources,
                materialized_root,
                progress_check=lambda: capacity.periodic_check(
                    f"materializing {task_plan['task_key']}/{shard_id}"
                ),
            )
        if not warmup_marker.is_file() and pending_sources:
            first = pending_sources[0]
            rgb = next(((key, member) for key, member in first.video_members if not task_plan["features"][key]["info"].get("is_depth_map")), None)
            if rgb is None:
                raise ConversionError("AgiBot warmup requires a non-depth camera")
            _warmup(materialized_root / rgb[1], fps=int(task_plan["fps"]), threads=args.encoder_threads_per_worker, marker_path=warmup_marker, task_fingerprint=str(task_plan["fingerprint"]))
        elif warmup_marker.is_file():
            _validate_warmup_marker(warmup_marker, threads=args.encoder_threads_per_worker)

        materialized_units = {
            unit.index: _unit(task_plan, source, layout=layout, materialized_root=materialized_root)
            for source, unit in zip(shard_sources, provisional, strict=True)
        }
        prepared = PreparedUnits(
            tuple(materialized_units[unit.index] for unit in prepared.reusable),
            tuple(materialized_units[unit.index] for unit in prepared.pending),
            prepared.repaired_markers,
            prepared.discarded_corrupt,
        )
        units = tuple(materialized_units[unit.index] for unit in provisional)
        by_index = {unit.index: unit for unit in prepared.pending}

        def on_result(result: Any) -> None:
            commit_one(by_index[result.index], trusted=True)

        def before_dispatch(unit: ParallelWorkUnit, active: tuple[ParallelWorkUnit, ...]) -> None:
            required = unit.estimated_temp_bytes + sum(item.estimated_temp_bytes for item in active)
            local_base = materialized_bytes if local_materialized else 0
            if local_base + required > args.max_local_inflight_bytes:
                raise ConversionError(
                    f"AgiBot local materialization and in-flight units exceed "
                    f"--max-local-inflight-bytes: required={local_base + required}, "
                    f"limit={args.max_local_inflight_bytes}"
                )
            capacity.check(f"dispatch {unit.key}", required_additional_bytes=required)

        def health() -> None:
            capacity.periodic_check(f"running {task_plan['task_key']}")

        if prepared.pending:
            run_parallel_work_units(
                prepared.pending,
                _build_unit,
                workers=effective_workers,
                on_result=on_result,
                before_dispatch=before_dispatch,
                health_check=health,
                health_check_interval_seconds=2.0,
            )
        upload_stats = {
            "workers": 1,
            "max_queue_units": 0,
            "submitted_units": submitted_uploads,
            "completed_units": completed_uploads,
            "elapsed_seconds": upload_elapsed_seconds,
            "synchronous": True,
        }
        for unit in units:
            read_committed_unit_marker(unit, partition_name=partition, resume_root=layout.resume)
        committed_count += len(candidates)
        if pending_sources:
            shutil.rmtree(materialized_root, ignore_errors=True)
        results.append({"shard_id": shard_id, "units": len(units), "reused_local": len(prepared.reusable), "effective_workers": effective_workers, "inflight_peak_bytes": inflight_peak, "upload": upload_stats, "materialization": "local" if local_materialized else "remote_fallback"})
        total_committed = previously_committed_units + committed_count
        if args.simulate_interruption_after_units is not None and total_committed >= args.simulate_interruption_after_units:
            raise ConversionError(f"simulated interruption after {total_committed} committed units")
    units = tuple(_unit(task_plan, source, layout=layout, materialized_root=Path("/unused")) for source in sources)
    markers = [read_committed_unit_marker(unit, partition_name=partition, resume_root=layout.resume) for unit in units]
    upload_seconds = sum(float(item.get("upload_elapsed_seconds", 0.0)) for item in markers)
    marker = {
        "schema_version": TASK_MARKER_VERSION, "status": "committed", "task_key": task_plan["task_key"],
        "fingerprint": task_plan["fingerprint"], "partition_name": partition, "task_plan": task_plan,
        "source": {"files": task_plan["source_files"], "metadata": task_plan["source_metadata"]},
        "index_ranges": {key: task_plan[key] for key in ("episode_start", "episode_end", "frame_start", "frame_end", "unit_start", "unit_end")},
        "targets": [record for marker in markers for record in marker.get("bulk", [])], "shards": results,
        "upload_seconds": upload_seconds, "committed_unix": time.time(),
    }
    marker["marker_fingerprint"] = canonical_fingerprint(marker)
    _write_immutable(_task_marker_path(layout, str(task_plan["task_key"])), marker, "AgiBot task commit marker")
    shutil.rmtree(layout.work / "tasks" / _slug(str(task_plan["task_key"])), ignore_errors=True)
    return marker, committed_count


def _combined_plan(task_plans: Sequence[Mapping[str, Any]], partition: str) -> DatasetConversionPlan:
    plans = [dataset_plan(plan) for plan in task_plans if plan["partition_name"] == partition]
    if not plans:
        raise ConversionError(f"cannot finalize empty AgiBot partition {partition}")
    first = plans[0]
    if any(plan.feature_schema() != first.feature_schema() or plan.fps != first.fps or plan.robot_type != first.robot_type for plan in plans[1:]):
        raise ConversionError(f"AgiBot partition schema changed: {partition}")
    episodes = tuple(episode for plan in plans for episode in plan.episodes)
    if len({episode.instruction for episode in episodes}) != len(plans):
        raise ConversionError(f"AgiBot partition has ambiguous duplicate task instructions: {partition}")
    return DatasetConversionPlan(first.dataset_uid, Path(first.dataset_uid) / partition, first.fps, first.measured_fps, first.robot_type, first.vector_features, first.camera_features, episodes, first.extra)


def _finalize(markers: Sequence[Mapping[str, Any]], *, layout: Any, catalog: Mapping[str, Any], capacity: StagingCapacityGuard, startup_seconds: float) -> list[dict[str, Any]]:
    task_plans = [dict(marker["task_plan"]) for marker in markers]
    partitions = sorted({str(plan["partition_name"]) for plan in task_plans})
    reports = []
    for partition in partitions:
        selected = [plan for plan in task_plans if plan["partition_name"] == partition]
        plan = _combined_plan(selected, partition)
        units = tuple(_unit(task, episode_source(row), layout=layout, materialized_root=Path("/unused")) for task in selected for row in task["episodes"])
        units = tuple(sorted(units, key=lambda unit: unit.index))
        finalize_direct_partition(plan, units, layout.final / partition, resume_root=layout.resume, reader_format="agibot_world_lerobot_v21_archives", parallel_evidence={"task_scoped": True, "task_order": [task["task_key"] for task in selected], "remote_revalidation": False, "video_operation": "byte-for-byte copy"})
        reports.append({"partition": partition, "episodes": len(plan.episodes), "frames": plan.num_frames, "tasks": len(selected)})
    provenance_root = layout.final / "source_metadata"
    for task in task_plans:
        for record in task["source_metadata"]:
            source = Path(str(record["cache_path"]))
            destination = provenance_root / _slug(str(task["task_key"])) / str(record["shard_id"])
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
    manifest = {
        "format": "agibot_world_lerobot_v3_collection", "schema_version": 1,
        "output_dataset_uid": layout.dataset_uid, "catalog_fingerprint": catalog["catalog_fingerprint"],
        "task_order": [task["task_key"] for task in task_plans], "partitions": reports,
        "field_mapping": {task["schema_fingerprint"]: task["mapping_table"] for task in task_plans},
        "resume": {"catalog": "local resume/task_catalog.json", "task_commit_markers": True, "committed_tasks_skip_source_and_remote_bulk": True},
        "performance": {"startup_seconds": startup_seconds, "task_upload_seconds": {marker["task_key"]: marker["upload_seconds"] for marker in markers}},
        "local_capacity": {"max_local_temp_bytes": None, "min_local_free_bytes": capacity.min_free_bytes, "peak_staging_bytes": capacity.peak_staging_bytes},
        "success_marker_required": True,
    }
    atomic_write_json(layout.final / "collection_manifest.json", manifest)
    publish_success(layout.final, fingerprint=str(catalog["catalog_fingerprint"]), evidence={"partitions": reports, "tasks": len(task_plans)})
    return reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-work-root", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output-dataset-uid", default="agibot_world")
    parser.add_argument("--inspect-only", "--dry-run", action="store_true", dest="inspect_only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=_positive, default=4)
    parser.add_argument("--encoder-threads-per-worker", type=_positive, default=8)
    parser.add_argument("--upload-workers", type=_positive, default=1)
    parser.add_argument("--max-local-inflight-bytes", type=_positive, default=DEFAULT_MAX_LOCAL_INFLIGHT_BYTES)
    parser.add_argument("--min-local-free-bytes", type=_positive, default=DEFAULT_MIN_LOCAL_FREE_BYTES)
    parser.add_argument("--task", action="append", default=[], help="exact catalog task key; repeatable")
    parser.add_argument("--start-task")
    parser.add_argument("--max-tasks", type=_positive)
    parser.add_argument("--max-shards-per-task", "--max-shards", dest="max_shards_per_task", type=_positive)
    parser.add_argument("--max-episodes-per-task", "--max-episodes", dest="max_episodes_per_task", type=_positive)
    parser.add_argument("--run-id")
    parser.add_argument("--simulate-interruption-after-units", type=_positive, help=argparse.SUPPRESS)
    return parser


def _ensure_task_plan(
    task: CatalogTask,
    *,
    layout: Any,
    args: argparse.Namespace,
    cursors: dict[str, dict[str, int]],
    capacity: StagingCapacityGuard,
) -> dict[str, Any]:
    path = _plan_path(layout, task.task_key)
    if path.is_file():
        plan = _load_plan(path)
        _validate_plan_catalog_identity(plan, task, args.output_dataset_uid)
        validate_source_file_records(plan["source_files"], args.raw_root)
        return plan
    cursor = cursors.setdefault(task.partition_name, {"episode": 0, "frame": 0, "unit": 0, "task": 0})
    plan = preflight_task(
        task, source_root=args.raw_root, task_cache_root=path.parent, output_uid=args.output_dataset_uid,
        max_shards=args.max_shards_per_task, max_episodes=args.max_episodes_per_task,
        episode_start=cursor["episode"], frame_start=cursor["frame"], unit_start=cursor["unit"],
        partition_task_index=cursor["task"], encoder_threads=args.encoder_threads_per_worker,
        workers=args.workers,
        progress_check=lambda: capacity.periodic_check(f"preflight {task.task_key}"),
    )
    _write_immutable(path, plan, "AgiBot task plan")
    return plan


def _advance(cursors: dict[str, dict[str, int]], plan: Mapping[str, Any]) -> None:
    cursor = cursors.setdefault(str(plan["partition_name"]), {"episode": 0, "frame": 0, "unit": 0, "task": 0})
    expected = (cursor["episode"], cursor["frame"], cursor["unit"], cursor["task"])
    actual = (int(plan["episode_start"]), int(plan["frame_start"]), int(plan["unit_start"]), int(plan["partition_task_index"]))
    if expected != actual:
        raise ConversionError(f"AgiBot task index plan is not contiguous for {plan['task_key']}: expected={expected}, actual={actual}")
    cursor.update(episode=int(plan["episode_end"]), frame=int(plan["frame_end"]), unit=int(plan["unit_end"]), task=cursor["task"] + 1)


def main(argv: Sequence[str] | None = None) -> int:
    started = time.monotonic()
    args = _parser().parse_args(argv)
    if args.upload_workers != 1:
        raise ConversionError("AgiBot low-space unit transactions require --upload-workers 1")
    args.raw_root, output = validate_source_and_output_roots(args.raw_root, _output_parent(args.output_root, args.output_dataset_uid))
    if (args.task or args.start_task or args.max_tasks or args.max_shards_per_task or args.max_episodes_per_task) and args.output_dataset_uid == "agibot_world":
        raise ConversionError("subset/smoke conversion requires an independent --output-dataset-uid")
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    layout = make_staging_layout(output_root=output, local_work_root=args.local_work_root, dataset_uid=args.output_dataset_uid, run_id=run_id, work_dir=args.local_work_root / ".conversion_work" / args.output_dataset_uid, resume_dir=args.local_work_root / ".conversion_resume" / args.output_dataset_uid, logs_dir=args.local_work_root / ".conversion_logs" / args.output_dataset_uid, temp_dir=args.local_work_root / ".conversion_work" / args.output_dataset_uid / "tmp")
    with exclusive_staging_lock(layout.lock):
        configure_runtime_environment(layout, create=True)
        catalog_payload_value, catalog = _load_or_create_catalog(args, layout)
        startup_seconds = time.monotonic() - started
        if not args.inspect_only and (layout.final / "_SUCCESS").is_file():
            raise FileExistsError(f"valid AgiBot output already exists: {layout.final}")
        cursors: dict[str, dict[str, int]] = {}
        markers: list[dict[str, Any]] = []
        plans: list[dict[str, Any]] = []
        capacity = StagingCapacityGuard(layout.local_root, max_staging_bytes=None, min_free_bytes=args.min_local_free_bytes, interval_seconds=2.0, usage_roots=(layout.work, layout.resume, layout.logs, layout.cache_root))
        committed_units = 0
        for task in catalog.tasks:
            marker = _read_task_marker(_task_marker_path(layout, task.task_key), task_key=task.task_key)
            if marker is not None:
                _validate_marker_catalog_identity(marker, task, args.output_dataset_uid)
                plan = dict(marker["task_plan"])
                markers.append(marker)
                plans.append(plan)
                committed_units += int(plan["unit_end"]) - int(plan["unit_start"])
                _advance(cursors, plan)
                shutil.rmtree(layout.work / "tasks" / _slug(task.task_key), ignore_errors=True)
                continue
            plan = _ensure_task_plan(task, layout=layout, args=args, cursors=cursors, capacity=capacity)
            plans.append(plan)
            _advance(cursors, plan)
            if args.inspect_only:
                continue
            if plans[:-1]:
                same = [prior for prior in plans[:-1] if prior["partition_name"] == plan["partition_name"]]
                if same and same[0]["schema_fingerprint"] != plan["schema_fingerprint"]:
                    raise ConversionError(f"AgiBot schema changed inside output partition {plan['partition_name']}")
                if any(prior["instruction"] == plan["instruction"] for prior in same):
                    raise ConversionError(
                        f"AgiBot source tasks share one instruction inside {plan['partition_name']}; "
                        "refusing an ambiguous task_index mapping"
                    )
            create_incomplete_output(layout.final, fingerprint=str(catalog_payload_value["catalog_fingerprint"]), run_id=run_id)
            atomic_write_json(layout.resume / "global_state.json", {"schema_version": 1, "catalog_fingerprint": catalog_payload_value["catalog_fingerprint"], "completed_task_keys": [item["task_key"] for item in markers], "current_task": plan["task_key"], "updated_unix": time.time()})
            marker, count = _run_task(plan, layout=layout, args=args, capacity=capacity, previously_committed_units=committed_units)
            committed_units += count
            markers.append(marker)
            atomic_write_json(layout.resume / "global_state.json", {"schema_version": 1, "catalog_fingerprint": catalog_payload_value["catalog_fingerprint"], "completed_task_keys": [item["task_key"] for item in markers], "current_task": None, "updated_unix": time.time()})
        if args.inspect_only:
            print(json.dumps({"catalog": {"tasks": len(catalog.tasks), "catalog_fingerprint": catalog_payload_value["catalog_fingerprint"], "unavailable_files": sum(len(task.unavailable_files) for task in catalog.tasks)}, "task_preflight": [{key: plan[key] for key in ("task_key", "partition_name", "instruction", "episode_start", "episode_end", "frame_start", "frame_end", "estimated_output_bytes", "estimated_local_peak_bytes", "estimated_local_peak_with_remote_materialization_bytes", "largest_unit_local_peak_bytes", "preflight_samples", "mapping_table")} for plan in plans], "startup_seconds": startup_seconds, "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False, indent=2))
            return 0
        if len(markers) != len(catalog.tasks):
            raise ConversionError("not every selected AgiBot task has a commit marker")
        reports = _finalize(markers, layout=layout, catalog=catalog_payload_value, capacity=capacity, startup_seconds=startup_seconds)
        shutil.rmtree(layout.work, ignore_errors=True)
        print(json.dumps({"published": str(layout.final), "tasks": len(markers), "partitions": reports}, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
