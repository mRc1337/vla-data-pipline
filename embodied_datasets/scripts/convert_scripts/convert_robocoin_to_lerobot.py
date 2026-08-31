"""Reliable task-sequential RoboCOIN LeRobot-v2.1 to v3.0 conversion.

RoboCOIN is a collection of publisher-defined task directories with multiple
incompatible robot schemas.  Compatible tasks are grouped into stable schema
partitions; every task is one direct-commit chunk and is fully converted,
validated, copied, remotely validated, and committed before the next task is
preflighted.  Source video containers are copied without decoding or
resampling.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

sys.dont_write_bytecode = True

from convert_core.checkpoint import atomic_write_json, canonical_fingerprint, read_json_object
from convert_core.direct_commit import (
    commit_verified_unit,
    committed_marker_path,
    finalize_direct_partition,
)
from convert_core.episode_spec import DatasetConversionPlan
from convert_core.errors import ConversionError
from convert_core.parallel import (
    ParallelWorkUnit,
    read_verified_unit_marker,
    verified_marker_path,
    write_verified_unit_marker,
)
from convert_core.staging import (
    create_incomplete_output,
    exclusive_staging_lock,
    publish_success,
    validate_source_and_output_roots,
)
from convert_core.storage import DiskGuard, directory_size, filesystem_snapshot
from convert_robogene_to_lerobot import (
    UnitPayload,
    _episode_metadata_table,
    _localise_data_file,
    _normalise_info,
    _validate_local_unit,
)
from readers.robocoin_reader import (
    ROBOCOIN_CONVERSION_POLICY_VERSION,
    RobocoinCatalog,
    RobocoinTaskCatalogEntry,
    build_robocoin_catalog,
    catalog_from_payload,
    catalog_to_payload,
    inspect_robocoin_task,
    task_plan_from_payload,
    task_plan_to_payload,
)
from readers.robogene_reader import RobogenePartition


DEFAULT_RAW_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw/robocoin")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0")
DEFAULT_LOCAL_WORK_ROOT = Path("/home/pai/zxw/robocoin_staging")
DEFAULT_MIN_LOCAL_FREE_BYTES = 200_000_000_000
COPY_BLOCK_BYTES = 64 * 1024 * 1024


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _output_uid(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or path.name != value or ".." in path.parts:
        raise argparse.ArgumentTypeError("output UID must be one path component")
    return value


def _run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"


def _task_plan_path(local_root: Path, task: str) -> Path:
    return local_root / "resume" / "task_plans" / f"{task}.json"


def _catalog_state_path(local_root: Path) -> Path:
    return local_root / "resume" / "robocoin_catalog.json"


def _copy_file_guarded(source: Path, destination: Path, guard: DiskGuard) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        while payload := incoming.read(COPY_BLOCK_BYTES):
            guard.check("RoboCOIN task copy", required_additional_bytes=len(payload))
            outgoing.write(payload)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _canonicalize_v3_parquet(path: Path, plan: DatasetConversionPlan) -> None:
    """Change only Arrow container shape to LeRobot-v3's canonical schema."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    changed = False
    for feature in plan.vector_features:
        key = feature.feature_key
        if key not in table.column_names:
            raise ConversionError(f"RoboCOIN source is missing declared field {key!r}: {path}")
        shape = feature.resolved_shape
        column = table[key]
        scalar_type = pa.from_numpy_dtype(np.dtype(feature.dtype))
        actual = column.type
        if shape == (1,) and (pa.types.is_list(actual) or pa.types.is_large_list(actual)):
            source_scalar_type = actual.value_type
            if not np.can_cast(
                np.dtype(source_scalar_type.to_pandas_dtype()),
                np.dtype(scalar_type.to_pandas_dtype()),
                casting="safe",
            ):
                raise ConversionError(
                    f"RoboCOIN field {key!r} would narrow {source_scalar_type} to "
                    f"{scalar_type} in {path}"
                )
            values = column.to_pylist()
            if any(not isinstance(value, list) or len(value) != 1 for value in values):
                raise ConversionError(f"RoboCOIN singleton field {key!r} has non-singleton values in {path}")
            replacement = pa.array([value[0] for value in values], type=scalar_type)
        elif len(shape) == 1 and shape != (1,) and (
            pa.types.is_list(actual) or pa.types.is_large_list(actual)
        ):
            source_scalar_type = actual.value_type
            if not np.can_cast(
                np.dtype(source_scalar_type.to_pandas_dtype()),
                np.dtype(scalar_type.to_pandas_dtype()),
                casting="safe",
            ):
                raise ConversionError(
                    f"RoboCOIN field {key!r} would narrow {source_scalar_type} to "
                    f"{scalar_type} in {path}"
                )
            values = column.to_pylist()
            if any(not isinstance(value, list) or len(value) != shape[0] for value in values):
                raise ConversionError(f"RoboCOIN field {key!r} violates declared shape {shape} in {path}")
            replacement = pa.array(values, type=pa.list_(scalar_type, shape[0]))
        else:
            continue
        index = table.schema.get_field_index(key)
        table = table.set_column(index, pa.field(key, replacement.type), replacement)
        changed = True
    if not changed:
        return
    temporary = path.with_name(f".{path.name}.v3-schema")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_episode(
    source: Any,
    *,
    local_index: int,
    unit: ParallelWorkUnit,
    root: Path,
    frame_offset: int,
    guard: DiskGuard,
) -> None:
    destination = root / "data" / "chunk-000" / f"file-{local_index:03d}.parquet"
    _copy_file_guarded(source.data_path, destination, guard)
    _localise_data_file(
        destination,
        source,
        global_episode=unit.episode_start + local_index,
        global_frame=unit.frame_start + frame_offset,
        global_task=unit.task_indices[local_index],
    )
    _canonicalize_v3_parquet(destination, unit.payload.partition.plan)
    for key, video_source in source.video_paths:
        _copy_file_guarded(
            video_source,
            root / "videos" / key / "chunk-000" / f"file-{local_index:03d}.mp4",
            guard,
        )


def _episode_output_paths(root: Path, source: Any, local_index: int) -> tuple[Path, tuple[Path, ...]]:
    data = root / "data" / "chunk-000" / f"file-{local_index:03d}.parquet"
    videos = tuple(
        root / "videos" / key / "chunk-000" / f"file-{local_index:03d}.mp4"
        for key, _source_path in source.video_paths
    )
    return data, videos


def _episode_checkpoint_path(root: Path, local_index: int) -> Path:
    return root / ".episode_checkpoints" / f"episode-{local_index:06d}.json"


def _episode_output_evidence(
    source: Any,
    *,
    local_index: int,
    unit: ParallelWorkUnit,
    root: Path,
    frame_offset: int,
) -> dict[str, Any]:
    """Validate one local episode without rereading its payload columns."""

    import pyarrow.parquet as pq
    from convert_core.lerobot_writer import _video_frame_count

    data_path, video_paths = _episode_output_paths(root, source, local_index)
    parquet = pq.ParquetFile(data_path)
    if parquet.metadata.num_rows != source.length:
        raise ConversionError(
            f"RoboCOIN episode checkpoint row count changed: {data_path}: "
            f"{parquet.metadata.num_rows} != {source.length}"
        )
    generated = pq.read_table(
        data_path,
        columns=["episode_index", "frame_index", "index", "task_index"],
        use_threads=False,
    )
    expected_episode = unit.episode_start + local_index
    expected_frame_start = unit.frame_start + frame_offset
    expected_task = unit.task_indices[local_index]
    samples = sorted({0, source.length // 2, source.length - 1})
    for row in samples:
        if int(generated["episode_index"][row].as_py()) != expected_episode:
            raise ConversionError(f"RoboCOIN episode index checkpoint mismatch: {data_path}")
        if int(generated["frame_index"][row].as_py()) != row:
            raise ConversionError(f"RoboCOIN frame index checkpoint mismatch: {data_path}")
        if int(generated["index"][row].as_py()) != expected_frame_start + row:
            raise ConversionError(f"RoboCOIN global frame checkpoint mismatch: {data_path}")
        if int(generated["task_index"][row].as_py()) != expected_task:
            raise ConversionError(f"RoboCOIN task index checkpoint mismatch: {data_path}")
    schema_text = str(parquet.schema_arrow)
    video_records = []
    camera_by_key = {
        camera.feature_key: camera for camera in unit.payload.partition.plan.camera_features
    }
    for (key, _source_path), video_path in zip(
        source.video_paths, video_paths, strict=True
    ):
        frames, height, width, fps, codec, pix_fmt = _video_frame_count(video_path)
        camera = camera_by_key[key]
        if (
            frames != source.length
            or (height, width) != (camera.height, camera.width)
            or fps is None
            or abs(fps - unit.payload.partition.plan.fps) > 1e-6
        ):
            raise ConversionError(f"RoboCOIN episode video checkpoint mismatch: {video_path}")
        video_records.append(
            {
                "path": video_path.relative_to(root).as_posix(),
                "size": video_path.stat().st_size,
                "frames": frames,
                "height": height,
                "width": width,
                "fps": fps,
                "codec": codec,
                "pixel_format": pix_fmt,
            }
        )
    return {
        "data": {
            "path": data_path.relative_to(root).as_posix(),
            "size": data_path.stat().st_size,
            "rows": parquet.metadata.num_rows,
            "schema_sha256": hashlib.sha256(schema_text.encode()).hexdigest(),
            "generated_samples": [
                {
                    "row": row,
                    "episode_index": int(generated["episode_index"][row].as_py()),
                    "frame_index": int(generated["frame_index"][row].as_py()),
                    "index": int(generated["index"][row].as_py()),
                    "task_index": int(generated["task_index"][row].as_py()),
                }
                for row in samples
            ],
        },
        "videos": video_records,
    }


def _discard_episode_output(root: Path, source: Any, local_index: int) -> None:
    data_path, video_paths = _episode_output_paths(root, source, local_index)
    data_path.unlink(missing_ok=True)
    for path in video_paths:
        path.unlink(missing_ok=True)
    _episode_checkpoint_path(root, local_index).unlink(missing_ok=True)


def _episode_checkpoint_valid(
    source: Any,
    *,
    local_index: int,
    unit: ParallelWorkUnit,
    root: Path,
    frame_offset: int,
) -> bool:
    marker_path = _episode_checkpoint_path(root, local_index)
    if not marker_path.is_file():
        return False
    try:
        marker = read_json_object(marker_path, "RoboCOIN episode checkpoint")
        if marker.get("schema_version") != 1:
            return False
        expected = {
            "task_fingerprint": unit.fingerprint,
            "unit_key": unit.key,
            "local_episode_index": local_index,
            "source_id": source.source_id,
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            return False
        return marker.get("output") == _episode_output_evidence(
            source,
            local_index=local_index,
            unit=unit,
            root=root,
            frame_offset=frame_offset,
        )
    except (ConversionError, OSError, ValueError):
        return False


def _build_episode_checkpoint(
    source: Any,
    *,
    local_index: int,
    unit: ParallelWorkUnit,
    root: Path,
    frame_offset: int,
    guard: DiskGuard,
) -> None:
    _discard_episode_output(root, source, local_index)
    try:
        _copy_episode(
            source,
            local_index=local_index,
            unit=unit,
            root=root,
            frame_offset=frame_offset,
            guard=guard,
        )
        output = _episode_output_evidence(
            source,
            local_index=local_index,
            unit=unit,
            root=root,
            frame_offset=frame_offset,
        )
        atomic_write_json(
            _episode_checkpoint_path(root, local_index),
            {
                "schema_version": 1,
                "task_fingerprint": unit.fingerprint,
                "unit_key": unit.key,
                "local_episode_index": local_index,
                "source_id": source.source_id,
                "output": output,
            },
        )
    except BaseException:
        _discard_episode_output(root, source, local_index)
        raise


def _build_task_unit(unit: ParallelWorkUnit, *, workers: int, guard: DiskGuard) -> dict[str, Any]:
    """Build one whole task locally; workers only parallelize episodes inside it."""

    payload = unit.payload
    if not isinstance(payload, UnitPayload):
        raise ConversionError(f"invalid RoboCOIN task payload for {unit.key}")
    root = Path(unit.target_path)
    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root / "meta", ignore_errors=True)
    (root / "conversion_manifest.json").unlink(missing_ok=True)
    verified_marker_path(unit).unlink(missing_ok=True)
    offsets: list[int] = []
    cursor = 0
    for source in payload.sources:
        offsets.append(cursor)
        cursor += source.length
    try:
        pending: list[tuple[int, Any]] = []
        for index, source in enumerate(payload.sources):
            if _episode_checkpoint_valid(
                source,
                local_index=index,
                unit=unit,
                root=root,
                frame_offset=offsets[index],
            ):
                continue
            _discard_episode_output(root, source, index)
            pending.append((index, source))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="robocoin-task") as pool:
            futures = [
                pool.submit(
                    _build_episode_checkpoint,
                    source,
                    local_index=index,
                    unit=unit,
                    root=root,
                    frame_offset=offsets[index],
                    guard=guard,
                )
                for index, source in pending
            ]
            for future in futures:
                future.result()
        import pyarrow.parquet as pq

        metadata = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            _episode_metadata_table(payload.sources, fps=payload.partition.plan.fps),
            metadata,
            compression="snappy",
            use_dictionary=True,
        )
        info = _normalise_info(payload.partition.source_info, payload.partition.plan)
        v3_info_keys = {
            "codebase_version", "fps", "features", "total_episodes", "total_frames",
            "total_tasks", "chunks_size", "data_files_size_in_mb",
            "video_files_size_in_mb", "data_path", "video_path", "robot_type",
            "splits", "tools",
        }
        info = {key: value for key, value in info.items() if key in v3_info_keys}
        info.update(
            total_episodes=len(payload.sources),
            total_frames=unit.weight,
            total_tasks=len(set(source.instruction for source in payload.sources)),
            splits={"train": f"0:{len(payload.sources)}"},
        )
        atomic_write_json(root / "meta" / "info.json", info)
        atomic_write_json(
            root / "conversion_manifest.json",
            {
                "source_format": "robocoin_lerobot_v2.1",
                "conversion_policy_version": ROBOCOIN_CONVERSION_POLICY_VERSION,
                "task": unit.key,
                "schema_policy": payload.partition.plan.extra.get("schema_policy"),
                "source_parquet_schema_variants": payload.partition.plan.extra.get(
                    "source_parquet_schema_variants"
                ),
                "source_references": [
                    {
                        "source_id": source.source_id,
                        "data_path": str(source.data_path),
                        "data_size": source.data_bytes,
                        "videos": [
                            {"field": key, "path": str(path), "size": path.stat().st_size}
                            for key, path in source.video_paths
                        ],
                    }
                    for source in payload.sources
                ],
                "episode_range": [unit.episode_start, unit.episode_end],
                "frame_range": [unit.frame_start, unit.frame_end],
            },
        )
        _validate_local_unit(unit)
        write_verified_unit_marker(unit)
    except BaseException:
        # Per-episode markers are durable resume points.  Only the episode
        # whose write failed is discarded by _build_episode_checkpoint.
        verified_marker_path(unit).unlink(missing_ok=True)
        raise
    return {"frames": unit.weight, "bytes": directory_size(root)}


def _task_unit(
    entry: RobocoinTaskCatalogEntry,
    partition: RobogenePartition,
    *,
    partition_instructions: list[str],
    task_fingerprint: str,
    local_root: Path,
    run_id: str,
    estimated_peak_bytes: int,
) -> ParallelWorkUnit:
    task_indices = {task: index for index, task in enumerate(partition_instructions)}
    sources = partition.episodes
    if len(sources) != entry.total_episodes or partition.plan.num_frames != entry.total_frames:
        raise ConversionError(
            f"RoboCOIN task plan totals changed from catalog for {entry.key}: "
            f"episodes={len(sources)}/{entry.total_episodes}, "
            f"frames={partition.plan.num_frames}/{entry.total_frames}"
        )
    return ParallelWorkUnit(
        index=entry.partition_unit_index,
        key=entry.key,
        dataset_uid=partition.plan.dataset_uid,
        target_path=str(local_root / "work" / run_id / entry.partition / entry.key),
        episode_start=entry.partition_episode_start,
        episode_end=entry.partition_episode_start + entry.total_episodes,
        frame_start=entry.partition_frame_start,
        frame_end=entry.partition_frame_start + entry.total_frames,
        task_indices=tuple(task_indices[episode.instruction] for episode in sources),
        weight=entry.total_frames,
        estimated_memory_bytes=max(source.data_bytes for source in partition.episodes),
        estimated_temp_bytes=estimated_peak_bytes,
        fingerprint=task_fingerprint,
        payload=UnitPayload(partition, partition.episodes),
    )


def _validate_task_plan_sources(partition: RobogenePartition, raw_root: Path) -> None:
    approved = raw_root.resolve(strict=False)
    for record in partition.source_files:
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ConversionError("invalid source path in cached RoboCOIN task plan")
        path = raw_root / relative
        if not path.resolve(strict=False).is_relative_to(approved):
            raise ConversionError(f"cached RoboCOIN source escapes raw root: {path}")
        stat = path.stat()
        if stat.st_size != int(record.get("size", -1)) or stat.st_mtime_ns != int(record.get("mtime_ns", -1)):
            raise ConversionError(f"RoboCOIN task source fingerprint changed: {path}")


def _load_or_preflight_task(
    entry: RobocoinTaskCatalogEntry,
    *,
    args: argparse.Namespace,
    raw_root: Path,
    output_root: Path,
    allow_cached: bool,
    skip_source_validation: bool = False,
) -> tuple[RobogenePartition, str, dict[str, Any], bool]:
    path = _task_plan_path(args.local_work_root, entry.key)
    if allow_cached and path.is_file():
        partition, fingerprint, evidence = task_plan_from_payload(
            read_json_object(path, "RoboCOIN task plan")
        )
        if not skip_source_validation:
            _validate_task_plan_sources(partition, raw_root)
        return partition, fingerprint, evidence, True
    partition, fingerprint, evidence = inspect_robocoin_task(
        entry,
        output_uid=args.output_dataset_uid,
        output_root=output_root,
        limit_episodes=args.limit_episodes_per_task,
    )
    if not args.inspect_only:
        payload = task_plan_to_payload(
            partition, task_fingerprint=fingerprint, evidence=evidence
        )
        if path.exists():
            existing = read_json_object(path, "RoboCOIN task plan")
            if existing != payload:
                raise ConversionError(f"immutable RoboCOIN task plan changed: {path}")
        else:
            atomic_write_json(path, payload)
    return partition, fingerprint, evidence, False


def _warmup(partition: RobogenePartition, root: Path, encoder_threads: int) -> dict[str, Any]:
    """Decode and CPU-encode 30 real frames once, then remove the artifact."""

    from convert_core.lerobot_writer import _video_frame_count
    import av

    source = partition.episodes[0].video_paths[0][1]
    work = root / "warmup"
    output = work / "cpu-h264-30-frames.mp4"
    started = time.monotonic()
    try:
        work.mkdir(parents=True, exist_ok=True)
        count = 0
        with av.open(str(source), mode="r") as incoming, av.open(str(output), mode="w") as outgoing:
            input_stream = incoming.streams.video[0]
            stream = outgoing.add_stream("h264", rate=partition.plan.fps)
            stream.width = int(input_stream.width)
            stream.height = int(input_stream.height)
            stream.pix_fmt = "yuv420p"
            stream.thread_count = encoder_threads
            for frame in incoming.decode(input_stream):
                for packet in stream.encode(frame):
                    outgoing.mux(packet)
                count += 1
                if count == 30:
                    break
            for packet in stream.encode():
                outgoing.mux(packet)
        frames, height, width, fps, codec, pix_fmt = _video_frame_count(output)
        if count != 30 or frames != 30 or fps is None or abs(fps - partition.plan.fps) > 1e-6:
            raise ConversionError(f"RoboCOIN CPU warmup validation failed: {frames=} {fps=}")
        return {
            "frames": frames,
            "height": height,
            "width": width,
            "fps": fps,
            "codec": codec,
            "pixel_format": pix_fmt,
            "encoder_threads": encoder_threads,
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _catalog_options(args: argparse.Namespace, raw_root: Path) -> dict[str, Any]:
    return {
        "conversion_policy_version": ROBOCOIN_CONVERSION_POLICY_VERSION,
        "raw_root": str(raw_root),
        "output_uid": args.output_dataset_uid,
        "tasks": sorted(args.task or []),
        "start_task": args.start_task,
        "max_tasks": args.max_tasks,
        "limit_episodes_per_task": args.limit_episodes_per_task,
        "video_conversion": "container-copy-no-resample",
    }


def _load_catalog(
    args: argparse.Namespace, raw_root: Path
) -> tuple[RobocoinCatalog, str, str, bool]:
    state_path = _catalog_state_path(args.local_work_root)
    options = _catalog_options(args, raw_root)
    if args.resume and state_path.is_file():
        state = read_json_object(state_path, "RoboCOIN catalog state")
        if state.get("options") != options:
            raise ConversionError("RoboCOIN resume selection/output options changed")
        catalog = catalog_from_payload(dict(state["catalog"]))
        fingerprint = canonical_fingerprint({"catalog": catalog_to_payload(catalog), "options": options})
        if state.get("fingerprint") != fingerprint:
            raise ConversionError("RoboCOIN catalog fingerprint changed")
        return catalog, fingerprint, str(state["run_id"]), True
    catalog = build_robocoin_catalog(
        raw_root,
        task_names=set(args.task) if args.task else None,
        start_task=args.start_task,
        max_tasks=args.max_tasks,
        limit_episodes_per_task=args.limit_episodes_per_task,
    )
    fingerprint = canonical_fingerprint({"catalog": catalog_to_payload(catalog), "options": options})
    run_id = args.run_id or _run_id()
    if not args.inspect_only:
        atomic_write_json(
            state_path,
            {
                "schema_version": 2,
                "conversion_policy_version": ROBOCOIN_CONVERSION_POLICY_VERSION,
                "fingerprint": fingerprint,
                "run_id": run_id,
                "options": options,
                "catalog": catalog_to_payload(catalog),
            },
        )
    return catalog, fingerprint, run_id, False


def _combined_partition_plan(
    name: str,
    task_plans: list[RobogenePartition],
    *,
    output_uid: str,
    output_root: Path,
) -> DatasetConversionPlan:
    first = task_plans[0].plan
    episodes = tuple(episode for task in task_plans for episode in task.plan.episodes)
    return replace(
        first,
        dataset_uid=f"{output_uid}-{name}",
        output_path=output_root / output_uid / name,
        episodes=episodes,
        extra={
            **first.extra,
            "source_tasks": [task.plan.extra["source_task"] for task in task_plans],
            "task_chunk_alignment": True,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-work-root", type=Path, default=DEFAULT_LOCAL_WORK_ROOT)
    parser.add_argument("--output-dataset-uid", type=_output_uid, default="robocoin")
    parser.add_argument("--min-local-free-bytes", type=_positive_int, default=DEFAULT_MIN_LOCAL_FREE_BYTES)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--encoder-threads-per-worker", type=_positive_int, default=8)
    parser.add_argument("--upload-workers", type=_positive_int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inspect-only", "--dry-run", dest="inspect_only", action="store_true")
    parser.add_argument("--task", action="append", help="exact top-level RoboCOIN task key; repeatable")
    parser.add_argument("--start-task")
    parser.add_argument("--max-tasks", type=_positive_int)
    parser.add_argument("--limit-episodes-per-task", type=_positive_int)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--skip-warmup", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-task", type=_positive_int, help=argparse.SUPPRESS)
    parser.add_argument("--fail-upload-task", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_root, output_root = validate_source_and_output_roots(args.raw_root, args.output_root)
    if raw_root.resolve(strict=False) != DEFAULT_RAW_ROOT.resolve(strict=False):
        raise ConversionError(f"--raw-root must be the approved RoboCOIN source {DEFAULT_RAW_ROOT}")
    if output_root.resolve(strict=False) != DEFAULT_OUTPUT_ROOT.resolve(strict=False):
        raise ConversionError(f"--output-root must be the approved staging root {DEFAULT_OUTPUT_ROOT}")
    if args.local_work_root.resolve(strict=False) != DEFAULT_LOCAL_WORK_ROOT.resolve(strict=False):
        raise ConversionError(f"--local-work-root must be {DEFAULT_LOCAL_WORK_ROOT}")
    if args.upload_workers != 1:
        raise ConversionError("RoboCOIN task-sequential commit requires --upload-workers 1")
    subset_selected = bool(args.task or args.start_task or args.max_tasks or args.limit_episodes_per_task)
    if subset_selected and not args.inspect_only and args.output_dataset_uid == "robocoin":
        raise ConversionError("subset conversion requires an independent --output-dataset-uid")
    final_root = output_root / args.output_dataset_uid
    if not args.inspect_only and (final_root / "_SUCCESS").exists():
        raise ConversionError(f"completed output already exists: {final_root}")

    if args.inspect_only:
        catalog = build_robocoin_catalog(
            raw_root,
            task_names=set(args.task) if args.task else None,
            start_task=args.start_task,
            max_tasks=args.max_tasks,
            limit_episodes_per_task=args.limit_episodes_per_task,
        )
        print(json.dumps(catalog.fingerprint_payload, ensure_ascii=False, indent=2))
        for entry in catalog.tasks:
            partition, fingerprint, evidence = inspect_robocoin_task(
                entry,
                output_uid=args.output_dataset_uid,
                output_root=output_root,
                limit_episodes=args.limit_episodes_per_task,
            )
            print(
                f"preflight task={entry.key} partition={entry.partition} "
                f"episodes={len(partition.episodes)} frames={partition.plan.num_frames} "
                f"files={evidence['source_file_count']} peak_bytes={evidence['estimated_peak_bytes']} "
                f"fingerprint={fingerprint}"
            )
        return 0

    args.local_work_root.mkdir(parents=True, exist_ok=True)
    lock = args.local_work_root / "resume" / "robocoin.lock"
    with exclusive_staging_lock(lock):
        invocation_started = time.monotonic()
        catalog, catalog_fingerprint, run_id, resumed_catalog = _load_catalog(args, raw_root)
        catalog_load_seconds = time.monotonic() - invocation_started
        if (final_root / "_INCOMPLETE").exists() and not args.resume:
            raise ConversionError("incomplete RoboCOIN output exists; rerun with --resume")
        for path in (
            args.local_work_root / "work" / run_id,
            args.local_work_root / "resume" / "task_plans",
            args.local_work_root / "cache" / run_id,
            args.local_work_root / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)
        cache_root = args.local_work_root / "cache" / run_id
        runtime_paths = {
            "TMPDIR": args.local_work_root / "work" / run_id / "tmp",
            "XDG_CACHE_HOME": cache_root / "xdg",
            "HF_HOME": cache_root / "huggingface",
            "PYTHONPYCACHEPREFIX": cache_root / "pycache",
        }
        for key, path in runtime_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        guard = DiskGuard(
            args.local_work_root,
            usage_roots=(args.local_work_root,),
            min_free_bytes=args.min_local_free_bytes,
            max_local_bytes=None,
            interval_seconds=1.0,
            enforce_policy_floor=False,
        )
        create_incomplete_output(final_root, fingerprint=catalog_fingerprint, run_id=run_id)
        task_plans: dict[str, list[RobogenePartition]] = {}
        task_units: dict[str, list[ParallelWorkUnit]] = {}
        progress_path = args.local_work_root / "resume" / "robocoin_progress.json"
        if args.resume and progress_path.is_file():
            progress = read_json_object(progress_path, "RoboCOIN progress log")
            if progress.get("catalog_fingerprint") != catalog_fingerprint:
                raise ConversionError("RoboCOIN progress log fingerprint changed")
            task_reports = list(progress.get("task_events", []))
            run_events = list(progress.get("run_events", []))
        else:
            task_reports = []
            run_events = []
        run_events.append(
            {
                "catalog_resumed": resumed_catalog,
                "catalog_startup_seconds": catalog_load_seconds,
                "unix_time": time.time(),
            }
        )

        def save_progress() -> None:
            atomic_write_json(
                progress_path,
                {
                    "schema_version": 1,
                    "catalog_fingerprint": catalog_fingerprint,
                    "run_events": run_events,
                    "task_events": task_reports,
                },
            )

        save_progress()
        physical_schemas: dict[str, str] = {}
        partition_instructions: dict[str, list[str]] = {}
        warmup_marker = args.local_work_root / "resume" / "robocoin_warmup.json"

        for entry in catalog.tasks:
            plan_path = _task_plan_path(args.local_work_root, entry.key)
            marker_path = (
                args.local_work_root
                / "resume"
                / "committed"
                / entry.partition
                / f"unit-{entry.partition_unit_index:06d}.json"
            )
            committed = False
            if marker_path.is_file():
                marker = read_json_object(marker_path, "RoboCOIN task commit marker")
                committed = marker.get("status") == "verified" and marker.get("unit_key") == entry.key
                if committed and not plan_path.is_file():
                    raise ConversionError(
                        f"committed task {entry.key} is missing its compact local task plan; "
                        "refusing to rescan committed source"
                    )
            started = time.monotonic()
            partition, task_fingerprint, evidence, cached = _load_or_preflight_task(
                entry,
                args=args,
                raw_root=raw_root,
                output_root=output_root,
                allow_cached=args.resume,
                skip_source_validation=committed,
            )
            schema = str(partition.plan.extra["parquet_schema_fingerprint"])
            previous_schema = physical_schemas.setdefault(entry.partition, schema)
            if previous_schema != schema:
                raise ConversionError(
                    f"catalog-compatible RoboCOIN tasks have different physical schemas in {entry.partition}"
                )
            instruction_order = partition_instructions.setdefault(entry.partition, [])
            for episode in partition.plan.episodes:
                if episode.instruction not in instruction_order:
                    instruction_order.append(episode.instruction)
            unit = _task_unit(
                entry,
                partition,
                partition_instructions=instruction_order,
                task_fingerprint=task_fingerprint,
                local_root=args.local_work_root,
                run_id=run_id,
                estimated_peak_bytes=int(evidence["estimated_peak_bytes"]),
            )
            task_plans.setdefault(entry.partition, []).append(partition)
            task_units.setdefault(entry.partition, []).append(unit)
            if committed:
                task_reports.append(
                    {
                        "task": entry.key,
                        "status": "skipped-committed-marker",
                        "startup_seconds": time.monotonic() - started,
                        "source_files_read": 0,
                    }
                )
                save_progress()
                continue
            snapshot = filesystem_snapshot(args.local_work_root)
            required = int(evidence["estimated_peak_bytes"])
            if snapshot.available_bytes - args.min_local_free_bytes < required:
                raise ConversionError(
                    f"RoboCOIN task {entry.key} needs estimated local peak {required} bytes, "
                    f"but only {snapshot.available_bytes - args.min_local_free_bytes} bytes are available above reserve"
                )
            if not args.skip_warmup and not warmup_marker.is_file():
                warmup = _warmup(partition, cache_root, args.encoder_threads_per_worker)
                atomic_write_json(
                    warmup_marker,
                    {"schema_version": 1, "catalog_fingerprint": catalog_fingerprint, **warmup},
                )
            conversion_started = time.monotonic()
            reuse_verified = False
            if args.resume and Path(unit.target_path).is_dir() and verified_marker_path(unit).is_file():
                read_verified_unit_marker(unit)
                reuse_verified = True
            else:
                _build_task_unit(unit, workers=args.workers, guard=guard)
            local_seconds = time.monotonic() - conversion_started
            if args.fail_upload_task == entry.key:
                raise ConversionError(
                    f"injected RoboCOIN upload failure for {entry.key}; verified local task retained"
                )
            upload_started = time.monotonic()
            marker = commit_verified_unit(
                unit,
                partition_name=entry.partition,
                partition_root=final_root / entry.partition,
                resume_root=args.local_work_root / "resume",
                verify_remote_sha256=args.output_dataset_uid != "robocoin",
                trust_verified_marker=reuse_verified,
            )
            task_reports.append(
                {
                    "task": entry.key,
                    "status": "committed",
                    "catalog_resumed": resumed_catalog,
                    "task_plan_cached": cached,
                    "reused_verified_local": reuse_verified,
                    "source_files_read": evidence["source_file_count"],
                    "local_conversion_seconds": local_seconds,
                    "upload_seconds": time.monotonic() - upload_started,
                    "commit_marker": str(marker_path.relative_to(args.local_work_root)),
                    "remote_files": len(marker.get("bulk", [])),
                }
            )
            save_progress()
            if args.stop_after_task is not None and len(task_reports) >= args.stop_after_task:
                raise ConversionError(
                    f"injected stop after {args.stop_after_task} committed RoboCOIN task(s)"
                )

        partition_reports = []
        for name in sorted(task_plans):
            plan = _combined_partition_plan(
                name,
                task_plans[name],
                output_uid=args.output_dataset_uid,
                output_root=output_root,
            )
            units = tuple(task_units[name])
            finalize_direct_partition(
                plan,
                units,
                final_root / name,
                resume_root=args.local_work_root / "resume",
                reader_format="robocoin_lerobot_v21",
                parallel_evidence={
                    "task_sequential": True,
                    "workers_within_task": args.workers,
                    "encoder_threads_per_worker": args.encoder_threads_per_worker,
                    "upload_workers": 1,
                    "fixed_local_cap_bytes": None,
                    "min_local_free_bytes": args.min_local_free_bytes,
                },
            )
            partition_reports.append(
                {
                    "partition": name,
                    "episodes": len(plan.episodes),
                    "frames": plan.num_frames,
                    "tasks": [task.plan.extra["source_task"] for task in task_plans[name]],
                }
            )
        actual_plans_by_task = {
            str(partition.plan.extra["source_task"]): partition
            for partitions in task_plans.values()
            for partition in partitions
        }
        root_task_order: list[str] = []
        for entry in catalog.tasks:
            for episode in actual_plans_by_task[entry.key].plan.episodes:
                if episode.instruction not in root_task_order:
                    root_task_order.append(episode.instruction)
        root_task_index = {task: index for index, task in enumerate(root_task_order)}
        manifest = {
            "format": "lerobot_v3_0_schema_partitioned_collection",
            "source_format": "robocoin_lerobot_v2.1",
            "conversion_policy_version": ROBOCOIN_CONVERSION_POLICY_VERSION,
            "catalog_fingerprint": catalog_fingerprint,
            "schema_policies": {
                task: partition.plan.extra["schema_policy"]
                for task, partition in sorted(actual_plans_by_task.items())
            },
            "source_parquet_schema_variants": {
                task: partition.plan.extra["source_parquet_schema_variants"]
                for task, partition in sorted(actual_plans_by_task.items())
            },
            "task_order": [entry.key for entry in catalog.tasks],
            "root_global_ranges": [
                {
                    "task": entry.key,
                    "episode_range": [entry.root_episode_start, entry.root_episode_start + entry.total_episodes],
                    "frame_range": [entry.root_frame_start, entry.root_frame_start + entry.total_frames],
                    "partition": entry.partition,
                    "partition_episode_range": [entry.partition_episode_start, entry.partition_episode_start + entry.total_episodes],
                    "partition_frame_range": [entry.partition_frame_start, entry.partition_frame_start + entry.total_frames],
                    "chunk_index": entry.partition_unit_index,
                    "root_task_indices": sorted(
                        {
                            root_task_index[episode.instruction]
                            for episode in actual_plans_by_task[entry.key].plan.episodes
                        }
                    ),
                }
                for entry in catalog.tasks
            ],
            "partitions": partition_reports,
            "root_tasks": [
                {"task_index": index, "task": task}
                for index, task in enumerate(root_task_order)
            ],
            "tasks": task_reports,
            "run_events": run_events,
            "success_marker_required": True,
        }
        atomic_write_json(final_root / "conversion_manifest.json", manifest)
        publish_success(
            final_root,
            fingerprint=catalog_fingerprint,
            evidence={"conversion_manifest": "conversion_manifest.json", "partitions": partition_reports},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
