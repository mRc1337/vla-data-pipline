"""Bounded, resumable RoboGene v2.1 to LeRobot v3.0 conversion.

The release already contains one Parquet and MP4 per legacy episode.  This
converter deliberately keeps that granularity: workers copy a complete unit to
local POSIX storage, rebase only generated indices, validate it, and the
uploader copies directly to deterministic final v3 chunk paths.  Finalization
only writes compact v3 metadata and statistics; it never concatenates bulk
Parquet or MP4 files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any

sys.dont_write_bytecode = True

from convert_core.checkpoint import atomic_write_json, canonical_fingerprint, read_json_object
from convert_core.direct_commit import (
    commit_verified_unit,
    finalize_direct_partition,
    prepare_direct_commits,
)
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
from convert_core.storage import DiskGuard, directory_size
from readers.robogene_reader import (
    RobogeneCatalog,
    RobogeneEpisodeSource,
    RobogenePartition,
    _normalise_info,
    catalog_from_payload,
    catalog_to_payload,
    inspect_robogene,
    validate_catalog_source_files,
)


DEFAULT_RAW_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw/robogene")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0")
DEFAULT_LOCAL_WORK_ROOT = Path.home() / "robogene_staging"
DEFAULT_MAX_LOCAL_TEMP_BYTES = 100_000_000_000
DEFAULT_MIN_LOCAL_FREE_BYTES = 200_000_000_000
COPY_BLOCK_BYTES = 64 * 1024 * 1024


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"


def _validate_run_id(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or path.name != value or ".." in path.parts:
        raise ConversionError(f"invalid RoboGene run id: {value!r}")
    return value


def _copy_stream(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        while chunk := incoming.read(COPY_BLOCK_BYTES):
            outgoing.write(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _set_column(table: Any, name: str, values: Any) -> Any:
    import pyarrow as pa

    index = table.schema.get_field_index(name)
    if index < 0:
        raise ConversionError(f"RoboGene Parquet is missing generated column {name!r}")
    return table.set_column(index, table.schema.field(index), pa.array(values, type=table.schema.field(index).type))


def _localise_data_file(
    path: Path,
    source: RobogeneEpisodeSource,
    *,
    global_episode: int,
    global_frame: int,
    global_task: int,
) -> None:
    """Rebase only v2.1 generated columns while retaining every payload field."""

    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if table.num_rows != source.length:
        raise ConversionError(f"local data row count changed for {path}: {table.num_rows} != {source.length}")
    frame_values = table["frame_index"].to_pylist()
    if [int(value) for value in frame_values] != list(range(source.length)):
        raise ConversionError(f"source frame_index is not contiguous in {path}")
    table = _set_column(table, "episode_index", np.full(source.length, global_episode, dtype=np.int64))
    table = _set_column(table, "index", np.arange(global_frame, global_frame + source.length, dtype=np.int64))
    table = _set_column(table, "task_index", np.full(source.length, global_task, dtype=np.int64))
    temporary = path.with_name(f".{path.name}.localising")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        os.replace(temporary, path)  # local same-filesystem publication only
    finally:
        temporary.unlink(missing_ok=True)


def _episode_metadata_table(
    sources: tuple[RobogeneEpisodeSource, ...], *, fps: int
) -> Any:
    import pyarrow as pa

    rows: list[dict[str, Any]] = []
    frame_cursor = 0
    for local_episode, source in enumerate(sources):
        row: dict[str, Any] = {
            "episode_index": local_episode,
            "tasks": [source.instruction],
            "length": source.length,
            "data/chunk_index": 0,
            "data/file_index": local_episode,
            "dataset_from_index": frame_cursor,
            "dataset_to_index": frame_cursor + source.length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for key, _ in source.video_paths:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = local_episode
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = source.length / float(fps)
        stats = source.stats.get("stats")
        if not isinstance(stats, dict):
            raise ConversionError(f"missing episode stats for {source.source_id}")
        for feature, values in stats.items():
            if not isinstance(values, dict):
                raise ConversionError(f"invalid stats for {source.source_id}:{feature}")
            for stat, value in values.items():
                row[f"stats/{feature}/{stat}"] = value
        rows.append(row)
        frame_cursor += source.length
    return pa.Table.from_pylist(rows)


@dataclass(frozen=True)
class UnitPayload:
    partition: RobogenePartition
    sources: tuple[RobogeneEpisodeSource, ...]


def _validate_local_unit(unit: ParallelWorkUnit) -> None:
    import pyarrow.parquet as pq
    from convert_core.lerobot_writer import _video_frame_count, validate_parquet_feature_schema

    payload = unit.payload
    if not isinstance(payload, UnitPayload):
        raise ConversionError(f"invalid RoboGene work payload for {unit.key}")
    root = Path(unit.target_path)
    data = sorted((root / "data").rglob("*.parquet"))
    if len(data) != len(payload.sources):
        raise ConversionError(f"unit {unit.key} has {len(data)} data files, expected {len(payload.sources)}")
    rows = 0
    for path, source in zip(data, payload.sources, strict=True):
        footer = pq.ParquetFile(path).metadata
        if footer.num_rows != source.length:
            raise ConversionError(f"unit {unit.key} row count changed in {path}")
        rows += footer.num_rows
    if rows != unit.weight:
        raise ConversionError(f"unit {unit.key} frame total changed: {rows} != {unit.weight}")
    validate_parquet_feature_schema(payload.partition.plan, root)
    metadata = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not metadata.is_file() or pq.ParquetFile(metadata).metadata.num_rows != len(payload.sources):
        raise ConversionError(f"unit {unit.key} episode metadata is incomplete")
    for source_index, source in enumerate(payload.sources):
        camera_by_key = {camera.feature_key: camera for camera in payload.partition.plan.camera_features}
        for key, _ in source.video_paths:
            path = root / "videos" / key / "chunk-000" / f"file-{source_index:03d}.mp4"
            frames, height, width, fps, _codec, _pix_fmt = _video_frame_count(path)
            camera = camera_by_key[key]
            if frames != source.length or (height, width) != (camera.height, camera.width) or fps is None or abs(fps - payload.partition.plan.fps) > 1e-6:
                raise ConversionError(f"unit {unit.key} video validation failed: {path}")


def _encoder_warmup(partition: RobogenePartition, local_root: Path) -> None:
    """Decode and CPU-encode 30 real frames once, then remove the artifact."""

    from convert_core.lerobot_writer import _video_frame_count

    if not partition.episodes or not partition.episodes[0].video_paths:
        raise ConversionError("RoboGene encoder warmup has no source video")
    warmup_root = local_root / "warmup"
    output = warmup_root / "cpu-h264-warmup.mp4"
    try:
        import av

        source = partition.episodes[0].video_paths[0][1]
        warmup_root.mkdir(parents=True, exist_ok=True)
        count = 0
        with av.open(str(source), mode="r") as incoming, av.open(str(output), mode="w") as outgoing:
            input_stream = incoming.streams.video[0]
            stream = outgoing.add_stream("h264", rate=partition.plan.fps)
            stream.width, stream.height = int(input_stream.width), int(input_stream.height)
            stream.pix_fmt = "yuv420p"
            for frame in incoming.decode(input_stream):
                for packet in stream.encode(frame):
                    outgoing.mux(packet)
                count += 1
                if count == 30:
                    break
            for packet in stream.encode():
                outgoing.mux(packet)
        frames, height, width, fps, codec, pix_fmt = _video_frame_count(output)
        camera = partition.plan.camera_features[0]
        if count != 30 or frames != 30 or (height, width) != (camera.height, camera.width) or fps is None or abs(fps - partition.plan.fps) > 1e-6 or codec != "h264" or pix_fmt != "yuv420p":
            raise ConversionError(f"RoboGene CPU encoder warmup failed validation: frames={frames} size={width}x{height} fps={fps} codec={codec} pix_fmt={pix_fmt}")
    except Exception as exc:
        if isinstance(exc, ConversionError):
            raise
        raise ConversionError(f"RoboGene CPU encoder warmup failed: {exc}") from exc
    finally:
        shutil.rmtree(warmup_root, ignore_errors=True)


def _encoder_warmup_once(
    catalog: RobogeneCatalog, *, local_root: Path, fingerprint: str, run_id: str
) -> None:
    marker_path = local_root / "resume" / "robogene_encoder_warmup.json"
    if marker_path.is_file():
        marker = read_json_object(marker_path, "RoboGene encoder warmup marker")
        if marker.get("fingerprint") == fingerprint:
            return
        marker_path.unlink(missing_ok=True)
    _encoder_warmup(catalog.partitions[0], local_root / "cache" / run_id)
    atomic_write_json(
        marker_path,
        {"schema_version": 1, "fingerprint": fingerprint, "status": "passed"},
    )


def _build_unit(unit: ParallelWorkUnit) -> dict[str, Any]:
    payload = unit.payload
    if not isinstance(payload, UnitPayload):
        raise ConversionError(f"invalid RoboGene work payload for {unit.key}")
    root = Path(unit.target_path)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    frame_cursor = 0
    try:
        for local_index, source in enumerate(payload.sources):
            destination = root / "data" / "chunk-000" / f"file-{local_index:03d}.parquet"
            _copy_stream(source.data_path, destination)
            _localise_data_file(
                destination,
                source,
                global_episode=unit.episode_start + local_index,
                global_frame=unit.frame_start + frame_cursor,
                global_task=unit.task_indices[local_index],
            )
            for key, video_source in source.video_paths:
                _copy_stream(video_source, root / "videos" / key / "chunk-000" / f"file-{local_index:03d}.mp4")
            frame_cursor += source.length
        metadata = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq

        pq.write_table(_episode_metadata_table(payload.sources, fps=payload.partition.plan.fps), metadata, compression="snappy", use_dictionary=True)
        info = _normalise_info(payload.partition.source_info, payload.partition.plan)
        info.update(total_episodes=len(payload.sources), total_frames=unit.weight, total_tasks=len(set(source.instruction for source in payload.sources)), splits={"train": f"0:{len(payload.sources)}"})
        atomic_write_json(root / "meta" / "info.json", info)
        atomic_write_json(root / "conversion_manifest.json", {"source_format": "lerobot_v2.1", "unit": unit.key, "sources": [source.source_id for source in payload.sources]})
        _validate_local_unit(unit)
        write_verified_unit_marker(unit)
    except BaseException:
        # An incomplete work unit is not a valid resume point.
        shutil.rmtree(root, ignore_errors=True)
        root.with_name(f"{root.name}.verified.json").unlink(missing_ok=True)
        raise
    return {"unit": unit.key, "frames": unit.weight, "local_bytes": directory_size(root)}


def _units_for_partition(
    partition: RobogenePartition,
    *,
    work_root: Path,
    run_id: str,
    max_local_temp_bytes: int,
    max_inflight_units: int,
    fingerprint: str,
) -> tuple[ParallelWorkUnit, ...]:
    # The scheduler, rather than arbitrary per-worker slicing, enforces the
    # global local quota.  A single source episode may therefore use the
    # complete quota when it is the only inflight unit.
    per_unit_limit = max_local_temp_bytes
    groups: list[list[RobogeneEpisodeSource]] = []
    current: list[RobogeneEpisodeSource] = []
    current_peak = 0
    for source in partition.episodes:
        estimate = source.estimated_peak_bytes
        if estimate > max_local_temp_bytes:
            raise ConversionError(f"source episode {source.source_id} peak {estimate} exceeds --max-local-temp-bytes")
        if current and (current_peak + estimate > per_unit_limit or len(current) >= 1000):
            groups.append(current)
            current, current_peak = [], 0
        current.append(source)
        current_peak += estimate
    if current:
        groups.append(current)
    task_index = {task: index for index, task in enumerate(dict.fromkeys(ep.instruction for ep in partition.plan.episodes))}
    units: list[ParallelWorkUnit] = []
    episode_cursor = 0
    frame_cursor = 0
    for index, group in enumerate(groups):
        frames = sum(source.length for source in group)
        unit = ParallelWorkUnit(
            index=index,
            key=f"{partition.name}/unit-{index:06d}",
            dataset_uid=partition.plan.dataset_uid,
            target_path=str(work_root / "work" / run_id / partition.name / f"unit-{index:06d}"),
            episode_start=episode_cursor,
            episode_end=episode_cursor + len(group),
            frame_start=frame_cursor,
            frame_end=frame_cursor + frames,
            task_indices=tuple(task_index[source.instruction] for source in group),
            weight=frames,
            estimated_memory_bytes=max(source.data_bytes for source in group),
            estimated_temp_bytes=sum(source.estimated_peak_bytes for source in group),
            fingerprint=fingerprint,
            payload=UnitPayload(partition, tuple(group)),
        )
        if unit.estimated_temp_bytes > max_local_temp_bytes:
            raise ConversionError(f"unit {unit.key} exceeds --max-local-temp-bytes; split the unit")
        units.append(unit)
        episode_cursor += len(group)
        frame_cursor += frames
    return tuple(units)


class LocalReservation:
    """Global local byte/unit reservation, held until OSS validation frees bulk."""

    def __init__(
        self,
        root: Path,
        guard: DiskGuard,
        max_bytes: int,
        max_units: int,
        *,
        baseline_bytes: int | None = None,
    ) -> None:
        self.root, self.guard, self.max_bytes, self.max_units = root, guard, max_bytes, max_units
        self.baseline_bytes = directory_size(root) if baseline_bytes is None else baseline_bytes
        if self.baseline_bytes < 0:
            raise ValueError("baseline_bytes cannot be negative")
        self._condition = threading.Condition()
        self._reserved: dict[int, int] = {}
        self._failure: BaseException | None = None

    def try_reserve(self, unit: ParallelWorkUnit) -> bool:
        """Reserve a unit if the current local quota can accommodate it.

        This operation is deliberately non-blocking.  The coordinator must be
        able to return to its completion wait when the next reservation does
        not fit; otherwise an upload that would free the reservation can never
        be observed by the coordinator.
        """

        if unit.estimated_temp_bytes > self.max_bytes:
            raise ConversionError(f"unit {unit.key} exceeds local quota")
        with self._condition:
            if self._failure is not None:
                raise ConversionError("previous upload failure stopped the pipeline") from self._failure
            actual = directory_size(self.root)
            reserved = sum(self._reserved.values())
            # ``actual`` includes cache/logs/checkpoints and any partial local
            # units already being built.  Every reservation can still grow to
            # its complete estimate, so add it rather than taking a maximum.
            task_usage = max(0, actual - self.baseline_bytes)
            projected = task_usage + reserved + unit.estimated_temp_bytes
            if len(self._reserved) >= self.max_units or projected > self.max_bytes:
                return False
            self.guard.check(
                "before dispatch",
                required_additional_bytes=max(0, projected - task_usage),
                inflight_units=len(self._reserved) + 1,
            )
            self._reserved[unit.index] = unit.estimated_temp_bytes
            return True

    def reserve(self, unit: ParallelWorkUnit) -> None:
        """Block until a reservation is available.

        Retained for callers that explicitly want blocking semantics.  The
        partition coordinator uses :meth:`try_reserve` so it can continue
        servicing completed uploads while waiting for space.
        """

        while not self.try_reserve(unit):
            with self._condition:
                self._condition.wait(timeout=0.5)

    def release(self, unit: ParallelWorkUnit) -> None:
        with self._condition:
            self._reserved.pop(unit.index, None)
            self._condition.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self._condition:
            self._failure = exc
            self._condition.notify_all()


def _prepare_local_units(
    units: tuple[ParallelWorkUnit, ...],
) -> tuple[tuple[ParallelWorkUnit, ...], tuple[ParallelWorkUnit, ...], int]:
    """Separate verified local units from units that must be rebuilt.

    A process can be interrupted after a unit has been fully validated but
    before its direct commit completes.  Such a unit is already occupying
    local space and must be uploaded first; treating it as a fresh build makes
    the quota calculation wait for space that the upload itself would free.
    Unverified leftovers are safe to discard because they are not resume
    points and the builder recreates them from the frozen catalog.
    """

    verified: list[ParallelWorkUnit] = []
    rebuild: list[ParallelWorkUnit] = []
    verified_bytes = 0
    for unit in units:
        target = Path(unit.target_path)
        marker = verified_marker_path(unit)
        if target.is_dir() and marker.is_file():
            try:
                # Do not re-hash a potentially multi-dozen-GB unit on the
                # coordinator thread.  The uploader performs the full
                # inventory validation before copying; this fast check only
                # decides whether the local unit can be queued as a resume
                # upload.
                read_verified_unit_marker(unit)
            except (ConversionError, OSError, ValueError):
                shutil.rmtree(target, ignore_errors=True)
                marker.unlink(missing_ok=True)
                rebuild.append(unit)
            else:
                verified.append(unit)
                marker_payload = read_verified_unit_marker(unit)
                verified_bytes += sum(
                    int(record.get("size", 0))
                    for record in marker_payload.get("inventory", [])
                    if isinstance(record, dict)
                )
            continue
        marker.unlink(missing_ok=True)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        rebuild.append(unit)
    return tuple(verified), tuple(rebuild), verified_bytes


def _run_partition(
    partition: RobogenePartition,
    *,
    final_root: Path,
    local_root: Path,
    run_id: str,
    fingerprint: str,
    args: argparse.Namespace,
    guard: DiskGuard,
) -> dict[str, Any]:
    partition_root = final_root / partition.name
    partition_root.mkdir(parents=True, exist_ok=True)
    units = _units_for_partition(partition, work_root=local_root, run_id=run_id, max_local_temp_bytes=args.max_local_temp_bytes, max_inflight_units=args.max_inflight_units, fingerprint=fingerprint)
    resume_root = local_root / "resume"
    prepared = prepare_direct_commits(units, partition_name=partition.name, partition_root=partition_root, resume_root=resume_root)
    verified_local, pending, verified_local_bytes = _prepare_local_units(prepared.uncommitted)
    baseline_usage = max(0, directory_size(local_root) - verified_local_bytes)
    ledger = LocalReservation(
        local_root,
        guard,
        args.max_local_temp_bytes,
        args.max_inflight_units,
        baseline_bytes=baseline_usage,
    )
    builds: dict[Future[Any], ParallelWorkUnit] = {}
    uploads: dict[Future[Any], ParallelWorkUnit] = {}
    cursor = 0
    started = time.monotonic()

    def submit_build(pool: ThreadPoolExecutor, unit: ParallelWorkUnit) -> bool:
        if not ledger.try_reserve(unit):
            return False
        builds[pool.submit(_build_unit, unit)] = unit
        return True

    def submit_upload(pool: ThreadPoolExecutor, unit: ParallelWorkUnit, *, trust_verified_marker: bool = False) -> None:
        uploads[pool.submit(commit_verified_unit, unit, partition_name=partition.name, partition_root=partition_root, resume_root=resume_root, trust_verified_marker=trust_verified_marker)] = unit

    verified_cursor = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="robogene-convert") as builders, ThreadPoolExecutor(max_workers=args.upload_workers, thread_name_prefix="robogene-upload") as uploader:
            while verified_cursor < len(verified_local) or cursor < len(pending) or builds or uploads:
                while verified_cursor < len(verified_local) and len(builds) + len(uploads) < args.max_inflight_units:
                    # Resume uploads must still validate their local bulk in
                    # the uploader thread.  Keeping that work off the
                    # coordinator is essential when the unit is tens of GB.
                    submit_upload(uploader, verified_local[verified_cursor])
                    verified_cursor += 1
                while cursor < len(pending) and len(builds) < args.workers and len(builds) + len(uploads) < args.max_inflight_units:
                    if not submit_build(builders, pending[cursor]):
                        break
                    cursor += 1
                active = set(builds) | set(uploads)
                if not active:
                    if verified_cursor < len(verified_local) or cursor < len(pending):
                        raise ConversionError(
                            "local quota cannot dispatch the next RoboGene unit; "
                            "reduce unit size or free local staging space"
                        )
                    break
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    if future in builds:
                        unit = builds.pop(future)
                        future.result()
                        submit_upload(uploader, unit)
                    else:
                        unit = uploads.pop(future)
                        future.result()
                        ledger.release(unit)
    except BaseException as exc:
        ledger.fail(exc)
        raise
    finalize_direct_partition(
        partition.plan,
        units,
        partition_root,
        resume_root=resume_root,
        reader_format="robogene_lerobot_v21",
        parallel_evidence={
            "workers": args.workers,
            "upload_workers": args.upload_workers,
            "encoder_threads_per_worker": args.encoder_threads_per_worker,
            "local_max_bytes": args.max_local_temp_bytes,
            "pipeline": "local-unit-copy-rebase-upload",
        },
    )
    return {
        "partition": partition.name,
        "split": partition.split,
        "schema_fingerprint": partition.schema_fingerprint,
        "robot_type": partition.plan.robot_type,
        "episodes": len(partition.episodes),
        "frames": partition.plan.num_frames,
        "units": len(units),
        "elapsed_seconds": time.monotonic() - started,
        "empty_tasks": list(partition.empty_tasks),
        "sample_evidence": list(partition.sample_evidence),
    }


def _catalog_summary(catalog: RobogeneCatalog) -> dict[str, Any]:
    return {
        "partitions": [
            {"name": partition.name, "episodes": len(partition.episodes), "frames": partition.plan.num_frames, "robot_type": partition.plan.robot_type, "features": partition.plan.feature_schema(), "empty_tasks": list(partition.empty_tasks), "sample_evidence": list(partition.sample_evidence)}
            for partition in catalog.partitions
        ],
        "field_mapping": list(catalog.mapping_table),
    }


def _resume_state_path(local_root: Path) -> Path:
    return local_root / "resume" / "robogene_preflight.json"


def _conversion_options(args: argparse.Namespace, raw_root: Path) -> dict[str, Any]:
    return {
        "raw_root": str(raw_root),
        "output_uid": "robogene",
        "task_names": sorted(args.task or []),
        "limit_tasks": args.limit_tasks,
        "limit_episodes": args.limit_episodes,
        "limit_shards": args.limit_shards,
        "max_local_temp_bytes": args.max_local_temp_bytes,
        "encoder_threads_per_worker": args.encoder_threads_per_worker,
        "video_encoding": {"source_codec": "h264", "target_codec": "h264", "target_pix_fmt": "yuv420p", "mode": "copy"},
        "partition_rules": ["top-level split", "Parquet schema fingerprint"],
    }


def _load_catalog_for_run(
    args: argparse.Namespace, raw_root: Path, *, state_path: Path
) -> tuple[RobogeneCatalog, str, str]:
    """Use a frozen, re-stat-validated catalog for resume without re-listing source."""

    options = _conversion_options(args, raw_root)
    if args.resume and state_path.is_file():
        from convert_core.checkpoint import read_json_object

        state = read_json_object(state_path, "RoboGene preflight resume state")
        if state.get("options") != options:
            raise ConversionError("RoboGene resume options changed; use the original selection and encoder settings")
        catalog_payload = state.get("catalog")
        if not isinstance(catalog_payload, dict):
            raise ConversionError("RoboGene preflight resume catalog is missing")
        catalog = catalog_from_payload(catalog_payload)
        if catalog.fingerprint_payload.get("source_root") != str(raw_root):
            raise ConversionError("RoboGene resume source root changed")
        validate_catalog_source_files(catalog, raw_root)
        fingerprint = canonical_fingerprint({"catalog": catalog.fingerprint_payload, "options": options})
        if state.get("fingerprint") != fingerprint:
            raise ConversionError("RoboGene resume fingerprint changed")
        run_id = state.get("run_id")
        if not isinstance(run_id, str):
            raise ConversionError("RoboGene resume state has an invalid run id")
        _validate_run_id(run_id)
        if args.run_id is not None and args.run_id != run_id:
            raise ConversionError("--run-id does not match the unfinished RoboGene run")
        return catalog, fingerprint, run_id
    catalog = inspect_robogene(
        raw_root,
        limit_tasks=args.limit_tasks,
        limit_episodes=args.limit_episodes,
        limit_shards=args.limit_shards,
        task_names=set(args.task) if args.task else None,
    )
    fingerprint = canonical_fingerprint({"catalog": catalog.fingerprint_payload, "options": options})
    run_id = _validate_run_id(args.run_id or _run_id())
    atomic_write_json(
        state_path,
        {"schema_version": 1, "fingerprint": fingerprint, "run_id": run_id, "options": options, "catalog": catalog_to_payload(catalog)},
    )
    return catalog, fingerprint, run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-work-root", type=Path, default=DEFAULT_LOCAL_WORK_ROOT)
    parser.add_argument("--max-local-temp-bytes", type=_positive_int, default=DEFAULT_MAX_LOCAL_TEMP_BYTES)
    parser.add_argument("--min-local-free-bytes", type=_positive_int, default=DEFAULT_MIN_LOCAL_FREE_BYTES)
    parser.add_argument("--max-inflight-units", type=_positive_int, default=4)
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument("--encoder-threads-per-worker", type=_positive_int, default=8)
    parser.add_argument("--upload-workers", type=_positive_int, default=1)
    parser.add_argument("--skip-encoder-warmup", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inspect-only", "--dry-run", dest="inspect_only", action="store_true")
    parser.add_argument("--limit-tasks", type=_positive_int)
    parser.add_argument("--limit-episodes", type=_positive_int)
    parser.add_argument("--limit-shards", type=_positive_int)
    parser.add_argument("--task", action="append", help="exact RoboGene task directory name; repeatable")
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_root, output_root = validate_source_and_output_roots(args.raw_root, args.output_root)
    if raw_root.resolve(strict=False) != DEFAULT_RAW_ROOT.resolve(strict=False):
        raise ConversionError(f"--raw-root must be the approved RoboGene source root {DEFAULT_RAW_ROOT}")
    if output_root.resolve(strict=False) != DEFAULT_OUTPUT_ROOT.resolve(strict=False):
        raise ConversionError(f"--output-root must be the approved staging root {DEFAULT_OUTPUT_ROOT}")
    if args.local_work_root.resolve(strict=False) != DEFAULT_LOCAL_WORK_ROOT.resolve(strict=False):
        raise ConversionError(f"--local-work-root must be the approved local staging root {DEFAULT_LOCAL_WORK_ROOT}")
    if args.upload_workers > args.max_inflight_units:
        raise ConversionError("--upload-workers cannot exceed --max-inflight-units")
    final_root = output_root / "robogene"
    if not args.inspect_only and (final_root / "_SUCCESS").is_file():
        raise ConversionError(f"valid RoboGene output already exists: {final_root}")
    if args.inspect_only:
        catalog = inspect_robogene(
            raw_root,
            limit_tasks=args.limit_tasks,
            limit_episodes=args.limit_episodes,
            limit_shards=args.limit_shards,
            task_names=set(args.task) if args.task else None,
        )
        if not catalog.partitions:
            raise ConversionError("RoboGene selection contains no non-empty task episodes")
        summary = _catalog_summary(catalog)
        for partition in summary["partitions"]:
            print(f"preflight partition={partition['name']} episodes={partition['episodes']} frames={partition['frames']} robot_type={partition['robot_type']} empty_tasks={len(partition['empty_tasks'])}")
        print(f"preflight field_mappings={len(summary['field_mapping'])}")
        return 0
    args.local_work_root.mkdir(parents=True, exist_ok=True)
    lock = args.local_work_root / "resume" / "robogene.lock"
    with exclusive_staging_lock(lock):
        if (final_root / "_INCOMPLETE").exists() and not args.resume:
            raise ConversionError("incomplete RoboGene output exists; rerun with --resume")
        if (final_root / "_INCOMPLETE").exists() and args.resume and not _resume_state_path(args.local_work_root).is_file():
            raise ConversionError("unfinished RoboGene output has no local preflight state; refusing unsafe resume")
        catalog, fingerprint, run_id = _load_catalog_for_run(
            args, raw_root, state_path=_resume_state_path(args.local_work_root)
        )
        if not catalog.partitions:
            raise ConversionError("RoboGene selection contains no non-empty task episodes")
        for path in (args.local_work_root / "work" / run_id, args.local_work_root / "resume", args.local_work_root / "logs", args.local_work_root / "cache" / run_id):
            path.mkdir(parents=True, exist_ok=True)
        cache_root = args.local_work_root / "cache" / run_id
        runtime_environment = {
            "TMPDIR": args.local_work_root / "work" / run_id / "tmp", "TMP": args.local_work_root / "work" / run_id / "tmp", "TEMP": args.local_work_root / "work" / run_id / "tmp",
            "XDG_CACHE_HOME": cache_root / "xdg", "HF_HOME": cache_root / "huggingface", "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
            "TORCH_HOME": cache_root / "torch", "MPLCONFIGDIR": cache_root / "matplotlib", "PYTHONPYCACHEPREFIX": cache_root / "pycache", "CUDA_CACHE_PATH": cache_root / "cuda",
            "VLA_DATASETS_CACHE_ROOT": cache_root / "datasets", "TORCH_EXTENSIONS_DIR": cache_root / "torch-extensions", "NUMBA_CACHE_DIR": cache_root / "numba",
            "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers", "PIP_CACHE_DIR": cache_root / "pip",
        }
        for key, path in runtime_environment.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        guard = DiskGuard(args.local_work_root, usage_roots=(args.local_work_root,), min_free_bytes=args.min_local_free_bytes, max_local_bytes=args.max_local_temp_bytes, interval_seconds=1.0)
        if not args.skip_encoder_warmup:
            _encoder_warmup_once(catalog, local_root=args.local_work_root, fingerprint=fingerprint, run_id=run_id)
        create_incomplete_output(final_root, fingerprint=fingerprint, run_id=run_id)
        reports = [_run_partition(partition, final_root=final_root, local_root=args.local_work_root, run_id=run_id, fingerprint=fingerprint, args=args, guard=guard) for partition in catalog.partitions]
        manifest = {"format": "lerobot_v3_0", "source_format": "robogene_lerobot_v21", "fingerprint": fingerprint, "partitions": reports, "field_mapping": list(catalog.mapping_table), "local_runtime_root": str(args.local_work_root), "success_marker_required": True}
        atomic_write_json(final_root / "conversion_manifest.json", manifest)
        publish_success(final_root, fingerprint=fingerprint, evidence={"partitions": reports, "conversion_manifest": "conversion_manifest.json"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
