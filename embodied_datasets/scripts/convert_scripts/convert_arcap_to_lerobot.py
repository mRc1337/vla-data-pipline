"""Reliably stage the official ARCap release as five LeRobot v3 partitions.

The coordinator freezes all indices before dispatch, batches complete ARCap
phase groups into bounded work units, and uses isolated persistent workers.
Verified Parquet chunks are moved directly into their deterministic final
paths; only compact metadata is aggregated by the coordinator.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Sequence
import uuid

# Set this before importing repository modules so the converter itself never
# creates bytecode objects on the full /home filesystem.
sys.dont_write_bytecode = True

from convert_core.checkpoint import (
    RESUME_SCHEMA_VERSION,
    atomic_write_json,
    build_resume_payload,
    canonical_fingerprint,
    read_json_object,
)
from convert_core.direct_commit import (
    commit_verified_unit,
    finalize_direct_partition,
    prepare_direct_commits,
)
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import convert_dataset, plan_summary, validate_written_dataset
from convert_core.equivalence import verify_lerobot_equivalence
from convert_core.parallel import (
    ParallelWorkUnit,
    PlanUnitSlice,
    isolated_unit_plan,
    prepare_work_units,
    run_parallel_work_units,
    validate_inflight_budget,
    write_verified_unit_marker,
)
from convert_core.performance import ProcessTreeSampler
from convert_core.staging import (
    StagingLayout,
    create_incomplete_output,
    exclusive_staging_lock,
    make_staging_layout,
    publish_success,
    sha256_file,
    validate_contained_path,
    validate_no_runtime_paths_outside_root,
    validate_source_and_output_roots,
)
from convert_core.storage import (
    StagingQuotaGuard,
    format_staging_quota_check,
    scoped_size,
)
from readers.arcap_hdf5_reader import (
    PARTITION_SPECS,
    ARCapPartitionInfo,
    inspect_partition,
    iter_frames,
)


GIB = 1024**3
MIB = 1024**2
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0"
)
DEFAULT_RAW_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw/arcap")
DEFAULT_MAX_STAGING_BYTES = 80 * GIB
DEFAULT_MAX_INFLIGHT_BYTES = 16 * GIB
DEFAULT_WORKER_MEMORY_LIMIT_BYTES = 8 * GIB
DEFAULT_MAX_FRAMES_PER_UNIT = 8_000
FULL_EXPECTED_OUTPUT_BYTES = 61_390_182_738
FULL_TOTAL_FRAMES = 231_923
OFFICIAL_PARTITIONS = tuple(spec.name for spec in PARTITION_SPECS)


@dataclass(frozen=True)
class _WorkerPayload:
    plan: Any
    local_index: int
    local_key: str
    eta_interval_seconds: float
    encoder_threads: int
    conversion_options: dict[str, Any]
    logs_dir: str
    run_id: str


_WORKER_SLOT = -1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    """Append one bounded event and flush it for ``tail -f`` consumers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "unix_time": time.time(),
        **event,
    }
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def _coordinator_log_path(layout: StagingLayout) -> Path:
    return layout.logs / f"{layout.run_id}.coordinator.jsonl"


def _worker_log_path(payload: _WorkerPayload) -> Path:
    return Path(payload.logs_dir) / f"{payload.run_id}.worker-{_WORKER_SLOT:02d}.jsonl"


def _process_io() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip())
    except (OSError, ValueError):
        return {}
    return values


def _usage_bytes_and_objects(paths: Sequence[Path]) -> tuple[int, int]:
    """Count regular files without following symlinks or double-counting roots."""

    roots = sorted(
        {path.resolve(strict=False) for path in paths},
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    selected: list[Path] = []
    for root in roots:
        if not any(root.is_relative_to(parent) for parent in selected):
            selected.append(root)
    total = 0
    objects = 0
    pending = [path for path in selected if path.exists()]
    while pending:
        root = pending.pop()
        if root.is_file():
            total += root.stat().st_size
            objects += 1
            continue
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    objects += 1
            except OSError:
                continue
    return total, objects


def _bulk_checksum(root: Path) -> dict[str, Any]:
    """Digest deterministic frame/metadata files, excluding run-specific markers."""

    records: list[tuple[str, int, str]] = []
    for relative_root in (Path("data"), Path("meta"), Path("videos")):
        base = root / relative_root
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            records.append(
                (path.relative_to(root).as_posix(), path.stat().st_size, sha256_file(path))
            )
    digest = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"sha256": digest, "files": len(records), "bytes": sum(row[1] for row in records)}


def _phase_group_slices(plan: Any, max_frames: int) -> tuple[PlanUnitSlice, ...]:
    """Greedily batch whole, contiguous phase groups up to ``max_frames``."""

    if max_frames <= 0 or not plan.episodes:
        raise ConversionError("work-unit frame limit and plan must be non-empty")
    tasks = list(dict.fromkeys(episode.instruction for episode in plan.episodes))
    task_index = {task: index for index, task in enumerate(tasks)}
    groups: list[tuple[int, int, int]] = []
    start = 0
    while start < len(plan.episodes):
        group = int(plan.episodes[start].extra["phase_group_index"])
        end = start + 1
        while (
            end < len(plan.episodes)
            and int(plan.episodes[end].extra["phase_group_index"]) == group
        ):
            end += 1
        expected_size = int(plan.episodes[start].extra["phase_group_size"])
        if end - start != expected_size:
            raise ConversionError(f"phase group {group} is incomplete in {plan.dataset_uid}")
        groups.append((start, end, sum(ep.num_frames for ep in plan.episodes[start:end])))
        start = end

    slices: list[PlanUnitSlice] = []
    group_cursor = 0
    frame_cursor = 0
    while group_cursor < len(groups):
        episode_start = groups[group_cursor][0]
        episode_end = episode_start
        frames = 0
        first_group = group_cursor
        while group_cursor < len(groups):
            candidate = groups[group_cursor][2]
            if frames and frames + candidate > max_frames:
                break
            episode_end = groups[group_cursor][1]
            frames += candidate
            group_cursor += 1
        frame_end = frame_cursor + frames
        slices.append(
            PlanUnitSlice(
                index=len(slices),
                key=(
                    f"{plan.output_path.name}/phase-groups-"
                    f"{first_group:05d}-{group_cursor - 1:05d}"
                ),
                episode_start=episode_start,
                episode_end=episode_end,
                frame_start=frame_cursor,
                frame_end=frame_end,
                task_indices=tuple(
                    task_index[episode.instruction]
                    for episode in plan.episodes[episode_start:episode_end]
                ),
            )
        )
        frame_cursor = frame_end
    return tuple(slices)


def _configure_runtime_environment(layout: StagingLayout, *, create: bool) -> dict[str, str]:
    """Use the ARCap-mandated ``runtime_cache`` directory, never $HOME/tmp."""

    cache = layout.work / "runtime_cache"
    values = {
        "TMPDIR": layout.temp,
        "TMP": layout.temp,
        "TEMP": layout.temp,
        "XDG_CACHE_HOME": cache / "xdg",
        "HF_HOME": cache / "huggingface",
        "HF_DATASETS_CACHE": cache / "huggingface" / "datasets",
        "TORCH_HOME": cache / "torch",
        "MPLCONFIGDIR": cache / "matplotlib",
        "VLA_DATASETS_CACHE_ROOT": cache / "datasets",
        "CUDA_CACHE_PATH": cache / "cuda",
        "TORCH_EXTENSIONS_DIR": cache / "torch-extensions",
        "NUMBA_CACHE_DIR": cache / "numba",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
    }
    if create:
        for path in set(values.values()):
            path.mkdir(parents=True, exist_ok=True)
    rendered = {key: str(path) for key, path in values.items()}
    os.environ.update(rendered)
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
        }
    )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    return rendered


def _select_run_id(output_root: Path, dataset_uid: str, resume_dir: Path | None) -> str:
    root = resume_dir or output_root / ".conversion_resume" / dataset_uid
    state_path = root / "collection.json"
    if state_path.is_file():
        state = read_json_object(state_path, "ARCap collection state")
        runtime = state.get("runtime")
        if isinstance(runtime, dict) and isinstance(runtime.get("run_id"), str):
            return str(runtime["run_id"])
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]


def _layout(args: argparse.Namespace) -> StagingLayout:
    approved = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    if args.output_root.resolve(strict=False) != approved:
        raise ConversionError(
            f"--output-root must be the approved staging root {approved}"
        )
    run_id = _select_run_id(args.output_root, args.output_dataset_uid, args.resume_dir)
    return make_staging_layout(
        output_root=args.output_root,
        dataset_uid=args.output_dataset_uid,
        run_id=run_id,
        work_dir=args.work_dir,
        resume_dir=args.resume_dir,
        logs_dir=args.logs_dir,
        temp_dir=args.temp_dir,
    )


def _selected_specs(names: Sequence[str] | None) -> tuple[Any, ...]:
    requested = set(names or OFFICIAL_PARTITIONS)
    return tuple(spec for spec in PARTITION_SPECS if spec.name in requested)


def _inspect_infos(args: argparse.Namespace, layout: StagingLayout) -> list[ARCapPartitionInfo]:
    infos = [
        inspect_partition(
            args.raw_root / spec.filename,
            raw_dataset_root=args.raw_root,
            collection_output=layout.final,
            max_phase_groups=args.max_phase_groups,
            verify_sha256=args.verify_source_sha256,
            full_lowdim_scan=args.full_lowdim_scan,
            full_pointcloud_scan=args.full_pointcloud_scan,
        )
        for spec in _selected_specs(args.partition)
    ]
    if not infos:
        raise ConversionError("partition selection is empty")
    if args.max_work_units is not None:
        remaining = args.max_work_units
        limited: list[ARCapPartitionInfo] = []
        for info in infos:
            if remaining <= 0:
                break
            slices = _phase_group_slices(info.plan, args.max_frames_per_unit)
            take = min(remaining, len(slices))
            end = slices[take - 1].episode_end
            plan = replace(info.plan, episodes=info.plan.episodes[:end])
            limited.append(replace(info, plan=plan))
            remaining -= take
        infos = limited
    return infos


def _conversion_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arcap_converter_schema_version": 1,
        "max_frames_per_unit": args.max_frames_per_unit,
        "acceleration_mode": args.acceleration_mode,
        "workers": args.workers,
        "max_inflight_units": args.max_inflight_units,
        "encoder_threads_per_worker": args.encoder_threads_per_worker,
        "worker_memory_limit_bytes": args.worker_memory_limit_bytes,
        "pointcloud_transform": "identity",
        "timestamp_expression": "frame_index / 10",
        "video_features": 0,
    }


def _build_units(
    infos: Sequence[ARCapPartitionInfo],
    layout: StagingLayout,
    args: argparse.Namespace,
) -> tuple[tuple[ParallelWorkUnit, ...], dict[str, tuple[ParallelWorkUnit, ...]]]:
    options = _conversion_options(args)
    global_units: list[ParallelWorkUnit] = []
    local_by_partition: dict[str, tuple[ParallelWorkUnit, ...]] = {}
    for info in infos:
        local_units: list[ParallelWorkUnit] = []
        for item in _phase_group_slices(info.plan, args.max_frames_per_unit):
            target = layout.work / "units" / info.spec.name / f"unit-{item.index:06d}"
            unit_plan = isolated_unit_plan(
                info.plan,
                item,
                dataset_uid=f"{args.output_dataset_uid}_{info.spec.name}__unit_{item.index:06d}",
                target_path=target,
            )
            unit_options = {
                **options,
                "partition": info.spec.name,
                "local_unit_index": item.index,
                "local_unit_key": item.key,
            }
            fingerprint = canonical_fingerprint(
                build_resume_payload(
                    unit_plan,
                    reader_format="arcap_hdf5",
                    conversion_options=unit_options,
                )
            )
            payload = _WorkerPayload(
                plan=unit_plan,
                local_index=item.index,
                local_key=item.key,
                eta_interval_seconds=args.eta_interval_seconds,
                encoder_threads=args.encoder_threads_per_worker,
                conversion_options=unit_options,
                logs_dir=str(layout.logs),
                run_id=layout.run_id,
            )
            global_unit = ParallelWorkUnit(
                index=len(global_units),
                key=item.key,
                dataset_uid=unit_plan.dataset_uid,
                target_path=str(target),
                episode_start=item.episode_start,
                episode_end=item.episode_end,
                frame_start=item.frame_start,
                frame_end=item.frame_end,
                task_indices=item.task_indices,
                weight=item.frame_end - item.frame_start,
                estimated_memory_bytes=args.worker_memory_limit_bytes,
                estimated_temp_bytes=(item.frame_end - item.frame_start) * 500_000,
                fingerprint=fingerprint,
                payload=payload,
            )
            local_units.append(_local_unit(global_unit))
            global_units.append(global_unit)
        local_by_partition[info.spec.name] = tuple(local_units)
    return tuple(global_units), local_by_partition


def _local_unit(unit: ParallelWorkUnit) -> ParallelWorkUnit:
    payload = unit.payload
    if not isinstance(payload, _WorkerPayload):
        raise ConversionError(f"invalid ARCap worker payload for {unit.key}")
    return replace(unit, index=payload.local_index, key=payload.local_key)


def _initialize_worker(_slot: int, memory_limit: int) -> None:
    global _WORKER_SLOT
    _WORKER_SLOT = _slot
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    def enforce_rss() -> None:
        while True:
            try:
                status = Path("/proc/self/status").read_text(encoding="utf-8")
                rss_kib = next(
                    int(line.split()[1])
                    for line in status.splitlines()
                    if line.startswith("VmRSS:")
                )
                if rss_kib * 1024 > memory_limit:
                    print(
                        f"worker RSS limit exceeded: {rss_kib * 1024} > {memory_limit}",
                        file=sys.stderr,
                        flush=True,
                    )
                    os._exit(86)
            except (OSError, StopIteration, ValueError):
                os._exit(87)
            time.sleep(0.25)

    threading.Thread(target=enforce_rss, name="arcap-rss-guard", daemon=True).start()


def _convert_unit(unit: ParallelWorkUnit) -> dict[str, Any]:
    payload = unit.payload
    if not isinstance(payload, _WorkerPayload):
        raise ConversionError(f"invalid ARCap worker payload for {unit.key}")
    local = _local_unit(unit)
    target = Path(unit.target_path)
    log_path = _worker_log_path(payload)
    started = time.monotonic()
    cpu_started = time.process_time()
    io_started = _process_io()
    identity = {
        "worker_slot": _WORKER_SLOT,
        "pid": os.getpid(),
        "unit": unit.key,
        "unit_index": unit.index,
        "partition": payload.plan.output_path.parent.name,
        "local_unit_index": payload.local_index,
        "episodes": len(payload.plan.episodes),
        "frames": payload.plan.num_frames,
        "tasks": list(
            dict.fromkeys(episode.instruction for episode in payload.plan.episodes)
        ),
        "first_source_episode": payload.plan.episodes[0].source_relative_path,
        "last_source_episode": payload.plan.episodes[-1].source_relative_path,
        "first_phase_group": int(
            payload.plan.episodes[0].extra["phase_group_index"]
        ),
        "last_phase_group": int(
            payload.plan.episodes[-1].extra["phase_group_index"]
        ),
    }
    _append_jsonl(log_path, {"event": "unit_started", **identity})
    try:
        for stale in target.parent.glob(f".{target.name}.incomplete-*"):
            if stale.is_dir():
                _remove_tree_with_retries(stale)
            else:
                stale.unlink()
        with _bounded_dataset_cache(target.parent / f".{target.name}.datasets-cache"):
            convert_dataset(
                payload.plan,
                lambda episode: iter_frames(payload.plan, episode),
                reader_format="arcap_hdf5",
                resume=False,
                eta_interval_seconds=payload.eta_interval_seconds,
                encoder_threads=payload.encoder_threads,
                conversion_options=payload.conversion_options,
                batch_metadata_writes=True,
            )
        write_verified_unit_marker(local)
    except BaseException as exc:
        _append_jsonl(
            log_path,
            {
                "event": "unit_failed",
                **identity,
                "wall_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ossfs_errors": int(isinstance(exc, OSError)),
                "ossfs_retries": 0,
            },
        )
        raise
    io_finished = _process_io()
    wall = time.monotonic() - started
    cpu = time.process_time() - cpu_started
    result = {
        **identity,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "frames_per_second": payload.plan.num_frames / wall if wall else 0.0,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "unit_output_bytes": scoped_size((target,)),
        "read_bytes": max(0, io_finished.get("read_bytes", 0) - io_started.get("read_bytes", 0)),
        "write_bytes": max(0, io_finished.get("write_bytes", 0) - io_started.get("write_bytes", 0)),
        "read_chars": max(0, io_finished.get("rchar", 0) - io_started.get("rchar", 0)),
        "write_chars": max(0, io_finished.get("wchar", 0) - io_started.get("wchar", 0)),
        "failures": 0,
        "retries": 0,
        "encoder_errors": 0,
        "ossfs_errors": 0,
        "ossfs_retries": 0,
    }
    _append_jsonl(log_path, {"event": "unit_succeeded", **result})
    return result


def _validate_unit(unit: ParallelWorkUnit) -> None:
    payload = unit.payload
    if not isinstance(payload, _WorkerPayload):
        raise ConversionError(f"invalid ARCap worker payload for {unit.key}")
    target = Path(unit.target_path)
    with _bounded_dataset_cache(target.parent / f".{target.name}.validation-cache"):
        validate_written_dataset(payload.plan, target)


@contextmanager
def _bounded_dataset_cache(path: Path):
    """Isolate and remove the materialized Arrow cache after one validation."""

    keys = ("VLA_DATASETS_CACHE_ROOT", "HF_DATASETS_CACHE")
    previous = {key: os.environ.get(key) for key in keys}
    path.mkdir(parents=True, exist_ok=True)
    os.environ.update({key: str(path) for key in keys})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if path.exists():
            _remove_tree_with_retries(path)


def _collection_payload(
    infos: Sequence[ARCapPartitionInfo], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "kind": "arcap_collection",
        "output_dataset_uid": args.output_dataset_uid,
        "partitions": [
            build_resume_payload(
                info.plan,
                reader_format="arcap_hdf5",
                conversion_options=_conversion_options(args),
            )
            for info in infos
        ],
        "options": _conversion_options(args),
    }


def _prepare_collection_state(
    layout: StagingLayout, payload: dict[str, Any], *, require_existing: bool
) -> str:
    path = layout.resume / "collection.json"
    fingerprint = canonical_fingerprint(payload)
    if path.is_file():
        state = read_json_object(path, "ARCap collection state")
        if state.get("fingerprint") != fingerprint:
            raise ConversionError(
                "ARCap resume fingerprint changed; source, selection, schema, worker, "
                "memory, or storage-independent conversion arguments differ"
            )
        if state.get("runtime") != layout.as_dict():
            raise ConversionError("ARCap resume runtime paths or run_id changed")
        return fingerprint
    if require_existing:
        raise ConversionError(f"--resume requires existing state: {path}")
    atomic_write_json(
        path,
        {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "configuration": payload,
            "runtime": layout.as_dict(),
        },
    )
    return fingerprint


def _estimate(infos: Sequence[ARCapPartitionInfo], units: Sequence[ParallelWorkUnit]) -> dict[str, Any]:
    frames = sum(info.plan.num_frames for info in infos)
    expected = math.ceil(FULL_EXPECTED_OUTPUT_BYTES * frames / FULL_TOTAL_FRAMES)
    upper = math.ceil(expected * 1.15)
    inflight = sum(
        sorted((unit.estimated_temp_bytes for unit in units), reverse=True)[:4]
    )
    return {
        "method": (
            "real ARCap LeRobot benchmark scaled by selected frames; 15% conservative "
            "Parquet margin; statvfs/df excluded from quota evidence"
        ),
        "expected_output_bytes": expected,
        "conservative_output_bytes": upper,
        "metadata_checkpoint_upper_bytes": 512 * MIB,
        "four_unit_inflight_upper_bytes": inflight,
        "expected_full_output_bytes": FULL_EXPECTED_OUTPUT_BYTES,
        "oss_object_quota_verified": False,
    }


def _manifest(
    infos: Sequence[ARCapPartitionInfo],
    local: dict[str, tuple[ParallelWorkUnit, ...]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_uid": args.output_dataset_uid,
        "format": "lerobot_v3_0_collection",
        "source_dataset": "Ericcsr/ARCap",
        "partitions": [
            {
                "name": info.spec.name,
                "relative_path": info.spec.name,
                "episodes": len(info.plan.episodes),
                "frames": info.plan.num_frames,
                "source_relative_path": info.source_relative_path,
                "source_schema_fingerprint": info.schema_fingerprint,
                "source_sha256": info.spec.sha256,
                "phase_group_size": info.spec.phase_group_size,
                "instruction": info.spec.instruction,
                "work_units": len(local[info.spec.name]),
            }
            for info in infos
        ],
        "partition_rule": "fixed official HDF5 schema and robot layout",
        "publication": "marker-gated in-place direct commit; no collection rename/copy",
        "cleaning_or_128d_mapping_applied": False,
        "video_features": 0,
        "parallel": {
            "persistent_workers": args.workers,
            "max_inflight_units": args.max_inflight_units,
            "deterministic_plan_order": [
                unit.key for info in infos for unit in local[info.spec.name]
            ],
        },
    }


def _validate_success(final: Path) -> dict[str, Any]:
    if (final / "_INCOMPLETE").exists():
        raise ConversionError(f"published output still has _INCOMPLETE: {final}")
    marker = read_json_object(final / "_SUCCESS", "ARCap success marker")
    manifest = final / "collection_manifest.json"
    if not manifest.is_file() or marker.get("collection_manifest_sha256") != sha256_file(manifest):
        raise ConversionError(f"invalid ARCap success publication: {final}")
    return marker


def _validate_published_collection(
    final: Path,
    infos: Sequence[ARCapPartitionInfo],
    *,
    expected_fingerprint: str,
    cache_root: Path,
) -> dict[str, Any]:
    """Reopen every published partition before accepting ``--skip-existing``."""

    marker = _validate_success(final)
    if marker.get("fingerprint") != expected_fingerprint:
        raise ConversionError(
            "published ARCap output fingerprint differs from the requested source, "
            "selection, schema, or conversion configuration"
        )
    manifest = read_json_object(final / "collection_manifest.json", "ARCap collection manifest")
    records = manifest.get("partitions")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ConversionError("ARCap collection manifest partitions must be a list")
    expected_names = [info.spec.name for info in infos]
    if [record.get("name") for record in records if isinstance(record, dict)] != expected_names:
        raise ConversionError("ARCap collection manifest partition order or selection changed")
    for info, record in zip(infos, records, strict=True):
        expected = {
            "name": info.spec.name,
            "relative_path": info.spec.name,
            "episodes": len(info.plan.episodes),
            "frames": info.plan.num_frames,
            "source_relative_path": info.source_relative_path,
            "source_schema_fingerprint": info.schema_fingerprint,
            "source_sha256": info.spec.sha256,
            "phase_group_size": info.spec.phase_group_size,
            "instruction": info.spec.instruction,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ConversionError(
                    f"published ARCap partition {info.spec.name} changed {key}: "
                    f"{record.get(key)!r} != {value!r}"
                )
        partition_root = final / info.spec.name
        with _bounded_dataset_cache(cache_root / info.spec.name):
            validate_written_dataset(info.plan, partition_root)
        partition_manifest = read_json_object(
            partition_root / "conversion_manifest.json",
            f"ARCap {info.spec.name} conversion manifest",
        )
        partition_expected = {
            "dataset_uid": info.plan.dataset_uid,
            "source_format": "arcap_hdf5",
            "robot_type": info.plan.robot_type,
            "fps": info.plan.fps,
            "num_episodes": len(info.plan.episodes),
            "num_frames": info.plan.num_frames,
            "num_video_features": 0,
        }
        for key, value in partition_expected.items():
            if partition_manifest.get(key) != value:
                raise ConversionError(
                    f"published ARCap partition manifest {info.spec.name} changed {key}"
                )
    return marker


def _write_log(layout: StagingLayout, status: str, **values: Any) -> None:
    atomic_write_json(
        layout.logs / f"{layout.run_id}.json",
        {
            "schema_version": 1,
            "status": status,
            "updated_unix": time.time(),
            "runtime": layout.as_dict(),
            **values,
        },
    )


def _write_event(layout: StagingLayout, event: str, **values: Any) -> None:
    _append_jsonl(
        _coordinator_log_path(layout),
        {"event": event, "run_id": layout.run_id, "pid": os.getpid(), **values},
    )


def _remove_tree_with_retries(path: Path, *, attempts: int = 8) -> None:
    """Tolerate OSSFS's briefly stale directory listings after file removal."""

    for attempt in range(attempts):
        if not path.exists():
            return
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        time.sleep(0.1 * (attempt + 1))
    raise OSError(f"could not remove runtime tree after {attempts} attempts: {path}")


class _RunProgress:
    """Unit-boundary aggregate progress suitable for nohup and JSONL tailing."""

    def __init__(
        self,
        *,
        layout: StagingLayout,
        args: argparse.Namespace,
        units: Sequence[ParallelWorkUnit],
        pending: Sequence[ParallelWorkUnit],
        reused: int,
        estimate: dict[str, Any],
        active_workers: int,
    ) -> None:
        self.layout = layout
        self.args = args
        self.units = tuple(units)
        self.total_frames = sum(unit.weight for unit in units)
        pending_indices = {unit.index for unit in pending}
        self.completed_indices = {
            unit.index for unit in units if unit.index not in pending_indices
        }
        self.completed_frames = sum(
            unit.weight for unit in units if unit.index in self.completed_indices
        )
        self.initial_completed_frames = self.completed_frames
        self.reused = reused
        self.estimate = estimate
        self.active_workers = active_workers
        self.active: dict[int, str] = {}
        self.worker_metrics: dict[int, dict[str, Any]] = {}
        self.started = time.monotonic()
        self.last_emit = 0.0

    def dispatched(self, unit: ParallelWorkUnit, check: Any | None = None) -> None:
        self.active[unit.index] = unit.key
        self.emit("unit_dispatched", check=check)

    def completed(self, unit: ParallelWorkUnit, value: Any) -> None:
        self.active.pop(unit.index, None)
        if unit.index not in self.completed_indices:
            self.completed_indices.add(unit.index)
            self.completed_frames += unit.weight
        if isinstance(value, dict) and isinstance(value.get("worker_slot"), int):
            self.worker_metrics[int(value["worker_slot"])] = dict(value)

    def _snapshot(self, event: str, check: Any | None, *, inventory: bool) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - self.started)
        converted = max(0, self.completed_frames - self.initial_completed_frames)
        rate = converted / elapsed if elapsed else 0.0
        remaining_frames = max(0, self.total_frames - self.completed_frames)
        eta = remaining_frames / rate if rate else (0.0 if not remaining_frames else None)
        completed_units = len(self.completed_indices)
        snapshot: dict[str, Any] = {
            "event": event,
            "completed_units": completed_units,
            "total_units": len(self.units),
            "completed_frames": self.completed_frames,
            "total_frames": self.total_frames,
            "percent": 100.0 * self.completed_frames / self.total_frames,
            "aggregate_frames_per_second": rate,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "reused_verified_units": self.reused,
            "acceleration_mode": self.args.acceleration_mode,
            "active_workers": len(self.active),
            "configured_active_workers": self.active_workers,
            "configured_workers": self.args.workers,
            "active_units": [self.active[index] for index in sorted(self.active)],
            "worker_current_unit_source": str(
                self.layout.logs / f"{self.layout.run_id}.worker-*.jsonl"
            ),
            "encoder": "none (ARCap has no video features)",
            "encoder_threads_per_worker": self.args.encoder_threads_per_worker,
            "worker_failures": 0,
            "worker_retries": 0,
            "ossfs_errors": 0,
            "ossfs_retries": 0,
            "per_worker_latest": {
                str(slot): {
                    key: metrics[key]
                    for key in (
                        "unit",
                        "frames_per_second",
                        "wall_seconds",
                        "cpu_seconds",
                        "peak_rss_bytes",
                    )
                    if key in metrics
                }
                for slot, metrics in sorted(self.worker_metrics.items())
            },
            "estimated_remaining_output_bytes": math.ceil(
                self.estimate["expected_output_bytes"]
                * remaining_frames
                / self.total_frames
            ),
        }
        if check is not None:
            snapshot["quota"] = check.as_dict()
        if inventory:
            roots = {
                "output": (self.layout.final,),
                "work": (self.layout.work,),
                "checkpoint": (self.layout.resume,),
                "logs": (self.layout.logs,),
            }
            snapshot["storage"] = {
                name: {"bytes": values[0], "objects": values[1]}
                for name, paths in roots.items()
                for values in (_usage_bytes_and_objects(paths),)
            }
        return snapshot

    def emit(
        self,
        event: str,
        *,
        check: Any | None = None,
        force: bool = False,
        inventory: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last_emit < self.args.eta_interval_seconds:
            return
        snapshot = self._snapshot(event, check, inventory=inventory)
        _write_event(self.layout, "progress", progress=snapshot)
        _write_log(self.layout, "running", progress=snapshot)
        print(json.dumps({"progress": snapshot}, sort_keys=True), file=sys.stderr, flush=True)
        self.last_emit = now


def _run_conversion(args: argparse.Namespace) -> int:
    validate_source_and_output_roots(args.raw_root, args.output_root)
    layout = _layout(args)
    environment = _configure_runtime_environment(layout, create=False)
    validate_no_runtime_paths_outside_root(layout)
    infos = _inspect_infos(args, layout)
    units, local = _build_units(infos, layout, args)
    estimate = _estimate(infos, units)
    report = {
        "runtime": layout.as_dict(),
        "runtime_environment": environment,
        "partitions": [plan_summary(info.plan) for info in infos],
        "work_units": len(units),
        "storage_estimate": estimate,
        "payload_scan": {
            "metadata_schema_reference": "all selected episodes",
            "sampled_pointcloud_frames": sum(
                info.payload_scan["sampled_pointcloud_frames"] for info in infos
            ),
            "full_lowdim_scan": args.full_lowdim_scan,
            "full_pointcloud_scan": args.full_pointcloud_scan,
        },
    }
    print(json.dumps(report, indent=2))
    if args.inspect_only or args.dry_run or args.estimate_storage:
        return 0

    payload = _collection_payload(infos, args)
    expected_fingerprint = canonical_fingerprint(payload)
    success = layout.final / "_SUCCESS"
    incomplete = layout.final / "_INCOMPLETE"
    if success.is_file():
        _validate_success(layout.final)
        if not args.skip_existing:
            raise FileExistsError(f"published output already exists: {layout.final}")
        layout.create_runtime_directories()
        _configure_runtime_environment(layout, create=True)
        validate_no_runtime_paths_outside_root(layout)
        try:
            with exclusive_staging_lock(layout.lock):
                _write_event(layout, "skip_validation_started")
                _validate_published_collection(
                    layout.final,
                    infos,
                    expected_fingerprint=expected_fingerprint,
                    cache_root=layout.work / "published-validation-cache",
                )
                _remove_tree_with_retries(layout.work)
                _remove_tree_with_retries(layout.resume)
                _write_event(layout, "skip_validation_succeeded")
                _write_log(
                    layout,
                    "skipped_valid_existing",
                    fingerprint=expected_fingerprint,
                )
        except BaseException as exc:
            with suppress(Exception):
                _write_event(
                    layout,
                    "skip_validation_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                _write_log(
                    layout,
                    "failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise
        layout.lock.unlink(missing_ok=True)
        print(f"skipped fully revalidated published ARCap collection: {layout.final}")
        return 0
    if layout.final.exists() and not incomplete.is_file():
        raise ConversionError(f"existing output is neither complete nor resumable: {layout.final}")
    if incomplete.is_file() and not args.resume:
        raise ConversionError(f"incomplete output exists; rerun with --resume: {layout.final}")

    layout.create_runtime_directories()
    _configure_runtime_environment(layout, create=True)
    validate_no_runtime_paths_outside_root(layout)
    guard = StagingQuotaGuard(
        staging_roots=(layout.final, layout.work, layout.resume, layout.logs, layout.lock),
        inflight_roots=(layout.work,),
        max_staging_bytes=args.max_staging_bytes,
        max_inflight_bytes=args.max_inflight_bytes,
        max_inflight_units=args.max_inflight_units,
        interval_seconds=args.storage_check_interval_seconds,
    )
    active_workers = min(args.workers, args.max_inflight_units, len(units))
    if not args.benchmark_child and args.max_phase_groups is None and active_workers < 2:
        raise ConversionError("formal ARCap conversion requires at least two active work units")
    validate_inflight_budget(
        units,
        active_workers,
        memory_budget_bytes=active_workers * args.worker_memory_limit_bytes,
        temp_budget_bytes=args.max_inflight_bytes,
    )

    lock_acquired = False
    try:
        with exclusive_staging_lock(layout.lock):
            lock_acquired = True
            try:
                fingerprint = _prepare_collection_state(
                    layout,
                    payload,
                    require_existing=args.resume and incomplete.is_file(),
                )
                create_incomplete_output(
                    layout.final, fingerprint=fingerprint, run_id=layout.run_id
                )
                _write_event(
                    layout,
                    "run_started",
                    fingerprint=fingerprint,
                    report=report,
                )
                _write_log(layout, "running", fingerprint=fingerprint, report=report)
                pending_global: list[ParallelWorkUnit] = []
                reused = 0
                for info in infos:
                    partition = info.spec.name
                    partition_root = info.plan.output_path
                    partition_root.mkdir(parents=True, exist_ok=True)
                    direct = prepare_direct_commits(
                        local[partition],
                        partition_name=partition,
                        partition_root=partition_root,
                        resume_root=layout.resume,
                    )
                    prepared = (
                        prepare_work_units(
                            direct.uncommitted,
                            _validate_unit,
                            require_complete_plan=False,
                        )
                        if direct.uncommitted
                        else None
                    )
                    reused += len(direct.committed) + (
                        len(prepared.reusable) if prepared else 0
                    )
                    for local_unit in prepared.reusable if prepared else ():
                        commit_verified_unit(
                            local_unit,
                            partition_name=partition,
                            partition_root=partition_root,
                            resume_root=layout.resume,
                        )
                    wanted = {
                        unit.index for unit in (prepared.pending if prepared else ())
                    }
                    pending_global.extend(
                        unit
                        for unit in units
                        if isinstance(unit.payload, _WorkerPayload)
                        and unit.payload.plan.output_path.parent.name == partition
                        and unit.payload.local_index in wanted
                    )
                print(
                    f"resume reused {reused}/{len(units)} verified work units",
                    file=sys.stderr,
                    flush=True,
                )
                pending_frames = sum(unit.weight for unit in pending_global)
                pending_workers = min(active_workers, len(pending_global))
                check = guard.check(
                    "before worker dispatch",
                    required_staging_bytes=math.ceil(
                        estimate["conservative_output_bytes"]
                        * pending_frames
                        / sum(unit.weight for unit in units)
                    ),
                    required_inflight_bytes=sum(
                        sorted(
                            (
                                unit.estimated_temp_bytes
                                for unit in pending_global
                            ),
                            reverse=True,
                        )[:pending_workers]
                    ),
                    inflight_units=pending_workers,
                )
                print(format_staging_quota_check(check), file=sys.stderr, flush=True)
                progress = _RunProgress(
                    layout=layout,
                    args=args,
                    units=units,
                    pending=pending_global,
                    reused=reused,
                    estimate=estimate,
                    active_workers=active_workers,
                )
                progress.emit(
                    "resume_state", check=check, force=True, inventory=True
                )
                by_global_index = {unit.index: unit for unit in pending_global}

                def commit_result(result: Any) -> None:
                    global_unit = by_global_index[result.index]
                    local_unit = _local_unit(global_unit)
                    payload_value = global_unit.payload
                    assert isinstance(payload_value, _WorkerPayload)
                    partition = payload_value.plan.output_path.parent.name
                    commit_verified_unit(
                        local_unit,
                        partition_name=partition,
                        partition_root=layout.final / partition,
                        resume_root=layout.resume,
                    )
                    progress.completed(global_unit, result.value)
                    quota = guard.check(
                        f"committed {global_unit.key}",
                        inflight_units=len(progress.active),
                    )
                    progress.emit(
                        "unit_committed", check=quota, force=True, inventory=True
                    )

                def before_dispatch(
                    unit: ParallelWorkUnit, active: tuple[ParallelWorkUnit, ...]
                ) -> None:
                    quota = guard.check(
                        f"dispatch {unit.key}",
                        required_inflight_bytes=unit.estimated_temp_bytes,
                        inflight_units=len(active) + 1,
                    )
                    progress.dispatched(unit, quota)

                def health_check() -> None:
                    quota = guard.check(
                        "workers active", inflight_units=len(progress.active)
                    )
                    progress.emit(
                        "health_check", check=quota, force=True, inventory=True
                    )

                completion_order: tuple[str, ...] = ()
                if pending_global:
                    result = run_parallel_work_units(
                        pending_global,
                        _convert_unit,
                        workers=active_workers,
                        on_result=commit_result,
                        initializer=_initialize_worker,
                        initargs=(args.worker_memory_limit_bytes,),
                        health_check=health_check,
                        health_check_interval_seconds=args.storage_check_interval_seconds,
                        before_dispatch=before_dispatch,
                    )
                    completion_order = result.completion_order

                for info in infos:
                    partition = info.spec.name
                    verified = prepare_direct_commits(
                        local[partition],
                        partition_name=partition,
                        partition_root=info.plan.output_path,
                        resume_root=layout.resume,
                    )
                    if verified.uncommitted:
                        raise ConversionError(
                            f"partition {partition} has uncommitted work units after workers"
                        )
                    with _bounded_dataset_cache(
                        layout.work / "final-validation-cache" / partition
                    ):
                        finalize_direct_partition(
                            info.plan,
                            local[partition],
                            info.plan.output_path,
                            resume_root=layout.resume,
                            reader_format="arcap_hdf5",
                            parallel_evidence={
                                "workers": active_workers,
                                "persistent_worker_pool": True,
                                "worker_memory_rss_watchdog_bytes": (
                                    args.worker_memory_limit_bytes
                                ),
                                "worker_memory_check_interval_seconds": 0.25,
                                "encoder_threads_per_worker": (
                                    args.encoder_threads_per_worker
                                ),
                                "encoder_unused_no_video_features": True,
                                "completion_order": list(completion_order),
                                "reused_verified_units": reused,
                            },
                        )

                manifest_path = layout.final / "collection_manifest.json"
                atomic_write_json(manifest_path, _manifest(infos, local, args))
                guard.check("final publication")
                publish_success(
                    layout.final,
                    fingerprint=fingerprint,
                    evidence={
                        "dataset_uid": args.output_dataset_uid,
                        "partitions": [info.spec.name for info in infos],
                        "collection_manifest_sha256": sha256_file(manifest_path),
                    },
                )
                _validate_published_collection(
                    layout.final,
                    infos,
                    expected_fingerprint=fingerprint,
                    cache_root=layout.work / "published-validation-cache",
                )
                progress.emit("published", force=True, inventory=True)
                _write_event(layout, "run_succeeded", fingerprint=fingerprint)
                _write_log(layout, "succeeded", fingerprint=fingerprint)
                _remove_tree_with_retries(layout.work)
                _remove_tree_with_retries(layout.resume)
            except BaseException as exc:
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                with suppress(Exception):
                    _write_event(
                        layout,
                        f"run_{status}",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        resume_hint="rerun the same command with --resume",
                    )
                    _write_log(
                        layout,
                        status,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        resume_hint="rerun the same command with --resume",
                    )
                raise
    except BaseException as exc:
        if not lock_acquired:
            with suppress(Exception):
                atomic_write_json(
                    layout.logs / f"{layout.run_id}.lock-refused-{os.getpid()}.json",
                    {
                        "schema_version": 1,
                        "status": "lock_refused",
                        "updated_unix": time.time(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "runtime": layout.as_dict(),
                    },
                )
        raise
    layout.lock.unlink(missing_ok=True)
    print(f"validated and marker-published ARCap collection: {layout.final}")
    return 0


def _strip_option(argv: list[str], option: str, *, many: bool = False) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == option:
            index += 1
            if many:
                while index < len(argv) and not argv[index].startswith("--"):
                    index += 1
            elif index < len(argv):
                index += 1
            continue
        if argv[index].startswith(option + "="):
            index += 1
            continue
        result.append(argv[index])
        index += 1
    return result


def _summarize_worker_logs(logs: Path) -> dict[str, Any]:
    by_worker: dict[int, dict[str, Any]] = {}
    failures = 0
    for path in sorted(logs.glob("*.worker-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            slot = int(event.get("worker_slot", -1))
            row = by_worker.setdefault(
                slot,
                {
                    "worker_slot": slot,
                    "units": 0,
                    "episodes": 0,
                    "frames": 0,
                    "busy_wall_seconds": 0.0,
                    "cpu_seconds": 0.0,
                    "peak_rss_bytes": 0,
                    "peak_unit_output_bytes": 0,
                    "read_bytes": 0,
                    "write_bytes": 0,
                    "read_chars": 0,
                    "write_chars": 0,
                    "failures": 0,
                    "retries": 0,
                    "encoder_errors": 0,
                    "ossfs_errors": 0,
                    "ossfs_retries": 0,
                },
            )
            if event.get("event") == "unit_failed":
                row["failures"] += 1
                failures += 1
            if event.get("event") != "unit_succeeded":
                continue
            row["units"] += 1
            for key in ("episodes", "frames", "read_bytes", "write_bytes", "read_chars", "write_chars"):
                row[key] += int(event.get(key, 0))
            row["busy_wall_seconds"] += float(event.get("wall_seconds", 0.0))
            row["cpu_seconds"] += float(event.get("cpu_seconds", 0.0))
            row["peak_rss_bytes"] = max(
                row["peak_rss_bytes"], int(event.get("peak_rss_bytes", 0))
            )
            row["peak_unit_output_bytes"] = max(
                row["peak_unit_output_bytes"], int(event.get("unit_output_bytes", 0))
            )
            row["retries"] += int(event.get("retries", 0))
            row["encoder_errors"] += int(event.get("encoder_errors", 0))
            row["ossfs_errors"] += int(event.get("ossfs_errors", 0))
            row["ossfs_retries"] += int(event.get("ossfs_retries", 0))
    workers = []
    for slot in sorted(by_worker):
        row = by_worker[slot]
        wall = float(row["busy_wall_seconds"])
        row["frames_per_busy_second"] = row["frames"] / wall if wall else 0.0
        row["average_cpu_cores_while_busy"] = row["cpu_seconds"] / wall if wall else 0.0
        workers.append(row)
    return {
        "workers": workers,
        "failures": failures,
        "retries": sum(row["retries"] for row in workers),
        "encoder_errors": sum(row["encoder_errors"] for row in workers),
        "ossfs_errors": sum(row["ossfs_errors"] for row in workers),
        "ossfs_retries": sum(row["ossfs_retries"] for row in workers),
    }


def _benchmark_output_summary(root: Path) -> dict[str, Any]:
    manifest = read_json_object(root / "collection_manifest.json", "benchmark manifest")
    partitions = list(manifest["partitions"])
    total_videos = 0
    checksums: dict[str, Any] = {}
    for record in partitions:
        name = str(record["name"])
        info = read_json_object(root / name / "meta/info.json", f"benchmark {name} info")
        total_videos += int(info.get("total_videos", 0))
        checksums[name] = _bulk_checksum(root / name)
    combined = hashlib.sha256(
        json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "partition_names": [str(row["name"]) for row in partitions],
        "episodes": sum(int(row["episodes"]) for row in partitions),
        "frames": sum(int(row["frames"]) for row in partitions),
        "videos": total_videos,
        "bulk_checksums": checksums,
        "combined_bulk_sha256": combined,
        "output_bytes": scoped_size((root,)),
    }


def _benchmark_equivalence(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_summary = _benchmark_output_summary(reference)
    candidate_summary = _benchmark_output_summary(candidate)
    if (
        reference_summary["partition_names"]
        != candidate_summary["partition_names"]
    ):
        raise ConversionError("benchmark partition selection changed between worker counts")
    reports = {}
    for name in reference_summary["partition_names"]:
        reports[name] = verify_lerobot_equivalence(
            reference / name,
            candidate / name,
            compare_video_frames=True,
            storage_layout_independent=True,
        ).as_dict()
    return {
        "semantic_equivalence": True,
        "partition_reports": reports,
        "bulk_checksums_equal": (
            reference_summary["bulk_checksums"]
            == candidate_summary["bulk_checksums"]
        ),
        "reference_combined_bulk_sha256": reference_summary[
            "combined_bulk_sha256"
        ],
        "candidate_combined_bulk_sha256": candidate_summary[
            "combined_bulk_sha256"
        ],
    }


def _run_benchmarks(raw_argv: list[str], args: argparse.Namespace) -> int:
    if args.max_phase_groups is None and args.max_work_units is None:
        raise ConversionError("--benchmark-workers requires a bounded real-data selection")
    candidates = sorted(set(args.benchmark_workers))
    if not candidates or candidates[0] != 1 or len(candidates) < 2:
        raise ConversionError("--benchmark-workers must include 1 and at least one parallel candidate")
    base = _strip_option(raw_argv, "--benchmark-workers", many=True)
    base = _strip_option(base, "--workers")
    base = _strip_option(base, "--output-dataset-uid")
    base = _strip_option(base, "--resume-dir")
    base = _strip_option(base, "--work-dir")
    base = _strip_option(base, "--logs-dir")
    base = _strip_option(base, "--temp-dir")
    base = _strip_option(base, "--benchmark-report")
    records: list[dict[str, Any]] = []
    token = uuid.uuid4().hex[:10]
    runtime_paths: list[tuple[Path, Path, Path, Path, Path]] = []
    baseline_output: Path | None = None
    report: dict[str, Any] | None = None
    try:
        for workers in candidates:
            uid = f"arcap-benchmark-{token}-w{workers}"
            output = args.output_root / uid
            work = args.output_root / ".conversion_work" / "arcap" / uid
            resume = args.output_root / ".conversion_resume" / "arcap" / uid
            logs = args.output_root / ".conversion_logs" / "arcap" / uid
            lock = args.output_root / ".conversion_locks" / f"{uid}.lock"
            runtime_paths.append((output, logs, work, resume, lock))
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *base,
                "--workers", str(workers),
                "--max-inflight-units", str(workers),
                "--output-dataset-uid", uid,
                "--work-dir", str(work),
                "--resume-dir", str(resume),
                "--logs-dir", str(logs),
                "--temp-dir", str(work / "tmp"),
                "--benchmark-child",
            ]
            sampler = ProcessTreeSampler(work.parent, interval_seconds=0.25)
            sampler.start()
            try:
                completed = subprocess.run(command, check=False)
            finally:
                metrics = sampler.stop()
            worker_metrics = _summarize_worker_logs(logs)
            if completed.returncode:
                raise ConversionError(
                    f"benchmark child w{workers} failed with exit {completed.returncode}; "
                    f"worker_failures={worker_metrics['failures']}"
                )
            output_summary = _benchmark_output_summary(output)
            wall = metrics.wall_seconds
            row: dict[str, Any] = {
                "workers": workers,
                "wall_seconds": wall,
                "episodes": output_summary["episodes"],
                "frames": output_summary["frames"],
                "videos": output_summary["videos"],
                "frames_per_second": output_summary["frames"] / wall,
                "episodes_per_second": output_summary["episodes"] / wall,
                "cpu_seconds": metrics.cpu_seconds,
                "average_cpu_cores": metrics.average_cpu_cores,
                "average_cpu_percent": metrics.average_cpu_cores * 100.0,
                "peak_process_tree_rss_bytes": metrics.peak_rss_bytes,
                "peak_temp_bytes": metrics.peak_temp_bytes,
                "io_wait_seconds": metrics.io_wait_seconds,
                "process_tree_read_bytes": metrics.read_bytes,
                "process_tree_write_bytes": metrics.write_bytes,
                "process_tree_read_chars": metrics.read_chars,
                "process_tree_write_chars": metrics.write_chars,
                "physical_io_counters_available": metrics.io_counters_available,
                "physical_read_bytes_per_second": metrics.read_bytes / wall,
                "physical_write_bytes_per_second": metrics.write_bytes / wall,
                "userspace_read_chars_per_second": metrics.read_chars / wall,
                "userspace_write_chars_per_second": metrics.write_chars / wall,
                "staging_output_bytes": output_summary["output_bytes"],
                "staging_output_bytes_per_second": output_summary["output_bytes"] / wall,
                "combined_bulk_sha256": output_summary["combined_bulk_sha256"],
                "worker_metrics": worker_metrics,
                "failures": worker_metrics["failures"],
                "retries": worker_metrics["retries"],
                "encoder_errors": worker_metrics["encoder_errors"],
                "ossfs_errors": worker_metrics["ossfs_errors"],
                "ossfs_retries": worker_metrics["ossfs_retries"],
            }
            if baseline_output is None:
                baseline_output = output
                row["equivalence_to_w1"] = {
                    "semantic_equivalence": True,
                    "bulk_checksums_equal": True,
                    "reference_combined_bulk_sha256": output_summary[
                        "combined_bulk_sha256"
                    ],
                    "candidate_combined_bulk_sha256": output_summary[
                        "combined_bulk_sha256"
                    ],
                }
            else:
                row["equivalence_to_w1"] = _benchmark_equivalence(
                    baseline_output, output
                )
            records.append(row)

        baseline = records[0]["frames_per_second"]
        for row in records:
            row["speedup"] = row["frames_per_second"] / baseline
            row["parallel_efficiency"] = row["speedup"] / row["workers"]
        parallel = [row for row in records if row["workers"] > 1]
        gate_passed = any(
            row["speedup"] > 1.0
            and row["equivalence_to_w1"]["semantic_equivalence"]
            for row in parallel
        )
        selected = max(records, key=lambda row: row["frames_per_second"])
        report = {
            "schema_version": 1,
            "benchmark_token": token,
            "benchmark": records,
            "parallel_gate_passed": gate_passed,
            "selected_workers": selected["workers"] if gate_passed else None,
            "selection_rule": (
                "fastest semantically equivalent candidate with positive speedup over W1"
            ),
            "nvenc_used": False,
            "outputs_cleaned": False,
        }
    finally:
        cleanup_errors = []
        for output, logs, work_parent, resume, lock in runtime_paths:
            for path in (output, logs, work_parent, resume):
                if path.exists():
                    try:
                        _remove_tree_with_retries(path)
                    except OSError as exc:
                        cleanup_errors.append(str(exc))
            lock.unlink(missing_ok=True)
        residual = [
            str(path)
            for paths in runtime_paths
            for path in paths
            if path.exists()
        ]
        if report is not None:
            report["outputs_cleaned"] = not cleanup_errors and not residual
            report["cleanup_errors"] = cleanup_errors
            report["cleanup_residual_paths"] = residual
            if args.benchmark_report is not None:
                report_path = validate_contained_path(
                    args.benchmark_report,
                    args.output_root,
                    label="benchmark report",
                )
                atomic_write_json(report_path, report)
        if cleanup_errors or residual:
            raise OSError(
                "benchmark cleanup left residual paths: "
                + "; ".join([*cleanup_errors, *residual])
            )
    assert report is not None
    print(json.dumps(report, indent=2))
    if not report["parallel_gate_passed"]:
        raise ConversionError(
            "real-data benchmark found no semantically equivalent parallel speedup"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dataset-uid", default="arcap")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--partition", action="append", choices=OFFICIAL_PARTITIONS)
    parser.add_argument("--max-phase-groups", "--phase-group-limit", type=_positive_int)
    parser.add_argument("--max-work-units", type=_positive_int)
    parser.add_argument("--max-frames-per-unit", type=_positive_int, default=DEFAULT_MAX_FRAMES_PER_UNIT)
    parser.add_argument("--workers", "--partition-workers", type=_positive_int, default=4)
    parser.add_argument("--max-inflight-units", type=_positive_int, default=4)
    parser.add_argument("--max-staging-bytes", type=_positive_int, default=DEFAULT_MAX_STAGING_BYTES)
    parser.add_argument("--max-inflight-bytes", type=_positive_int, default=DEFAULT_MAX_INFLIGHT_BYTES)
    parser.add_argument("--worker-memory-limit-bytes", type=_positive_int, default=DEFAULT_WORKER_MEMORY_LIMIT_BYTES)
    parser.add_argument("--encoder-threads-per-worker", type=_positive_int, default=1)
    parser.add_argument("--storage-check-interval-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--eta-interval-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--acceleration-mode", choices=("hardware", "parallel"), default="parallel")
    parser.add_argument("--benchmark-workers", nargs="+", type=_positive_int)
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        help="optional JSON report path below --output-root",
    )
    parser.add_argument("--benchmark-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--estimate-storage", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-source-sha256", action="store_true")
    parser.add_argument("--full-lowdim-scan", action="store_true")
    parser.add_argument("--full-pointcloud-scan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(raw_argv)
    if args.overwrite:
        parser.error("--overwrite is disabled for marker-published OSSFS datasets")
    if args.acceleration_mode != "parallel":
        parser.error("hardware mode is unavailable: all four A800 NVENC preflights failed")
    if args.max_inflight_units > args.workers:
        parser.error("--max-inflight-units cannot exceed --workers")
    if not args.raw_root.is_dir():
        parser.error(f"raw root does not exist: {args.raw_root}")
    bounded = args.max_phase_groups is not None or args.max_work_units is not None
    if bounded and args.output_dataset_uid == "arcap" and not (
        args.dry_run or args.inspect_only or args.estimate_storage
    ) and not args.benchmark_workers:
        parser.error("bounded conversion requires an independent --output-dataset-uid")
    if not bounded and args.workers < 2:
        parser.error("formal conversion cannot run with one worker")
    if args.benchmark_workers and args.benchmark_child:
        parser.error("benchmark parent and child modes are mutually exclusive")
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        try:
            if args.benchmark_workers:
                return _run_benchmarks(raw_argv, args)
            return _run_conversion(args)
        except KeyboardInterrupt:
            print(
                "interrupted; verified work units retained; rerun with --resume",
                file=sys.stderr,
            )
            return 130
        except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
            details = [f"{type(exc).__name__}: {exc}"]
            cause = exc.__cause__
            while cause is not None:
                details.append(f"{type(cause).__name__}: {cause}")
                cause = cause.__cause__
            print("error: " + " <- ".join(details), file=sys.stderr)
            return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
