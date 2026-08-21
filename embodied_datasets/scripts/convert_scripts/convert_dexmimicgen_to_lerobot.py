"""Convert the official DexMimicGen HDF5 release to nine LeRobot v3 partitions.

The source containers have nine genuinely different schemas.  The coordinator
therefore freezes one plan per container, runs only isolated partition writers,
and publishes a collection root after every child and XML sidecar revalidates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
from typing import Any, Sequence
import uuid
import zlib

from convert_core.checkpoint import (
    RESUME_SCHEMA_VERSION,
    atomic_write_json,
    build_resume_payload,
    canonical_fingerprint,
    exclusive_resume_lock,
    read_json_object,
)
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import (
    convert_dataset,
    plan_summary,
    validate_video_files,
    validate_written_dataset,
)
from convert_core.parallel import (
    ParallelWorkUnit,
    inflight_estimate,
    run_parallel_work_units,
)
from convert_core.storage import (
    StagingQuotaGuard,
    directory_size,
    filesystem_snapshot,
    format_staging_quota_check,
    make_storage_estimate,
    validate_paths_within_root,
)
from readers.dexmimicgen_hdf5_reader import (
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    DexMimicGenPartitionInfo,
    inspect_partition,
    iter_frames,
)


GIB = 1024**3
MIB = 1024**2
PARTITION_FILES = (
    "generated/two_arm_box_cleanup.hdf5",
    "generated/two_arm_can_sort_random.hdf5",
    "generated/two_arm_coffee.hdf5",
    "generated/two_arm_drawer_cleanup.hdf5",
    "generated/two_arm_lift_tray.hdf5",
    "generated/two_arm_pouring.hdf5",
    "generated/two_arm_threading.hdf5",
    "generated/two_arm_three_piece_assembly.hdf5",
    "generated/two_arm_transport.hdf5",
)
PARTITION_NAMES = tuple(
    Path(value).stem.removeprefix("two_arm_") for value in PARTITION_FILES
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0"
)
DEFAULT_RAW_ROOT = Path(
    "/mnt/data/embodied_datasets/public_datasets_raw/dexmimicgen"
)
MAX_TOTAL_ENCODER_THREADS = 32
DEFAULT_MAX_STAGING_BYTES = 64 * GIB
DEFAULT_MAX_INFLIGHT_BYTES = 8 * GIB
DEFAULT_EPISODES_PER_PART = 64
INCOMPLETE_MARKER = "_INCOMPLETE"
SUCCESS_MARKER = "_SUCCESS"


@dataclass(frozen=True)
class RuntimePaths:
    output_root: Path
    final_output: Path
    work_dir: Path
    resume_dir: Path
    logs_dir: Path
    temp_dir: Path
    cache_dir: Path
    lock_path: Path


@dataclass(frozen=True)
class _WorkerPayload:
    plan: Any
    temp_dir: str
    resume_dir: str
    log_path: str
    staging_roots: tuple[str, ...]
    inflight_roots: tuple[str, ...]
    max_staging_bytes: int
    max_inflight_bytes: int
    max_inflight_units: int
    storage_check_interval_seconds: float
    encoder_threads: int
    encoder_threads_per_worker: int
    encoder_queue_maxsize: int
    eta_interval_seconds: float
    conversion_options: dict[str, Any]
    partition_output_upper_bytes: int
    episodes_per_part: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    output_root = args.output_root
    final_output = args.output or output_root / "dexmimicgen"
    run_id = final_output.name
    local_runtime_root = args.local_runtime_root
    if local_runtime_root is None:
        work_dir = args.work_dir or (
            output_root / ".conversion_work" / "dexmimicgen" / run_id
        )
    else:
        local_runtime_root = local_runtime_root.expanduser().absolute()
        work_dir = args.work_dir or local_runtime_root / "work"
    resume_dir = args.resume_dir or (
        output_root / ".conversion_resume" / "dexmimicgen"
    )
    logs_dir = args.logs_dir or (
        output_root / ".conversion_logs" / "dexmimicgen"
    )
    temp_dir = args.temp_dir or work_dir / "temp"
    cache_dir = work_dir / "cache"
    lock_path = output_root / ".conversion_locks" / "dexmimicgen.lock"
    resolved = validate_paths_within_root(
        output_root,
        {
            "final output": final_output,
            "--resume-dir": resume_dir,
            "--logs-dir": logs_dir,
            "conversion lock": lock_path,
        },
        required_root=DEFAULT_OUTPUT_ROOT,
    )
    runtime_root = local_runtime_root or output_root
    resolved.update(
        validate_paths_within_root(
            runtime_root,
            {
                "--work-dir": work_dir,
                "--temp-dir": temp_dir,
                "runtime cache": cache_dir,
            },
        )
    )
    root = output_root.expanduser().absolute().resolve(strict=False)
    if resolved["final output"] == root or resolved["final output"].name.startswith("."):
        raise ConversionError("final output must be one non-hidden dataset directory below output root")
    if not _is_relative_to(resolved["--temp-dir"], resolved["--work-dir"]):
        raise ConversionError("--temp-dir must be inside --work-dir")
    if not _is_relative_to(resolved["runtime cache"], resolved["--work-dir"]):
        raise ConversionError("runtime cache must be inside --work-dir")
    isolated = (
        resolved["final output"],
        resolved["--work-dir"],
        resolved["--resume-dir"],
        resolved["--logs-dir"],
    )
    for index, first in enumerate(isolated):
        for second in isolated[index + 1 :]:
            if _is_relative_to(first, second) or _is_relative_to(second, first):
                raise ConversionError(
                    f"runtime roots must not overlap: {first} and {second}"
                )
    return RuntimePaths(
        output_root=root,
        final_output=resolved["final output"],
        work_dir=resolved["--work-dir"],
        resume_dir=resolved["--resume-dir"],
        logs_dir=resolved["--logs-dir"],
        temp_dir=resolved["--temp-dir"],
        cache_dir=resolved["runtime cache"],
        lock_path=resolved["conversion lock"],
    )


def _configure_runtime_environment(paths: RuntimePaths, *, create: bool) -> dict[str, str]:
    values = {
        "TMPDIR": paths.temp_dir,
        "TMP": paths.temp_dir,
        "TEMP": paths.temp_dir,
        "XDG_CACHE_HOME": paths.cache_dir / "xdg",
        "HF_HOME": paths.cache_dir / "huggingface",
        "HF_DATASETS_CACHE": paths.cache_dir / "huggingface" / "datasets",
        "TORCH_HOME": paths.cache_dir / "torch",
        "MPLCONFIGDIR": paths.cache_dir / "matplotlib",
        "VLA_DATASETS_CACHE_ROOT": paths.cache_dir / "datasets",
    }
    if create:
        for path in set(values.values()):
            path.mkdir(parents=True, exist_ok=True)
    rendered = {key: str(path) for key, path in values.items()}
    os.environ.update(rendered)
    # ``tempfile`` caches its selected directory after the first lookup.  An
    # imported dependency may have populated that cache before main() gets a
    # chance to redirect the environment, so reset it explicitly instead of
    # trusting TMPDIR alone.  Pointing at a not-yet-created staging path in
    # estimate-only mode is intentional: an unexpected write must fail rather
    # than fall back to /tmp.
    tempfile.tempdir = rendered["TMPDIR"]
    return rendered


def _append_log(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "unix_time": time.time(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _rgb_encoder() -> Any:
    try:
        from lerobot.configs.video import RGBEncoderConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lerobot==0.6.0 is required") from exc
    return RGBEncoderConfig(
        vcodec="h264",
        crf=18,
        preset="fast",
        extra_options={"tune": "zerolatency"},
    )


def _selected_files(partitions: Sequence[str] | None) -> tuple[str, ...]:
    if not partitions:
        return PARTITION_FILES
    selected = set(partitions)
    return tuple(
        source for source, name in zip(PARTITION_FILES, PARTITION_NAMES) if name in selected
    )


def _build_infos(
    raw_root: Path,
    final_output: Path,
    *,
    partitions: Sequence[str] | None,
    max_episodes: int | None,
    episodes_per_part: int,
) -> list[DexMimicGenPartitionInfo]:
    infos = []
    for relative in _selected_files(partitions):
        info = inspect_partition(
            raw_root / relative,
            raw_dataset_root=raw_root,
            collection_output=final_output,
            max_episodes=max_episodes,
        )
        encoding = {
            "source_storage": "HDF5 uint8 RGB frames",
            "target_codec": "h264",
            "target_quality": 18,
            "target_preset": "fast",
            "target_tune": "zerolatency",
            "target_pix_fmt": "yuv420p",
            "container": "fragmented MP4",
            "movflags": "frag_keyframe+empty_moov+default_base_moof",
            "streaming": True,
            "video_reencoded": True,
            "video_reencoding_lossy": True,
            "hardware_encoder": False,
            "hardware_rejection": (
                "h264/hevc/av1 NVENC real encode failed on every local A800"
            ),
        }
        episodes = tuple(
            replace(
                episode,
                extra={
                    **episode.extra,
                    "checkpoint_unit": (
                        f"{info.partition_name}/part-{episode_index // episodes_per_part:05d}"
                    ),
                    "checkpoint_part_index": episode_index // episodes_per_part,
                },
            )
            for episode_index, episode in enumerate(info.plan.episodes)
        )
        infos.append(
            replace(
                info,
                plan=replace(
                    info.plan,
                    episodes=episodes,
                    extra={
                        **info.plan.extra,
                        "video_encoding": encoding,
                        "checkpointing": {
                            "granularity": "contiguous episode part",
                            "episodes_per_part": episodes_per_part,
                            "per_frame_markers": False,
                            "per_episode_markers": False,
                        },
                    },
                ),
            )
        )
    if not infos:
        raise ConversionError("partition selection is empty")
    return infos


def _storage_estimate(
    infos: Sequence[DexMimicGenPartitionInfo],
    output: Path,
    *,
    workers: int,
    existing_output_bytes: int = 0,
) -> tuple[Any, Any]:
    snapshot = filesystem_snapshot(output)
    numeric = sum(info.selected_numeric_logical_bytes for info in infos)
    image = sum(info.selected_image_logical_bytes for info in infos)
    # Ratios come from the documented full-source scan and real Transport
    # LeRobot benchmark. Ranges deliberately cover observed ratio by 0.5x–2x.
    data_expected = math.ceil(numeric * (9_851_640_279 / 9_038_202_091))
    data_lower = math.ceil(data_expected * 0.69)
    data_upper = math.ceil(data_expected * 1.38)
    video_expected = math.ceil(image * (3_611_207_197 / 202_894_327_440))
    video_lower = math.ceil(video_expected * 0.5)
    video_upper = math.ceil(video_expected * 2.0)
    metadata_upper = 256 * MIB
    checkpoint_upper = 256 * MIB
    active = min(workers, len(infos))
    inflight_upper = active * 2 * GIB
    encoder_temp_upper = active * 512 * MIB
    final_upper = data_upper + video_upper + metadata_upper
    estimate = make_storage_estimate(
        data_expected_bytes=data_expected,
        data_range_bytes=(data_lower, data_upper),
        video_expected_bytes=video_expected,
        video_range_bytes=(video_lower, video_upper),
        metadata_upper_bytes=metadata_upper,
        checkpoint_state_upper_bytes=checkpoint_upper,
        inflight_upper_bytes=inflight_upper,
        encoder_temp_upper_bytes=encoder_temp_upper,
        final_upper_bytes=final_upper,
        existing_output_bytes=existing_output_bytes,
        snapshot=snapshot,
        safety_reserve_bytes=0,
        method=(
            "numeric logical bytes scaled by full preflight Parquet ratio; image logical "
            "bytes scaled by real Transport H.264 CRF18 benchmark; video range 0.5x–2x; "
            "bounded per-partition inflight and encoder overhead; OSSFS statvfs is "
            "informational and --max-staging-bytes is the authoritative quota"
        ),
    )
    return snapshot, estimate


def _resume_accounting(resume_dir: Path) -> tuple[int, int]:
    """Read lightweight checkpoint ledgers without walking remote snapshots."""

    output_bytes = 0
    state_bytes = 0
    for state_path in sorted(resume_dir.glob("partitions/*/state.json")):
        try:
            state_bytes += state_path.stat().st_size
            state = read_json_object(state_path, "DexMimicGen resume state")
        except (ConversionError, OSError):
            continue
        records = state.get("inventory", [])
        if not isinstance(records, list):
            continue
        output_bytes += sum(
            int(record.get("size", 0))
            for record in records
            if isinstance(record, dict)
        )
    return output_bytes, state_bytes


def _remaining_staging_reservation(
    estimate: Any,
    paths: RuntimePaths,
    *,
    existing_output_bytes: int,
    resume_state_bytes: int,
) -> int:
    """Return conservative additional bytes across final, resume, and work roots."""

    remaining_final = max(
        0,
        estimate.final_output_conservative_upper_bytes
        - existing_output_bytes,
    )
    remaining_checkpoint = max(
        0,
        estimate.checkpoint_state_upper_bytes - resume_state_bytes,
    )
    remaining_work = max(
        0,
        estimate.maximum_inflight_units_upper_bytes
        + estimate.encoder_and_temp_upper_bytes
        - directory_size(paths.work_dir),
    )
    return remaining_final + remaining_checkpoint + remaining_work


def _conversion_options(payload: _WorkerPayload) -> dict[str, Any]:
    return {
        **payload.conversion_options,
        "encoder_threads_per_camera": payload.encoder_threads,
        "encoder_threads_per_worker_budget": payload.encoder_threads_per_worker,
        "encoder_queue_maxsize": payload.encoder_queue_maxsize,
        "blocking_streaming_encoding": True,
    }


def _convert_partition(unit: ParallelWorkUnit) -> dict[str, Any]:
    payload = unit.payload
    if not isinstance(payload, _WorkerPayload):
        raise ConversionError(f"invalid DexMimicGen worker payload for {unit.key}")
    temp_dir = Path(payload.temp_dir) / unit.key
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update({key: str(temp_dir) for key in ("TMPDIR", "TMP", "TEMP")})
    tempfile.tempdir = str(temp_dir)
    guard = StagingQuotaGuard(
        staging_roots=[Path(value) for value in payload.staging_roots],
        inflight_roots=[Path(value) for value in payload.inflight_roots],
        max_staging_bytes=payload.max_staging_bytes,
        max_inflight_bytes=payload.max_inflight_bytes,
        max_inflight_units=payload.max_inflight_units,
        interval_seconds=payload.storage_check_interval_seconds,
    )
    start_check = guard.check(
        f"partition {unit.key} start",
        required_staging_bytes=payload.partition_output_upper_bytes,
        required_inflight_bytes=unit.estimated_temp_bytes,
        inflight_units=1,
    )
    print(format_staging_quota_check(start_check), file=sys.stderr, flush=True)
    _append_log(Path(payload.log_path), "partition_start", partition=unit.key)
    active_checkpoint_unit: str | None = None

    def checked_frames(episode: Any):
        nonlocal active_checkpoint_unit
        checkpoint_unit = str(episode.extra["checkpoint_unit"])
        if checkpoint_unit != active_checkpoint_unit:
            check = guard.check(
                f"checkpoint part {checkpoint_unit} start",
                required_inflight_bytes=unit.estimated_temp_bytes,
                inflight_units=1,
            )
            print(format_staging_quota_check(check), file=sys.stderr, flush=True)
            active_checkpoint_unit = checkpoint_unit
        yield from iter_frames(payload.plan, episode)

    def frame_hook(_episode: Any, _index: int) -> None:
        check = guard.periodic_check(
            f"partition {unit.key} frames",
            required_inflight_bytes=unit.estimated_temp_bytes,
            inflight_units=1,
        )
        if check is not None:
            print(format_staging_quota_check(check), file=sys.stderr, flush=True)

    try:
        convert_dataset(
            payload.plan,
            checked_frames,
            reader_format="dexmimicgen_hdf5",
            resume=True,
            eta_interval_seconds=payload.eta_interval_seconds,
            rgb_encoder=_rgb_encoder(),
            streaming_encoding=True,
            blocking_streaming_encoding=True,
            encoder_queue_maxsize=payload.encoder_queue_maxsize,
            encoder_threads=payload.encoder_threads,
            frame_completed_hook=frame_hook,
            conversion_options=_conversion_options(payload),
            metadata_buffer_size=payload.episodes_per_part,
            resume_data_root=payload.plan.output_path,
            resume_state_root=(
                Path(payload.resume_dir) / "partitions" / unit.key
            ),
            resume_lock_path=(
                Path(payload.resume_dir) / "partition_locks" / f"{unit.key}.lock"
            ),
            publish_on_complete=False,
            cleanup_resume_state=False,
            rebuild_corrupt_checkpoint=True,
            batch_metadata_writes=True,
            encoder_temp_root=temp_dir,
            fragmented_mp4_writes=True,
        )
        validate_written_dataset(payload.plan, Path(unit.target_path))
        videos = validate_video_files(
            payload.plan, Path(unit.target_path), expected_frames=payload.plan.num_frames
        )
        result = {
            "partition": unit.key,
            "episodes": len(payload.plan.episodes),
            "frames": payload.plan.num_frames,
            "video_files": sum(len(rows) for rows in videos.values()),
        }
        _append_log(Path(payload.log_path), "partition_complete", **result)
        return result
    except BaseException as exc:
        _append_log(
            Path(payload.log_path),
            "partition_failed",
            partition=unit.key,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _work_units(
    infos: Sequence[DexMimicGenPartitionInfo],
    args: argparse.Namespace,
    *,
    paths: RuntimePaths,
    final_output_upper_bytes: int,
) -> tuple[ParallelWorkUnit, ...]:
    units = []
    episode_cursor = 0
    frame_cursor = 0
    task_names = list(dict.fromkeys(info.plan.episodes[0].instruction for info in infos))
    task_indices = {task: index for index, task in enumerate(task_names)}
    for index, info in enumerate(infos):
        plan = info.plan
        cameras = len(plan.camera_features)
        encoder_threads = max(1, args.encoder_threads_per_worker // cameras)
        payload = _WorkerPayload(
            plan=plan,
            temp_dir=str(paths.temp_dir),
            resume_dir=str(paths.resume_dir),
            log_path=str(paths.logs_dir / f"partition-{info.partition_name}.jsonl"),
            # Final output and resume snapshots are on OSSFS. Their committed
            # bytes are accounted from checkpoint inventory; runtime scans
            # must stay on local work storage to avoid remote tree walks.
            staging_roots=(str(paths.work_dir),),
            inflight_roots=(str(paths.work_dir),),
            max_staging_bytes=args.max_staging_bytes,
            max_inflight_bytes=args.max_inflight_bytes,
            max_inflight_units=args.max_inflight_units,
            storage_check_interval_seconds=args.storage_check_interval_seconds,
            encoder_threads=encoder_threads,
            encoder_threads_per_worker=args.encoder_threads_per_worker,
            encoder_queue_maxsize=args.encoder_queue_maxsize,
            eta_interval_seconds=args.eta_interval_seconds,
            conversion_options={
                "codec": "h264",
                "crf": 18,
                "preset": "fast",
                "tune": "zerolatency",
                "pix_fmt": "yuv420p",
                "mp4_movflags": "frag_keyframe+empty_moov+default_base_moof",
                "partition_worker_count": args.workers,
                "checkpoint_episodes_per_part": args.episodes_per_checkpoint_part,
                "in_place_final_compatible_chunks": True,
            },
            partition_output_upper_bytes=max(
                1,
                final_output_upper_bytes * plan.num_frames
                // sum(item.plan.num_frames for item in infos),
            ),
            episodes_per_part=args.episodes_per_checkpoint_part,
        )
        fingerprint = canonical_fingerprint(
            build_resume_payload(
                plan,
                reader_format="dexmimicgen_hdf5",
                conversion_options=_conversion_options(payload),
            )
        )
        episode_end = episode_cursor + len(plan.episodes)
        frame_end = frame_cursor + plan.num_frames
        units.append(
            ParallelWorkUnit(
                index=index,
                key=info.partition_name,
                dataset_uid=plan.dataset_uid,
                target_path=str(plan.output_path),
                episode_start=episode_cursor,
                episode_end=episode_end,
                frame_start=frame_cursor,
                frame_end=frame_end,
                task_indices=tuple(
                    task_indices[episode.instruction] for episode in plan.episodes
                ),
                weight=plan.num_frames,
                estimated_memory_bytes=2 * GIB,
                estimated_temp_bytes=512 * MIB,
                fingerprint=fingerprint,
                payload=payload,
            )
        )
        episode_cursor = episode_end
        frame_cursor = frame_end
    return tuple(units)


def _validate_partition(unit: ParallelWorkUnit) -> None:
    payload = unit.payload
    if not isinstance(payload, _WorkerPayload):
        raise ConversionError(f"invalid worker payload for {unit.key}")
    validate_written_dataset(payload.plan, Path(unit.target_path))
    validate_video_files(
        payload.plan, Path(unit.target_path), expected_frames=payload.plan.num_frames
    )


def _write_model_sidecars(
    infos: Sequence[DexMimicGenPartitionInfo], workspace: Path
) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required") from exc
    root = workspace / "source_models"
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for info in infos:
        with h5py.File(info.source_path, "r") as source:
            for reference in info.model_references:
                value = source[f"data/{reference.demo_name}"].attrs["model_file"]
                xml = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                raw = xml.encode("utf-8")
                digest = hashlib.sha256(raw).hexdigest()
                if digest != reference.sha256 or len(raw) != reference.uncompressed_bytes:
                    raise ConversionError(
                        f"model XML identity changed for {info.partition_name}/{reference.demo_name}"
                    )
                path = root / f"{digest}.xml.zlib"
                compressed = zlib.compress(raw, level=9)
                if path.exists():
                    existing = path.read_bytes()
                    if zlib.decompress(existing) != raw:
                        raise ConversionError(f"conflicting model sidecar: {path}")
                else:
                    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
                    temporary.write_bytes(compressed)
                    if zlib.decompress(temporary.read_bytes()) != raw:
                        temporary.unlink(missing_ok=True)
                        raise ConversionError(f"model sidecar verification failed: {path}")
                    temporary.replace(path)
                records.setdefault(
                    digest,
                    {
                        "sha256": digest,
                        "relative_path": path.relative_to(workspace).as_posix(),
                        "uncompressed_bytes": len(raw),
                        "compressed_bytes": path.stat().st_size,
                        "codec": "zlib level 9",
                    },
                )
    return [records[key] for key in sorted(records)]


def _collection_manifest(
    infos: Sequence[DexMimicGenPartitionInfo],
    sidecars: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    paths: RuntimePaths,
    runtime_environment: dict[str, str],
) -> dict[str, Any]:
    global_episode = 0
    partitions = []
    episode_mapping = []
    for info in infos:
        partitions.append(
            {
                "name": info.partition_name,
                "dataset_uid": info.plan.dataset_uid,
                "source_relative_path": info.source_relative_path,
                "source_schema_fingerprint": info.schema_fingerprint,
                "source_env_name": info.plan.extra["source_env_name"],
                "robot_type": info.plan.robot_type,
                "episodes": len(info.plan.episodes),
                "frames": info.plan.num_frames,
                "all_source_episodes": info.all_episode_count,
                "all_source_frames": info.all_frame_count,
                "episode_length_summary": info.episode_length_summary,
                "features": info.plan.feature_schema(),
            }
        )
        for local_index, episode in enumerate(info.plan.episodes):
            episode_mapping.append(
                {
                    "collection_episode_index": global_episode,
                    "partition": info.partition_name,
                    "lerobot_episode_index": local_index,
                    "lerobot_task_index": 0,
                    "source_episode_id": episode.extra["source_episode_id"],
                    "source": episode.source_relative_path,
                    "source_task": episode.extra["source_task"],
                    "instruction": episode.instruction,
                    "source_model_sha256": episode.extra["source_model_sha256"],
                    "num_frames": episode.num_frames,
                }
            )
            global_episode += 1
    return {
        "format": "lerobot_v3_0_collection",
        "dataset_uid": paths.final_output.name,
        "source_dataset": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "partition_reason": (
            "the nine official HDF5 containers have distinct fixed schemas; no padding, "
            "casting, field fabrication, or 128-D mapping is allowed"
        ),
        "partitions": partitions,
        "episodes": episode_mapping,
        "task_index_mapping": {
            info.partition_name: {
                "0": info.plan.episodes[0].instruction,
                "source_env_name": info.plan.extra["source_env_name"],
            }
            for info in infos
        },
        "model_sidecars": list(sidecars),
        "parallel": {
            "requested_workers": args.workers,
            "active_partition_workers": min(
                args.workers, args.max_inflight_units, len(infos)
            ),
            "max_inflight_units": args.max_inflight_units,
            "encoder_threads_per_worker_budget": args.encoder_threads_per_worker,
            "actual_encoder_threads_per_camera": {
                info.partition_name: max(
                    1, args.encoder_threads_per_worker // len(info.plan.camera_features)
                )
                for info in infos
            },
            "deterministic_partition_order": [info.partition_name for info in infos],
            "worker_outputs_are_isolated": True,
        },
        "checkpointing": {
            "episodes_per_part": args.episodes_per_checkpoint_part,
            "state_root": str(paths.resume_dir),
            "partitions_write_final_compatible_chunks_in_place": True,
            "per_frame_markers": False,
            "per_episode_markers": False,
        },
        "publication": {
            "protocol": "_INCOMPLETE then _SUCCESS; no complete-directory rename",
            "success_marker": SUCCESS_MARKER,
            "incomplete_marker": INCOMPLETE_MARKER,
        },
        "storage_limits": {
            "max_staging_bytes": args.max_staging_bytes,
            "max_inflight_bytes": args.max_inflight_bytes,
            "max_inflight_units": args.max_inflight_units,
            "storage_check_interval_seconds": args.storage_check_interval_seconds,
        },
        "runtime_environment": runtime_environment,
        "video_encoding": infos[0].plan.extra["video_encoding"],
        "payload_scan_coverage": {
            "metadata_schema_reference_scan": "all source episodes in selected partitions",
            "full_payload_scan": False,
        },
    }


def _collection_payload(infos: Sequence[DexMimicGenPartitionInfo], args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "codec": "h264",
        "crf": 18,
        "preset": "fast",
        "tune": "zerolatency",
        "pix_fmt": "yuv420p",
        "mp4_movflags": "frag_keyframe+empty_moov+default_base_moof",
        "workers": args.workers,
        "max_inflight_units": args.max_inflight_units,
        "encoder_threads_per_worker": args.encoder_threads_per_worker,
        "encoder_queue_maxsize": args.encoder_queue_maxsize,
        "blocking_streaming_encoding": True,
        "episodes_per_checkpoint_part": args.episodes_per_checkpoint_part,
        "in_place_final_compatible_chunks": True,
    }
    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "kind": "dexmimicgen_collection",
        "partitions": [
            build_resume_payload(
                info.plan,
                reader_format="dexmimicgen_hdf5",
                conversion_options={
                    **common,
                    "encoder_threads_per_camera": max(
                        1,
                        args.encoder_threads_per_worker
                        // len(info.plan.camera_features),
                    ),
                },
            )
            for info in infos
        ],
        "options": common,
    }


def _prepare_collection_resume(
    final_output: Path,
    state_root: Path,
    payload: dict[str, Any],
    *,
    require_existing: bool = False,
) -> None:
    state_path = state_root / "collection.json"
    fingerprint = canonical_fingerprint(payload)
    if state_path.exists():
        if not final_output.is_dir():
            raise ConversionError(
                f"collection resume state exists without output data: {final_output}"
            )
        state = read_json_object(state_path, "DexMimicGen collection resume state")
        if state.get("fingerprint") != fingerprint:
            raise ConversionError(
                "collection resume fingerprint changed; restore the original source, "
                "selection, worker, and encoder arguments or move resume paths aside"
            )
        return
    if require_existing:
        raise ConversionError(
            f"--resume requested but collection state is missing: {state_path}"
        )
    final_output.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        state_path,
        {"fingerprint": fingerprint, "configuration": payload},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--local-runtime-root",
        type=Path,
        help=(
            "local root for work/temp/cache; final output, resume state, logs, "
            "and lock remain under --output-root"
        ),
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--partition", action="append", choices=PARTITION_NAMES)
    parser.add_argument("--max-episodes", type=_positive_int)
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument("--max-inflight-units", type=_positive_int)
    parser.add_argument("--max-staging-bytes", type=_positive_int)
    parser.add_argument(
        "--max-inflight-bytes", type=_positive_int, default=DEFAULT_MAX_INFLIGHT_BYTES
    )
    parser.add_argument(
        "--episodes-per-checkpoint-part",
        type=_positive_int,
        default=DEFAULT_EPISODES_PER_PART,
    )
    parser.add_argument("--encoder-threads-per-worker", type=_positive_int, default=5)
    parser.add_argument("--encoder-queue-maxsize", type=_positive_int, default=30)
    parser.add_argument("--eta-interval-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--estimate-storage", action="store_true")
    parser.add_argument("--max-local-bytes", type=_positive_int)
    parser.add_argument("--min-free-bytes", type=_nonnegative_int)
    parser.add_argument(
        "--storage-check-interval-seconds",
        "--disk-check-interval-seconds",
        dest="storage_check_interval_seconds",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-codec", default="h264")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * MIB):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_success_marker(
    final_output: Path, *, allow_stale_incomplete: bool = False
) -> dict[str, Any]:
    marker = read_json_object(final_output / SUCCESS_MARKER, "success marker")
    manifest = final_output / "collection_manifest.json"
    if marker.get("status") != "success" or not manifest.is_file():
        raise ConversionError(f"invalid success publication at {final_output}")
    if marker.get("collection_manifest_sha256") != _sha256_file(manifest):
        raise ConversionError(f"success marker manifest checksum changed at {final_output}")
    if (final_output / INCOMPLETE_MARKER).exists() and not allow_stale_incomplete:
        raise ConversionError(f"success output still contains {INCOMPLETE_MARKER}: {final_output}")
    return marker


def _recover_stale_success(paths: RuntimePaths) -> None:
    """Finish marker cleanup after a crash immediately following `_SUCCESS`."""

    incomplete_path = paths.final_output / INCOMPLETE_MARKER
    with exclusive_resume_lock(paths.lock_path):
        _validate_success_marker(paths.final_output, allow_stale_incomplete=True)
        incomplete_path.unlink()
        _validate_success_marker(paths.final_output)
        shutil.rmtree(paths.resume_dir, ignore_errors=True)
        shutil.rmtree(paths.work_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (args.resume, args.skip_existing, args.overwrite)) > 1:
        parser.error("--resume, --skip-existing, and --overwrite are mutually exclusive")
    if args.overwrite:
        parser.error(
            "--overwrite is retained for CLI compatibility but unsupported by the in-place "
            "OSSFS publication protocol; use a new test UID or explicitly archive the old output"
        )
    if args.video_codec != "h264":
        parser.error("only CPU h264 is approved; NVENC and unbenchmarked codecs are rejected")
    if args.max_staging_bytes is not None and args.max_local_bytes is not None:
        if args.max_staging_bytes != args.max_local_bytes:
            parser.error("--max-local-bytes and --max-staging-bytes disagree")
    args.max_staging_bytes = (
        args.max_staging_bytes
        or args.max_local_bytes
        or DEFAULT_MAX_STAGING_BYTES
    )
    if args.min_free_bytes not in (None, 0):
        parser.error(
            "--min-free-bytes is not meaningful on OSSFS; use --max-staging-bytes"
        )
    if args.max_inflight_units is None:
        args.max_inflight_units = min(4, args.workers)
    if args.max_inflight_units > args.workers:
        parser.error("--max-inflight-units cannot exceed --workers")
    if args.workers * args.encoder_threads_per_worker > MAX_TOTAL_ENCODER_THREADS:
        parser.error(
            f"worker encoder budgets exceed the {MAX_TOTAL_ENCODER_THREADS}-thread limit"
        )
    bounded = args.max_episodes is not None or bool(args.partition)
    if not bounded and args.workers <= 1:
        parser.error("formal full conversion requires at least two deterministic partition workers")
    if not bounded and (
        args.workers != 4
        or args.max_inflight_units != 4
        or args.encoder_threads_per_worker != 5
    ):
        parser.error(
            "formal full conversion requires the real-sample winner: 4 workers, "
            "4 inflight partitions, and a 5-thread per-worker encoder budget"
        )
    if bounded and args.output is None and not args.estimate_storage:
        parser.error("bounded smoke conversion requires an independent --output test UID")
    if not args.raw_root.is_dir():
        parser.error(f"raw root does not exist: {args.raw_root}")

    paths: RuntimePaths | None = None
    log_path: Path | None = None
    try:
        paths = _runtime_paths(args)
        # Redirect caches and temporary-file discovery before source
        # inspection or any lazy dependency import.  The estimate-only path
        # does not create these directories.
        _configure_runtime_environment(paths, create=False)
        infos = _build_infos(
            args.raw_root,
            paths.final_output,
            partitions=args.partition,
            max_episodes=args.max_episodes,
            episodes_per_part=args.episodes_per_checkpoint_part,
        )
        existing_output_bytes, resume_state_bytes = _resume_accounting(paths.resume_dir)
        snapshot, estimate = _storage_estimate(
            infos,
            paths.final_output,
            workers=min(args.workers, args.max_inflight_units),
            existing_output_bytes=existing_output_bytes,
        )
        # The final output is on OSSFS.  Account its committed bytes from the
        # checkpoint inventory rather than recursively stat-ing the remote tree.
        current_scoped_usage = (
            existing_output_bytes
            + resume_state_bytes
            + directory_size(paths.work_dir)
            + (paths.lock_path.stat().st_size if paths.lock_path.exists() else 0)
        )
        remaining_staging = _remaining_staging_reservation(
            estimate,
            paths,
            existing_output_bytes=existing_output_bytes,
            resume_state_bytes=resume_state_bytes,
        )
        report = {
            "filesystem_informational_only": snapshot.as_dict(),
            "paths": {key: str(value) for key, value in paths.__dict__.items()},
            "selection": {
                "partitions": [info.partition_name for info in infos],
                "episodes": sum(len(info.plan.episodes) for info in infos),
                "frames": sum(info.plan.num_frames for info in infos),
                "source_hdf5_bytes": sum(info.source_path.stat().st_size for info in infos),
                "numeric_logical_bytes": sum(info.selected_numeric_logical_bytes for info in infos),
                "image_logical_bytes": sum(info.selected_image_logical_bytes for info in infos),
                "camera_frames": sum(info.selected_camera_frames for info in infos),
                # The raw release is on OSSFS.  Do not recursively walk the
                # remote root here: that scan can block the coordinator in
                # uninterruptible I/O before any worker starts.  The fixed
                # partition list has already been inspected by _build_infos,
                # so report the selected source files directly.
                "source_root_files": len(infos),
                "source_root_logical_bytes": sum(
                    info.source_path.stat().st_size for info in infos
                ),
                "partition_distribution": [
                    {
                        "partition": info.partition_name,
                        "all_episodes": info.all_episode_count,
                        "all_frames": info.all_frame_count,
                        "source_hdf5_bytes": info.source_path.stat().st_size,
                        "episode_length": info.episode_length_summary,
                    }
                    for info in infos
                ],
            },
            "estimate": estimate.as_dict(),
            "quota": {
                "current_scoped_usage_bytes": current_scoped_usage,
                "max_staging_bytes": args.max_staging_bytes,
                "max_inflight_bytes": args.max_inflight_bytes,
                "max_inflight_units": args.max_inflight_units,
                "capacity_sufficient": (
                    current_scoped_usage + remaining_staging
                    <= args.max_staging_bytes
                ),
            },
            "estimate_evidence": {
                "metadata_schema_reference_coverage": "all episodes in selected HDF5 files",
                "video_compression_sample": "Transport demo_0, 375 frames, 5 cameras",
                "video_error_range": "0.5x to 2x measured H.264 ratio",
                "full_payload_scan": False,
            },
        }
        print(json.dumps(report, indent=2))
        print(json.dumps({"plans": [plan_summary(info.plan) for info in infos]}, indent=2))
        if args.estimate_storage:
            return 0

        success_path = paths.final_output / SUCCESS_MARKER
        incomplete_path = paths.final_output / INCOMPLETE_MARKER
        if success_path.exists():
            _validate_success_marker(
                paths.final_output,
                allow_stale_incomplete=incomplete_path.exists(),
            )
            if incomplete_path.exists():
                # A crash after the durable success write but before marker cleanup
                # leaves a fully published collection.  Complete that idempotent
                # cleanup only while holding the collection-wide lock.
                _recover_stale_success(paths)
            if args.skip_existing:
                print(f"skipped valid successful collection: {paths.final_output}")
                return 0
            raise FileExistsError(
                f"valid {SUCCESS_MARKER} already exists; refusing overwrite: {paths.final_output}"
            )
        if paths.final_output.exists() and not incomplete_path.is_file():
            raise ConversionError(
                f"existing output has neither valid {SUCCESS_MARKER} nor {INCOMPLETE_MARKER}: "
                f"{paths.final_output}"
            )
        if incomplete_path.exists() and not args.resume:
            raise ConversionError(
                f"incomplete output exists; rerun the identical command with --resume: "
                f"{paths.final_output}"
            )
        if args.resume and not incomplete_path.is_file():
            raise ConversionError(
                f"--resume requires {incomplete_path}; no incomplete conversion was found"
            )

        if current_scoped_usage + remaining_staging > args.max_staging_bytes:
            raise ConversionError(
                "estimated remaining staging peak exceeds --max-staging-bytes: "
                f"{current_scoped_usage} + {remaining_staging} > "
                f"{args.max_staging_bytes}"
            )

        runtime_environment = _configure_runtime_environment(paths, create=True)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = paths.logs_dir / f"run-{paths.final_output.name}.jsonl"
        _append_log(log_path, "start", report=report)
        payload = _collection_payload(infos, args)
        units = _work_units(
            infos,
            args,
            paths=paths,
            final_output_upper_bytes=estimate.final_output_conservative_upper_bytes,
        )
        active_workers = min(args.workers, args.max_inflight_units, len(units))
        if active_workers < args.workers:
            print(
                f"requested {args.workers} workers but selection has {len(units)} "
                f"isolated partitions; running {active_workers} worker(s)",
                file=sys.stderr,
                flush=True,
            )
        inflight = inflight_estimate(units, active_workers)
        if inflight.temp_bytes > args.max_inflight_bytes:
            raise ConversionError(
                f"planned inflight temporary bytes {inflight.temp_bytes} exceed "
                f"--max-inflight-bytes={args.max_inflight_bytes}"
            )

        with exclusive_resume_lock(paths.lock_path):
                if not paths.final_output.exists():
                    paths.final_output.mkdir(parents=True, exist_ok=False)
                    atomic_write_json(
                        incomplete_path,
                        {
                            "status": "incomplete",
                            "fingerprint": canonical_fingerprint(payload),
                            "started_unix": time.time(),
                        },
                    )
                _prepare_collection_resume(
                    paths.final_output,
                    paths.resume_dir,
                    payload,
                    require_existing=args.resume,
                )
                guard = StagingQuotaGuard(
                    staging_roots=(paths.work_dir,),
                    inflight_roots=(paths.work_dir,),
                    max_staging_bytes=args.max_staging_bytes,
                    max_inflight_bytes=args.max_inflight_bytes,
                    max_inflight_units=args.max_inflight_units,
                    interval_seconds=args.storage_check_interval_seconds,
                )
                check = guard.check(
                    "before partition dispatch",
                    required_staging_bytes=remaining_staging,
                    required_inflight_bytes=inflight.temp_bytes,
                    inflight_units=inflight.workers,
                )
                print(format_staging_quota_check(check), file=sys.stderr, flush=True)

                pending = []
                for unit in units:
                    target = Path(unit.target_path)
                    if (target / "conversion_manifest.json").is_file():
                        try:
                            _validate_partition(unit)
                        except (ConversionError, OSError, RuntimeError, ValueError):
                            pending.append(unit)
                        else:
                            print(f"[{unit.key}] reused verified partition", file=sys.stderr)
                    else:
                        pending.append(unit)

                def partition_completed(result: Any) -> None:
                    result_check = guard.check(
                        f"partition {result.key} completed",
                        inflight_units=0,
                    )
                    print(
                        format_staging_quota_check(result_check),
                        file=sys.stderr,
                        flush=True,
                    )

                def before_partition_dispatch(
                    unit: ParallelWorkUnit,
                    active: tuple[ParallelWorkUnit, ...],
                ) -> None:
                    payload = unit.payload
                    if not isinstance(payload, _WorkerPayload):
                        raise ConversionError(
                            f"invalid DexMimicGen worker payload for {unit.key}"
                        )
                    dispatch_check = guard.check(
                        f"before partition {unit.key} dispatch",
                        required_staging_bytes=payload.partition_output_upper_bytes,
                        required_inflight_bytes=unit.estimated_temp_bytes,
                        inflight_units=len(active) + 1,
                    )
                    print(
                        format_staging_quota_check(dispatch_check),
                        file=sys.stderr,
                        flush=True,
                    )

                def scheduler_health_check() -> None:
                    health_check = guard.check(
                        "parallel scheduler periodic check",
                        inflight_units=active_workers,
                    )
                    print(
                        format_staging_quota_check(health_check),
                        file=sys.stderr,
                        flush=True,
                    )

                if pending:
                    run_parallel_work_units(
                        pending,
                        _convert_partition,
                        workers=min(active_workers, len(pending)),
                        on_result=partition_completed,
                        health_check=scheduler_health_check,
                        health_check_interval_seconds=(
                            args.storage_check_interval_seconds
                        ),
                        before_dispatch=before_partition_dispatch,
                    )
                for unit in units:
                    _validate_partition(unit)
                sidecars = _write_model_sidecars(infos, paths.final_output)
                manifest_path = paths.final_output / "collection_manifest.json"
                atomic_write_json(
                    manifest_path,
                    _collection_manifest(
                        infos,
                        sidecars,
                        args,
                        paths,
                        runtime_environment,
                    ),
                )
                final_check = guard.check("before success publication", inflight_units=0)
                print(format_staging_quota_check(final_check), file=sys.stderr, flush=True)
                atomic_write_json(
                    success_path,
                    {
                        "status": "success",
                        "fingerprint": canonical_fingerprint(payload),
                        "collection_manifest_sha256": _sha256_file(manifest_path),
                        "partitions": len(infos),
                        "episodes": sum(len(info.plan.episodes) for info in infos),
                        "frames": sum(info.plan.num_frames for info in infos),
                        "completed_unix": time.time(),
                    },
                )
                incomplete_path.unlink()
                _validate_success_marker(paths.final_output)
                shutil.rmtree(paths.resume_dir, ignore_errors=True)
                shutil.rmtree(paths.work_dir, ignore_errors=True)
                _append_log(log_path, "success", output=str(paths.final_output))
        print(f"validated successful collection: {paths.final_output}")
        return 0
    except KeyboardInterrupt:
        if log_path is not None:
            _append_log(log_path, "interrupted")
        print("interrupted; verified parts were retained; rerun with --resume", file=sys.stderr)
        return 130
    except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        if log_path is not None:
            _append_log(log_path, "failed", error=f"{type(exc).__name__}: {exc}")
        print(f"error: {exc}", file=sys.stderr)
        return 1


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> bool:
        return False


if __name__ == "__main__":
    previous = signal.getsignal(signal.SIGTERM)

    def _sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)
    try:
        sys.exit(main())
    finally:
        signal.signal(signal.SIGTERM, previous)
