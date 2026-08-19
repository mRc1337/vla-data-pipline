#!/usr/bin/env python3
"""Reliably convert the official MimicGen release to LeRobot v3.0.

The release is heterogeneous, so every source HDF5 container becomes one
fixed-schema LeRobot partition below a collection root.  Full conversion is
never implicit: this CLI only writes when neither --inspect-only nor
--dry-run is supplied by the caller.
"""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
import uuid

# The converter's own modules must not create ``__pycache__`` under the source
# checkout in /home.  Formal commands also set PYTHONDONTWRITEBYTECODE before
# interpreter startup; this protects direct CLI invocation as early as Python
# permits inside the script.
sys.dont_write_bytecode = True

from convert_core.checkpoint import (
    RESUME_SCHEMA_VERSION,
    atomic_write_json,
    canonical_fingerprint,
    exclusive_resume_lock,
    read_json_object,
    resume_paths,
)
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import (
    convert_dataset,
    plan_summary,
    publish_temporary_output,
    validate_written_dataset,
)
from convert_core.parallel import (
    ParallelWorkResult,
    ParallelWorkUnit,
    run_parallel_work_units,
    validate_inflight_budget,
)
from convert_core.performance import ProcessTreeSampler
from convert_core.progress import EtaProgress
from convert_core.runtime_layout import (
    ConversionRuntimeLayout,
    StagingCapacityGuard,
    StructuredRunLog,
    build_runtime_layout,
    redirected_runtime_environment,
    validate_streaming_runtime_filesystems,
)
from convert_core.storage import directory_size
from readers.robomimic_hdf5_reader import RobomimicPartitionInfo, inspect_partition, iter_frames


SOURCE_REPO = "amandlek/mimicgen_datasets"
SOURCE_REVISION = "33016f8a62c02334f929f2913af8fdd2a8a129e1"
OFFICIAL_CODE_REVISION = "72bd767c255545f462e7ccfb2731f2e5d4c1d9bb"
DEFAULT_RAW_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw")
APPROVED_OUTPUT_ROOT = Path(
    "/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0"
)
# ``--staging-root`` remains as a legacy alias for the parent of
# ``--output-root``.  New commands should pass --output-root explicitly.
DEFAULT_STAGING_ROOT = APPROVED_OUTPUT_ROOT.parent
DEFAULT_SOURCE_DIRECTORY = "minicgen"
DEFAULT_DATASET_UID = "mimicgen"
DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS = 300.0
CHECKPOINT_VALIDATION_VERSION = 1
MAX_TOTAL_ENCODER_THREADS = 32
# LeRobot, PyArrow, HDF5, and encoder state measured 1.3--1.9 GiB per
# concurrently active worker on the bounded real benchmark.  Use 2 GiB before
# adding queued-frame storage so the pre-dispatch budget remains conservative.
WORKER_BASE_MEMORY_BYTES = 2 * 1024**3
WORKER_BASE_TEMP_BYTES = 512 * 1024**2
DEFAULT_RESOURCE_BUDGET_FRACTION = 0.8
MAX_BENCHMARK_PARTITIONS = 8
MAX_BENCHMARK_EPISODES_PER_PARTITION = 10
DEFAULT_ENCODER_QUEUE_MAXSIZE = 30
DEFAULT_STORAGE_CHECK_INTERVAL_SECONDS = 10.0
# State, partition markers, structured logs, sentinels, and manifests are tiny
# relative to video data.  Reserve a fixed conservative allowance so capacity
# checks never recursively scan OSSFS metadata directories at dispatch time.
RUNTIME_METADATA_RESERVE_BYTES = 256 * 1024**2
FORBIDDEN_NVENC_CODECS = frozenset({"h264_nvenc", "hevc_nvenc", "av1_nvenc"})


class ResumeValidationUnavailable(ConversionError):
    """A checkpoint could not be validated without risking an indefinite wait."""


@dataclass(frozen=True)
class PartitionWorkerPayload:
    """Everything a spawned writer needs; all indices are coordinator-owned."""

    info: RobomimicPartitionInfo
    plan_index: int
    episode_indices: tuple[int, ...]
    task_indices: tuple[int, ...]
    video_codec: str
    video_quality: int
    video_preset: str
    encoder_threads: int
    probe_timeout_seconds: float
    cache_path: str
    temp_path: str
    marker_path: str | None
    fingerprint: str
    max_unit_bytes: int | None
    storage_check_interval_seconds: float


@dataclass(frozen=True)
class PartitionWorkerOutput:
    partition_name: str
    plan_index: int
    episodes: int
    frames: int
    elapsed_seconds: float
    validation: dict[str, Any]


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


def _source_root(raw_root: Path, source_directory: str) -> Path:
    requested = raw_root / source_directory
    if requested.is_dir():
        return requested
    if source_directory == "mimicgen" and (raw_root / DEFAULT_SOURCE_DIRECTORY).is_dir():
        discovered = raw_root / DEFAULT_SOURCE_DIRECTORY
        print(
            f"warning: requested source directory {requested} is absent; using discovered official "
            f"release directory {discovered}",
            file=sys.stderr,
            flush=True,
        )
        return discovered
    raise ConversionError(f"source directory does not exist: {requested}")


def _select_source_files(
    source_root: Path,
    *,
    categories: set[str],
    partitions: set[str],
    max_partitions: int | None,
) -> list[Path]:
    paths = sorted(source_root.glob("*/*.hdf5"), key=lambda path: path.relative_to(source_root).as_posix())
    if categories:
        paths = [path for path in paths if path.parent.name in categories]
    if partitions:
        normalized = {value.removesuffix(".hdf5") for value in partitions}
        paths = [
            path
            for path in paths
            if path.relative_to(source_root).with_suffix("").as_posix() in normalized
            or path.stem in normalized
        ]
    if max_partitions is not None:
        paths = paths[:max_partitions]
    if not paths:
        raise ConversionError("no MimicGen HDF5 partitions matched the requested selection")
    return paths


def inspect_collection(
    source_root: Path,
    collection_output: Path,
    source_files: list[Path],
    *,
    max_episodes: int | None,
    eta_interval_seconds: float,
    workers: int = 1,
) -> list[RobomimicPartitionInfo]:
    progress = EtaProgress(
        "mimicgen preflight", len(source_files), "partitions", interval_seconds=eta_interval_seconds
    )
    if workers <= 0:
        raise ValueError("inspect workers must be positive")
    if workers == 1:
        infos = [
            inspect_partition(
                path,
                raw_dataset_root=source_root,
                collection_output=collection_output,
                max_episodes=max_episodes,
            )
            for path in source_files
        ]
        for index, info in enumerate(infos):
            progress.update(index + 1, context=info.source_relative_path, force=True)
    else:
        indexed: dict[int, RobomimicPartitionInfo] = {}
        with ProcessPoolExecutor(
            max_workers=min(workers, len(source_files)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(
                    inspect_partition,
                    path,
                    raw_dataset_root=source_root,
                    collection_output=collection_output,
                    max_episodes=max_episodes,
                ): index
                for index, path in enumerate(source_files)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                info = future.result()
                indexed[index] = info
                progress.update(completed, context=info.source_relative_path, force=True)
        infos = [indexed[index] for index in range(len(source_files))]
    for info in infos:
        info.plan.extra["source_revision"] = SOURCE_REVISION
        if info.dangling_split_references:
            counts = ", ".join(
                f"{name}={len(values)}" for name, values in info.dangling_split_references.items()
            )
            print(
                f"warning: {info.source_relative_path} has dangling mask references ({counts}); "
                "raw lists are retained in manifests and nonexistent episodes are not fabricated",
                file=sys.stderr,
                flush=True,
            )
    return infos


def _collection_summary(infos: list[RobomimicPartitionInfo], output: Path) -> dict[str, Any]:
    return {
        "dataset_uid": output.name,
        "output": str(output),
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "partitions": len(infos),
        "selected_episodes": sum(len(info.plan.episodes) for info in infos),
        "selected_frames": sum(info.plan.num_frames for info in infos),
        "source_episodes": sum(info.all_episode_count for info in infos),
        "source_frames": sum(info.all_frame_count for info in infos),
        "dangling_split_references": {
            info.source_relative_path: {
                split: list(values) for split, values in info.dangling_split_references.items()
            }
            for info in infos
            if info.dangling_split_references
        },
        "plans": [plan_summary(info.plan) for info in infos],
    }


def _video_encoding_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "storage": "video",
        "source_storage": "uncompressed uint8 RGB arrays in HDF5",
        "remux": False,
        "reencoded": True,
        "lossy": args.video_quality is not None,
        "codec": args.video_codec,
        "crf": args.video_quality,
        "preset": args.video_preset,
        "pix_fmt": "yuv420p",
        "streaming": True,
        "encoder_threads": args.encoder_threads,
    }


def _rgb_encoder_values(codec: str, quality: int, preset: str) -> Any:
    if codec in FORBIDDEN_NVENC_CODECS:
        raise ConversionError(
            f"{codec} is disabled: this host has no usable NVENC; use a CPU codec such as h264"
        )
    try:
        from lerobot.configs.video import RGBEncoderConfig
    except ImportError as exc:
        raise RuntimeError("lerobot==0.6.0 is required") from exc
    return RGBEncoderConfig(
        vcodec=codec,
        crf=quality,
        preset=preset,
        pix_fmt="yuv420p",
    )


def _rgb_encoder(args: argparse.Namespace) -> Any:
    return _rgb_encoder_values(args.video_codec, args.video_quality, args.video_preset)


def _encoder_preflight(info: RobomimicPartitionInfo, args: argparse.Namespace) -> None:
    """Actually encode and reopen two real source frames on local storage."""

    episode = info.plan.episodes[0]
    short_episode = replace(episode, num_frames=min(2, episode.num_frames))
    configured_work = getattr(args, "work_dir", None)
    preflight_parent = (
        Path(configured_work) / "preflight"
        if configured_work is not None
        else args.staging_root / ".encoder-preflight"
    )
    preflight_parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="mimicgen-", dir=preflight_parent) as directory:
            output = Path(directory) / "dataset"
            plan = replace(
                info.plan,
                dataset_uid="mimicgen_encoder_preflight",
                output_path=output,
                episodes=(short_episode,),
            )
            queue_size = short_episode.num_frames + 1
            convert_dataset(
                plan,
                lambda selected: iter_frames(plan, selected),
                reader_format="robomimic_hdf5",
                rgb_encoder=_rgb_encoder(args),
                streaming_encoding=True,
                blocking_streaming_encoding=True,
                encoder_queue_maxsize=queue_size,
                encoder_threads=args.encoder_threads,
                batch_metadata_writes=True,
                encoder_temp_root=Path(directory) / "encoder-temp",
                fragmented_mp4_writes=True,
            )
            _validate_video_streams(plan, output, args.video_codec)
    finally:
        with contextlib.suppress(OSError):
            preflight_parent.rmdir()
    print("video encoder preflight passed on real MimicGen frames", file=sys.stderr, flush=True)


def _ffprobe(
    path: Path, *, timeout_seconds: float = DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,"
            "nb_read_frames,nb_frames,pix_fmt"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ResumeValidationUnavailable(
            f"ffprobe timed out after {timeout_seconds:g}s for {path}; checkpoint was preserved"
        ) from exc
    except OSError as exc:
        raise ResumeValidationUnavailable(
            f"could not start ffprobe for {path}: {exc}; checkpoint was preserved"
        ) from exc
    if result.returncode != 0:
        raise ConversionError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ConversionError(f"{path}: expected exactly one video stream")
    return streams[0]


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def _validate_video_streams(
    plan: Any,
    root: Path,
    requested_codec: str,
    *,
    probe_timeout_seconds: float = DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    expected_codec = {"h264": "h264", "libsvtav1": "av1", "libaom-av1": "av1"}.get(
        requested_codec, requested_codec.replace("_nvenc", "")
    )
    validated_streams: list[dict[str, Any]] = []
    for camera in plan.camera_features:
        camera_root = root / "videos" / camera.feature_key
        paths = sorted(camera_root.rglob("*.mp4")) if camera_root.is_dir() else []
        if not paths:
            raise ConversionError(f"written dataset has no videos for {camera.feature_key}")
        frame_count = 0
        for index, path in enumerate(paths, start=1):
            relative_path = path.relative_to(root).as_posix()
            print(
                f"[video validation] {plan.dataset_uid} {camera.feature_key} "
                f"{index}/{len(paths)}: {relative_path} "
                f"(timeout {probe_timeout_seconds:g}s)",
                file=sys.stderr,
                flush=True,
            )
            stream = _ffprobe(path, timeout_seconds=probe_timeout_seconds)
            frames = stream.get("nb_read_frames") or stream.get("nb_frames")
            if frames in {None, "N/A"}:
                raise ConversionError(f"{path}: ffprobe did not report a frame count")
            frame_count += int(frames)
            if (int(stream["height"]), int(stream["width"])) != (camera.height, camera.width):
                raise ConversionError(f"{path}: unexpected resolution")
            measured_rate = stream.get("avg_frame_rate") or stream["r_frame_rate"]
            if not math.isclose(_fraction(measured_rate), plan.fps, abs_tol=1e-6):
                raise ConversionError(f"{path}: unexpected average FPS {measured_rate}")
            if stream.get("codec_name") != expected_codec:
                raise ConversionError(
                    f"{path}: codec is {stream.get('codec_name')!r}, expected {expected_codec!r}"
                )
            validated_streams.append(
                {
                    "path": relative_path,
                    "frames": int(frames),
                    "codec": stream.get("codec_name"),
                    "width": int(stream["width"]),
                    "height": int(stream["height"]),
                    "r_frame_rate": stream["r_frame_rate"],
                    "avg_frame_rate": stream.get("avg_frame_rate"),
                }
            )
        if frame_count != plan.num_frames:
            raise ConversionError(
                f"{camera.feature_key}: videos contain {frame_count} frames, expected {plan.num_frames}"
            )
    return validated_streams


def _checkpoint_file_inventory(root: Path) -> list[dict[str, Any]]:
    """Return a cheap identity for an already fully validated immutable partition."""
    try:
        entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
        inventory: list[dict[str, Any]] = []
        for path in entries:
            if path.is_symlink():
                raise ConversionError(f"checkpoint contains a symbolic link: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ConversionError(f"checkpoint contains a non-regular file: {path}")
            stat = path.stat()
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    except OSError as exc:
        raise ResumeValidationUnavailable(
            f"could not inventory checkpoint {root}: {exc}; checkpoint was preserved"
        ) from exc
    if not inventory:
        raise ConversionError(f"checkpoint contains no files: {root}")
    return inventory


def _checkpoint_validation(
    root: Path, video_streams: list[dict[str, Any]]
) -> dict[str, Any]:
    files = _checkpoint_file_inventory(root)
    return {
        "version": CHECKPOINT_VALIDATION_VERSION,
        "file_fingerprint": canonical_fingerprint({"files": files}),
        "files": files,
        "video_streams": video_streams,
    }


def _validate_checkpoint_fingerprint(root: Path, validation: Any) -> None:
    if not isinstance(validation, dict) or validation.get("version") != CHECKPOINT_VALIDATION_VERSION:
        raise ConversionError("checkpoint has no supported persisted validation")
    expected_files = validation.get("files")
    expected_fingerprint = validation.get("file_fingerprint")
    if not isinstance(expected_files, list) or not isinstance(expected_fingerprint, str):
        raise ConversionError("checkpoint persisted validation is incomplete")
    actual_files = _checkpoint_file_inventory(root)
    actual_fingerprint = canonical_fingerprint({"files": actual_files})
    if actual_fingerprint != expected_fingerprint or actual_files != expected_files:
        raise ConversionError("checkpoint file fingerprint changed")


def _planned_task_indices(plan: Any) -> tuple[int, ...]:
    task_indices: dict[str, int] = {}
    result: list[int] = []
    for episode in plan.episodes:
        result.append(task_indices.setdefault(episode.instruction, len(task_indices)))
    return tuple(result)


def _partition_resource_estimate(info: RobomimicPartitionInfo) -> tuple[int, int]:
    """Bound the full streaming queues plus one partition's temporary output."""

    queue_frames = max(episode.num_frames for episode in info.plan.episodes) + 1
    image_bytes_per_frame = sum(
        camera.height * camera.width * 3 for camera in info.plan.camera_features
    )
    # StreamingVideoEncoder copies each enqueued image.  Count both the HDF5
    # frame and queue copy, plus a fixed Python/Arrow/encoder allowance.
    memory_bytes = WORKER_BASE_MEMORY_BYTES + 2 * queue_frames * image_bytes_per_frame
    # Scale the source container by selected frames for bounded smoke runs.  A
    # full conversion still uses the complete source size as its conservative
    # encoded-output/temp ceiling.
    source_bytes = info.source_path.stat().st_size
    selected_source_bytes = math.ceil(
        source_bytes * info.plan.num_frames / max(1, info.all_frame_count)
    )
    temp_bytes = WORKER_BASE_TEMP_BYTES + min(source_bytes, selected_source_bytes)
    return memory_bytes, temp_bytes


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    raise ConversionError("could not determine MemAvailable for the worker memory budget")


def _resource_budgets(args: argparse.Namespace, temp_root: Path) -> tuple[int, int]:
    configured_memory = getattr(args, "memory_budget_gib", None)
    configured_temp = getattr(args, "temp_budget_gib", None)
    memory = (
        int(configured_memory * 1024**3)
        if configured_memory is not None
        else int(_available_memory_bytes() * DEFAULT_RESOURCE_BUDGET_FRACTION)
    )
    temp = (
        int(configured_temp * 1024**3)
        if configured_temp is not None
        else int(shutil.disk_usage(temp_root).free * DEFAULT_RESOURCE_BUDGET_FRACTION)
    )
    return memory, temp


def _build_parallel_work_units(
    infos: list[RobomimicPartitionInfo],
    pending: list[RobomimicPartitionInfo],
    temporary: Path,
    cache_root: Path,
    resume_state: Path,
    fingerprint: str,
    args: argparse.Namespace,
) -> list[ParallelWorkUnit]:
    plan_indices = {info.partition_name: index for index, info in enumerate(infos)}
    units: list[ParallelWorkUnit] = []
    episode_cursor = 0
    frame_cursor = 0
    for unit_index, info in enumerate(pending):
        memory_bytes, temp_bytes = _partition_resource_estimate(info)
        plan_index = plan_indices[info.partition_name]
        marker = (
            resume_state / "partitions" / f"{info.partition_name}.json"
            if args.resume
            else None
        )
        payload = PartitionWorkerPayload(
            info=info,
            plan_index=plan_index,
            episode_indices=tuple(range(len(info.plan.episodes))),
            task_indices=_planned_task_indices(info.plan),
            video_codec=args.video_codec,
            video_quality=args.video_quality,
            video_preset=args.video_preset,
            encoder_threads=args.encoder_threads,
            probe_timeout_seconds=getattr(
                args, "resume_probe_timeout", DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS
            ),
            cache_path=str(cache_root / info.partition_name),
            temp_path=str(cache_root.parent / "temp" / info.partition_name),
            marker_path=str(marker) if marker is not None else None,
            fingerprint=fingerprint,
            max_unit_bytes=temp_bytes,
            storage_check_interval_seconds=getattr(
                args,
                "storage_check_interval_seconds",
                DEFAULT_STORAGE_CHECK_INTERVAL_SECONDS,
            ),
        )
        units.append(
            ParallelWorkUnit(
                index=unit_index,
                key=info.partition_name,
                dataset_uid=info.plan.dataset_uid,
                target_path=str(temporary / info.partition_name),
                episode_start=episode_cursor,
                episode_end=episode_cursor + len(info.plan.episodes),
                frame_start=frame_cursor,
                frame_end=frame_cursor + info.plan.num_frames,
                task_indices=payload.task_indices,
                weight=info.plan.num_frames,
                estimated_memory_bytes=memory_bytes,
                estimated_temp_bytes=temp_bytes,
                fingerprint=fingerprint,
                payload=payload,
            )
        )
        episode_cursor += len(info.plan.episodes)
        frame_cursor += info.plan.num_frames
    return units


def _convert_partition_worker(unit: ParallelWorkUnit) -> PartitionWorkerOutput:
    payload = unit.payload
    if not isinstance(payload, PartitionWorkerPayload):
        raise ConversionError(f"invalid payload for partition {unit.key}")
    info = payload.info
    expected_episode_indices = tuple(range(len(info.plan.episodes)))
    if payload.episode_indices != expected_episode_indices:
        raise ConversionError(f"coordinator episode assignment changed for {unit.key}")
    if payload.task_indices != _planned_task_indices(info.plan):
        raise ConversionError(f"coordinator task assignment changed for {unit.key}")

    encoding_args = argparse.Namespace(
        video_codec=payload.video_codec,
        video_quality=payload.video_quality,
        video_preset=payload.video_preset,
        encoder_threads=payload.encoder_threads,
    )
    plan = replace(
        info.plan,
        output_path=Path(unit.target_path),
        extra={**info.plan.extra, "video_encoding": _video_encoding_payload(encoding_args)},
    )
    queue_size = DEFAULT_ENCODER_QUEUE_MAXSIZE
    cache_path = Path(payload.cache_path)
    temp_path = Path(payload.temp_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)
    previous_cache = os.environ.get("VLA_DATASETS_CACHE_ROOT")
    previous_temp = {
        key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")
    }
    os.environ["VLA_DATASETS_CACHE_ROOT"] = str(cache_path)
    for key in previous_temp:
        os.environ[key] = str(temp_path)
    unit_guard = StagingCapacityGuard(
        temp_path.parent,
        max_staging_bytes=payload.max_unit_bytes,
        max_inflight_bytes=payload.max_unit_bytes,
        max_inflight_units=1,
        interval_seconds=payload.storage_check_interval_seconds,
    )
    unit_guard.check(
        f"partition {unit.key} start",
        current_staging_bytes=0,
        inflight_bytes=unit.estimated_temp_bytes,
        inflight_units=1,
    )
    started_at = time.monotonic()
    print(
        f"[worker] starting plan {payload.plan_index + 1}: {unit.key} "
        f"({len(plan.episodes)} episodes / {plan.num_frames} frames)",
        file=sys.stderr,
        flush=True,
    )
    completed_frames = 0

    def frame_completed(_episode: Any, _index: int) -> None:
        nonlocal completed_frames
        completed_frames += 1
        if unit_guard.periodic_check(
            f"partition {unit.key} frame {completed_frames}",
            current_staging_bytes=(
                directory_size(plan.output_path)
                + directory_size(cache_path)
                + directory_size(temp_path)
            ),
            inflight_bytes=0,
            inflight_units=1,
        ) is not None:
            print(
                f"[storage] {unit.key}: checked after {completed_frames} frames",
                file=sys.stderr,
                flush=True,
            )
    try:
        convert_dataset(
            plan,
            lambda episode: iter_frames(plan, episode),
            reader_format="robomimic_hdf5",
            rgb_encoder=_rgb_encoder_values(
                payload.video_codec, payload.video_quality, payload.video_preset
            ),
            streaming_encoding=True,
            blocking_streaming_encoding=True,
            encoder_queue_maxsize=queue_size,
            encoder_threads=payload.encoder_threads,
            batch_metadata_writes=True,
            encoder_temp_root=temp_path,
            fragmented_mp4_writes=True,
            frame_completed_hook=frame_completed,
        )
        validate_written_dataset(plan, plan.output_path)
        streams = _validate_video_streams(
            plan,
            plan.output_path,
            payload.video_codec,
            probe_timeout_seconds=payload.probe_timeout_seconds,
        )
        validation = _checkpoint_validation(plan.output_path, streams)
        if payload.marker_path is not None:
            atomic_write_json(
                Path(payload.marker_path),
                {
                    "resume_schema_version": RESUME_SCHEMA_VERSION,
                    "fingerprint": payload.fingerprint,
                    "partition": info.partition_name,
                    "episodes": len(plan.episodes),
                    "frames": plan.num_frames,
                    "validation": validation,
                },
            )
        elapsed = time.monotonic() - started_at
        print(
            f"[worker] completed plan {payload.plan_index + 1}: {unit.key} in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return PartitionWorkerOutput(
            partition_name=info.partition_name,
            plan_index=payload.plan_index,
            episodes=len(plan.episodes),
            frames=plan.num_frames,
            elapsed_seconds=elapsed,
            validation=validation,
        )
    finally:
        if previous_cache is None:
            os.environ.pop("VLA_DATASETS_CACHE_ROOT", None)
        else:
            os.environ["VLA_DATASETS_CACHE_ROOT"] = previous_cache
        for key, value in previous_temp.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(cache_path, ignore_errors=True)
        shutil.rmtree(temp_path, ignore_errors=True)


def _fingerprint_payload(
    infos: list[RobomimicPartitionInfo],
    source_root: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "source_root": str(source_root.resolve()),
        "output": str(output.resolve()),
        "selection": {
            "category": sorted(args.category),
            "partition": sorted(args.partition),
            "max_partitions": args.max_partitions,
            "max_episodes": args.max_episodes,
        },
        "video_encoding": _video_encoding_payload(args),
        "partition_rule": "one source HDF5 container per fixed-schema LeRobot dataset",
        "partitions": [
            {
                "name": info.partition_name,
                "source": info.source_relative_path,
                "source_stat": {
                    "size": info.source_path.stat().st_size,
                    "mtime_ns": info.source_path.stat().st_mtime_ns,
                },
                "schema": [[key, list(shape), dtype] for key, shape, dtype in info.source_schema],
                "features": info.plan.feature_schema(),
                "robot_type": info.plan.robot_type,
                "fps": info.plan.fps,
                "episodes": [
                    {
                        "uid": episode.episode_uid,
                        "frames": episode.num_frames,
                        "task": episode.instruction,
                        "splits": list(episode.extra.get("source_splits", ())),
                    }
                    for episode in info.plan.episodes
                ],
                "dangling_split_references": {
                    key: list(value) for key, value in info.dangling_split_references.items()
                },
            }
            for info in infos
        ],
    }


def _changed_resume_sections(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    return sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))


def _prepare_resume(
    infos: list[RobomimicPartitionInfo],
    data_root: Path,
    state_root: Path,
    state_payload: dict[str, Any],
    fingerprint: str,
    video_codec: str,
    *,
    validation_mode: str = "fast",
    probe_timeout_seconds: float = DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS,
    allowed_data_entries: set[str] | None = None,
) -> tuple[list[RobomimicPartitionInfo], int, int]:
    normalized_payload = json.loads(json.dumps(state_payload, ensure_ascii=False, allow_nan=False))
    expected_state = {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": normalized_payload,
    }
    state_path = state_root / "state.json"
    if state_root.exists():
        if not state_root.is_dir() or not state_path.is_file():
            raise ConversionError(f"resume state is incomplete: {state_root}")
        actual = read_json_object(state_path, "resume state")
        if actual != expected_state:
            changed = _changed_resume_sections(
                expected_state.get("fingerprint_payload", {}), actual.get("fingerprint_payload", {})
            )
            raise ConversionError(
                "resume checkpoint does not match this conversion; changed sections: "
                + ", ".join(changed or ["fingerprint/state schema"])
            )
    else:
        state_root.mkdir(parents=True)
        atomic_write_json(state_path, expected_state)
    data_root.mkdir(parents=True, exist_ok=True)
    markers_root = state_root / "partitions"
    markers_root.mkdir(exist_ok=True)
    # This cache is expendable and never a checkpoint. A killed coordinator can
    # leave it behind, so clear it before validating durable partition markers.
    shutil.rmtree(state_root / "worker-cache", ignore_errors=True)
    # Older single-worker runs allowed the writer's default datasets cache to
    # land inside the resume data root.  It is expendable rather than an
    # unknown checkpoint entry, so remove it before the strict inventory check.
    shutil.rmtree(data_root / ".lerobot-datasets-cache", ignore_errors=True)

    expected_names = {info.partition_name for info in infos}
    # A SIGKILL or machine loss can leave the generic writer's unfinalized
    # sibling. It has no completion marker and must never be mixed with a new
    # run; remove only names that resolve to a known partition prefix.
    for path in list(data_root.iterdir()):
        if any(path.name.startswith(f".{name}.incomplete-") for name in expected_names):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    allowed_names = expected_names | set(allowed_data_entries or ())
    unexpected = sorted(path.name for path in data_root.iterdir() if path.name not in allowed_names)
    if unexpected:
        raise ConversionError(f"resume data contains unexpected entries: {unexpected}")
    pending: list[RobomimicPartitionInfo] = []
    reused_frames = reused_parts = 0
    for info in infos:
        root = data_root / info.partition_name
        marker = markers_root / f"{info.partition_name}.json"
        expected_marker_base = {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "partition": info.partition_name,
            "episodes": len(info.plan.episodes),
            "frames": info.plan.num_frames,
        }
        valid = False
        if root.is_dir() and marker.is_file():
            try:
                actual_marker = read_json_object(marker, "partition marker")
                if all(actual_marker.get(key) == value for key, value in expected_marker_base.items()):
                    checkpoint_plan = replace(info.plan, output_path=root)
                    persisted_validation = actual_marker.get("validation")
                    if validation_mode == "fast" and persisted_validation is not None:
                        print(
                            f"[resume] checking file fingerprint for {info.partition_name}",
                            file=sys.stderr,
                            flush=True,
                        )
                        try:
                            _validate_checkpoint_fingerprint(root, persisted_validation)
                        except ConversionError as exc:
                            if isinstance(exc, ResumeValidationUnavailable):
                                raise
                            print(
                                f"[resume] file fingerprint check failed for {info.partition_name}; "
                                f"running full validation: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            persisted_validation = None
                    if validation_mode == "full" or persisted_validation is None:
                        print(
                            f"[resume] fully validating {info.partition_name}",
                            file=sys.stderr,
                            flush=True,
                        )
                        validate_written_dataset(checkpoint_plan, root)
                        streams = _validate_video_streams(
                            checkpoint_plan,
                            root,
                            video_codec,
                            probe_timeout_seconds=probe_timeout_seconds,
                        )
                        persisted_validation = _checkpoint_validation(root, streams)
                        atomic_write_json(
                            marker,
                            {**expected_marker_base, "validation": persisted_validation},
                        )
                    valid = True
            except ResumeValidationUnavailable:
                raise
            except Exception as exc:
                print(
                    f"[resume] invalid checkpoint {info.partition_name}; rebuilding: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if valid:
            reused_parts += 1
            reused_frames += info.plan.num_frames
            continue
        marker.unlink(missing_ok=True)
        if root.exists():
            if not root.is_dir():
                raise ConversionError(f"resume partition path is not a directory: {root}")
            shutil.rmtree(root)
        pending.append(info)
    print(
        f"[resume] reused {reused_parts}/{len(infos)} verified partitions / {reused_frames} frames",
        file=sys.stderr,
        flush=True,
    )
    return pending, reused_parts, reused_frames


def _manifest(
    infos: list[RobomimicPartitionInfo], output_uid: str, args: argparse.Namespace, root: Path
) -> dict[str, Any]:
    return {
        "format": "lerobot_v3_0_collection",
        "dataset_uid": output_uid,
        "source_dataset": "MimicGen CoRL 2023 official release",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "partition_reason": (
            "Source HDF5 containers differ by robot, state/object schema, optional fields, and joint/gripper "
            "shape; each container is a deterministic fixed-schema partition."
        ),
        "video_encoding": _video_encoding_payload(args),
        "task_index_note": "LeRobot task_index values are generated from the documented natural-language mapping.",
        "partitions": [
            {
                "path": info.partition_name,
                "source": info.source_relative_path,
                "source_env_name": info.plan.extra["source_env_name"],
                "robot_type": info.plan.robot_type,
                "fps": info.plan.fps,
                "episodes": len(info.plan.episodes),
                "frames": info.plan.num_frames,
                "video_features": len(info.plan.camera_features),
                "video_files": len(list((root / info.partition_name / "videos").rglob("*.mp4"))),
                "features": info.plan.feature_schema(),
                "dangling_split_references": {
                    key: list(value) for key, value in info.dangling_split_references.items()
                },
            }
            for info in infos
        ],
        "total_partitions": len(infos),
        "total_episodes": sum(len(info.plan.episodes) for info in infos),
        "total_frames": sum(info.plan.num_frames for info in infos),
        "total_video_features": sum(len(info.plan.camera_features) for info in infos),
        "total_video_files": sum(
            len(list((root / info.partition_name / "videos").rglob("*.mp4"))) for info in infos
        ),
    }


def convert_collection(
    infos: list[RobomimicPartitionInfo],
    output: Path,
    args: argparse.Namespace,
    source_root: Path,
) -> Path:
    workers = int(getattr(args, "workers", 1))
    if workers <= 0:
        raise ConversionError("workers must be positive")
    if workers * args.encoder_threads > MAX_TOTAL_ENCODER_THREADS:
        raise ConversionError(
            f"workers * encoder threads must not exceed {MAX_TOTAL_ENCODER_THREADS}: "
            f"{workers} * {args.encoder_threads}"
        )
    if output.exists() and args.skip_existing:
        print(f"skipped existing output: {output}")
        return output
    if output.exists() and (args.resume or not args.overwrite):
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    resume_data, resume_state, resume_lock = resume_paths(output)
    state_payload = _fingerprint_payload(infos, source_root, output, args)
    fingerprint = canonical_fingerprint(state_payload)
    resume_validation = getattr(args, "resume_validation", "fast")
    probe_timeout_seconds = getattr(
        args, "resume_probe_timeout", DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS
    )
    lock_context = exclusive_resume_lock(resume_lock) if args.resume else _null_context()
    completed_successfully = False
    with lock_context:
        if args.resume:
            temporary = resume_data
            pending, _reused_parts, reused_frames = _prepare_resume(
                infos,
                temporary,
                resume_state,
                state_payload,
                fingerprint,
                args.video_codec,
                validation_mode=resume_validation,
                probe_timeout_seconds=probe_timeout_seconds,
            )
        else:
            temporary = output.with_name(f".{output.name}.incomplete-{uuid.uuid4().hex}")
            temporary.mkdir()
            pending = list(infos)
            reused_frames = 0
        cache_root = (
            resume_state / "worker-cache"
            if args.resume
            else temporary.with_name(f"{temporary.name}.worker-cache")
        )
        progress = EtaProgress(
            f"{output.name} convert",
            sum(info.plan.num_frames for info in infos),
            "frames",
            interval_seconds=args.eta_interval_seconds,
            initial_completed=reused_frames,
        )
        progress.update(reused_frames, context=f"reused {len(infos) - len(pending)} checkpoints", force=True)
        partition_validations: dict[str, dict[str, Any]] = {}
        if args.resume:
            for info in infos:
                marker = resume_state / "partitions" / f"{info.partition_name}.json"
                if marker.is_file():
                    validation = read_json_object(marker, "partition marker").get("validation")
                    if isinstance(validation, dict):
                        partition_validations[info.partition_name] = validation
        try:
            if workers == 1:
                rgb_encoder = _rgb_encoder(args)
                for part_index, info in enumerate(pending):
                    plan = replace(
                        info.plan,
                        output_path=temporary / info.partition_name,
                        extra={**info.plan.extra, "video_encoding": _video_encoding_payload(args)},
                    )
                    queue_size = max(episode.num_frames for episode in plan.episodes) + 1

                    def frame_done(_episode: Any, _index: int) -> None:
                        progress.update(progress.completed + 1, context=info.partition_name)

                    def episode_done(_episode: Any, episode_index: int) -> None:
                        progress.update(
                            progress.completed,
                            context=(
                                f"{info.partition_name} episode {episode_index + 1}/"
                                f"{len(plan.episodes)} saved"
                            ),
                            force=True,
                        )

                    serial_cache = cache_root / info.partition_name
                    serial_cache.mkdir(parents=True, exist_ok=True)
                    previous_cache = os.environ.get("VLA_DATASETS_CACHE_ROOT")
                    os.environ["VLA_DATASETS_CACHE_ROOT"] = str(serial_cache)
                    try:
                        convert_dataset(
                            plan,
                            lambda episode: iter_frames(plan, episode),
                            reader_format="robomimic_hdf5",
                            rgb_encoder=rgb_encoder,
                            streaming_encoding=True,
                            encoder_queue_maxsize=queue_size,
                            encoder_threads=args.encoder_threads,
                            frame_completed_hook=frame_done,
                            episode_completed_hook=episode_done,
                        )
                    finally:
                        if previous_cache is None:
                            os.environ.pop("VLA_DATASETS_CACHE_ROOT", None)
                        else:
                            os.environ["VLA_DATASETS_CACHE_ROOT"] = previous_cache
                        shutil.rmtree(serial_cache, ignore_errors=True)
                    validate_written_dataset(plan, plan.output_path)
                    streams = _validate_video_streams(
                        plan,
                        plan.output_path,
                        args.video_codec,
                        probe_timeout_seconds=probe_timeout_seconds,
                    )
                    validation = _checkpoint_validation(plan.output_path, streams)
                    partition_validations[info.partition_name] = validation
                    if args.resume:
                        marker = resume_state / "partitions" / f"{info.partition_name}.json"
                        atomic_write_json(
                            marker,
                            {
                                "resume_schema_version": RESUME_SCHEMA_VERSION,
                                "fingerprint": fingerprint,
                                "partition": info.partition_name,
                                "episodes": len(plan.episodes),
                                "frames": plan.num_frames,
                                "validation": validation,
                            },
                        )
                    print(
                        f"completed partition {part_index + 1}/{len(pending)}: "
                        f"{info.partition_name}",
                        file=sys.stderr,
                        flush=True,
                    )
            elif pending:
                units = _build_parallel_work_units(
                    infos,
                    pending,
                    temporary,
                    cache_root,
                    resume_state,
                    fingerprint,
                    args,
                )
                memory_budget, temp_budget = _resource_budgets(args, temporary.parent)
                estimate = validate_inflight_budget(
                    units,
                    workers,
                    memory_budget_bytes=memory_budget,
                    temp_budget_bytes=temp_budget,
                )
                print(
                    "parallel inflight budget: "
                    f"workers={estimate.workers}, estimated memory="
                    f"{estimate.memory_bytes / 1024**3:.2f} GiB/"
                    f"{memory_budget / 1024**3:.2f} GiB, estimated temporary="
                    f"{estimate.temp_bytes / 1024**3:.2f} GiB/"
                    f"{temp_budget / 1024**3:.2f} GiB",
                    file=sys.stderr,
                    flush=True,
                )
                cache_root.mkdir(parents=True, exist_ok=True)

                def partition_done(result: ParallelWorkResult) -> None:
                    value = result.value
                    if not isinstance(value, PartitionWorkerOutput):
                        raise ConversionError(f"invalid worker result for {result.key}")
                    partition_validations[result.key] = value.validation
                    progress.update(
                        progress.completed + value.frames,
                        context=(
                            f"{result.key} verified ({value.episodes} episodes in "
                            f"{value.elapsed_seconds:.1f}s)"
                        ),
                        force=True,
                    )

                parallel_result = run_parallel_work_units(
                    units,
                    _convert_partition_worker,
                    workers=workers,
                    on_result=partition_done,
                )
                print(
                    "worker completion order: " + ", ".join(parallel_result.completion_order),
                    file=sys.stderr,
                    flush=True,
                )
            atomic_write_json(
                temporary / "collection_manifest.json", _manifest(infos, output.name, args, temporary)
            )
            for info in infos:
                plan = replace(info.plan, output_path=temporary / info.partition_name)
                validation = partition_validations.get(info.partition_name)
                if validation is None:
                    raise ConversionError(
                        f"missing completed validation for {info.partition_name} before publication"
                    )
                _validate_checkpoint_fingerprint(plan.output_path, validation)
            publish_temporary_output(temporary, output, overwrite=args.overwrite)
            if args.resume and resume_state.exists():
                shutil.rmtree(resume_state)
            progress.finish(context="all partitions validated and atomically published")
            completed_successfully = True
        except BaseException:
            if not args.resume and temporary.exists():
                shutil.rmtree(temporary)
            elif args.resume:
                print(f"checkpoint retained for --resume: {temporary}", file=sys.stderr, flush=True)
            raise
        finally:
            shutil.rmtree(cache_root, ignore_errors=True)
    if completed_successfully:
        resume_lock.unlink(missing_ok=True)
    return output


INCOMPLETE_SENTINEL = "_INCOMPLETE"
SUCCESS_SENTINEL = "_SUCCESS"
DIRECT_LAYOUT_VERSION = 1


def _validation_size(validation: dict[str, Any]) -> int:
    files = validation.get("files")
    if not isinstance(files, list):
        raise ConversionError("partition validation has no durable file inventory")
    total = 0
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("size"), int):
            raise ConversionError("partition validation contains an invalid file size")
        total += int(record["size"])
    return total


def _direct_runtime_bytes(
    layout: ConversionRuntimeLayout,
    validations: dict[str, dict[str, Any]],
) -> int:
    committed = sum(_validation_size(value) for value in validations.values())
    # Active worker cache/temp bytes are already included in the conservative
    # inflight estimate.  Do not rescan any runtime directory: OSSFS metadata
    # traversal is slow and a worker may delete its cache between scandir
    # calls.  This keeps every coordinator check O(number of completed units).
    compact = RUNTIME_METADATA_RESERVE_BYTES
    for name in (INCOMPLETE_SENTINEL, SUCCESS_SENTINEL, "collection_manifest.json"):
        path = layout.output_path / name
        if path.is_file():
            compact += path.stat().st_size
    return committed + compact


def _write_incomplete_sentinel(
    layout: ConversionRuntimeLayout,
    *,
    fingerprint: str,
) -> None:
    atomic_write_json(
        layout.output_path / INCOMPLETE_SENTINEL,
        {
            "layout_version": DIRECT_LAYOUT_VERSION,
            "dataset_uid": layout.output_path.name,
            "fingerprint": fingerprint,
            "run_id": layout.run_id,
            "started_unix": time.time(),
        },
    )


def _validate_success_sentinel(
    infos: list[RobomimicPartitionInfo],
    output: Path,
    fingerprint: str,
) -> dict[str, Any]:
    success = read_json_object(output / SUCCESS_SENTINEL, "MimicGen success sentinel")
    if success.get("layout_version") != DIRECT_LAYOUT_VERSION:
        raise ConversionError(f"unsupported success sentinel at {output}")
    if success.get("fingerprint") != fingerprint:
        raise ConversionError("existing _SUCCESS belongs to a different conversion fingerprint")
    if (output / INCOMPLETE_SENTINEL).exists():
        raise ConversionError("output contains both _SUCCESS and _INCOMPLETE")
    if not (output / "collection_manifest.json").is_file():
        raise ConversionError("successful output is missing collection_manifest.json")
    validations = success.get("partition_validations")
    if not isinstance(validations, dict):
        raise ConversionError("success sentinel is missing partition validations")
    for info in infos:
        validation = validations.get(info.partition_name)
        if not isinstance(validation, dict):
            raise ConversionError(
                f"success sentinel is missing validation for {info.partition_name}"
            )
        _validate_checkpoint_fingerprint(output / info.partition_name, validation)
    return success


def convert_collection_staged(
    infos: list[RobomimicPartitionInfo],
    layout: ConversionRuntimeLayout,
    args: argparse.Namespace,
    source_root: Path,
) -> Path:
    """Acquire the run lock before any output, work, or log mutation."""

    with exclusive_resume_lock(layout.lock_path):
        return _convert_collection_staged_locked(infos, layout, args, source_root)


def _convert_collection_staged_locked(
    infos: list[RobomimicPartitionInfo],
    layout: ConversionRuntimeLayout,
    args: argparse.Namespace,
    source_root: Path,
) -> Path:
    """Write final-format partitions directly under a sentinel-published root."""

    output = layout.output_path
    requested_workers = int(args.workers)
    max_inflight_units = int(args.max_inflight_units)
    workers = min(requested_workers, max_inflight_units)
    if workers <= 0:
        raise ConversionError("effective worker count must be positive")
    if requested_workers * args.encoder_threads > MAX_TOTAL_ENCODER_THREADS:
        raise ConversionError(
            f"workers * encoder threads must not exceed {MAX_TOTAL_ENCODER_THREADS}"
        )
    state_payload = {
        **_fingerprint_payload(infos, source_root, output, args),
        "direct_layout_version": DIRECT_LAYOUT_VERSION,
    }
    fingerprint = canonical_fingerprint(state_payload)
    log = StructuredRunLog(layout)
    log.write(
        "start",
        {
            "output": str(output),
            "workers": requested_workers,
            "max_inflight_units": max_inflight_units,
            "fingerprint": fingerprint,
        },
    )
    success_path = output / SUCCESS_SENTINEL
    incomplete_path = output / INCOMPLETE_SENTINEL
    if success_path.is_file():
        _validate_success_sentinel(infos, output, fingerprint)
        if args.skip_existing:
            print(f"skipped verified successful output: {output}")
            return output
        if not args.overwrite:
            raise FileExistsError(
                f"verified _SUCCESS already exists; refusing to overwrite {output}"
            )
    if args.overwrite:
        if output.exists():
            shutil.rmtree(output)
        if layout.resume_dir.exists():
            shutil.rmtree(layout.resume_dir)
    elif output.exists() and not args.resume:
        raise ConversionError(
            f"unfinished output exists at {output}; rerun the identical command with --resume"
        )

    layout.output_root.mkdir(parents=True, exist_ok=True)
    layout.work_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    _write_incomplete_sentinel(layout, fingerprint=fingerprint)
    capacity = StagingCapacityGuard(
        layout.output_root,
        max_staging_bytes=args.max_staging_bytes,
        max_inflight_bytes=args.max_inflight_bytes,
        max_inflight_units=max_inflight_units,
        interval_seconds=args.storage_check_interval_seconds,
    )
    run_args = argparse.Namespace(**vars(args))
    # Direct staging always records durable partition markers, including the
    # first run; --resume only controls whether an existing incomplete root is
    # accepted at entry.
    run_args.resume = True
    run_args.workers = workers
    resume_validation = getattr(args, "resume_validation", "fast")
    probe_timeout = getattr(
        args, "resume_probe_timeout", DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS
    )
    completed_successfully = False
    try:
        with _null_context():
            pending, _reused_parts, reused_frames = _prepare_resume(
                infos,
                output,
                layout.resume_dir,
                state_payload,
                fingerprint,
                args.video_codec,
                validation_mode=resume_validation,
                probe_timeout_seconds=probe_timeout,
                allowed_data_entries={
                    INCOMPLETE_SENTINEL,
                    SUCCESS_SENTINEL,
                    "collection_manifest.json",
                },
            )
            partition_validations: dict[str, dict[str, Any]] = {}
            for info in infos:
                marker = layout.resume_dir / "partitions" / f"{info.partition_name}.json"
                if marker.is_file():
                    validation = read_json_object(marker, "partition marker").get(
                        "validation"
                    )
                    if isinstance(validation, dict):
                        partition_validations[info.partition_name] = validation
            progress = EtaProgress(
                f"{output.name} convert",
                sum(info.plan.num_frames for info in infos),
                "frames",
                interval_seconds=args.eta_interval_seconds,
                initial_completed=reused_frames,
            )
            progress.update(
                reused_frames,
                context=f"reused {len(infos) - len(pending)} checkpoints",
                force=True,
            )
            cache_root = layout.work_dir / "workers"
            if pending:
                units = _build_parallel_work_units(
                    infos,
                    pending,
                    output,
                    cache_root,
                    layout.resume_dir,
                    fingerprint,
                    run_args,
                )
                memory_budget, legacy_temp_budget = _resource_budgets(
                    args, layout.work_dir
                )
                temp_budget = (
                    args.max_inflight_bytes
                    if args.max_inflight_bytes is not None
                    else legacy_temp_budget
                )
                estimate = validate_inflight_budget(
                    units,
                    workers,
                    memory_budget_bytes=memory_budget,
                    temp_budget_bytes=temp_budget,
                )
                current_bytes = _direct_runtime_bytes(layout, partition_validations)
                capacity.check(
                    "initial worker dispatch",
                    current_staging_bytes=current_bytes,
                    inflight_bytes=estimate.temp_bytes,
                    inflight_units=estimate.workers,
                )
                log.write(
                    "dispatch",
                    {
                        "pending_units": len(units),
                        "inflight_units": estimate.workers,
                        "inflight_bytes": estimate.temp_bytes,
                    },
                )

                def partition_done(result: ParallelWorkResult) -> None:
                    value = result.value
                    if not isinstance(value, PartitionWorkerOutput):
                        raise ConversionError(f"invalid worker result for {result.key}")
                    partition_validations[result.key] = value.validation
                    current = _direct_runtime_bytes(layout, partition_validations)
                    capacity.check(
                        f"partition {result.key} completion",
                        current_staging_bytes=current,
                        inflight_bytes=0,
                        inflight_units=0,
                    )
                    log.write(
                        "partition_validated",
                        {
                            "partition": result.key,
                            "frames": value.frames,
                            "elapsed_seconds": value.elapsed_seconds,
                        },
                    )
                    progress.update(
                        progress.completed + value.frames,
                        context=f"{result.key} verified",
                        force=True,
                    )

                def before_partition_dispatch(
                    unit: ParallelWorkUnit,
                    active: tuple[ParallelWorkUnit, ...],
                ) -> None:
                    frontier = (*active, unit)
                    capacity.check(
                        f"partition {unit.key} dispatch",
                        current_staging_bytes=_direct_runtime_bytes(
                            layout, partition_validations
                        ),
                        inflight_bytes=sum(
                            item.estimated_temp_bytes for item in frontier
                        ),
                        inflight_units=len(frontier),
                    )

                parallel_result = run_parallel_work_units(
                    units,
                    _convert_partition_worker,
                    workers=workers,
                    on_result=partition_done,
                    before_dispatch=before_partition_dispatch,
                )
                log.write(
                    "workers_complete",
                    {"completion_order": list(parallel_result.completion_order)},
                )
            atomic_write_json(
                output / "collection_manifest.json",
                _manifest(infos, output.name, args, output),
            )
            for info in infos:
                validation = partition_validations.get(info.partition_name)
                if validation is None:
                    raise ConversionError(
                        f"missing completed validation for {info.partition_name}"
                    )
                _validate_checkpoint_fingerprint(
                    output / info.partition_name, validation
                )
            final_bytes = _direct_runtime_bytes(layout, partition_validations)
            capacity.check(
                "final validation",
                current_staging_bytes=final_bytes,
                inflight_bytes=0,
                inflight_units=0,
            )
            atomic_write_json(
                success_path,
                {
                    "layout_version": DIRECT_LAYOUT_VERSION,
                    "dataset_uid": output.name,
                    "fingerprint": fingerprint,
                    "completed_unix": time.time(),
                    "partitions": len(infos),
                    "episodes": sum(len(info.plan.episodes) for info in infos),
                    "frames": sum(info.plan.num_frames for info in infos),
                    "partition_validations": partition_validations,
                },
            )
            incomplete_path.unlink()
            if layout.resume_dir.exists():
                shutil.rmtree(layout.resume_dir)
            progress.finish(context="_SUCCESS written; _INCOMPLETE removed")
            completed_successfully = True
            log.write("success", {"staging_bytes": final_bytes})
    except BaseException as exc:
        log.write("failure", {"type": type(exc).__name__, "message": str(exc)})
        print(
            f"checkpoint retained under {layout.resume_dir}; output remains _INCOMPLETE",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        if layout.work_dir.exists():
            shutil.rmtree(layout.work_dir)
        if layout.temp_dir != layout.work_dir and layout.temp_dir.exists():
            shutil.rmtree(layout.temp_dir)
    if not completed_successfully:  # pragma: no cover - defensive
        raise ConversionError("direct staged conversion did not complete")
    return output


def _json_lines(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _verify_partition_equivalence(reference: Path, candidate: Path) -> None:
    import pyarrow.parquet as pq

    reference_files = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*")
        if path.is_file() and path.suffix != ".mp4"
    )
    candidate_files = sorted(
        path.relative_to(candidate).as_posix()
        for path in candidate.rglob("*")
        if path.is_file() and path.suffix != ".mp4"
    )
    if candidate_files != reference_files:
        raise ConversionError(
            f"parallel output files differ for {reference.name}: "
            f"reference={reference_files}, candidate={candidate_files}"
        )
    for relative in reference_files:
        left = reference / relative
        right = candidate / relative
        if left.suffix == ".parquet":
            left_table = pq.read_table(left)
            right_table = pq.read_table(right)
            if not left_table.schema.equals(right_table.schema, check_metadata=True):
                raise ConversionError(f"Parquet schema differs: {relative}")
            if not left_table.equals(right_table):
                raise ConversionError(f"Parquet values differ: {relative}")
        elif left.suffix == ".json":
            if json.loads(left.read_text(encoding="utf-8")) != json.loads(
                right.read_text(encoding="utf-8")
            ):
                raise ConversionError(f"JSON metadata differs: {relative}")
        elif left.suffix == ".jsonl":
            if _json_lines(left) != _json_lines(right):
                raise ConversionError(f"JSONL metadata differs: {relative}")
        elif left.read_bytes() != right.read_bytes():
            raise ConversionError(f"non-video output differs: {relative}")

    reference_videos = sorted(
        path.relative_to(reference).as_posix() for path in reference.rglob("*.mp4")
    )
    candidate_videos = sorted(
        path.relative_to(candidate).as_posix() for path in candidate.rglob("*.mp4")
    )
    if candidate_videos != reference_videos:
        raise ConversionError(
            f"video paths differ for {reference.name}: "
            f"reference={reference_videos}, candidate={candidate_videos}"
        )
    stream_fields = (
        "codec_name",
        "width",
        "height",
        "nb_read_frames",
        "pix_fmt",
    )
    for relative in reference_videos:
        left_stream = _ffprobe(reference / relative)
        right_stream = _ffprobe(candidate / relative)
        left_signature = {key: left_stream.get(key) for key in stream_fields}
        right_signature = {key: right_stream.get(key) for key in stream_fields}
        if left_signature != right_signature:
            raise ConversionError(f"video stream metadata differs: {relative}")
        left_rate = _fraction(
            left_stream.get("avg_frame_rate") or left_stream["r_frame_rate"]
        )
        right_rate = _fraction(
            right_stream.get("avg_frame_rate") or right_stream["r_frame_rate"]
        )
        if not math.isclose(left_rate, right_rate, rel_tol=0.0, abs_tol=1e-9):
            raise ConversionError(f"video frame rate differs: {relative}")


def verify_collection_equivalence(
    infos: list[RobomimicPartitionInfo],
    reference: Path,
    candidate: Path,
    *,
    video_codec: str,
) -> dict[str, int]:
    """Prove plan-order metadata, values, indices, and streams are equivalent."""

    reference_manifest = read_json_object(
        reference / "collection_manifest.json", "reference collection manifest"
    )
    candidate_manifest = read_json_object(
        candidate / "collection_manifest.json", "candidate collection manifest"
    )
    if candidate_manifest != reference_manifest:
        raise ConversionError("serial and parallel collection manifests differ")
    video_files = 0
    for info in infos:
        reference_partition = reference / info.partition_name
        candidate_partition = candidate / info.partition_name
        reference_plan = replace(info.plan, output_path=reference_partition)
        candidate_plan = replace(info.plan, output_path=candidate_partition)
        validate_written_dataset(reference_plan, reference_partition)
        validate_written_dataset(candidate_plan, candidate_partition)
        reference_streams = _validate_video_streams(
            reference_plan, reference_partition, video_codec
        )
        candidate_streams = _validate_video_streams(
            candidate_plan, candidate_partition, video_codec
        )
        if reference_streams != candidate_streams:
            raise ConversionError(f"video evidence differs for {info.partition_name}")
        video_files += len(reference_streams)
        _verify_partition_equivalence(reference_partition, candidate_partition)
    return {
        "partitions": len(infos),
        "episodes": sum(len(info.plan.episodes) for info in infos),
        "frames": sum(info.plan.num_frames for info in infos),
        "video_files": video_files,
    }


def benchmark_worker_counts(
    infos: list[RobomimicPartitionInfo],
    args: argparse.Namespace,
    source_root: Path,
    layout: ConversionRuntimeLayout,
) -> dict[str, Any]:
    worker_counts = list(args.benchmark_workers)
    total_frames = sum(info.plan.num_frames for info in infos)
    benchmark_parent = layout.work_dir / "benchmark"
    benchmark_parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    output_by_workers: dict[int, Path] = {}
    with tempfile.TemporaryDirectory(
        prefix=".mimicgen-benchmark-", dir=benchmark_parent
    ) as directory:
        benchmark_root = Path(directory)
        for workers in worker_counts:
            run_root = benchmark_root / f"workers-{workers}"
            run_root.mkdir()
            run_layout = build_runtime_layout(
                output_root=run_root,
                dataset_uid="mimicgen_benchmark_sample",
                run_id=f"workers-{workers}",
                lock_name="mimicgen.lock",
            )
            output = run_layout.output_path
            run_args = argparse.Namespace(**vars(args))
            run_args.resume = False
            run_args.skip_existing = False
            run_args.overwrite = False
            run_args.workers = workers
            run_args.benchmark_workers = None
            sampler = ProcessTreeSampler(run_root)
            sampler.start()
            try:
                convert_collection_staged(infos, run_layout, run_args, source_root)
            finally:
                metrics = sampler.stop()
            output_by_workers[workers] = output
            row = {
                "workers": workers,
                "encoder_threads_per_worker": args.encoder_threads,
                "frames": total_frames,
                "frames_per_second": total_frames / metrics.wall_seconds,
                **metrics.as_dict(),
            }
            results.append(row)

        baseline = next(row for row in results if row["workers"] == 1)
        for row in results:
            row["speedup"] = baseline["wall_seconds"] / row["wall_seconds"]
            row["average_cpu_percent"] = row["average_cpu_cores"] * 100.0

        equivalence: dict[str, Any] = {}
        reference = output_by_workers[1]
        for workers in worker_counts:
            if workers == 1:
                continue
            equivalence[str(workers)] = verify_collection_equivalence(
                infos,
                reference,
                output_by_workers[workers],
                video_codec=args.video_codec,
            )
        report = {
            "sample": {
                "source_partitions": [info.source_relative_path for info in infos],
                "partitions": len(infos),
                "episodes": sum(len(info.plan.episodes) for info in infos),
                "frames": total_frames,
            },
            "runs": results,
            "equivalence_verified": equivalence,
            "artifacts_cleaned": True,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=APPROVED_OUTPUT_ROOT,
        help="Validated staging root containing final data, checkpoints, and logs.",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        help=(
            "Legacy alias for the parent of lerobot_v3_0; retained for CLI compatibility. "
            "New commands should use --output-root."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Local POSIX work directory required for streaming video encoding.",
    )
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Temporary directory below --work-dir on a local POSIX filesystem.",
    )
    parser.add_argument("--source-directory", default="mimicgen")
    parser.add_argument("--dataset-uid", default=DEFAULT_DATASET_UID)
    parser.add_argument("--category", action="append", default=[], help="Select source/core/object/robot/large_interpolation.")
    parser.add_argument("--partition", action="append", default=[], help="Select relative partition stem, e.g. core/square_d0.")
    parser.add_argument("--max-partitions", type=_positive_int)
    parser.add_argument("--max-episodes", type=_positive_int)
    parser.add_argument("--dry-run", "--inspect-only", dest="inspect_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-validation",
        choices=("fast", "full"),
        default="fast",
        help="Validate unchanged checkpoints by file fingerprint (fast) or rescan all videos (full).",
    )
    parser.add_argument(
        "--resume-probe-timeout",
        type=_positive_float,
        default=DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="Maximum ffprobe time per video; timeout preserves the checkpoint and aborts.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eta-interval-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--inspect-workers", type=_positive_int, default=4)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Independent CPU HDF5 partition writers (default: 4).",
    )
    parser.add_argument(
        "--benchmark-workers",
        type=_positive_int,
        nargs="+",
        metavar="N",
        help="Benchmark bounded real samples with identical inputs; include 1 as the baseline.",
    )
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--video-quality", type=int, default=18)
    parser.add_argument("--video-preset", default="fast")
    parser.add_argument(
        "--encoder-threads-per-worker",
        "--encoder-threads",
        dest="encoder_threads",
        type=_positive_int,
        default=4,
        help="CPU encoder threads per writer; --encoder-threads remains a compatible alias.",
    )
    parser.add_argument(
        "--memory-budget-gib",
        type=_positive_float,
        help="Optional inflight worker memory ceiling; default is 80%% of MemAvailable.",
    )
    parser.add_argument(
        "--temp-budget-gib",
        type=_positive_float,
        help="Optional inflight temporary-space ceiling; default is 80%% of free space.",
    )
    parser.add_argument(
        "--max-staging-bytes",
        type=_positive_int,
        help="Stop before committed plus estimated inflight data exceeds this byte limit.",
    )
    parser.add_argument(
        "--max-inflight-bytes",
        type=_positive_int,
        help="Maximum estimated bytes across concurrently active partition units.",
    )
    parser.add_argument(
        "--max-inflight-units",
        type=_positive_int,
        help="Maximum concurrently active partition units (default: --workers).",
    )
    parser.add_argument(
        "--storage-check-interval-seconds",
        type=_positive_float,
        default=DEFAULT_STORAGE_CHECK_INTERVAL_SECONDS,
        metavar="SECONDS",
        help="Minimum interval between worker-side storage checks (default: 10).",
    )
    parser.add_argument("--skip-encoder-preflight", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.staging_root is not None:
        legacy_output_root = args.staging_root / "lerobot_v3_0"
        if args.output_root != APPROVED_OUTPUT_ROOT and (
            args.output_root.resolve(strict=False) != legacy_output_root.resolve(strict=False)
        ):
            parser.error("--output-root and legacy --staging-root resolve to different roots")
        args.output_root = legacy_output_root
    if args.max_inflight_units is None:
        args.max_inflight_units = args.workers
    elif args.max_inflight_units < args.workers:
        print(
            f"worker dispatch explicitly capped from {args.workers} to "
            f"{args.max_inflight_units} by --max-inflight-units",
            file=sys.stderr,
            flush=True,
        )
    if args.video_codec in FORBIDDEN_NVENC_CODECS:
        parser.error(
            f"{args.video_codec} is disabled on this host; use a CPU codec such as h264"
        )
    enabled = sum(bool(value) for value in (args.resume, args.skip_existing, args.overwrite))
    if enabled > 1:
        parser.error("--resume, --skip-existing, and --overwrite are mutually exclusive")
    if args.workers * args.encoder_threads > MAX_TOTAL_ENCODER_THREADS:
        parser.error(
            f"--workers * --encoder-threads-per-worker must not exceed "
            f"{MAX_TOTAL_ENCODER_THREADS}"
        )
    if args.benchmark_workers is not None:
        if any((args.resume, args.skip_existing, args.overwrite, args.inspect_only)):
            parser.error(
                "--benchmark-workers cannot be combined with resume, skip, overwrite, or inspect-only"
            )
        if len(args.benchmark_workers) != len(set(args.benchmark_workers)):
            parser.error("--benchmark-workers values must be unique")
        if 1 not in args.benchmark_workers:
            parser.error("--benchmark-workers must include 1 as the serial baseline")
        if any(
            workers * args.encoder_threads > MAX_TOTAL_ENCODER_THREADS
            for workers in args.benchmark_workers
        ):
            parser.error(
                f"every benchmark workers * encoder threads value must not exceed "
                f"{MAX_TOTAL_ENCODER_THREADS}"
            )
        if args.max_episodes is None or args.max_episodes > MAX_BENCHMARK_EPISODES_PER_PARTITION:
            parser.error(
                "--benchmark-workers requires --max-episodes between 1 and "
                f"{MAX_BENCHMARK_EPISODES_PER_PARTITION}"
            )
        if not args.partition and args.max_partitions is None:
            parser.error("--benchmark-workers requires --partition or --max-partitions")
        if args.max_partitions is not None and args.max_partitions > MAX_BENCHMARK_PARTITIONS:
            parser.error(
                f"benchmark --max-partitions must not exceed {MAX_BENCHMARK_PARTITIONS}"
            )
    if (args.max_episodes is not None or args.max_partitions is not None or args.category or args.partition) and (
        args.dataset_uid == DEFAULT_DATASET_UID
        and not args.inspect_only
        and args.benchmark_workers is None
    ):
        parser.error("limited conversion requires an explicit smoke --dataset-uid")
    try:
        layout = build_runtime_layout(
            output_root=args.output_root,
            dataset_uid=args.dataset_uid,
            work_dir=args.work_dir,
            resume_dir=args.resume_dir,
            logs_dir=args.logs_dir,
            temp_dir=args.temp_dir,
            required_output_root=APPROVED_OUTPUT_ROOT,
        )
        if not args.inspect_only:
            validate_streaming_runtime_filesystems(layout)
    except ConversionError as exc:
        parser.error(str(exc))
    args.output_root = layout.output_root
    args.work_dir = layout.work_dir
    args.resume_dir = layout.resume_dir
    args.logs_dir = layout.logs_dir
    args.temp_dir = layout.temp_dir
    source_root = _source_root(args.raw_root, args.source_directory)
    output = layout.output_path
    runtime_context = (
        _null_context() if args.inspect_only else redirected_runtime_environment(layout)
    )
    with runtime_context:
        paths = _select_source_files(
            source_root,
            categories=set(args.category),
            partitions=set(args.partition),
            max_partitions=args.max_partitions,
        )
        if args.benchmark_workers is not None and len(paths) > MAX_BENCHMARK_PARTITIONS:
            parser.error(
                f"benchmark selection resolved to {len(paths)} partitions; maximum is "
                f"{MAX_BENCHMARK_PARTITIONS}"
            )
        infos = inspect_collection(
            source_root,
            output,
            paths,
            max_episodes=args.max_episodes,
            eta_interval_seconds=args.eta_interval_seconds,
            workers=args.inspect_workers,
        )
        print(json.dumps(_collection_summary(infos, output), ensure_ascii=False, indent=2))
        if args.inspect_only:
            print("preflight validated; no output written")
            return 0
        if args.benchmark_workers is not None:
            if not args.skip_encoder_preflight:
                _encoder_preflight(infos[0], args)
            benchmark_worker_counts(infos, args, source_root, layout)
            return 0
        if not args.skip_encoder_preflight:
            _encoder_preflight(infos[0], args)
        convert_collection_staged(infos, layout, args, source_root)
    print(
        f"wrote {len(infos)} partitions / {sum(len(i.plan.episodes) for i in infos)} episodes / "
        f"{sum(i.plan.num_frames for i in infos)} frames to {output}"
    )
    return 0


if __name__ == "__main__":
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("received SIGTERM")

    signal.signal(signal.SIGTERM, _sigterm)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; verified partition checkpoints were retained when --resume was used", file=sys.stderr)
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
