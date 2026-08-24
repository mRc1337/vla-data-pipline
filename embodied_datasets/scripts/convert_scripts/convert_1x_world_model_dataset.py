"""Convert 1X World Model Challenge v1.1/v2.0 tokens to a LeRobot collection.

This is a thin collection/decoder layer.  Source inspection lives in
``readers.one_x_world_model_reader`` and all durable writing, validation,
checkpoint snapshots, locking, and atomic publication live in
``convert_core``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from convert_core.checkpoint import (
    RESUME_SCHEMA_VERSION,
    atomic_write_json,
    build_resume_payload,
    canonical_fingerprint,
    exclusive_resume_lock,
    read_json_object,
    resume_paths,
)
from convert_core.dataset_config import load_dataset_config
from convert_core.direct_commit import (
    DirectCommitUploader,
    finalize_direct_partition,
    globalize_unit_data_files,
    prepare_direct_commits,
    read_committed_unit_marker,
)
from convert_core.errors import ConversionError
from convert_core.equivalence import verify_lerobot_equivalence
from convert_core.lerobot_writer import (
    convert_dataset,
    plan_summary,
    publish_temporary_output,
    validate_video_files,
    validate_written_dataset,
)
from convert_core.parallel import (
    ParallelWorkUnit,
    aggregate_lerobot_work_units,
    isolated_unit_plan,
    prepare_work_units,
    run_parallel_work_units,
    split_plan_into_units,
    validate_inflight_budget,
    validate_verified_unit_marker,
    verified_marker_path,
    write_verified_unit_marker,
)
from convert_core.performance import ProcessTreeSampler
from convert_core.staging import (
    DEFAULT_LOCAL_WORK_ROOT,
    DEFAULT_MAX_LOCAL_TEMP_BYTES,
    DEFAULT_MIN_LOCAL_FREE_BYTES,
    DEFAULT_OUTPUT_ROOT,
    RUNTIME_ENVIRONMENT_KEYS,
    StagingCapacityGuard,
    StagingLayout,
    configure_runtime_environment,
    create_incomplete_output,
    exclusive_staging_lock,
    make_staging_layout,
    publish_success,
    validate_no_runtime_paths_outside_root,
    validate_source_and_output_roots,
)
from readers.one_x_world_model_reader import OneXWorldModelReader


VERSIONS = ("v1.1", "v2.0")
VIDEO_CODECS = ("libsvtav1", "h264", "hevc", "h264_nvenc", "hevc_nvenc")
PREFLIGHT_FRAME_COUNT = 60
NVENC_PRESETS = {
    "default": 0,
    "slow": 1,
    "medium": 2,
    "fast": 3,
    "hp": 4,
    "hq": 5,
    "bd": 6,
    "ll": 7,
    "llhq": 8,
    "llhp": 9,
    "lossless": 10,
    "losslesshp": 11,
    "p1": 12,
    "p2": 13,
    "p3": 14,
    "p4": 15,
    "p5": 16,
    "p6": 17,
    "p7": 18,
}

GIB = 1024**3
MIB = 1024**2
DEFAULT_PARALLEL_QUEUE_SIZE = 64
DEFAULT_ENCODER_THREADS_PER_WORKER = 8
MAX_TOTAL_ENCODER_THREADS = 32
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True)
class _OneXWorkerPayload:
    plan: Any
    video_codec: str
    video_quality: int
    video_preset: str | None
    queue_size: int
    encoder_threads: int
    decoder_cpu_threads: int | None
    v1_postprocess_device: str
    eta_interval_seconds: float
    conversion_options: dict[str, Any]


_WORKER_READER: OneXWorldModelReader | None = None
_WORKER_PREFLIGHTED: set[str] = set()


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


def _preset(codec: str, value: str | None) -> str | int | None:
    if codec not in {"h264_nvenc", "hevc_nvenc"}:
        return value
    selected = value or "p4"
    if selected in NVENC_PRESETS:
        return NVENC_PRESETS[selected]
    if selected.isdecimal():
        return int(selected)
    raise ConversionError(
        f"unsupported NVENC preset {selected!r}; choose one of {sorted(NVENC_PRESETS)}"
    )


def _rgb_encoder(codec: str, quality: int, preset: str | None) -> Any:
    try:
        from lerobot.configs.video import RGBEncoderConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lerobot==0.6.0 is required") from exc
    return RGBEncoderConfig(vcodec=codec, crf=quality, preset=_preset(codec, preset))


def _preflight_encoder(plan, encoder: Any) -> None:
    """Open a real encoder session and encode a short 60-frame sample."""

    try:
        import av
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyAV is required for video encoder preflight") from exc
    camera = plan.camera_features[0]
    buffer = io.BytesIO()
    try:
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream(
            encoder.vcodec,
            rate=plan.fps,
            options=encoder.get_codec_options(as_strings=True),
        )
        stream.width = camera.width
        stream.height = camera.height
        stream.pix_fmt = encoder.pix_fmt
        sample = np.zeros(
            (camera.height, camera.width, 3), dtype=np.uint8
        )
        for _ in range(PREFLIGHT_FRAME_COUNT):
            frame = av.VideoFrame.from_ndarray(sample, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception as exc:
        raise ConversionError(
            f"{encoder.vcodec} encoder preflight failed at {camera.width}x{camera.height}: {exc}. "
            "FFmpeg listing an encoder does not prove hardware/runtime support."
        ) from exc


def _enable_fragmented_streaming_mp4() -> None:
    """Make LeRobot's streaming MP4 append-only for OSSFS writes.

    A conventional MP4 muxer seeks while finalizing container metadata, but
    the mounted object filesystem rejects that seek with ``EINVAL``.  A
    fragmented MP4 writes its initialization metadata first and appends media
    fragments, without changing the encoded frames or final LeRobot paths.
    """

    from lerobot.datasets import video_utils

    marker = "_vla_fragmented_streaming_mp4"
    if getattr(video_utils, marker, False):
        return
    original_open = video_utils.av.open

    def open_fragmented_stream(
        file: Any, mode: str = "r", *open_args: Any, **open_kwargs: Any
    ) -> Any:
        if mode == "w" and str(file).endswith("_streaming.mp4"):
            options = dict(open_kwargs.get("options") or {})
            options["movflags"] = (
                "frag_keyframe+empty_moov+default_base_moof+negative_cts_offsets"
            )
            open_kwargs["options"] = options
        return original_open(file, mode, *open_args, **open_kwargs)

    video_utils.av.open = open_fragmented_stream
    setattr(video_utils, marker, True)


def _visible_cuda_devices() -> tuple[str, ...]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        devices = tuple(item.strip() for item in configured.split(",") if item.strip())
        if not devices or configured.strip() == "-1":
            raise ConversionError("parallel decoding requires visible CUDA devices")
        return devices
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConversionError(f"cannot enumerate CUDA devices with nvidia-smi: {exc}") from exc
    devices = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if not devices:
        raise ConversionError("parallel decoding requires at least one CUDA device")
    return devices


def _initialize_one_x_worker(slot: int, devices: tuple[str, ...]) -> None:
    """Bind a spawned worker before any CUDA runtime is imported."""

    global _WORKER_READER, _WORKER_PREFLIGHTED
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[slot]
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    )
    # The source tokens are deterministic, so decoder output must not depend
    # on which identical A800 handles a work unit.  Configure torch only after
    # binding the spawned process; the reader deliberately imports torch
    # lazily, so no CUDA context exists before this point.
    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for attribute in (
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_reduced_precision_reduction",
    ):
        if hasattr(torch.backends.cuda.matmul, attribute):
            setattr(torch.backends.cuda.matmul, attribute, False)
    _WORKER_READER = OneXWorldModelReader()
    _WORKER_PREFLIGHTED = set()


def _convert_one_x_work_unit(unit: ParallelWorkUnit) -> dict[str, Any]:
    global _WORKER_READER, _WORKER_PREFLIGHTED
    payload = unit.payload
    if not isinstance(payload, _OneXWorkerPayload):
        raise ConversionError(f"invalid 1X worker payload for {unit.key!r}")
    reader = _WORKER_READER
    if reader is None:
        reader = OneXWorldModelReader()
        _WORKER_READER = reader
    version = str(payload.plan.extra["source_version"])
    if version == "v1.1" and payload.decoder_cpu_threads is not None:
        import torch

        torch.set_num_threads(payload.decoder_cpu_threads)
    runtime_decoder = dict(payload.plan.extra.get("decoder", {}))
    runtime_decoder["v1_postprocess_device"] = payload.v1_postprocess_device
    runtime_plan = replace(
        payload.plan,
        extra={**payload.plan.extra, "decoder": runtime_decoder},
    )
    encoder = _rgb_encoder(
        payload.video_codec, payload.video_quality, payload.video_preset
    )
    _enable_fragmented_streaming_mp4()
    if version not in _WORKER_PREFLIGHTED:
        print(f"[{unit.key} worker] decoder preflight started", flush=True)
        reader.preflight_decoder(runtime_plan)
        print(f"[{unit.key} worker] decoder preflight completed", flush=True)
        _preflight_encoder(runtime_plan, encoder)
        print(f"[{unit.key} worker] encoder preflight completed", flush=True)
        _WORKER_PREFLIGHTED.add(version)
    print(f"[{unit.key} worker] conversion started", flush=True)
    convert_dataset(
        runtime_plan,
        lambda episode: reader.iter_frames(runtime_plan, episode),
        reader_format="one_x_world_model",
        resume=True,
        eta_interval_seconds=payload.eta_interval_seconds,
        rgb_encoder=encoder,
        streaming_encoding=True,
        blocking_streaming_encoding=True,
        encoder_queue_maxsize=payload.queue_size,
        encoder_threads=payload.encoder_threads,
        fragmented_mp4_writes=True,
        conversion_options=payload.conversion_options,
    )
    validate_written_dataset(payload.plan, Path(unit.target_path))
    validate_video_files(
        payload.plan,
        Path(unit.target_path),
        expected_frames=unit.weight,
    )
    globalize_unit_data_files(unit)
    write_verified_unit_marker(unit)
    return {
        "unit_index": unit.index,
        "unit_key": unit.key,
        "episodes": unit.episode_end - unit.episode_start,
        "frames": unit.weight,
        "cuda_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _parallel_threads(args: argparse.Namespace) -> int:
    return (
        args.encoder_threads_per_worker
        or args.encoder_threads
        or DEFAULT_ENCODER_THREADS_PER_WORKER
    )


def _parallel_queue_size(args: argparse.Namespace) -> int:
    return args.encoder_queue_maxsize or DEFAULT_PARALLEL_QUEUE_SIZE


def _build_parallel_work_units(
    plan: Any,
    work_root: Path,
    args: argparse.Namespace,
    conversion_options: dict[str, Any],
) -> tuple[ParallelWorkUnit, ...]:
    # Reader checkpoints are also decoder-context boundaries. In particular,
    # v2 token blocks can straddle episodes; splitting a shard at an episode
    # boundary changes cache/batch context and therefore reconstructed pixels.
    slices = split_plan_into_units(plan)
    queue_size = _parallel_queue_size(args)
    threads = _parallel_threads(args)
    units_root = work_root / "parallel-units" / plan.output_path.name
    units: list[ParallelWorkUnit] = []
    for item in slices:
        target = units_root / f"unit-{item.index:06d}"
        dataset_uid = f"{plan.dataset_uid}__unit_{item.index:06d}"
        unit_plan = isolated_unit_plan(
            plan,
            item,
            dataset_uid=dataset_uid,
            target_path=target,
        )
        unit_options = {
            **conversion_options,
            "parallel_schema_version": 1,
            "parallel_unit_index": item.index,
            "parallel_unit_key": item.key,
            "blocking_streaming_encoding": True,
            "encoder_queue_maxsize": queue_size,
            "encoder_threads_per_worker": threads,
        }
        fingerprint = canonical_fingerprint(
            build_resume_payload(
                unit_plan,
                reader_format="one_x_world_model",
                conversion_options=unit_options,
            )
        )
        camera_bytes = sum(
            camera.height * camera.width * 3 for camera in unit_plan.camera_features
        )
        memory_estimate = 8 * GIB + queue_size * camera_bytes
        temp_estimate = max(16 * MIB, item.frame_end - item.frame_start << 14)
        payload = _OneXWorkerPayload(
            plan=unit_plan,
            video_codec=args.video_codec,
            video_quality=args.video_quality,
            video_preset=args.video_preset,
            queue_size=queue_size,
            encoder_threads=threads,
            decoder_cpu_threads=args.decoder_cpu_threads,
            v1_postprocess_device=args.v1_postprocess_device,
            eta_interval_seconds=args.eta_interval_seconds,
            conversion_options=unit_options,
        )
        units.append(
            ParallelWorkUnit(
                index=item.index,
                key=item.key,
                dataset_uid=dataset_uid,
                target_path=str(target),
                episode_start=item.episode_start,
                episode_end=item.episode_end,
                frame_start=item.frame_start,
                frame_end=item.frame_end,
                task_indices=item.task_indices,
                weight=item.frame_end - item.frame_start,
                estimated_memory_bytes=memory_estimate,
                estimated_temp_bytes=temp_estimate,
                fingerprint=fingerprint,
                payload=payload,
            )
        )
    return tuple(units)


def _validate_parallel_unit_output(unit: ParallelWorkUnit) -> None:
    payload = unit.payload
    if not isinstance(payload, _OneXWorkerPayload):
        raise ConversionError(f"invalid 1X worker payload for {unit.key!r}")
    if verified_marker_path(unit).is_file():
        validate_verified_unit_marker(unit)
        return
    validate_written_dataset(payload.plan, Path(unit.target_path))
    validate_video_files(payload.plan, Path(unit.target_path), expected_frames=unit.weight)
    globalize_unit_data_files(unit)


def _convert_parallel_partition(
    plan: Any,
    work_root: Path,
    resume_root: Path,
    args: argparse.Namespace,
    conversion_options: dict[str, Any],
    devices: tuple[str, ...],
    capacity_guard: StagingCapacityGuard,
) -> None:
    assert args.workers is not None
    units = _build_parallel_work_units(plan, work_root, args, conversion_options)
    estimate = validate_inflight_budget(
        units,
        args.workers,
        memory_budget_bytes=int(args.inflight_memory_budget_gb * GIB),
        temp_budget_bytes=min(
            int(args.inflight_temp_budget_gb * GIB),
            args.max_inflight_bytes,
        ),
    )
    capacity_guard.wait_for_capacity(
        f"dispatching {plan.output_path.name}",
        required_additional_bytes=estimate.temp_bytes,
    )
    plan.output_path.mkdir(parents=True, exist_ok=True)
    direct = prepare_direct_commits(
        units,
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume_root,
    )
    if direct.uncommitted:
        prepared = prepare_work_units(
            direct.uncommitted,
            _validate_parallel_unit_output,
            require_complete_plan=False,
        )
    else:
        prepared = None

    reused_work = prepared.reusable if prepared is not None else ()
    pending = prepared.pending if prepared is not None else ()
    completion_order: tuple[str, ...] = ()
    uploader = DirectCommitUploader(
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume_root,
        workers=args.upload_workers,
        max_queue_units=args.max_upload_queue_units,
    )
    upload_stats: dict[str, int] = {}
    try:
        for unit in reused_work:
            uploader.submit(unit, trust_verified_marker=False)
        if pending:
            by_index = {unit.index: unit for unit in pending}

            def commit_result(result: Any) -> None:
                uploader.submit(
                    by_index[result.index], trust_verified_marker=True
                )

            def before_dispatch(
                unit: ParallelWorkUnit,
                _active: tuple[ParallelWorkUnit, ...],
            ) -> None:
                capacity_guard.wait_for_capacity(
                    f"dispatching local unit {unit.key}",
                    required_additional_bytes=unit.estimated_temp_bytes,
                    abort_check=uploader.raise_if_failed,
                )

            def health_check() -> None:
                uploader.raise_if_failed()
                capacity_guard.check(
                    f"workers active for {plan.output_path.name}"
                )

            result = run_parallel_work_units(
                pending,
                _convert_one_x_work_unit,
                workers=args.workers,
                on_result=commit_result,
                initializer=_initialize_one_x_worker,
                initargs=(devices,),
                before_dispatch=before_dispatch,
                health_check=health_check,
                health_check_interval_seconds=args.storage_check_interval_seconds,
            )
            completion_order = result.completion_order
        upload_stats = uploader.close()
    except BaseException:
        uploader.close(raise_on_failure=False)
        raise
    for unit in units:
        read_committed_unit_marker(
            unit,
            partition_name=plan.output_path.name,
            resume_root=resume_root,
        )
    finalize_direct_partition(
        plan,
        units,
        plan.output_path,
        resume_root=resume_root,
        reader_format="one_x_world_model",
        parallel_evidence={
            "workers": args.workers,
            "encoder_threads_per_worker": _parallel_threads(args),
            "total_encoder_threads": args.workers * _parallel_threads(args),
            "encoder_queue_maxsize": _parallel_queue_size(args),
            "inflight_memory_estimate_bytes": estimate.memory_bytes,
            "inflight_temp_estimate_bytes": estimate.temp_bytes,
            "max_inflight_bytes": args.max_inflight_bytes,
            "max_inflight_units": args.max_inflight_units,
            "upload_pipeline": upload_stats,
            "local_capacity": {
                "max_local_temp_bytes": capacity_guard.max_staging_bytes,
                "min_local_free_bytes": capacity_guard.min_free_bytes,
                "peak_accounted_bytes": capacity_guard.peak_staging_bytes,
            },
            "reused_committed_units": [unit.key for unit in direct.committed],
            "reused_work_units": [unit.key for unit in reused_work],
            "repaired_markers": (
                list(prepared.repaired_markers) if prepared is not None else []
            ),
            "discarded_corrupt_units": [
                *direct.discarded_corrupt,
                *(prepared.discarded_corrupt if prepared is not None else ()),
            ],
            "worker_completion_order": list(completion_order),
            "cuda_devices": list(devices[: args.workers]),
            "deterministic_cuda": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "tf32": False,
                "reduced_precision_reduction": False,
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG", DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
                ),
            },
            "video_codec_policy": "CPU libx264/libx265/libsvtav1 only; NVENC forbidden",
        },
    )


def _strip_cli_option(argv: list[str], option: str, *, many: bool = False) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == option:
            index += 1
            if many:
                while index < len(argv) and not argv[index].startswith("--"):
                    index += 1
            elif index < len(argv):
                index += 1
            continue
        if value.startswith(f"{option}="):
            index += 1
            continue
        result.append(value)
        index += 1
    return result


def _semantic_collection_manifest(path: Path) -> dict[str, Any]:
    value = read_json_object(path, "collection manifest")
    value = dict(value)
    value.pop("dataset_uid", None)
    value.pop("parallel", None)
    partitions = []
    for partition in value.get("partitions", []):
        normalized = dict(partition)
        normalized.pop("dataset_uid", None)
        partitions.append(normalized)
    value["partitions"] = partitions
    return value


def _run_worker_benchmarks(
    raw_argv: list[str], args: argparse.Namespace
) -> int:
    assert args.benchmark_workers
    base_argv = _strip_cli_option(raw_argv, "--benchmark-workers", many=True)
    base_argv = _strip_cli_option(base_argv, "--benchmark-report")
    base_argv = _strip_cli_option(base_argv, "--output-dataset-uid")
    sample_limits = [
        value
        for value in (args.max_episodes, args.max_checkpoint_units)
        if value is not None
    ]
    sample_units = min(sample_limits)
    base_argv = _strip_cli_option(base_argv, "--max-episodes")
    base_argv = _strip_cli_option(base_argv, "--max-checkpoint-units")
    for option in (
        "--local-work-root",
        "--work-dir",
        "--resume-dir",
        "--logs-dir",
        "--temp-dir",
    ):
        base_argv = _strip_cli_option(base_argv, option)
    base_argv.extend(
        [
            "--max-checkpoint-units",
            str(sample_units),
            "--sample-one-episode-per-checkpoint-unit",
        ]
    )
    records: list[dict[str, Any]] = []
    outputs: dict[int, Path] = {}
    baseline_wall: float | None = None
    for workers in args.benchmark_workers:
        uid = f"{args.output_dataset_uid}_benchmark_w{workers}"
        final = args.output_root / uid
        local_root = args.local_work_root / "benchmarks" / uid
        work = local_root / ".conversion_work" / uid / "benchmark"
        resume = local_root / ".conversion_resume" / uid
        logs = local_root / ".conversion_logs" / uid
        temp = work / "tmp"
        # Sampling the whole staging root turns every 0.2-second probe into a
        # TB-scale OSSFS walk. Child paths are explicit, so their complete
        # final/work/resume/log footprint can be sampled in bounded time.
        sampler = ProcessTreeSampler((final, work, resume, logs))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_argv,
            "--benchmark-child",
            "--output-dataset-uid",
            uid,
            "--local-work-root",
            str(local_root),
            "--work-dir",
            str(work),
            "--resume-dir",
            str(resume),
            "--logs-dir",
            str(logs),
            "--temp-dir",
            str(temp),
            "--workers",
            str(workers),
        ]
        sampler.start()
        completed = subprocess.run(command)
        metrics = sampler.stop()
        if completed.returncode != 0:
            raise ConversionError(
                f"benchmark child for {workers} worker(s) exited {completed.returncode}"
            )
        manifest = read_json_object(final / "collection_manifest.json", "benchmark manifest")
        total_frames = sum(int(partition["frames"]) for partition in manifest["partitions"])
        if baseline_wall is None:
            baseline_wall = metrics.wall_seconds
        record = {
            "workers": workers,
            "encoder_threads_per_worker": _parallel_threads(args),
            "total_encoder_threads": workers * _parallel_threads(args),
            "wall_seconds": metrics.wall_seconds,
            "frames": total_frames,
            "frames_per_second": total_frames / metrics.wall_seconds,
            "speedup": baseline_wall / metrics.wall_seconds,
            "cpu_seconds": metrics.cpu_seconds,
            "average_cpu_cores": metrics.average_cpu_cores,
            "average_cpu_percent": metrics.average_cpu_cores * 100.0,
            "peak_rss_bytes": metrics.peak_rss_bytes,
            "read_bytes": metrics.read_bytes,
            "write_bytes": metrics.write_bytes,
            "read_chars": metrics.read_chars,
            "write_chars": metrics.write_chars,
            "peak_temp_bytes": metrics.peak_temp_bytes,
            "io_counters_available": metrics.io_counters_available,
        }
        records.append(record)
        outputs[workers] = final

    reference_workers = args.benchmark_workers[0]
    reference = outputs[reference_workers]
    equivalence: dict[str, Any] = {}
    invalid_workers: dict[int, str] = {}
    reference_collection = _semantic_collection_manifest(
        reference / "collection_manifest.json"
    )
    for workers in args.benchmark_workers[1:]:
        candidate = outputs[workers]
        key = f"{reference_workers}_vs_{workers}"
        comparison: dict[str, Any] = {"equivalent": True, "partitions": {}}
        if _semantic_collection_manifest(candidate / "collection_manifest.json") != reference_collection:
            comparison = {
                "equivalent": False,
                "error": (
                    f"collection manifest differs for {reference_workers} and "
                    f"{workers} workers"
                ),
                "partitions": {},
            }
            invalid_workers[workers] = str(comparison["error"])
            equivalence[key] = comparison
            continue
        for partition in sorted(path.name for path in reference.iterdir() if path.is_dir()):
            try:
                partition_report = verify_lerobot_equivalence(
                    reference / partition,
                    candidate / partition,
                    compare_video_frames=True,
                    storage_layout_independent=True,
                )
            except (ConversionError, OSError, RuntimeError, ValueError) as exc:
                comparison["equivalent"] = False
                comparison["partitions"][partition] = {
                    "equivalent": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                invalid_workers[workers] = f"{partition}: {type(exc).__name__}: {exc}"
            else:
                comparison["partitions"][partition] = {
                    "equivalent": True,
                    **partition_report.as_dict(),
                }
        equivalence[key] = comparison

    for record in records:
        workers = int(record["workers"])
        record["equivalence_to_baseline"] = (
            "baseline" if workers == reference_workers else
            "failed" if workers in invalid_workers else "passed"
        )
        record["valid_for_formal_conversion"] = workers not in invalid_workers

    valid_records = [
        row for row in records if bool(row["valid_for_formal_conversion"])
    ]
    fastest = max(valid_records, key=lambda row: float(row["frames_per_second"]))
    rejected = [
        {"workers": workers, "reason": reason}
        for workers, reason in sorted(invalid_workers.items())
    ]
    if invalid_workers:
        recommendation_reason = (
            "multi-worker output failed exact equivalence; formal conversion is restricted "
            "to the 1-worker baseline"
        )
    elif int(fastest["workers"]) == reference_workers:
        recommendation_reason = (
            "no equivalent multi-worker configuration produced a positive end-to-end speedup"
        )
    else:
        recommendation_reason = (
            "fastest equivalent configuration in this run; repeat the benchmark before "
            "treating the speedup as reproducible"
        )
    report = {
        "schema_version": 1,
        "sample": {
            "versions": args.version or list(VERSIONS),
            "checkpoint_units": sample_units,
            "episodes_per_checkpoint_unit": 1,
            "selection_reason": (
                "v2 token blocks can cross episode boundaries; sampling one first episode "
                "from each decoder-context checkpoint preserves those boundaries before "
                "empirical worker-count equivalence checking"
            ),
            "video_codec": args.video_codec,
            "video_quality": args.video_quality,
            "video_preset": args.video_preset,
        },
        "runs": records,
        "equivalence": equivalence,
        "bottleneck_assessment": {
            "formal_scaling_blocker": (
                "official bfloat16 Cosmos decoder output changes with worker/GPU process "
                "context, violating exact output equivalence"
            ),
            "resource_interpretation": (
                "inspect average_cpu_cores, physical read/write bytes, and peak temp in runs; "
                "the observed W1 profile is not CPU- or physical-I/O-saturated, while W2/W4 "
                "parallelize decoder/encoder/write work but are semantically invalid"
            ),
        },
        "parallel_eligible": not invalid_workers and int(fastest["workers"]) > 1,
        "rejected_worker_counts": rejected,
        "formal_conversion_allowed_workers": [
            int(row["workers"]) for row in valid_records
        ],
        "fastest": {
            "workers": fastest["workers"],
            "encoder_threads_per_worker": fastest["encoder_threads_per_worker"],
            "frames_per_second": fastest["frames_per_second"],
        },
        "recommendation": {
            "workers": int(fastest["workers"]),
            "encoder_threads_per_worker": int(
                fastest["encoder_threads_per_worker"]
            ),
            "reason": recommendation_reason,
        },
    }
    report_path = args.benchmark_report or (
        args.output_root / "benchmarks" / f"{args.output_dataset_uid}_workers.json"
    )
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"benchmark report: {report_path}")
    if invalid_workers:
        print(
            "benchmark rejected multi-worker formal conversion because exact equivalence failed",
            file=sys.stderr,
        )
    return 0


def _select(
    plan,
    *,
    max_episodes: int | None,
    max_units: int | None,
    one_episode_per_unit: bool = False,
):
    episodes = list(plan.episodes)
    if max_units is not None:
        units: list[str] = []
        selected = []
        for episode in episodes:
            unit = str(episode.extra["checkpoint_unit"])
            if unit not in units:
                if len(units) >= max_units:
                    break
                units.append(unit)
            if not one_episode_per_unit or not selected or unit != str(
                selected[-1].extra["checkpoint_unit"]
            ):
                selected.append(episode)
        episodes = selected
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ConversionError("selection produced no episodes")
    return replace(plan, episodes=tuple(episodes))


def _plans(args: argparse.Namespace, workspace: Path) -> tuple[OneXWorldModelReader, list[Any]]:
    config = load_dataset_config(args.config)
    if config.format != "one_x_world_model":
        raise ConversionError(f"expected format=one_x_world_model in {args.config}")
    reader = OneXWorldModelReader()
    plans = []
    versions = args.version or list(VERSIONS)
    for version in versions:
        suffix = version.replace(".", "_")
        partition_name = suffix
        dataset_uid = f"{args.output_dataset_uid}_{suffix}"
        splits = [f"train_{version}", f"val_{version}"]
        version_config = config.model_copy(
            update={
                "dataset_uid": dataset_uid,
                "one_x_version": version,
                "one_x_splits": splits,
                "one_x_v1_decoder_repo": str(args.v1_decoder_repo) if args.v1_decoder_repo else None,
                "one_x_cosmos_decoder_path": (
                    str(args.cosmos_decoder_path) if args.cosmos_decoder_path else None
                ),
                "one_x_decode_batch_size": args.decode_batch_size,
                "one_x_v1_postprocess_device": args.v1_postprocess_device,
                "one_x_decoder_cpu_threads": args.decoder_cpu_threads,
                "one_x_v1_checkpoint_segments": args.v1_checkpoint_segments,
            }
        )
        plan = reader.build_plan(version_config, args.raw_root, args.staging_root)
        plan = replace(plan, output_path=workspace / partition_name)
        plan = _select(
            plan,
            max_episodes=args.max_episodes,
            max_units=args.max_checkpoint_units,
            one_episode_per_unit=args.sample_one_episode_per_checkpoint_unit,
        )
        video_encoding = dict(plan.extra["video_encoding"])
        video_encoding.update(
            {
                "target_codec": args.video_codec,
                "target_quality": args.video_quality,
                "target_preset": _preset(args.video_codec, args.video_preset),
                "target_pix_fmt": "yuv420p",
                "streaming": True,
                "video_reencoded": True,
                # Even qp/crf=0 is not claimed pixel-lossless here: the RGB to
                # yuv420p conversion subsamples chroma and the container output
                # is validated semantically, not by pixel identity.
                "video_reencoding_lossy": True,
            }
        )
        plan = replace(
            plan,
            extra={
                **plan.extra,
                "converter": "convert_1x_world_model_dataset.py",
                "video_encoding": video_encoding,
            },
        )
        plans.append(plan)
    return reader, plans


def _collection_payload(plans: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    options = {
        "video_codec": args.video_codec,
        "video_quality": args.video_quality,
        "video_preset": _preset(args.video_codec, args.video_preset),
        "decode_batch_size": args.decode_batch_size,
        "encoder_queue_maxsize": args.encoder_queue_maxsize,
        "encoder_threads": args.encoder_threads,
    }
    if args.workers is not None:
        options.update(
            {
                "encoder_threads_per_worker": _parallel_threads(args),
                "blocking_streaming_encoding": True,
                "fragmented_mp4_writes": True,
                "encoder_queue_maxsize": _parallel_queue_size(args),
                "parallel_schema_version": 1,
                "direct_final_chunks": True,
                "workers": args.workers,
            }
        )
    options["storage"] = {
        "local_work_root": str(args.local_work_root),
        "max_local_temp_bytes": args.max_local_temp_bytes,
        "min_local_free_bytes": args.min_local_free_bytes,
        "upload_workers": args.upload_workers,
        "max_upload_queue_units": args.max_upload_queue_units,
        "max_inflight_bytes": args.max_inflight_bytes,
        "max_inflight_units": args.max_inflight_units,
        "storage_check_interval_seconds": args.storage_check_interval_seconds,
    }
    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "kind": "1x_world_model_collection",
        "output_dataset_uid": args.output_dataset_uid,
        "partitions": [
            build_resume_payload(
                plan,
                reader_format="one_x_world_model",
                conversion_options=options,
            )
            for plan in plans
        ],
        "options": options,
    }


def _prepare_collection_resume(
    data_root: Path,
    state_root: Path,
    payload: dict[str, Any],
) -> None:
    state_path = state_root / "collection.json"
    fingerprint = canonical_fingerprint(payload)
    if state_path.exists():
        state = read_json_object(state_path, "collection resume state")
        if state.get("fingerprint") != fingerprint:
            raise ConversionError(
                "collection resume fingerprint changed; use the original source/selection/decoder/"
                f"encoder arguments or move {data_root} and {state_root} aside"
            )
        return
    if data_root.exists():
        raise ConversionError(f"collection resume data exists without state: {data_root}")
    data_root.mkdir(parents=True)
    atomic_write_json(
        state_path,
        {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "configuration": payload,
        },
    )


def _collection_manifest(plans: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    manifest = {
        "format": "lerobot_v3_0_collection",
        "dataset_uid": args.output_dataset_uid,
        "source_dataset": "1x-technologies/worldmodel",
        "source_revision": plans[0].extra["source_revision"],
        "partitions": [
            {
                "name": plan.output_path.name,
                "dataset_uid": plan.dataset_uid,
                "source_version": plan.extra["source_version"],
                "source_splits": plan.extra["source_splits"],
                "episodes": len(plan.episodes),
                "frames": plan.num_frames,
                "features": plan.feature_schema(),
            }
            for plan in plans
        ],
        "partition_reason": (
            "v1.1 and v2.0 have different robot fields, tokenizers, and feature schemas; "
            "zero padding or field fabrication is forbidden"
        ),
        "excluded": plans[0].extra["unsupported_source_components"],
        "task_mapping": {
            "0": "Unspecified task; the source dataset provides no instruction."
        },
        "video_encoding": [plan.extra["video_encoding"] for plan in plans],
    }
    if args.workers is not None:
        manifest["parallel"] = {
            "workers": args.workers,
            "encoder_threads_per_worker": _parallel_threads(args),
            "total_encoder_threads": args.workers * _parallel_threads(args),
            "encoder_queue_maxsize": _parallel_queue_size(args),
            "deterministic_aggregation": True,
            "deterministic_cuda_decoding": True,
        }
    return manifest


def _option_was_explicit(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def _uses_legacy_collection_workflow(
    raw_argv: Sequence[str], args: argparse.Namespace
) -> bool:
    """Keep the old serial path only for callers of the deprecated CLI alias.

    Production defaults and every worker-based run use the marker-published
    direct-commit workflow.  The narrow compatibility branch preserves
    existing integrations that explicitly pass ``--staging-root`` and rely on
    the generic serial writer's sibling checkpoint layout.
    """

    new_options = (
        "--output-root",
        "--local-work-root",
        "--work-dir",
        "--resume-dir",
        "--logs-dir",
        "--temp-dir",
        "--max-staging-bytes",
        "--max-local-temp-bytes",
        "--min-local-free-bytes",
        "--upload-workers",
        "--max-inflight-bytes",
        "--max-inflight-units",
        "--storage-check-interval-seconds",
    )
    return (
        args.staging_root is not None
        and args.workers is None
        and not args.benchmark_child
        and not any(_option_was_explicit(raw_argv, option) for option in new_options)
    )


def _resolve_output_root(
    parser: argparse.ArgumentParser,
    raw_argv: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    if args.staging_root is not None:
        if _option_was_explicit(raw_argv, "--output-root"):
            parser.error("--staging-root and --output-root are mutually exclusive")
        return args.staging_root / "lerobot_v3_0"
    return args.output_root


def _resume_state_path(_output_root: Path, args: argparse.Namespace) -> Path:
    resume = args.resume_dir or (
        args.local_work_root / ".conversion_resume" / args.output_dataset_uid
    )
    return resume / "collection.json"


def _select_run_id(output_root: Path, args: argparse.Namespace) -> str:
    state_path = _resume_state_path(output_root, args)
    if state_path.is_file():
        state = read_json_object(state_path, "direct collection resume state")
        runtime = state.get("runtime")
        if not isinstance(runtime, dict) or not isinstance(runtime.get("run_id"), str):
            raise ConversionError(f"resume state has no run_id: {state_path}")
        return str(runtime["run_id"])
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]


def _prepare_direct_collection_state(
    layout: StagingLayout,
    payload: dict[str, Any],
) -> str:
    state_path = layout.resume / "collection.json"
    fingerprint = canonical_fingerprint(payload)
    expected_runtime = layout.as_dict()
    if state_path.exists():
        state = read_json_object(state_path, "direct collection resume state")
        if state.get("fingerprint") != fingerprint:
            raise ConversionError(
                "collection resume fingerprint changed; use the original source/selection/"
                f"decoder/encoder arguments or move {layout.resume} aside"
            )
        if state.get("runtime") != expected_runtime:
            raise ConversionError(
                f"runtime layout changed for resume state {state_path}; use the original "
                "--work-dir/--resume-dir/--logs-dir/--temp-dir arguments"
            )
        return fingerprint
    layout.resume.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        state_path,
        {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "configuration": payload,
            "runtime": expected_runtime,
        },
    )
    return fingerprint


def _write_run_record(
    layout: StagingLayout,
    *,
    status: str,
    detail: Mapping[str, Any],
) -> None:
    atomic_write_json(
        layout.logs / f"{layout.run_id}.json",
        {
            "schema_version": 1,
            "run_id": layout.run_id,
            "status": status,
            "updated_unix": time.time(),
            "layout": layout.as_dict(),
            "environment": dict(layout.environment),
            **dict(detail),
        },
    )


def _run_legacy_collection(
    args: argparse.Namespace,
    *,
    devices: tuple[str, ...],
) -> int:
    """Original small/serial workflow retained for explicit legacy callers."""

    assert args.staging_root is not None
    final = args.staging_root / "lerobot_v3_0" / args.output_dataset_uid
    if final.exists():
        if args.skip_existing:
            print(f"skipped existing collection: {final}")
            return 0
        if not args.overwrite:
            print(f"error: output already exists: {final}", file=sys.stderr)
            return 1

    if args.resume:
        workspace, state_root, lock_path = resume_paths(final)
    else:
        workspace = final.with_name(f".{final.name}.incomplete-{uuid.uuid4().hex}")
        state_root = (
            workspace.with_name(f"{workspace.name}.parallel-state")
            if args.workers is not None
            else None
        )
        lock_path = None
    try:
        reader, plans = _plans(args, workspace)
        print(
            json.dumps(
                {
                    "collection": str(final),
                    "partitions": [plan_summary(plan) for plan in plans],
                },
                indent=2,
            )
        )
        if args.dry_run:
            print("inspect-only complete; no decoder loaded and no output written")
            return 0

        payload = _collection_payload(plans, args)
        encoder = _rgb_encoder(args.video_codec, args.video_quality, args.video_preset)
        for plan in plans:
            reader.preflight_decoder(plan)
            _preflight_encoder(plan, encoder)
        longest = max(episode.num_frames for plan in plans for episode in plan.episodes)
        queue_size = args.encoder_queue_maxsize or longest + 1
        if queue_size <= longest:
            raise ConversionError(
                f"streaming queue {queue_size} must exceed longest episode ({longest}) "
                "so frames cannot drop"
            )
        lock_context = exclusive_resume_lock(lock_path) if args.resume else _nullcontext()
        with lock_context:
            if args.resume:
                assert state_root is not None
                _prepare_collection_resume(workspace, state_root, payload)
            else:
                workspace.mkdir(parents=True, exist_ok=False)
            for plan in plans:
                if plan.output_path.exists():
                    validate_written_dataset(plan, plan.output_path)
                    validate_video_files(
                        plan, plan.output_path, expected_frames=plan.num_frames
                    )
                    print(
                        f"[{plan.dataset_uid}] reused completed collection partition",
                        flush=True,
                    )
                    continue
                convert_dataset(
                    plan,
                    lambda episode, _plan=plan: reader.iter_frames(_plan, episode),
                    reader_format="one_x_world_model",
                    resume=args.resume,
                    eta_interval_seconds=args.eta_interval_seconds,
                    rgb_encoder=encoder,
                    streaming_encoding=True,
                    encoder_queue_maxsize=queue_size,
                    encoder_threads=args.encoder_threads,
                    conversion_options=payload["options"],
                )
            for plan in plans:
                validate_written_dataset(plan, plan.output_path)
                validate_video_files(plan, plan.output_path, expected_frames=plan.num_frames)
            atomic_write_json(
                workspace / "collection_manifest.json",
                _collection_manifest(plans, args),
            )
            publish_temporary_output(workspace, final, overwrite=args.overwrite)
            if state_root is not None:
                shutil.rmtree(state_root)
        if args.resume and lock_path is not None:
            lock_path.unlink(missing_ok=True)
        print(f"validated and published collection: {final}")
        return 0
    except KeyboardInterrupt:
        print("interrupted; the last verified resume unit was retained", file=sys.stderr)
        return 130
    except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        if not args.resume and workspace.exists():
            shutil.rmtree(workspace)
        if not args.resume and state_root is not None and state_root.exists():
            shutil.rmtree(state_root)
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_staging_collection(
    args: argparse.Namespace,
    *,
    output_root: Path,
    devices: tuple[str, ...],
) -> int:
    assert args.workers is not None
    previous_environment = {
        key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS
    }
    previous_tempdir = tempfile.tempdir
    try:
        validate_source_and_output_roots(args.raw_root, output_root)
        run_id = _select_run_id(output_root, args)
        layout = make_staging_layout(
            output_root=output_root,
            local_work_root=args.local_work_root,
            dataset_uid=args.output_dataset_uid,
            run_id=run_id,
            work_dir=args.work_dir,
            resume_dir=args.resume_dir,
            logs_dir=args.logs_dir,
            temp_dir=args.temp_dir,
        )
        configure_runtime_environment(layout, create=False)
        validate_no_runtime_paths_outside_root(layout)
        # Reader output paths are replaced immediately by _plans; this keeps
        # its legacy constructor contract without changing parsing semantics.
        args.staging_root = output_root.parent
        reader, plans = _plans(args, layout.final)
        print(
            json.dumps(
                {
                    "collection": str(layout.final),
                    "runtime": layout.as_dict(),
                    "partitions": [plan_summary(plan) for plan in plans],
                },
                indent=2,
            )
        )
        if args.dry_run:
            print("inspect-only complete; no decoder loaded and no output written")
            return 0
        if (layout.final / "_SUCCESS").is_file():
            if args.skip_existing:
                print(f"skipped valid published collection: {layout.final}")
                return 0
            if not args.overwrite:
                raise FileExistsError(
                    f"valid published output already exists: {layout.final}"
                )
        layout.create_runtime_directories()
        configure_runtime_environment(layout, create=True)
        validate_no_runtime_paths_outside_root(layout)

        payload = _collection_payload(plans, args)
        source_fingerprint = canonical_fingerprint(payload)
        lock_acquired = False
        with exclusive_staging_lock(layout.lock):
            lock_acquired = True
            if args.overwrite:
                if layout.final.exists():
                    shutil.rmtree(layout.final)
                if layout.resume.exists():
                    shutil.rmtree(layout.resume)
                layout.resume.mkdir(parents=True)
            success = layout.final / "_SUCCESS"
            incomplete = layout.final / "_INCOMPLETE"
            if success.is_file():
                if args.skip_existing:
                    print(f"skipped valid published collection: {layout.final}")
                    return 0
                raise FileExistsError(f"valid published output already exists: {layout.final}")
            if layout.final.exists() and not incomplete.is_file():
                raise ConversionError(
                    f"existing output has neither _SUCCESS nor _INCOMPLETE: {layout.final}"
                )
            if incomplete.is_file() and not args.resume and not args.overwrite:
                raise ConversionError(
                    f"incomplete output exists: {layout.final}; rerun with --resume"
                )
            fingerprint = _prepare_direct_collection_state(layout, payload)
            create_incomplete_output(
                layout.final, fingerprint=fingerprint, run_id=layout.run_id
            )
            _write_run_record(
                layout,
                status="running",
                detail={"fingerprint": fingerprint, "workers": args.workers},
            )
            capacity = StagingCapacityGuard(
                layout.local_root,
                max_staging_bytes=args.max_local_temp_bytes,
                min_free_bytes=args.min_local_free_bytes,
                interval_seconds=args.storage_check_interval_seconds,
                usage_roots=(
                    layout.work,
                    layout.resume,
                    layout.logs,
                    layout.cache_root,
                ),
            )
            capacity.check("conversion startup")
            for plan in plans:
                _convert_parallel_partition(
                    plan,
                    layout.work,
                    layout.resume,
                    args,
                    payload["options"],
                    devices,
                    capacity,
                )
            # Each unit was fully validated on local disk before upload.
            # Finalization validates metadata and consumes the per-object
            # size/range/container evidence; do not decode all remote videos.
            if canonical_fingerprint(_collection_payload(plans, args)) != source_fingerprint:
                raise ConversionError("source files changed during conversion")
            atomic_write_json(
                layout.final / "collection_manifest.json",
                _collection_manifest(plans, args),
            )
            validate_no_runtime_paths_outside_root(layout)
            capacity.check("final publication")
            publish_success(
                layout.final,
                fingerprint=fingerprint,
                evidence={
                    "dataset_uid": args.output_dataset_uid,
                    "partitions": [plan.output_path.name for plan in plans],
                    "workers": args.workers,
                },
            )
            _write_run_record(
                layout,
                status="succeeded",
                detail={"fingerprint": fingerprint, "workers": args.workers},
            )
            if layout.work.exists():
                shutil.rmtree(layout.work)
            try:
                layout.work.parent.rmdir()
            except OSError:
                pass
            unit_metadata = layout.resume / "unit_metadata"
            if unit_metadata.exists():
                shutil.rmtree(unit_metadata)
            atomic_write_json(
                layout.resume / "completed.json",
                {
                    "schema_version": 1,
                    "fingerprint": fingerprint,
                    "published_output": str(layout.final),
                    "published_unix": time.time(),
                },
            )
        layout.lock.unlink(missing_ok=True)
        print(f"validated and marker-published collection: {layout.final}")
        return 0
    except KeyboardInterrupt:
        if "lock_acquired" in locals() and lock_acquired:
            layout.lock.unlink(missing_ok=True)
        if "layout" in locals() and layout.logs.exists():
            _write_run_record(layout, status="interrupted", detail={})
        print("interrupted; verified work-unit checkpoints were retained", file=sys.stderr)
        return 130
    except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        if "lock_acquired" in locals() and lock_acquired:
            layout.lock.unlink(missing_ok=True)
        if "layout" in locals() and layout.logs.exists():
            _write_run_record(
                layout,
                status="failed",
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        # ``main()`` is also exercised in-process by callers and tests.  Keep
        # the required staging-only environment for the whole conversion, but
        # do not leave tempfile or cache variables pointing at a work tree
        # that successful publication just removed.
        for key, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        tempfile.tempdir = previous_tempdir


def _build_parser() -> argparse.ArgumentParser:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=script_root / "configs" / "1x_world_model_dataset.yaml",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/data/embodied_datasets/public_datasets_raw"),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        help=(
            "deprecated compatibility alias: output root becomes "
            "<staging-root>/lerobot_v3_0"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--local-work-root", type=Path, default=DEFAULT_LOCAL_WORK_ROOT
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--output-dataset-uid", default="1x_world_model_dataset")
    parser.add_argument("--version", action="append", choices=VERSIONS)
    parser.add_argument("--dry-run", "--inspect-only", dest="dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=_positive_int)
    parser.add_argument("--max-checkpoint-units", type=_positive_int)
    parser.add_argument(
        "--sample-one-episode-per-checkpoint-unit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--v1-decoder-repo", type=Path)
    parser.add_argument("--cosmos-decoder-path", type=Path)
    parser.add_argument("--decode-batch-size", type=_positive_int, default=8)
    parser.add_argument(
        "--v1-postprocess-device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="device for v1 output clamp/uint8/NHWC conversion",
    )
    parser.add_argument(
        "--decoder-cpu-threads",
        type=_positive_int,
        help="PyTorch intra-op CPU threads per conversion worker",
    )
    parser.add_argument("--v1-checkpoint-segments", type=_positive_int, default=128)
    parser.add_argument("--eta-interval-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--video-codec", choices=VIDEO_CODECS, default="libsvtav1")
    parser.add_argument("--video-quality", type=int, choices=range(52), default=30)
    parser.add_argument("--video-preset")
    parser.add_argument("--encoder-queue-maxsize", type=_positive_int)
    parser.add_argument("--encoder-threads", type=_positive_int)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        help="spawn isolated deterministic conversion workers; omitted preserves legacy serial mode",
    )
    parser.add_argument("--encoder-threads-per-worker", type=_positive_int)
    parser.add_argument("--upload-workers", type=_positive_int, default=1)
    parser.add_argument("--max-upload-queue-units", type=_positive_int, default=2)
    parser.add_argument(
        "--benchmark-workers",
        type=_positive_int,
        nargs="+",
        help="run identical subset conversions for each worker count and verify equivalence",
    )
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--benchmark-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--inflight-memory-budget-gb", type=_positive_float, default=64.0
    )
    parser.add_argument(
        "--inflight-temp-budget-gb", type=_positive_float, default=800.0
    )
    parser.add_argument(
        "--max-local-temp-bytes",
        "--max-staging-bytes",
        dest="max_local_temp_bytes",
        type=_positive_int,
        default=DEFAULT_MAX_LOCAL_TEMP_BYTES,
    )
    parser.add_argument(
        "--min-local-free-bytes",
        type=_positive_int,
        default=DEFAULT_MIN_LOCAL_FREE_BYTES,
    )
    parser.add_argument(
        "--max-inflight-bytes",
        type=_positive_int,
        default=64 * GIB,
    )
    parser.add_argument("--max-inflight-units", type=_positive_int, default=4)
    parser.add_argument(
        "--storage-check-interval-seconds",
        type=_positive_float,
        default=10.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    if (
        Path(args.output_dataset_uid).name != args.output_dataset_uid
        or args.output_dataset_uid in {"", ".", ".."}
    ):
        parser.error("--output-dataset-uid must be one safe path component")
    if sum(bool(value) for value in (args.resume, args.skip_existing, args.overwrite)) > 1:
        parser.error("--resume, --skip-existing, and --overwrite are mutually exclusive")
    if args.workers is not None and args.benchmark_workers:
        parser.error("--workers and --benchmark-workers are mutually exclusive")
    if args.benchmark_child and args.benchmark_workers:
        parser.error("internal --benchmark-child cannot be combined with --benchmark-workers")
    if (
        args.encoder_threads is not None
        and args.encoder_threads_per_worker is not None
    ):
        parser.error("use only one of --encoder-threads and --encoder-threads-per-worker")
    if args.encoder_threads_per_worker is not None and not (
        args.workers is not None or args.benchmark_workers
    ):
        parser.error("--encoder-threads-per-worker requires --workers or --benchmark-workers")
    if args.video_codec in {"h264_nvenc", "hevc_nvenc"}:
        parser.error("NVENC is unsupported on all four A800 GPUs; use CPU h264/hevc/libsvtav1")
    if (
        args.sample_one_episode_per_checkpoint_unit
        and args.max_checkpoint_units is None
    ):
        parser.error("internal checkpoint sampling requires --max-checkpoint-units")
    subset = args.max_episodes is not None or args.max_checkpoint_units is not None
    if args.benchmark_child and not subset:
        parser.error("internal --benchmark-child requires a bounded subset")
    legacy_workflow = _uses_legacy_collection_workflow(raw_argv, args)
    output_root = _resolve_output_root(parser, raw_argv, args)
    args.output_root = output_root
    if not legacy_workflow and args.workers is None and not args.benchmark_workers:
        # The scalable path always uses isolated work units, including W1. It
        # must never silently fall back to the legacy serial writer.
        args.workers = 1
    if args.workers is not None and args.workers > 1 and not args.benchmark_child:
        parser.error(
            "--workers > 1 is disabled for formal conversion because real-sample exact "
            "equivalence failed; use --benchmark-workers 1 2 4 on a bounded subset for "
            "diagnostics, or use --workers 1"
        )
    if subset and args.output_dataset_uid == "1x_world_model_dataset" and not args.dry_run:
        parser.error("smoke/subset conversion requires an independent --output-dataset-uid")
    requested_workers = (
        list(args.benchmark_workers) if args.benchmark_workers else [args.workers]
    )
    requested_workers = [value for value in requested_workers if value is not None]
    if requested_workers:
        if len(requested_workers) != len(set(requested_workers)):
            parser.error("worker counts must be unique")
        threads = _parallel_threads(args)
        for workers in requested_workers:
            if workers > args.max_inflight_units:
                parser.error(
                    f"{workers} workers exceeds --max-inflight-units="
                    f"{args.max_inflight_units}; increase the explicit bound"
                )
            if workers * threads > MAX_TOTAL_ENCODER_THREADS:
                parser.error(
                    f"{workers} workers x {threads} encoder threads exceeds the "
                    f"{MAX_TOTAL_ENCODER_THREADS}-thread limit"
                )
        if args.dry_run:
            devices = ()
        else:
            try:
                devices = _visible_cuda_devices()
            except ConversionError as exc:
                parser.error(str(exc))
            if max(requested_workers) > len(devices):
                parser.error(
                    f"requested {max(requested_workers)} workers but only "
                    f"{len(devices)} CUDA device(s) are visible"
                )
    else:
        devices = ()
    if args.benchmark_workers:
        if not subset:
            parser.error("--benchmark-workers requires --max-episodes or --max-checkpoint-units")
        if args.dry_run:
            parser.error("--benchmark-workers cannot be combined with --dry-run")
        if args.benchmark_workers[0] != 1:
            parser.error("--benchmark-workers must start with the 1-worker baseline")
        try:
            return _run_worker_benchmarks(raw_argv, args)
        except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if legacy_workflow:
        return _run_legacy_collection(args, devices=devices)
    return _run_staging_collection(args, output_root=output_root, devices=devices)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, _type, _value, _traceback):
        return False


if __name__ == "__main__":
    previous = signal.getsignal(signal.SIGTERM)

    def _sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)
    try:
        sys.exit(main())
    finally:
        signal.signal(signal.SIGTERM, previous)
