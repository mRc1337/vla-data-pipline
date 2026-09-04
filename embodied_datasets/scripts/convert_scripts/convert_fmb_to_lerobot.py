"""Bounded, resumable FMB ``.npy-in-ZIP`` to LeRobot v3.0 conversion.

The coordinator freezes the ZIP index and all global indices before dispatch.
Workers write one complete LeRobot unit on local POSIX staging; verified bulk
files are copied to their preassigned OSS-final paths and deleted locally only
after size/footer/header/sample validation.  Finalisation writes metadata and
stats only and publishes ``_SUCCESS`` last.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterator, Sequence

sys.dont_write_bytecode = True

from convert_core.checkpoint import atomic_write_json, canonical_fingerprint, read_json_object
from convert_core.direct_commit import (
    DirectCommitUploader,
    finalize_direct_partition,
    globalize_unit_data_files,
    prepare_direct_commits,
)
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import (
    validate_video_files,
    validate_written_dataset,
    write_dataset,
)
from convert_core.parallel import (
    ParallelWorkUnit,
    isolated_unit_plan,
    prepare_work_units,
    run_parallel_work_units,
    split_plan_into_units,
    validate_work_units,
    write_verified_unit_marker,
)
from convert_core.staging import (
    DEFAULT_MAX_LOCAL_TEMP_BYTES,
    DEFAULT_MIN_LOCAL_FREE_BYTES,
    DEFAULT_OUTPUT_ROOT,
    create_incomplete_output,
    exclusive_staging_lock,
    publish_success,
    validate_source_and_output_roots,
    StagingCapacityGuard,
)
from readers.fmb_npy_reader import (
    FmbCatalog,
    FmbPartition,
    catalog_from_payload,
    catalog_to_payload,
    inspect_fmb,
    iter_fmb_frames,
    remap_catalog_to_extracted_root,
    stage_fmb_unit_sources,
    validate_extracted_fmb_root,
    validate_catalog_sources,
)
from convert_core.dataset_config import load_dataset_config


DEFAULT_RAW_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted")
DEFAULT_LOCAL_WORK_ROOT = Path.home() / "functional_manipulation_benchmark_fmb_staging"
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "functional_manipulation_benchmark_fmb.yaml"
DEFAULT_ENCODER_THREADS = 8
COPY_FINGERPRINT_VERSION = 1
FMB_VIDEO_CODEC = "h264"
FMB_VIDEO_PIX_FMT = "yuv420p"
FMB_VIDEO_GOP = 2
FMB_VIDEO_CRF = 30


@dataclass(frozen=True)
class FmbWorkerPayload:
    plan: Any
    raw_root: str
    encoder_threads: int
    conversion_options: dict[str, Any]


def _fmb_rgb_encoder() -> Any:
    from lerobot.configs import RGBEncoderConfig

    return RGBEncoderConfig(
        vcodec=FMB_VIDEO_CODEC,
        pix_fmt=FMB_VIDEO_PIX_FMT,
        g=FMB_VIDEO_GOP,
        crf=FMB_VIDEO_CRF,
    )


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _warmup_frames(value: str) -> int:
    result = _positive(value)
    if not 30 <= result <= 60:
        raise argparse.ArgumentTypeError("warmup frame count must be between 30 and 60")
    return result


def _run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"


def _unit_datasets_cache_root(
    runtime_cache_root: Path, partition_name: str, unit_index: int
) -> Path:
    """Return the private, reproducible HF cache for one conversion unit."""

    if Path(partition_name).name != partition_name:
        raise ConversionError(f"invalid FMB partition name for cache: {partition_name!r}")
    return runtime_cache_root / "units" / partition_name / f"unit-{unit_index:06d}"


@contextmanager
def _fmb_datasets_cache(cache_root: Path) -> Iterator[None]:
    """Route LeRobot/HF parquet intermediates to one unit-owned directory."""

    cache_root.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("VLA_DATASETS_CACHE_ROOT")
    os.environ["VLA_DATASETS_CACHE_ROOT"] = str(cache_root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VLA_DATASETS_CACHE_ROOT", None)
        else:
            os.environ["VLA_DATASETS_CACHE_ROOT"] = previous


def _cleanup_unit_datasets_cache(
    runtime_cache_root: Path, partition_name: str, unit_index: int
) -> None:
    """Delete only a unit cache after its committed marker is durable."""

    cache_root = _unit_datasets_cache_root(runtime_cache_root, partition_name, unit_index)
    if cache_root.exists():
        if cache_root.is_symlink():
            raise ConversionError(f"refusing to remove symlinked FMB cache: {cache_root}")
        shutil.rmtree(cache_root)
        print(f"[cache-cleanup] removed committed unit cache {cache_root}", flush=True)


def _cleanup_legacy_datasets_cache(local_root: Path) -> None:
    """Remove the pre-fix shared cache; preflight/checkpoints are elsewhere."""

    cache_root = local_root / "cache" / "runtime" / "datasets"
    if cache_root.exists():
        if cache_root.is_symlink():
            raise ConversionError(f"refusing to remove symlinked legacy cache: {cache_root}")
        print(f"[cache-cleanup] removing legacy shared cache {cache_root}", flush=True)
        shutil.rmtree(cache_root)


def _plan_payload(plan: Any) -> dict[str, Any]:
    return {
        "dataset_uid": plan.dataset_uid,
        "fps": plan.fps,
        "robot_type": plan.robot_type,
        "features": plan.feature_schema(),
        "episodes": [
            {"uid": e.episode_uid, "source": e.source_relative_path, "task": e.instruction, "frames": e.num_frames, "extra": e.extra}
            for e in plan.episodes
        ],
    }


def _worker(unit: ParallelWorkUnit) -> dict[str, Any]:
    payload = unit.payload
    if not isinstance(payload, FmbWorkerPayload):
        raise ConversionError(f"invalid FMB worker payload: {unit.key}")
    root = Path(unit.target_path)
    root.parent.mkdir(parents=True, exist_ok=True)
    plan = payload.plan
    source_root = root.parent / f".{root.name}.source"
    datasets_cache_root = Path(payload.conversion_options["datasets_cache_root"])
    print(f"[{unit.key}] staging FMB source members to local POSIX: {source_root}", flush=True)
    try:
        stage_fmb_unit_sources(
            plan,
            Path(payload.raw_root),
            source_root,
            progress=lambda message: print(f"[{unit.key}] {message}", flush=True),
        )
        with _fmb_datasets_cache(datasets_cache_root):
            write_dataset(
                plan,
                lambda episode: iter_fmb_frames(plan, episode, source_root),
                root,
                rgb_encoder=_fmb_rgb_encoder(),
                streaming_encoding=True,
                blocking_streaming_encoding=True,
                encoder_queue_maxsize=30,
                encoder_threads=payload.encoder_threads,
                encoder_temp_root=root / "encoder-temp",
                deferred_video_concatenation=True,
                batch_metadata_writes=True,
                fragmented_mp4_writes=False,
            )
            validate_written_dataset(plan, root)
            validate_video_files(plan, root, expected_frames=unit.weight)
        globalize_unit_data_files(unit)
        write_verified_unit_marker(unit)
        print(f"[{unit.key}] local unit verified", flush=True)
        return {"unit": unit.key, "frames": unit.weight, "episodes": unit.episode_end - unit.episode_start}
    finally:
        # Source members are reproducible from raw_root and must not be hashed
        # into the verified output inventory or survive a successful commit.
        shutil.rmtree(source_root, ignore_errors=True)


def _validate_existing_unit(unit: ParallelWorkUnit) -> None:
    payload = unit.payload
    if not isinstance(payload, FmbWorkerPayload):
        raise ConversionError(f"invalid FMB unit payload: {unit.key}")
    root = Path(unit.target_path)
    validate_written_dataset(payload.plan, root)
    validate_video_files(payload.plan, root, expected_frames=unit.weight)


def _make_units(
    partition: FmbPartition,
    *,
    work_root: Path,
    raw_root: Path,
    workers: int,
    encoder_threads: int,
    episodes_per_unit: int,
    fingerprint: str,
    runtime_cache_root: Path,
) -> tuple[ParallelWorkUnit, ...]:
    slices = split_plan_into_units(partition.plan, max_episodes_per_unit=episodes_per_unit)
    units: list[ParallelWorkUnit] = []
    for item in slices:
        target = work_root / "units" / partition.name / f"unit-{item.index:06d}"
        unit_plan = isolated_unit_plan(
            partition.plan,
            item,
            dataset_uid=f"{partition.name}__unit_{item.index:06d}",
            target_path=target,
        )
        unit_fingerprint = canonical_fingerprint({
            "schema": COPY_FINGERPRINT_VERSION,
            "dataset": fingerprint,
            "unit": item.__dict__,
            "plan": _plan_payload(unit_plan),
            "encoder_threads": encoder_threads,
            "video": unit_plan.extra.get("video_encoding"),
        })
        source_bytes = sum(int(episode.extra.get("source_uncompressed_bytes", 0)) for episode in unit_plan.episodes)
        # ZipInfo sizes are uncompressed NumPy payload sizes.  Reserve source
        # plus generated output/encoder headroom before dispatch.
        estimated = max(256 * 1024 * 1024, source_bytes * 2 + 2 * 1024**3)
        units.append(ParallelWorkUnit(
            index=item.index,
            key=item.key,
            dataset_uid=unit_plan.dataset_uid,
            target_path=str(target),
            episode_start=item.episode_start,
            episode_end=item.episode_end,
            frame_start=item.frame_start,
            frame_end=item.frame_end,
            task_indices=item.task_indices,
            weight=item.frame_end - item.frame_start,
            estimated_memory_bytes=max(512 * 1024**2, min(source_bytes, 8 * 1024**3)),
            estimated_temp_bytes=estimated,
            fingerprint=unit_fingerprint,
            payload=FmbWorkerPayload(
                unit_plan,
                str(raw_root),
                encoder_threads,
                {
                    "workers": workers,
                    "datasets_cache_root": str(
                        _unit_datasets_cache_root(runtime_cache_root, partition.name, item.index)
                    ),
                },
            ),
        ))
    validate_work_units(units)
    return tuple(units)


def _select_partition(partition: FmbPartition, args: argparse.Namespace) -> FmbPartition:
    episodes = list(partition.plan.episodes)
    if args.task:
        selected = set(args.task)
        episodes = [episode for episode in episodes if episode.instruction in selected]
        if not episodes:
            raise ConversionError(f"requested FMB tasks are absent from {partition.name}: {sorted(selected)}")
    if args.max_tasks is not None:
        tasks = list(dict.fromkeys(episode.instruction for episode in episodes))[: args.max_tasks]
        episodes = [episode for episode in episodes if episode.instruction in set(tasks)]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise ConversionError(f"empty FMB selection for {partition.name}")
    return FmbPartition(
        partition.name,
        replace(partition.plan, episodes=tuple(episodes)),
        tuple(next(entry for entry in partition.entries if entry.member == episode.extra["member"]) for episode in episodes),
        partition.schema_fingerprint,
    )


def _capacity_guard(local_root: Path, args: argparse.Namespace) -> StagingCapacityGuard:
    local_root.mkdir(parents=True, exist_ok=True)
    return StagingCapacityGuard(
        local_root,
        max_staging_bytes=args.max_local_temp_bytes,
        min_free_bytes=args.min_local_free_bytes,
        interval_seconds=5.0,
        usage_roots=(local_root,),
    )


def _configure_local_runtime(local_root: Path, uid: str, run_id: str) -> None:
    """Keep library temp files and caches inside the bounded local root."""

    temp_root = local_root / "tmp" / uid / run_id
    cache_root = local_root / "cache" / "runtime"
    for path in (
        temp_root,
        cache_root,
        cache_root / "xdg",
        cache_root / "huggingface",
        cache_root / "torch",
        cache_root / "matplotlib",
        cache_root / "datasets",
        cache_root / "cuda",
        cache_root / "torch-extensions",
        cache_root / "numba",
    ):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "TMPDIR": str(temp_root),
            "TMP": str(temp_root),
            "TEMP": str(temp_root),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "TORCH_HOME": str(cache_root / "torch"),
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "VLA_DATASETS_CACHE_ROOT": str(cache_root / "datasets"),
            "CUDA_CACHE_PATH": str(cache_root / "cuda"),
            "TORCH_EXTENSIONS_DIR": str(cache_root / "torch-extensions"),
            "NUMBA_CACHE_DIR": str(cache_root / "numba"),
        }
    )
    tempfile.tempdir = str(temp_root)


def _run_encoder_warmup(
    catalog: FmbCatalog,
    *,
    raw_root: Path,
    local_root: Path,
    fingerprint: str,
    frames: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Encode one bounded local sample, validate it, and remove all output."""

    partition = catalog.partitions[0]
    source_episode = partition.plan.episodes[0]
    if source_episode.num_frames < frames:
        raise ConversionError(
            f"encoder warmup needs {frames} frames but the selected sample has "
            f"only {source_episode.num_frames}"
        )
    warmup_episode = replace(
        source_episode,
        num_frames=frames,
        extra={**source_episode.extra, "allow_frame_prefix": True},
    )
    warmup_uid = f"{partition.name}__encoder_warmup"
    warmup_root = local_root / "warmup" / fingerprint[:24]
    warmup_cache_root = local_root / "cache" / "runtime" / "warmup" / fingerprint[:24]
    warmup_plan = replace(
        partition.plan,
        dataset_uid=warmup_uid,
        output_path=warmup_root,
        episodes=(warmup_episode,),
    )
    guard = _capacity_guard(local_root, args)
    guard.wait_for_capacity(
        "encoder warmup",
        required_additional_bytes=4 * 1024**3,
    )
    started = time.monotonic()
    warmup_source_root = local_root / "warmup" / f"{fingerprint[:24]}.source"
    try:
        warmup_root.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[warmup] staging source member to local POSIX: {warmup_source_root}",
            flush=True,
        )
        stage_fmb_unit_sources(
            warmup_plan,
            raw_root,
            warmup_source_root,
            progress=lambda message: print(f"[warmup] {message}", flush=True),
        )
        with _fmb_datasets_cache(warmup_cache_root):
            write_dataset(
                warmup_plan,
                lambda episode: iter_fmb_frames(warmup_plan, episode, warmup_source_root),
                warmup_root,
                rgb_encoder=_fmb_rgb_encoder(),
                streaming_encoding=True,
                blocking_streaming_encoding=True,
                encoder_queue_maxsize=30,
                encoder_threads=args.encoder_threads_per_worker,
                encoder_temp_root=warmup_root / "encoder-temp",
                batch_metadata_writes=True,
                fragmented_mp4_writes=False,
                deferred_video_concatenation=True,
            )
        validate_written_dataset(warmup_plan, warmup_root)
        validate_video_files(warmup_plan, warmup_root, expected_frames=warmup_plan.num_frames)
        elapsed = max(time.monotonic() - started, 1e-9)
        return {
            "source": warmup_episode.source_relative_path,
            "frames": warmup_plan.num_frames,
            "wall_seconds": elapsed,
            "frames_per_second": warmup_plan.num_frames / elapsed,
            "peak_local_bytes": guard.peak_staging_bytes,
        }
    finally:
        shutil.rmtree(warmup_root, ignore_errors=True)
        shutil.rmtree(warmup_source_root, ignore_errors=True)
        shutil.rmtree(warmup_cache_root, ignore_errors=True)


def _convert_partition(
    partition: FmbPartition,
    *,
    args: argparse.Namespace,
    raw_root: Path,
    output_root: Path,
    local_root: Path,
    work_run_root: Path,
    resume_root: Path,
    fingerprint: str,
) -> dict[str, Any]:
    started = time.monotonic()
    final = output_root / partition.name
    if (final / "_SUCCESS").exists() and not args.resume:
        raise FileExistsError(f"valid FMB partition already exists: {final}")
    final.mkdir(parents=True, exist_ok=True)
    work_root = work_run_root / partition.name
    work_root.mkdir(parents=True, exist_ok=True)
    plan = replace(partition.plan, output_path=final)
    partition = replace(partition, plan=plan)
    runtime_cache_root = local_root / "cache" / "runtime"
    units = _make_units(
        partition,
        work_root=work_root,
        raw_root=raw_root,
        workers=args.workers,
        encoder_threads=args.encoder_threads_per_worker,
        episodes_per_unit=args.episodes_per_unit,
        fingerprint=fingerprint,
        runtime_cache_root=runtime_cache_root,
    )
    for unit in units:
        stale_source_root = Path(unit.target_path).parent / f".{Path(unit.target_path).name}.source"
        if stale_source_root.exists():
            print(f"[resume] removing stale local source staging {stale_source_root}", flush=True)
            shutil.rmtree(stale_source_root)
    guard = _capacity_guard(local_root, args)
    direct = prepare_direct_commits(
        units,
        partition_name=partition.name,
        partition_root=final,
        resume_root=resume_root,
        ossfs_io_timeout_seconds=args.ossfs_io_timeout_seconds,
    )
    for unit in direct.committed:
        _cleanup_unit_datasets_cache(runtime_cache_root, partition.name, unit.index)
    prepared = prepare_work_units(direct.uncommitted, _validate_existing_unit, require_complete_plan=False) if direct.uncommitted else None
    reusable = prepared.reusable if prepared else ()
    pending = prepared.pending if prepared else ()
    uploader = DirectCommitUploader(
        partition_name=partition.name,
        partition_root=final,
        resume_root=resume_root,
        workers=args.upload_workers,
        max_queue_units=args.max_inflight_units,
        on_committed=lambda unit: _cleanup_unit_datasets_cache(
            runtime_cache_root, partition.name, unit.index
        ),
        ossfs_io_timeout_seconds=args.ossfs_io_timeout_seconds,
    )
    try:
        for unit in reusable:
            uploader.submit(unit, trust_verified_marker=False)
        by_index = {unit.index: unit for unit in pending}
        def on_result(result: Any) -> None:
            uploader.submit(by_index[result.index], trust_verified_marker=True)
        def before_dispatch(unit: ParallelWorkUnit, active: tuple[ParallelWorkUnit, ...]) -> None:
            if len(active) + 1 > args.max_inflight_units:
                raise ConversionError(f"--max-inflight-units={args.max_inflight_units} is below active worker frontier")
            # Active workers have already started growing files, so their
            # current bytes are included in the measured usage but their
            # remaining peak must still be reserved before another unit is
            # admitted.  This prevents a concurrent frontier from exceeding
            # the hard local limit between periodic checks.
            active_peak_reservation = sum(item.estimated_temp_bytes for item in active)
            guard.wait_for_capacity(
                f"dispatch {unit.key}",
                required_additional_bytes=active_peak_reservation + unit.estimated_temp_bytes,
                abort_check=uploader.raise_if_failed,
            )
        def health_check() -> None:
            uploader.raise_if_failed()
            guard.periodic_check(f"workers active for {partition.name}")
        completion: tuple[str, ...] = ()
        if pending:
            result = run_parallel_work_units(pending, _worker, workers=args.workers, on_result=on_result, before_dispatch=before_dispatch, health_check=health_check, health_check_interval_seconds=5.0)
            completion = result.completion_order
        upload_stats = uploader.close()
    except BaseException:
        uploader.close(raise_on_failure=False)
        raise
    for unit in units:
        from convert_core.direct_commit import read_committed_unit_marker
        read_committed_unit_marker(unit, partition_name=partition.name, resume_root=resume_root)
    finalize_direct_partition(
        plan,
        units,
        final,
        resume_root=resume_root,
        reader_format="fmb_npy",
        parallel_evidence={"workers": args.workers, "encoder_threads_per_worker": args.encoder_threads_per_worker, "upload_workers": args.upload_workers, "max_local_temp_bytes": args.max_local_temp_bytes, "min_local_free_bytes": args.min_local_free_bytes, "max_inflight_units": args.max_inflight_units, "ossfs_io_timeout_seconds": args.ossfs_io_timeout_seconds, "upload_attempts": 3, "peak_reservation": "active unit estimates plus next dispatch", "runtime_cache_policy": "per-unit datasets cache removed after durable direct commit; committed caches swept on resume", "reused_units": [u.key for u in direct.committed], "worker_completion_order": list(completion), "peak_local_bytes": guard.peak_staging_bytes, "wall_seconds": time.monotonic() - started},
    )
    elapsed = max(time.monotonic() - started, 1e-9)
    return {"partition": partition.name, "episodes": len(plan.episodes), "frames": plan.num_frames, "wall_seconds": elapsed, "frames_per_second": plan.num_frames / elapsed, "upload": upload_stats}


def _catalog_for_run(config: Any, raw_root: Path, cache_root: Path, args: argparse.Namespace) -> FmbCatalog:
    cache = cache_root / "preflight.json"
    cache_scope = {
        # Bump when the catalog/fingerprint semantics change.  In particular,
        # the current schema ignores per-episode frame counts and fixed-width
        # Unicode storage widths when grouping logical feature schemas.
        "schema_version": 3,
        "raw_root": str(raw_root.resolve()),
        "max_shards": args.max_shards,
    }
    if cache.is_file():
        payload = read_json_object(cache, "FMB preflight cache")
        # A complete ZIP preflight can be reused after the same shards are
        # extracted to OSSFS.  Re-scan only member stat/size metadata; avoid
        # reading every large NPY payload again just to rebuild the catalog.
        if (
            payload.get("config_text") == args.config.read_text(encoding="utf-8")
            and payload.get("cache_scope", {}).get("schema_version") == cache_scope["schema_version"]
            and payload.get("cache_scope", {}).get("max_shards") == cache_scope["max_shards"]
            and payload.get("cache_scope", {}).get("raw_root") != cache_scope["raw_root"]
        ):
            previous = catalog_from_payload(payload["catalog"], config, raw_root)
            migrated = remap_catalog_to_extracted_root(previous, config, raw_root)
            if migrated is not None:
                atomic_write_json(
                    cache,
                    {
                        "cache_scope": cache_scope,
                        "config": str(args.config),
                        "config_text": args.config.read_text(encoding="utf-8"),
                        "catalog": catalog_to_payload(migrated),
                    },
                )
                if getattr(args, "trust_preflight_source", False):
                    print("[preflight] trusting complete cached source inventory", flush=True)
                else:
                    validate_catalog_sources(migrated, raw_root)
                return migrated
        if (
            payload.get("config_text") == args.config.read_text(encoding="utf-8")
            and payload.get("cache_scope") == cache_scope
        ):
            catalog = catalog_from_payload(payload["catalog"], config, raw_root)
            if getattr(args, "trust_preflight_source", False):
                print("[preflight] trusting complete cached source inventory", flush=True)
            else:
                validate_catalog_sources(catalog, raw_root)
            return catalog
        # A different shard scope/config is a different preflight identity.
        # Fall through to a fresh scan and replace the cache atomically; never
        # reinterpret a partial catalog as a full source inventory.
    preflight_root = cache_root.parent / ".fmb-preflight-source"
    try:
        catalog = inspect_fmb(
            config,
            raw_root,
            max_episodes=None,
            max_shards=args.max_shards,
            local_preflight_root=preflight_root,
            progress=lambda message: print(f"[preflight] {message}", flush=True),
        )
        atomic_write_json(cache, {"cache_scope": cache_scope, "config": str(args.config), "config_text": args.config.read_text(encoding="utf-8"), "catalog": catalog_to_payload(catalog)})
    finally:
        shutil.rmtree(preflight_root, ignore_errors=True)
    return catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-work-root", type=Path, default=DEFAULT_LOCAL_WORK_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inspect-only", "--dry-run", dest="inspect_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trust-preflight-source",
        action="store_true",
        help="skip repeated OSSFS per-NPY stat validation after a complete preflight",
    )
    parser.add_argument("--max-local-temp-bytes", type=_positive, default=DEFAULT_MAX_LOCAL_TEMP_BYTES)
    parser.add_argument("--min-local-free-bytes", type=_positive, default=DEFAULT_MIN_LOCAL_FREE_BYTES)
    parser.add_argument("--max-inflight-units", type=_positive, default=8)
    parser.add_argument("--workers", type=_positive, default=4)
    parser.add_argument("--encoder-threads-per-worker", type=_positive, default=DEFAULT_ENCODER_THREADS)
    parser.add_argument("--upload-workers", type=_positive, default=1)
    parser.add_argument(
        "--ossfs-io-timeout-seconds",
        type=_positive_float,
        default=900.0,
        help="per-file no-progress timeout for OSSFS copy and validation",
    )
    parser.add_argument("--episodes-per-unit", type=_positive, default=8)
    parser.add_argument("--max-episodes", type=_positive)
    parser.add_argument("--max-tasks", type=_positive)
    parser.add_argument("--max-shards", type=_positive)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--output-dataset-uid")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--benchmark-report", type=Path, help="write conversion timing and resource evidence as JSON")
    parser.add_argument("--warmup-frames", type=_warmup_frames, help="encode and validate one local-only 30-60 frame sample, then delete it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers > args.max_inflight_units:
        raise SystemExit("--workers cannot exceed --max-inflight-units")
    raw_root, output_root = validate_source_and_output_roots(args.raw_root, args.output_root)
    validate_extracted_fmb_root(raw_root)
    _, local_root = validate_source_and_output_roots(raw_root, args.local_work_root)
    _, local_root = validate_source_and_output_roots(output_root, local_root)
    if args.benchmark_report is not None:
        report_path = args.benchmark_report.resolve(strict=False)
        if not report_path.is_relative_to(local_root) or report_path == local_root:
            raise SystemExit("--benchmark-report must be inside --local-work-root")
    config = load_dataset_config(args.config)
    if config.dataset_uid != "functional_manipulation_benchmark_fmb" or config.format != "fmb_npy":
        raise SystemExit("FMB converter requires the functional_manipulation_benchmark_fmb fmb_npy config")
    uid = args.output_dataset_uid or config.dataset_uid
    has_subset = bool(args.max_episodes or args.max_tasks or args.task or args.max_shards)
    if args.warmup_frames and (args.max_episodes or args.max_tasks or args.task):
        raise SystemExit("--warmup-frames cannot be combined with episode/task selection")
    if has_subset and not args.output_dataset_uid and not args.inspect_only:
        raise SystemExit("subset options require --output-dataset-uid so the canonical dataset cannot be overwritten")
    if Path(uid).name != uid:
        raise SystemExit("--output-dataset-uid must be one path component")
    collection_root = output_root / uid
    # Checkpoints and worker bulk are scoped by output UID/run so independent
    # smoke or benchmark datasets cannot collide.  The catalog cache is shared
    # because it is independently fingerprinted by source/config/shard scope.
    logs_root = local_root / "logs" / uid
    cache_root = local_root / "cache"
    for path in (logs_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)
    catalog = _catalog_for_run(config, raw_root, cache_root, args)
    fingerprint = canonical_fingerprint({
        "schema": COPY_FINGERPRINT_VERSION,
        "catalog": catalog.fingerprint_payload,
        "field_mapping": list(catalog.mapping_table),
        "config": args.config.read_text(encoding="utf-8"),
        "fps": 10,
        "video_encoding": {"codec": FMB_VIDEO_CODEC, "pix_fmt": FMB_VIDEO_PIX_FMT, "gop": FMB_VIDEO_GOP, "crf": FMB_VIDEO_CRF, "streaming": True},
        "partition_rules": ["single_object vs multi_object", "schema fingerprint", "archive/member lexical order"],
        "output_uid": uid,
        "encoder_threads": args.encoder_threads_per_worker,
        "episodes_per_unit": args.episodes_per_unit,
        "selection": {"max_episodes": args.max_episodes, "max_tasks": args.max_tasks, "task": list(args.task), "max_shards": args.max_shards},
    })
    run_id = args.run_id or fingerprint[:24]
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be one path component")
    resume_root = local_root / "resume" / uid
    work_run_root = local_root / "work" / uid / run_id
    _configure_local_runtime(local_root, uid, run_id)
    summary = {"dataset_uid": uid, "fingerprint": fingerprint, "partitions": [{"name": p.name, "episodes": len(p.plan.episodes), "frames": p.plan.num_frames, "fps": p.plan.fps, "features": p.plan.feature_schema()} for p in catalog.partitions], "mapping_table": list(catalog.mapping_table), "sample_evidence": list(catalog.sample_evidence)}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if args.inspect_only:
        return 0
    if args.warmup_frames:
        warmup = _run_encoder_warmup(
            catalog,
            raw_root=raw_root,
            local_root=local_root,
            fingerprint=fingerprint,
            frames=args.warmup_frames,
            args=args,
        )
        print(json.dumps({"warmup": warmup}, ensure_ascii=False, indent=2))
        return 0
    if (collection_root / "_SUCCESS").exists() and not args.resume:
        raise SystemExit(f"valid published FMB output exists: {collection_root}; use --resume only for an incomplete run")
    create_incomplete_output(collection_root, fingerprint=fingerprint, run_id=run_id)
    selected = tuple(_select_partition(partition, args) for partition in catalog.partitions)
    with exclusive_staging_lock(local_root / "fmb.lock"):
        # Older runs used one shared HF parquet cache under
        # cache/runtime/datasets.  It is reproducible and is not referenced by
        # resume markers, so reclaim it once the collection lock is held.  New
        # units use private caches and remove them after direct commit.
        _cleanup_legacy_datasets_cache(local_root)
        reports = [_convert_partition(partition, args=args, raw_root=raw_root, output_root=collection_root, local_root=local_root, work_run_root=work_run_root, resume_root=resume_root, fingerprint=fingerprint) for partition in selected]
    manifest = {"schema_version": 1, "dataset_uid": uid, "fingerprint": fingerprint, "source": "Functional Manipulation Benchmark", "partitions": reports, "catalog": catalog.fingerprint_payload}
    atomic_write_json(collection_root / "collection_manifest.json", manifest)
    if args.benchmark_report:
        atomic_write_json(args.benchmark_report, {"schema_version": 1, "dataset_uid": uid, "fingerprint": fingerprint, "workers": args.workers, "encoder_threads_per_worker": args.encoder_threads_per_worker, "partitions": reports})
    publish_success(collection_root, fingerprint=fingerprint, evidence={"partitions": reports, "run_id": run_id})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConversionError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"caused by: {exc.__cause__}", file=sys.stderr)
        raise SystemExit(1)
