#!/usr/bin/env python3
"""Reliably convert the official MimicGen release to LeRobot v3.0.

The release is heterogeneous, so every source HDF5 container becomes one
fixed-schema LeRobot partition below a collection root.  Full conversion is
never implicit: this CLI only writes when neither --inspect-only nor
--dry-run is supplied by the caller.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any, Sequence
import uuid

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
from convert_core.progress import EtaProgress
from readers.robomimic_hdf5_reader import RobomimicPartitionInfo, inspect_partition, iter_frames


SOURCE_REPO = "amandlek/mimicgen_datasets"
SOURCE_REVISION = "33016f8a62c02334f929f2913af8fdd2a8a129e1"
OFFICIAL_CODE_REVISION = "72bd767c255545f462e7ccfb2731f2e5d4c1d9bb"
DEFAULT_RAW_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw")
DEFAULT_STAGING_ROOT = Path("/home/pai/zxw/mimicgen_staging")
DEFAULT_SOURCE_DIRECTORY = "minicgen"
DEFAULT_DATASET_UID = "mimicgen"
DEFAULT_RESUME_PROBE_TIMEOUT_SECONDS = 300.0
CHECKPOINT_VALIDATION_VERSION = 1


class ResumeValidationUnavailable(ConversionError):
    """A checkpoint could not be validated without risking an indefinite wait."""


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


def _rgb_encoder(args: argparse.Namespace) -> Any:
    try:
        from lerobot.configs.video import RGBEncoderConfig
    except ImportError as exc:
        raise RuntimeError("lerobot==0.6.0 is required") from exc
    return RGBEncoderConfig(
        vcodec=args.video_codec,
        crf=args.video_quality,
        preset=args.video_preset,
        pix_fmt="yuv420p",
    )


def _encoder_preflight(info: RobomimicPartitionInfo, args: argparse.Namespace) -> None:
    """Actually encode and reopen two real source frames on local storage."""

    episode = info.plan.episodes[0]
    short_episode = replace(episode, num_frames=min(2, episode.num_frames))
    preflight_parent = args.staging_root / ".encoder-preflight"
    preflight_parent.mkdir(parents=True, exist_ok=True)
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
            encoder_queue_maxsize=queue_size,
            encoder_threads=args.encoder_threads,
        )
        _validate_video_streams(plan, output, args.video_codec)
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
        "stream=codec_name,width,height,r_frame_rate,nb_read_frames,nb_frames,pix_fmt",
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
            if not math.isclose(_fraction(stream["r_frame_rate"]), plan.fps, abs_tol=1e-6):
                raise ConversionError(f"{path}: unexpected FPS {stream['r_frame_rate']}")
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
    unexpected = sorted(path.name for path in data_root.iterdir() if path.name not in expected_names)
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
                        context=f"{info.partition_name} episode {episode_index + 1}/{len(plan.episodes)} saved",
                        force=True,
                    )

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
                    f"completed partition {part_index + 1}/{len(pending)}: {info.partition_name}",
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
    if completed_successfully:
        resume_lock.unlink(missing_ok=True)
    return output


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
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
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--video-quality", type=int, default=18)
    parser.add_argument("--video-preset", default="fast")
    parser.add_argument("--encoder-threads", type=_positive_int, default=4)
    parser.add_argument("--skip-encoder-preflight", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    enabled = sum(bool(value) for value in (args.resume, args.skip_existing, args.overwrite))
    if enabled > 1:
        parser.error("--resume, --skip-existing, and --overwrite are mutually exclusive")
    if (args.max_episodes is not None or args.max_partitions is not None or args.category or args.partition) and (
        args.dataset_uid == DEFAULT_DATASET_UID and not args.inspect_only
    ):
        parser.error("limited conversion requires an explicit smoke --dataset-uid")
    source_root = _source_root(args.raw_root, args.source_directory)
    output = args.staging_root / "lerobot_v3_0" / args.dataset_uid
    paths = _select_source_files(
        source_root,
        categories=set(args.category),
        partitions=set(args.partition),
        max_partitions=args.max_partitions,
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
    if args.skip_existing and output.exists():
        print(f"skipped existing output: {output}")
        return 0
    if not args.skip_encoder_preflight:
        _encoder_preflight(infos[0], args)
    convert_collection(infos, output, args, source_root)
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
