"""Convert 1X World Model Challenge v1.1/v2.0 tokens to a LeRobot collection.

This is a thin collection/decoder layer.  Source inspection lives in
``readers.one_x_world_model_reader`` and all durable writing, validation,
checkpoint snapshots, locking, and atomic publication live in
``convert_core``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import shutil
import signal
import sys
from typing import Any, Sequence
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
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import (
    convert_dataset,
    plan_summary,
    publish_temporary_output,
    validate_video_files,
    validate_written_dataset,
)
from readers.one_x_world_model_reader import OneXWorldModelReader


VERSIONS = ("v1.1", "v2.0")
VIDEO_CODECS = ("libsvtav1", "h264", "hevc", "h264_nvenc", "hevc_nvenc")
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
    """Open a real encoder session and encode one 256x256 RGB frame."""

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
        frame = av.VideoFrame.from_ndarray(
            np.zeros((camera.height, camera.width, 3), dtype=np.uint8), format="rgb24"
        )
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


def _select(plan, *, max_episodes: int | None, max_units: int | None):
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
                "one_x_v1_checkpoint_segments": args.v1_checkpoint_segments,
            }
        )
        plan = reader.build_plan(version_config, args.raw_root, args.staging_root)
        plan = replace(plan, output_path=workspace / partition_name)
        plan = _select(
            plan,
            max_episodes=args.max_episodes,
            max_units=args.max_checkpoint_units,
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
    return {
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
        default=Path("/home/pai/zxw/1x_world_model_dataset_staging"),
    )
    parser.add_argument("--output-dataset-uid", default="1x_world_model_dataset")
    parser.add_argument("--version", action="append", choices=VERSIONS)
    parser.add_argument("--dry-run", "--inspect-only", dest="dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=_positive_int)
    parser.add_argument("--max-checkpoint-units", type=_positive_int)
    parser.add_argument("--v1-decoder-repo", type=Path)
    parser.add_argument("--cosmos-decoder-path", type=Path)
    parser.add_argument("--decode-batch-size", type=_positive_int, default=8)
    parser.add_argument("--v1-checkpoint-segments", type=_positive_int, default=128)
    parser.add_argument("--eta-interval-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--video-codec", choices=VIDEO_CODECS, default="libsvtav1")
    parser.add_argument("--video-quality", type=int, choices=range(52), default=30)
    parser.add_argument("--video-preset")
    parser.add_argument("--encoder-queue-maxsize", type=_positive_int)
    parser.add_argument("--encoder-threads", type=_positive_int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if (
        Path(args.output_dataset_uid).name != args.output_dataset_uid
        or args.output_dataset_uid in {"", ".", ".."}
    ):
        parser.error("--output-dataset-uid must be one safe path component")
    if sum(bool(value) for value in (args.resume, args.skip_existing, args.overwrite)) > 1:
        parser.error("--resume, --skip-existing, and --overwrite are mutually exclusive")
    subset = args.max_episodes is not None or args.max_checkpoint_units is not None
    if subset and args.output_dataset_uid == "1x_world_model_dataset" and not args.dry_run:
        parser.error("smoke/subset conversion requires an independent --output-dataset-uid")
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
        state_root = lock_path = None
    try:
        reader, plans = _plans(args, workspace)
        print(json.dumps({"collection": str(final), "partitions": [plan_summary(plan) for plan in plans]}, indent=2))
        if args.dry_run:
            print("inspect-only complete; no decoder loaded and no output written")
            return 0

        encoder = _rgb_encoder(args.video_codec, args.video_quality, args.video_preset)
        for plan in plans:
            reader.preflight_decoder(plan)
            _preflight_encoder(plan, encoder)
        longest = max(episode.num_frames for plan in plans for episode in plan.episodes)
        queue_size = args.encoder_queue_maxsize or longest + 1
        if queue_size <= longest:
            raise ConversionError(
                f"streaming queue {queue_size} must exceed longest episode ({longest}) so frames cannot drop"
            )

        payload = _collection_payload(plans, args)
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
                    validate_video_files(plan, plan.output_path, expected_frames=plan.num_frames)
                    print(f"[{plan.dataset_uid}] reused completed collection partition", flush=True)
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
            atomic_write_json(workspace / "collection_manifest.json", _collection_manifest(plans, args))
            publish_temporary_output(workspace, final, overwrite=args.overwrite)
            if args.resume and state_root is not None:
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
        print(f"error: {exc}", file=sys.stderr)
        return 1


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
