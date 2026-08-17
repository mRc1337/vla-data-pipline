"""Convert raw Mobile ALOHA HDF5 episodes directly to LeRobot v3.0.

The converter is intentionally separate from ``process_scripts``: it only
ports raw values and RGB frames into staging. It does not clean, normalize,
resample, canonicalize, or upload the dataset.

Expected input layout::

    <raw-root>/<dataset-uid>/<source task directory>/episode_*.hdf5

For the 21 directories in the public release, the converter maps the source
directory identifiers to the natural-language task annotations published in
the official LeRobot datasets. Unknown directory names are retained verbatim
so the lower-level helpers remain usable with local Mobile ALOHA recordings.

The public Mobile ALOHA release is heterogeneous: mobile episodes have a
14-D arm action plus a 2-D base action and three cameras, while static
co-training episodes have only the 14-D arm action and four cameras; effort
is also absent from part of the static data. LeRobot requires one fixed
feature schema per dataset, so the CLI automatically partitions the source
without inventing zero actions, zero effort, or placeholder video. Output is
written as a small collection below::

    <staging-root>/lerobot_v3_0/<dataset-uid>

Each child directory is an independently loadable LeRobot v3.0 dataset and
``collection_manifest.json`` records the partition mapping. A homogeneous
source still produces a one-part collection through the CLI. The lower-level
``inspect_dataset``/``convert_dataset`` functions remain available for callers
that already know their source has one schema.

This file is now a thin, ALOHA-specific layer over ``convert_core/`` (the
LeRobot writer, HDF5 header/camera/FPS helpers) -- everything here that is
genuinely generic HDF5 handling was extracted to
``convert_core.hdf5_common``/``convert_core.lerobot_writer`` and is imported
back in under its original private name so every call site below is
unchanged. What's left is Mobile ALOHA's own domain logic: the dual-arm +
mobile-base 14+2 action-layout ambiguity (``separate_14_plus_2`` vs.
``combined_16``, cross-validated against ``/base_action``) and its fixed
feature schema. A single unambiguous-action HDF5 dataset (most other
robots) doesn't need a new file like this one -- see
``readers/hdf5_reader.py`` and ``convert_dataset.py`` for the config-driven
general case.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any, Callable, Iterator, Sequence
import uuid

import numpy as np

from convert_core.errors import ConversionError
from convert_core.hdf5_common import (
    FPS_RELATIVE_TOLERANCE,
    CameraSpec,
    FpsEvidence,
    as_finite_positive_float as _as_finite_positive_float,
    camera_fps as _camera_fps,
    discover_cameras as _discover_cameras,
    get_hdf5_object as _get_hdf5_object,
    natural_sort_key as _natural_sort_key,
    normalize_hdf5_key as _normalize_hdf5_key,
    read_rgb_frame as _read_rgb_frame,
    relative_difference as _relative_difference,
    require_dataset as _require_dataset,
    require_h5py as _require_h5py,
    validate_numeric_matrix as _validate_numeric_matrix,
)
from convert_core.lerobot_writer import (
    publish_temporary_output as _publish_temporary_output,
    resolve_dataset_uids as _dataset_uids,
)

ARM_DIM = 14
BASE_DIM = 2
HDF5_SUFFIXES = {".h5", ".hdf5"}
DEFAULT_FPS = 50.0
DEFAULT_ETA_INTERVAL_SECONDS = 10.0
RESUME_STATE_VERSION = 1
NVENC_CODECS = {"h264_nvenc", "hevc_nvenc"}
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

ARM_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)
BASE_ACTION_NAMES = ("base_linear_velocity", "base_angular_velocity")

# The public release stores task identity only in directory names.  Use the
# natural-language annotations published with the official LeRobot versions
# instead of exposing download/processing identifiers ("truncated",
# "compressed", dates, and the co-training container directory) as prompts.
MOBILE_ALOHA_TASKS = {
    "aloha_mobile_cabinet": (
        "Open the top cabinet, store the pot inside it, then close the cabinet."
    ),
    "aloha_mobile_chair_truncated": (
        "Push the chairs in front of the desk to place them against it."
    ),
    "aloha_mobile_elevator_truncated": "Take the elevator to the 1st floor.",
    "aloha_mobile_shrimp_truncated": (
        "Sauté the raw shrimp on both sides, then serve it in the bowl."
    ),
    "aloha_mobile_wash_pan": (
        "Pick up the pan, rinse it in the sink, and then place it in the drying rack."
    ),
    "aloha_mobile_wipe_wine": (
        "Pick up the wet cloth on the faucet and use it to clean the spilled wine "
        "on the table and underneath the glass."
    ),
    "aloha_static_cotraining_datasets/12_01_ziploc_slide_50_compressed": (
        "Slide open the ziploc bag."
    ),
    "aloha_static_cotraining_datasets/1_22_cups_open_compressed": (
        "Pick up the plastic cup and open its lid."
    ),
    "aloha_static_cotraining_datasets/aloha_coffee_compressed": (
        "Place the coffee capsule inside the capsule container, then place the cup "
        "onto the center of the cup tray, then push the 'Hot Water' and 'Travel Mug' buttons."
    ),
    "aloha_static_cotraining_datasets/aloha_coffee_new_compressed": (
        "Place the coffee capsule inside the capsule container, then place the cup "
        "onto the center of the cup tray."
    ),
    "aloha_static_cotraining_datasets/aloha_fork_pick_up_compressed": (
        "Pick up the fork and place it on the plate."
    ),
    "aloha_static_cotraining_datasets/aloha_pingpong_test_compressed": (
        "Transfer one of the two balls in the right glass into the left glass, "
        "then transfer it back to the right glass."
    ),
    "aloha_static_cotraining_datasets/aloha_pro_pencil_compressed": (
        "Pick up the pencil with the right arm, hand it over to the left arm, "
        "then place it back onto the table."
    ),
    "aloha_static_cotraining_datasets/aloha_screw_driver_compressed": (
        "Pick up the screwdriver with the right arm, hand it over to the left arm, "
        "then place it into the cup."
    ),
    "aloha_static_cotraining_datasets/aloha_towel_compressed": (
        "Pick up a piece of paper towel and place it on the spilled liquid."
    ),
    "aloha_static_cotraining_datasets/aloha_vinh_cup_compressed": (
        "Pick up the plastic cup with the right arm, then pop its lid open with the left arm."
    ),
    "aloha_static_cotraining_datasets/aloha_vinh_cup_left_compressed": (
        "Pick up the plastic cup with the left arm, then pop its lid open with the right arm."
    ),
    "aloha_static_cotraining_datasets/battery_compressed": (
        "Place the battery into the slot of the remote controller."
    ),
    "aloha_static_cotraining_datasets/candy_compressed": (
        "Pick up the candy and unwrap it."
    ),
    "aloha_static_cotraining_datasets/tape_compressed": (
        "Cut a small piece of tape from the tape dispenser, then place it on the cardboard box's edge."
    ),
    "aloha_static_cotraining_datasets/thread_velcro_compressed": (
        "Pick up the velcro cable tie with the left arm, then insert the end of the "
        "velcro tie into the other end's loop with the right arm."
    ),
}


def _format_duration(seconds: float) -> str:
    """Format an elapsed/ETA duration without relying on wall-clock dates."""
    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a finite positive number, got {value!r}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


class EtaProgress:
    """Rate-limited, newline-based progress suitable for terminals and nohup logs."""

    def __init__(
        self,
        label: str,
        total: int,
        unit: str,
        *,
        interval_seconds: float = DEFAULT_ETA_INTERVAL_SECONDS,
        clock: Any = time.monotonic,
        stream: Any = None,
    ) -> None:
        if total <= 0:
            raise ValueError(f"ETA progress total must be positive, got {total}")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError(
                f"ETA refresh interval must be finite and positive, got {interval_seconds!r}"
            )
        self.label = label
        self.total = total
        self.unit = unit
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.stream = stream if stream is not None else sys.stderr
        self.started_at = float(clock())
        self.last_emitted_at = self.started_at
        self.completed = 0
        self.rate_baseline_completed = 0

    def mark_precompleted(self, amount: int) -> None:
        """Account for durable work found in a checkpoint without inflating the new-run rate."""
        if amount < 0 or self.completed + amount > self.total:
            raise ValueError(f"invalid precompleted amount {amount} for {self.completed}/{self.total}")
        self.completed += amount
        self.rate_baseline_completed = self.completed
        self.started_at = float(self.clock())
        self.last_emitted_at = self.started_at

    def update(self, completed: int, *, context: str | None = None, force: bool = False) -> None:
        if completed < self.completed:
            raise ValueError(
                f"ETA progress cannot move backwards: {completed} < {self.completed}"
            )
        self.completed = min(completed, self.total)
        now = float(self.clock())
        if not force and now - self.last_emitted_at < self.interval_seconds:
            return

        elapsed = max(0.0, now - self.started_at)
        newly_completed = self.completed - self.rate_baseline_completed
        rate = newly_completed / elapsed if elapsed > 0 and newly_completed > 0 else 0.0
        if self.completed >= self.total:
            eta_text = _format_duration(0)
        elif rate > 0:
            eta_text = _format_duration((self.total - self.completed) / rate)
        else:
            eta_text = "--:--:--"
        percent = 100.0 * self.completed / self.total
        rate_text = f"{rate:.2f} {self.unit}/s" if rate > 0 else f"-- {self.unit}/s"
        context_text = f" | {context}" if context else ""
        print(
            f"{self.label} {self.completed}/{self.total} {self.unit} ({percent:5.1f}%)"
            f" | {rate_text} | elapsed {_format_duration(elapsed)} | ETA {eta_text}{context_text}",
            file=self.stream,
            flush=True,
        )
        self.last_emitted_at = now

    def advance(self, amount: int = 1, *, context: str | None = None, force: bool = False) -> None:
        self.update(self.completed + amount, context=context, force=force)

    def finish(self, *, context: str | None = None) -> None:
        self.update(self.total, context=context, force=True)


@dataclass(frozen=True)
class VideoEncodingConfig:
    """Video writer settings used by both the CLI and conversion manifest.

    Streaming mode bypasses LeRobot's PNG round-trip.  For offline conversion
    the queue is required to hold at least one complete longest episode, which
    makes LeRobot's queue-full/drop branch unreachable and preserves every
    source frame.
    """

    streaming: bool = False
    codec: str = "libsvtav1"
    quality: int = 30
    preset: str | int | None = None
    encoder_queue_maxsize: int | None = None
    preflight: bool = True

    def __post_init__(self) -> None:
        if self.codec not in VIDEO_CODECS:
            raise ValueError(f"unsupported video codec {self.codec!r}; choose one of {VIDEO_CODECS}")
        if not 0 <= self.quality <= 51:
            raise ValueError(f"video quality/QP must be between 0 and 51, got {self.quality}")
        if self.encoder_queue_maxsize is not None and self.encoder_queue_maxsize <= 0:
            raise ValueError("encoder queue size must be positive")

    @property
    def effective_preset(self) -> str | int | None:
        if self.codec in NVENC_CODECS:
            preset = "p4" if self.preset is None else self.preset
            if isinstance(preset, str):
                try:
                    return NVENC_PRESETS[preset]
                except KeyError:
                    if preset.isdecimal():
                        return int(preset)
                    raise ValueError(
                        f"unsupported NVENC preset {preset!r}; choose one of {tuple(NVENC_PRESETS)}"
                    ) from None
            return preset
        return self.preset


@dataclass(frozen=True)
class EpisodeSpec:
    source_path: Path
    source_relative_path: str
    instruction: str
    num_frames: int
    action_layout: str
    has_base_action: bool
    cameras: tuple[CameraSpec, ...]
    skipped_depth_keys: tuple[str, ...]
    has_velocity: bool
    has_effort: bool
    fps_evidence: tuple[FpsEvidence, ...]


@dataclass(frozen=True)
class ConversionPlan:
    dataset_uid: str
    raw_dataset_root: Path
    output_path: Path
    fps: int
    measured_fps: float
    state_key: str
    action_key: str
    base_action_key: str
    velocity_key: str
    effort_key: str
    images_key: str
    uncompressed_color_order: str
    episodes: tuple[EpisodeSpec, ...]
    robot_type: str = "mobile_aloha"

    @property
    def num_frames(self) -> int:
        return sum(episode.num_frames for episode in self.episodes)


@dataclass(frozen=True)
class ConversionCollectionPlan:
    dataset_uid: str
    raw_dataset_root: Path
    output_path: Path
    partitions: tuple[ConversionPlan, ...]

    @property
    def num_frames(self) -> int:
        return sum(partition.num_frames for partition in self.partitions)

    @property
    def num_episodes(self) -> int:
        return sum(len(partition.episodes) for partition in self.partitions)


def _resume_paths(output_path: Path) -> tuple[Path, Path, Path]:
    """Return the deterministic checkpoint, signature state, and advisory lock paths."""
    prefix = f".{output_path.name}"
    return (
        output_path.with_name(f"{prefix}.checkpoint"),
        output_path.with_name(f"{prefix}.resume.json"),
        output_path.with_name(f"{prefix}.resume.lock"),
    )


@contextlib.contextmanager
def _resume_lock(lock_path: Path) -> Iterator[None]:
    """Prevent two converter processes from appending to the same checkpoint."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConversionError(
                f"another conversion process is using this resume checkpoint (lock: {lock_path})"
            ) from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _episode_resume_record(episode: EpisodeSpec) -> dict[str, Any]:
    stat = episode.source_path.stat()
    return {
        "source": episode.source_relative_path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "instruction": episode.instruction,
        "num_frames": episode.num_frames,
        "action_layout": episode.action_layout,
        "has_base_action": episode.has_base_action,
        "has_velocity": episode.has_velocity,
        "has_effort": episode.has_effort,
        "cameras": [asdict(camera) for camera in episode.cameras],
    }


def _plan_resume_record(plan: ConversionPlan) -> dict[str, Any]:
    return {
        "dataset_uid": plan.dataset_uid,
        "fps": plan.fps,
        "robot_type": plan.robot_type,
        "hdf5_keys": {
            "state": plan.state_key,
            "action": plan.action_key,
            "base_action": plan.base_action_key,
            "velocity": plan.velocity_key,
            "effort": plan.effort_key,
            "images": plan.images_key,
        },
        "uncompressed_color_order": plan.uncompressed_color_order,
        "features": _feature_schema(plan),
        "episodes": [_episode_resume_record(episode) for episode in plan.episodes],
    }


def _resume_signature(
    plan_or_collection: ConversionPlan | ConversionCollectionPlan,
    video_encoding: VideoEncodingConfig,
) -> tuple[str, dict[str, Any]]:
    if isinstance(plan_or_collection, ConversionCollectionPlan):
        payload: dict[str, Any] = {
            "kind": "collection",
            "dataset_uid": plan_or_collection.dataset_uid,
            "partitions": [_plan_resume_record(plan) for plan in plan_or_collection.partitions],
        }
    else:
        payload = {"kind": "dataset", "plan": _plan_resume_record(plan_or_collection)}
    payload["video_encoding"] = {
        **asdict(video_encoding),
        "preset": video_encoding.effective_preset,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def _prepare_resume_state(
    state_path: Path,
    plan_or_collection: ConversionPlan | ConversionCollectionPlan,
    video_encoding: VideoEncodingConfig,
    *,
    checkpoint_exists: bool,
) -> None:
    signature, payload = _resume_signature(plan_or_collection, video_encoding)
    if state_path.exists():
        if not state_path.is_file():
            raise ConversionError(f"resume state path is not a file: {state_path}")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(f"cannot read resume state {state_path}: {exc}") from exc
        if state.get("version") != RESUME_STATE_VERSION or state.get("signature") != signature:
            raise ConversionError(
                "resume checkpoint does not match the current source files, schema, or video "
                f"settings: {state_path}; use the original arguments or remove the checkpoint"
            )
        return

    if checkpoint_exists:
        if not state_path.is_file():
            raise ConversionError(
                f"resume checkpoint exists but its state file is missing: {state_path}; "
                "move or remove the checkpoint explicitly before starting over"
            )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_state = state_path.with_name(f".{state_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_state.write_text(
        json.dumps(
            {
                "version": RESUME_STATE_VERSION,
                "signature": signature,
                "configuration": payload,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_state.replace(state_path)


def _cleanup_checkpoint_tempdirs(checkpoint_path: Path) -> None:
    """Remove only LeRobot streaming-encoder temp directories left by an interrupted episode."""
    for path in checkpoint_path.iterdir():
        if path.is_dir() and path.name.startswith("tmp"):
            shutil.rmtree(path)


def _resolve_instruction(raw_dataset_root: Path, episode_path: Path) -> str:
    relative_parent = episode_path.parent.relative_to(raw_dataset_root)
    if relative_parent == Path("."):
        raise ConversionError(
            f"{episode_path}: episode is directly below the dataset root and has no instruction subdirectory"
        )
    source_task = relative_parent.as_posix()
    return MOBILE_ALOHA_TASKS.get(source_task, source_task)


def _inspect_episode(
    episode_path: Path,
    *,
    raw_dataset_root: Path,
    state_key: str,
    action_key: str,
    base_action_key: str,
    velocity_key: str,
    effort_key: str,
    images_key: str,
    fallback_fps: float | None,
) -> EpisodeSpec:
    h5py = _require_h5py()
    instruction = _resolve_instruction(raw_dataset_root, episode_path)
    with h5py.File(episode_path, "r") as h5_file:
        state = _require_dataset(h5_file, state_key, episode_path)
        num_frames, _ = _validate_numeric_matrix(
            state, source_path=episode_path, key=state_key, expected_width=ARM_DIM
        )
        action = _require_dataset(h5_file, action_key, episode_path)
        _, action_width = _validate_numeric_matrix(
            action,
            source_path=episode_path,
            key=action_key,
            expected_width=(ARM_DIM, ARM_DIM + BASE_DIM),
            expected_frames=num_frames,
        )
        base_action = _get_hdf5_object(h5_file, base_action_key)
        if base_action is not None and not isinstance(base_action, h5py.Dataset):
            raise ConversionError(f"{episode_path}: {_normalize_hdf5_key(base_action_key)} is not a dataset")
        if base_action is not None:
            _validate_numeric_matrix(
                base_action,
                source_path=episode_path,
                key=base_action_key,
                expected_width=BASE_DIM,
                expected_frames=num_frames,
            )

        if action_width == ARM_DIM:
            if base_action is None:
                action_layout = "arm_only_14"
            else:
                action_layout = "separate_14_plus_2"
        else:
            action_layout = "combined_16"
            if base_action is not None and not np.allclose(
                np.asarray(action[:, ARM_DIM:]), np.asarray(base_action[:]), rtol=1e-5, atol=1e-6
            ):
                raise ConversionError(
                    f"{episode_path}: combined action columns 14:16 disagree with "
                    f"{_normalize_hdf5_key(base_action_key)}"
                )
            if base_action is not None:
                action_layout = "combined_16_verified_by_separate_base"

        optional_presence: dict[str, bool] = {}
        for label, key in (("velocity", velocity_key), ("effort", effort_key)):
            optional = _get_hdf5_object(h5_file, key)
            optional_presence[label] = optional is not None
            if optional is not None:
                if not isinstance(optional, h5py.Dataset):
                    raise ConversionError(f"{episode_path}: {_normalize_hdf5_key(key)} is not a dataset")
                _validate_numeric_matrix(
                    optional,
                    source_path=episode_path,
                    key=key,
                    expected_width=ARM_DIM,
                    expected_frames=num_frames,
                )

        cameras, skipped_depth = _discover_cameras(h5_file, images_key, episode_path)
        for camera in cameras:
            camera_dataset = h5_file[camera.source_key]
            if camera_dataset.shape[0] != num_frames:
                raise ConversionError(
                    f"{episode_path}: {camera.source_key} has {camera_dataset.shape[0]} frames, expected {num_frames}"
                )
        fps_evidence = tuple(
            _camera_fps(
                h5_file,
                camera,
                images_key=images_key,
                num_frames=num_frames,
                source_path=episode_path,
                raw_dataset_root=raw_dataset_root,
                fallback_fps=fallback_fps,
            )
            for camera in cameras
        )

    relative_path = episode_path.relative_to(raw_dataset_root).as_posix()
    return EpisodeSpec(
        source_path=episode_path,
        source_relative_path=relative_path,
        instruction=instruction,
        num_frames=num_frames,
        action_layout=action_layout,
        has_base_action=action_width == ARM_DIM + BASE_DIM or base_action is not None,
        cameras=cameras,
        skipped_depth_keys=skipped_depth,
        has_velocity=optional_presence["velocity"],
        has_effort=optional_presence["effort"],
        fps_evidence=fps_evidence,
    )


def _validate_dataset_uid(dataset_uid: str) -> None:
    if Path(dataset_uid).name != dataset_uid or dataset_uid in {"", ".", ".."}:
        raise ConversionError(f"dataset UID must be one path component, got {dataset_uid!r}")


def _episode_paths(raw_dataset_root: Path) -> tuple[Path, ...]:
    episode_paths = tuple(
        sorted(
            (
                path
                for path in raw_dataset_root.rglob("*")
                if path.is_file() and path.suffix.casefold() in HDF5_SUFFIXES
            ),
            key=_natural_sort_key,
        )
    )
    if not episode_paths:
        raise ConversionError(f"no .h5 or .hdf5 episodes found below {raw_dataset_root}")
    return episode_paths


def _camera_schema(episode: EpisodeSpec) -> tuple[tuple[str, int, int], ...]:
    return tuple((camera.feature_key, camera.height, camera.width) for camera in episode.cameras)


def _resolved_fps(episodes: Sequence[EpisodeSpec]) -> tuple[int, float]:
    fps_values = [evidence.value for episode in episodes for evidence in episode.fps_evidence]
    measured_fps = float(np.median(fps_values))
    inconsistent = [
        value
        for value in fps_values
        if _relative_difference(value, measured_fps) > FPS_RELATIVE_TOLERANCE
    ]
    if inconsistent:
        details = ", ".join(f"{value:.6g}" for value in fps_values)
        raise ConversionError(f"camera/episode FPS values differ by more than 2%: {details}")
    integer_fps = int(round(measured_fps))
    if integer_fps <= 0 or _relative_difference(measured_fps, integer_fps) > FPS_RELATIVE_TOLERANCE:
        raise ConversionError(
            f"measured FPS {measured_fps:.6g} is not within 2% of an integer; "
            "LeRobot video encoding requires an integer FPS and this converter does not resample"
        )
    return integer_fps, measured_fps


def _episode_schema_key(episode: EpisodeSpec) -> tuple[Any, ...]:
    integer_fps, _ = _resolved_fps((episode,))
    return (
        episode.has_base_action,
        episode.has_velocity,
        episode.has_effort,
        _camera_schema(episode),
        integer_fps,
    )


def _build_plan(
    *,
    dataset_uid: str,
    raw_dataset_root: Path,
    output_path: Path,
    episodes: Sequence[EpisodeSpec],
    state_key: str,
    action_key: str,
    base_action_key: str,
    velocity_key: str,
    effort_key: str,
    images_key: str,
    uncompressed_color_order: str,
) -> ConversionPlan:
    if not episodes:
        raise ConversionError("cannot build a conversion plan without episodes")
    reference = episodes[0]
    for episode in episodes[1:]:
        if _camera_schema(episode) != _camera_schema(reference):
            raise ConversionError(
                f"{episode.source_path}: camera schema {_camera_schema(episode)} "
                f"differs from first episode schema {_camera_schema(reference)}"
            )
        for label in ("has_base_action", "has_velocity", "has_effort"):
            if getattr(episode, label) != getattr(reference, label):
                feature = label.removeprefix("has_").replace("_", " ")
                raise ConversionError(
                    f"{episode.source_path}: {feature} feature presence differs across episodes"
                )

    integer_fps, measured_fps = _resolved_fps(episodes)
    return ConversionPlan(
        dataset_uid=dataset_uid,
        raw_dataset_root=raw_dataset_root,
        output_path=output_path,
        fps=integer_fps,
        measured_fps=measured_fps,
        state_key=_normalize_hdf5_key(state_key),
        action_key=_normalize_hdf5_key(action_key),
        base_action_key=_normalize_hdf5_key(base_action_key),
        velocity_key=_normalize_hdf5_key(velocity_key),
        effort_key=_normalize_hdf5_key(effort_key),
        images_key=_normalize_hdf5_key(images_key),
        uncompressed_color_order=uncompressed_color_order,
        episodes=tuple(episodes),
        robot_type="mobile_aloha" if reference.has_base_action else "aloha_static",
    )


def _inspect_all_episodes(
    *,
    raw_root: Path,
    dataset_uid: str,
    state_key: str,
    action_key: str,
    base_action_key: str,
    velocity_key: str,
    effort_key: str,
    images_key: str,
    fps: float | None,
    eta_interval_seconds: float,
    episode_limit: int | None = None,
) -> tuple[Path, tuple[EpisodeSpec, ...]]:
    _validate_dataset_uid(dataset_uid)
    if fps is not None and _as_finite_positive_float(fps) is None:
        raise ConversionError(f"--fps must be finite and positive, got {fps!r}")
    raw_dataset_root = raw_root / dataset_uid
    if not raw_dataset_root.is_dir():
        raise ConversionError(f"raw dataset directory does not exist: {raw_dataset_root}")
    paths = _episode_paths(raw_dataset_root)
    if episode_limit is not None:
        if episode_limit <= 0:
            raise ConversionError(f"episode limit must be positive, got {episode_limit}")
        paths = paths[:episode_limit]
    episodes: list[EpisodeSpec] = []
    progress = EtaProgress(
        f"[{dataset_uid}] inspect",
        len(paths),
        "episodes",
        interval_seconds=eta_interval_seconds,
    )
    for index, path in enumerate(paths, start=1):
        episodes.append(
            _inspect_episode(
                path,
                raw_dataset_root=raw_dataset_root,
                state_key=state_key,
                action_key=action_key,
                base_action_key=base_action_key,
                velocity_key=velocity_key,
                effort_key=effort_key,
                images_key=images_key,
                fallback_fps=fps,
            )
        )
        progress.update(
            index,
            context=path.relative_to(raw_dataset_root).as_posix(),
            force=index == len(paths),
        )
    return raw_dataset_root, tuple(episodes)


def inspect_dataset(
    *,
    raw_root: Path,
    staging_root: Path,
    dataset_uid: str,
    state_key: str = "/observations/qpos",
    action_key: str = "/action",
    base_action_key: str = "/base_action",
    velocity_key: str = "/observations/qvel",
    effort_key: str = "/observations/effort",
    images_key: str = "/observations/images",
    fps: float | None = DEFAULT_FPS,
    uncompressed_color_order: str = "bgr",
    eta_interval_seconds: float = DEFAULT_ETA_INTERVAL_SECONDS,
    episode_limit: int | None = None,
) -> ConversionPlan:
    raw_dataset_root, episodes = _inspect_all_episodes(
        raw_root=raw_root,
        dataset_uid=dataset_uid,
        state_key=state_key,
        action_key=action_key,
        base_action_key=base_action_key,
        velocity_key=velocity_key,
        effort_key=effort_key,
        images_key=images_key,
        fps=fps,
        eta_interval_seconds=eta_interval_seconds,
        episode_limit=episode_limit,
    )
    return _build_plan(
        dataset_uid=dataset_uid,
        raw_dataset_root=raw_dataset_root,
        output_path=staging_root / "lerobot_v3_0" / dataset_uid,
        episodes=episodes,
        state_key=state_key,
        action_key=action_key,
        base_action_key=base_action_key,
        velocity_key=velocity_key,
        effort_key=effort_key,
        images_key=images_key,
        uncompressed_color_order=uncompressed_color_order,
    )


def inspect_dataset_collection(
    *,
    raw_root: Path,
    staging_root: Path,
    dataset_uid: str,
    state_key: str = "/observations/qpos",
    action_key: str = "/action",
    base_action_key: str = "/base_action",
    velocity_key: str = "/observations/qvel",
    effort_key: str = "/observations/effort",
    images_key: str = "/observations/images",
    fps: float | None = DEFAULT_FPS,
    uncompressed_color_order: str = "bgr",
    eta_interval_seconds: float = DEFAULT_ETA_INTERVAL_SECONDS,
    episode_limit: int | None = None,
) -> ConversionCollectionPlan:
    raw_dataset_root, episodes = _inspect_all_episodes(
        raw_root=raw_root,
        dataset_uid=dataset_uid,
        state_key=state_key,
        action_key=action_key,
        base_action_key=base_action_key,
        velocity_key=velocity_key,
        effort_key=effort_key,
        images_key=images_key,
        fps=fps,
        eta_interval_seconds=eta_interval_seconds,
        episode_limit=episode_limit,
    )
    grouped: dict[tuple[Any, ...], list[EpisodeSpec]] = {}
    for episode in episodes:
        grouped.setdefault(_episode_schema_key(episode), []).append(episode)

    collection_output = staging_root / "lerobot_v3_0" / dataset_uid
    partitions: list[ConversionPlan] = []
    for index, (schema_key, partition_episodes) in enumerate(sorted(grouped.items())):
        has_base, has_velocity, has_effort, cameras, integer_fps = schema_key
        motion = "mobile" if has_base else "static"
        velocity = "velocity" if has_velocity else "no-velocity"
        effort = "effort" if has_effort else "no-effort"
        partition_name = (
            f"part-{index:03d}-{motion}-{velocity}-{effort}-{len(cameras)}cams-{integer_fps}fps"
        )
        partition_uid = f"{dataset_uid}_{partition_name.replace('-', '_')}"
        partitions.append(
            _build_plan(
                dataset_uid=partition_uid,
                raw_dataset_root=raw_dataset_root,
                output_path=collection_output / partition_name,
                episodes=partition_episodes,
                state_key=state_key,
                action_key=action_key,
                base_action_key=base_action_key,
                velocity_key=velocity_key,
                effort_key=effort_key,
                images_key=images_key,
                uncompressed_color_order=uncompressed_color_order,
            )
        )

    return ConversionCollectionPlan(
        dataset_uid=dataset_uid,
        raw_dataset_root=raw_dataset_root,
        output_path=collection_output,
        partitions=tuple(partitions),
    )


def _feature_schema(plan: ConversionPlan) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ARM_DIM,),
            "names": list(ARM_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (ARM_DIM,),
            "names": list(ARM_NAMES),
        },
    }
    if plan.episodes[0].has_base_action:
        features["action.base"] = {
            "dtype": "float32",
            "shape": (BASE_DIM,),
            "names": list(BASE_ACTION_NAMES),
        }
    if plan.episodes[0].has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (ARM_DIM,),
            "names": list(ARM_NAMES),
        }
    if plan.episodes[0].has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (ARM_DIM,),
            "names": list(ARM_NAMES),
        }
    for camera in plan.episodes[0].cameras:
        features[camera.feature_key] = {
            "dtype": "video",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _manifest(
    plan: ConversionPlan,
    video_encoding: VideoEncodingConfig,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "format": "lerobot_v3_0",
        "converter": "convert_mobile_aloha_to_lerobot.py",
        "dataset_uid": plan.dataset_uid,
        "robot_type": plan.robot_type,
        "raw_dataset_root": str(plan.raw_dataset_root.resolve()),
        "fps": plan.fps,
        "measured_fps": plan.measured_fps,
        "fps_relative_tolerance": FPS_RELATIVE_TOLERANCE,
        "num_episodes": len(plan.episodes),
        "num_frames": plan.num_frames,
        "video_encoding": {
            **asdict(video_encoding),
            "preset": video_encoding.effective_preset,
        },
        "conversion_metrics": {
            "elapsed_seconds": elapsed_seconds,
            "frames_per_second": plan.num_frames / elapsed_seconds if elapsed_seconds > 0 else None,
        },
        "hdf5_keys": {
            "state": plan.state_key,
            "action": plan.action_key,
            "base_action": plan.base_action_key,
            "velocity": plan.velocity_key,
            "effort": plan.effort_key,
            "images": plan.images_key,
        },
        "uncompressed_color_order": plan.uncompressed_color_order,
        "features": _feature_schema(plan),
        "episodes": [
            {
                "source": episode.source_relative_path,
                "instruction": episode.instruction,
                "num_frames": episode.num_frames,
                "action_layout": episode.action_layout,
                "cameras": [asdict(camera) for camera in episode.cameras],
                "skipped_depth_keys": list(episode.skipped_depth_keys),
                "fps_evidence": [asdict(evidence) for evidence in episode.fps_evidence],
            }
            for episode in plan.episodes
        ],
    }


def plan_summary(plan: ConversionPlan) -> dict[str, Any]:
    return {
        "dataset_uid": plan.dataset_uid,
        "input": str(plan.raw_dataset_root),
        "output": str(plan.output_path),
        "robot_type": plan.robot_type,
        "fps": plan.fps,
        "measured_fps": plan.measured_fps,
        "episodes": len(plan.episodes),
        "frames": plan.num_frames,
        "tasks": sorted({episode.instruction for episode in plan.episodes}),
        "features": _feature_schema(plan),
        "skipped_depth_keys": sorted(
            {key for episode in plan.episodes for key in episode.skipped_depth_keys}
        ),
    }


def _collection_manifest(
    collection: ConversionCollectionPlan,
    video_encoding: VideoEncodingConfig,
) -> dict[str, Any]:
    return {
        "format": "lerobot_v3_0_collection",
        "converter": "convert_mobile_aloha_to_lerobot.py",
        "dataset_uid": collection.dataset_uid,
        "raw_dataset_root": str(collection.raw_dataset_root.resolve()),
        "num_partitions": len(collection.partitions),
        "num_episodes": collection.num_episodes,
        "num_frames": collection.num_frames,
        "video_encoding": {
            **asdict(video_encoding),
            "preset": video_encoding.effective_preset,
        },
        "partitions": [
            {
                "name": partition.output_path.name,
                "path": partition.output_path.name,
                "repo_id": partition.dataset_uid,
                "robot_type": partition.robot_type,
                "fps": partition.fps,
                "num_episodes": len(partition.episodes),
                "num_frames": partition.num_frames,
                "features": _feature_schema(partition),
            }
            for partition in collection.partitions
        ],
    }


def collection_summary(collection: ConversionCollectionPlan) -> dict[str, Any]:
    return {
        "dataset_uid": collection.dataset_uid,
        "input": str(collection.raw_dataset_root),
        "output": str(collection.output_path),
        "partitions": len(collection.partitions),
        "episodes": collection.num_episodes,
        "frames": collection.num_frames,
        "partition_plans": [plan_summary(partition) for partition in collection.partitions],
    }


def _episode_arrays(h5_file: Any, plan: ConversionPlan, episode: EpisodeSpec) -> dict[str, np.ndarray]:
    state = np.asarray(h5_file[plan.state_key][:], dtype=np.float32)
    raw_action = np.asarray(h5_file[plan.action_key][:], dtype=np.float32)
    if raw_action.shape[1] == ARM_DIM:
        arm_action = raw_action
    else:
        arm_action = raw_action[:, :ARM_DIM]
    arrays = {
        "observation.state": state,
        "action": arm_action,
    }
    if episode.has_base_action:
        if raw_action.shape[1] == ARM_DIM:
            arrays["action.base"] = np.asarray(h5_file[plan.base_action_key][:], dtype=np.float32)
        else:
            arrays["action.base"] = raw_action[:, ARM_DIM : ARM_DIM + BASE_DIM]
    if episode.has_velocity:
        arrays["observation.velocity"] = np.asarray(h5_file[plan.velocity_key][:], dtype=np.float32)
    if episode.has_effort:
        arrays["observation.effort"] = np.asarray(h5_file[plan.effort_key][:], dtype=np.float32)
    return arrays


def _rgb_encoder(video_encoding: VideoEncodingConfig) -> Any:
    try:
        from lerobot.configs.video import RGBEncoderConfig
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("lerobot==0.6.0 is required; install the project's requirements.txt") from exc

    return RGBEncoderConfig(
        vcodec=video_encoding.codec,
        crf=video_encoding.quality,
        preset=video_encoding.effective_preset,
    )


def _streaming_queue_size(plan: ConversionPlan, video_encoding: VideoEncodingConfig) -> int:
    """Return a queue capacity that cannot drop an offline episode's frames."""
    required = max(episode.num_frames for episode in plan.episodes) + 1
    requested = video_encoding.encoder_queue_maxsize
    if requested is None:
        return required
    if requested < required:
        raise ConversionError(
            f"streaming encoder queue {requested} is too small for the longest episode "
            f"({required - 1} frames); use at least {required} so offline conversion cannot drop frames"
        )
    return requested


def _preflight_video_encoder(
    plan: ConversionPlan,
    video_encoding: VideoEncodingConfig,
) -> None:
    """Open all camera encoder sessions and encode a frame before conversion.

    LeRobot's codec discovery only proves that FFmpeg was compiled with an
    encoder.  This probe also proves that the device, driver, permissions and
    simultaneous session count work in the current process.
    """
    if not video_encoding.preflight or video_encoding.codec not in NVENC_CODECS:
        return

    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency of lerobot[dataset]
        raise RuntimeError("PyAV is required for the NVENC preflight") from exc

    camera_specs = plan.episodes[0].cameras
    resources: list[tuple[Any, Any, io.BytesIO]] = []
    try:
        encoder = _rgb_encoder(video_encoding)
        for camera in camera_specs:
            buffer = io.BytesIO()
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
                np.zeros((camera.height, camera.width, 3), dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
            resources.append((container, stream, buffer))

        # Keep every session open until all cameras have successfully encoded.
        for container, stream, _ in resources:
            for packet in stream.encode():
                container.mux(packet)
        for container, _, _ in resources:
            container.close()
        resources.clear()
    except Exception as exc:
        raise ConversionError(
            f"{video_encoding.codec} preflight failed while opening "
            f"{len(camera_specs)} simultaneous {camera_specs[0].width}x{camera_specs[0].height} "
            "camera encoders; FFmpeg codec presence alone does not prove NVENC hardware access. "
            f"Original error: {exc}"
        ) from exc
    finally:
        for container, _, _ in resources:
            with contextlib.suppress(Exception):
                container.close()


def _video_stream_summary(video_paths: Sequence[Path]) -> tuple[int, set[str], set[float]]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency of lerobot[dataset]
        raise RuntimeError("PyAV is required to validate written videos") from exc

    total_frames = 0
    codecs: set[str] = set()
    frame_rates: set[float] = set()
    for path in video_paths:
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            codecs.add(stream.codec.canonical_name)
            rate = stream.average_rate or stream.base_rate
            if rate is not None:
                frame_rates.add(float(rate))
            frame_count = int(stream.frames or 0)
            if frame_count <= 0:
                frame_count = sum(1 for _ in container.decode(stream))
            total_frames += frame_count
    return total_frames, codecs, frame_rates


def _validate_video_streams(
    plan: ConversionPlan,
    temporary_path: Path,
    video_encoding: VideoEncodingConfig,
) -> None:
    expected_codec = {
        "libsvtav1": "av1",
        "libaom-av1": "av1",
        "h264": "h264",
        "h264_nvenc": "h264",
        "hevc": "hevc",
        "hevc_nvenc": "hevc",
    }[video_encoding.codec]
    for camera in plan.episodes[0].cameras:
        paths = sorted((temporary_path / "videos" / camera.feature_key).rglob("*.mp4"))
        if not paths:
            raise ConversionError(f"written dataset has no video files for {camera.feature_key}")
        frame_count, codecs, frame_rates = _video_stream_summary(paths)
        if frame_count != plan.num_frames:
            raise ConversionError(
                f"written videos for {camera.feature_key} contain {frame_count} frames, "
                f"expected {plan.num_frames}"
            )
        if codecs != {expected_codec}:
            raise ConversionError(
                f"written videos for {camera.feature_key} use codecs {sorted(codecs)}, "
                f"expected {expected_codec}"
            )
        if any(not math.isclose(rate, plan.fps, rel_tol=0.0, abs_tol=1e-9) for rate in frame_rates):
            raise ConversionError(
                f"written videos for {camera.feature_key} use frame rates {sorted(frame_rates)}, "
                f"expected {plan.fps}"
            )


def _write_dataset(
    plan: ConversionPlan,
    temporary_path: Path,
    *,
    progress: EtaProgress,
    video_encoding: VideoEncodingConfig,
    resume_existing: bool = False,
    episode_completed_hook: Callable[[ConversionPlan, int], None] | None = None,
) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("lerobot==0.6.0 is required; install the project's requirements.txt") from exc

    h5py = _require_h5py()
    _preflight_video_encoder(plan, video_encoding)
    encoder_queue_maxsize = (
        _streaming_queue_size(plan, video_encoding) if video_encoding.streaming else 30
    )
    if resume_existing:
        _cleanup_checkpoint_tempdirs(temporary_path)
        dataset = LeRobotDataset.resume(
            repo_id=plan.dataset_uid,
            root=temporary_path,
            streaming_encoding=video_encoding.streaming,
            encoder_queue_maxsize=encoder_queue_maxsize,
            rgb_encoder=_rgb_encoder(video_encoding),
        )
        # LeRobot defaults to flushing episode metadata every ten episodes. A
        # checkpoint must expose every completed episode to the next process.
        dataset.meta._metadata_buffer_size = 1
    else:
        dataset = LeRobotDataset.create(
            repo_id=plan.dataset_uid,
            fps=plan.fps,
            root=temporary_path,
            features=_feature_schema(plan),
            robot_type=plan.robot_type,
            use_videos=True,
            metadata_buffer_size=1,
            streaming_encoding=video_encoding.streaming,
            encoder_queue_maxsize=encoder_queue_maxsize,
            rgb_encoder=_rgb_encoder(video_encoding),
        )
    try:
        completed_episodes = _validate_completed_episodes(plan, dataset)
        completed_frames = sum(
            episode.num_frames for episode in plan.episodes[:completed_episodes]
        )
        if completed_frames:
            progress.mark_precompleted(completed_frames)
            print(
                f"[{plan.dataset_uid}] resuming after {completed_episodes}/{len(plan.episodes)} "
                f"episodes ({completed_frames} durable frames)",
                file=sys.stderr,
                flush=True,
            )
        for episode_index, episode in enumerate(
            plan.episodes[completed_episodes:], start=completed_episodes
        ):
            print(
                f"[{plan.dataset_uid}] episode {episode_index + 1}/{len(plan.episodes)}: "
                f"{episode.source_relative_path} ({episode.num_frames} frames)",
                file=sys.stderr,
            )
            with h5py.File(episode.source_path, "r") as h5_file:
                arrays = _episode_arrays(h5_file, plan, episode)
                for frame_index in range(episode.num_frames):
                    frame: dict[str, Any] = {
                        key: values[frame_index] for key, values in arrays.items()
                    }
                    frame["task"] = episode.instruction
                    for camera in episode.cameras:
                        frame[camera.feature_key] = _read_rgb_frame(
                            h5_file[camera.source_key],
                            frame_index,
                            camera,
                            source_path=episode.source_path,
                            uncompressed_color_order=plan.uncompressed_color_order,
                        )
                    dataset.add_frame(frame)
                    progress.advance(
                        context=(
                            f"{plan.output_path.name} episode "
                            f"{episode_index + 1}/{len(plan.episodes)}"
                        )
                    )
            dataset.save_episode()
            # Force an update after save_episode so video encoder/muxer time is
            # reflected immediately in the global rate and ETA.
            progress.update(
                progress.completed,
                context=(
                    f"{plan.output_path.name} episode "
                    f"{episode_index + 1}/{len(plan.episodes)} saved"
                ),
                force=True,
            )
            if episode_completed_hook is not None:
                episode_completed_hook(plan, episode_index)
        print(f"[{plan.dataset_uid}] finalizing dataset metadata and videos", file=sys.stderr, flush=True)
        dataset.finalize()
    except BaseException:
        # Preserve every fully saved episode and discard only the current
        # in-memory/streaming episode before closing parquet footers.
        with contextlib.suppress(Exception):
            dataset.clear_episode_buffer(delete_images=True)
        with contextlib.suppress(Exception):
            dataset.finalize()
        raise


def _validate_completed_episodes(plan: ConversionPlan, dataset: Any) -> int:
    completed = int(dataset.meta.total_episodes)
    if completed > len(plan.episodes):
        raise ConversionError(
            f"checkpoint has {completed} episodes, but the current plan has only {len(plan.episodes)}"
        )
    expected_frames = sum(episode.num_frames for episode in plan.episodes[:completed])
    if int(dataset.meta.total_frames) != expected_frames:
        raise ConversionError(
            f"checkpoint metadata has {dataset.meta.total_frames} frames for {completed} episodes, "
            f"expected {expected_frames}"
        )
    episode_rows = dataset.meta.episodes
    if completed and (episode_rows is None or len(episode_rows) != completed):
        raise ConversionError(
            f"checkpoint exposes {0 if episode_rows is None else len(episode_rows)} episode rows, "
            f"but info.json reports {completed}"
        )
    for episode_index in range(completed):
        expected = plan.episodes[episode_index]
        row = episode_rows[episode_index]
        if int(row["length"]) != expected.num_frames:
            raise ConversionError(
                f"checkpoint episode {episode_index} has {int(row['length'])} frames, "
                f"expected {expected.num_frames}"
            )
        if set(row["tasks"]) != {expected.instruction}:
            raise ConversionError(
                f"checkpoint episode {episode_index} tasks are {row['tasks']}, "
                f"expected {[expected.instruction]}"
            )
    return completed


def _validate_written_dataset(
    plan: ConversionPlan,
    temporary_path: Path,
    video_encoding: VideoEncodingConfig,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.dataset_uid, root=temporary_path)
    if dataset.num_episodes != len(plan.episodes):
        raise ConversionError(
            f"written dataset has {dataset.num_episodes} episodes, expected {len(plan.episodes)}"
        )
    if len(dataset) != plan.num_frames:
        raise ConversionError(f"written dataset has {len(dataset)} frames, expected {plan.num_frames}")
    missing = set(_feature_schema(plan)) - set(dataset.meta.features)
    if missing:
        raise ConversionError(f"written dataset is missing features: {sorted(missing)}")
    written_fps = float(dataset.meta.fps)
    if not math.isclose(written_fps, plan.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ConversionError(f"written dataset FPS is {written_fps}, expected {plan.fps}")
    written_tasks = set(dataset.meta.tasks.index.tolist())
    expected_tasks = {episode.instruction for episode in plan.episodes}
    if written_tasks != expected_tasks:
        raise ConversionError(
            f"written dataset tasks are {sorted(written_tasks)}, expected {sorted(expected_tasks)}"
        )
    episode_rows = dataset.meta.episodes
    for episode_index, expected in enumerate(plan.episodes):
        row = episode_rows[episode_index]
        if int(row["length"]) != expected.num_frames:
            raise ConversionError(
                f"written episode {episode_index} has {int(row['length'])} frames, expected {expected.num_frames}"
            )
        if set(row["tasks"]) != {expected.instruction}:
            raise ConversionError(
                f"written episode {episode_index} tasks are {row['tasks']}, expected {[expected.instruction]}"
            )
    del dataset
    _validate_video_streams(plan, temporary_path, video_encoding)


def convert_dataset(
    plan: ConversionPlan,
    *,
    overwrite: bool = False,
    resume: bool = False,
    eta_interval_seconds: float = DEFAULT_ETA_INTERVAL_SECONDS,
    progress: EtaProgress | None = None,
    video_encoding: VideoEncodingConfig | None = None,
    episode_completed_hook: Callable[[ConversionPlan, int], None] | None = None,
) -> Path:
    video_encoding = video_encoding or VideoEncodingConfig()
    if not video_encoding.streaming and video_encoding.encoder_queue_maxsize is not None:
        raise ConversionError("encoder queue size is only meaningful with streaming encoding")
    if resume and overwrite:
        raise ConversionError("resume and overwrite are mutually exclusive")
    output_path = plan.output_path
    if output_path.exists() and (resume or not overwrite):
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        temporary_path, state_path, lock_path = _resume_paths(output_path)
    else:
        temporary_path = output_path.with_name(f".{output_path.name}.incomplete-{uuid.uuid4().hex}")
        state_path = lock_path = None
    owns_progress = progress is None
    if progress is None:
        progress = EtaProgress(
            f"[{plan.dataset_uid}] convert",
            plan.num_frames,
            "frames",
            interval_seconds=eta_interval_seconds,
        )
    lock_context = _resume_lock(lock_path) if resume else contextlib.nullcontext()
    with lock_context:
        checkpoint_existed = temporary_path.exists()
        if resume:
            assert state_path is not None
            _prepare_resume_state(
                state_path,
                plan,
                video_encoding,
                checkpoint_exists=checkpoint_existed,
            )
            if checkpoint_existed and not (temporary_path / "meta" / "info.json").is_file():
                raise ConversionError(
                    f"resume checkpoint is incomplete or corrupt: {temporary_path} "
                    "(missing meta/info.json)"
                )
        try:
            started_at = time.monotonic()
            _write_dataset(
                plan,
                temporary_path,
                progress=progress,
                video_encoding=video_encoding,
                resume_existing=checkpoint_existed,
                episode_completed_hook=episode_completed_hook,
            )
            _validate_written_dataset(plan, temporary_path, video_encoding)
            elapsed_seconds = time.monotonic() - started_at
            (temporary_path / "conversion_manifest.json").write_text(
                json.dumps(
                    _manifest(plan, video_encoding, elapsed_seconds=elapsed_seconds),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            _publish_temporary_output(temporary_path, output_path, overwrite=overwrite)
            if resume:
                state_path.unlink(missing_ok=True)
            if owns_progress:
                progress.finish(context="conversion and validation complete")
        except BaseException:
            if not resume and temporary_path.exists():
                shutil.rmtree(temporary_path)
            elif resume:
                print(
                    f"[{plan.dataset_uid}] checkpoint retained for --resume: {temporary_path}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
    return output_path


def convert_collection(
    collection: ConversionCollectionPlan,
    *,
    overwrite: bool = False,
    resume: bool = False,
    eta_interval_seconds: float = DEFAULT_ETA_INTERVAL_SECONDS,
    video_encoding: VideoEncodingConfig | None = None,
    episode_completed_hook: Callable[[ConversionPlan, int], None] | None = None,
) -> Path:
    video_encoding = video_encoding or VideoEncodingConfig()
    if resume and overwrite:
        raise ConversionError("resume and overwrite are mutually exclusive")
    output_path = collection.output_path
    if output_path.exists() and (resume or not overwrite):
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        temporary_path, state_path, lock_path = _resume_paths(output_path)
    else:
        temporary_path = output_path.with_name(f".{output_path.name}.incomplete-{uuid.uuid4().hex}")
        state_path = lock_path = None
    progress = EtaProgress(
        f"[{collection.dataset_uid}] convert",
        collection.num_frames,
        "frames",
        interval_seconds=eta_interval_seconds,
    )
    lock_context = _resume_lock(lock_path) if resume else contextlib.nullcontext()
    with lock_context:
        checkpoint_existed = temporary_path.exists()
        if resume:
            assert state_path is not None
            _prepare_resume_state(
                state_path,
                collection,
                video_encoding,
                checkpoint_exists=checkpoint_existed,
            )
            temporary_path.mkdir(parents=True, exist_ok=True)
        try:
            for partition in collection.partitions:
                temporary_plan = replace(
                    partition,
                    output_path=temporary_path / partition.output_path.name,
                )
                if resume and temporary_plan.output_path.exists():
                    _validate_written_dataset(
                        temporary_plan, temporary_plan.output_path, video_encoding
                    )
                    progress.mark_precompleted(temporary_plan.num_frames)
                    print(
                        f"[{partition.dataset_uid}] reusing completed partition checkpoint",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                convert_dataset(
                    temporary_plan,
                    resume=resume,
                    eta_interval_seconds=eta_interval_seconds,
                    progress=progress,
                    video_encoding=video_encoding,
                    episode_completed_hook=episode_completed_hook,
                )
            (temporary_path / "collection_manifest.json").write_text(
                json.dumps(
                    _collection_manifest(collection, video_encoding), ensure_ascii=False, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            # Nested partition locks are advisory files only. The collection
            # lock serializes all access while they are removed before publish.
            for nested_lock in temporary_path.glob(".*.resume.lock"):
                nested_lock.unlink(missing_ok=True)
            _publish_temporary_output(temporary_path, output_path, overwrite=overwrite)
            if resume:
                state_path.unlink(missing_ok=True)
            progress.finish(context="all partitions validated and published")
        except BaseException:
            if not resume and temporary_path.exists():
                shutil.rmtree(temporary_path)
            elif resume:
                print(
                    f"[{collection.dataset_uid}] collection checkpoint retained for --resume: "
                    f"{temporary_path}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path, help="Path to public_datasets_raw.")
    parser.add_argument("--staging-root", required=True, type=Path, help="Path to public_datasets_staging.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset-uid", help="Convert one dataset directory below --raw-root.")
    selection.add_argument("--all", action="store_true", help="Convert every dataset directory below --raw-root.")
    parser.add_argument("--inspect-only", action="store_true", help="Validate and print plans without writing output.")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip dataset UIDs whose v3 output already exists."
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite", action="store_true", help="Replace an existing UID only after new output validates."
    )
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep a deterministic checkpoint and resume at the next completed episode after "
            "an interruption. Source files and encoding arguments must remain unchanged."
        ),
    )
    parser.add_argument(
        "--fps",
        type=_positive_finite_float,
        default=DEFAULT_FPS,
        help=(
            "Fallback FPS used only when raw camera metadata is absent "
            f"(default: {DEFAULT_FPS:g}, matching the public Mobile ALOHA release)."
        ),
    )
    parser.add_argument(
        "--eta-interval-seconds",
        type=_positive_finite_float,
        default=DEFAULT_ETA_INTERVAL_SECONDS,
        help=(
            "Minimum seconds between ETA log lines "
            f"(default: {DEFAULT_ETA_INTERVAL_SECONDS:g}; episode completions always print)."
        ),
    )
    parser.add_argument(
        "--episode-limit",
        type=_positive_int,
        help=(
            "Inspect and convert only the first N naturally sorted episodes. "
            "Intended for representative smoke/performance tests, not full publication."
        ),
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Encode frames directly to video and bypass the temporary PNG round-trip.",
    )
    parser.add_argument(
        "--video-codec",
        choices=VIDEO_CODECS,
        default="libsvtav1",
        help="Video encoder (use h264_nvenc or hevc_nvenc only on NVENC-capable GPUs).",
    )
    parser.add_argument(
        "--video-quality",
        type=int,
        choices=range(0, 52),
        default=30,
        metavar="0..51",
        help="CRF for software codecs or constant QP for NVENC (default: 30).",
    )
    parser.add_argument(
        "--video-preset",
        help="Codec-specific preset; NVENC defaults to p4 when omitted.",
    )
    parser.add_argument(
        "--encoder-queue-maxsize",
        type=_positive_int,
        help=(
            "Streaming frames buffered per camera. Defaults to longest episode + 1; "
            "smaller values are rejected to make frame dropping impossible."
        ),
    )
    parser.add_argument(
        "--no-video-preflight",
        action="store_false",
        dest="video_preflight",
        help="Skip the real NVENC session probe (not recommended).",
    )
    parser.set_defaults(video_preflight=True)
    parser.add_argument("--state-key", default="/observations/qpos")
    parser.add_argument("--action-key", default="/action")
    parser.add_argument("--base-action-key", default="/base_action")
    parser.add_argument("--velocity-key", default="/observations/qvel")
    parser.add_argument("--effort-key", default="/observations/effort")
    parser.add_argument("--images-key", default="/observations/images")
    parser.add_argument(
        "--uncompressed-color-order",
        choices=("bgr", "rgb"),
        default="bgr",
        help=(
            "Channel order of raw uncompressed HDF5 camera arrays. Compressed JPEG/PNG frames are always "
            "decoded as RGB (default: bgr)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing and --overwrite cannot be used together")
    if args.skip_existing and args.resume:
        parser.error("--skip-existing and --resume cannot be used together")

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt("received SIGTERM")

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        video_encoding = VideoEncodingConfig(
            streaming=args.streaming_encoding,
            codec=args.video_codec,
            quality=args.video_quality,
            preset=args.video_preset,
            encoder_queue_maxsize=args.encoder_queue_maxsize,
            preflight=args.video_preflight,
        )
        uids = _dataset_uids(args.raw_root, args.dataset_uid, args.all)
        converted = 0
        skipped = 0
        for uid in uids:
            expected_output = args.staging_root / "lerobot_v3_0" / uid
            if expected_output.exists() and args.skip_existing and not args.inspect_only:
                print(f"[{uid}] skipped existing output: {expected_output}")
                skipped += 1
                continue
            collection = inspect_dataset_collection(
                raw_root=args.raw_root,
                staging_root=args.staging_root,
                dataset_uid=uid,
                state_key=args.state_key,
                action_key=args.action_key,
                base_action_key=args.base_action_key,
                velocity_key=args.velocity_key,
                effort_key=args.effort_key,
                images_key=args.images_key,
                fps=args.fps,
                uncompressed_color_order=args.uncompressed_color_order,
                eta_interval_seconds=args.eta_interval_seconds,
                episode_limit=args.episode_limit,
            )
            print(json.dumps(collection_summary(collection), ensure_ascii=False, indent=2))
            if not args.inspect_only:
                output = convert_collection(
                    collection,
                    overwrite=args.overwrite,
                    resume=args.resume,
                    eta_interval_seconds=args.eta_interval_seconds,
                    video_encoding=video_encoding,
                )
                print(
                    f"[{uid}] wrote {len(collection.partitions)} partitions / "
                    f"{collection.num_episodes} episodes / {collection.num_frames} frames to {output}"
                )
                converted += 1
        if args.inspect_only:
            print(f"validated {len(uids)} dataset(s); no output written")
        else:
            print(f"completed: converted={converted}, skipped={skipped}")
        return 0
    except KeyboardInterrupt:
        print(
            "interrupted; a durable checkpoint was retained if --resume was enabled",
            file=sys.stderr,
        )
        return 130
    except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    sys.exit(main())
