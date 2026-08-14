"""Convert raw Mobile ALOHA HDF5 episodes directly to LeRobot v3.0.

The converter is intentionally separate from ``process_scripts``: it only
ports raw values and RGB frames into staging. It does not clean, normalize,
resample, canonicalize, or upload the dataset.

Expected input layout::

    <raw-root>/<dataset-uid>/<language instruction>/episode_*.hdf5

Output is always written below::

    <staging-root>/lerobot_v3_0/<dataset-uid>

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
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
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


@dataclass(frozen=True)
class EpisodeSpec:
    source_path: Path
    source_relative_path: str
    instruction: str
    num_frames: int
    action_layout: str
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

    @property
    def num_frames(self) -> int:
        return sum(episode.num_frames for episode in self.episodes)


def _resolve_instruction(raw_dataset_root: Path, episode_path: Path) -> str:
    relative_parent = episode_path.parent.relative_to(raw_dataset_root)
    if relative_parent == Path("."):
        raise ConversionError(
            f"{episode_path}: episode is directly below the dataset root and has no instruction subdirectory"
        )
    return relative_parent.as_posix()


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
                raise ConversionError(
                    f"{episode_path}: 14-D arm action requires {_normalize_hdf5_key(base_action_key)} with 2 columns"
                )
            action_layout = "separate_14_plus_2"
        else:
            action_layout = "combined_16"
            if base_action is not None and not np.allclose(
                np.asarray(action[:, ARM_DIM:]), np.asarray(base_action[:]), rtol=1e-5, atol=1e-6
            ):
                raise ConversionError(
                    f"{episode_path}: combined action columns 14:16 disagree with {_normalize_hdf5_key(base_action_key)}"
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
        cameras=cameras,
        skipped_depth_keys=skipped_depth,
        has_velocity=optional_presence["velocity"],
        has_effort=optional_presence["effort"],
        fps_evidence=fps_evidence,
    )


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
    fps: float | None = None,
    uncompressed_color_order: str = "bgr",
) -> ConversionPlan:
    if Path(dataset_uid).name != dataset_uid or dataset_uid in {"", ".", ".."}:
        raise ConversionError(f"dataset UID must be one path component, got {dataset_uid!r}")
    if fps is not None and _as_finite_positive_float(fps) is None:
        raise ConversionError(f"--fps must be finite and positive, got {fps!r}")
    raw_dataset_root = raw_root / dataset_uid
    if not raw_dataset_root.is_dir():
        raise ConversionError(f"raw dataset directory does not exist: {raw_dataset_root}")
    episode_paths = sorted(
        (path for path in raw_dataset_root.rglob("*") if path.is_file() and path.suffix.casefold() in HDF5_SUFFIXES),
        key=_natural_sort_key,
    )
    if not episode_paths:
        raise ConversionError(f"no .h5 or .hdf5 episodes found below {raw_dataset_root}")

    episodes = tuple(
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
        for path in episode_paths
    )
    reference_cameras = tuple(
        (item.feature_key, item.height, item.width) for item in episodes[0].cameras
    )
    for episode in episodes[1:]:
        cameras = tuple((item.feature_key, item.height, item.width) for item in episode.cameras)
        if cameras != reference_cameras:
            raise ConversionError(
                f"{episode.source_path}: camera schema {cameras} differs from first episode schema {reference_cameras}"
            )
        if episode.has_velocity != episodes[0].has_velocity:
            raise ConversionError(f"{episode.source_path}: velocity feature presence differs across episodes")
        if episode.has_effort != episodes[0].has_effort:
            raise ConversionError(f"{episode.source_path}: effort feature presence differs across episodes")

    fps_values = [evidence.value for episode in episodes for evidence in episode.fps_evidence]
    measured_fps = float(np.median(fps_values))
    inconsistent = [value for value in fps_values if _relative_difference(value, measured_fps) > FPS_RELATIVE_TOLERANCE]
    if inconsistent:
        details = ", ".join(f"{value:.6g}" for value in fps_values)
        raise ConversionError(f"camera/episode FPS values differ by more than 2%: {details}")
    integer_fps = int(round(measured_fps))
    if integer_fps <= 0 or _relative_difference(measured_fps, integer_fps) > FPS_RELATIVE_TOLERANCE:
        raise ConversionError(
            f"measured FPS {measured_fps:.6g} is not within 2% of an integer; "
            "LeRobot video encoding requires an integer FPS and this converter does not resample"
        )

    return ConversionPlan(
        dataset_uid=dataset_uid,
        raw_dataset_root=raw_dataset_root,
        output_path=staging_root / "lerobot_v3_0" / dataset_uid,
        fps=integer_fps,
        measured_fps=measured_fps,
        state_key=_normalize_hdf5_key(state_key),
        action_key=_normalize_hdf5_key(action_key),
        base_action_key=_normalize_hdf5_key(base_action_key),
        velocity_key=_normalize_hdf5_key(velocity_key),
        effort_key=_normalize_hdf5_key(effort_key),
        images_key=_normalize_hdf5_key(images_key),
        uncompressed_color_order=uncompressed_color_order,
        episodes=episodes,
    )


def _feature_schema(plan: ConversionPlan) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ARM_DIM,),
            "names": [f"arm_{index}" for index in range(ARM_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ARM_DIM,),
            "names": [f"arm_{index}" for index in range(ARM_DIM)],
        },
        "action.base": {
            "dtype": "float32",
            "shape": (BASE_DIM,),
            "names": ["base_0", "base_1"],
        },
    }
    if plan.episodes[0].has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (ARM_DIM,),
            "names": [f"arm_{index}" for index in range(ARM_DIM)],
        }
    if plan.episodes[0].has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (ARM_DIM,),
            "names": [f"arm_{index}" for index in range(ARM_DIM)],
        }
    for camera in plan.episodes[0].cameras:
        features[camera.feature_key] = {
            "dtype": "video",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _manifest(plan: ConversionPlan) -> dict[str, Any]:
    return {
        "format": "lerobot_v3_0",
        "converter": "convert_mobile_aloha_to_lerobot.py",
        "dataset_uid": plan.dataset_uid,
        "raw_dataset_root": str(plan.raw_dataset_root.resolve()),
        "fps": plan.fps,
        "measured_fps": plan.measured_fps,
        "fps_relative_tolerance": FPS_RELATIVE_TOLERANCE,
        "num_episodes": len(plan.episodes),
        "num_frames": plan.num_frames,
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


def _episode_arrays(h5_file: Any, plan: ConversionPlan, episode: EpisodeSpec) -> dict[str, np.ndarray]:
    state = np.asarray(h5_file[plan.state_key][:], dtype=np.float32)
    raw_action = np.asarray(h5_file[plan.action_key][:], dtype=np.float32)
    if raw_action.shape[1] == ARM_DIM:
        arm_action = raw_action
        base_action = np.asarray(h5_file[plan.base_action_key][:], dtype=np.float32)
    else:
        arm_action = raw_action[:, :ARM_DIM]
        base_action = raw_action[:, ARM_DIM : ARM_DIM + BASE_DIM]
    arrays = {
        "observation.state": state,
        "action": arm_action,
        "action.base": base_action,
    }
    if episode.has_velocity:
        arrays["observation.velocity"] = np.asarray(h5_file[plan.velocity_key][:], dtype=np.float32)
    if episode.has_effort:
        arrays["observation.effort"] = np.asarray(h5_file[plan.effort_key][:], dtype=np.float32)
    return arrays


def _write_dataset(plan: ConversionPlan, temporary_path: Path) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("lerobot==0.6.0 is required; install the project's requirements.txt") from exc

    h5py = _require_h5py()
    dataset = LeRobotDataset.create(
        repo_id=plan.dataset_uid,
        fps=plan.fps,
        root=temporary_path,
        features=_feature_schema(plan),
        robot_type="mobile_aloha",
        use_videos=True,
    )
    try:
        for episode_index, episode in enumerate(plan.episodes):
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
            dataset.save_episode()
        dataset.finalize()
    except BaseException:
        # LeRobot owns its worker cleanup; keep the original conversion error.
        del dataset
        raise


def _validate_written_dataset(plan: ConversionPlan, temporary_path: Path) -> None:
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


def convert_dataset(plan: ConversionPlan, *, overwrite: bool = False) -> Path:
    output_path = plan.output_path
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.incomplete-{uuid.uuid4().hex}")
    try:
        _write_dataset(plan, temporary_path)
        _validate_written_dataset(plan, temporary_path)
        (temporary_path / "conversion_manifest.json").write_text(
            json.dumps(_manifest(plan), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish_temporary_output(temporary_path, output_path, overwrite=overwrite)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
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
    parser.add_argument("--skip-existing", action="store_true", help="Skip dataset UIDs whose v3 output already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing UID only after new output validates.")
    parser.add_argument("--fps", type=float, help="Fallback FPS used only when raw camera metadata is absent.")
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

    try:
        uids = _dataset_uids(args.raw_root, args.dataset_uid, args.all)
        converted = 0
        skipped = 0
        for uid in uids:
            expected_output = args.staging_root / "lerobot_v3_0" / uid
            if expected_output.exists() and args.skip_existing and not args.inspect_only:
                print(f"[{uid}] skipped existing output: {expected_output}")
                skipped += 1
                continue
            plan = inspect_dataset(
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
            )
            print(json.dumps(plan_summary(plan), ensure_ascii=False, indent=2))
            if not args.inspect_only:
                output = convert_dataset(plan, overwrite=args.overwrite)
                print(f"[{uid}] wrote {len(plan.episodes)} episodes / {plan.num_frames} frames to {output}")
                converted += 1
        if args.inspect_only:
            print(f"validated {len(uids)} dataset(s); no output written")
        else:
            print(f"completed: converted={converted}, skipped={skipped}")
        return 0
    except (ConversionError, FileExistsError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
