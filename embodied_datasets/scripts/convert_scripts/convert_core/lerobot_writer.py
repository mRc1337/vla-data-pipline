"""Generic LeRobot v3.0 writer: create -> add_frame -> save_episode -> finalize,
plus write-then-validate-then-atomically-publish, extracted from
``convert_mobile_aloha_to_lerobot.py``'s ``_write_dataset``/``_validate_written_dataset``/
``_publish_temporary_output``/``convert_dataset``. Every reader (hdf5/rlds/
raw_image_json) funnels through this one code path, so a fix or a new safety
check here benefits all of them at once instead of being copy-pasted per
format.

This module only knows about :mod:`convert_core.episode_spec`'s dataclasses;
it never imports h5py/tensorflow_datasets/PIL or looks at a source path.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Iterator
import uuid

from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError

IterFrames = Callable[[EpisodePlan], Iterator[dict[str, Any]]]


def plan_summary(plan: DatasetConversionPlan) -> dict[str, Any]:
    return {
        "dataset_uid": plan.dataset_uid,
        "output": str(plan.output_path),
        "robot_type": plan.robot_type,
        "fps": plan.fps,
        "measured_fps": plan.measured_fps,
        "episodes": len(plan.episodes),
        "frames": plan.num_frames,
        "tasks": sorted({episode.instruction for episode in plan.episodes}),
        "features": plan.feature_schema(),
    }


def build_manifest(plan: DatasetConversionPlan, *, reader_format: str) -> dict[str, Any]:
    return {
        "format": "lerobot_v3_0",
        "converter": "convert_dataset.py",
        "source_format": reader_format,
        "dataset_uid": plan.dataset_uid,
        "robot_type": plan.robot_type,
        "fps": plan.fps,
        "measured_fps": plan.measured_fps,
        "num_episodes": len(plan.episodes),
        "num_frames": plan.num_frames,
        "features": plan.feature_schema(),
        "episodes": [
            {
                "episode_uid": episode.episode_uid,
                "source": episode.source_relative_path,
                "instruction": episode.instruction,
                "num_frames": episode.num_frames,
            }
            for episode in plan.episodes
        ],
    }


def write_dataset(plan: DatasetConversionPlan, iter_frames: IterFrames, temporary_path: Path) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("lerobot==0.6.0 is required; install the project's requirements.txt") from exc

    dataset = LeRobotDataset.create(
        repo_id=plan.dataset_uid,
        fps=plan.fps,
        root=temporary_path,
        features=plan.feature_schema(),
        robot_type=plan.robot_type,
        use_videos=True,
    )
    try:
        for episode_index, episode in enumerate(plan.episodes):
            print(
                f"[{plan.dataset_uid}] episode {episode_index + 1}/{len(plan.episodes)}: "
                f"{episode.source_relative_path} ({episode.num_frames} frames)",
                file=sys.stderr,
            )
            for frame in iter_frames(episode):
                dataset.add_frame(frame)
            dataset.save_episode()
        dataset.finalize()
    except BaseException:
        # LeRobot owns its worker cleanup; keep the original conversion error.
        del dataset
        raise


def validate_written_dataset(plan: DatasetConversionPlan, temporary_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.dataset_uid, root=temporary_path)
    if dataset.num_episodes != len(plan.episodes):
        raise ConversionError(
            f"written dataset has {dataset.num_episodes} episodes, expected {len(plan.episodes)}"
        )
    if len(dataset) != plan.num_frames:
        raise ConversionError(f"written dataset has {len(dataset)} frames, expected {plan.num_frames}")
    missing = set(plan.feature_schema()) - set(dataset.meta.features)
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


def publish_temporary_output(temporary_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if not output_path.exists():
        temporary_path.rename(output_path)
        return
    if not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    backup_path = output_path.with_name(f".{output_path.name}.backup-{uuid.uuid4().hex}")
    output_path.rename(backup_path)
    try:
        temporary_path.rename(output_path)
    except BaseException:
        backup_path.rename(output_path)
        raise
    else:
        shutil.rmtree(backup_path, ignore_errors=True)


def convert_dataset(
    plan: DatasetConversionPlan,
    iter_frames: IterFrames,
    *,
    reader_format: str,
    overwrite: bool = False,
) -> Path:
    output_path = plan.output_path
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.incomplete-{uuid.uuid4().hex}")
    try:
        write_dataset(plan, iter_frames, temporary_path)
        validate_written_dataset(plan, temporary_path)
        (temporary_path / "conversion_manifest.json").write_text(
            json.dumps(build_manifest(plan, reader_format=reader_format), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_temporary_output(temporary_path, output_path, overwrite=overwrite)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
    return output_path


def resolve_dataset_uids(raw_root: Path, requested_uid: str | None, convert_all: bool) -> list[str]:
    if convert_all:
        if not raw_root.is_dir():
            raise ConversionError(f"raw root does not exist: {raw_root}")
        uids = sorted((path.name for path in raw_root.iterdir() if path.is_dir()), key=str.casefold)
        if not uids:
            raise ConversionError(f"raw root contains no dataset directories: {raw_root}")
        return uids
    if requested_uid is None:
        raise ConversionError("provide exactly one of --dataset-uid or --all")
    return [requested_uid]
