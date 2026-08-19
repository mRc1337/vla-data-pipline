#!/usr/bin/env python3
"""Convert the heterogeneous RoboVerse v2 trajectory release to LeRobot v3.

The released files contain control/state trajectories, not rendered camera
observations.  The converter therefore never fabricates images and never runs
a video encoder.  Files with unequal action/state stream lengths are published
as linked action and state parts, preserving every sample without padding,
truncation, or an invented alignment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import json
import math
import multiprocessing
from pathlib import Path
import signal
import shutil
import sys
from typing import Any, Iterable
import uuid

import numpy as np

from convert_core.checkpoint import (
    RESUME_SCHEMA_VERSION,
    atomic_write_json,
    canonical_fingerprint,
    exclusive_resume_lock,
    read_json_object,
    resume_paths,
)
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan, VectorFeatureSpec
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import (
    publish_temporary_output,
    validate_written_dataset,
    write_dataset,
)
from convert_core.progress import EtaProgress
from readers.roboverse_v2_reader import (
    SOURCE_DATASET,
    SOURCE_REVISION,
    CollectionPlan,
    FeatureColumn,
    NumericStatistics,
    PartPlan,
    SourceEpisode,
    column_array,
    dtype_promotion_dict,
    inspect_collection,
    iter_part_frames,
    load_source_file,
    merge_numeric_statistics,
    numeric_statistics_dict,
)


CONVERTER_VERSION = 10
ORDINAL_FPS = 1
STATE_FILE = "state.json"
MARKERS_DIR = "parts"
LEROBOT_CACHE_DIR = ".lerobot-datasets-cache"


def _parse_fps_overrides(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ConversionError(f"FPS override must be SUITE=INTEGER, got {value!r}")
        suite, raw_fps = value.split("=", 1)
        suite = suite.strip()
        try:
            fps = int(raw_fps)
        except ValueError as exc:
            raise ConversionError(f"FPS override must be an integer, got {value!r}") from exc
        if not suite or fps <= 0 or str(fps) != raw_fps.strip():
            raise ConversionError(f"FPS override must be SUITE=positive-integer, got {value!r}")
        if suite in result and result[suite] != fps:
            raise ConversionError(f"conflicting FPS overrides for {suite!r}")
        result[suite] = fps
    return result


def _part_fps(part: PartPlan, overrides: dict[str, int], allow_ordinal: bool) -> tuple[int, str]:
    if part.source_suite in overrides:
        return overrides[part.source_suite], "user_override_without_source_timestamps"
    if not allow_ordinal:
        raise ConversionError(
            f"source suite {part.source_suite!r} has no timestamps/FPS in the downloaded release; "
            "conversion requires either --fps-override SUITE=INTEGER with external evidence or "
            "--allow-ordinal-timebase to encode source step_index as timestamp at nominal fps=1"
        )
    return ORDINAL_FPS, "ordinal_source_step_index"


def _part_dataset_uid(dataset_uid: str, part: PartPlan) -> str:
    return f"{dataset_uid}/{part.part_id}"


def _episode_plan(part: PartPlan, episode: SourceEpisode) -> EpisodePlan:
    return EpisodePlan(
        episode_uid=episode.episode_uid,
        source_relative_path=episode.source_file.relative_path,
        instruction=episode.task_text,
        num_frames=part.episode_frames(episode),
        extra={
            "source_episode_index": episode.source_episode_index,
            "source_robot": episode.robot_name,
            "source_split": episode.source_split,
            "source_suite": episode.source_suite,
            "source_task": episode.source_task,
            "source_task_origin": episode.task_origin,
            "stream_kind": part.stream_kind,
        },
    )


def _dataset_plan(
    part: PartPlan,
    output_path: Path,
    dataset_uid: str,
    fps_overrides: dict[str, int],
    allow_ordinal: bool,
) -> DatasetConversionPlan:
    fps, time_basis = _part_fps(part, fps_overrides, allow_ordinal)
    vectors = tuple(
        VectorFeatureSpec(
            feature_key=column.feature_key,
            dim=int(math.prod(column.shape)),
            names=column.names,
            dtype=column.dtype,
            shape=column.shape,
        )
        for column in part.feature_columns
    )
    return DatasetConversionPlan(
        dataset_uid=_part_dataset_uid(dataset_uid, part),
        output_path=output_path,
        fps=fps,
        measured_fps=float(fps),
        robot_type=part.robot_name,
        vector_features=vectors,
        camera_features=(),
        episodes=tuple(_episode_plan(part, episode) for episode in part.episodes),
        extra={
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "time_basis": time_basis,
            "stream_kind": part.stream_kind,
        },
    )


def _column_manifest(column: FeatureColumn) -> dict[str, Any]:
    return {
        "source_path": list(column.source_path),
        "target_key": column.feature_key,
        "shape": list(column.shape),
        "dtype": column.dtype,
        "names": list(column.names) if column.names is not None else None,
        "source_components": (
            list(column.source_components) if column.source_components is not None else None
        ),
        "source_component_indices": (
            list(column.source_component_indices)
            if column.source_component_indices is not None
            else None
        ),
        "source_component_shapes": (
            [list(shape) for shape in column.source_component_shapes]
            if column.source_component_shapes is not None
            else None
        ),
        "split_reason": column.split_reason,
        "source_dtype_options": (
            [list(options) for options in column.source_dtype_options]
            if column.source_dtype_options is not None
            else None
        ),
        "transform": (
            "lossless_exact_dtype_promotion"
            if column.source_dtype_options is not None
            else "identity_split_named_mapping"
            if column.split_named_mapping
            else "identity_flatten_named_mapping"
            if column.source_components is not None
            else "identity_array"
        ),
        "lossy": False,
    }


def _episode_dtype_promotions(
    part: PartPlan,
    episode: SourceEpisode,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if part.stream_kind in {"action", "aligned"}:
        result.extend(
            {"stream": "action", **dtype_promotion_dict(item)}
            for item in episode.action_dtype_promotions
        )
    if part.stream_kind in {"state", "aligned"}:
        result.extend(
            {"stream": "state", **dtype_promotion_dict(item)}
            for item in episode.state_dtype_promotions
        )
    return result


def _feature_statistics_manifest(
    column: FeatureColumn,
    statistics: NumericStatistics,
) -> dict[str, Any]:
    return {
        "target_key": column.feature_key,
        "dtype": column.dtype,
        "shape": list(column.shape),
        "names": list(column.names) if column.names is not None else None,
        **numeric_statistics_dict(statistics),
    }


def _written_task_mapping(output_path: Path, dataset_uid: str) -> dict[str, str]:
    """Read the task indices LeRobot actually assigned instead of guessing them."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=dataset_uid, root=output_path)
    mapping = {
        str(int(row["task_index"])): str(task)
        for task, row in dataset.meta.tasks.iterrows()
    }
    del dataset
    return dict(sorted(mapping.items(), key=lambda item: int(item[0])))


def _part_manifest(
    part: PartPlan,
    plan: DatasetConversionPlan,
    time_basis: str,
    task_index_mapping: dict[str, str],
) -> dict[str, Any]:
    return {
        "format": "lerobot_v3_0",
        "converter": Path(__file__).name,
        "converter_version": CONVERTER_VERSION,
        "source_dataset": SOURCE_DATASET,
        "source_revision": plan.extra.get("source_revision"),
        "source_format": "roboverse_v2_pickle",
        "part_id": part.part_id,
        "source_suite": part.source_suite,
        "robot_type": part.robot_name,
        "stream_kind": part.stream_kind,
        "fps": plan.fps,
        "time_basis": time_basis,
        "physical_timestamps_available": False,
        "num_episodes": len(part.episodes),
        "num_frames": part.num_frames,
        "features": {
            key: value | {"shape": list(value["shape"])}
            for key, value in plan.feature_schema().items()
        },
        "field_mapping": [_column_manifest(column) for column in part.feature_columns],
        "source_numeric_statistics": [
            _feature_statistics_manifest(column, statistics)
            for column, statistics in zip(
                part.feature_columns,
                part.feature_statistics,
                strict=True,
            )
        ],
        "unrepresented_empty_state_fields": [list(path) for path in part.empty_state_fields],
        "episodes": [
            {
                "target_episode_index": index,
                "source_episode_uid": episode.episode_uid,
                "source_relative_path": episode.source_file.relative_path,
                "source_robot": episode.robot_name,
                "source_episode_index": episode.source_episode_index,
                "source_split": episode.source_split,
                "source_task": episode.source_task,
                "source_task_origin": episode.task_origin,
                "source_action_count": episode.action_count,
                "source_state_count": episode.state_count,
                "source_empty_state_fields": [list(path) for path in episode.state_empty_fields],
                "target_length": part.episode_frames(episode),
                "lossless_dtype_promotions": _episode_dtype_promotions(part, episode),
            }
            for index, episode in enumerate(part.episodes)
        ],
        "task_index_mapping": task_index_mapping,
        "dtype_casts": [
            {
                "source_episode_uid": episode.episode_uid,
                **promotion,
            }
            for episode in part.episodes
            for promotion in _episode_dtype_promotions(part, episode)
        ],
        "reordered_fields": [],
        "dropped_fields": [],
        "video": {"present_in_source": False, "encoded": False, "remuxed": False},
    }


def _source_frame(part: PartPlan, episode: SourceEpisode, frame_index: int) -> dict[str, np.ndarray]:
    data = load_source_file(episode.source_file.path)
    raw_episode = data[episode.robot_name][episode.source_episode_index]
    if part.stream_kind == "state":
        primary = raw_episode["states"][frame_index]
    else:
        primary = raw_episode["actions"][frame_index]
    result: dict[str, np.ndarray] = {}
    if part.stream_kind == "aligned":
        state = raw_episode["states"][frame_index]
        for column in part.feature_columns:
            container = primary if column.feature_key.startswith("action") else state
            result[column.feature_key] = column_array(container, column)
    else:
        for column in part.feature_columns:
            result[column.feature_key] = column_array(primary, column)
    return result


def _validate_part_samples(part: PartPlan, plan: DatasetConversionPlan, output_path: Path) -> None:
    """Compare stored Arrow values against first/middle/last source samples."""

    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.dataset_uid, root=output_path)
    info = read_json_object(output_path / "meta" / "info.json", "LeRobot info metadata")
    written_statistics = read_json_object(
        output_path / "meta" / "stats.json",
        "LeRobot statistics metadata",
    )
    if info.get("codebase_version") != "v3.0":
        raise ConversionError(
            f"{part.part_id}: codebase_version is {info.get('codebase_version')!r}, expected 'v3.0'"
        )
    if int(info.get("total_episodes", -1)) != len(part.episodes) or int(
        info.get("total_frames", -1)
    ) != part.num_frames:
        raise ConversionError(f"{part.part_id}: info.json episode/frame totals are incorrect")
    for column in part.feature_columns:
        actual_names = dataset.meta.features[column.feature_key].get("names")
        expected_names = list(column.names) if column.names is not None else None
        if actual_names != expected_names:
            raise ConversionError(
                f"{part.part_id}: written names for {column.feature_key} are "
                f"{actual_names!r}, expected {expected_names!r}"
            )
    for column, source_statistics in zip(
        part.feature_columns,
        part.feature_statistics,
        strict=True,
    ):
        row = written_statistics.get(column.feature_key)
        if not isinstance(row, dict):
            raise ConversionError(
                f"{part.part_id}: stats.json has no entry for {column.feature_key}"
            )
        source_width = int(math.prod(column.shape))
        # LeRobot/Hugging Face stores ArrayND statistics as one aggregate
        # value across all array components.  One-dimensional vectors retain
        # component-wise statistics.  Full component-wise source statistics
        # remain in the conversion and collection manifests in both cases.
        written_width = 1 if len(column.shape) >= 2 else source_width
        flattened_ranges: dict[str, np.ndarray] = {}
        for key in ("min", "max"):
            value = row.get(key)
            if not isinstance(value, list):
                raise ConversionError(
                    f"{part.part_id}: stats.json {column.feature_key}.{key} has invalid width"
                )
            flattened = np.asarray(value).reshape(-1)
            if flattened.size != written_width:
                raise ConversionError(
                    f"{part.part_id}: stats.json {column.feature_key}.{key} has invalid width"
                )
            flattened_ranges[key] = flattened
        if row.get("count") != [source_statistics.frames]:
            raise ConversionError(
                f"{part.part_id}: stats.json {column.feature_key}.count is "
                f"{row.get('count')!r}, expected {[source_statistics.frames]}"
            )
        if len(column.shape) >= 2:
            nonfinite = sum(source_statistics.nan_count) + sum(
                source_statistics.positive_infinity_count
            ) + sum(source_statistics.negative_infinity_count)
            finite_min = [value for value in source_statistics.finite_min if value is not None]
            finite_max = [value for value in source_statistics.finite_max if value is not None]
            if nonfinite == 0 and finite_min and finite_max and (
                not np.array_equal(flattened_ranges["min"][0], min(finite_min))
                or not np.array_equal(flattened_ranges["max"][0], max(finite_max))
            ):
                raise ConversionError(
                    f"{part.part_id}: stats.json aggregate range for {column.feature_key} "
                    "does not match the full source scan"
                )
            continue
        for component_index in range(source_width):
            nonfinite = sum(
                counts[component_index]
                for counts in (
                    source_statistics.nan_count,
                    source_statistics.positive_infinity_count,
                    source_statistics.negative_infinity_count,
                )
            )
            if nonfinite == 0 and (
                not np.array_equal(
                    np.asarray(flattened_ranges["min"][component_index]),
                    np.asarray(source_statistics.finite_min[component_index]),
                )
                or not np.array_equal(
                    np.asarray(flattened_ranges["max"][component_index]),
                    np.asarray(source_statistics.finite_max[component_index]),
                )
            ):
                raise ConversionError(
                    f"{part.part_id}: stats.json range for {column.feature_key} "
                    f"component {component_index} differs from the full source scan"
                )
    if list(output_path.rglob("*.mp4")):
        raise ConversionError(f"{part.part_id}: videos were written although the source has no cameras")
    episode_indices = sorted({0, len(part.episodes) // 2, len(part.episodes) - 1})
    for target_episode_index in episode_indices:
        episode = part.episodes[target_episode_index]
        length = part.episode_frames(episode)
        metadata = dataset.meta.episodes[target_episode_index]
        data_path = output_path / dataset.meta.get_data_file_path(target_episode_index)
        table = pq.read_table(
            data_path,
            columns=[
                "index",
                "frame_index",
                "episode_index",
                "timestamp",
                "task_index",
                *[column.feature_key for column in part.feature_columns],
            ],
        )
        indices = table["index"].to_numpy(zero_copy_only=False)
        global_start = int(metadata["dataset_from_index"])
        expected_task_index = int(dataset.meta.tasks.loc[episode.task_text]["task_index"])
        for local_index in sorted({0, length // 2, length - 1}):
            positions = np.flatnonzero(indices == global_start + local_index)
            if len(positions) != 1:
                raise ConversionError(
                    f"{part.part_id}: cannot locate stored episode={target_episode_index} frame={local_index}"
                )
            stored_row = int(positions[0])
            if int(table["frame_index"][stored_row].as_py()) != local_index:
                raise ConversionError(f"{part.part_id}: stored frame_index is incorrect")
            if int(table["episode_index"][stored_row].as_py()) != target_episode_index:
                raise ConversionError(f"{part.part_id}: stored episode_index is incorrect")
            if int(table["task_index"][stored_row].as_py()) != expected_task_index:
                raise ConversionError(f"{part.part_id}: stored task_index is incorrect")
            stored_timestamp = float(table["timestamp"][stored_row].as_py())
            expected_timestamp = local_index / plan.fps
            if not math.isclose(stored_timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1e-6):
                raise ConversionError(
                    f"{part.part_id}: timestamp {stored_timestamp} differs from {expected_timestamp}"
                )
            expected = _source_frame(part, episode, local_index)
            for column in part.feature_columns:
                stored = np.asarray(table[column.feature_key][stored_row].as_py()).reshape(-1)
                source = expected[column.feature_key].reshape(-1)
                if not np.array_equal(stored, source, equal_nan=True):
                    raise ConversionError(
                        f"{part.part_id}: stored {column.feature_key} differs at "
                        f"episode={target_episode_index} frame={local_index}"
                    )
    del dataset


def _convert_part(
    part: PartPlan,
    output_path: Path,
    dataset_uid: str,
    fps_overrides: dict[str, int],
    allow_ordinal: bool,
    eta_interval_seconds: float,
) -> tuple[str, int, int]:
    plan = _dataset_plan(part, output_path, dataset_uid, fps_overrides, allow_ordinal)
    progress = EtaProgress(
        f"part:{part.part_id}",
        part.num_frames,
        "frame",
        interval_seconds=eta_interval_seconds,
    )
    completed = 0

    def iter_frames(ep_plan: EpisodePlan):
        nonlocal completed
        episode = next(item for item in part.episodes if item.episode_uid == ep_plan.episode_uid)
        for frame in iter_part_frames(part, episode):
            completed += 1
            progress.update(
                completed,
                context=(
                    f"suite={part.source_suite} part={part.part_id} "
                    f"split={episode.source_split} task={episode.source_task} "
                    f"episode={episode.source_episode_index}"
                ),
            )
            yield frame

    try:
        write_dataset(plan, iter_frames, output_path)
        validate_written_dataset(plan, output_path)
        _validate_part_samples(part, plan, output_path)
        _, time_basis = _part_fps(part, fps_overrides, allow_ordinal)
        task_mapping = _written_task_mapping(output_path, plan.dataset_uid)
        atomic_write_json(
            output_path / "conversion_manifest.json",
            _part_manifest(part, plan, time_basis, task_mapping),
        )
        progress.finish(context=f"suite={part.source_suite} part={part.part_id}")
    except BaseException:
        if output_path.exists():
            shutil.rmtree(output_path)
        raise
    return part.part_id, len(part.episodes), part.num_frames


def _fingerprint_payload(
    collection: CollectionPlan,
    *,
    dataset_uid: str,
    fps_overrides: dict[str, int],
    allow_ordinal: bool,
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "converter_version": CONVERTER_VERSION,
        "source_dataset": SOURCE_DATASET,
        "source_revision": collection.source_revision,
        "source_root": str(collection.source_root),
        "dataset_uid": dataset_uid,
        "selection": selection,
        "timebase": {
            "allow_ordinal": allow_ordinal,
            "ordinal_fps": ORDINAL_FPS,
            "fps_overrides": fps_overrides,
        },
        "video": {"present": False, "codec": None, "preset": None, "remux": False},
        "files": [asdict(source) | {"path": str(source.path)} for source in collection.files],
        "source_issues": list(collection.source_issues),
        "source_sidecar_payloads": list(collection.sidecar_payloads),
        "auxiliary_source_files": list(collection.auxiliary_files),
        "calvin_instruction_aliases": {
            key: list(value) for key, value in collection.calvin_instruction_aliases.items()
        },
        "parts": [
            {
                "part_id": part.part_id,
                "suite": part.source_suite,
                "robot": part.robot_name,
                "stream_kind": part.stream_kind,
                "features": [_column_manifest(column) for column in part.feature_columns],
                "source_numeric_statistics": [
                    _feature_statistics_manifest(column, statistics)
                    for column, statistics in zip(
                        part.feature_columns,
                        part.feature_statistics,
                        strict=True,
                    )
                ],
                "empty_state_fields": [list(path) for path in part.empty_state_fields],
                "episodes": [
                    {
                        "uid": episode.episode_uid,
                        "source": episode.source_file.relative_path,
                        "source_episode_index": episode.source_episode_index,
                        "action_count": episode.action_count,
                        "state_count": episode.state_count,
                        "empty_state_fields": [list(path) for path in episode.state_empty_fields],
                        "task": episode.source_task,
                        "task_origin": episode.task_origin,
                        "split": episode.source_split,
                        "lossless_dtype_promotions": _episode_dtype_promotions(
                            part,
                            episode,
                        ),
                    }
                    for episode in part.episodes
                ],
            }
            for part in collection.parts
        ],
        "partition_rule": "source_file+robot+stream_alignment+ordered_feature_schema",
    }


def _marker_payload(fingerprint: str, part: PartPlan) -> dict[str, Any]:
    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "collection_fingerprint": fingerprint,
        "part_id": part.part_id,
        "episodes": len(part.episodes),
        "frames": part.num_frames,
        "schema": canonical_fingerprint({"features": [_column_manifest(c) for c in part.feature_columns]}),
    }


def _changed_fingerprint_sections(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    return sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))


def _prepare_resume(
    collection: CollectionPlan,
    data_root: Path,
    state_root: Path,
    fingerprint_payload: dict[str, Any],
    dataset_uid: str,
    fps_overrides: dict[str, int],
    allow_ordinal: bool,
) -> tuple[list[PartPlan], int, int]:
    fingerprint = canonical_fingerprint(fingerprint_payload)
    expected_state = {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "collection_fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
    }
    state_path = state_root / STATE_FILE
    if state_root.exists():
        if not state_root.is_dir() or not state_path.is_file():
            raise ConversionError(f"resume state is incomplete: {state_root}")
        actual = read_json_object(state_path, "RoboVerse resume state")
        if actual != expected_state:
            changed = _changed_fingerprint_sections(
                actual.get("fingerprint_payload", {}), fingerprint_payload
            )
            raise ConversionError(
                f"resume checkpoint fingerprint mismatch in sections {changed}: {state_root}; "
                "use the original source/configuration or move the checkpoint aside"
            )
    else:
        state_root.mkdir(parents=True)
        atomic_write_json(state_path, expected_state)
    if data_root.exists() and not data_root.is_dir():
        raise ConversionError(f"resume data path is not a directory: {data_root}")
    (data_root / "parts").mkdir(parents=True, exist_ok=True)
    markers = state_root / MARKERS_DIR
    markers.mkdir(exist_ok=True)
    expected_ids = {part.part_id for part in collection.parts}
    unexpected_markers = sorted(path.name for path in markers.glob("*.json") if path.stem not in expected_ids)
    if unexpected_markers:
        raise ConversionError(f"resume state contains unexpected part markers: {unexpected_markers}")
    unexpected_parts = sorted(
        path.name
        for path in (data_root / "parts").iterdir()
        if path.name not in expected_ids and path.name != LEROBOT_CACHE_DIR
    )
    if unexpected_parts:
        raise ConversionError(f"resume data contains unexpected parts: {unexpected_parts}")
    pending: list[PartPlan] = []
    reused_episodes = 0
    reused_frames = 0
    for part in collection.parts:
        output = data_root / "parts" / part.part_id
        marker = markers / f"{part.part_id}.json"
        expected_marker = _marker_payload(fingerprint, part)
        valid_marker = False
        if marker.is_file():
            try:
                valid_marker = read_json_object(marker, "RoboVerse part marker") == expected_marker
            except ConversionError:
                valid_marker = False
        if valid_marker and output.is_dir():
            try:
                plan = _dataset_plan(part, output, dataset_uid, fps_overrides, allow_ordinal)
                validate_written_dataset(plan, output)
                _validate_part_samples(part, plan, output)
                _, time_basis = _part_fps(part, fps_overrides, allow_ordinal)
                expected_manifest = _part_manifest(
                    part,
                    plan,
                    time_basis,
                    _written_task_mapping(output, plan.dataset_uid),
                )
                if read_json_object(
                    output / "conversion_manifest.json",
                    f"RoboVerse part manifest for {part.part_id}",
                ) != expected_manifest:
                    raise ConversionError("part manifest does not match the checkpoint plan")
            except Exception as exc:
                print(f"[resume] invalid part {part.part_id}; rebuilding: {exc}", file=sys.stderr, flush=True)
            else:
                reused_episodes += len(part.episodes)
                reused_frames += part.num_frames
                continue
        marker.unlink(missing_ok=True)
        if output.exists():
            shutil.rmtree(output)
        pending.append(part)
    print(
        f"[resume] reused {len(collection.parts)-len(pending)}/{len(collection.parts)} verified parts, "
        f"{reused_episodes} episodes, {reused_frames} frames",
        file=sys.stderr,
        flush=True,
    )
    return pending, reused_episodes, reused_frames


def _iter_source_episode_rows(collection: CollectionPlan) -> Iterable[dict[str, Any]]:
    by_uid: dict[str, SourceEpisode] = {}
    output_parts: dict[str, list[dict[str, Any]]] = {}
    for part in collection.parts:
        for target_index, episode in enumerate(part.episodes):
            by_uid[episode.episode_uid] = episode
            output_parts.setdefault(episode.episode_uid, []).append(
                {
                    "part_id": part.part_id,
                    "stream_kind": part.stream_kind,
                    "target_episode_index": target_index,
                    "target_length": part.episode_frames(episode),
                }
            )
    for uid, episode in sorted(by_uid.items(), key=lambda item: (
        item[1].source_file.relative_path,
        item[1].robot_name,
        item[1].source_episode_index,
    )):
        yield {
            "source_episode_uid": uid,
            "source_relative_path": episode.source_file.relative_path,
            "source_robot": episode.robot_name,
            "source_episode_index": episode.source_episode_index,
            "source_suite": episode.source_suite,
            "source_task": episode.source_task,
            "source_task_origin": episode.task_origin,
            "source_split": episode.source_split,
            "source_action_count": episode.action_count,
            "source_state_count": episode.state_count,
            "source_empty_state_fields": [list(path) for path in episode.state_empty_fields],
            "lossless_dtype_promotions": [
                {"stream": "action", **dtype_promotion_dict(item)}
                for item in episode.action_dtype_promotions
            ]
            + [
                {"stream": "state", **dtype_promotion_dict(item)}
                for item in episode.state_dtype_promotions
            ],
            "output_parts": output_parts[uid],
            "static_source_payload": episode.static_payload,
        }


def _write_source_episode_index(collection: CollectionPlan, output_root: Path) -> None:
    path = output_root / "source_episodes.jsonl"
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in _iter_source_episode_rows(collection):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _collection_manifest(
    collection: CollectionPlan,
    *,
    dataset_uid: str,
    fps_overrides: dict[str, int],
    allow_ordinal: bool,
) -> dict[str, Any]:
    mismatched = sorted(
        {
            episode.episode_uid
            for part in collection.parts
            for episode in part.episodes
            if episode.action_count != episode.state_count and episode.action_count and episode.state_count
        }
    )
    return {
        "format": "lerobot_v3_0_collection",
        "converter": Path(__file__).name,
        "converter_version": CONVERTER_VERSION,
        "dataset_uid": dataset_uid,
        "source_dataset": SOURCE_DATASET,
        "source_revision": collection.source_revision,
        "source_root": str(collection.source_root),
        "source_files": [
            {
                "relative_path": source.relative_path,
                "size": source.size,
                "mtime_ns": source.mtime_ns,
            }
            for source in collection.files
        ],
        "source_sidecars_not_trajectories": list(collection.sidecars),
        "source_sidecar_payloads": list(collection.sidecar_payloads),
        "auxiliary_source_files": list(collection.auxiliary_files),
        "source_issues": list(collection.source_issues),
        "duplicate_source_aliases": [
            {"alias": alias, "canonical": canonical} for alias, canonical in collection.duplicate_aliases
        ],
        "parts": [
            {
                "part_id": part.part_id,
                "path": f"parts/{part.part_id}",
                "source_suite": part.source_suite,
                "robot_type": part.robot_name,
                "stream_kind": part.stream_kind,
                "episodes": len(part.episodes),
                "frames": part.num_frames,
                "fps": _part_fps(part, fps_overrides, allow_ordinal)[0],
                "time_basis": _part_fps(part, fps_overrides, allow_ordinal)[1],
                "features": [_column_manifest(column) for column in part.feature_columns],
                "source_numeric_statistics": [
                    _feature_statistics_manifest(column, statistics)
                    for column, statistics in zip(
                        part.feature_columns,
                        part.feature_statistics,
                        strict=True,
                    )
                ],
                "unrepresented_empty_state_fields": [list(path) for path in part.empty_state_fields],
            }
            for part in collection.parts
        ],
        "source_episode_count": collection.num_source_episodes,
        "output_episode_count": collection.num_output_episodes,
        "output_frame_count": collection.num_output_frames,
        "partition_rule": "source_file+robot+stream_alignment+ordered_feature_schema",
        "mismatched_stream_episode_uids": mismatched,
        "stream_alignment_policy": (
            "equal-length action/state streams share frame indices; unequal streams are separate linked parts"
        ),
        "timebase": {
            "physical_timestamps_available": False,
            "ordinal_timebase_enabled": allow_ordinal,
            "ordinal_fps": ORDINAL_FPS if allow_ordinal else None,
            "fps_overrides": fps_overrides,
        },
        "calvin_instruction_aliases": {
            key: list(value) for key, value in collection.calvin_instruction_aliases.items()
        },
        "source_episode_index": "source_episodes.jsonl",
        "semantic_changes": {
            "field_drops": [],
            "empty_state_fields": (
                "explicit null/empty mappings carry no numeric samples; their paths are retained in manifests"
            ),
            "dtype_casts": [
                {
                    "source_episode_uid": episode.episode_uid,
                    "stream": stream,
                    **dtype_promotion_dict(promotion),
                }
                for episode in {
                    episode.episode_uid: episode
                    for part in collection.parts
                    for episode in part.episodes
                }.values()
                for stream, promotions in (
                    ("action", episode.action_dtype_promotions),
                    ("state", episode.state_dtype_promotions),
                )
                for promotion in promotions
            ],
            "reorders": [],
            "normalization": [],
            "video_reencoding": [],
            "task_index": "LeRobot-local deterministic mapping recorded in every part manifest",
            "partitions": "required by robot, ordered schema, source file, and action/state alignment",
        },
    }


def _validate_collection(collection: CollectionPlan, root: Path, dataset_uid: str, fps_overrides: dict[str, int], allow_ordinal: bool) -> None:
    for part in collection.parts:
        output = root / "parts" / part.part_id
        plan = _dataset_plan(part, output, dataset_uid, fps_overrides, allow_ordinal)
        validate_written_dataset(plan, output)
        _validate_part_samples(part, plan, output)
        part_manifest_path = output / "conversion_manifest.json"
        actual_part_manifest = read_json_object(
            part_manifest_path, f"RoboVerse part manifest for {part.part_id}"
        )
        _, time_basis = _part_fps(part, fps_overrides, allow_ordinal)
        expected_part_manifest = _part_manifest(
            part,
            plan,
            time_basis,
            _written_task_mapping(output, plan.dataset_uid),
        )
        if actual_part_manifest != expected_part_manifest:
            raise ConversionError(f"part manifest does not match validated data: {part.part_id}")
    manifest_path = root / "collection_manifest.json"
    source_index_path = root / "source_episodes.jsonl"
    if not manifest_path.is_file() or not source_index_path.is_file():
        raise ConversionError(f"collection metadata is incomplete: {root}")
    try:
        with source_index_path.open("r", encoding="utf-8") as handle:
            expected_rows = iter(_iter_source_episode_rows(collection))
            line_number = 0
            while True:
                line = handle.readline()
                expected_row = next(expected_rows, None)
                if not line and expected_row is None:
                    break
                line_number += 1
                if not line or expected_row is None or json.loads(line) != expected_row:
                    raise ConversionError(
                        f"source episode index does not match the validated conversion plan at line {line_number}"
                    )
    except json.JSONDecodeError as exc:
        raise ConversionError(f"cannot read source episode index {source_index_path}: {exc}") from exc
    except OSError as exc:
        raise ConversionError(f"cannot read source episode index {source_index_path}: {exc}") from exc
    manifest = read_json_object(manifest_path, "RoboVerse collection manifest")
    expected = _collection_manifest(
        collection,
        dataset_uid=dataset_uid,
        fps_overrides=fps_overrides,
        allow_ordinal=allow_ordinal,
    )
    if manifest != expected:
        raise ConversionError("collection manifest does not match the validated conversion plan")


def convert_collection(
    collection: CollectionPlan,
    *,
    output_path: Path,
    dataset_uid: str,
    resume: bool,
    overwrite: bool,
    skip_existing: bool,
    fps_overrides: dict[str, int],
    allow_ordinal: bool,
    eta_interval_seconds: float,
    workers: int,
    selection: dict[str, Any],
) -> Path:
    if workers <= 0:
        raise ConversionError("workers must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data, checkpoint_state, lock_path = resume_paths(output_path)
    fingerprint_payload = _fingerprint_payload(
        collection,
        dataset_uid=dataset_uid,
        fps_overrides=fps_overrides,
        allow_ordinal=allow_ordinal,
        selection=selection,
    )
    fingerprint = canonical_fingerprint(fingerprint_payload)
    with exclusive_resume_lock(lock_path):
        if output_path.exists():
            if skip_existing or resume:
                _validate_collection(collection, output_path, dataset_uid, fps_overrides, allow_ordinal)
                shutil.rmtree(output_path / "parts" / LEROBOT_CACHE_DIR, ignore_errors=True)
                if resume:
                    if checkpoint_data.exists():
                        raise ConversionError(
                            f"validated final output and resume data both exist: "
                            f"{output_path} and {checkpoint_data}; move the ambiguous resume data aside"
                        )
                    if checkpoint_state.exists():
                        if not checkpoint_state.is_dir():
                            raise ConversionError(
                                f"resume state beside validated output is not a directory: {checkpoint_state}"
                            )
                        shutil.rmtree(checkpoint_state)
                print(f"[skip] verified existing RoboVerse output: {output_path}", file=sys.stderr, flush=True)
                return output_path
            if not overwrite:
                raise FileExistsError(f"output already exists: {output_path}")
        if resume:
            temporary = checkpoint_data
            pending, reused_episodes, reused_frames = _prepare_resume(
                collection,
                checkpoint_data,
                checkpoint_state,
                fingerprint_payload,
                dataset_uid,
                fps_overrides,
                allow_ordinal,
            )
        else:
            temporary = output_path.with_name(f".{output_path.name}.incomplete-{uuid.uuid4().hex}")
            temporary.mkdir()
            (temporary / "parts").mkdir()
            pending = list(collection.parts)
            reused_episodes = reused_frames = 0
        progress = EtaProgress(
            "roboverse-convert",
            collection.num_output_frames,
            "frame",
            interval_seconds=eta_interval_seconds,
            initial_completed=reused_frames,
        )
        completed_frames = reused_frames
        try:
            if workers == 1:
                for part in pending:
                    _convert_part(
                        part,
                        temporary / "parts" / part.part_id,
                        dataset_uid,
                        fps_overrides,
                        allow_ordinal,
                        eta_interval_seconds,
                    )
                    completed_frames += part.num_frames
                    progress.update(completed_frames, context=f"part={part.part_id}", force=True)
                    if resume:
                        atomic_write_json(
                            checkpoint_state / MARKERS_DIR / f"{part.part_id}.json",
                            _marker_payload(fingerprint, part),
                        )
            elif pending:
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=context) as executor:
                    pending_iter = iter(pending)
                    futures: dict[Any, PartPlan] = {}

                    def submit_one() -> bool:
                        try:
                            next_part = next(pending_iter)
                        except StopIteration:
                            return False
                        future = executor.submit(
                            _convert_part,
                            next_part,
                            temporary / "parts" / next_part.part_id,
                            dataset_uid,
                            fps_overrides,
                            allow_ordinal,
                            eta_interval_seconds,
                        )
                        futures[future] = next_part
                        return True

                    for _ in range(min(len(pending), workers * 2)):
                        submit_one()
                    while futures:
                        done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                        for future in done:
                            part = futures.pop(future)
                            future.result()
                            completed_frames += part.num_frames
                            progress.update(completed_frames, context=f"part={part.part_id}", force=True)
                            if resume:
                                atomic_write_json(
                                    checkpoint_state / MARKERS_DIR / f"{part.part_id}.json",
                                    _marker_payload(fingerprint, part),
                                )
                            submit_one()
            _write_source_episode_index(collection, temporary)
            atomic_write_json(
                temporary / "collection_manifest.json",
                _collection_manifest(
                    collection,
                    dataset_uid=dataset_uid,
                    fps_overrides=fps_overrides,
                    allow_ordinal=allow_ordinal,
                ),
            )
            _validate_collection(collection, temporary, dataset_uid, fps_overrides, allow_ordinal)
            # LeRobot/Datasets creates this cache beside a dataset while
            # reopening it. It is validation scratch space, not publishable
            # collection content.
            shutil.rmtree(temporary / "parts" / LEROBOT_CACHE_DIR, ignore_errors=True)
            publish_temporary_output(temporary, output_path, overwrite=overwrite)
            if resume:
                shutil.rmtree(checkpoint_state)
            progress.finish(context=f"published={output_path}")
        except BaseException:
            if not resume and temporary.exists():
                shutil.rmtree(temporary)
            raise
    return output_path


def _summary(collection: CollectionPlan) -> dict[str, Any]:
    suites: dict[str, dict[str, Any]] = {}
    suite_tasks: dict[str, set[str]] = {}
    suite_robots: dict[str, set[str]] = {}
    suite_splits: dict[str, dict[str, int]] = {}

    def suite_row(suite: str) -> dict[str, Any]:
        return suites.setdefault(
            suite,
            {
                "parts": 0,
                "output_episodes": 0,
                "frames": 0,
                "source_episodes": 0,
                "source_action_frames": 0,
                "source_state_frames": 0,
                "source_action_length_min": None,
                "source_action_length_max": None,
                "source_state_length_min": None,
                "source_state_length_max": None,
                "aligned_episodes": 0,
                "unequal_stream_episodes": 0,
                "action_only_episodes": 0,
                "state_only_episodes": 0,
            },
        )

    for part in collection.parts:
        row = suite_row(part.source_suite)
        row["parts"] += 1
        row["output_episodes"] += len(part.episodes)
        row["frames"] += part.num_frames

    source_episodes = {
        episode.episode_uid: episode
        for part in collection.parts
        for episode in part.episodes
    }
    for episode in source_episodes.values():
        row = suite_row(episode.source_suite)
        row["source_episodes"] += 1
        row["source_action_frames"] += episode.action_count
        row["source_state_frames"] += episode.state_count
        if episode.action_count:
            current = row["source_action_length_min"]
            row["source_action_length_min"] = (
                episode.action_count if current is None else min(current, episode.action_count)
            )
            current = row["source_action_length_max"]
            row["source_action_length_max"] = (
                episode.action_count if current is None else max(current, episode.action_count)
            )
        if episode.state_count:
            current = row["source_state_length_min"]
            row["source_state_length_min"] = (
                episode.state_count if current is None else min(current, episode.state_count)
            )
            current = row["source_state_length_max"]
            row["source_state_length_max"] = (
                episode.state_count if current is None else max(current, episode.state_count)
            )
        if episode.action_count and episode.state_count:
            key = "aligned_episodes" if episode.action_count == episode.state_count else "unequal_stream_episodes"
        elif episode.action_count:
            key = "action_only_episodes"
        else:
            key = "state_only_episodes"
        row[key] += 1
        suite_tasks.setdefault(episode.source_suite, set()).add(episode.source_task)
        suite_robots.setdefault(episode.source_suite, set()).add(episode.robot_name)
        split_counts = suite_splits.setdefault(episode.source_suite, {})
        split_counts[episode.source_split] = split_counts.get(episode.source_split, 0) + 1

    for suite, row in suites.items():
        row["source_tasks"] = len(suite_tasks.get(suite, set()))
        row["robots"] = sorted(suite_robots.get(suite, set()))
        row["source_splits"] = dict(sorted(suite_splits.get(suite, {}).items()))

    inventory: dict[
        tuple[Any, ...],
        tuple[FeatureColumn, NumericStatistics, int, int],
    ] = {}
    for part in collection.parts:
        for column, statistics in zip(
            part.feature_columns,
            part.feature_statistics,
            strict=True,
        ):
            key = (part.source_suite, part.robot_name, part.stream_kind, column.signature())
            current = inventory.get(key)
            if current is None:
                inventory[key] = (column, statistics, 1, len(part.episodes))
            else:
                inventory[key] = (
                    column,
                    merge_numeric_statistics(current[1], statistics),
                    current[2] + 1,
                    current[3] + len(part.episodes),
                )
    feature_inventory = []
    for key in sorted(inventory, key=lambda item: repr(item).casefold()):
        suite, robot, stream_kind, _ = key
        column, statistics, parts, episodes = inventory[key]
        feature_inventory.append(
            {
                "source_suite": suite,
                "robot_type": robot,
                "stream_kind": stream_kind,
                "parts": parts,
                "episodes": episodes,
                **_column_manifest(column),
                **numeric_statistics_dict(statistics),
            }
        )
    return {
        "source_dataset": SOURCE_DATASET,
        "source_revision": collection.source_revision,
        "source_root": str(collection.source_root),
        "source_files": len(collection.files),
        "source_sidecars": len(collection.sidecars),
        "auxiliary_source_files": len(collection.auxiliary_files),
        "source_issues": list(collection.source_issues),
        "duplicate_aliases": len(collection.duplicate_aliases),
        "source_episodes": collection.num_source_episodes,
        "source_action_frames": sum(episode.action_count for episode in source_episodes.values()),
        "source_state_frames": sum(episode.state_count for episode in source_episodes.values()),
        "output_parts": len(collection.parts),
        "output_episodes": collection.num_output_episodes,
        "output_frames": collection.num_output_frames,
        "suites": suites,
        "feature_inventory": feature_inventory,
        "lossless_dtype_promotions": sum(
            len(episode.action_dtype_promotions) + len(episode.state_dtype_promotions)
            for episode in source_episodes.values()
        ),
        "physical_timestamps_available": False,
        "trajectory_images_or_videos_available": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/data/embodied_datasets/public_datasets_raw"))
    parser.add_argument("--staging-root", type=Path, default=Path("/home/pai/zxw/roboverse_staging"))
    parser.add_argument("--dataset-uid", default="roboverse")
    parser.add_argument(
        "--source-directory",
        default="roboverse",
        help="Directory below --raw-root; separate from the output dataset UID for smoke runs",
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --inspect-only")
    parser.add_argument(
        "--inspection-report",
        type=Path,
        help="Atomically write the complete machine-readable inspection summary to this path",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-ordinal-timebase", action="store_true")
    parser.add_argument(
        "--allow-lossless-dtype-promotion",
        action="store_true",
        help=(
            "Permit only exact integer/bool-to-existing-float promotion for a component whose "
            "dtype changes within an episode; original dtype runs are retained in provenance"
        ),
    )
    parser.add_argument(
        "--allow-source-sidecar-issues",
        action="store_true",
        help="Allow explicitly reported non-trajectory sidecar issues; never suppresses trajectory errors",
    )
    parser.add_argument("--fps-override", action="append", default=[], metavar="SUITE=INTEGER")
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Select an exact source task or generated task text; repeat to select multiple tasks",
    )
    parser.add_argument("--source-path", action="append", default=[])
    parser.add_argument("--max-source-files", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--part", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--eta-interval-seconds", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        if args.workers <= 0:
            raise ConversionError("--workers must be positive")
        if Path(args.dataset_uid).name != args.dataset_uid or args.dataset_uid in {"", ".", ".."}:
            raise ConversionError("--dataset-uid must be one safe path component")
        if not math.isfinite(args.eta_interval_seconds) or args.eta_interval_seconds <= 0:
            raise ConversionError("--eta-interval-seconds must be finite and positive")
        fps_overrides = _parse_fps_overrides(args.fps_override)
        raw_root = args.raw_root.expanduser().resolve()
        source_root = (raw_root / args.source_directory).resolve()
        if not source_root.is_relative_to(raw_root):
            raise ConversionError("--source-directory must stay below --raw-root")
        preflight_progress: EtaProgress | None = None

        def preflight_update(completed: int, total: int, path: str, episodes: int) -> None:
            nonlocal preflight_progress
            if preflight_progress is None:
                preflight_progress = EtaProgress(
                    "roboverse-preflight",
                    total,
                    "file",
                    interval_seconds=args.eta_interval_seconds,
                )
            preflight_progress.update(completed, context=f"source={path} episodes={episodes}")

        selection = {
            "suites": sorted(set(args.suite)),
            "tasks": sorted(set(args.task)),
            "source_paths": sorted(set(args.source_path)),
            "max_source_files": args.max_source_files,
            "max_episodes": args.max_episodes,
            "parts": sorted(set(args.part)),
            "source_directory": args.source_directory,
            "allow_source_sidecar_issues": args.allow_source_sidecar_issues,
            "allow_lossless_dtype_promotion": args.allow_lossless_dtype_promotion,
        }
        collection = inspect_collection(
            source_root,
            suites=set(args.suite) or None,
            source_paths=set(args.source_path) or None,
            tasks=set(args.task) or None,
            max_source_files=args.max_source_files,
            max_episodes=args.max_episodes,
            allow_lossless_dtype_promotion=args.allow_lossless_dtype_promotion,
            progress_callback=preflight_update,
        )
        if preflight_progress is not None:
            preflight_progress.finish(context=f"episodes={collection.num_source_episodes}")
        if args.part:
            requested = set(args.part)
            selected = tuple(part for part in collection.parts if part.part_id in requested)
            missing = requested - {part.part_id for part in selected}
            if missing:
                raise ConversionError(f"requested parts do not exist: {sorted(missing)}")
            collection = CollectionPlan(
                collection.source_root,
                collection.source_revision,
                collection.files,
                selected,
                collection.sidecars,
                collection.duplicate_aliases,
                collection.calvin_instruction_aliases,
                collection.source_issues,
                collection.sidecar_payloads,
                collection.auxiliary_files,
            )
        summary = _summary(collection)
        if args.inspection_report is not None:
            atomic_write_json(args.inspection_report.expanduser().resolve(), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        blocking_issues = [issue for issue in collection.source_issues if issue.get("blocking")]
        sidecar_issues = [issue for issue in collection.source_issues if not issue.get("blocking")]
        if args.inspect_only or args.dry_run:
            if blocking_issues:
                print(
                    f"error: source preflight found {len(blocking_issues)} blocking trajectory issue(s)",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            return 0
        if blocking_issues:
            raise ConversionError(
                f"source preflight found {len(blocking_issues)} unreadable or invalid trajectory file(s); "
                "these cannot be acknowledged or skipped"
            )
        if sidecar_issues and not args.allow_source_sidecar_issues:
            raise ConversionError(
                f"source preflight found {len(sidecar_issues)} sidecar issue(s); "
                "inspect the report and pass --allow-source-sidecar-issues only if these official "
                "non-trajectory placeholders are acceptable"
            )
        # Validate the time policy before creating any staging/checkpoint path.
        for part in collection.parts:
            _part_fps(part, fps_overrides, args.allow_ordinal_timebase)
        staging_root = args.staging_root.expanduser().resolve()
        if staging_root.is_relative_to(Path("/mnt/data")):
            raise ConversionError(
                "--staging-root must be server-local storage, not the /mnt/data OSSFS/FUSE mount"
            )
        output = staging_root / "lerobot_v3_0" / args.dataset_uid
        convert_collection(
            collection,
            output_path=output,
            dataset_uid=args.dataset_uid,
            resume=args.resume,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            fps_overrides=fps_overrides,
            allow_ordinal=args.allow_ordinal_timebase,
            eta_interval_seconds=args.eta_interval_seconds,
            workers=args.workers,
            selection=selection,
        )
        print(f"wrote RoboVerse collection: {output}", flush=True)
        return 0
    except (ConversionError, FileExistsError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
