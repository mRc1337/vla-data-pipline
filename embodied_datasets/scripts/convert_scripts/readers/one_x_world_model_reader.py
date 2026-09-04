"""Reader for the official 1X World Model Challenge token datasets.

The public directory contains two genuinely different releases.  v1.1 stores
one MAGVIT2 token grid per source frame plus five separately named robot
arrays.  v2.0 stores causal Cosmos video-token blocks plus one 25-value state
array.  This reader preserves those schemas as separate LeRobot datasets and
uses the official ``segment_ids``/``segment_idx`` values as episode bounds.

Inspection is read-only and never imports either neural decoder.  Decoder
dependencies and weights are loaded only by :meth:`iter_frames`, allowing a
full structural preflight on machines that do not have the GPU stack yet.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np

from convert_core.dataset_config import DatasetConversionConfig
from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError


HF_DATASET = "1x-technologies/worldmodel"
HF_REVISION = "42e3e12fff6848b511583ba6e8afa7f82ef9014e"
FPS = 30
DECODED_HEIGHT = 256
DECODED_WIDTH = 256
TASK_PLACEHOLDER = "Unspecified task; the source dataset provides no instruction."
V2_TEMPORAL_BLOCK = 17
V2_TOKEN_SHAPE = (3, 32, 32)
V2_CODEBOOK_SIZE = 64_000
SCAN_ROWS = 65_536

V2_STATE_NAMES = (
    "hip_yaw",
    "hip_roll",
    "hip_pitch",
    "knee_pitch",
    "ankle_roll",
    "ankle_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_yaw",
    "left_wrist_pitch",
    "left_wrist_roll",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_elbow_yaw",
    "right_wrist_pitch",
    "right_wrist_roll",
    "neck_pitch",
    "left_hand_closure",
    "right_hand_closure",
    "linear_velocity",
    "angular_velocity",
)

V1_JOINT_NAMES = V2_STATE_NAMES[:21]
V1_FIELDS = (
    ("action.joint_position", "joint_pos", 21, V1_JOINT_NAMES),
    ("action.neck_desired", "neck_desired", 3, None),
    ("action.driving_command", "driving_command", 2, None),
    ("action.left_hand_closure", "l_hand_closure", 1, ("left_hand_closure",)),
    ("action.right_hand_closure", "r_hand_closure", 1, ("right_hand_closure",)),
)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"metadata must contain a JSON object: {path}")
    return value


def _positive_int(value: Any, *, path: Path, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConversionError(f"{path}: {key} must be a positive integer, got {value!r}")
    return value


def _require_file(path: Path, *, size: int | None = None) -> None:
    if not path.is_file():
        raise ConversionError(f"missing source file: {path}")
    if size is not None and path.stat().st_size != size:
        raise ConversionError(
            f"{path}: size is {path.stat().st_size} bytes, expected exactly {size} bytes"
        )


def _source_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _verify_split_inventory(split_root: Path, expected_paths: list[Path]) -> None:
    """Reject regular files that the conversion plan would silently ignore."""

    expected = {path.resolve() for path in expected_paths}
    observed = {path.resolve() for path in split_root.rglob("*") if path.is_file()}
    unexpected = sorted(path.relative_to(split_root).as_posix() for path in observed - expected)
    missing = sorted(path.relative_to(split_root).as_posix() for path in expected - observed)
    if missing or unexpected:
        raise ConversionError(
            f"{split_root}: source inventory mismatch; missing={missing}, "
            f"unreferenced={unexpected}"
        )


def _selected_splits(
    configured: list[str],
    *,
    defaults: list[str],
    allowed: set[str],
    version: str,
) -> list[str]:
    splits = list(configured or defaults)
    duplicates = sorted(
        {split for split in splits if splits.count(split) > 1}
    )
    if duplicates:
        raise ConversionError(
            f"duplicate {version} split selection would duplicate source episodes: {duplicates}"
        )
    invalid = sorted(set(splits) - allowed)
    if invalid:
        raise ConversionError(f"invalid {version} split(s): {invalid}")
    return splits


def _integer_summary(
    values: np.ndarray,
    *,
    path: Path,
    lower: int,
    upper_exclusive: int,
) -> dict[str, int]:
    minimum: int | None = None
    maximum: int | None = None
    for start in range(0, len(values), SCAN_ROWS):
        chunk = np.asarray(values[start : start + SCAN_ROWS])
        chunk_min = int(chunk.min())
        chunk_max = int(chunk.max())
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
    assert minimum is not None and maximum is not None
    if minimum < lower or maximum >= upper_exclusive:
        raise ConversionError(
            f"{path}: token range [{minimum}, {maximum}] is outside "
            f"[{lower}, {upper_exclusive})"
        )
    return {"min": minimum, "max": maximum}


def _float_summary(values: np.ndarray, *, path: Path) -> dict[str, list[float]]:
    width = int(values.shape[1])
    minimum = np.full(width, np.inf, dtype=np.float64)
    maximum = np.full(width, -np.inf, dtype=np.float64)
    for start in range(0, len(values), SCAN_ROWS):
        chunk = np.asarray(values[start : start + SCAN_ROWS])
        finite = np.isfinite(chunk)
        if not bool(finite.all()):
            row, column = np.argwhere(~finite)[0]
            source_row = start + int(row)
            raise ConversionError(
                f"{path}: non-finite float at row {source_row}, column {int(column)}: "
                f"{chunk[row, column]!r}"
            )
        minimum = np.minimum(minimum, chunk.min(axis=0).astype(np.float64))
        maximum = np.maximum(maximum, chunk.max(axis=0).astype(np.float64))
    return {"min": minimum.tolist(), "max": maximum.tolist()}


def _runs(segment_ids: np.ndarray, *, path: Path) -> list[tuple[int, int, int]]:
    if segment_ids.size == 0:
        raise ConversionError(f"{path}: empty segment array")
    if np.any(segment_ids[1:] < segment_ids[:-1]):
        index = int(np.flatnonzero(segment_ids[1:] < segment_ids[:-1])[0] + 1)
        raise ConversionError(
            f"{path}: segment IDs decrease at frame {index}: "
            f"{int(segment_ids[index - 1])} -> {int(segment_ids[index])}"
        )
    changes = np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1
    starts = np.concatenate((np.array([0]), changes))
    ends = np.concatenate((changes, np.array([segment_ids.size])))
    return [(int(segment_ids[start]), int(start), int(end)) for start, end in zip(starts, ends)]


def _revision_from_cache(source_root: Path) -> str:
    metadata = source_root / ".cache" / "huggingface" / "download" / "README.md.metadata"
    if not metadata.is_file():
        return HF_REVISION
    first_line = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    if len(first_line) != 40:
        raise ConversionError(f"invalid Hugging Face revision in {metadata}: {first_line!r}")
    return first_line


def _split_paths_v2(split_root: Path, shard_index: int) -> tuple[Path, Path, Path, Path]:
    metadata_dir = split_root / "metadata"
    state_dir = split_root / "robot_states"
    segment_dir = split_root / "segment_indices"
    video_dir = split_root / "videos"
    metadata_path = (
        metadata_dir / f"metadata_{shard_index}.json"
        if metadata_dir.is_dir()
        else split_root / f"metadata_{shard_index}.json"
    )
    state_path = (
        state_dir / f"states_{shard_index}.bin"
        if state_dir.is_dir()
        else split_root / f"states_{shard_index}.bin"
    )
    segment_path = (
        segment_dir / f"segment_idx_{shard_index}.bin"
        if segment_dir.is_dir()
        else split_root / f"segment_idx_{shard_index}.bin"
    )
    video_path = (
        video_dir / f"video_{shard_index}.bin"
        if video_dir.is_dir()
        else split_root / f"video_{shard_index}.bin"
    )
    return metadata_path, state_path, segment_path, video_path


def _episode(
    *,
    version: str,
    split: str,
    segment_id: int,
    spans: list[dict[str, Any]],
    source_root: Path,
    checkpoint_unit: str,
) -> EpisodePlan:
    num_frames = sum(int(span["end"]) - int(span["start"]) for span in spans)
    source_spans = []
    for span in spans:
        record: dict[str, Any] = {
            "start": int(span["start"]),
            "end": int(span["end"]),
        }
        if "shard_index" in span:
            record["shard_index"] = int(span["shard_index"])
        for key in ("video_path", "state_path"):
            if key in span:
                record[key.removesuffix("_path")] = Path(span[key]).relative_to(
                    source_root
                ).as_posix()
        if "action_paths" in span:
            record["actions"] = {
                name: Path(path).relative_to(source_root).as_posix()
                for name, path in span["action_paths"].items()
            }
        source_spans.append(record)
    return EpisodePlan(
        episode_uid=f"{split}:segment:{segment_id}",
        source_relative_path=f"{split}/segment:{segment_id}",
        instruction=TASK_PLACEHOLDER,
        num_frames=num_frames,
        extra={
            "source_split": split,
            "source_splits": (split,),
            "source_task": None,
            "source_segment_id": segment_id,
            "source_version": version,
            "spans": tuple(spans),
            "source_spans": tuple(source_spans),
            "checkpoint_unit": checkpoint_unit,
            "source_root": source_root,
        },
    )


class OneXWorldModelReader:
    """Strict reader for the official tokenized 1X challenge release."""

    def __init__(self) -> None:
        self._v1_decoder: Any = None
        self._v2_decoder: Any = None
        self._v2_block_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def build_plan(
        self, config: DatasetConversionConfig, raw_root: Path, staging_root: Path
    ) -> DatasetConversionPlan:
        source_name = config.source_directory or "1x_world_model_dataset"
        source_root = raw_root / source_name
        if not source_root.is_dir():
            raise ConversionError(f"raw 1X dataset directory does not exist: {source_root}")
        if config.one_x_include_test:
            raise ConversionError(
                "test_v2.0 cannot be converted faithfully: each challenge sample contains one "
                "Cosmos token block (17 decoded frames) but 64 state rows, and supplies neither "
                "segment boundaries nor a documented per-frame alignment"
            )
        if config.one_x_version == "v1.1":
            return self._build_v1(config, source_root, staging_root)
        return self._build_v2(config, source_root, staging_root)

    def _base_extra(
        self,
        config: DatasetConversionConfig,
        source_root: Path,
        splits: list[str],
        source_files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        relative_paths = [str(row["relative_path"]) for row in source_files]
        duplicate_paths = sorted(
            {path for path in relative_paths if relative_paths.count(path) > 1}
        )
        if duplicate_paths:
            raise ConversionError(
                f"source files are referenced more than once: {duplicate_paths}"
            )
        return {
            "source_dataset": HF_DATASET,
            "source_revision": _revision_from_cache(source_root),
            "source_relative_path": source_root.name,
            "source_root": source_root,
            "source_splits": splits,
            "source_files": sorted(source_files, key=lambda row: row["relative_path"]),
            "task_provenance": {
                "source_has_instruction": False,
                "lerobot_required_placeholder": TASK_PLACEHOLDER,
            },
            "timestamp_provenance": {
                "source_has_per_frame_timestamps": False,
                "source_metadata_hz": FPS,
                "lerobot_timestamp": "frame_index / 30",
                "resampled": False,
            },
            "decoder": {
                "v1_decoder_repo": config.one_x_v1_decoder_repo,
                "v2_decoder_path": config.one_x_cosmos_decoder_path,
                "batch_size": config.one_x_decode_batch_size,
            },
            "unsupported_source_components": {
                "test_v2.0": (
                    "450 challenge samples have 17 decoded video frames and 64 state rows per "
                    "sample with no documented one-to-one alignment or segment metadata"
                )
            },
        }

    def _build_v1(
        self,
        config: DatasetConversionConfig,
        source_root: Path,
        staging_root: Path,
    ) -> DatasetConversionPlan:
        splits = _selected_splits(
            config.one_x_splits,
            defaults=["train_v1.1", "val_v1.1"],
            allowed={"train_v1.1", "val_v1.1"},
            version="v1.1",
        )
        source_files: list[dict[str, Any]] = []
        episodes: list[EpisodePlan] = []
        split_summaries: dict[str, Any] = {}

        checkpoint = source_root / "magvit2.ckpt"
        _require_file(checkpoint)
        source_files.append(_source_record(checkpoint, source_root))
        for split in splits:
            split_root = source_root / split
            metadata_path = split_root / "metadata.json"
            segment_path = split_root / "segment_ids.bin"
            video_path = split_root / "video.bin"
            metadata = _json_object(metadata_path)
            num_frames = _positive_int(metadata.get("num_images"), path=metadata_path, key="num_images")
            expected = {
                "token_dtype": "uint32",
                "s": 16,
                "h": 16,
                "w": 16,
                "vocab_size": 262144,
                "hz": FPS,
            }
            mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
            if mismatches:
                raise ConversionError(f"{metadata_path}: unsupported v1.1 metadata values: {mismatches}")
            _require_file(segment_path, size=num_frames * np.dtype(np.int32).itemsize)
            _require_file(video_path, size=num_frames * 16 * 16 * np.dtype(np.uint32).itemsize)
            video_tokens = np.memmap(
                video_path, dtype=np.uint32, mode="r", shape=(num_frames, 16, 16)
            )
            token_range = _integer_summary(
                video_tokens,
                path=video_path,
                lower=0,
                upper_exclusive=int(metadata["vocab_size"]),
            )
            action_paths: dict[str, Path] = {}
            action_ranges: dict[str, dict[str, list[float]]] = {}
            for _feature, source_name, width, _names in V1_FIELDS:
                path = split_root / "actions" / f"{source_name}.bin"
                _require_file(path, size=num_frames * width * np.dtype(np.float32).itemsize)
                action_paths[source_name] = path
                values = np.memmap(
                    path, dtype=np.float32, mode="r", shape=(num_frames, width)
                )
                action_ranges[source_name] = _float_summary(values, path=path)

            segment_ids = np.memmap(segment_path, dtype=np.int32, mode="r", shape=(num_frames,))
            split_runs = _runs(segment_ids, path=segment_path)
            for ordinal, (segment_id, start, end) in enumerate(split_runs):
                checkpoint_index = ordinal // config.one_x_v1_checkpoint_segments
                span = {
                    "start": start,
                    "end": end,
                    "video_path": video_path,
                    "action_paths": action_paths,
                    "num_source_frames": num_frames,
                }
                episodes.append(
                    _episode(
                        version="v1.1",
                        split=split,
                        segment_id=segment_id,
                        spans=[span],
                        source_root=source_root,
                        checkpoint_unit=f"{split}/segment_batch_{checkpoint_index:05d}",
                    )
                )
            paths = [metadata_path, segment_path, video_path, *action_paths.values()]
            _verify_split_inventory(split_root, paths)
            source_files.extend(_source_record(path, source_root) for path in paths)
            lengths = np.asarray([end - start for _, start, end in split_runs])
            split_summaries[split] = {
                "frames": num_frames,
                "episodes": len(split_runs),
                "min_episode_frames": int(lengths.min()),
                "max_episode_frames": int(lengths.max()),
                "source_file_count": len(paths),
                "unreferenced_files": [],
                "numeric_ranges": {
                    "video_tokens": token_range,
                    "actions": action_ranges,
                },
            }

        extra = self._base_extra(config, source_root, splits, source_files)
        extra.update(
            {
                "source_version": "v1.1",
                "partition_rules": {
                    "partition_key": "source release version",
                    "partition_value": "v1.1",
                    "reason": (
                        "v1.1 has five separately stored robot fields and MAGVIT2 tokens; "
                        "it is incompatible with the v2.0 state/Cosmos schema"
                    ),
                },
                "split_summaries": split_summaries,
                "field_mapping": [
                    {
                        "source": f"actions/{source_name}.bin",
                        "source_shape": ["N"] if width == 1 else ["N", width],
                        "source_dtype": "float32",
                        "semantics": source_name,
                        "role_evidence": (
                            "stored below actions/ and described collectively as raw actions by "
                            "the official repository; the dataset card also calls these states, "
                            "closures, etc., so command/target/measured status is unpublished"
                        ),
                        "unit": "not published by source",
                        "lerobot": feature,
                        "transform": (
                            "reshape source scalar to a length-one LeRobot array without casting"
                            if width == 1
                            else "none"
                        ),
                        "dtype_cast": None,
                        "field_reordered": False,
                        "lossy": False,
                    }
                    for feature, source_name, width, _names in V1_FIELDS
                ],
                "video_encoding": {
                    "source": "MAGVIT2 discrete uint32 token grid [16,16]",
                    "source_fps": FPS,
                    "decoded_resolution": [DECODED_HEIGHT, DECODED_WIDTH],
                    "decode_is_source_reconstruction": True,
                    "direct_remux_possible": False,
                },
            }
        )
        vector_features = tuple(
            VectorFeatureSpec(feature_key=feature, dim=width, names=names)
            for feature, _source_name, width, names in V1_FIELDS
        )
        return DatasetConversionPlan(
            dataset_uid=config.dataset_uid,
            output_path=staging_root / "lerobot_v3_0" / config.dataset_uid,
            fps=FPS,
            measured_fps=float(FPS),
            robot_type=config.robot_type,
            vector_features=vector_features,
            camera_features=(
                CameraFeatureSpec(
                    feature_key="observation.images.head",
                    height=DECODED_HEIGHT,
                    width=DECODED_WIDTH,
                ),
            ),
            episodes=tuple(episodes),
            extra=extra,
        )

    def _build_v2(
        self,
        config: DatasetConversionConfig,
        source_root: Path,
        staging_root: Path,
    ) -> DatasetConversionPlan:
        splits = _selected_splits(
            config.one_x_splits,
            defaults=["train_v2.0", "val_v2.0"],
            allowed={"train_v2.0", "val_v2.0"},
            version="v2.0",
        )
        source_files: list[dict[str, Any]] = []
        episodes: list[EpisodePlan] = []
        split_summaries: dict[str, Any] = {}

        for split in splits:
            split_root = source_root / split
            metadata_path = split_root / "metadata.json"
            metadata = _json_object(metadata_path)
            num_shards = _positive_int(metadata.get("num_shards"), path=metadata_path, key="num_shards")
            declared_frames = _positive_int(metadata.get("num_images"), path=metadata_path, key="num_images")
            if metadata.get("hz") != FPS:
                raise ConversionError(f"{metadata_path}: expected hz={FPS}, got {metadata.get('hz')!r}")
            if metadata.get("query") is not None:
                raise ConversionError(
                    f"{metadata_path}: query/task metadata is unexpectedly non-null: {metadata.get('query')!r}"
                )
            source_files.append(_source_record(metadata_path, source_root))
            split_paths = [metadata_path]
            mutable: list[dict[str, Any]] = []
            observed_frames = 0
            previous_segment: int | None = None
            state_minimum = np.full(25, np.inf, dtype=np.float64)
            state_maximum = np.full(25, -np.inf, dtype=np.float64)
            token_minimum: int | None = None
            token_maximum: int | None = None

            for shard_index in range(num_shards):
                shard_metadata_path, state_path, segment_path, video_path = _split_paths_v2(
                    split_root, shard_index
                )
                shard_metadata = _json_object(shard_metadata_path)
                shard_frames = _positive_int(
                    shard_metadata.get("shard_num_frames"),
                    path=shard_metadata_path,
                    key="shard_num_frames",
                )
                if shard_metadata.get("shard_ind") != shard_index:
                    raise ConversionError(
                        f"{shard_metadata_path}: shard_ind is {shard_metadata.get('shard_ind')!r}, "
                        f"expected {shard_index}"
                    )
                _require_file(state_path, size=shard_frames * 25 * np.dtype(np.float32).itemsize)
                _require_file(segment_path, size=shard_frames * np.dtype(np.int32).itemsize)
                token_blocks = math.ceil(shard_frames / V2_TEMPORAL_BLOCK)
                _require_file(
                    video_path,
                    size=token_blocks * math.prod(V2_TOKEN_SHAPE) * np.dtype(np.int32).itemsize,
                )
                source_files.extend(
                    _source_record(path, source_root)
                    for path in (shard_metadata_path, state_path, segment_path, video_path)
                )
                split_paths.extend(
                    (shard_metadata_path, state_path, segment_path, video_path)
                )
                state_values = np.memmap(
                    state_path, dtype=np.float32, mode="r", shape=(shard_frames, 25)
                )
                state_range = _float_summary(state_values, path=state_path)
                state_minimum = np.minimum(state_minimum, state_range["min"])
                state_maximum = np.maximum(state_maximum, state_range["max"])
                token_values = np.memmap(
                    video_path,
                    dtype=np.int32,
                    mode="r",
                    shape=(token_blocks, *V2_TOKEN_SHAPE),
                )
                token_range = _integer_summary(
                    token_values,
                    path=video_path,
                    lower=0,
                    upper_exclusive=V2_CODEBOOK_SIZE,
                )
                token_minimum = (
                    token_range["min"]
                    if token_minimum is None
                    else min(token_minimum, token_range["min"])
                )
                token_maximum = (
                    token_range["max"]
                    if token_maximum is None
                    else max(token_maximum, token_range["max"])
                )
                segment_ids = np.memmap(
                    segment_path, dtype=np.int32, mode="r", shape=(shard_frames,)
                )
                shard_runs = _runs(segment_ids, path=segment_path)
                first_segment = shard_runs[0][0]
                if previous_segment is not None and first_segment < previous_segment:
                    raise ConversionError(
                        f"{segment_path}: first segment {first_segment} is below previous shard's "
                        f"last segment {previous_segment}"
                    )
                for segment_id, start, end in shard_runs:
                    span = {
                        "start": start,
                        "end": end,
                        "shard_index": shard_index,
                        "state_path": state_path,
                        "video_path": video_path,
                        "num_source_frames": shard_frames,
                    }
                    if mutable and mutable[-1]["segment_id"] == segment_id:
                        mutable[-1]["spans"].append(span)
                    else:
                        mutable.append({"segment_id": segment_id, "spans": [span]})
                previous_segment = shard_runs[-1][0]
                observed_frames += shard_frames

            if observed_frames != declared_frames:
                raise ConversionError(
                    f"{metadata_path}: shard frames sum to {observed_frames}, expected {declared_frames}"
                )
            _verify_split_inventory(split_root, split_paths)
            split_lengths: list[int] = []
            for item in mutable:
                spans = item["spans"]
                last_shard = int(spans[-1]["shard_index"])
                plan_episode = _episode(
                    version="v2.0",
                    split=split,
                    segment_id=int(item["segment_id"]),
                    spans=spans,
                    source_root=source_root,
                    checkpoint_unit=f"{split}/shard_{last_shard:05d}",
                )
                episodes.append(plan_episode)
                split_lengths.append(plan_episode.num_frames)
            lengths = np.asarray(split_lengths)
            split_summaries[split] = {
                "frames": declared_frames,
                "shards": num_shards,
                "episodes": len(mutable),
                "min_episode_frames": int(lengths.min()),
                "max_episode_frames": int(lengths.max()),
                "source_file_count": len(split_paths),
                "unreferenced_files": [],
                "numeric_ranges": {
                    "video_tokens": {"min": token_minimum, "max": token_maximum},
                    "states": {
                        "names": list(V2_STATE_NAMES),
                        "min": state_minimum.tolist(),
                        "max": state_maximum.tolist(),
                    },
                },
            }

        extra = self._base_extra(config, source_root, splits, source_files)
        extra.update(
            {
                "source_version": "v2.0",
                "partition_rules": {
                    "partition_key": "source release version",
                    "partition_value": "v2.0",
                    "reason": (
                        "v2.0 has one 25-value state and causal Cosmos tokens; it is "
                        "incompatible with the v1.1 multi-field/MAGVIT2 schema"
                    ),
                },
                "split_summaries": split_summaries,
                "field_mapping": [
                    {
                        "source": "states_{shard}.bin",
                        "source_shape": ["N", 25],
                        "source_dtype": "float32",
                        "semantics": "robot state (official dataset-card wording)",
                        "unit": "not published by source",
                        "lerobot": "observation.state",
                        "transform": "none",
                        "dtype_cast": None,
                        "field_reordered": False,
                        "lossy": False,
                    }
                ],
                "video_encoding": {
                    "source": "Cosmos-Tokenizer-DV8x8x8 int32 tokens [ceil(N/17),3,32,32]",
                    "source_fps": FPS,
                    "decoded_resolution": [DECODED_HEIGHT, DECODED_WIDTH],
                    "decode_is_source_reconstruction": True,
                    "direct_remux_possible": False,
                },
            }
        )
        return DatasetConversionPlan(
            dataset_uid=config.dataset_uid,
            output_path=staging_root / "lerobot_v3_0" / config.dataset_uid,
            fps=FPS,
            measured_fps=float(FPS),
            robot_type=config.robot_type,
            vector_features=(
                VectorFeatureSpec(
                    feature_key="observation.state", dim=25, names=V2_STATE_NAMES
                ),
            ),
            camera_features=(
                CameraFeatureSpec(
                    feature_key="observation.images.head",
                    height=DECODED_HEIGHT,
                    width=DECODED_WIDTH,
                ),
            ),
            episodes=tuple(episodes),
            extra=extra,
        )

    @contextmanager
    def _official_v1_import_path(self, repo: Path) -> Iterator[None]:
        text = str(repo)
        sys.path.insert(0, text)
        try:
            yield
        finally:
            if sys.path and sys.path[0] == text:
                sys.path.pop(0)

    def _get_v1_decoder(self, plan: DatasetConversionPlan) -> Any:
        if self._v1_decoder is not None:
            return self._v1_decoder
        repo_value = plan.extra["decoder"].get("v1_decoder_repo")
        if not repo_value:
            raise ConversionError(
                "v1.1 decoding requires one_x_v1_decoder_repo pointing to an official "
                "1x-technologies/1Xgpt checkout"
            )
        repo = Path(repo_value)
        if not (repo / "magvit2" / "models" / "lfqgan.py").is_file():
            raise ConversionError(f"invalid official 1Xgpt checkout: {repo}")
        checkpoint = Path(plan.extra["source_root"]) / "magvit2.ckpt"
        _require_file(checkpoint)
        try:
            with self._official_v1_import_path(repo):
                from magvit2.config import VQConfig
                from magvit2.models.lfqgan import VQModel
                import torch
        except ImportError as exc:
            raise ConversionError(
                f"cannot import the official MAGVIT2 decoder from {repo}: {exc}"
            ) from exc
        if not torch.cuda.is_available():
            raise ConversionError("the official MAGVIT2 decoder requires a CUDA GPU")
        model = VQModel(VQConfig(), ckpt_path=str(checkpoint))
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        postprocess_device = plan.extra["decoder"].get(
            "v1_postprocess_device", "cpu"
        )
        if postprocess_device not in {"cpu", "gpu"}:
            raise ConversionError(
                "v1_postprocess_device must be either 'cpu' or 'gpu'"
            )

        def decode(tokens: np.ndarray) -> np.ndarray:
            from einops import rearrange

            batch = torch.from_numpy(tokens.astype(np.int64, copy=True))
            with torch.no_grad(), model.ema_scope():
                quant = model.quantize.get_codebook_entry(
                    rearrange(batch, "b h w -> b (h w)"),
                    bhwc=batch.shape + (model.quantize.codebook_dim,),
                ).flip(1)
                output = model.decode(quant.to(device="cuda", dtype=torch.bfloat16))
                if postprocess_device == "gpu":
                    output = torch.clamp((output + 1) * 127.5, 0, 255)
                    output = output.to(dtype=torch.uint8).permute(
                        0, 2, 3, 1
                    ).contiguous()
                    return output.cpu().numpy()
                output = torch.clamp(
                    (output.detach().cpu() + 1) * 127.5, 0, 255
                )
            return output.to(dtype=torch.uint8).permute(0, 2, 3, 1).numpy()

        self._v1_decoder = decode
        return decode

    def _get_v2_decoder(self, plan: DatasetConversionPlan) -> Any:
        if self._v2_decoder is not None:
            return self._v2_decoder
        decoder_value = plan.extra["decoder"].get("v2_decoder_path")
        if not decoder_value:
            raise ConversionError(
                "v2.0 decoding requires one_x_cosmos_decoder_path pointing to "
                "Cosmos-Tokenizer-DV8x8x8/decoder.jit"
            )
        decoder_path = Path(decoder_value)
        _require_file(decoder_path)
        try:
            import torch
            from cosmos_tokenizer.utils import tensor2numpy
            from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        except ImportError as exc:
            raise ConversionError(
                "v2.0 decoding requires the official NVIDIA Cosmos-Tokenizer package: "
                f"{exc}"
            ) from exc
        if not torch.cuda.is_available():
            raise ConversionError("the Cosmos video decoder requires a CUDA GPU")
        decoder = CausalVideoTokenizer(checkpoint_dec=str(decoder_path))
        if decoder._dec_model is None:
            raise ConversionError(f"Cosmos decoder did not load from {decoder_path}")

        def decode(tokens: np.ndarray) -> np.ndarray:
            # Memmaps are read-only; copy before exposing the array to torch so
            # a decoder implementation can never mutate source-backed memory.
            batch = torch.from_numpy(np.array(tokens, copy=True)).to(device="cuda")
            with torch.no_grad():
                output = decoder.decode(batch)
            frames = np.asarray(tensor2numpy(output))
            if frames.ndim != 5 or frames.shape[1:] != (
                V2_TEMPORAL_BLOCK,
                DECODED_HEIGHT,
                DECODED_WIDTH,
                3,
            ):
                raise ConversionError(
                    f"Cosmos decoder returned shape {frames.shape}, expected "
                    f"[B,{V2_TEMPORAL_BLOCK},{DECODED_HEIGHT},{DECODED_WIDTH},3]"
                )
            if frames.dtype != np.uint8:
                raise ConversionError(
                    f"Cosmos tensor2numpy returned {frames.dtype}, expected uint8 RGB"
                )
            return frames

        self._v2_decoder = decode
        return decode

    def preflight_decoder(self, plan: DatasetConversionPlan) -> None:
        """Decode one short real sample and validate every output contract."""

        episode = plan.episodes[0]
        sample_count = min(60, episode.num_frames)
        frames = self.iter_frames(plan, episode)
        for _ in range(sample_count):
            next(frames)

    @staticmethod
    def _validate_frame_contract(
        plan: DatasetConversionPlan,
        episode: EpisodePlan,
        frame: dict[str, Any],
    ) -> None:
        expected_keys = {
            "task",
            *(feature.feature_key for feature in plan.vector_features),
            *(camera.feature_key for camera in plan.camera_features),
        }
        if set(frame) != expected_keys:
            raise ConversionError(
                f"{episode.episode_uid}: frame keys are {sorted(frame)}, "
                f"expected {sorted(expected_keys)}"
            )
        if frame["task"] != episode.instruction:
            raise ConversionError(
                f"{episode.episode_uid}: frame task differs from episode instruction"
            )
        for feature in plan.vector_features:
            value = frame[feature.feature_key]
            expected_dtype = np.dtype(feature.dtype)
            expected_shape = feature.resolved_shape
            if not isinstance(value, np.ndarray):
                raise ConversionError(
                    f"{episode.episode_uid}: {feature.feature_key} must be np.ndarray, "
                    f"got {type(value).__name__}"
                )
            if value.dtype != expected_dtype or value.shape != expected_shape:
                raise ConversionError(
                    f"{episode.episode_uid}: {feature.feature_key} is "
                    f"{value.shape}/{value.dtype}, expected {expected_shape}/{expected_dtype}"
                )
        for camera in plan.camera_features:
            image = frame[camera.feature_key]
            expected_shape = (camera.height, camera.width, 3)
            if not isinstance(image, np.ndarray):
                raise ConversionError(
                    f"{episode.episode_uid}: {camera.feature_key} must be np.ndarray, "
                    f"got {type(image).__name__}"
                )
            if image.dtype != np.uint8 or image.shape != expected_shape:
                raise ConversionError(
                    f"{episode.episode_uid}: {camera.feature_key} is "
                    f"{image.shape}/{image.dtype}, expected {expected_shape}/uint8"
                )

    def _v2_frames_for_span(
        self,
        plan: DatasetConversionPlan,
        video_path: Path,
        num_source_frames: int,
        start: int,
        end: int,
    ) -> Iterator[np.ndarray]:
        block_count = math.ceil(num_source_frames / V2_TEMPORAL_BLOCK)
        tokens = np.memmap(
            video_path,
            dtype=np.int32,
            mode="r",
            shape=(block_count, *V2_TOKEN_SHAPE),
        )
        decode = self._get_v2_decoder(plan)
        batch_size = int(plan.extra["decoder"]["batch_size"])
        first_block = start // V2_TEMPORAL_BLOCK
        final_block = (end - 1) // V2_TEMPORAL_BLOCK
        block_index = first_block
        while block_index <= final_block:
            key = (str(video_path), block_index)
            if key not in self._v2_block_cache:
                batch_end = min(final_block + 1, block_index + batch_size)
                decoded = decode(np.asarray(tokens[block_index:batch_end]))
                for offset, block in enumerate(decoded):
                    cache_key = (str(video_path), block_index + offset)
                    self._v2_block_cache[cache_key] = block
                    self._v2_block_cache.move_to_end(cache_key)
                while len(self._v2_block_cache) > max(8, batch_size * 4):
                    self._v2_block_cache.popitem(last=False)
            block = self._v2_block_cache[key]
            block_start = block_index * V2_TEMPORAL_BLOCK
            local_start = max(start, block_start) - block_start
            local_end = min(end, block_start + V2_TEMPORAL_BLOCK) - block_start
            for frame in block[local_start:local_end]:
                yield frame
            block_index += 1

    def iter_frames(
        self, plan: DatasetConversionPlan, episode: EpisodePlan
    ) -> Iterator[dict[str, Any]]:
        version = episode.extra["source_version"]
        if version == "v1.1":
            decoder = self._get_v1_decoder(plan)
            batch_size = int(plan.extra["decoder"]["batch_size"])
            span = episode.extra["spans"][0]
            start, end = int(span["start"]), int(span["end"])
            video = np.memmap(
                span["video_path"],
                dtype=np.uint32,
                mode="r",
                shape=(int(span["num_source_frames"]), 16, 16),
            )
            arrays = {
                feature: np.memmap(
                    span["action_paths"][source_name],
                    dtype=np.float32,
                    mode="r",
                    shape=(int(span["num_source_frames"]), width),
                )
                for feature, source_name, width, _names in V1_FIELDS
            }
            for batch_start in range(start, end, batch_size):
                batch_end = min(end, batch_start + batch_size)
                images = decoder(np.asarray(video[batch_start:batch_end]))
                for offset, image in enumerate(images):
                    source_index = batch_start + offset
                    frame = {
                        feature: np.array(values[source_index], dtype=np.float32, copy=True)
                        for feature, values in arrays.items()
                    }
                    frame["observation.images.head"] = image
                    frame["task"] = episode.instruction
                    self._validate_frame_contract(plan, episode, frame)
                    yield frame
            return

        yielded = 0
        for span in episode.extra["spans"]:
            start, end = int(span["start"]), int(span["end"])
            states = np.memmap(
                span["state_path"],
                dtype=np.float32,
                mode="r",
                shape=(int(span["num_source_frames"]), 25),
            )
            images = self._v2_frames_for_span(
                plan,
                span["video_path"],
                int(span["num_source_frames"]),
                start,
                end,
            )
            for source_index, image in zip(range(start, end), images, strict=True):
                frame = {
                    "observation.state": np.asarray(states[source_index]),
                    "observation.images.head": image,
                    "task": episode.instruction,
                }
                self._validate_frame_contract(plan, episode, frame)
                yield frame
                yielded += 1
        if yielded != episode.num_frames:
            raise ConversionError(
                f"{episode.episode_uid}: decoded {yielded} frames, expected {episode.num_frames}"
            )
