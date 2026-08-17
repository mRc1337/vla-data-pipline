"""Reliably migrate NVIDIA GR00T Teleop Sim from LeRobot v2.0 to v3.0.

The official release contains 24 independent LeRobot v2.0 datasets (one per
RoboCasa task) plus secondary HDF5 files.  This converter keeps those source
dataset boundaries as a v3 collection.  Within every part it follows the two
official LeRobot migrations:

* v2.0 -> v2.1: compute per-episode statistics;
* v2.1 -> v3.0: pack episode parquet files, remux (never re-encode) episode
  videos, and write parquet task/episode metadata.

The release's frame-level ``task_index`` always points at a CamelCase source
identifier.  The authoritative natural-language label is instead stored per
episode in ``meta/episodes.jsonl`` as ``remarks`` and is repeated in the HDF5
``ep_meta.lang`` field.  We remap only ``task_index`` to those official labels;
the old value and source identifier remain in conversion metadata.  Robot
state/action values, episode order/boundaries, timestamps, annotations, and
encoded video packets are otherwise retained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import filecmp
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
import uuid

import numpy as np
import yaml

from convert_core.errors import ConversionError
from convert_core.lerobot_writer import publish_temporary_output
from convert_core.progress import EtaProgress


SOURCE_REPO = "nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim"
SOURCE_COMMIT = "09c6de8af50168090e7e9cc01e1ec3bce788de24"
SOURCE_DATASET_URL = f"https://huggingface.co/datasets/{SOURCE_REPO}/tree/{SOURCE_COMMIT}"
PAPER_URL = "https://arxiv.org/abs/2503.14734"
ISAAC_GROOT_REPO = "https://github.com/NVIDIA/Isaac-GR00T"
ISAAC_GROOT_COMMIT = "376ba890cff8c9de64d71d982772a9c36185fdd7"
ROBOCASA_GR1_REPO = "https://github.com/robocasa/robocasa-gr1-tabletop-tasks"
ROBOCASA_GR1_COMMIT = "4840e671596f93ca03651524b9f72ffb1aadfeff"
ROBOSUITE_REPO = "https://github.com/ARISE-Initiative/robosuite"
ROBOSUITE_COMMIT = "a071383d53568ab798eb315c0e95357911be922d"
LEROBOT_V033_COMMIT = "b883328e6c95681ca90a18b102e4ae5e1f91e2bf"
RESUME_SCHEMA_VERSION = 1
RESUME_STATE_FILE = "state.json"
RESUME_PARTS_DIR = "parts"
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "gr00t_teleop_sim.yaml"
VIDEO_KEY = "observation.images.ego_view"
REQUIRED_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "next.reward",
    "next.done",
    "task_index",
    "annotation.human.fine_action",
    "annotation.human.coarse_action",
    "episode_index",
    "index",
)

# Exact source order.  Arm/waist/leg names come from the GR1 MJCF.  The six
# Fourier-hand coordinates are proven by the source qpos extraction order
# (pinky/ring/middle/index intermediate, thumb pitch, thumb yaw).  Disabled
# neck/leg coordinates already exist as zero-valued fields in the official v2
# release; this converter does not synthesize them.
GR1_NAMES = (
    "robot0_l_shoulder_pitch",
    "robot0_l_shoulder_roll",
    "robot0_l_shoulder_yaw",
    "robot0_l_elbow_pitch",
    "robot0_l_wrist_yaw",
    "robot0_l_wrist_roll",
    "robot0_l_wrist_pitch",
    "gripper0_left_L_pinky_intermediate_joint",
    "gripper0_left_L_ring_intermediate_joint",
    "gripper0_left_L_middle_intermediate_joint",
    "gripper0_left_L_index_intermediate_joint",
    "gripper0_left_L_thumb_proximal_pitch_joint",
    "gripper0_left_L_thumb_proximal_yaw_joint",
    "robot0_l_leg_hip_roll",
    "robot0_l_leg_hip_yaw",
    "robot0_l_leg_hip_pitch",
    "robot0_l_leg_knee_pitch",
    "robot0_l_leg_ankle_pitch",
    "robot0_l_leg_ankle_roll",
    "robot0_head_yaw",
    "robot0_head_roll",
    "robot0_head_pitch",
    "robot0_r_shoulder_pitch",
    "robot0_r_shoulder_roll",
    "robot0_r_shoulder_yaw",
    "robot0_r_elbow_pitch",
    "robot0_r_wrist_yaw",
    "robot0_r_wrist_roll",
    "robot0_r_wrist_pitch",
    "gripper0_right_R_pinky_intermediate_joint",
    "gripper0_right_R_ring_intermediate_joint",
    "gripper0_right_R_middle_intermediate_joint",
    "gripper0_right_R_index_intermediate_joint",
    "gripper0_right_R_thumb_proximal_pitch_joint",
    "gripper0_right_R_thumb_proximal_yaw_joint",
    "robot0_r_leg_hip_roll",
    "robot0_r_leg_hip_yaw",
    "robot0_r_leg_hip_pitch",
    "robot0_r_leg_knee_pitch",
    "robot0_r_leg_ankle_pitch",
    "robot0_r_leg_ankle_roll",
    "robot0_torso_waist_yaw",
    "robot0_torso_waist_pitch",
    "robot0_torso_waist_roll",
)


@dataclass(frozen=True)
class Config:
    dataset_uid: str
    source_lerobot_subdir: str
    source_hdf5_subdir: str
    source_part_glob: str
    source_version: str
    target_version: str
    robot_type: str
    fps: int
    task_source: str
    preserve_meta_files: tuple[str, ...]
    data_file_size_in_mb: int
    video_file_size_in_mb: int
    crosscheck_hdf5: bool


@dataclass(frozen=True)
class Episode:
    episode_index: int
    source_parquet: Path
    source_video: Path
    length: int
    instruction: str
    mapped_task_index: int
    source_task_index: int
    source_metadata: dict[str, Any]


@dataclass
class Part:
    source_name: str
    source_task: str
    source_root: Path
    hdf5_path: Path
    output_name: str
    source_info: dict[str, Any]
    features: dict[str, dict[str, Any]]
    episodes: list[Episode]
    tasks: dict[int, str]
    video_info: dict[str, Any]
    hdf5_schemas: set[tuple[tuple[str, tuple[int, ...], str], ...]] = field(default_factory=set)

    @property
    def frames(self) -> int:
        return sum(ep.length for ep in self.episodes)


@dataclass
class Collection:
    config: Config
    raw_dataset_root: Path
    output_path: Path
    parts: list[Part]

    @property
    def episodes(self) -> int:
        return sum(len(part.episodes) for part in self.parts)

    @property
    def frames(self) -> int:
        return sum(part.frames for part in self.parts)


def official_references() -> dict[str, Any]:
    """Pinned public sources used for schema and migration decisions."""

    return {
        "dataset_card_and_files": {"url": SOURCE_DATASET_URL, "commit": SOURCE_COMMIT},
        "paper": PAPER_URL,
        "official_loader": {"repo": ISAAC_GROOT_REPO, "commit": ISAAC_GROOT_COMMIT},
        "simulation_and_key_converter": {
            "repo": ROBOCASA_GR1_REPO,
            "commit": ROBOCASA_GR1_COMMIT,
        },
        "robot_and_fourier_hand_mjcf": {"repo": ROBOSUITE_REPO, "commit": ROBOSUITE_COMMIT},
        "lerobot_v2_0_to_v2_1": {
            "repo": "https://github.com/huggingface/lerobot",
            "tag": "v0.3.3",
            "commit": LEROBOT_V033_COMMIT,
            "scripts": [
                "src/lerobot/datasets/v21/convert_dataset_v20_to_v21.py",
                "src/lerobot/datasets/v21/convert_stats.py",
            ],
        },
        "lerobot_v2_1_to_v3_0": {
            "installed_version": "0.6.0",
            "script": "lerobot.scripts.convert_dataset_v21_to_v30",
        },
    }


def field_mapping() -> list[dict[str, str]]:
    """Machine-readable source-to-target mapping included in every manifest."""

    unchanged = "copied bit-exactly into packed Parquet"
    return [
        {
            "source": "LeRobot v2 data:observation.state",
            "target": "data:observation.state",
            "operation": unchanged,
            "basis": "source Parquet schema + modality.json + official GR1/Fourier MJCF",
        },
        {
            "source": "LeRobot v2 data:action",
            "target": "data:action",
            "operation": unchanged,
            "basis": "source Parquet schema + modality.json + official absolute-action key converter",
        },
        {
            "source": "LeRobot v2 episode MP4:observation.images.ego_view",
            "target": "videos:observation.images.ego_view",
            "operation": "H.264 packet remux into packed MP4; no decode/re-encode",
            "basis": "dataset card, source info.json/video stream, official LeRobot v3 migration",
        },
        {
            "source": "LeRobot v2 data:timestamp",
            "target": "data:timestamp",
            "operation": unchanged,
            "basis": "source Parquet; cadence cross-checked with 20 Hz source metadata",
        },
        {
            "source": "LeRobot v2 data:next.reward",
            "target": "data:next.reward",
            "operation": unchanged,
            "basis": "source Parquet schema",
        },
        {
            "source": "LeRobot v2 data:next.done",
            "target": "data:next.done",
            "operation": unchanged,
            "basis": "source Parquet schema",
        },
        {
            "source": "episodes.jsonl:remarks (verified against HDF5 ep_meta.lang)",
            "target": "tasks.parquet task + data:task_index + episodes.parquet:tasks",
            "operation": "natural-language task table creation and task_index remap only",
            "basis": "official per-episode annotations in both distributed representations",
        },
        {
            "source": "LeRobot v2 data:annotation.human.fine_action",
            "target": "data:annotation.human.fine_action",
            "operation": unchanged,
            "basis": "source Parquet schema + modality.json",
        },
        {
            "source": "LeRobot v2 data:annotation.human.coarse_action",
            "target": "data:annotation.human.coarse_action",
            "operation": unchanged,
            "basis": "source Parquet schema + modality.json",
        },
        {
            "source": "LeRobot v2 data:episode_index",
            "target": "data:episode_index",
            "operation": unchanged,
            "basis": "source part boundary and contiguous episode order",
        },
        {
            "source": "LeRobot v2 data:index",
            "target": "data:index",
            "operation": unchanged,
            "basis": "source part-local contiguous frame order",
        },
        {
            "source": "episodes.jsonl fields",
            "target": "episodes.parquet fields plus source_episode_index/source_task_index/source_task",
            "operation": "preserved; tasks replaced with verified natural-language instruction",
            "basis": "source episode metadata",
        },
        {
            "source": "info.json:splits",
            "target": "info.json:splits",
            "operation": "copied unchanged for every retained source part",
            "basis": "official source split metadata",
        },
        {
            "source": "embodiment.json/modality.json/metadata.json/initial_actions.npz",
            "target": "meta files with the same names",
            "operation": "copied byte-for-byte when present",
            "basis": "GR00T sidecars distributed with each source LeRobot part",
        },
        {
            "source": "HDF5 actions/states/action_dict/model_file and non-language ep_meta fields",
            "target": "not merged into the v3 data schema",
            "operation": "secondary representation only; schema recorded and language/length link checked",
            "basis": "request preference for existing legacy LeRobot representation; HDF5 is heterogeneous and image-free",
        },
    ]


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConversionError(f"config must contain a YAML mapping: {path}")
    expected = {f.name for f in Config.__dataclass_fields__.values()}
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ConversionError(f"invalid config keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
    string_fields = (
        "dataset_uid",
        "source_lerobot_subdir",
        "source_hdf5_subdir",
        "source_part_glob",
        "source_version",
        "target_version",
        "robot_type",
        "task_source",
    )
    for name in string_fields:
        if not isinstance(raw[name], str) or not raw[name]:
            raise ConversionError(f"config field {name} must be a non-empty string")
    for name in ("fps", "data_file_size_in_mb", "video_file_size_in_mb"):
        if isinstance(raw[name], bool) or not isinstance(raw[name], int):
            raise ConversionError(f"config field {name} must be an integer")
    if not isinstance(raw["crosscheck_hdf5"], bool):
        raise ConversionError("config field crosscheck_hdf5 must be boolean")
    preserve = raw["preserve_meta_files"]
    if not isinstance(preserve, (list, tuple)) or not all(
        isinstance(name, str) and name and Path(name).name == name for name in preserve
    ):
        raise ConversionError("config field preserve_meta_files must contain plain file names")
    raw["preserve_meta_files"] = tuple(raw["preserve_meta_files"])
    config = Config(**raw)
    if config.source_version != "v2.0" or config.target_version != "v3.0":
        raise ConversionError("this converter supports exactly LeRobot v2.0 -> v3.0")
    if config.task_source != "episode_remarks":
        raise ConversionError("task_source must be episode_remarks for this release")
    if config.fps <= 0 or config.data_file_size_in_mb <= 0 or config.video_file_size_in_mb <= 0:
        raise ConversionError("fps and output file-size limits must be positive")
    if config.robot_type != "GR1ArmsAndWaistFourierHands":
        raise ConversionError("robot_type must match the official GR1 embodiment")
    return config


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read JSON {path}: {exc}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ConversionError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc
    return rows


def _require_dependencies() -> tuple[Any, Any, Any]:
    try:
        import av
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install requirements.txt in the project .venv") from exc
    return av, pa, pq


def _infer_actual_features(source_info: dict[str, Any], parquet_schema: Any, video_info: dict[str, Any]) -> dict:
    import pyarrow as pa

    features = json.loads(json.dumps(source_info["features"]))
    for key, feature in features.items():
        if feature["dtype"] == "video":
            feature.pop("video_info", None)
            feature["info"] = dict(video_info)
            continue
        field_type = parquet_schema.field(key).type
        if pa.types.is_list(field_type) or pa.types.is_large_list(field_type) or pa.types.is_fixed_size_list(field_type):
            dtype = str(field_type.value_type)
        else:
            dtype = str(field_type)
        aliases = {"double": "float64", "float": "float32", "boolean": "bool"}
        feature["dtype"] = aliases.get(dtype, dtype)
        # LeRobot's official v2.1 -> v3.0 migration adds the dataset FPS to
        # every non-video feature.
        feature["fps"] = int(source_info["fps"])
        if key in {"observation.state", "action"}:
            feature["names"] = list(GR1_NAMES)
    return features


def _normalized_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a comparison-only copy with JSON tuple/list differences removed."""

    return json.loads(json.dumps(features))


def _video_metadata(path: Path) -> dict[str, Any]:
    from lerobot.datasets.video_utils import get_video_info

    info = get_video_info(path)
    info["video.video_backend"] = "pyav"
    return info


def _part_output_name(index: int, source_task: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in source_task).strip("-")
    return f"part-{index:03d}-{safe}"


def _validate_metadata_basics(source_root: Path, config: Config) -> tuple[dict, list[dict], str]:
    meta = source_root / "meta"
    required = ["info.json", "episodes.jsonl", "tasks.jsonl", "modality.json", "embodiment.json"]
    missing = [name for name in required if not (meta / name).is_file()]
    if missing:
        raise ConversionError(f"{source_root}: missing metadata files {missing}")
    info = _json(meta / "info.json")
    if info.get("codebase_version") != config.source_version:
        raise ConversionError(f"{source_root}: expected {config.source_version}, got {info.get('codebase_version')}")
    if info.get("robot_type") != config.robot_type or int(info.get("fps", -1)) != config.fps:
        raise ConversionError(f"{source_root}: robot_type/fps differs from config")
    expected_features = set(REQUIRED_COLUMNS) | {VIDEO_KEY}
    features = info.get("features")
    if not isinstance(features, dict) or set(features) != expected_features:
        raise ConversionError(f"{source_root}: info.json feature keys differ from the inspected release")
    if features["observation.state"].get("shape") != [44] or features["action"].get("shape") != [44]:
        raise ConversionError(f"{source_root}: info.json state/action shape is not [44]")
    if features[VIDEO_KEY].get("shape") != [256, 256, 3] or features[VIDEO_KEY].get("dtype") != "video":
        raise ConversionError(f"{source_root}: info.json ego video descriptor differs from the release")
    episodes = _jsonl(meta / "episodes.jsonl")
    if len(episodes) != info.get("total_episodes"):
        raise ConversionError(f"{source_root}: episodes.jsonl count differs from info.json")
    if info.get("total_videos") != len(episodes):
        raise ConversionError(f"{source_root}: source video/episode counts differ")
    source_tasks = _jsonl(meta / "tasks.jsonl")
    task_one = [row["task"] for row in source_tasks if row.get("task_index") == 1]
    if len(task_one) != 1:
        raise ConversionError(f"{source_root}: expected exactly one legacy task_index=1")
    return info, episodes, task_one[0]


def inspect_collection(
    config: Config,
    raw_root: Path,
    staging_root: Path,
    *,
    dataset_uid: str | None = None,
    source_task_filters: set[str] | None = None,
    max_episodes_per_part: int | None = None,
    eta_interval_seconds: float = 10.0,
    full_video_scan: bool = True,
) -> Collection:
    av, pa, pq = _require_dependencies()
    raw_dataset_root = raw_root / config.dataset_uid
    source_parent = raw_dataset_root / config.source_lerobot_subdir
    if not source_parent.is_dir():
        raise ConversionError(f"missing LeRobot source directory: {source_parent}")
    source_dirs = sorted(p for p in source_parent.glob(config.source_part_glob) if p.is_dir())
    if source_task_filters:
        source_dirs = [p for p in source_dirs if p.name.removeprefix("gr1_unified.") in source_task_filters]
    if not source_dirs:
        raise ConversionError("no source parts selected")

    parts: list[Part] = []
    total_expected = 0
    headers: list[tuple[Path, dict, list[dict], str]] = []
    for source_root in source_dirs:
        info, legacy_episodes, source_task = _validate_metadata_basics(source_root, config)
        headers.append((source_root, info, legacy_episodes, source_task))
        total_expected += min(len(legacy_episodes), max_episodes_per_part or len(legacy_episodes))

    progress = EtaProgress("preflight", total_expected, "episode", interval_seconds=eta_interval_seconds)
    completed = 0
    reference_features: dict | None = None
    for part_index, (source_root, info, legacy_episodes, source_task) in enumerate(headers):
        if source_root.name != f"gr1_unified.{source_task}":
            raise ConversionError(f"{source_root}: directory and legacy task disagree")
        selected = legacy_episodes[:max_episodes_per_part] if max_episodes_per_part else legacy_episodes
        instructions: list[str] = []
        for row in selected:
            instruction = row.get("remarks")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ConversionError(f"{source_root}: episode {row.get('episode_index')} lacks natural-language remarks")
            if instruction not in instructions:
                instructions.append(instruction)
        tasks = {0: "", **{i + 1: task for i, task in enumerate(instructions)}}
        task_to_index = {task: index for index, task in tasks.items()}

        part_eps: list[Episode] = []
        reference_schema = None
        first_video_info = None
        expected_global_index = 0
        for position, row in enumerate(selected):
            ep_idx = row.get("episode_index")
            if ep_idx != position:
                raise ConversionError(f"{source_root}: non-contiguous episode index {ep_idx} at row {position}")
            parquet_path = source_root / info["data_path"].format(episode_chunk=ep_idx // info["chunks_size"], episode_index=ep_idx)
            video_path = source_root / info["video_path"].format(
                episode_chunk=ep_idx // info["chunks_size"], episode_index=ep_idx, video_key=VIDEO_KEY
            )
            if not parquet_path.is_file() or not video_path.is_file():
                raise ConversionError(f"{source_root}: episode {ep_idx} data/video missing")
            parquet_file = pq.ParquetFile(parquet_path)
            schema = parquet_file.schema_arrow.remove_metadata()
            if set(schema.names) != set(REQUIRED_COLUMNS):
                raise ConversionError(f"{parquet_path}: columns are {schema.names}, expected {list(REQUIRED_COLUMNS)}")
            for key in ("observation.state", "action"):
                field_type = schema.field(key).type
                if not (
                    (
                        pa.types.is_list(field_type)
                        or pa.types.is_large_list(field_type)
                        or pa.types.is_fixed_size_list(field_type)
                    )
                    and pa.types.is_float64(field_type.value_type)
                ):
                    raise ConversionError(f"{parquet_path}: {key} must be a float64 list")
            expected_scalar_types = {
                "timestamp": pa.float64(),
                "next.reward": pa.float64(),
                "next.done": pa.bool_(),
                "task_index": pa.int64(),
                "annotation.human.fine_action": pa.int64(),
                "annotation.human.coarse_action": pa.int64(),
                "episode_index": pa.int64(),
                "index": pa.int64(),
            }
            for key, expected_type in expected_scalar_types.items():
                if schema.field(key).type != expected_type:
                    raise ConversionError(
                        f"{parquet_path}: {key} has dtype {schema.field(key).type}, expected {expected_type}"
                    )
            if reference_schema is None:
                reference_schema = schema
            elif not schema.equals(reference_schema):
                raise ConversionError(f"{parquet_path}: parquet schema differs within source part")
            rows = parquet_file.metadata.num_rows
            if rows != row.get("length"):
                raise ConversionError(f"{parquet_path}: {rows} rows but metadata length={row.get('length')}")
            if rows <= 0:
                raise ConversionError(f"{parquet_path}: empty episodes are not supported")
            check = pq.read_table(
                parquet_path,
                columns=["timestamp", "task_index", "episode_index", "index", "observation.state", "action"],
            )
            timestamp = check["timestamp"].to_numpy(zero_copy_only=False)
            # The release uses a stable 0.0499999523 s simulator tick rather
            # than the exact decimal 0.05.  Validate the cadence and local
            # reset without rewriting those official timestamp values.
            expected_step = 1.0 / config.fps
            if (
                len(timestamp) != rows
                or not np.isclose(timestamp[0], 0.0, rtol=0, atol=1e-9)
                or not np.allclose(np.diff(timestamp), expected_step, rtol=0, atol=1e-6)
            ):
                raise ConversionError(f"{parquet_path}: timestamps are not a monotonic {config.fps} Hz cadence")
            for vector_key in ("observation.state", "action"):
                lengths = pa.compute.list_value_length(check[vector_key]).to_numpy()
                if not np.all(lengths == 44):
                    raise ConversionError(f"{parquet_path}: {vector_key} contains non-44-D rows")
            if set(check["task_index"].to_pylist()) != {1} or set(check["episode_index"].to_pylist()) != {ep_idx}:
                raise ConversionError(f"{parquet_path}: legacy task/episode index mismatch")
            indices = check["index"].to_numpy(zero_copy_only=False)
            if not np.array_equal(indices, np.arange(expected_global_index, expected_global_index + rows)):
                raise ConversionError(f"{parquet_path}: global frame index is not contiguous")
            expected_global_index += rows

            if full_video_scan or position in {0, len(selected) // 2, len(selected) - 1}:
                with av.open(str(video_path)) as container:
                    stream = container.streams.video[0]
                    stream_info = (
                        stream.width,
                        stream.height,
                        int(stream.base_rate),
                        stream.codec_context.name,
                        stream.pix_fmt,
                        int(stream.frames),
                    )
                if stream_info[:5] != (256, 256, config.fps, "h264", "yuv420p"):
                    raise ConversionError(f"{video_path}: unexpected video schema {stream_info}")
                if stream_info[5] not in {0, rows}:
                    raise ConversionError(f"{video_path}: video frames={stream_info[5]}, expected {rows}")
            if first_video_info is None:
                first_video_info = _video_metadata(video_path)

            instruction = row["remarks"]
            part_eps.append(
                Episode(
                    episode_index=ep_idx,
                    source_parquet=parquet_path,
                    source_video=video_path,
                    length=rows,
                    instruction=instruction,
                    mapped_task_index=task_to_index[instruction],
                    source_task_index=1,
                    source_metadata=dict(row),
                )
            )
            completed += 1
            progress.update(completed, context=f"{source_task} episode {ep_idx}")

        assert reference_schema is not None and first_video_info is not None
        if max_episodes_per_part is None and expected_global_index != info["total_frames"]:
            raise ConversionError(f"{source_root}: frame total differs from info.json")
        features = _infer_actual_features(info, reference_schema, first_video_info)
        if reference_features is None:
            reference_features = features
        elif _normalized_features(features) != _normalized_features(reference_features):
            raise ConversionError(f"{source_root}: LeRobot feature schema differs across source parts")
        parts.append(
            Part(
                source_name=source_root.name,
                source_task=source_task,
                source_root=source_root,
                hdf5_path=raw_dataset_root / config.source_hdf5_subdir / f"{source_task}.hdf5",
                output_name=_part_output_name(part_index, source_task),
                source_info=info,
                features=features,
                episodes=part_eps,
                tasks=tasks,
                video_info=first_video_info,
            )
        )
    progress.finish(context="schema/timestamp/video headers valid")

    collection = Collection(
        config=config,
        raw_dataset_root=raw_dataset_root,
        output_path=staging_root / "lerobot_v3_0" / (dataset_uid or config.dataset_uid),
        parts=parts,
    )
    if config.crosscheck_hdf5:
        _crosscheck_hdf5(collection, eta_interval_seconds=eta_interval_seconds)
    return collection


def _crosscheck_hdf5(collection: Collection, *, eta_interval_seconds: float) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required") from exc
    total = collection.episodes
    progress = EtaProgress("hdf5-crosscheck", total, "episode", interval_seconds=eta_interval_seconds)
    completed = 0
    for part in collection.parts:
        if not part.hdf5_path.is_file():
            raise ConversionError(f"missing secondary HDF5 source {part.hdf5_path}")
        with h5py.File(part.hdf5_path, "r") as h5_file:
            data = h5_file.get("data")
            if data is None:
                raise ConversionError(f"{part.hdf5_path}: missing /data")
            if len(part.episodes) == part.source_info["total_episodes"] and len(data) != len(part.episodes):
                raise ConversionError(f"{part.hdf5_path}: HDF5/LeRobot episode counts differ")
            for episode in part.episodes:
                trajectory = episode.source_metadata.get("trajectory_id", "")
                try:
                    demo_number = int(trajectory.rsplit("-", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ConversionError(f"{part.source_name}: invalid trajectory_id {trajectory!r}") from exc
                demo_key = f"demo_{demo_number}"
                if demo_key not in data:
                    raise ConversionError(f"{part.hdf5_path}: missing {demo_key}")
                demo = data[demo_key]
                if int(demo.attrs.get("num_samples", -1)) != episode.length:
                    raise ConversionError(f"{part.hdf5_path}:{demo_key}: length differs from LeRobot")
                ep_meta = json.loads(demo.attrs["ep_meta"])
                if ep_meta.get("lang") != episode.instruction:
                    raise ConversionError(f"{part.hdf5_path}:{demo_key}: lang differs from episodes remarks")
                schema = _hdf5_schema(demo)
                part.hdf5_schemas.add(schema)
                completed += 1
                progress.update(completed, context=f"{part.source_task} {demo_key}")
    progress.finish(context="trajectory_id/length/language valid")


def _hdf5_schema(group: Any, prefix: str = "") -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Return the recursive per-frame dataset schema below an HDF5 group."""

    import h5py

    fields: list[tuple[str, tuple[int, ...], str]] = []
    for name, obj in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(obj, h5py.Dataset):
            fields.append((path, tuple(obj.shape[1:]), str(obj.dtype)))
        elif isinstance(obj, h5py.Group):
            fields.extend(_hdf5_schema(obj, path))
    return tuple(sorted(fields))


def summary(collection: Collection) -> dict[str, Any]:
    return {
        "dataset_uid": collection.output_path.name,
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_format": "LeRobot v2.0 collection (HDF5 cross-check)",
        "target_format": "LeRobot v3.0 collection",
        "output": str(collection.output_path),
        "parts": len(collection.parts),
        "episodes": collection.episodes,
        "frames": collection.frames,
        "fps": collection.config.fps,
        "robot_type": collection.config.robot_type,
        "tasks_including_unused_empty_legacy_task": sum(len(p.tasks) for p in collection.parts),
        "features": collection.parts[0].features,
        "part_summaries": [
            {
                "source": part.source_name,
                "output": part.output_name,
                "episodes": len(part.episodes),
                "frames": part.frames,
                "instructions": list(part.tasks.values()),
                "hdf5_schema_variants": len(part.hdf5_schemas),
            }
            for part in collection.parts
        ],
    }


def _decode_sampled_rgb(video_path: Path, length: int) -> np.ndarray:
    import av
    from lerobot.datasets.compute_stats import sample_indices

    wanted = sample_indices(length)
    positions = {frame_index: output_index for output_index, frame_index in enumerate(wanted)}
    frames = np.empty((len(wanted), 3, 256, 256), dtype=np.uint8)
    found = 0
    with av.open(str(video_path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            output_index = positions.get(frame_index)
            if output_index is not None:
                frames[output_index] = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                found += 1
            if found == len(wanted):
                break
    if found != len(wanted):
        raise ConversionError(f"{video_path}: decoded {found}/{len(wanted)} requested frames")
    return frames


def _numeric_array(column: Any) -> np.ndarray:
    values = column.to_pylist()
    if values and isinstance(values[0], list):
        return np.asarray(values)
    return np.asarray(values)


def _episode_stats(part: Part, episode: Episode, table: Any) -> dict[str, dict[str, np.ndarray]]:
    from lerobot.datasets.compute_stats import get_feature_stats

    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, feature in part.features.items():
        if feature["dtype"] == "video":
            array = _decode_sampled_rgb(episode.source_video, episode.length)
            feature_stats = get_feature_stats(array, axis=(0, 2, 3), keepdims=True)
            stats[key] = {
                name: value if name == "count" else np.squeeze(value / 255.0, axis=0)
                for name, value in feature_stats.items()
            }
            continue
        array = _numeric_array(table[key])
        stats[key] = get_feature_stats(array, axis=0, keepdims=array.ndim == 1)
    return stats


def _pack_data_and_stats(
    part: Part,
    new_root: Path,
    config: Config,
    *,
    eta_interval_seconds: float,
) -> tuple[list[dict], list[dict]]:
    import pandas as pd
    from lerobot.datasets.io_utils import get_parquet_file_size_in_mb
    from lerobot.datasets.utils import DEFAULT_CHUNK_SIZE, DEFAULT_DATA_PATH, update_chunk_file_indices

    _, pa, pq = _require_dependencies()
    chunk_index = file_index = 0
    size_mb = 0.0
    frame_offset = 0
    pending_frames = []
    episode_metadata = []
    episode_stats = []
    progress = EtaProgress(f"{part.output_name}:data+stats", len(part.episodes), "episode", interval_seconds=eta_interval_seconds)

    def flush() -> None:
        nonlocal pending_frames
        if not pending_frames:
            return
        path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(pending_frames, ignore_index=True).to_parquet(path, index=False)
        pending_frames = []

    for position, episode in enumerate(part.episodes):
        ep_size = get_parquet_file_size_in_mb(episode.source_parquet)
        if pending_frames and size_mb + ep_size >= config.data_file_size_in_mb:
            flush()
            chunk_index, file_index = update_chunk_file_indices(chunk_index, file_index, DEFAULT_CHUNK_SIZE)
            size_mb = 0.0
        table = pq.read_table(episode.source_parquet)
        frame = table.to_pandas()
        frame["task_index"] = np.full(episode.length, episode.mapped_task_index, dtype=np.int64)
        stats_table = pa.Table.from_pandas(frame, preserve_index=False)
        episode_stats.append(_episode_stats(part, episode, stats_table))
        pending_frames.append(frame)
        episode_metadata.append(
            {
                "episode_index": position,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": frame_offset,
                "dataset_to_index": frame_offset + episode.length,
            }
        )
        frame_offset += episode.length
        size_mb += ep_size
        progress.update(position + 1, context=f"source episode {episode.episode_index}")
    flush()
    progress.finish(context="parquet packed; episode stats computed")
    return episode_metadata, episode_stats


def _pack_videos(
    part: Part,
    new_root: Path,
    config: Config,
    *,
    eta_interval_seconds: float,
) -> list[dict]:
    from lerobot.datasets.io_utils import get_file_size_in_mb
    from lerobot.datasets.utils import DEFAULT_CHUNK_SIZE, DEFAULT_VIDEO_PATH, update_chunk_file_indices
    from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s

    chunk_index = file_index = 0
    size_mb = duration = 0.0
    pending: list[Path] = []
    records: list[dict] = []
    max_mb = config.video_file_size_in_mb
    progress = EtaProgress(f"{part.output_name}:video", len(part.episodes), "episode", interval_seconds=eta_interval_seconds)

    def flush() -> None:
        if not pending:
            return
        output = new_root / DEFAULT_VIDEO_PATH.format(video_key=VIDEO_KEY, chunk_index=chunk_index, file_index=file_index)
        concatenate_video_files(pending, output, compatibility_check=True)

    for position, episode in enumerate(part.episodes):
        ep_size = get_file_size_in_mb(episode.source_video)
        if pending and size_mb + ep_size >= max_mb:
            flush()
            pending.clear()
            chunk_index, file_index = update_chunk_file_indices(chunk_index, file_index, DEFAULT_CHUNK_SIZE)
            size_mb = duration = 0.0
        ep_duration = get_video_duration_in_s(episode.source_video)
        records.append(
            {
                "episode_index": position,
                f"videos/{VIDEO_KEY}/chunk_index": chunk_index,
                f"videos/{VIDEO_KEY}/file_index": file_index,
                f"videos/{VIDEO_KEY}/from_timestamp": duration,
                f"videos/{VIDEO_KEY}/to_timestamp": duration + ep_duration,
            }
        )
        pending.append(episode.source_video)
        size_mb += ep_size
        duration += ep_duration
        progress.update(position + 1, context=f"source episode {episode.episode_index}")
    flush()
    progress.finish(context="H.264 packets remuxed without re-encoding")
    return records


def _write_info(part: Part, output_root: Path, config: Config) -> None:
    from lerobot.datasets.io_utils import write_info
    from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH, DatasetInfo

    info = DatasetInfo(
        codebase_version="v3.0",
        fps=config.fps,
        features=part.features,
        total_episodes=len(part.episodes),
        total_frames=part.frames,
        total_tasks=len(part.tasks),
        chunks_size=1000,
        data_files_size_in_mb=config.data_file_size_in_mb,
        video_files_size_in_mb=config.video_file_size_in_mb,
        data_path=DEFAULT_DATA_PATH,
        video_path=DEFAULT_VIDEO_PATH,
        robot_type=config.robot_type,
        splits=dict(part.source_info.get("splits", {"train": "0:100"})),
    )
    write_info(info, output_root)


def _write_tasks(part: Part, output_root: Path) -> None:
    import pandas as pd
    from lerobot.datasets.io_utils import write_tasks

    task_strings = [part.tasks[index] for index in sorted(part.tasks)]
    frame = pd.DataFrame(
        {"task_index": sorted(part.tasks)}, index=pd.Index(task_strings, name="task")
    )
    write_tasks(frame, output_root)


def _write_episode_metadata(
    part: Part,
    output_root: Path,
    data_records: list[dict],
    video_records: list[dict],
    episode_stats: list[dict],
) -> None:
    from datasets import Dataset
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_episodes, write_stats
    from lerobot.utils.utils import flatten_dict

    rows = []
    for position, (episode, data, video, stats) in enumerate(
        zip(part.episodes, data_records, video_records, episode_stats, strict=True)
    ):
        legacy = dict(episode.source_metadata)
        legacy["episode_index"] = position
        legacy["tasks"] = [episode.instruction]
        legacy["source_episode_index"] = episode.episode_index
        legacy["source_task_index"] = episode.source_task_index
        legacy["source_task"] = part.source_task
        row = {**data, **video, **legacy, **flatten_dict({"stats": stats})}
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        rows.append(row)
    write_episodes(Dataset.from_list(rows), output_root)
    write_stats(aggregate_stats(episode_stats), output_root)


def _copy_source_metadata(part: Part, output_root: Path, config: Config) -> None:
    meta_out = output_root / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    for name in config.preserve_meta_files:
        source = part.source_root / "meta" / name
        if source.is_file():
            shutil.copy2(source, meta_out / name)


def _part_manifest(part: Part, output_root: Path, config: Config) -> None:
    copied_meta_files = [
        name for name in config.preserve_meta_files if (part.source_root / "meta" / name).is_file()
    ]
    hdf5_schema_variants = [
        [
            {"path": path, "per_frame_shape": list(shape), "dtype": dtype}
            for path, shape, dtype in schema
        ]
        for schema in sorted(part.hdf5_schemas)
    ]
    target_video_files = len(list((output_root / "videos" / VIDEO_KEY).rglob("*.mp4")))
    manifest = {
        "format": "lerobot_v3_0",
        "converter": Path(__file__).name,
        "official_migration_basis": [
            "lerobot.datasets.v21.convert_dataset_v20_to_v21",
            "lerobot.scripts.convert_dataset_v21_to_v30",
        ],
        "official_references": official_references(),
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_dataset": part.source_name,
        "source_version": "v2.0",
        "source_hdf5": part.hdf5_path.name,
        "target_version": "v3.0",
        "episodes": len(part.episodes),
        "frames": part.frames,
        "fps": config.fps,
        "robot_type": config.robot_type,
        "source_splits": part.source_info.get("splits", {}),
        "target_splits": part.source_info.get("splits", {}),
        "copied_meta_files": copied_meta_files,
        "task_mapping": {
            "source_frame_task_index": 1,
            "source_task": part.source_task,
            "target_tasks": part.tasks,
            "basis": "meta/episodes.jsonl remarks, cross-checked against HDF5 ep_meta.lang",
        },
        "video": {
            "operation": "packet remux only",
            "reencoded": False,
            "key": VIDEO_KEY,
            "source_episode_files": len(part.episodes),
            "target_packed_files": target_video_files,
            "note": "LeRobot v3.0 info.json has no total_videos field; counts are recorded here",
        },
        "hdf5_crosscheck": {
            "role": "secondary representation; not merged into target features",
            "trajectory_link": "episodes.jsonl trajectory_id suffix -> /data/demo_<id>",
            "checks": ["episode count", "episode length", "ep_meta.lang"],
            "schema_variants": hdf5_schema_variants,
        },
        "value_changes": ["task_index remapped to per-episode official natural-language instruction"],
        "preserved_without_numeric_conversion": [
            key for key in REQUIRED_COLUMNS if key != "task_index"
        ],
        "field_mapping": field_mapping(),
        "features": part.features,
        "episodes_traceability": [
            {
                "target_episode_index": position,
                "source_episode_index": ep.episode_index,
                "source_trajectory_id": ep.source_metadata.get("trajectory_id"),
                "instruction": ep.instruction,
                "source_task_index": ep.source_task_index,
                "target_task_index": ep.mapped_task_index,
                "length": ep.length,
            }
            for position, ep in enumerate(part.episodes)
        ],
    }
    (output_root / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def convert_part(part: Part, output_root: Path, config: Config, *, eta_interval_seconds: float) -> None:
    output_root.mkdir(parents=True)
    _write_info(part, output_root, config)
    _write_tasks(part, output_root)
    data_records, stats = _pack_data_and_stats(
        part, output_root, config, eta_interval_seconds=eta_interval_seconds
    )
    video_records = _pack_videos(
        part, output_root, config, eta_interval_seconds=eta_interval_seconds
    )
    _write_episode_metadata(part, output_root, data_records, video_records, stats)
    _copy_source_metadata(part, output_root, config)
    _part_manifest(part, output_root, config)
    validate_part(part, output_root, config)


def _decode_source_frame(path: Path, index: int) -> np.ndarray:
    import av
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index == index:
                return frame.to_ndarray(format="rgb24")
    raise ConversionError(f"{path}: cannot decode frame {index}")


def validate_part(part: Part, output_root: Path, config: Config) -> None:
    import datasets.config

    for name in config.preserve_meta_files:
        source = part.source_root / "meta" / name
        if not source.is_file():
            continue
        target = output_root / "meta" / name
        if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
            raise ConversionError(f"{output_root}: preserved metadata differs for {name}")

    # Dataset.from_parquet writes cache/lock files even for a completely local
    # dataset.  Keep validation independent of the caller's HOME permissions
    # and ensure those transient files are part of atomic failure cleanup.
    old_cache = datasets.config.HF_DATASETS_CACHE
    validation_cache = output_root / ".validation-cache"
    validation_cache.mkdir()
    datasets.config.HF_DATASETS_CACHE = validation_cache
    try:
        _validate_part_with_lerobot(part, output_root, config)
    finally:
        datasets.config.HF_DATASETS_CACHE = old_cache
        shutil.rmtree(validation_cache, ignore_errors=True)


def _validate_part_with_lerobot(part: Part, output_root: Path, config: Config) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import DEFAULT_DATA_PATH

    dataset = LeRobotDataset(repo_id=part.output_name, root=output_root)
    if dataset.num_episodes != len(part.episodes) or len(dataset) != part.frames:
        raise ConversionError(f"{output_root}: LeRobot reopen count mismatch")
    if (
        _normalized_features(dataset.meta.features) != _normalized_features(part.features)
        or int(dataset.meta.fps) != config.fps
    ):
        raise ConversionError(f"{output_root}: LeRobot feature/fps metadata mismatch")
    if (
        set(dataset.meta.tasks.index) != set(part.tasks.values())
        or dataset.meta.info.total_tasks != len(part.tasks)
        or dataset.meta.info.splits != part.source_info.get("splits", {})
        or dataset.meta.video_keys != [VIDEO_KEY]
    ):
        raise ConversionError(f"{output_root}: tasks table mismatch")
    for task_index, task in part.tasks.items():
        if int(dataset.meta.tasks.loc[task, "task_index"]) != task_index:
            raise ConversionError(f"{output_root}: task index mismatch for {task!r}")

    expected_from_index = 0
    data_groups: dict[Path, list[tuple[int, Episode, Any]]] = {}
    for target_ep_index, episode in enumerate(part.episodes):
        row = dataset.meta.episodes[target_ep_index]
        expected_to_index = expected_from_index + episode.length
        actual_tasks = set(row["tasks"])
        trace_values = (
            int(row["source_episode_index"]),
            int(row["source_task_index"]),
            row["source_task"],
            row["trajectory_id"],
        )
        expected_trace_values = (
            episode.episode_index,
            episode.source_task_index,
            part.source_task,
            episode.source_metadata.get("trajectory_id"),
        )
        if (
            int(row["episode_index"]) != target_ep_index
            or int(row["length"]) != episode.length
            or int(row["dataset_from_index"]) != expected_from_index
            or int(row["dataset_to_index"]) != expected_to_index
            or actual_tasks != {episode.instruction}
            or row["remarks"] != episode.instruction
            or trace_values != expected_trace_values
        ):
            raise ConversionError(f"{output_root}: episode metadata mismatch at {target_ep_index}")
        data_path = output_root / dataset.meta.get_data_file_path(target_ep_index)
        video_path = output_root / dataset.meta.get_video_file_path(target_ep_index, VIDEO_KEY)
        if not data_path.is_file() or not video_path.is_file():
            raise ConversionError(f"{output_root}: episode file reference missing at {target_ep_index}")
        data_groups.setdefault(data_path, []).append((target_ep_index, episode, row))
        video_from = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
        video_to = float(row[f"videos/{VIDEO_KEY}/to_timestamp"])
        expected_duration = episode.length / config.fps
        if video_to <= video_from or not np.isclose(
            video_to - video_from, expected_duration, rtol=0, atol=1 / config.fps
        ):
            raise ConversionError(f"{output_root}: video duration mismatch at {target_ep_index}")
        expected_from_index = expected_to_index
    if expected_from_index != part.frames:
        raise ConversionError(f"{output_root}: episode boundaries do not cover all frames")

    _, _, pq = _require_dependencies()
    # Independently scan the lightweight index columns of every packed file.
    # This proves that all episode metadata boundaries describe the actual
    # serialized rows, rather than merely agreeing with metadata we generated.
    for data_path, entries in data_groups.items():
        packed_indices = pq.read_table(
            data_path, columns=["episode_index", "index", "task_index"]
        )
        file_offset = 0
        for target_ep_index, episode, row in entries:
            segment = packed_indices.slice(file_offset, episode.length)
            expected_global = np.arange(
                int(row["dataset_from_index"]), int(row["dataset_to_index"]), dtype=np.int64
            )
            if (
                segment.num_rows != episode.length
                or not np.array_equal(
                    segment["episode_index"].to_numpy(zero_copy_only=False),
                    np.full(episode.length, target_ep_index, dtype=np.int64),
                )
                or not np.array_equal(
                    segment["index"].to_numpy(zero_copy_only=False), expected_global
                )
                or not np.array_equal(
                    segment["task_index"].to_numpy(zero_copy_only=False),
                    np.full(episode.length, episode.mapped_task_index, dtype=np.int64),
                )
            ):
                raise ConversionError(
                    f"{output_root}: packed episode boundary/index mismatch at {target_ep_index}"
                )
            file_offset += episode.length
        if file_offset != packed_indices.num_rows:
            raise ConversionError(f"{output_root}: packed data file has unreferenced rows: {data_path}")

    sample_eps = sorted({0, len(part.episodes) // 2, len(part.episodes) - 1})
    for target_ep_index in sample_eps:
        episode = part.episodes[target_ep_index]
        source = pq.read_table(episode.source_parquet)
        row = dataset.meta.episodes[target_ep_index]
        global_start = int(row["dataset_from_index"])
        packed_path = output_root / DEFAULT_DATA_PATH.format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        packed = pq.read_table(packed_path)
        packed_indices = packed["index"].to_numpy(zero_copy_only=False)
        for local_index in sorted({0, episode.length // 2, episode.length - 1}):
            item = dataset[global_start + local_index]
            source_global_index = source["index"][local_index].as_py()
            packed_positions = np.flatnonzero(packed_indices == source_global_index)
            if len(packed_positions) != 1:
                raise ConversionError(
                    f"{output_root}: packed index lookup failed ep={target_ep_index} frame={local_index}"
                )
            packed_index = int(packed_positions[0])
            # Prove serialized values are bit-exact.  LeRobot/Hugging Face's
            # Python transform presents fixed-size float64 sequences as
            # float32 torch tensors, so reader validation below separately
            # checks that documented presentation cast.
            for key in REQUIRED_COLUMNS:
                expected_stored = (
                    episode.mapped_task_index if key == "task_index" else source[key][local_index].as_py()
                )
                actual_stored = packed[key][packed_index].as_py()
                if not np.array_equal(np.asarray(actual_stored), np.asarray(expected_stored)):
                    raise ConversionError(
                        f"{output_root}: stored {key} mismatch ep={target_ep_index} frame={local_index}"
                    )
            for key in ("observation.state", "action"):
                actual = item[key].detach().cpu().numpy()
                expected = np.asarray(source[key][local_index].as_py())
                if not np.array_equal(actual, expected.astype(actual.dtype)):
                    raise ConversionError(
                        f"{output_root}: reader {key} mismatch ep={target_ep_index} frame={local_index}"
                    )
            if item["task"] != episode.instruction:
                raise ConversionError(f"{output_root}: task lookup mismatch ep={target_ep_index}")
            image = item[VIDEO_KEY].detach().cpu().numpy().transpose(1, 2, 0)
            if np.issubdtype(image.dtype, np.floating):
                image = np.rint(image * 255).clip(0, 255).astype(np.uint8)
            expected_image = _decode_source_frame(episode.source_video, local_index)
            if not np.array_equal(image, expected_image):
                # Decoder implementations can differ by one code value.  Remuxing
                # is already packet-exact; tolerate only that known decode effect.
                if np.max(np.abs(image.astype(np.int16) - expected_image.astype(np.int16))) > 1:
                    raise ConversionError(f"{output_root}: video frame mismatch ep={target_ep_index} frame={local_index}")
    del dataset


def _collection_manifest(collection: Collection, temporary_path: Path) -> None:
    manifest = {
        "format": "lerobot_v3_0_collection",
        "dataset_uid": collection.output_path.name,
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_version": "v2.0",
        "target_version": "v3.0",
        "parts": [
            {
                "source": part.source_name,
                "path": part.output_name,
                "source_task": part.source_task,
                "episodes": len(part.episodes),
                "frames": part.frames,
                "tasks": list(part.tasks.values()),
            }
            for part in collection.parts
        ],
        "total_episodes": collection.episodes,
        "total_frames": collection.frames,
        "partition_reason": "preserve the 24 official source LeRobot dataset/split boundaries",
        "schema_partitions": 1,
    }
    (temporary_path / "collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _resume_fingerprint_payload(collection: Collection) -> dict[str, Any]:
    """Return the semantic inputs that must match before reusing converted parts."""

    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "raw_dataset_root": str(collection.raw_dataset_root.resolve()),
        "output_path": str(collection.output_path.resolve()),
        "config": asdict(collection.config),
        "parts": [
            {
                "source_name": part.source_name,
                "source_task": part.source_task,
                "source_root": str(part.source_root.resolve()),
                "source_hdf5": str(part.hdf5_path.resolve()),
                "output_name": part.output_name,
                "source_info": part.source_info,
                "features": part.features,
                "tasks": part.tasks,
                "hdf5_schemas": [
                    [[path, list(shape), dtype] for path, shape, dtype in schema]
                    for schema in sorted(part.hdf5_schemas)
                ],
                "episodes": [
                    {
                        "episode_index": episode.episode_index,
                        "source_parquet": str(episode.source_parquet.resolve()),
                        "source_video": str(episode.source_video.resolve()),
                        "length": episode.length,
                        "instruction": episode.instruction,
                        "mapped_task_index": episode.mapped_task_index,
                        "source_task_index": episode.source_task_index,
                        "source_metadata": episode.source_metadata,
                    }
                    for episode in part.episodes
                ],
            }
            for part in collection.parts
        ],
    }


def _resume_fingerprint(collection: Collection) -> str:
    payload = _resume_fingerprint_payload(collection)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resume_paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output.with_name(f".{output.name}.resume"),
        output.with_name(f".{output.name}.resume-state"),
        output.with_name(f".{output.name}.resume.lock"),
    )


def _resume_marker_path(state_root: Path, part: Part) -> Path:
    return state_root / RESUME_PARTS_DIR / f"{part.output_name}.json"


def _resume_marker_payload(fingerprint: str, part: Part) -> dict[str, Any]:
    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "collection_fingerprint": fingerprint,
        "part": part.output_name,
        "source": part.source_name,
        "episodes": len(part.episodes),
        "frames": part.frames,
    }


def _read_resume_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"{description} must contain a JSON object: {path}")
    return value


def _prepare_resume_workspace(
    collection: Collection,
    data_root: Path,
    state_root: Path,
    fingerprint: str,
) -> list[Part]:
    """Create/check a checkpoint and return only parts that still need conversion."""

    expected_state = {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "collection_fingerprint": fingerprint,
        "dataset_uid": collection.output_path.name,
        "raw_dataset_root": str(collection.raw_dataset_root.resolve()),
        "output_path": str(collection.output_path.resolve()),
        "parts": [part.output_name for part in collection.parts],
        "total_episodes": collection.episodes,
        "total_frames": collection.frames,
    }
    state_path = state_root / RESUME_STATE_FILE
    if state_root.exists():
        if not state_root.is_dir() or not state_path.is_file():
            raise ConversionError(
                f"resume state is incomplete: {state_root}; move it aside to start a new checkpoint"
            )
        actual_state = _read_resume_json(state_path, "resume state")
        if actual_state != expected_state:
            raise ConversionError(
                f"resume checkpoint does not match this conversion: {state_root}; "
                "use the original arguments or move the checkpoint aside"
            )
    else:
        state_root.mkdir(parents=True)
        _write_json_atomic(state_path, expected_state)

    if data_root.exists() and not data_root.is_dir():
        raise ConversionError(f"resume data path is not a directory: {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    markers_root = state_root / RESUME_PARTS_DIR
    markers_root.mkdir(exist_ok=True)

    expected_data_names = {part.output_name for part in collection.parts} | {
        "collection_manifest.json"
    }
    unexpected_data = sorted(
        path.name for path in data_root.iterdir() if path.name not in expected_data_names
    )
    if unexpected_data:
        raise ConversionError(f"resume data contains unexpected entries: {unexpected_data}")
    expected_marker_names = {f"{part.output_name}.json" for part in collection.parts}
    unexpected_markers = sorted(
        path.name for path in markers_root.iterdir() if path.name not in expected_marker_names
    )
    if unexpected_markers:
        raise ConversionError(f"resume state contains unexpected part markers: {unexpected_markers}")

    pending: list[Part] = []
    reused = 0
    for part in collection.parts:
        part_root = data_root / part.output_name
        marker_path = _resume_marker_path(state_root, part)
        expected_marker = _resume_marker_payload(fingerprint, part)
        marker_matches = False
        if marker_path.is_file():
            try:
                marker_matches = _read_resume_json(marker_path, "part checkpoint") == expected_marker
            except ConversionError:
                marker_matches = False
        if part_root.is_dir() and marker_matches:
            print(
                f"[resume] validating completed part: {part.source_task}",
                file=sys.stderr,
                flush=True,
            )
            try:
                validate_part(part, part_root, collection.config)
            except Exception as exc:
                print(
                    f"[resume] checkpoint invalid; rebuilding {part.source_task}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                reused += 1
                continue
        marker_path.unlink(missing_ok=True)
        if part_root.exists():
            if not part_root.is_dir():
                raise ConversionError(f"resume part path is not a directory: {part_root}")
            shutil.rmtree(part_root)
        pending.append(part)

    print(
        f"[resume] reused {reused}/{len(collection.parts)} verified parts; "
        f"pending {len(pending)}",
        file=sys.stderr,
        flush=True,
    )
    return pending


def _acquire_resume_lock(lock_path: Path) -> int:
    import fcntl

    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ConversionError(f"another resume process is using {lock_path}") from exc
    return descriptor


def _release_resume_lock(descriptor: int) -> None:
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _convert_part_worker(
    part: Part,
    output_root: Path,
    config: Config,
    eta_interval_seconds: float,
    marker_path: Path | None = None,
    marker_payload: dict[str, Any] | None = None,
) -> str:
    """Process-pool entry point; each part owns a disjoint output directory."""

    convert_part(part, output_root, config, eta_interval_seconds=eta_interval_seconds)
    if marker_path is not None:
        if marker_payload is None:
            raise ValueError("resume marker payload is required")
        _write_json_atomic(marker_path, marker_payload)
    return part.output_name


def convert_collection(
    collection: Collection,
    *,
    overwrite: bool,
    eta_interval_seconds: float,
    workers: int = 1,
    resume: bool = False,
) -> Path:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    output = collection.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    resume_data, resume_state, resume_lock = _resume_paths(output)
    lock_descriptor = _acquire_resume_lock(resume_lock) if resume else None
    try:
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if resume:
            fingerprint = _resume_fingerprint(collection)
            temporary = resume_data
            pending_parts = _prepare_resume_workspace(
                collection, temporary, resume_state, fingerprint
            )
        else:
            fingerprint = None
            temporary = output.with_name(f".{output.name}.incomplete-{uuid.uuid4().hex}")
            temporary.mkdir()
            pending_parts = list(collection.parts)

        worker_count = min(workers, len(pending_parts)) if pending_parts else 0
        if worker_count == 1:
            for index, part in enumerate(pending_parts):
                print(
                    f"[collection] pending part {index+1}/{len(pending_parts)}: {part.source_task}",
                    file=sys.stderr,
                    flush=True,
                )
                convert_part(
                    part,
                    temporary / part.output_name,
                    collection.config,
                    eta_interval_seconds=eta_interval_seconds,
                )
                if resume:
                    assert fingerprint is not None
                    _write_json_atomic(
                        _resume_marker_path(resume_state, part),
                        _resume_marker_payload(fingerprint, part),
                    )
        elif worker_count > 1:
            print(
                f"[collection] converting {len(pending_parts)} pending parts with "
                f"{worker_count} workers",
                file=sys.stderr,
                flush=True,
            )
            # Spawn avoids inheriting PyArrow/PyAV/HDF5 runtime state from the
            # completed preflight. Parts read common sources but write disjoint
            # directories, so no writer or Hugging Face cache is shared.
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            )
            futures: dict[Future[str], Part] = {}
            try:
                for part in pending_parts:
                    marker_path = _resume_marker_path(resume_state, part) if resume else None
                    marker_payload = (
                        _resume_marker_payload(fingerprint, part)
                        if resume and fingerprint is not None
                        else None
                    )
                    future = executor.submit(
                        _convert_part_worker,
                        part,
                        temporary / part.output_name,
                        collection.config,
                        eta_interval_seconds,
                        marker_path,
                        marker_payload,
                    )
                    futures[future] = part
                completed = 0
                for future in as_completed(futures):
                    part = futures[future]
                    future.result()
                    completed += 1
                    print(
                        f"[collection] completed pending part {completed}/{len(pending_parts)}: "
                        f"{part.source_task}",
                        file=sys.stderr,
                        flush=True,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
        _collection_manifest(collection, temporary)
        publish_temporary_output(temporary, output, overwrite=overwrite)
        if resume and resume_state.exists():
            try:
                shutil.rmtree(resume_state)
            except OSError as exc:
                print(
                    f"warning: could not remove resume state {resume_state}: {exc}",
                    file=sys.stderr,
                )
    except BaseException:
        if "temporary" in locals() and temporary.exists() and not resume:
            shutil.rmtree(temporary)
        raise
    finally:
        if lock_descriptor is not None:
            _release_resume_lock(lock_descriptor)
    return output


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--dataset-uid", help="Output UID override, useful for a smoke conversion.")
    parser.add_argument("--inspect-only", "--dry-run", dest="inspect_only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep verified part checkpoints and resume an interrupted conversion.",
    )
    parser.add_argument("--eta-interval-seconds", type=_positive_float, default=10.0)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Convert independent source parts in parallel processes (preflight remains ordered).",
    )
    parser.add_argument("--source-task", action="append", help="Select an exact task suffix; repeatable.")
    parser.add_argument("--max-episodes-per-part", type=_positive_int, help="Explicit subset for smoke tests only.")
    parser.add_argument(
        "--sample-video-headers",
        action="store_true",
        help="Inspect only first/middle/last video headers per part (full scan is the safe default).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing and --overwrite are mutually exclusive")
    try:
        config = load_config(args.config)
        output = args.staging_root / "lerobot_v3_0" / (args.dataset_uid or config.dataset_uid)
        if output.exists() and args.skip_existing and not args.inspect_only:
            print(f"skipped existing output: {output}")
            return 0
        collection = inspect_collection(
            config,
            args.raw_root,
            args.staging_root,
            dataset_uid=args.dataset_uid,
            source_task_filters=set(args.source_task) if args.source_task else None,
            max_episodes_per_part=args.max_episodes_per_part,
            eta_interval_seconds=args.eta_interval_seconds,
            full_video_scan=not args.sample_video_headers,
        )
        print(json.dumps(summary(collection), ensure_ascii=False, indent=2))
        if args.inspect_only:
            print("preflight complete; no output written")
            return 0
        written = convert_collection(
            collection,
            overwrite=args.overwrite,
            eta_interval_seconds=args.eta_interval_seconds,
            workers=args.workers,
            resume=args.resume,
        )
        print(f"wrote verified LeRobot v3 collection: {written}")
        return 0
    except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
