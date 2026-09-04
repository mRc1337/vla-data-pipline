"""Read the released DexCap HDF5 files without materialising the dataset.

The HDF5 files are already post-processed by the official DexCap release.  In
particular, ``actions[t]`` is the next-frame 46-D EEF target (the files are
named ``5actiongap`` but the stored action relationship is one frame).  This
reader keeps the source arrays and dtypes intact; the generic writer only
changes their LeRobot field names.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from convert_core.checkpoint import canonical_fingerprint
from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError


SOURCE_DATASET = "j96w/DexCap"
SOURCE_REVISION = "released-dexcap-data"
DEFAULT_FPS = 10
POINT_SHAPE = (10_000, 6)
IMAGE_SHAPE = (84, 84, 3)

REQUIRED_DATASETS: tuple[tuple[str, tuple[int, ...], str], ...] = (
    ("actions", (46,), "float64"),
    ("dones", (), "int64"),
    ("glove_states", (63,), "float64"),
    ("obs/agentview_image", IMAGE_SHAPE, "uint8"),
    ("obs/label", (), "int64"),
    ("obs/pointcloud", POINT_SHAPE, "float64"),
    ("obs/robot0_eef_hand", (32,), "float64"),
    ("obs/robot0_eef_pos", (6,), "float64"),
    ("obs/robot0_eef_quat", (8,), "float64"),
    ("rewards", (), "float64"),
    ("states", (16,), "float64"),
)


@dataclass(frozen=True)
class DexCapPartitionSpec:
    name: str
    filename: str
    instruction: str
    task_basis: str
    robot_type: str = "dexcap"
    phase_group_size: int = 1
    source_bytes: int = 0
    sha256: str = ""


PARTITION_SPECS = (
    DexCapPartitionSpec(
        name="packaging_wild",
        filename="hand_packaging_wild_1-20_5actiongap_10000points.hdf5",
        instruction="Package objects in the wild using the DexCap hand.",
        task_basis="DexCap release filename and dataset card task description",
    ),
    DexCapPartitionSpec(
        name="wiping",
        filename="hand_wiping_1-14_5actiongap_10000points.hdf5",
        instruction="Wipe a surface using the DexCap hand.",
        task_basis="DexCap release filename and dataset card task description",
    ),
)
PARTITIONS_BY_NAME = {item.name: item for item in PARTITION_SPECS}
PARTITIONS_BY_FILE = {item.filename: item for item in PARTITION_SPECS}


@dataclass(frozen=True)
class DexCapPartitionInfo:
    spec: DexCapPartitionSpec
    source_path: Path
    source_relative_path: str
    plan: DatasetConversionPlan
    all_episode_count: int
    all_frame_count: int
    selected_logical_bytes: int
    source_schema: tuple[tuple[str, tuple[int, ...], str], ...]
    schema_fingerprint: str
    episode_length_summary: dict[str, int]
    payload_scan: dict[str, Any]


def partition_spec(value: str | Path) -> DexCapPartitionSpec:
    key = Path(value).name
    if key in PARTITIONS_BY_FILE:
        return PARTITIONS_BY_FILE[key]
    try:
        return PARTITIONS_BY_NAME[str(value)]
    except KeyError as exc:
        raise ConversionError(
            f"unknown DexCap partition {value!r}; expected {sorted(PARTITIONS_BY_NAME)}"
        ) from exc


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("h5py is required for DexCap conversion") from exc
    return h5py


def _demo_names(data: Any) -> list[str]:
    names = list(data.keys())
    if not names or any(not name.startswith("demo_") for name in names):
        raise ConversionError("DexCap HDF5 data group must contain demo_N groups")
    try:
        return sorted(names, key=lambda value: int(value.removeprefix("demo_")))
    except ValueError as exc:
        raise ConversionError("DexCap demo names must have numeric suffixes") from exc


def _schema(demo: Any) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    rows: list[tuple[str, tuple[int, ...], str]] = []
    for path, tail_shape, dtype in REQUIRED_DATASETS:
        if path not in demo:
            raise ConversionError(f"DexCap episode is missing {path}")
        dataset = demo[path]
        if tuple(dataset.shape[1:]) != tail_shape:
            raise ConversionError(
                f"DexCap {path} shape tail is {dataset.shape[1:]}, expected {tail_shape}"
            )
        if str(dataset.dtype) != dtype:
            raise ConversionError(
                f"DexCap {path} dtype is {dataset.dtype}, expected {dtype}"
            )
        rows.append((path, tail_shape, dtype))
    return tuple(rows)


def _sample_payload(demo: Any, frames: int) -> dict[str, Any]:
    positions = sorted({0, frames // 2, frames - 1})
    pointcloud = demo["obs/pointcloud"]
    image = demo["obs/agentview_image"]
    for index in positions:
        points = np.asarray(pointcloud[index])
        pixels = np.asarray(image[index])
        if not np.isfinite(points).all():
            raise ConversionError("sampled DexCap point cloud contains non-finite values")
        if pixels.shape != IMAGE_SHAPE:
            raise ConversionError("sampled DexCap image shape changed")
    if frames > 1:
        sample_indices = sorted({0, frames // 2, frames - 2})
        for index in sample_indices:
            expected = np.concatenate(
                (
                    np.asarray(demo["obs/robot0_eef_pos"][index + 1]),
                    np.asarray(demo["obs/robot0_eef_quat"][index + 1]),
                    np.asarray(demo["obs/robot0_eef_hand"][index + 1]),
                )
            )
            if not np.array_equal(np.asarray(demo["actions"][index]), expected):
                raise ConversionError(
                    "DexCap action relationship changed: expected actions[t] "
                    "to equal the concatenated EEF observation at t+1"
                )
    return {
        "sampled_frames": len(positions),
        "sampled_pointcloud_frames": len(positions),
        "pointcloud_shape": list(POINT_SHAPE),
        "image_shape": list(IMAGE_SHAPE),
        "action_is_next_observation": True,
    }


def _frame_fields(demo: Any, index: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(demo[path][index])
        for path in (
            "actions",
            "glove_states",
            "obs/robot0_eef_hand",
            "obs/robot0_eef_pos",
            "obs/robot0_eef_quat",
            "obs/pointcloud",
            "states",
            "rewards",
            "dones",
            "obs/label",
        )
    )


def inspect_partition(
    source_path: Path,
    *,
    raw_dataset_root: Path,
    collection_output: Path,
    max_phase_groups: int | None = None,
    verify_sha256: bool = False,
    full_lowdim_scan: bool = False,
    full_pointcloud_scan: bool = False,
) -> DexCapPartitionInfo:
    h5py = _require_h5py()
    source_path = source_path.absolute()
    raw_dataset_root = raw_dataset_root.absolute()
    spec = partition_spec(source_path.name)
    if not source_path.is_file():
        raise ConversionError(f"DexCap source file does not exist: {source_path}")
    try:
        relative = source_path.relative_to(raw_dataset_root).as_posix()
    except ValueError as exc:
        raise ConversionError("DexCap source must be below the raw root") from exc

    digest = ""
    if verify_sha256:
        digest_maker = hashlib.sha256()
        with source_path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest_maker.update(chunk)
        digest = digest_maker.hexdigest()

    with h5py.File(source_path, "r") as handle:
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise ConversionError(f"DexCap file has no /data group: {source_path}")
        names = _demo_names(handle["data"])
        reference_schema = _schema(handle["data"][names[0]])
        lengths: list[int] = []
        logical_bytes = 0
        for name in names:
            demo = handle["data"][name]
            schema = _schema(demo)
            if schema != reference_schema:
                raise ConversionError(f"DexCap episode schema changed at {name}")
            frames = int(demo["actions"].shape[0])
            if frames <= 0 or int(demo.attrs.get("num_samples", frames)) != frames:
                raise ConversionError(f"DexCap episode {name} has invalid frame count")
            if any(int(demo[path].shape[0]) != frames for path, _, _ in REQUIRED_DATASETS):
                raise ConversionError(f"DexCap episode {name} fields are not time aligned")
            lengths.append(frames)
            logical_bytes += sum(int(demo[path].size * demo[path].dtype.itemsize) for path, _, _ in REQUIRED_DATASETS)

        selected_count = len(names) if max_phase_groups is None else min(max_phase_groups, len(names))
        if selected_count <= 0:
            raise ConversionError("DexCap selection must contain at least one episode")
        selected_names = names[:selected_count]
        selected_lengths = lengths[:selected_count]
        sample = _sample_payload(handle["data"][selected_names[0]], selected_lengths[0])
        if len(selected_names) > 1:
            sample.update(_sample_payload(handle["data"][selected_names[-1]], selected_lengths[-1]))

    spec = replace(spec, source_bytes=source_path.stat().st_size, sha256=digest)

    episodes = tuple(
        EpisodePlan(
            episode_uid=f"{spec.name}/demo_{index}",
            source_relative_path=f"{relative}::data/{name}",
            instruction=spec.instruction,
            num_frames=length,
            extra={
                "source_task": spec.name,
                "source_episode_id": name,
                "checkpoint_unit": f"demo_{index}",
                "phase_group_index": index,
                "phase_group_size": 1,
            },
        )
        for index, (name, length) in enumerate(zip(selected_names, selected_lengths, strict=True))
    )
    vectors = (
        VectorFeatureSpec("action", 46, dtype="float64"),
        VectorFeatureSpec("observation.glove_state", 63, dtype="float64"),
        VectorFeatureSpec("observation.eef_hand", 32, dtype="float64"),
        VectorFeatureSpec("observation.eef_position", 6, dtype="float64"),
        VectorFeatureSpec("observation.eef_quaternion", 8, dtype="float64"),
        VectorFeatureSpec("observation.pointcloud", 60_000, shape=POINT_SHAPE, dtype="float64"),
        VectorFeatureSpec("observation.source_state", 16, dtype="float64"),
        # LeRobot's feature contract requires a one-element vector at frame
        # ingestion; the writer canonicalises it back to a scalar Arrow value.
        VectorFeatureSpec("source.reward", 1, dtype="float64"),
        VectorFeatureSpec("source.done", 1, dtype="int64"),
        VectorFeatureSpec("source.label", 1, dtype="int64"),
    )
    plan = DatasetConversionPlan(
        dataset_uid=f"dexcap_{spec.name}",
        output_path=Path(collection_output) / spec.name,
        fps=DEFAULT_FPS,
        measured_fps=float(DEFAULT_FPS),
        robot_type=spec.robot_type,
        vector_features=vectors,
        camera_features=(CameraFeatureSpec("observation.images.agentview", 84, 84),),
        episodes=episodes,
        extra={
            "converter": "convert_dexcap_to_lerobot.py",
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_files": [relative],
            "source_root": str(raw_dataset_root),
            "source_relative_path": relative,
            "source_schema": [list(row) for row in reference_schema],
            "source_data_attributes": {"total": len(names)},
            "field_mapping": [
                {"source": "actions", "target": "action", "conversion": "identity", "lossy": False},
                {"source": "glove_states", "target": "observation.glove_state", "conversion": "identity", "lossy": False},
                {"source": "obs/robot0_eef_hand", "target": "observation.eef_hand", "conversion": "identity", "lossy": False},
                {"source": "obs/robot0_eef_pos", "target": "observation.eef_position", "conversion": "identity", "lossy": False},
                {"source": "obs/robot0_eef_quat", "target": "observation.eef_quaternion", "conversion": "identity", "lossy": False},
                {"source": "obs/pointcloud", "target": "observation.pointcloud", "conversion": "identity", "lossy": False},
                {"source": "states", "target": "observation.source_state", "conversion": "identity", "lossy": False},
                {"source": "rewards", "target": "source.reward", "conversion": "identity", "lossy": False},
                {"source": "dones", "target": "source.done", "conversion": "identity", "lossy": False},
                {"source": "obs/label", "target": "source.label", "conversion": "identity", "lossy": False},
                {"source": "obs/agentview_image", "target": "observation.images.agentview", "conversion": "video encode h264 CPU", "lossy": True},
            ],
            "action_semantics": "stored action[t] equals the concatenated EEF observation at t+1; no shift or repair applied",
            "timestamp_provenance": "fixed 10 Hz release convention; source files contain no timestamps",
            "pointcloud_semantics": "released 10,000-point XYZRGB array; identity mapping",
            "payload_scan_coverage": {
                "metadata_all_episodes": True,
                "sampled_payload_first_middle_last": True,
                "full_lowdim_scan": full_lowdim_scan,
                "full_pointcloud_scan": full_pointcloud_scan,
            },
            "source_sha256": digest,
        },
    )
    schema_fingerprint = canonical_fingerprint({"schema": reference_schema, "features": plan.feature_schema()})
    return DexCapPartitionInfo(
        spec=spec,
        source_path=source_path,
        source_relative_path=relative,
        plan=plan,
        all_episode_count=len(names),
        all_frame_count=sum(lengths),
        selected_logical_bytes=sum(selected_lengths) * (46 * 8 + 63 * 8 + 32 * 8 + 6 * 8 + 8 * 8 + 60_000 * 8 + 16 * 8 + 8 + 8 + 8 + 84 * 84 * 3),
        source_schema=reference_schema,
        schema_fingerprint=schema_fingerprint,
        episode_length_summary={"min": min(lengths), "max": max(lengths), "total": sum(lengths)},
        payload_scan=sample,
    )


def iter_frames(plan: DatasetConversionPlan, episode: EpisodePlan) -> Iterator[dict[str, Any]]:
    h5py = _require_h5py()
    source = Path(plan.extra["source_files"][0])
    # The plan stores the raw-root-relative path so worker processes can reopen
    # the source independently without serialising HDF5 handles.
    source_path = Path(plan.extra["source_root"]) / source if not source.is_absolute() else source
    demo_name = str(episode.extra["source_episode_id"])
    with h5py.File(source_path, "r") as handle:
        demo = handle["data"][demo_name]
        for index in range(episode.num_frames):
            action, glove, hand, pos, quat, pointcloud, state, reward, done, label = _frame_fields(demo, index)
            yield {
                "action": action,
                "observation.glove_state": glove,
                "observation.eef_hand": hand,
                "observation.eef_position": pos,
                "observation.eef_quaternion": quat,
                "observation.pointcloud": pointcloud,
                "observation.source_state": state,
                "source.reward": np.asarray([reward]),
                "source.done": np.asarray([done]),
                "source.label": np.asarray([label]),
                "observation.images.agentview": np.asarray(demo["obs/agentview_image"][index]),
                "task": episode.instruction,
            }
