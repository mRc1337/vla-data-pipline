"""Semantic reader for the official ARCap HDF5 release.

ARCap publishes five robomimic-shaped containers, but they do not contain
robomimic ``env_args`` or simulator XML metadata.  Their semantics instead
come from the official ARCap collection code, the official DexCap dataset
builder, the paper, and the released model checkpoints.  This reader keeps
that dataset-specific interpretation out of the generic LeRobot writer.

The source already stores 10,000-point XYZRGB point clouds and target actions.
No point sampling, normalization, padding, canonicalization, or 128-D mapping
is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from convert_core.checkpoint import canonical_fingerprint
from convert_core.episode_spec import (
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError
from convert_core.hdf5_common import (
    hdf5_leaf_schema,
    require_h5py,
    robomimic_demo_sort_key,
    validate_time_major_group,
)


SOURCE_DATASET = "Ericcsr/ARCap"
SOURCE_REVISION = "2cadcdbf4cec4ed9bc7023959583e82c8f4f0203"
OFFICIAL_CODE_REVISION = "8fe21e533d2af8549b8c880ff331445dc0a42dbf"
BUILDER_REPOSITORY = "j96w/DexCap"
BUILDER_REVISION = "5806229f00d73e70a284314ded788859fa90b89a"
DEFAULT_FPS = 10
POINT_COUNT = 10_000
POINT_COMPONENTS = ("x_m", "y_m", "z_m", "red", "green", "blue")
PANDA_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
FIN_RAY_COMMAND = ("fin_ray_command",)

# The official right-hand URDF names its movable joints ``0`` ... ``15``.
# These stable labels retain the same order while using the actual parent and
# child links, avoiding invented joint_N placeholders.
LEAP_JOINTS = (
    "mcp_joint_to_pip",
    "palm_lower_to_mcp_joint",
    "pip_to_dip",
    "dip_to_fingertip",
    "mcp_joint_2_to_pip_2",
    "palm_lower_to_mcp_joint_2",
    "pip_2_to_dip_2",
    "dip_2_to_fingertip_2",
    "mcp_joint_3_to_pip_3",
    "palm_lower_to_mcp_joint_3",
    "pip_3_to_dip_3",
    "dip_3_to_fingertip_3",
    "palm_lower_to_pip_4",
    "pip_4_to_thumb_pip",
    "thumb_pip_to_thumb_dip",
    "thumb_dip_to_thumb_fingertip",
)
EEF_NAMES = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")


@dataclass(frozen=True)
class ARCapPartitionSpec:
    name: str
    filename: str
    sha256: str
    source_bytes: int
    phase_group_size: int
    arm_names: tuple[str, ...]
    hand_names: tuple[str, ...]
    has_end_effector: bool
    instruction: str
    task_basis: str
    robot_type: str

    @property
    def joint_action_names(self) -> tuple[str, ...]:
        return (*self.arm_names, *self.hand_names)

    @property
    def end_effector_action_names(self) -> tuple[str, ...]:
        return (*EEF_NAMES, *self.hand_names)


CLUTTER_INSTRUCTION = (
    "Picking and placing a tennis ball with obstacles using a dexterous LEAP hand."
)

PARTITION_SPECS: tuple[ARCapPartitionSpec, ...] = (
    ARCapPartitionSpec(
        name="assemble",
        filename="assemble_arcap.hdf5",
        sha256="e0dcc483fd514521c67dd5a552e2f336638419b5592daec2aa528cbe0a917012",
        source_bytes=27_872_679_936,
        phase_group_size=3,
        arm_names=PANDA_JOINTS,
        hand_names=FIN_RAY_COMMAND,
        has_end_effector=True,
        instruction=(
            "Assemble a long-horizon, three-stage Lego tower with a Fin-ray "
            "parallel-jaw gripper."
        ),
        task_basis="ARCap paper section IV-D and official assemble_lego model name",
        robot_type="panda_fin_ray",
    ),
    ARCapPartitionSpec(
        name="clutter",
        filename="clutter_arcap.hdf5",
        sha256="5318e44edef385caf106b8133f365700bd0b5c6b92daead02e9cb4c8eac7b3ae",
        source_bytes=19_279_966_160,
        phase_group_size=3,
        arm_names=PANDA_JOINTS,
        hand_names=LEAP_JOINTS,
        has_end_effector=True,
        instruction=CLUTTER_INSTRUCTION,
        task_basis="verbatim ARCap paper section IV-B task description",
        robot_type="panda_leap_hand",
    ),
    ARCapPartitionSpec(
        name="clutter_users",
        filename="clutter_arcap_users.hdf5",
        sha256="dcea52fcacf21c3e54a612f78bd2e295d7de71d8b8d8391f0b2484fdfabb8685",
        source_bytes=21_868_455_008,
        phase_group_size=3,
        arm_names=PANDA_JOINTS,
        hand_names=LEAP_JOINTS,
        has_end_effector=True,
        instruction=CLUTTER_INSTRUCTION,
        task_basis="verbatim ARCap paper section IV-B task description; user-study cohort",
        robot_type="panda_leap_hand",
    ),
    ARCapPartitionSpec(
        name="open_bottle",
        filename="open_bottle_arcap.hdf5",
        sha256="1c0f7b445a8e6d4c5f2a84d8ed9fa60dea3a239f7c5ebcccafd234e7631715de",
        source_bytes=25_607_833_480,
        phase_group_size=2,
        arm_names=tuple(
            [f"left_{name}" for name in PANDA_JOINTS]
            + [f"right_{name}" for name in PANDA_JOINTS]
        ),
        hand_names=("left_fin_ray_command", *tuple(f"right_{name}" for name in LEAP_JOINTS)),
        has_end_effector=False,
        instruction="Open a bottle using both robot arms.",
        task_basis=(
            "official open_bottle dataset/model identifiers and website bimanual task"
        ),
        robot_type="bimanual_panda_left_fin_ray_right_leap_hand",
    ),
    ARCapPartitionSpec(
        name="wild",
        filename="wild_arcap.hdf5",
        sha256="52d681e890cb24145d23770e2043d7847b3d825bb0f282b6794e33bf05e8e11f",
        source_bytes=16_824_032_800,
        phase_group_size=3,
        arm_names=PANDA_JOINTS,
        hand_names=LEAP_JOINTS,
        has_end_effector=True,
        instruction=CLUTTER_INSTRUCTION,
        task_basis=(
            "official model checkpoint config path wild_tennis_3gap_test.hdf5 plus "
            "verbatim ARCap paper tennis task description"
        ),
        robot_type="panda_leap_hand",
    ),
)
PARTITIONS_BY_NAME = {spec.name: spec for spec in PARTITION_SPECS}
PARTITIONS_BY_FILE = {spec.filename: spec for spec in PARTITION_SPECS}


@dataclass(frozen=True)
class ARCapPartitionInfo:
    spec: ARCapPartitionSpec
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


def partition_spec(value: str | Path) -> ARCapPartitionSpec:
    key = Path(value).name
    if key in PARTITIONS_BY_FILE:
        return PARTITIONS_BY_FILE[key]
    if str(value) in PARTITIONS_BY_NAME:
        return PARTITIONS_BY_NAME[str(value)]
    raise ConversionError(
        f"unknown ARCap partition {value!r}; expected {sorted(PARTITIONS_BY_NAME)}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_schema(spec: ARCapPartitionSpec) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    rows: list[tuple[str, tuple[int, ...], str]] = [
        ("actions", (len(spec.joint_action_names),), "float64"),
        ("dones", (), "int64"),
        ("obs/pointcloud", (POINT_COUNT, len(POINT_COMPONENTS)), "float64"),
        ("obs/robot0_arm_joints", (len(spec.arm_names),), "float64"),
        ("obs/robot0_hand_joints", (len(spec.hand_names),), "float64"),
        ("rewards", (), "float64"),
        ("states", (), "float64"),
    ]
    if spec.has_end_effector:
        rows.extend(
            [
                ("actions2", (len(spec.end_effector_action_names),), "float64"),
                ("obs/robot0_eef_pos", (3,), "float64"),
                ("obs/robot0_eef_quat", (4,), "float64"),
            ]
        )
    return tuple(sorted(rows))


def _feature_mapping(spec: ARCapPartitionSpec) -> tuple[tuple[str, str, tuple[str, ...] | None], ...]:
    rows: list[tuple[str, str, tuple[str, ...] | None]] = [
        ("actions", "action", spec.joint_action_names),
        ("dones", "source.done", ("done",)),
        ("obs/pointcloud", "observation.pointcloud", ("point", "component")),
        ("obs/robot0_arm_joints", "observation.arm_joint_position", spec.arm_names),
        ("obs/robot0_hand_joints", "observation.hand_joint_position", spec.hand_names),
        ("rewards", "source.reward", ("reward",)),
        ("states", "source.state", ("source_state",)),
    ]
    if spec.has_end_effector:
        rows.extend(
            [
                ("actions2", "action.end_effector", spec.end_effector_action_names),
                ("obs/robot0_eef_pos", "observation.end_effector_position", EEF_NAMES[:3]),
                (
                    "obs/robot0_eef_quat",
                    "observation.end_effector_quaternion_xyzw",
                    EEF_NAMES[3:],
                ),
            ]
        )
    return tuple(rows)


def _sample_pointclouds(data: Any, demo_names: list[str]) -> dict[str, Any]:
    selected_episodes = sorted(
        {0, len(demo_names) // 2, len(demo_names) - 1}
    )
    frames = 0
    xyz_min = np.full(3, np.inf)
    xyz_max = np.full(3, -np.inf)
    rgb_min = np.full(3, np.inf)
    rgb_max = np.full(3, -np.inf)
    for episode_index in selected_episodes:
        dataset = data[demo_names[episode_index]]["obs/pointcloud"]
        for frame_index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            value = np.asarray(dataset[frame_index])
            if value.shape != (POINT_COUNT, 6) or value.dtype != np.float64:
                raise ConversionError(
                    f"point-cloud sample has changed shape/dtype: {value.shape} {value.dtype}"
                )
            if not np.isfinite(value).all():
                raise ConversionError("sampled ARCap point cloud contains non-finite values")
            xyz_min = np.minimum(xyz_min, value[:, :3].min(axis=0))
            xyz_max = np.maximum(xyz_max, value[:, :3].max(axis=0))
            rgb_min = np.minimum(rgb_min, value[:, 3:].min(axis=0))
            rgb_max = np.maximum(rgb_max, value[:, 3:].max(axis=0))
            frames += 1
    if (rgb_min < 0).any() or (rgb_max > 1).any():
        raise ConversionError(
            f"sampled ARCap RGB values are outside [0,1]: {rgb_min} .. {rgb_max}"
        )
    return {
        "sampled_pointcloud_frames": frames,
        "xyz_min_m": xyz_min.tolist(),
        "xyz_max_m": xyz_max.tolist(),
        "rgb_min": rgb_min.tolist(),
        "rgb_max": rgb_max.tolist(),
    }


def _scan_pointcloud_payload(data: Any, demo_names: list[str]) -> int:
    checked = 0
    for demo_name in demo_names:
        dataset = data[demo_name]["obs/pointcloud"]
        block = max(1, min(16, len(dataset)))
        for start in range(0, len(dataset), block):
            value = np.asarray(dataset[start : start + block])
            if not np.isfinite(value).all():
                raise ConversionError(
                    f"data/{demo_name}/obs/pointcloud contains non-finite values"
                )
            if (value[..., 3:] < 0).any() or (value[..., 3:] > 1).any():
                raise ConversionError(
                    f"data/{demo_name}/obs/pointcloud RGB is outside [0,1]"
                )
            checked += len(value)
    return checked


def _scan_lowdim_payload(
    data: Any,
    demo_names: list[str],
    spec: ARCapPartitionSpec,
) -> dict[str, Any]:
    terminal_joint_matches = 0
    terminal_eef_matches = 0
    group_done_counts: list[int] = []
    for group_start in range(0, len(demo_names), spec.phase_group_size):
        group = demo_names[group_start : group_start + spec.phase_group_size]
        done_count = 0
        for demo_name in group:
            demo = data[demo_name]
            arm = np.asarray(demo["obs/robot0_arm_joints"][:])
            hand = np.asarray(demo["obs/robot0_hand_joints"][:])
            actions = np.asarray(demo["actions"][:])
            joint = np.concatenate((arm, hand), axis=1)
            if not np.array_equal(actions[:-1], joint[1:]):
                raise ConversionError(
                    f"data/{demo_name}/actions no longer equals the following joint state"
                )
            terminal_joint_matches += int(np.array_equal(actions[-1], joint[-1]))
            if spec.has_end_effector:
                eef = np.concatenate(
                    (
                        np.asarray(demo["obs/robot0_eef_pos"][:]),
                        np.asarray(demo["obs/robot0_eef_quat"][:]),
                        hand,
                    ),
                    axis=1,
                )
                actions2 = np.asarray(demo["actions2"][:])
                if not np.array_equal(actions2[:-1], eef[1:]):
                    raise ConversionError(
                        f"data/{demo_name}/actions2 no longer equals the following EEF state"
                    )
                terminal_eef_matches += int(np.array_equal(actions2[-1], eef[-1]))
            for source_key in (
                "actions",
                "obs/robot0_arm_joints",
                "obs/robot0_hand_joints",
                "rewards",
                "states",
                *(("actions2", "obs/robot0_eef_pos", "obs/robot0_eef_quat")
                  if spec.has_end_effector else ()),
            ):
                if not np.isfinite(np.asarray(demo[source_key][:])).all():
                    raise ConversionError(f"data/{demo_name}/{source_key} contains non-finite values")
            if np.count_nonzero(np.asarray(demo["rewards"][:])):
                raise ConversionError(f"data/{demo_name}/rewards is no longer the released zero field")
            if np.count_nonzero(np.asarray(demo["states"][:])):
                raise ConversionError(f"data/{demo_name}/states is no longer the released zero field")
            dones = np.asarray(demo["dones"][:])
            if not np.isin(dones, (0, 1)).all():
                raise ConversionError(f"data/{demo_name}/dones is not binary")
            nonzero = np.flatnonzero(dones)
            if len(nonzero) > 1 or (len(nonzero) == 1 and nonzero[0] != len(dones) - 1):
                raise ConversionError(f"data/{demo_name}/dones changed terminal placement")
            done_count += len(nonzero)
        if done_count != 1:
            raise ConversionError(
                f"phase group beginning {group[0]} has {done_count} done flags; expected one"
            )
        group_done_counts.append(done_count)
    return {
        "full_lowdim_frames": sum(int(data[name].attrs["num_samples"]) for name in demo_names),
        "phase_groups": len(group_done_counts),
        "one_done_per_phase_group": True,
        "terminal_joint_action_matches": terminal_joint_matches,
        "terminal_eef_action_matches": terminal_eef_matches if spec.has_end_effector else None,
        "terminal_actions_preserved_without_repair": True,
    }


def inspect_partition(
    source_path: Path,
    *,
    raw_dataset_root: Path,
    collection_output: Path,
    max_phase_groups: int | None = None,
    verify_sha256: bool = False,
    full_lowdim_scan: bool = False,
    full_pointcloud_scan: bool = False,
) -> ARCapPartitionInfo:
    """Inspect one official container without writing output."""

    h5py = require_h5py()
    source_path = source_path.absolute()
    raw_dataset_root = raw_dataset_root.absolute()
    spec = partition_spec(source_path.name)
    if max_phase_groups is not None and max_phase_groups <= 0:
        raise ConversionError("max_phase_groups must be positive")
    try:
        relative = source_path.relative_to(raw_dataset_root).as_posix()
    except ValueError as exc:
        raise ConversionError(f"source file is outside ARCap root: {source_path}") from exc
    stat = source_path.stat()
    if stat.st_size != spec.source_bytes:
        raise ConversionError(
            f"{source_path}: size {stat.st_size} differs from official {spec.source_bytes}"
        )
    actual_sha = sha256_file(source_path) if verify_sha256 else None
    if actual_sha is not None and actual_sha != spec.sha256:
        raise ConversionError(
            f"{source_path}: SHA-256 {actual_sha} differs from official {spec.sha256}"
        )

    with h5py.File(source_path, "r") as h5_file:
        if set(h5_file.keys()) != {"data"} or h5_file.attrs:
            raise ConversionError(
                f"{source_path}: expected the attribute-free official data root"
            )
        data = h5_file["data"]
        expected_data_attrs = {"mean_init_arm", "mean_init_hand", "total"}
        if spec.has_end_effector:
            expected_data_attrs.update({"mean_init_pos", "mean_init_quat"})
        if set(data.attrs) != expected_data_attrs:
            raise ConversionError(
                f"{source_path}: data-group attributes differ from the official release: "
                f"expected {sorted(expected_data_attrs)}, got {sorted(data.attrs)}"
            )
        expected_attr_shapes = {
            "mean_init_arm": (len(spec.arm_names),),
            "mean_init_hand": (len(spec.hand_names),),
            "total": (),
        }
        if spec.has_end_effector:
            expected_attr_shapes.update({"mean_init_pos": (3,), "mean_init_quat": (4,)})
        data_attributes: dict[str, Any] = {}
        for key, expected_shape in expected_attr_shapes.items():
            value = np.asarray(data.attrs[key])
            if value.shape != expected_shape or not np.isfinite(value).all():
                raise ConversionError(
                    f"{source_path}: data attribute {key} has invalid shape/value: "
                    f"{value.shape}"
                )
            data_attributes[key] = value.item() if not expected_shape else value.tolist()
        demo_names = sorted(data.keys(), key=robomimic_demo_sort_key)
        if not demo_names:
            raise ConversionError(f"{source_path}: contains no demo_N episodes")
        ids = [robomimic_demo_sort_key(name) for name in demo_names]
        if ids != list(range(len(ids))):
            raise ConversionError(f"{source_path}: demo IDs are not continuous from zero")
        if len(demo_names) % spec.phase_group_size:
            raise ConversionError(
                f"{source_path}: {len(demo_names)} episodes is not divisible by phase group "
                f"size {spec.phase_group_size}"
            )
        reference_schema = hdf5_leaf_schema(data[demo_names[0]])
        expected_schema = _expected_schema(spec)
        if reference_schema != expected_schema:
            raise ConversionError(
                f"{source_path}: schema differs from official release\n"
                f"expected={expected_schema}\nactual={reference_schema}"
            )
        lengths: dict[str, int] = {}
        all_frames = 0
        for demo_name in demo_names:
            demo = data[demo_name]
            if set(demo.attrs) != {"num_samples"}:
                raise ConversionError(
                    f"{source_path}:data/{demo_name}: expected only num_samples attribute"
                )
            num_samples = int(demo.attrs["num_samples"])
            if num_samples <= 0:
                raise ConversionError(f"{source_path}:data/{demo_name}: empty episode")
            validate_time_major_group(
                demo,
                reference_schema,
                expected_frames=num_samples,
                description=f"{source_path}:data/{demo_name}",
            )
            lengths[demo_name] = num_samples
            all_frames += num_samples
        if int(data.attrs["total"]) != all_frames:
            raise ConversionError(
                f"{source_path}: data.total={int(data.attrs['total'])} but episodes contain "
                f"{all_frames} frames"
            )

        selected_group_count = (
            min(max_phase_groups, len(demo_names) // spec.phase_group_size)
            if max_phase_groups is not None
            else len(demo_names) // spec.phase_group_size
        )
        selected_names = demo_names[: selected_group_count * spec.phase_group_size]
        payload_scan = _sample_pointclouds(data, demo_names)
        if full_lowdim_scan:
            payload_scan.update(_scan_lowdim_payload(data, demo_names, spec))
        if full_pointcloud_scan:
            payload_scan["full_pointcloud_frames"] = _scan_pointcloud_payload(
                data, demo_names
            )
        selected_logical_bytes = 0
        for demo_name in selected_names:
            for source_key, tail_shape, dtype in reference_schema:
                selected_logical_bytes += (
                    lengths[demo_name]
                    * (int(np.prod(tail_shape)) if tail_shape else 1)
                    * np.dtype(dtype).itemsize
                )

    mapping_by_source = {row[0]: (row[1], row[2]) for row in _feature_mapping(spec)}
    vectors: list[VectorFeatureSpec] = []
    field_mapping: list[dict[str, Any]] = []
    for source_key, tail_shape, dtype in reference_schema:
        feature_key, names = mapping_by_source[source_key]
        output_shape = tail_shape or (1,)
        vectors.append(
            VectorFeatureSpec(
                feature_key=feature_key,
                dim=int(np.prod(output_shape)),
                names=names,
                dtype=dtype,
                shape=output_shape,
            )
        )
        field_mapping.append(
            {
                "source_key": f"data/<demo>/{source_key}",
                "source_shape": ["T", *tail_shape],
                "source_dtype": dtype,
                "lerobot_key": feature_key,
                "transform": "scalar wrapped as shape [1]" if not tail_shape else "identity",
                "lossy": False,
            }
        )
    episodes = tuple(
        EpisodePlan(
            episode_uid=f"{relative}::data/{demo_name}",
            source_relative_path=f"{relative}::data/{demo_name}",
            instruction=spec.instruction,
            num_frames=lengths[demo_name],
            extra={
                "source_path": source_path,
                "demo_name": demo_name,
                "source_episode_id": demo_name,
                "source_partition": spec.name,
                "source_task": spec.name,
                "phase_group_index": episode_index // spec.phase_group_size,
                "phase_offset": episode_index % spec.phase_group_size,
                "phase_group_size": spec.phase_group_size,
                "checkpoint_unit": (
                    f"{spec.name}/phase-group-"
                    f"{episode_index // spec.phase_group_size:05d}"
                ),
            },
        )
        for episode_index, demo_name in enumerate(selected_names)
    )
    schema_payload = [
        {"key": key, "tail_shape": list(shape), "dtype": dtype}
        for key, shape, dtype in reference_schema
    ]
    schema_fingerprint = canonical_fingerprint({"leaves": schema_payload})
    plan = DatasetConversionPlan(
        dataset_uid=f"arcap_{spec.name}",
        output_path=collection_output / spec.name,
        fps=DEFAULT_FPS,
        measured_fps=float(DEFAULT_FPS),
        robot_type=spec.robot_type,
        vector_features=tuple(vectors),
        camera_features=(),
        episodes=episodes,
        extra={
            "converter": "convert_arcap_to_lerobot.py",
            "source_root": str(raw_dataset_root),
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_relative_path": relative,
            "source_files": [
                {
                    "relative_path": relative,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "official_lfs_sha256": spec.sha256,
                    "verified_sha256": actual_sha,
                }
            ],
            "source_splits": [],
            "field_mapping": field_mapping,
            "task_provenance": {
                "source_task_id": spec.name,
                "instruction": spec.instruction,
                "basis": spec.task_basis,
                "paper": "arXiv:2410.08464",
                "official_model_revision": "13ce736462ce8ca70f5b8ac2696168ec8a31dbfb",
            },
            "timestamp_provenance": {
                "source_timestamps_present": False,
                "fps": DEFAULT_FPS,
                "timestamp_expression": "frame_index / 10",
                "basis": (
                    "official 30 Hz collection default followed by builder gap=3 phase "
                    "construction; released HDF5 discarded original jitter"
                ),
                "synthetic": True,
                "resampled_by_staging": False,
            },
            "partition_rules": {
                "rule": "one official fixed-schema HDF5 container per LeRobot partition",
                "partition": spec.name,
                "source_schema_fingerprint": schema_fingerprint,
                "phase_group_size": spec.phase_group_size,
                "no_padding_cast_reorder_or_normalization": True,
            },
            "action_semantics": {
                "action": "following arm+hand joint target except released terminal value",
                "action_end_effector": (
                    "following XYZ+quaternion-XYZW+hand target except released terminal value"
                    if spec.has_end_effector
                    else None
                ),
                "terminal_values_preserved": True,
                "fin_ray_command": {"open": -1, "close": 1},
                "open_bottle_order": (
                    "left Panda 7, right Panda 7; left Fin-ray command, right LEAP 16"
                    if spec.name == "open_bottle"
                    else None
                ),
            },
            "quaternion_convention": "xyzw",
            "pointcloud_semantics": {
                "shape": [POINT_COUNT, 6],
                "component_names": list(POINT_COMPONENTS),
                "xyz_unit": "metre",
                "rgb_range": [0.0, 1.0],
                "source_sampling": (
                    "official builder already randomly samples or repeats to 10000 points"
                ),
                "staging_transform": "identity",
            },
            "unsupported_source_components": {
                "absent": (
                    ["actions2", "obs/robot0_eef_pos", "obs/robot0_eef_quat"]
                    if not spec.has_end_effector
                    else []
                ),
                "fabricated": [],
            },
            "payload_scan_coverage": {
                "metadata_schema_reference_scan": "all episodes",
                "sampled_pointcloud_scan": payload_scan["sampled_pointcloud_frames"],
                "full_lowdim_scan": full_lowdim_scan,
                "full_pointcloud_scan": full_pointcloud_scan,
                **payload_scan,
            },
            "source_builder": {
                "repository": BUILDER_REPOSITORY,
                "revision": BUILDER_REVISION,
                "official_collection_code_revision": OFFICIAL_CODE_REVISION,
            },
            "source_data_attributes": data_attributes,
            "selected_episode_count": len(selected_names),
            "all_episode_count": len(demo_names),
            "all_frame_count": all_frames,
        },
    )
    values = list(lengths.values())
    return ARCapPartitionInfo(
        spec=spec,
        source_path=source_path,
        source_relative_path=relative,
        plan=plan,
        all_episode_count=len(demo_names),
        all_frame_count=all_frames,
        selected_logical_bytes=selected_logical_bytes,
        source_schema=reference_schema,
        schema_fingerprint=schema_fingerprint,
        episode_length_summary={
            "min": min(values),
            "p50": int(np.percentile(values, 50)),
            "p95": int(np.percentile(values, 95)),
            "max": max(values),
        },
        payload_scan=payload_scan,
    )


def iter_frames(
    plan: DatasetConversionPlan,
    episode: EpisodePlan,
) -> Iterator[dict[str, Any]]:
    """Materialize only one bounded source episode and preserve values exactly."""

    h5py = require_h5py()
    source_path = Path(episode.extra["source_path"])
    demo_name = str(episode.extra["demo_name"])
    mapping = plan.extra["field_mapping"]
    with h5py.File(source_path, "r") as h5_file:
        demo = h5_file[f"data/{demo_name}"]
        arrays = {
            item["lerobot_key"]: np.asarray(
                demo[item["source_key"].removeprefix("data/<demo>/")][:]
            )
            for item in mapping
        }
        for frame_index in range(episode.num_frames):
            frame: dict[str, Any] = {"task": episode.instruction}
            for item in mapping:
                value = arrays[item["lerobot_key"]][frame_index]
                if not item["source_shape"][1:]:
                    value = np.asarray(value).reshape(1)
                frame[item["lerobot_key"]] = value
            yield frame


class ARCapHdf5Reader:
    """Registry adapter for inspecting one configured ARCap partition."""

    def build_plan(
        self,
        config: Any,
        raw_root: Path,
        staging_root: Path,
    ) -> DatasetConversionPlan:
        source_file = getattr(config, "source_file", None)
        if not source_file:
            raise ConversionError(
                "format=arcap_hdf5 requires source_file for the generic single-partition CLI"
            )
        source_directory = getattr(config, "source_directory", None) or config.dataset_uid
        dataset_root = raw_root / source_directory
        return inspect_partition(
            dataset_root / source_file,
            raw_dataset_root=dataset_root,
            collection_output=staging_root / "lerobot_v3_0" / config.dataset_uid,
        ).plan

    def iter_frames(
        self,
        plan: DatasetConversionPlan,
        episode: EpisodePlan,
    ) -> Iterator[dict[str, Any]]:
        yield from iter_frames(plan, episode)
