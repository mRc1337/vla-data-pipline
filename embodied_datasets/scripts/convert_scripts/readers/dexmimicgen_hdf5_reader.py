"""Semantic reader for the official DexMimicGen robomimic HDF5 release.

Every source container is a fixed-schema partition, but the nine containers
are mutually heterogeneous.  This reader preserves every time-major leaf,
derives simulator-state and robot joint names from the exact per-episode MJCF,
and records model XML identities for lossless collection-level sidecars.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator
import xml.etree.ElementTree as ET

import numpy as np

from convert_core.checkpoint import canonical_fingerprint
from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError
from convert_core.hdf5_common import (
    decode_hdf5_text,
    hdf5_leaf_schema,
    require_h5py,
    robomimic_demo_sort_key,
    validate_time_major_group,
)


SOURCE_REPOSITORY = "MimicGen/dexmimicgen_datasets"
SOURCE_REVISION = "181967e10c6277a653e7c9761f3978312af3646a"
OFFICIAL_CODE_REVISION = "940e8a1b3ad70eb1925ada6b364b197de6bb2af9"
DEFAULT_FPS = 20


TASK_INSTRUCTIONS: dict[str, str] = {
    "TwoArmBoxCleanup": "Place the lid aligned on the box.",
    "TwoArmCanSortBlue": (
        "Place the red and blue cans into their matching red and blue bins."
    ),
    "TwoArmCoffee": "Insert the coffee pod into its holder and close the lid.",
    "TwoArmDrawerCleanup": (
        "Place the cleanup object in the drawer and close the drawer."
    ),
    "TwoArmLiftTray": (
        "Lift the tray with both contained objects above the table."
    ),
    "TwoArmPouring": (
        "Pour the ball into the bowl and place the bowl upright on the pad."
    ),
    "TwoArmThreading": "Insert the needle through the tripod ring.",
    "TwoArmThreePieceAssembly": "Assemble both pieces onto the base.",
    "TwoArmTransport": (
        "Move the payload into the target bin and the trash into the trash bin."
    ),
}


PANDA_ACTION_KEYS = (
    "right_rel_pos",
    "right_rel_rot_axis_angle",
    "right_gripper",
    "left_rel_pos",
    "left_rel_rot_axis_angle",
    "left_gripper",
)
HUMANOID_STORED_ACTION_KEYS = (
    "right_abs_pos",
    "right_abs_rot_axis_angle",
    "left_abs_pos",
    "left_abs_rot_axis_angle",
    "right_gripper",
    "left_gripper",
)
HUMANOID_TRAINING_ACTION_KEYS = (
    "right_abs_pos",
    "right_abs_rot_6d",
    "left_abs_pos",
    "left_abs_rot_6d",
    "right_gripper",
    "left_gripper",
)


@dataclass(frozen=True)
class ModelReference:
    demo_name: str
    sha256: str
    uncompressed_bytes: int


@dataclass(frozen=True)
class DexMimicGenPartitionInfo:
    source_path: Path
    source_relative_path: str
    partition_name: str
    env_args: dict[str, Any]
    source_schema: tuple[tuple[str, tuple[int, ...], str], ...]
    plan: DatasetConversionPlan
    all_episode_count: int
    all_frame_count: int
    selected_numeric_logical_bytes: int
    selected_image_logical_bytes: int
    selected_camera_frames: int
    model_references: tuple[ModelReference, ...]
    unique_model_count: int
    schema_fingerprint: str
    episode_length_summary: dict[str, int]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_mjcf(model_file: str, *, description: str) -> ET.Element:
    try:
        return ET.fromstring(model_file)
    except ET.ParseError as exc:
        raise ConversionError(f"{description}: model_file is invalid MJCF/XML: {exc}") from exc


def _joint_elements(root: ET.Element, *, description: str) -> list[ET.Element]:
    joints = root.findall(".//joint")
    if not joints:
        raise ConversionError(f"{description}: model_file contains no joints")
    for joint in joints:
        if not joint.attrib.get("name"):
            raise ConversionError(f"{description}: model_file contains an unnamed joint")
    return joints


def _qpos_names(name: str, joint_type: str) -> tuple[str, ...]:
    if joint_type in {"hinge", "slide"}:
        return (f"qpos({name})",)
    if joint_type == "ball":
        return tuple(f"qpos({name}).{axis}" for axis in ("qw", "qx", "qy", "qz"))
    if joint_type == "free":
        return tuple(
            f"qpos({name}).{axis}"
            for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")
        )
    raise ConversionError(f"unsupported MJCF joint type {joint_type!r} for {name}")


def _qvel_names(name: str, joint_type: str) -> tuple[str, ...]:
    if joint_type in {"hinge", "slide"}:
        return (f"qvel({name})",)
    if joint_type == "ball":
        return tuple(f"qvel({name}).{axis}" for axis in ("wx", "wy", "wz"))
    if joint_type == "free":
        return tuple(
            f"qvel({name}).{axis}"
            for axis in ("vx", "vy", "vz", "wx", "wy", "wz")
        )
    raise ConversionError(f"unsupported MJCF joint type {joint_type!r} for {name}")


def simulator_state_names(
    model_file: str,
    *,
    expected_width: int,
    description: str,
) -> tuple[str, ...]:
    """Name robosuite's flattened ``time + qpos + qvel`` simulator state."""

    root = _parse_mjcf(model_file, description=description)
    joints = _joint_elements(root, description=description)
    qpos: list[str] = []
    qvel: list[str] = []
    for joint in joints:
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "hinge")
        qpos.extend(_qpos_names(name, joint_type))
        qvel.extend(_qvel_names(name, joint_type))
    names = ("time", *qpos, *qvel)
    if len(names) != expected_width:
        raise ConversionError(
            f"{description}: states width is {expected_width}, but model_file describes "
            f"time + {len(qpos)} qpos + {len(qvel)} qvel = {len(names)} values"
        )
    return tuple(names)


def _joint_names(model_file: str, *, description: str) -> tuple[str, ...]:
    root = _parse_mjcf(model_file, description=description)
    return tuple(
        joint.attrib["name"]
        for joint in _joint_elements(root, description=description)
    )


def _component_names(prefix: str, width: int) -> tuple[str, ...]:
    if width == 1:
        return (prefix,)
    return tuple(f"{prefix}.source_component_{index}" for index in range(width))


def _action_field_names(key: str, width: int) -> tuple[str, ...] | None:
    if key.endswith("_pos") and width == 3:
        return tuple(f"{key}.{axis}" for axis in ("x", "y", "z"))
    if key.endswith("_rot_axis_angle") and width == 3:
        return tuple(f"{key}.{axis}" for axis in ("x", "y", "z"))
    if key.endswith("_gripper"):
        return _component_names(key, width)
    # The official config identifies rot_6d but does not publish labels for
    # its six stored components. Keep names absent instead of guessing row/
    # column semantics.
    return None


def _stored_action_layout(group: Any, *, description: str) -> tuple[str, ...]:
    action_dict = group.get("action_dict")
    if action_dict is None:
        raise ConversionError(f"{description}: missing action_dict")
    if "right_rel_pos" in action_dict:
        keys = PANDA_ACTION_KEYS
    elif "right_abs_pos" in action_dict:
        keys = HUMANOID_STORED_ACTION_KEYS
    else:
        raise ConversionError(f"{description}: action_dict has no supported action layout")
    missing = [key for key in keys if key not in action_dict]
    if missing:
        raise ConversionError(f"{description}: action_dict is missing {missing}")
    return keys


def _flattened_action_names(group: Any, *, description: str) -> tuple[str, ...]:
    keys = _stored_action_layout(group, description=description)
    names: list[str] = []
    arrays = []
    sample_count = min(64, int(group.attrs["num_samples"]))
    for key in keys:
        dataset = group[f"action_dict/{key}"]
        width = int(np.prod(dataset.shape[1:]))
        field_names = _action_field_names(key, width)
        if field_names is None:
            field_names = _component_names(key, width)
        names.extend(field_names)
        arrays.append(np.asarray(dataset[:sample_count]).reshape(sample_count, -1))
    actions = group["actions"]
    if actions.ndim != 2 or actions.shape[1] != len(names):
        raise ConversionError(
            f"{description}: actions width {actions.shape} does not match official stored "
            f"action_dict order width {len(names)}"
        )
    reconstructed = np.concatenate(arrays, axis=1)
    sampled_actions = np.asarray(actions[:sample_count])
    maximum_error = float(
        np.max(np.abs(sampled_actions - reconstructed), initial=0.0)
    )
    if maximum_error > 1e-6:
        raise ConversionError(
            f"{description}: actions disagree with action_dict order (max error {maximum_error})"
        )
    return tuple(names)


def _observation_names(
    source_key: str,
    shape: tuple[int, ...],
    *,
    joint_names: tuple[str, ...],
) -> tuple[str, ...] | None:
    if len(shape) != 1:
        return None
    width = shape[0]
    name = source_key.removeprefix("obs/")
    match = re.fullmatch(r"robot(\d+)_joint_(pos|vel|pos_sin|pos_cos)", name)
    if match:
        robot, kind = match.groups()
        selected = tuple(
            joint
            for joint in joint_names
            if joint.startswith(f"robot{robot}_")
        )
        if len(selected) != width:
            raise ConversionError(
                f"{source_key}: XML exposes {len(selected)} robot joints, expected {width}"
            )
        if kind == "pos":
            return selected
        if kind == "vel":
            return tuple(f"velocity({joint})" for joint in selected)
        function = "sin" if kind == "pos_sin" else "cos"
        return tuple(f"{function}({joint})" for joint in selected)

    match = re.fullmatch(
        r"robot(\d+)_(?:(right|left)_)?gripper_q(pos|vel)", name
    )
    if match:
        robot, side, kind = match.groups()
        prefix = f"gripper{robot}_{side}_" if side else f"gripper{robot}_"
        selected = tuple(joint for joint in joint_names if joint.startswith(prefix))
        if len(selected) != width:
            raise ConversionError(
                f"{source_key}: XML exposes {len(selected)} gripper joints, expected {width}"
            )
        if kind == "pos":
            return selected
        return tuple(f"velocity({joint})" for joint in selected)

    if name.endswith("_pos") and width == 3:
        return ("x", "y", "z")
    if "_pos_rel_" in name and width == 3:
        return ("x", "y", "z")
    if name.endswith("_quat") or "_quat_rel_" in name or name.endswith("_quat_site"):
        return ("qx", "qy", "qz", "qw") if width == 4 else None
    return None


def _feature_key(source_key: str) -> str:
    if source_key == "actions":
        return "action"
    if source_key == "states":
        return "observation.sim_state"
    if source_key.startswith("action_dict/"):
        return "source.action_dict." + source_key.removeprefix("action_dict/").replace("/", ".")
    if source_key.startswith("datagen_info/"):
        return "source.datagen_info." + source_key.removeprefix("datagen_info/").replace("/", ".")
    if source_key.startswith("obs/"):
        name = source_key.removeprefix("obs/")
        if name.endswith("_image"):
            return f"observation.images.{name.removesuffix('_image')}"
        return f"observation.{name}"
    raise ConversionError(f"no lossless DexMimicGen mapping for source field {source_key!r}")


def _names_for_leaf(
    source_key: str,
    shape: tuple[int, ...],
    *,
    action_names: tuple[str, ...],
    state_names: tuple[str, ...],
    joint_names: tuple[str, ...],
) -> tuple[str, ...] | None:
    if source_key == "actions":
        return action_names
    if source_key == "states":
        return state_names
    if source_key.startswith("action_dict/") and len(shape) == 1:
        return _action_field_names(source_key.rsplit("/", 1)[1], shape[0])
    if source_key.startswith("obs/"):
        return _observation_names(source_key, shape, joint_names=joint_names)
    if source_key.startswith("datagen_info/subtask_term_signals/") and shape == (1,):
        return (source_key.rsplit("/", 1)[1],)
    return None


def _partition_name(source_path: Path) -> str:
    stem = source_path.stem
    if not stem.startswith("two_arm_"):
        raise ConversionError(
            f"unexpected DexMimicGen container name {source_path.name!r}; expected two_arm_*.hdf5"
        )
    return stem.removeprefix("two_arm_")


def inspect_partition(
    source_path: Path,
    *,
    raw_dataset_root: Path,
    collection_output: Path,
    max_episodes: int | None = None,
) -> DexMimicGenPartitionInfo:
    """Perform a full metadata/schema/reference scan for one container."""

    h5py = require_h5py()
    source_path = source_path.absolute()
    raw_dataset_root = raw_dataset_root.absolute()
    try:
        relative = source_path.relative_to(raw_dataset_root).as_posix()
    except ValueError as exc:
        raise ConversionError(f"source file is outside DexMimicGen root: {source_path}") from exc
    partition = _partition_name(source_path)
    if max_episodes is not None and max_episodes <= 0:
        raise ConversionError("max_episodes must be positive")

    with h5py.File(source_path, "r") as h5_file:
        if set(h5_file.keys()) != {"data"}:
            raise ConversionError(
                f"{source_path}: expected only the official data root, got {sorted(h5_file.keys())}"
            )
        data = h5_file["data"]
        try:
            env_text = decode_hdf5_text(
                data.attrs["env_args"], description=f"{source_path}:data.attrs.env_args"
            )
            env_args = json.loads(env_text)
        except KeyError as exc:
            raise ConversionError(f"{source_path}: missing data.attrs.env_args") from exc
        except json.JSONDecodeError as exc:
            raise ConversionError(f"{source_path}: invalid env_args JSON: {exc}") from exc
        if not isinstance(env_args, dict):
            raise ConversionError(f"{source_path}: env_args must be an object")
        env_name = env_args.get("env_name")
        if env_name not in TASK_INSTRUCTIONS:
            raise ConversionError(
                f"{source_path}: no reviewed official success-predicate task for {env_name!r}"
            )
        env_kwargs = env_args.get("env_kwargs")
        if not isinstance(env_kwargs, dict):
            raise ConversionError(f"{source_path}: env_args.env_kwargs must be an object")
        if env_kwargs.get("env_lang") is not None:
            raise ConversionError(
                f"{source_path}: unexpected non-null env_lang; task provenance needs review"
            )
        robots = env_kwargs.get("robots")
        if (
            not isinstance(robots, list)
            or not robots
            or not all(isinstance(robot, str) and robot for robot in robots)
        ):
            raise ConversionError(f"{source_path}: invalid robots metadata {robots!r}")
        declared_frequency = env_kwargs.get("control_freq")
        if declared_frequency is not None and float(declared_frequency) != DEFAULT_FPS:
            raise ConversionError(
                f"{source_path}: control_freq={declared_frequency!r}, expected official {DEFAULT_FPS}"
            )

        demo_names = sorted(data.keys(), key=robomimic_demo_sort_key)
        if not demo_names:
            raise ConversionError(f"{source_path}: data contains no demo_N episodes")
        ids = [robomimic_demo_sort_key(name) for name in demo_names]
        if ids != list(range(len(ids))):
            raise ConversionError(
                f"{source_path}: episode IDs are not continuous demo_0..demo_{len(ids) - 1}"
            )
        reference = data[demo_names[0]]
        reference_schema = hdf5_leaf_schema(reference)
        if "actions" not in reference or "states" not in reference or "obs" not in reference:
            raise ConversionError(f"{source_path}: missing actions, states, or obs")
        state_width = int(reference["states"].shape[1])
        first_model = decode_hdf5_text(
            reference.attrs.get("model_file"),
            description=f"{source_path}:data/{demo_names[0]}.attrs.model_file",
        )
        state_names = simulator_state_names(
            first_model,
            expected_width=state_width,
            description=f"{source_path}:data/{demo_names[0]}",
        )
        joints = _joint_names(first_model, description=f"{source_path}:data/{demo_names[0]}")
        action_names = _flattened_action_names(
            reference, description=f"{source_path}:data/{demo_names[0]}"
        )
        panda_action_layout = "right_rel_pos" in reference["action_dict"]

        schema_paths = {key for key, _shape, _dtype in reference_schema}
        absent_components = [
            name for name in ("rewards", "dones", "timestamps", "masks") if name not in schema_paths
        ]
        image_rows = [
            (key, shape, dtype)
            for key, shape, dtype in reference_schema
            if key.startswith("obs/") and key.endswith("_image")
        ]
        actual_cameras = {key.removeprefix("obs/").removesuffix("_image") for key, _, _ in image_rows}
        declared_cameras = env_kwargs.get("camera_names")
        if not isinstance(declared_cameras, list) or set(declared_cameras) != actual_cameras:
            raise ConversionError(
                f"{source_path}: camera_names {declared_cameras!r} do not match image leaves "
                f"{sorted(actual_cameras)}"
            )
        for key, shape, dtype in image_rows:
            if shape != (84, 84, 3) or dtype != "uint8":
                raise ConversionError(
                    f"{source_path}:{key}: expected HWC 84x84x3 uint8 RGB, got {shape} {dtype}"
                )

        all_frames = 0
        episode_lengths: dict[str, int] = {}
        model_refs: dict[str, ModelReference] = {}
        state_name_cache: dict[str, tuple[str, ...]] = {_sha256_text(first_model): state_names}
        joint_name_cache: dict[str, tuple[str, ...]] = {_sha256_text(first_model): joints}
        for demo_name in demo_names:
            demo = data[demo_name]
            try:
                num_samples = int(demo.attrs["num_samples"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ConversionError(f"{source_path}:{demo_name}: invalid num_samples") from exc
            if num_samples <= 0:
                raise ConversionError(f"{source_path}:{demo_name}: num_samples must be positive")
            validate_time_major_group(
                demo,
                reference_schema,
                expected_frames=num_samples,
                description=f"{source_path}:data/{demo_name}",
            )
            model_file = decode_hdf5_text(
                demo.attrs.get("model_file"),
                description=f"{source_path}:data/{demo_name}.attrs.model_file",
            )
            model_sha = _sha256_text(model_file)
            names = state_name_cache.get(model_sha)
            if names is None:
                names = simulator_state_names(
                    model_file,
                    expected_width=state_width,
                    description=f"{source_path}:data/{demo_name}",
                )
                state_name_cache[model_sha] = names
                joint_name_cache[model_sha] = _joint_names(
                    model_file, description=f"{source_path}:data/{demo_name}"
                )
            if names != state_names:
                raise ConversionError(
                    f"{source_path}:data/{demo_name}: simulator-state names/order differ within partition"
                )
            if joint_name_cache[model_sha] != joints:
                raise ConversionError(
                    f"{source_path}:data/{demo_name}: joint names/order differ within partition"
                )
            encoded_size = len(model_file.encode("utf-8"))
            model_refs[demo_name] = ModelReference(demo_name, model_sha, encoded_size)
            episode_lengths[demo_name] = num_samples
            all_frames += num_samples

        try:
            declared_total = int(data.attrs["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionError(f"{source_path}: invalid data.attrs.total") from exc
        if declared_total != all_frames:
            raise ConversionError(
                f"{source_path}: data.attrs.total={declared_total}, summed frames={all_frames}"
            )

        selected_names = demo_names[:max_episodes] if max_episodes is not None else demo_names
        selected_frames = sum(episode_lengths[name] for name in selected_names)
        vector_features: list[VectorFeatureSpec] = []
        camera_features: list[CameraFeatureSpec] = []
        field_mapping: list[dict[str, Any]] = []
        numeric_bytes_per_frame = 0
        image_bytes_per_frame = 0
        for source_key, tail_shape, dtype in reference_schema:
            feature_key = _feature_key(source_key)
            is_image = source_key.startswith("obs/") and source_key.endswith("_image")
            values_per_frame = int(np.prod(tail_shape)) if tail_shape else 1
            bytes_per_frame = values_per_frame * np.dtype(dtype).itemsize
            if is_image:
                camera_features.append(
                    CameraFeatureSpec(
                        feature_key=feature_key,
                        height=tail_shape[0],
                        width=tail_shape[1],
                    )
                )
                image_bytes_per_frame += bytes_per_frame
                transform = "streamed uint8 RGB to H.264 video"
            else:
                output_shape = tail_shape or (1,)
                names = _names_for_leaf(
                    source_key,
                    output_shape,
                    action_names=action_names,
                    state_names=state_names,
                    joint_names=joints,
                )
                if names is not None and len(names) != int(np.prod(output_shape)):
                    raise ConversionError(
                        f"{source_path}:{source_key}: {len(names)} names do not match shape {output_shape}"
                    )
                vector_features.append(
                    VectorFeatureSpec(
                        feature_key=feature_key,
                        dim=int(np.prod(output_shape)),
                        names=names,
                        dtype=dtype,
                        shape=output_shape,
                    )
                )
                numeric_bytes_per_frame += bytes_per_frame
                transform = "scalar wrapped as shape [1]" if not tail_shape else "identity"
            field_mapping.append(
                {
                    "source_key": f"data/<demo>/{source_key}",
                    "source_shape": ["T", *tail_shape],
                    "source_dtype": dtype,
                    "lerobot_key": feature_key,
                    "transform": transform,
                    "lossy": is_image,
                }
            )

    stat = source_path.stat()
    episodes = tuple(
        EpisodePlan(
            episode_uid=f"{relative}::data/{demo_name}",
            source_relative_path=f"{relative}::data/{demo_name}",
            instruction=TASK_INSTRUCTIONS[str(env_name)],
            num_frames=episode_lengths[demo_name],
            extra={
                "source_path": source_path,
                "demo_name": demo_name,
                "source_episode_id": demo_name,
                "source_task": env_name,
                "source_model_sha256": model_refs[demo_name].sha256,
                "source_model_uncompressed_bytes": model_refs[demo_name].uncompressed_bytes,
                "checkpoint_unit": f"{relative}::data/{demo_name}",
            },
        )
        for demo_name in selected_names
    )
    schema_payload = [
        {"key": key, "tail_shape": list(shape), "dtype": dtype}
        for key, shape, dtype in reference_schema
    ]
    schema_fingerprint = canonical_fingerprint({"leaves": schema_payload})
    plan = DatasetConversionPlan(
        dataset_uid=f"dexmimicgen_{partition}",
        output_path=collection_output / partition,
        fps=DEFAULT_FPS,
        measured_fps=float(DEFAULT_FPS),
        robot_type="+".join(robots),
        vector_features=tuple(vector_features),
        camera_features=tuple(camera_features),
        episodes=episodes,
        extra={
            "converter": "convert_dexmimicgen_to_lerobot.py",
            "source_root": str(raw_dataset_root),
            "source_dataset": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_relative_path": relative,
            "source_env_name": env_name,
            "source_env_args": env_args,
            "source_files": [
                {
                    "relative_path": relative,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            ],
            "field_mapping": field_mapping,
            "task_provenance": {
                "source_env_lang": None,
                "source_env_name": env_name,
                "instruction": TASK_INSTRUCTIONS[str(env_name)],
                "basis": "reviewed description of official environment success predicate",
                "official_code_revision": OFFICIAL_CODE_REVISION,
            },
            "timestamp_provenance": {
                "source_timestamps_present": False,
                "fps": DEFAULT_FPS,
                "basis": (
                    "source control_freq" if declared_frequency is not None else
                    "official environment default control_freq and playback writer"
                ),
                "resampled": False,
            },
            "partition_rules": {
                "rule": "one fixed-schema official HDF5 container per LeRobot partition",
                "partition": partition,
                "source_schema_fingerprint": schema_fingerprint,
                "no_padding_or_cast": True,
            },
            "unsupported_source_components": {
                "absent": absent_components,
                "fabricated": [],
            },
            "action_semantics": {
                "stored_actions_order": list(
                    PANDA_ACTION_KEYS
                    if panda_action_layout
                    else HUMANOID_STORED_ACTION_KEYS
                ),
                "official_training_action_order": list(
                    PANDA_ACTION_KEYS
                    if panda_action_layout
                    else HUMANOID_TRAINING_ACTION_KEYS
                ),
                "actions_and_action_dict_both_preserved": True,
            },
            "quaternion_convention": {
                "observation_fields": "xyzw",
                "simulator_qpos_names": "MuJoCo free/ball joint wxyz",
            },
            "payload_scan_coverage": {
                "metadata_schema_reference_scan": "all episodes",
                "sampled_payload_scan": "external preflight evidence",
                "full_payload_scan": False,
            },
            "selected_episode_count": len(selected_names),
            "all_episode_count": len(demo_names),
            "all_frame_count": all_frames,
            "unique_model_xml_count": len(state_name_cache),
            "model_sidecar": {
                "codec": "zlib level 9",
                "identity": "sha256 of UTF-8 XML",
                "path_template": "source_models/<sha256>.xml.zlib",
            },
        },
    )
    return DexMimicGenPartitionInfo(
        source_path=source_path,
        source_relative_path=relative,
        partition_name=partition,
        env_args=env_args,
        source_schema=reference_schema,
        plan=plan,
        all_episode_count=len(demo_names),
        all_frame_count=all_frames,
        selected_numeric_logical_bytes=selected_frames * numeric_bytes_per_frame,
        selected_image_logical_bytes=selected_frames * image_bytes_per_frame,
        selected_camera_frames=selected_frames * len(camera_features),
        model_references=tuple(model_refs[name] for name in selected_names),
        unique_model_count=len(state_name_cache),
        schema_fingerprint=schema_fingerprint,
        episode_length_summary={
            "min": min(episode_lengths.values()),
            "p50": int(np.percentile(list(episode_lengths.values()), 50)),
            "p95": int(np.percentile(list(episode_lengths.values()), 95)),
            "max": max(episode_lengths.values()),
        },
    )


def iter_frames(
    plan: DatasetConversionPlan,
    episode: EpisodePlan,
) -> Iterator[dict[str, Any]]:
    """Materialize only the current episode's numeric arrays; stream images."""

    h5py = require_h5py()
    source_path = Path(episode.extra["source_path"])
    demo_name = str(episode.extra["demo_name"])
    field_mapping = plan.extra["field_mapping"]
    with h5py.File(source_path, "r") as h5_file:
        demo = h5_file[f"data/{demo_name}"]
        numeric: dict[str, np.ndarray] = {}
        cameras: list[tuple[str, Any]] = []
        for item in field_mapping:
            source_key = item["source_key"].replace("data/<demo>/", "")
            dataset = demo[source_key]
            if item["lossy"]:
                cameras.append((item["lerobot_key"], dataset))
            else:
                numeric[item["lerobot_key"]] = np.asarray(dataset[:])
        for frame_index in range(episode.num_frames):
            frame: dict[str, Any] = {"task": episode.instruction}
            for item in field_mapping:
                if item["lossy"]:
                    continue
                feature_key = item["lerobot_key"]
                value = numeric[feature_key][frame_index]
                if not item["source_shape"][1:]:
                    value = np.asarray(value).reshape(1)
                frame[feature_key] = value
            for feature_key, dataset in cameras:
                frame[feature_key] = np.asarray(dataset[frame_index])
            yield frame


class DexMimicGenHdf5Reader:
    """Registry adapter for debugging one configured official partition."""

    def build_plan(
        self,
        config: Any,
        raw_root: Path,
        staging_root: Path,
    ) -> DatasetConversionPlan:
        source_file = getattr(config, "source_file", None)
        if not source_file:
            raise ConversionError(
                "format=dexmimicgen_hdf5 requires source_file for the generic single-partition CLI"
            )
        source_directory = getattr(config, "source_directory", None) or config.dataset_uid
        dataset_root = raw_root / source_directory
        info = inspect_partition(
            dataset_root / source_file,
            raw_dataset_root=dataset_root,
            collection_output=staging_root / "lerobot_v3_0" / config.dataset_uid,
        )
        return info.plan

    def iter_frames(
        self,
        plan: DatasetConversionPlan,
        episode: EpisodePlan,
    ) -> Iterator[dict[str, Any]]:
        yield from iter_frames(plan, episode)
