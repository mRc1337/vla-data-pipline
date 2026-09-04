"""Reader for robomimic multi-episode HDF5 containers.

Unlike :mod:`readers.hdf5_reader` (one file per episode), robomimic stores
many ``data/demo_N`` groups in each HDF5 file.  The reader preserves every
per-frame leaf independently and never merges, pads, reorders, or casts it.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterator

import numpy as np

from convert_core.episode_spec import (
    CameraFeatureSpec,
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


TASK_INSTRUCTIONS = {
    "Coffee": "Insert the coffee pod into the coffee machine and close the lid.",
    "CoffeePreparation": (
        "Retrieve the mug and coffee pod, place the mug on the coffee machine, "
        "insert the pod, and close the lid."
    ),
    "HammerCleanup": "Place the hammer in the drawer and close the drawer.",
    "Kitchen": (
        "Put the bread in the pot, cook it on the stove, turn the stove off, "
        "and place the pot in the serving region."
    ),
    "MugCleanup": "Place the mug in the drawer and close the drawer.",
    "NutAssembly": "Place the nuts onto their matching pegs.",
    "PickPlace": "Place each object into its corresponding bin.",
    "Square": "Place the square nut onto the square peg.",
    "Stack": "Stack one block on top of the other block.",
    "StackThree": "Stack the three blocks in the required order.",
    "Threading": "Thread the needle through the tripod opening.",
    "ThreePieceAssembly": "Assemble the three pieces.",
}

ACTION_NAMES = (
    "normalized_delta_x",
    "normalized_delta_y",
    "normalized_delta_z",
    "normalized_delta_axis_angle_x",
    "normalized_delta_axis_angle_y",
    "normalized_delta_axis_angle_z",
    "gripper_command",
)


@dataclass(frozen=True)
class RobomimicPartitionInfo:
    source_path: Path
    source_relative_path: str
    partition_name: str
    env_args: dict[str, Any]
    source_schema: tuple[tuple[str, tuple[int, ...], str], ...]
    source_splits: dict[str, tuple[str, ...]]
    dangling_split_references: dict[str, tuple[str, ...]]
    plan: DatasetConversionPlan
    all_episode_count: int
    all_frame_count: int


def _task_instruction(env_name: str) -> str:
    base = re.sub(r"_(?:D\d+|O\d+)$", "", env_name)
    try:
        return TASK_INSTRUCTIONS[base]
    except KeyError:
        raise ConversionError(
            f"no evidence-backed natural-language task mapping for environment {env_name!r}"
        ) from None


def _model_joint_names(model_file: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = re.findall(r"<joint[^>]*name=[\"']([^\"']+)", model_file)
    arm = tuple(name for name in names if name.startswith("robot0_"))
    gripper = tuple(name for name in names if name.startswith("gripper0_"))
    return arm, gripper


def _feature_key(source_key: str) -> str:
    if source_key == "actions":
        return "action"
    if source_key == "states":
        return "observation.sim_state"
    if source_key == "rewards":
        return "source.reward"
    if source_key == "dones":
        return "source.done"
    if source_key.startswith("obs/"):
        name = source_key.removeprefix("obs/")
        if name.endswith("_image"):
            return f"observation.images.{name.removesuffix('_image')}"
        return f"observation.{name}"
    return f"source.{source_key.replace('/', '.')}"


def _names_for_leaf(
    source_key: str,
    shape: tuple[int, ...],
    *,
    arm_joint_names: tuple[str, ...],
    gripper_joint_names: tuple[str, ...],
) -> tuple[str, ...] | None:
    width = int(np.prod(shape)) if shape else 1
    if source_key == "actions":
        if width != len(ACTION_NAMES):
            raise ConversionError(f"actions has width {width}, expected {len(ACTION_NAMES)}")
        return ACTION_NAMES
    if source_key in {"rewards", "dones"}:
        return (source_key.removesuffix("s"),)
    if source_key.endswith("robot0_joint_pos"):
        return arm_joint_names if len(arm_joint_names) == width else None
    if source_key.endswith("robot0_joint_vel"):
        return tuple(f"velocity({name})" for name in arm_joint_names) if len(arm_joint_names) == width else None
    if source_key.endswith("robot0_joint_pos_sin"):
        return tuple(f"sin({name})" for name in arm_joint_names) if len(arm_joint_names) == width else None
    if source_key.endswith("robot0_joint_pos_cos"):
        return tuple(f"cos({name})" for name in arm_joint_names) if len(arm_joint_names) == width else None
    if source_key.endswith("robot0_gripper_qpos"):
        return gripper_joint_names if len(gripper_joint_names) == width else None
    if source_key.endswith("robot0_gripper_qvel"):
        return tuple(f"velocity({name})" for name in gripper_joint_names) if len(gripper_joint_names) == width else None
    if source_key.endswith("_eef_pos") or "_eef_pos_rel_" in source_key:
        return ("x", "y", "z") if width == 3 else None
    if source_key.endswith("_eef_quat") or "_eef_quat_rel_" in source_key:
        return ("qx", "qy", "qz", "qw") if width == 4 else None
    if source_key.endswith("_eef_vel_lin") or source_key.endswith("_eef_vel_ang"):
        return ("x", "y", "z") if width == 3 else None
    return None


def inspect_partition(
    source_path: Path,
    *,
    raw_dataset_root: Path,
    collection_output: Path,
    max_episodes: int | None = None,
) -> RobomimicPartitionInfo:
    """Fully validate one container and return a fixed-schema LeRobot plan."""

    h5py = require_h5py()
    relative = source_path.relative_to(raw_dataset_root).as_posix()
    partition_name = "--".join(source_path.relative_to(raw_dataset_root).with_suffix("").parts)
    with h5py.File(source_path, "r") as h5_file:
        if "data" not in h5_file:
            raise ConversionError(f"{source_path}: missing data group")
        data = h5_file["data"]
        try:
            env_args = json.loads(data.attrs["env_args"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ConversionError(f"{source_path}: invalid data.attrs['env_args']: {exc}") from exc
        env_name = str(env_args.get("env_name", ""))
        env_kwargs = env_args.get("env_kwargs", {})
        robots = env_kwargs.get("robots")
        if not isinstance(robots, list) or len(robots) != 1 or not isinstance(robots[0], str):
            raise ConversionError(f"{source_path}: expected exactly one named robot, got {robots!r}")
        control_freq = env_kwargs.get("control_freq")
        if not isinstance(control_freq, (int, float)) or control_freq <= 0:
            raise ConversionError(f"{source_path}: invalid control_freq {control_freq!r}")
        if int(control_freq) != float(control_freq):
            raise ConversionError(f"{source_path}: non-integer control_freq {control_freq!r} is unsupported")
        instruction = _task_instruction(env_name)

        demo_names = sorted(data.keys(), key=robomimic_demo_sort_key)
        if not demo_names:
            raise ConversionError(f"{source_path}: data group contains no episodes")
        first = data[demo_names[0]]
        reference_schema = hdf5_leaf_schema(first)
        model_file = first.attrs.get("model_file", "")
        if isinstance(model_file, bytes):
            model_file = model_file.decode("utf-8")
        arm_names, gripper_names = _model_joint_names(str(model_file))

        source_splits: dict[str, tuple[str, ...]] = {}
        if "mask" in h5_file:
            for split_name, dataset in h5_file["mask"].items():
                source_splits[split_name] = tuple(
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in dataset[:]
                )
        memberships: dict[str, list[str]] = {name: [] for name in demo_names}
        dangling_split_references: dict[str, tuple[str, ...]] = {}
        for split_name, members in source_splits.items():
            unknown = set(members) - set(demo_names)
            if unknown:
                dangling_split_references[split_name] = tuple(
                    sorted(unknown, key=robomimic_demo_sort_key)
                )
            for member in members:
                if member in memberships:
                    memberships[member].append(split_name)

        all_frames = 0
        episode_lengths: dict[str, int] = {}
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
                description=f"{source_path}:{demo_name}",
            )
            episode_lengths[demo_name] = num_samples
            all_frames += num_samples
        declared_total = int(data.attrs.get("total", -1))
        if declared_total != all_frames:
            raise ConversionError(
                f"{source_path}: data.attrs['total']={declared_total}, summed frames={all_frames}"
            )

        selected_names = demo_names[:max_episodes] if max_episodes is not None else demo_names
        vector_features: list[VectorFeatureSpec] = []
        camera_features: list[CameraFeatureSpec] = []
        field_map: list[dict[str, Any]] = []
        for source_key, tail_shape, dtype in reference_schema:
            feature_key = _feature_key(source_key)
            if source_key.startswith("obs/") and source_key.endswith("_image"):
                if dtype != "uint8" or len(tail_shape) != 3 or tail_shape[-1] != 3:
                    raise ConversionError(
                        f"{source_path}:{source_key}: expected uint8 HWC RGB, got {tail_shape} {dtype}"
                    )
                camera_features.append(
                    CameraFeatureSpec(feature_key=feature_key, height=tail_shape[0], width=tail_shape[1])
                )
                transform = "uint8 RGB frames encoded to video"
            else:
                lerobot_shape = tail_shape if tail_shape else (1,)
                vector_features.append(
                    VectorFeatureSpec(
                        feature_key=feature_key,
                        dim=int(np.prod(lerobot_shape)),
                        names=_names_for_leaf(
                            source_key,
                            lerobot_shape,
                            arm_joint_names=arm_names,
                            gripper_joint_names=gripper_names,
                        ),
                        dtype=dtype,
                        shape=lerobot_shape,
                    )
                )
                transform = "scalar wrapped as shape [1]" if not tail_shape else "identity"
            field_map.append(
                {
                    "source_key": f"data/<demo>/{source_key}",
                    "source_shape": ["T", *tail_shape],
                    "source_dtype": dtype,
                    "lerobot_key": feature_key,
                    "transform": transform,
                    "lossy": source_key.startswith("obs/") and source_key.endswith("_image"),
                }
            )

    episodes = tuple(
        EpisodePlan(
            episode_uid=f"{relative}::data/{demo_name}",
            source_relative_path=f"{relative}::data/{demo_name}",
            instruction=instruction,
            num_frames=episode_lengths[demo_name],
            extra={
                "source_path": source_path,
                "demo_name": demo_name,
                "source_splits": tuple(sorted(memberships[demo_name])),
            },
        )
        for demo_name in selected_names
    )
    plan = DatasetConversionPlan(
        dataset_uid=f"mimicgen_{partition_name.replace('--', '_')}",
        output_path=collection_output / partition_name,
        fps=int(control_freq),
        measured_fps=float(control_freq),
        robot_type=robots[0],
        vector_features=tuple(vector_features),
        camera_features=tuple(camera_features),
        episodes=episodes,
        extra={
            "source_dataset": "MimicGen CoRL 2023 official release",
            "source_relative_path": relative,
            "source_env_name": env_name,
            "source_env_args": env_args,
            "source_splits": {key: list(value) for key, value in source_splits.items()},
            "dangling_split_references": {
                key: list(value) for key, value in dangling_split_references.items()
            },
            "field_mapping": field_map,
            "selected_episode_count": len(selected_names),
            "all_episode_count": len(demo_names),
            "all_frame_count": all_frames,
        },
    )
    return RobomimicPartitionInfo(
        source_path=source_path,
        source_relative_path=relative,
        partition_name=partition_name,
        env_args=env_args,
        source_schema=reference_schema,
        source_splits=source_splits,
        dangling_split_references=dangling_split_references,
        plan=plan,
        all_episode_count=len(demo_names),
        all_frame_count=all_frames,
    )


def iter_frames(plan: DatasetConversionPlan, episode: EpisodePlan) -> Iterator[dict[str, Any]]:
    h5py = require_h5py()
    source_path: Path = episode.extra["source_path"]
    demo_name: str = episode.extra["demo_name"]
    field_map: list[dict[str, Any]] = plan.extra["field_mapping"]
    with h5py.File(source_path, "r") as h5_file:
        demo = h5_file[f"data/{demo_name}"]
        arrays = {
            item["lerobot_key"]: demo[item["source_key"].replace("data/<demo>/", "")]
            for item in field_map
        }
        for frame_index in range(episode.num_frames):
            frame: dict[str, Any] = {"task": episode.instruction}
            for item in field_map:
                source_key = item["source_key"].replace("data/<demo>/", "")
                value = np.asarray(arrays[item["lerobot_key"]][frame_index])
                if not demo[source_key].shape[1:]:
                    value = value.reshape(1)
                frame[item["lerobot_key"]] = value
            yield frame


class RobomimicHdf5Reader:
    """Registry adapter for a single configured robomimic container."""

    def build_plan(self, config: Any, raw_root: Path, staging_root: Path) -> DatasetConversionPlan:
        if not getattr(config, "source_file", None):
            raise ConversionError("format=robomimic_hdf5 requires source_file")
        source_dir = getattr(config, "source_directory", None) or config.dataset_uid
        raw_dataset_root = raw_root / source_dir
        info = inspect_partition(
            raw_dataset_root / config.source_file,
            raw_dataset_root=raw_dataset_root,
            collection_output=staging_root / "lerobot_v3_0" / config.dataset_uid,
        )
        return info.plan

    def iter_frames(self, plan: DatasetConversionPlan, episode: EpisodePlan) -> Iterator[dict[str, Any]]:
        yield from iter_frames(plan, episode)
