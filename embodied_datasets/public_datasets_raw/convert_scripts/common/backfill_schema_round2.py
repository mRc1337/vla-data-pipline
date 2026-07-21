"""Table-driven corrections applied during the 2026-07-13 registry schema
round-2 enum extension (see
docs/superpowers/specs/2026-07-13-registry-schema-round2-design.md
section 4). Every value below is already justified in the corresponding
dataset's own onboarding/re-verification report -- this module applies
known corrections, it does not research new ones. Mirrors the pattern of
common/backfill_schema_refinement.py from the first schema-refinement
round.
"""
from __future__ import annotations

from typing import Any, Dict

from .schema import DatasetConfig

FIELD_CORRECTIONS: Dict[str, Dict[str, Any]] = {
    "1x_world_model_dataset": {"robot_platform": "1x_eve"},
    "agibot_world": {"robot_platform": "agibot_g2"},
    "alfred": {"robot_platform": "virtual_agent", "action_space": "discrete_symbolic"},
    "autobag": {"robot_platform": "abb_yumi", "collection_method": "scripted"},
    "behavior_1k": {"robot_platform": "r1pro"},
    "behavior_robot_suite": {"robot_platform": "galaxea_r1"},
    "dobb_e": {"robot_platform": "hello_robot_stretch"},
    "ego4d": {
        "embodiment_class": "human_full_body",
        "license": "custom_research_eula",
    },
    "ego_exo4d": {
        "license": "custom_research_eula",
        "additional_modalities": ["audio", "eye_gaze", "imu", "point_cloud_3d_scan"],
    },
    "egoallo": {"embodiment_class": "human_full_body"},
    "epic_kitchens_100": {
        "license": "custom_research_eula",
        "additional_modalities": ["audio", "imu"],
    },
    "h2o": {"license": "custom_research_eula", "action_space": "mixed"},
    "handloom": {"robot_platform": "abb_yumi", "gripper_type": "cage_pinch"},
    "hot3d": {"additional_modalities": ["eye_gaze", "point_cloud_3d_scan"]},
    "humanoid_x": {"robot_platform": "unitree_h1_2"},
    "nvidia_physicalai_robotics_manipulation_kitchen": {
        "additional_modalities": ["semantic_segmentation"],
    },
    "nvidia_physicalai_robotics_manipulation_objects": {
        "additional_modalities": ["semantic_segmentation"],
        "license": "custom_research_eula",
    },
    "nvidia_physicalai_robotics_manipulation_singlearm": {
        "rotation_representation": "mixed",
    },
    "oakink2": {"license": "CC-BY-SA-4.0"},
    "open_x_embodiment": {"gripper_type": "mixed"},
    "ovmm": {"robot_platform": "hello_robot_stretch"},
    "partnr": {
        "robot_platform": "boston_dynamics_spot",
        "camera_views": ["third_person", "head", "wrist", "gripper_jaw"],
    },
    "pokeflex": {"license": "custom_research_eula"},
    "robocoin": {"gripper_type": "mixed"},
    "robocook": {"rotation_representation": "single_axis_angle"},
    "robogene": {"robot_platform": "franka_fr3"},
    "robomind": {"gripper_type": "mixed"},
    "roboomni": {
        "gripper_type": "mixed",
        "secondary_collection_methods": ["synthetic_multimodal_augmentation"],
    },
    "roboverse": {"rotation_representation": "mixed"},
    "rt_1": {"robot_platform": "everyday_robots_arm"},
    "teach": {
        "robot_platform": "virtual_agent",
        "license": "CDLA-Sharing-1.0",
        "action_space": "discrete_symbolic",
    },
    "vitra": {"gripper_type": "none"},
    "dexmimicgen": {"rotation_representation": "mixed"},
    "xr_1_dataset": {"rotation_representation": "mixed"},
}


def apply_round2_corrections(config: DatasetConfig) -> DatasetConfig:
    """Return a new DatasetConfig with this dataset's round-2 table-driven
    corrections applied. Fields not mentioned in FIELD_CORRECTIONS for
    this dataset id are left untouched."""
    data = config.model_dump(mode="json", exclude_none=True)
    data.update(FIELD_CORRECTIONS.get(config.id, {}))
    return DatasetConfig(**data)
