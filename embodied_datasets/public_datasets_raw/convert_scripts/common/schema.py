"""Pydantic models for the VLA dataset registry and per-dataset onboarding
config. Field names and enum values must stay in sync with
docs/superpowers/specs/2026-07-08-vla-data-pipeline-design.md sections 5.1-5.3.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class DownloadStatus(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"


class IntegrityStatus(str, Enum):
    NOT_VERIFIED = "not_verified"
    VERIFIED = "verified"
    FAILED = "failed"


class ConvertStatus(str, Enum):
    NOT_CONVERTED = "not_converted"
    CONVERTING = "converting"
    CONVERTED = "converted"
    FAILED = "failed"


class ProcessStatus(str, Enum):
    NOT_PROCESSED = "not_processed"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    priority: Priority = Priority.P2
    download_status: DownloadStatus = DownloadStatus.NOT_DOWNLOADED
    integrity_status: IntegrityStatus = IntegrityStatus.NOT_VERIFIED
    convert_status: ConvertStatus = ConvertStatus.NOT_CONVERTED
    process_status: ProcessStatus = ProcessStatus.NOT_PROCESSED
    raw_local_path: Optional[str] = None
    lerobot_v2_1_local_path: Optional[str] = None
    storage_size_gb: Optional[float] = None
    num_episodes: Optional[int] = None
    num_frames: Optional[int] = None
    duration_hours: Optional[float] = None


class LicenseEnum(str, Enum):
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    BSD_3_CLAUSE = "BSD-3-Clause"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"
    CC0_1_0 = "CC0-1.0"
    GPL_3_0 = "GPL-3.0"
    PROPRIETARY = "Proprietary"
    UNKNOWN = "Unknown"


class RawFormat(str, Enum):
    RLDS = "RLDS"
    HDF5 = "HDF5"
    LEROBOT = "LeRobot"
    ROS_BAG = "ROS_bag"
    MCAP = "MCAP"
    TFRECORD = "TFRecord"
    CUSTOM = "Custom"


class CollectionMethod(str, Enum):
    TELEOP = "teleop"
    AUTONOMOUS_POLICY = "autonomous_policy"
    UMI = "umi"
    EGOCENTRIC_HUMAN = "egocentric_human"
    SIMULATION = "simulation"
    HUMAN_TO_ROBOT_SYNTHESIS = "human_to_robot_synthesis"


class EmbodimentClass(str, Enum):
    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"
    HUMANOID = "humanoid"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    HUMAN_HAND = "human_hand"


class RobotPlatform(str, Enum):
    FRANKA_PANDA = "franka_panda"
    UR5 = "ur5"
    UR5E = "ur5e"
    AGILEX_ALOHA = "agilex_aloha"
    AGILEX_COBOT_MAGIC = "agilex_cobot_magic"
    XARM7 = "xarm7"
    KINOVA_GEN3 = "kinova_gen3"
    SAWYER = "sawyer"
    WIDOWX = "widowx"
    VIPERX = "viperx"
    AGIBOT_G1 = "agibot_g1"
    TIEN_KUNG = "tien_kung"
    ARX5 = "arx5"
    UNITREE_G1 = "unitree_g1"
    OTHER = "other"


class GripperType(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    DEXTEROUS_HAND = "dexterous_hand"
    SUCTION = "suction"
    NONE = "none"
    UNKNOWN = "unknown"


class ActionSpace(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    EEF_POSE = "eef_pose"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ActionFrame(str, Enum):
    DELTA = "delta"
    ABSOLUTE = "absolute"
    BOTH = "both"
    UNKNOWN = "unknown"


class RotationRepresentation(str, Enum):
    EULER_XYZ = "euler_xyz"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    AXIS_ANGLE = "axis_angle"
    ROTATION_MATRIX = "rotation_matrix"
    NONE = "none"
    UNKNOWN = "unknown"


class CameraView(str, Enum):
    THIRD_PERSON = "third_person"
    HEAD = "head"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    TOP = "top"
    FRONT = "front"
    SIDE = "side"
    OTHER = "other"


class DepthCoverage(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class UrdfSource(str, Enum):
    DATASET_REPO = "dataset_repo"
    MANUFACTURER_OFFICIAL = "manufacturer_official"
    COMMUNITY_REPO = "community_repo"
    NOT_FOUND = "not_found"


class ReviewStatus(str, Enum):
    PENDING_HUMAN_REVIEW = "pending_human_review"
    CONFIRMED = "confirmed"


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    source_url: Optional[str] = None
    license: Optional[LicenseEnum] = None
    raw_format: Optional[RawFormat] = None
    collection_method: Optional[CollectionMethod] = None
    embodiment_class: Optional[EmbodimentClass] = None
    robot_platform: Optional[RobotPlatform] = None
    num_arms: Optional[int] = Field(default=None, ge=0, le=2)
    dof_per_arm: Optional[int] = None
    gripper_type: Optional[GripperType] = None
    has_mobile_base: Optional[bool] = None
    action_space: Optional[ActionSpace] = None
    action_frame: Optional[ActionFrame] = None
    rotation_representation: Optional[RotationRepresentation] = None
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    fps: Optional[float] = None
    fps_variable: Optional[bool] = None
    num_camera_views: Optional[int] = None
    camera_views: List[CameraView] = Field(default_factory=list)
    has_camera_calibration: Optional[bool] = None
    depth_coverage: Optional[DepthCoverage] = None
    has_language_instruction: Optional[bool] = None
    num_task_types: Optional[int] = None
    urdf_available: Optional[bool] = None
    urdf_source: Optional[UrdfSource] = None
    expected_size_gb: Optional[float] = None
    expected_num_episodes: Optional[int] = None
    review_status: ReviewStatus = ReviewStatus.PENDING_HUMAN_REVIEW
    field_sources: Dict[str, str] = Field(default_factory=dict)
    suggested_new_enum_values: Dict[str, str] = Field(default_factory=dict)
