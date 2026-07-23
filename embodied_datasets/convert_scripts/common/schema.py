"""Pydantic models for the VLA dataset registry and per-dataset onboarding
config. Field names and enum values must stay in sync with
docs/superpowers/specs/2026-07-08-vla-data-pipeline-design.md sections 5.1-5.3,
docs/superpowers/specs/2026-07-10-registry-schema-refinement-design.md
sections 3-4, and
docs/superpowers/specs/2026-07-13-registry-schema-round2-design.md section 2.
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
    final_local_path: Optional[str] = None
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
    CC_BY_NC_SA_4_0 = "CC-BY-NC-SA-4.0"
    CC_BY_NC_ND_4_0 = "CC-BY-NC-ND-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC0_1_0 = "CC0-1.0"
    CDLA_SHARING_1_0 = "CDLA-Sharing-1.0"
    GPL_3_0 = "GPL-3.0"
    PROPRIETARY = "Proprietary"
    CUSTOM_RESEARCH_EULA = "custom_research_eula"
    UNKNOWN = "Unknown"


class RawFormat(str, Enum):
    RLDS = "RLDS"
    HDF5 = "HDF5"
    LEROBOT = "LeRobot"
    ROS_BAG = "ROS_bag"
    MCAP = "MCAP"
    TFRECORD = "TFRecord"
    VRS = "VRS"
    CUSTOM = "Custom"


class CollectionMethod(str, Enum):
    TELEOP = "teleop"
    KINESTHETIC = "kinesthetic"
    AUTONOMOUS_POLICY = "autonomous_policy"
    SCRIPTED = "scripted"
    UMI = "umi"
    EGOCENTRIC_HUMAN = "egocentric_human"
    EGO_EXO_HUMAN = "ego_exo_human"
    MOCAP_MULTIVIEW_HUMAN = "mocap_multiview_human"
    SIMULATION = "simulation"
    HUMAN_TO_ROBOT_SYNTHESIS = "human_to_robot_synthesis"
    SYNTHETIC_MULTIMODAL_AUGMENTATION = "synthetic_multimodal_augmentation"
    AR_HAPTIC_GUIDED = "ar_haptic_guided_synthesis"
    SCENE_ASSET_CURATION = "scene_asset_curation"


class EmbodimentClass(str, Enum):
    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"
    HALF_HUMANOID = "half_humanoid"
    HUMANOID = "humanoid"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    HUMAN_HAND = "human_hand"
    HUMAN_FULL_BODY = "human_full_body"
    QUADRUPED = "quadruped"


class RobotPlatform(str, Enum):
    FRANKA_PANDA = "franka_panda"
    FRANKA_FR3 = "franka_fr3"
    UR5 = "ur5"
    UR5E = "ur5e"
    AGILEX_ALOHA = "agilex_aloha"
    AGILEX_COBOT_MAGIC = "agilex_cobot_magic"
    XARM6 = "xarm6"
    XARM7 = "xarm7"
    KINOVA_GEN3 = "kinova_gen3"
    KUKA_IIWA = "kuka_iiwa"
    SAWYER = "sawyer"
    WIDOWX = "widowx"
    VIPERX = "viperx"
    ABB_YUMI = "abb_yumi"
    HELLO_ROBOT_STRETCH = "hello_robot_stretch"
    BOSTON_DYNAMICS_SPOT = "boston_dynamics_spot"
    EVERYDAY_ROBOTS_ARM = "everyday_robots_arm"
    AGIBOT_G1 = "agibot_g1"
    AGIBOT_G2 = "agibot_g2"
    GALBOT_G1 = "galbot_g1"
    TIEN_KUNG = "tien_kung"
    ARX5 = "arx5"
    UNITREE_G1 = "unitree_g1"
    UNITREE_H1 = "unitree_h1"
    UNITREE_H1_2 = "unitree_h1_2"
    UNITREE_ALIENGO = "unitree_aliengo"
    UNITREE_A1 = "unitree_a1"
    ANYMAL = "anymal"
    ATLAS = "atlas"
    FOURIER_GR1 = "fourier_gr1"
    FLEXIV_RIZON4 = "flexiv_rizon4"
    GALAXEA_R1 = "galaxea_r1"
    GALAXEA_R1_LITE = "galaxea_r1_lite"
    R1PRO = "r1pro"
    TOYOTA_ELEY = "toyota_eley"
    X1_EVE = "1x_eve"
    VIRTUAL_AGENT = "virtual_agent"
    OTHER = "other"


class GripperType(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    DEXTEROUS_HAND = "dexterous_hand"
    THREE_JAW = "three_jaw"
    CAGE_PINCH = "cage_pinch"
    SUCTION = "suction"
    MIXED = "mixed"
    NONE = "none"
    UNKNOWN = "unknown"


class ActionSpace(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    EEF_POSE = "eef_pose"
    DISCRETE_SYMBOLIC = "discrete_symbolic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ActionFrame(str, Enum):
    DELTA = "delta"
    ABSOLUTE = "absolute"
    BOTH = "both"
    RELATIVE_TRAJECTORY = "relative_trajectory"
    MIXED_DELTA_ABSOLUTE = "mixed_delta_absolute"
    UNKNOWN = "unknown"


class RotationRepresentation(str, Enum):
    EULER_XYZ = "euler_xyz"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    AXIS_ANGLE = "axis_angle"
    SINGLE_AXIS_ANGLE = "single_axis_angle"
    ROTATION_MATRIX = "rotation_matrix"
    MIXED = "mixed"
    NONE = "none"
    UNKNOWN = "unknown"


class CameraView(str, Enum):
    THIRD_PERSON = "third_person"
    HEAD = "head"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    WRIST = "wrist"
    GRIPPER_JAW = "gripper_jaw"
    BODY_WORN = "body_worn"
    WORMS_EYE = "worms_eye"
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


class ReleaseType(str, Enum):
    FIXED_EPISODE_DATASET = "fixed_episode_dataset"
    GENERATION_FRAMEWORK = "generation_framework"
    SCENE_PLATFORM = "scene_platform"
    RL_BENCHMARK_ENV = "rl_benchmark_env"


class HandPoseRepresentation(str, Enum):
    MANO = "mano"
    KEYPOINTS_3D = "keypoints_3d"
    JOINT_ANGLES = "joint_angles"
    NONE = "none"


class SensorModality(str, Enum):
    FORCE_TORQUE = "force_torque"
    TACTILE = "tactile"
    AUDIO = "audio"
    EYE_GAZE = "eye_gaze"
    IMU = "imu"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POINT_CLOUD_3D_SCAN = "point_cloud_3d_scan"


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    source_url: Optional[str] = None
    paper_url: Optional[str] = None
    license: Optional[LicenseEnum] = None
    raw_format: Optional[RawFormat] = None
    release_type: Optional[ReleaseType] = None
    collection_method: Optional[CollectionMethod] = None
    secondary_collection_methods: List[CollectionMethod] = Field(default_factory=list)
    is_multi_embodiment: Optional[bool] = None
    embodiment_class: Optional[EmbodimentClass] = None
    robot_platform: Optional[RobotPlatform] = None
    num_arms: Optional[int] = Field(default=None, ge=0, le=2)
    dof_per_arm: Optional[int] = None
    dof_per_hand: Optional[int] = None
    gripper_type: Optional[GripperType] = None
    hand_pose_representation: Optional[HandPoseRepresentation] = Field(
        default=None,
        description=(
            "Finger/hand-shape encoding (MANO / 3D keypoints / joint angles) for "
            "dexterous hands or bare-hand tracking. Complementary to, not a "
            "substitute for, rotation_representation below -- a dataset with a "
            "dexterous hand can have both set simultaneously."
        ),
    )
    has_mobile_base: Optional[bool] = None
    action_space: Optional[ActionSpace] = None
    action_frame: Optional[ActionFrame] = None
    rotation_representation: Optional[RotationRepresentation] = Field(
        default=None,
        description=(
            "Wrist/end-effector orientation encoding (euler/quaternion/6D/etc). "
            "Complementary to, not a substitute for, hand_pose_representation "
            "above, which encodes finger/hand shape instead."
        ),
    )
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    fps: Optional[float] = None
    fps_variable: Optional[bool] = None
    num_camera_views: Optional[int] = None
    camera_views: List[CameraView] = Field(default_factory=list)
    has_synchronized_multiview_rig: Optional[bool] = None
    has_camera_calibration: Optional[bool] = None
    depth_coverage: Optional[DepthCoverage] = None
    additional_modalities: List[SensorModality] = Field(default_factory=list)
    has_language_instruction: Optional[bool] = None
    num_task_types: Optional[int] = None
    urdf_available: Optional[bool] = None
    urdf_source: Optional[UrdfSource] = None
    expected_size_gb: Optional[float] = None
    expected_num_episodes: Optional[int] = None
    expected_duration_hours: Optional[float] = None
    num_subjects: Optional[int] = None
    num_scenes: Optional[int] = None
    num_objects: Optional[int] = None
    review_status: ReviewStatus = ReviewStatus.PENDING_HUMAN_REVIEW
    field_sources: Dict[str, str] = Field(default_factory=dict)
    suggested_new_enum_values: Dict[str, str] = Field(default_factory=dict)
