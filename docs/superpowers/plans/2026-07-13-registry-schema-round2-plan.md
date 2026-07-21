# Registry Schema Round-2 Enum Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 37 new enum values (no new fields, no new enum types) surfaced by two full onboarding/re-verification passes across 82 dataset configs, then apply the specific known corrections those passes already justified.

**Architecture:** Pure additive changes to the existing enum classes in `common/schema.py` (every new value is a new member of an *existing* enum class — no new `Enum` subclasses, no new `DatasetConfig` fields). A second table-driven backfill script (mirroring `common/backfill_schema_refinement.py` from the first round) applies the ~35 known corrections to the affected configs.

**Tech Stack:** Python 3.9, pydantic 2.10.3, PyYAML 6.0.2, pytest 8.3.4 (already installed, unchanged).

## Global Constraints

- Python 3.9 syntax only: no `X | Y` union syntax, no `match` statements.
- All pydantic models keep `ConfigDict(extra="forbid")`. This task adds no new fields, so no new defaults are needed.
- New enum member names use `UPPER_SNAKE_CASE`; string values follow the existing per-enum style (hyphenated `CC-BY-...` style for `LicenseEnum`, lowercase `snake_case` for the rest). Exception: `RobotPlatform.X1_EVE = "1x_eve"` — the string value starts with a digit (matches the real product name "1X EVE"), which is legal for an enum *value* even though it could not be a bare Python identifier; the member name itself (`X1_EVE`) still starts with a letter.
- The backfill script is table-driven only — every value it writes must already be justified in a dataset's own `suggested_new_enum_values`/`field_sources` from its onboarding or re-verification report. No new research.
- `_ENUM_FIELDS` in `common/onboarding_agent.py` maps field name to enum *class*, not to individual members — adding members to existing classes requires no change there. Task 1 includes a test proving this (the new values must appear in `build_onboarding_prompt()`'s output without touching `onboarding_agent.py`).
- Run tests with: `python3 -m pytest tests/ -v` from the repo root.

---

### Task 1: Schema enum value additions

**Files:**
- Modify: `embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py` (full-file replacement below)
- Modify: `tests/test_schema.py` (append new test functions; existing tests must keep passing unchanged)
- Modify: `tests/test_onboarding_agent.py` (append one new test function; existing tests must keep passing unchanged)

**Interfaces:**
- Consumes: nothing new — this task only adds members to enum classes that already exist.
- Produces: the 37 new enum members listed below. Task 2 (the backfill script) writes these exact string values into dataset configs.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py` (keep all existing content above; add these functions at the end, following the file's existing convention of `from common.schema import X` inside each test function):

```python
def test_robot_platform_round2_new_members():
    from common.schema import RobotPlatform

    assert RobotPlatform.FRANKA_FR3.value == "franka_fr3"
    assert RobotPlatform.XARM6.value == "xarm6"
    assert RobotPlatform.KUKA_IIWA.value == "kuka_iiwa"
    assert RobotPlatform.ABB_YUMI.value == "abb_yumi"
    assert RobotPlatform.HELLO_ROBOT_STRETCH.value == "hello_robot_stretch"
    assert RobotPlatform.BOSTON_DYNAMICS_SPOT.value == "boston_dynamics_spot"
    assert RobotPlatform.EVERYDAY_ROBOTS_ARM.value == "everyday_robots_arm"
    assert RobotPlatform.AGIBOT_G2.value == "agibot_g2"
    assert RobotPlatform.GALBOT_G1.value == "galbot_g1"
    assert RobotPlatform.UNITREE_H1_2.value == "unitree_h1_2"
    assert RobotPlatform.UNITREE_ALIENGO.value == "unitree_aliengo"
    assert RobotPlatform.UNITREE_A1.value == "unitree_a1"
    assert RobotPlatform.ANYMAL.value == "anymal"
    assert RobotPlatform.ATLAS.value == "atlas"
    assert RobotPlatform.GALAXEA_R1.value == "galaxea_r1"
    assert RobotPlatform.R1PRO.value == "r1pro"
    assert RobotPlatform.X1_EVE.value == "1x_eve"
    assert RobotPlatform.VIRTUAL_AGENT.value == "virtual_agent"


def test_collection_method_round2_new_members():
    from common.schema import CollectionMethod

    assert CollectionMethod.KINESTHETIC.value == "kinesthetic"
    assert CollectionMethod.SCRIPTED.value == "scripted"
    assert CollectionMethod.EGO_EXO_HUMAN.value == "ego_exo_human"
    assert CollectionMethod.MOCAP_MULTIVIEW_HUMAN.value == "mocap_multiview_human"
    assert (
        CollectionMethod.SYNTHETIC_MULTIMODAL_AUGMENTATION.value
        == "synthetic_multimodal_augmentation"
    )


def test_license_round2_new_members():
    from common.schema import LicenseEnum

    assert LicenseEnum.CC_BY_SA_4_0.value == "CC-BY-SA-4.0"
    assert LicenseEnum.CDLA_SHARING_1_0.value == "CDLA-Sharing-1.0"
    assert LicenseEnum.CUSTOM_RESEARCH_EULA.value == "custom_research_eula"


def test_embodiment_class_round2_new_members():
    from common.schema import EmbodimentClass

    assert EmbodimentClass.HUMAN_FULL_BODY.value == "human_full_body"
    assert EmbodimentClass.HALF_HUMANOID.value == "half_humanoid"


def test_gripper_type_round2_new_members():
    from common.schema import GripperType

    assert GripperType.MIXED.value == "mixed"
    assert GripperType.CAGE_PINCH.value == "cage_pinch"


def test_rotation_representation_round2_new_members():
    from common.schema import RotationRepresentation

    assert RotationRepresentation.MIXED.value == "mixed"
    assert RotationRepresentation.SINGLE_AXIS_ANGLE.value == "single_axis_angle"


def test_sensor_modality_round2_new_members():
    from common.schema import SensorModality

    assert SensorModality.IMU.value == "imu"
    assert SensorModality.SEMANTIC_SEGMENTATION.value == "semantic_segmentation"
    assert SensorModality.POINT_CLOUD_3D_SCAN.value == "point_cloud_3d_scan"


def test_action_space_round2_new_members():
    from common.schema import ActionSpace

    assert ActionSpace.DISCRETE_SYMBOLIC.value == "discrete_symbolic"


def test_camera_view_round2_new_members():
    from common.schema import CameraView

    assert CameraView.GRIPPER_JAW.value == "gripper_jaw"


def test_dataset_config_accepts_round2_enum_values():
    config = DatasetConfig(
        id="droid",
        name="DROID",
        robot_platform="abb_yumi",
        collection_method="ego_exo_human",
        license="custom_research_eula",
        embodiment_class="half_humanoid",
        gripper_type="mixed",
        rotation_representation="single_axis_angle",
        additional_modalities=["imu", "semantic_segmentation"],
        action_space="discrete_symbolic",
        camera_views=["gripper_jaw"],
    )
    assert config.robot_platform.value == "abb_yumi"
    assert config.collection_method.value == "ego_exo_human"
    assert config.license.value == "custom_research_eula"
    assert config.embodiment_class.value == "half_humanoid"
    assert config.gripper_type.value == "mixed"
    assert config.rotation_representation.value == "single_axis_angle"
    assert [m.value for m in config.additional_modalities] == [
        "imu",
        "semantic_segmentation",
    ]
    assert config.action_space.value == "discrete_symbolic"
    assert config.camera_views[0].value == "gripper_jaw"
```

Append to `tests/test_onboarding_agent.py` (keep all existing content above; add this function at the end):

```python
def test_build_onboarding_prompt_includes_round2_enum_values():
    prompt = build_onboarding_prompt("droid", "DROID", "https://droid-dataset.github.io/")
    assert "abb_yumi" in prompt
    assert "ego_exo_human" in prompt
    assert "custom_research_eula" in prompt
    assert "half_humanoid" in prompt
    assert "cage_pinch" in prompt
    assert "single_axis_angle" in prompt
    assert "imu" in prompt
    assert "discrete_symbolic" in prompt
    assert "gripper_jaw" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_schema.py tests/test_onboarding_agent.py -v`
Expected: FAIL — `AttributeError: FRANKA_FR3` (and similarly for every other new member referenced) since none of the new enum members exist yet.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_schema.py tests/test_onboarding_agent.py -v`
Expected: PASS (all existing tests + 10 new tests in `test_schema.py` + 1 new test in `test_onboarding_agent.py`)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests, including `test_paths.py`, `test_io.py`, `test_migrate_xlsx_to_registry.py`, `test_backfill_schema_refinement.py`, `test_generate_overview_readme.py`, `test_scaffolding.py`)

- [ ] **Step 6: Commit**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py tests/test_schema.py tests/test_onboarding_agent.py
git commit -m "feat: add 37 round-2 enum values surfaced by 82-dataset onboarding wave"
```

---

### Task 2: Round-2 backfill script

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/backfill_schema_round2.py`
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_round2.py`
- Test: `tests/test_backfill_schema_round2.py`
- Modify (via running the script for real, not by hand): the affected dataset config YAMLs under `embodied_datasets/public_datasets_raw/convert_scripts/configs/*.yaml`

**Interfaces:**
- Consumes: `DatasetConfig` from `common.schema` (Task 1); `load_dataset_config`/`save_dataset_config` from `common.io` (already exists, unchanged).
- Produces: `apply_round2_corrections(config: DatasetConfig) -> DatasetConfig` (pure function, used by the runner script below and by tests).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backfill_schema_round2.py`:

```python
from common.backfill_schema_round2 import apply_round2_corrections
from common.schema import DatasetConfig


def test_apply_round2_corrections_sets_robot_platform():
    config = DatasetConfig(id="rt_1", name="RT-1", robot_platform="other")
    updated = apply_round2_corrections(config)
    assert updated.robot_platform.value == "everyday_robots_arm"


def test_apply_round2_corrections_sets_multiple_fields_for_one_dataset():
    config = DatasetConfig(id="teach", name="TEACh", robot_platform="other")
    updated = apply_round2_corrections(config)
    assert updated.robot_platform.value == "virtual_agent"
    assert updated.license.value == "CDLA-Sharing-1.0"
    assert updated.action_space.value == "discrete_symbolic"


def test_apply_round2_corrections_overwrites_additional_modalities_list():
    config = DatasetConfig(
        id="ego_exo4d", name="Ego-Exo4D", additional_modalities=["audio", "eye_gaze"]
    )
    updated = apply_round2_corrections(config)
    assert [m.value for m in updated.additional_modalities] == [
        "audio",
        "eye_gaze",
        "imu",
        "point_cloud_3d_scan",
    ]


def test_apply_round2_corrections_splits_wrist_and_gripper_jaw_cameras():
    config = DatasetConfig(
        id="partnr", name="PARTNR", camera_views=["third_person", "head", "wrist"]
    )
    updated = apply_round2_corrections(config)
    assert [v.value for v in updated.camera_views] == [
        "third_person",
        "head",
        "wrist",
        "gripper_jaw",
    ]


def test_apply_round2_corrections_fixes_vitra_gripper_type():
    config = DatasetConfig(id="vitra", name="VITRA", gripper_type="dexterous_hand")
    updated = apply_round2_corrections(config)
    assert updated.gripper_type.value == "none"


def test_apply_round2_corrections_leaves_untouched_datasets_unchanged():
    config = DatasetConfig(
        id="droid", name="DROID", source_url="https://a", license="MIT"
    )
    updated = apply_round2_corrections(config)
    assert updated.source_url == "https://a"
    assert updated.license.value == "MIT"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_backfill_schema_round2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.backfill_schema_round2'`

- [ ] **Step 3: Write the implementation**

Create `embodied_datasets/public_datasets_raw/convert_scripts/common/backfill_schema_round2.py`:

```python
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
```

Create `embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_round2.py`:

```python
"""One-off script: apply the 2026-07-13 registry schema round-2 enum
extension's table-driven corrections
(common/backfill_schema_round2.py) to every existing dataset config.

Run from repo root:
    python3 embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_round2.py
"""
from __future__ import annotations

from pathlib import Path

from common.backfill_schema_round2 import apply_round2_corrections
from common.io import load_dataset_config, save_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = (
    REPO_ROOT
    / "embodied_datasets"
    / "public_datasets_raw"
    / "convert_scripts"
    / "configs"
)


def main() -> None:
    paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    for path in paths:
        config = load_dataset_config(path)
        updated = apply_round2_corrections(config)
        save_dataset_config(updated, path)
    print(f"backfilled {len(paths)} dataset configs in {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_backfill_schema_round2.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 6: Run the backfill script for real against the actual configs**

Run from repo root: `python3 embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_round2.py`
Expected output: `backfilled 82 dataset configs in .../configs`

- [ ] **Step 7: Spot-check the real output**

Run: `grep "^robot_platform" embodied_datasets/public_datasets_raw/convert_scripts/configs/rt_1.yaml` — expect `robot_platform: everyday_robots_arm`.
Run: `grep "^license" embodied_datasets/public_datasets_raw/convert_scripts/configs/oakink2.yaml` — expect `license: CC-BY-SA-4.0`.
Run: `grep "^gripper_type" embodied_datasets/public_datasets_raw/convert_scripts/configs/vitra.yaml` — expect `gripper_type: none`.
Run: `grep -A5 "^camera_views" embodied_datasets/public_datasets_raw/convert_scripts/configs/partnr.yaml` — expect both `- wrist` and `- gripper_jaw` present.

- [ ] **Step 8: Commit**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/backfill_schema_round2.py \
        embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_round2.py \
        tests/test_backfill_schema_round2.py \
        embodied_datasets/public_datasets_raw/convert_scripts/configs/
git commit -m "feat: apply round-2 schema corrections across 34 affected dataset configs"
```
