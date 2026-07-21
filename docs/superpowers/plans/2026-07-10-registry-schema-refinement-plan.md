# Registry Schema Refinement + Configurable Data Root + Overview README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable data-root path-resolution module, extend the dataset registry schema with the field/enum gaps surfaced by the 59-dataset onboarding wave, apply the known corrections to the existing configs, and add a semi-auto-generated top-level README for `embodied_datasets/`.

**Architecture:** All changes are additive to the existing Plan A foundation (`embodied_datasets/public_datasets_raw/convert_scripts/common/`). No existing field is removed or renamed; new enum values/fields are optional so all 59 existing config YAMLs keep validating unchanged until the backfill task explicitly updates them via a table-driven script (no new research).

**Tech Stack:** Python 3.9, pydantic 2.10.3, PyYAML 6.0.2, pytest 8.3.4 (versions pinned in `requirements.txt`, already installed).

## Global Constraints

- Python 3.9 syntax only: no `X | Y` union syntax, no `match` statements. Use `from __future__ import annotations` + `typing.Optional`/`List`/`Dict`/`Type`.
- All pydantic models keep `ConfigDict(extra="forbid")`. Every new field must have an explicit default (`None` or `Field(default_factory=...)`).
- New enum member names use `UPPER_SNAKE_CASE`; string values follow the existing per-enum style (lowercase `snake_case` for most enums, hyphenated `CC-BY-...` style for `LicenseEnum`, exact `VRS`/`HDF5`-style casing for `RawFormat`).
- Path-resolution functions take `data_root: Path` as an explicit parameter — no module-level global mutable state.
- `datasets_registry.yaml` and `convert_scripts/configs/*.yaml` are never touched by `--data-root` / `common/paths.py` — only the heavy data directories (`raw/`, `lerobot_v2_1_staging/`, `public_datasets/lerobot_v2_1/`, `urdf_assets/`) are.
- The backfill script (Task 4) is table-driven only — every value it writes must already be justified in a dataset's existing `field_sources`/`suggested_new_enum_values`. It must not perform any new research or invent values.
- Run tests with: `python3 -m pytest tests/ -v` from the repo root (`pythonpath` is already configured in `pyproject.toml`).

---

### Task 1: Configurable data-root path resolution

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Produces: `REPO_ROOT: Path`, `DEFAULT_DATA_ROOT: Path`, `resolve_data_root(cli_value: Optional[str]) -> Path`, `raw_dir(data_root: Path, dataset_id: str) -> Path`, `lerobot_v2_1_staging_dir(data_root: Path, dataset_id: str) -> Path`, `lerobot_v2_1_final_dir(data_root: Path, dataset_id: str) -> Path`, `urdf_assets_dir(data_root: Path, robot_platform: str) -> Path`. Later tasks (and future Plan B/C/D scripts) import these from `common.paths`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paths.py`:

```python
from pathlib import Path

from common.paths import (
    DEFAULT_DATA_ROOT,
    REPO_ROOT,
    lerobot_v2_1_final_dir,
    lerobot_v2_1_staging_dir,
    raw_dir,
    resolve_data_root,
    urdf_assets_dir,
)


def test_default_data_root_is_repo_embodied_datasets():
    assert DEFAULT_DATA_ROOT == REPO_ROOT / "embodied_datasets"


def test_resolve_data_root_defaults_when_none():
    assert resolve_data_root(None) == DEFAULT_DATA_ROOT


def test_resolve_data_root_defaults_when_empty_string():
    assert resolve_data_root("") == DEFAULT_DATA_ROOT


def test_resolve_data_root_uses_explicit_value(tmp_path):
    custom = tmp_path / "my_data"
    assert resolve_data_root(str(custom)) == custom.resolve()


def test_raw_dir():
    data_root = Path("/mnt/big_disk")
    assert raw_dir(data_root, "droid") == Path(
        "/mnt/big_disk/public_datasets_raw/droid/raw"
    )


def test_lerobot_v2_1_staging_dir():
    data_root = Path("/mnt/big_disk")
    assert lerobot_v2_1_staging_dir(data_root, "droid") == Path(
        "/mnt/big_disk/public_datasets_raw/droid/lerobot_v2_1_staging"
    )


def test_lerobot_v2_1_final_dir():
    data_root = Path("/mnt/big_disk")
    assert lerobot_v2_1_final_dir(data_root, "droid") == Path(
        "/mnt/big_disk/public_datasets/lerobot_v2_1/droid"
    )


def test_urdf_assets_dir():
    data_root = Path("/mnt/big_disk")
    assert urdf_assets_dir(data_root, "franka_panda") == Path(
        "/mnt/big_disk/urdf_assets/franka_panda"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.paths'`

- [ ] **Step 3: Write the implementation**

Create `embodied_datasets/public_datasets_raw/convert_scripts/common/paths.py`:

```python
"""Resolve the configurable data root and build paths under it for the
heavy data directories (raw/, lerobot_v2_1_staging/, public_datasets/,
urdf_assets/). datasets_registry.yaml and convert_scripts/configs/ always
stay inside the repo and are unaffected by this module -- see
docs/superpowers/specs/2026-07-10-registry-schema-refinement-design.md
section 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPO_ROOT / "embodied_datasets"


def resolve_data_root(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).resolve()
    return DEFAULT_DATA_ROOT


def raw_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "public_datasets_raw" / dataset_id / "raw"


def lerobot_v2_1_staging_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "public_datasets_raw" / dataset_id / "lerobot_v2_1_staging"


def lerobot_v2_1_final_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "public_datasets" / "lerobot_v2_1" / dataset_id


def urdf_assets_dir(data_root: Path, robot_platform: str) -> Path:
    return data_root / "urdf_assets" / robot_platform
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_paths.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/paths.py tests/test_paths.py
git commit -m "feat: add configurable data-root path resolution module"
```

---

### Task 2: Schema enum/field extensions

**Files:**
- Modify: `embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py` (full-file replacement below)
- Modify: `tests/test_schema.py` (append new test functions; existing tests must keep passing unchanged)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: enum classes `ReleaseType`, `HandPoseRepresentation`, `SensorModality`; new members on `LicenseEnum` (`CC_BY_NC_SA_4_0`, `CC_BY_NC_ND_4_0`), `RawFormat` (`VRS`), `CollectionMethod` (`AR_HAPTIC_GUIDED`, `SCENE_ASSET_CURATION`), `EmbodimentClass` (`QUADRUPED`), `RobotPlatform` (`UNITREE_H1`, `FOURIER_GR1`, `FLEXIV_RIZON4`, `GALAXEA_R1_LITE`, `TOYOTA_ELEY`), `GripperType` (`THREE_JAW`), `ActionFrame` (`RELATIVE_TRAJECTORY`, `MIXED_DELTA_ABSOLUTE`), `CameraView` (`WRIST`, `BODY_WORN`, `WORMS_EYE`); new `DatasetConfig` fields `release_type`, `is_multi_embodiment`, `paper_url`, `secondary_collection_methods`, `additional_modalities`, `has_synchronized_multiview_rig`, `dof_per_hand`, `hand_pose_representation`, `expected_duration_hours`, `num_subjects`, `num_scenes`, `num_objects`. Tasks 3, 4, 5 all import these.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py` (keep all existing content above; add these functions, and add `ReleaseType`, `HandPoseRepresentation`, `SensorModality` to the existing `from common.schema import ...` line so the top of the file reads `from common.schema import DatasetConfig, HandPoseRepresentation, ReleaseType, RegistryEntry, SensorModality`):

```python
def test_camera_view_new_members():
    from common.schema import CameraView

    assert CameraView.WRIST.value == "wrist"
    assert CameraView.BODY_WORN.value == "body_worn"
    assert CameraView.WORMS_EYE.value == "worms_eye"


def test_license_new_members():
    from common.schema import LicenseEnum

    assert LicenseEnum.CC_BY_NC_SA_4_0.value == "CC-BY-NC-SA-4.0"
    assert LicenseEnum.CC_BY_NC_ND_4_0.value == "CC-BY-NC-ND-4.0"


def test_robot_platform_new_members():
    from common.schema import RobotPlatform

    assert RobotPlatform.UNITREE_H1.value == "unitree_h1"
    assert RobotPlatform.FOURIER_GR1.value == "fourier_gr1"
    assert RobotPlatform.FLEXIV_RIZON4.value == "flexiv_rizon4"
    assert RobotPlatform.GALAXEA_R1_LITE.value == "galaxea_r1_lite"
    assert RobotPlatform.TOYOTA_ELEY.value == "toyota_eley"


def test_embodiment_class_quadruped():
    from common.schema import EmbodimentClass

    assert EmbodimentClass.QUADRUPED.value == "quadruped"


def test_collection_method_new_members():
    from common.schema import CollectionMethod

    assert CollectionMethod.AR_HAPTIC_GUIDED.value == "ar_haptic_guided_synthesis"
    assert CollectionMethod.SCENE_ASSET_CURATION.value == "scene_asset_curation"


def test_raw_format_vrs():
    from common.schema import RawFormat

    assert RawFormat.VRS.value == "VRS"


def test_gripper_type_three_jaw():
    from common.schema import GripperType

    assert GripperType.THREE_JAW.value == "three_jaw"


def test_action_frame_new_members():
    from common.schema import ActionFrame

    assert ActionFrame.RELATIVE_TRAJECTORY.value == "relative_trajectory"
    assert ActionFrame.MIXED_DELTA_ABSOLUTE.value == "mixed_delta_absolute"


def test_release_type_members():
    assert ReleaseType.FIXED_EPISODE_DATASET.value == "fixed_episode_dataset"
    assert ReleaseType.GENERATION_FRAMEWORK.value == "generation_framework"
    assert ReleaseType.SCENE_PLATFORM.value == "scene_platform"
    assert ReleaseType.RL_BENCHMARK_ENV.value == "rl_benchmark_env"


def test_hand_pose_representation_members():
    assert HandPoseRepresentation.MANO.value == "mano"
    assert HandPoseRepresentation.KEYPOINTS_3D.value == "keypoints_3d"
    assert HandPoseRepresentation.JOINT_ANGLES.value == "joint_angles"
    assert HandPoseRepresentation.NONE.value == "none"


def test_sensor_modality_members():
    assert SensorModality.FORCE_TORQUE.value == "force_torque"
    assert SensorModality.TACTILE.value == "tactile"
    assert SensorModality.AUDIO.value == "audio"
    assert SensorModality.EYE_GAZE.value == "eye_gaze"


def test_dataset_config_new_fields_default():
    config = DatasetConfig(id="droid", name="DROID")
    assert config.release_type is None
    assert config.is_multi_embodiment is None
    assert config.paper_url is None
    assert config.secondary_collection_methods == []
    assert config.additional_modalities == []
    assert config.has_synchronized_multiview_rig is None
    assert config.dof_per_hand is None
    assert config.hand_pose_representation is None
    assert config.expected_duration_hours is None
    assert config.num_subjects is None
    assert config.num_scenes is None
    assert config.num_objects is None


def test_dataset_config_new_fields_accept_valid_values():
    config = DatasetConfig(
        id="droid",
        name="DROID",
        release_type="generation_framework",
        is_multi_embodiment=True,
        paper_url="https://arxiv.org/abs/1234.5678",
        secondary_collection_methods=["simulation"],
        additional_modalities=["force_torque", "tactile"],
        has_synchronized_multiview_rig=True,
        dof_per_hand=16,
        hand_pose_representation="mano",
        expected_duration_hours=41.3,
        num_subjects=19,
        num_scenes=100,
        num_objects=50,
    )
    assert config.release_type.value == "generation_framework"
    assert config.secondary_collection_methods[0].value == "simulation"
    assert [m.value for m in config.additional_modalities] == [
        "force_torque",
        "tactile",
    ]
    assert config.hand_pose_representation.value == "mano"


def test_dataset_config_rejects_invalid_release_type():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", release_type="not_a_real_type")


def test_dataset_config_rejects_invalid_modality():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", additional_modalities=["smell"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'ReleaseType' from 'common.schema'`

- [ ] **Step 3: Write the implementation**

Replace the full contents of `embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py` with:

```python
"""Pydantic models for the VLA dataset registry and per-dataset onboarding
config. Field names and enum values must stay in sync with
docs/superpowers/specs/2026-07-08-vla-data-pipeline-design.md sections 5.1-5.3
and docs/superpowers/specs/2026-07-10-registry-schema-refinement-design.md
sections 3-4.
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
    VRS = "VRS"
    CUSTOM = "Custom"


class CollectionMethod(str, Enum):
    TELEOP = "teleop"
    AUTONOMOUS_POLICY = "autonomous_policy"
    UMI = "umi"
    EGOCENTRIC_HUMAN = "egocentric_human"
    SIMULATION = "simulation"
    HUMAN_TO_ROBOT_SYNTHESIS = "human_to_robot_synthesis"
    AR_HAPTIC_GUIDED = "ar_haptic_guided_synthesis"
    SCENE_ASSET_CURATION = "scene_asset_curation"


class EmbodimentClass(str, Enum):
    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"
    HUMANOID = "humanoid"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    HUMAN_HAND = "human_hand"
    QUADRUPED = "quadruped"


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
    UNITREE_H1 = "unitree_h1"
    FOURIER_GR1 = "fourier_gr1"
    FLEXIV_RIZON4 = "flexiv_rizon4"
    GALAXEA_R1_LITE = "galaxea_r1_lite"
    TOYOTA_ELEY = "toyota_eley"
    OTHER = "other"


class GripperType(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    DEXTEROUS_HAND = "dexterous_hand"
    THREE_JAW = "three_jaw"
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
    RELATIVE_TRAJECTORY = "relative_trajectory"
    MIXED_DELTA_ABSOLUTE = "mixed_delta_absolute"
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
    WRIST = "wrist"
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
    hand_pose_representation: Optional[HandPoseRepresentation] = None
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

Run: `python3 -m pytest tests/test_schema.py -v`
Expected: PASS (8 existing + 15 new = 23 total)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests, including `test_io.py`, `test_migrate_xlsx_to_registry.py`, `test_onboarding_agent.py`, `test_scaffolding.py`)

- [ ] **Step 6: Commit**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py tests/test_schema.py
git commit -m "feat: extend registry schema with enum values and fields surfaced by onboarding wave"
```

---

### Task 3: Sync onboarding agent prompt builder with new schema fields

**Files:**
- Modify: `embodied_datasets/public_datasets_raw/convert_scripts/common/onboarding_agent.py` (full-file replacement below)
- Modify: `tests/test_onboarding_agent.py` (append new test functions; existing tests must keep passing unchanged)

**Interfaces:**
- Consumes: `ReleaseType`, `HandPoseRepresentation`, `SensorModality` from `common.schema` (Task 2).
- Produces: same public functions as before (`build_onboarding_prompt`, `parse_and_validate_agent_output`) — signatures unchanged, only their internal `_ENUM_FIELDS` table and the non-enum field description text grow.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_onboarding_agent.py`:

```python
def test_build_onboarding_prompt_includes_new_enum_fields():
    prompt = build_onboarding_prompt("droid", "DROID", "https://droid-dataset.github.io/")
    assert "generation_framework" in prompt
    assert "mano" in prompt
    assert "force_torque" in prompt
    assert "three_jaw" in prompt


def test_build_onboarding_prompt_includes_new_scalar_field_descriptions():
    prompt = build_onboarding_prompt("droid", "DROID", "https://droid-dataset.github.io/")
    assert "dof_per_hand" in prompt
    assert "expected_duration_hours" in prompt
    assert "num_subjects" in prompt
    assert "is_multi_embodiment" in prompt
    assert "paper_url" in prompt


def test_parse_and_validate_agent_output_accepts_new_fields():
    yaml_text = """
id: droid
name: DROID
release_type: fixed_episode_dataset
is_multi_embodiment: false
paper_url: "https://arxiv.org/abs/1234.5678"
additional_modalities:
  - force_torque
hand_pose_representation: joint_angles
num_subjects: 5
review_status: pending_human_review
"""
    config = parse_and_validate_agent_output(yaml_text)
    assert config.release_type.value == "fixed_episode_dataset"
    assert config.additional_modalities[0].value == "force_torque"
    assert config.hand_pose_representation.value == "joint_angles"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_onboarding_agent.py -v`
Expected: FAIL — `test_build_onboarding_prompt_includes_new_enum_fields` and the two others fail because `generation_framework`/`mano`/`dof_per_hand`/etc. are not yet in the prompt output, and `release_type`/`additional_modalities`/`hand_pose_representation` are not yet accepted fields (the last assertion errors with `AttributeError: 'NoneType' object has no attribute 'value'`)

- [ ] **Step 3: Write the implementation**

Replace the full contents of `embodied_datasets/public_datasets_raw/convert_scripts/common/onboarding_agent.py` with:

```python
"""Prompt template and output validator for the dataset onboarding research
agent (design doc section 8, and the 2026-07-10 refinement section 3.4).
This module does not call any LLM -- it only (a) builds the research
prompt an Agent-tool task should receive, and (b) validates/parses
whatever YAML that task returns before it is trusted as a `DatasetConfig`.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Type

import yaml
from pydantic import ValidationError

from .schema import (
    ActionFrame,
    ActionSpace,
    CameraView,
    CollectionMethod,
    DatasetConfig,
    DepthCoverage,
    EmbodimentClass,
    GripperType,
    HandPoseRepresentation,
    LicenseEnum,
    RawFormat,
    ReleaseType,
    RobotPlatform,
    RotationRepresentation,
    SensorModality,
    UrdfSource,
)

_ENUM_FIELDS: Dict[str, Type[Enum]] = {
    "license": LicenseEnum,
    "raw_format": RawFormat,
    "release_type": ReleaseType,
    "collection_method": CollectionMethod,
    "secondary_collection_methods": CollectionMethod,
    "embodiment_class": EmbodimentClass,
    "robot_platform": RobotPlatform,
    "gripper_type": GripperType,
    "hand_pose_representation": HandPoseRepresentation,
    "action_space": ActionSpace,
    "action_frame": ActionFrame,
    "rotation_representation": RotationRepresentation,
    "depth_coverage": DepthCoverage,
    "additional_modalities": SensorModality,
    "urdf_source": UrdfSource,
    "camera_views": CameraView,
}


def _format_enum_choices(enum_cls: Type[Enum]) -> str:
    return ", ".join(member.value for member in enum_cls)


def build_onboarding_prompt(dataset_id: str, name: str, source_url: Optional[str]) -> str:
    enum_lines = "\n".join(
        f"- {field}: {_format_enum_choices(enum_cls)}"
        for field, enum_cls in _ENUM_FIELDS.items()
    )
    source_line = source_url or "(未提供，请自行搜索该数据集官网/论文)"
    return f"""\
你正在为数据集 "{name}"（id: {dataset_id}）调研结构化元数据。

可信来源（优先使用，找不到再自行搜索官网/论文）：
{source_line}

请仔细阅读官网和对应论文，为下列每个字段给出取值。每个枚举字段的值必须
严格从给定选项中选择；如果实际情况不在选项里，不要编造，而是把这个字段
留空，并在 suggested_new_enum_values 里写下建议新增的枚举值和理由。

枚举字段及可选值：
{enum_lines}

其他字段：num_arms(0/1/2的整数), dof_per_arm(整数), dof_per_hand(整数：灵巧
手/手指自由度，跟dof_per_arm是互补关系，不是替代), has_mobile_base(布尔),
state_dim/action_dim(整数), fps(数值), fps_variable(布尔),
num_camera_views(整数), has_camera_calibration(布尔),
has_synchronized_multiview_rig(布尔：多相机是否同步组成一个阵列，跟
camera_views里有哪些视角是不同维度), has_language_instruction(布尔),
num_task_types(整数), urdf_available(布尔), expected_size_gb(数值),
expected_num_episodes(整数), expected_duration_hours(数值：声明的总时长小时数),
num_subjects(整数：人类被试数量，仅适用于人类采集的数据集),
num_scenes(整数：场景/环境数量), num_objects(整数：交互物体数量),
is_multi_embodiment(布尔：是否同一份发布同时横跨多种具身形态，如果是就把
embodiment_class/robot_platform留空，不要强行选一个), paper_url(字符串：
对应论文链接，跟source_url分开存)。

对你填写的每一个字段，在 field_sources 里记录信息来源（URL 或论文章节）。

输出必须是可以直接解析为以下YAML结构的文本（不要用markdown代码块包裹），
顶层字段名必须与上面列出的字段名完全一致，另外必须包含:
id: {dataset_id}
name: {name}
review_status: pending_human_review
"""


def parse_and_validate_agent_output(yaml_text: str) -> DatasetConfig:
    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError("agent output must parse to a YAML mapping")
    try:
        return DatasetConfig(**raw)
    except ValidationError as exc:
        raise ValueError(f"agent output failed schema validation: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_onboarding_agent.py -v`
Expected: PASS (all existing + 3 new tests)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/onboarding_agent.py tests/test_onboarding_agent.py
git commit -m "feat: sync onboarding agent prompt with new schema enums and fields"
```

---

### Task 4: Backfill known corrections into the 59 existing dataset configs

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/backfill_schema_refinement.py`
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_refinement.py`
- Test: `tests/test_backfill_schema_refinement.py`
- Modify (via running the script for real, not by hand): all 59 files under `embodied_datasets/public_datasets_raw/convert_scripts/configs/*.yaml`

**Interfaces:**
- Consumes: `DatasetConfig` from `common.schema` (Task 2); `load_dataset_config`/`save_dataset_config` from `common.io` (already exists, unchanged).
- Produces: `apply_corrections(config: DatasetConfig) -> DatasetConfig` (pure function, used by the runner script below and by tests).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backfill_schema_refinement.py`:

```python
from common.backfill_schema_refinement import apply_corrections
from common.schema import DatasetConfig


def test_apply_corrections_overwrites_listed_camera_views():
    config = DatasetConfig(id="fastumi", name="FastUMI", camera_views=["other"])
    updated = apply_corrections(config)
    assert [v.value for v in updated.camera_views] == ["wrist"]


def test_apply_corrections_sets_fixed_final_camera_views_for_aloha_unleashed():
    config = DatasetConfig(
        id="aloha_unleashed", name="ALOHA Unleashed", camera_views=["top"]
    )
    updated = apply_corrections(config)
    assert [v.value for v in updated.camera_views] == [
        "top",
        "left_wrist",
        "right_wrist",
        "other",
        "worms_eye",
    ]


def test_apply_corrections_sets_multiple_fields_for_one_dataset():
    config = DatasetConfig(id="mv_umi", name="MV-UMI")
    updated = apply_corrections(config)
    assert [v.value for v in updated.camera_views] == ["third_person", "wrist"]
    assert updated.gripper_type.value == "three_jaw"
    assert updated.action_frame.value == "relative_trajectory"


def test_apply_corrections_body_worn_for_dexcap():
    config = DatasetConfig(id="dexcap", name="DexCap", camera_views=["third_person"])
    updated = apply_corrections(config)
    assert [v.value for v in updated.camera_views] == ["body_worn"]


def test_apply_corrections_sets_default_release_type():
    config = DatasetConfig(id="droid", name="DROID")
    updated = apply_corrections(config)
    assert updated.release_type.value == "fixed_episode_dataset"


def test_apply_corrections_sets_framework_release_type_and_multi_embodiment():
    config = DatasetConfig(id="robogen", name="RoboGen")
    updated = apply_corrections(config)
    assert updated.release_type.value == "generation_framework"
    assert updated.is_multi_embodiment is True


def test_apply_corrections_sets_scene_platform_release_type():
    config = DatasetConfig(id="grutopia", name="GRUtopia")
    updated = apply_corrections(config)
    assert updated.release_type.value == "scene_platform"
    assert updated.license.value == "CC-BY-NC-SA-4.0"
    assert updated.collection_method.value == "scene_asset_curation"


def test_apply_corrections_sets_rl_benchmark_release_type():
    config = DatasetConfig(id="humanoidbench", name="HumanoidBench")
    updated = apply_corrections(config)
    assert updated.release_type.value == "rl_benchmark_env"
    assert updated.robot_platform.value == "unitree_h1"
    assert updated.is_multi_embodiment is True


def test_apply_corrections_recolors_arcap_collection_method():
    config = DatasetConfig(id="arcap", name="ARCap", collection_method="umi")
    updated = apply_corrections(config)
    assert updated.collection_method.value == "ar_haptic_guided_synthesis"


def test_apply_corrections_preserves_fields_not_in_table():
    config = DatasetConfig(
        id="droid", name="DROID", source_url="https://a", license="MIT"
    )
    updated = apply_corrections(config)
    assert updated.source_url == "https://a"
    assert updated.license.value == "MIT"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_backfill_schema_refinement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.backfill_schema_refinement'`

- [ ] **Step 3: Write the implementation**

Create `embodied_datasets/public_datasets_raw/convert_scripts/common/backfill_schema_refinement.py`:

```python
"""Table-driven corrections applied to the 59 already-onboarded dataset
configs during the 2026-07-10 registry schema refinement (see
docs/superpowers/specs/2026-07-10-registry-schema-refinement-design.md
section 4). Every value below is already justified in the corresponding
dataset's field_sources/suggested_new_enum_values from its onboarding
report -- this module applies known corrections, it does not research
new ones.
"""
from __future__ import annotations

from typing import Any, Dict

from .schema import DatasetConfig

# Design doc section 4, table A: specific field-value corrections.
FIELD_CORRECTIONS: Dict[str, Dict[str, Any]] = {
    "functional_manipulation_benchmark_fmb": {
        "camera_views": ["side", "wrist"],
    },
    "furniturebench": {
        "camera_views": ["front", "wrist"],
    },
    "the_colosseum": {
        "camera_views": ["front", "top", "third_person", "wrist"],
    },
    "gensim2": {
        "camera_views": ["front", "side", "wrist"],
    },
    "fastumi": {
        "camera_views": ["wrist"],
    },
    "mv_umi": {
        "camera_views": ["third_person", "wrist"],
        "gripper_type": "three_jaw",
        "action_frame": "relative_trajectory",
    },
    "omniumi": {
        "camera_views": ["wrist"],
        "action_frame": "mixed_delta_absolute",
    },
    "dexcap": {
        "camera_views": ["body_worn"],
    },
    "aloha_unleashed": {
        "camera_views": ["top", "left_wrist", "right_wrist", "other", "worms_eye"],
    },
    "galaxea_open_world_dataset": {
        "license": "CC-BY-NC-SA-4.0",
    },
    "grutopia": {
        "license": "CC-BY-NC-SA-4.0",
        "collection_method": "scene_asset_curation",
    },
    "airexo_2": {
        "license": "CC-BY-NC-SA-4.0",
        "robot_platform": "flexiv_rizon4",
    },
    "yubi": {
        "license": "CC-BY-NC-SA-4.0",
        "robot_platform": "toyota_eley",
        "is_multi_embodiment": True,
    },
    "egodex": {
        "license": "CC-BY-NC-ND-4.0",
    },
    "bigym": {
        "robot_platform": "unitree_h1",
    },
    "humanoidbench": {
        "robot_platform": "unitree_h1",
        "is_multi_embodiment": True,
    },
    "arcap": {
        "collection_method": "ar_haptic_guided_synthesis",
    },
    "hot3d": {
        "raw_format": "VRS",
    },
    "assembly101": {
        "has_synchronized_multiview_rig": True,
    },
    "robogen": {
        "is_multi_embodiment": True,
    },
}

# Design doc section 4, table B: release_type classification. Any dataset
# id not listed here defaults to "fixed_episode_dataset" -- see
# apply_corrections().
RELEASE_TYPE_OVERRIDES: Dict[str, str] = {
    "robogen": "generation_framework",
    "gensim2": "generation_framework",
    "grutopia": "scene_platform",
    "humanoidbench": "rl_benchmark_env",
}


def apply_corrections(config: DatasetConfig) -> DatasetConfig:
    """Return a new DatasetConfig with this dataset's table-driven
    corrections applied. Fields not mentioned in FIELD_CORRECTIONS for
    this dataset id are left untouched; release_type is always set
    (falling back to "fixed_episode_dataset")."""
    data = config.model_dump(mode="json", exclude_none=True)
    data.update(FIELD_CORRECTIONS.get(config.id, {}))
    data["release_type"] = RELEASE_TYPE_OVERRIDES.get(config.id, "fixed_episode_dataset")
    return DatasetConfig(**data)
```

Create `embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_refinement.py`:

```python
"""One-off script: apply the 2026-07-10 registry schema refinement's
table-driven corrections (common/backfill_schema_refinement.py) to every
existing dataset config.

Run from repo root:
    python3 embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_refinement.py
"""
from __future__ import annotations

from pathlib import Path

from common.backfill_schema_refinement import apply_corrections
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
        updated = apply_corrections(config)
        save_dataset_config(updated, path)
    print(f"backfilled {len(paths)} dataset configs in {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_backfill_schema_refinement.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 6: Run the backfill script for real against the actual 59 configs**

Run from repo root: `python3 embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_refinement.py`
Expected output: `backfilled 59 dataset configs in .../configs`

- [ ] **Step 7: Spot-check the real output**

Run: `grep -A2 "^camera_views" embodied_datasets/public_datasets_raw/convert_scripts/configs/fastumi.yaml` — expect `- wrist` as the only entry.
Run: `grep "^release_type" embodied_datasets/public_datasets_raw/convert_scripts/configs/robogen.yaml` — expect `release_type: generation_framework`.
Run: `grep "^release_type" embodied_datasets/public_datasets_raw/convert_scripts/configs/droid.yaml` — expect `release_type: fixed_episode_dataset`.

- [ ] **Step 8: Commit**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/backfill_schema_refinement.py \
        embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_refinement.py \
        tests/test_backfill_schema_refinement.py \
        embodied_datasets/datasets_registry.yaml \
        embodied_datasets/public_datasets_raw/convert_scripts/configs/
git commit -m "feat: backfill known schema corrections and release_type across all 59 dataset configs"
```

---

### Task 5: Overview table generator

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/generate_overview_readme.py`
- Test: `tests/test_generate_overview_readme.py`

**Interfaces:**
- Consumes: `RegistryEntry`, `DatasetConfig` from `common.schema`; `load_registry`, `load_dataset_config` from `common.io` (both already exist, unchanged).
- Produces: `render_overview_table(entries: List[RegistryEntry], configs_by_id: Dict[str, DatasetConfig]) -> str`, `replace_marked_section(readme_text: str, table_markdown: str) -> str`, module-level constants `TABLE_START_MARKER = "<!-- AUTO-GENERATED TABLE START -->"` and `TABLE_END_MARKER = "<!-- AUTO-GENERATED TABLE END -->"`. Task 6 uses these exact marker strings in the README and runs this module's `main()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_overview_readme.py`:

```python
from common.generate_overview_readme import (
    TABLE_END_MARKER,
    TABLE_START_MARKER,
    render_overview_table,
    replace_marked_section,
)
from common.schema import DatasetConfig, RegistryEntry


def test_render_overview_table_sorts_by_priority_then_name():
    entries = [
        RegistryEntry(id="b", name="Bravo", priority="P1"),
        RegistryEntry(id="a", name="Alpha", priority="P0"),
        RegistryEntry(id="c", name="Charlie", priority="P0"),
    ]
    table = render_overview_table(entries, {})
    data_lines = table.splitlines()[2:]
    ids_in_order = [line.split("|")[1].strip() for line in data_lines]
    assert ids_in_order == ["a", "c", "b"]


def test_render_overview_table_includes_config_fields():
    entries = [RegistryEntry(id="droid", name="DROID")]
    configs = {
        "droid": DatasetConfig(
            id="droid",
            name="DROID",
            collection_method="teleop",
            embodiment_class="single_arm",
        )
    }
    table = render_overview_table(entries, configs)
    assert "teleop" in table
    assert "single_arm" in table


def test_render_overview_table_handles_missing_config():
    entries = [RegistryEntry(id="droid", name="DROID")]
    table = render_overview_table(entries, {})
    assert "droid" in table


def test_replace_marked_section_only_touches_between_markers():
    readme_text = (
        f"# Title\n\nIntro text.\n\n"
        f"{TABLE_START_MARKER}\nold table\n{TABLE_END_MARKER}\n\n"
        f"Footer text.\n"
    )
    updated = replace_marked_section(readme_text, "new table")
    assert "Intro text." in updated
    assert "Footer text." in updated
    assert "old table" not in updated
    assert "new table" in updated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generate_overview_readme.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.generate_overview_readme'`

- [ ] **Step 3: Write the implementation**

Create `embodied_datasets/public_datasets_raw/convert_scripts/common/generate_overview_readme.py`:

```python
"""Render the auto-generated dataset overview table for
embodied_datasets/README.md (see
docs/superpowers/specs/2026-07-10-registry-schema-refinement-design.md
section 5). Reads datasets_registry.yaml + convert_scripts/configs/*.yaml
and replaces only the content between the AUTO-GENERATED TABLE markers in
the README -- everything else in the file is hand-written and left as-is.

Run from repo root:
    python3 embodied_datasets/public_datasets_raw/convert_scripts/common/generate_overview_readme.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .io import load_dataset_config, load_registry
from .schema import DatasetConfig, RegistryEntry

TABLE_START_MARKER = "<!-- AUTO-GENERATED TABLE START -->"
TABLE_END_MARKER = "<!-- AUTO-GENERATED TABLE END -->"

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}

_COLUMNS = [
    "id",
    "name",
    "priority",
    "download_status",
    "convert_status",
    "process_status",
    "review_status",
    "collection_method",
    "embodiment_class",
]


def render_overview_table(
    entries: List[RegistryEntry], configs_by_id: Dict[str, DatasetConfig]
) -> str:
    sorted_entries = sorted(
        entries, key=lambda e: (_PRIORITY_ORDER[e.priority.value], e.name)
    )
    lines = [
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "---|" * len(_COLUMNS),
    ]
    for entry in sorted_entries:
        config = configs_by_id.get(entry.id)
        review_status = config.review_status.value if config else ""
        collection_method = (
            config.collection_method.value
            if config and config.collection_method
            else ""
        )
        embodiment_class = (
            config.embodiment_class.value
            if config and config.embodiment_class
            else ""
        )
        row = [
            entry.id,
            entry.name,
            entry.priority.value,
            entry.download_status.value,
            entry.convert_status.value,
            entry.process_status.value,
            review_status,
            collection_method,
            embodiment_class,
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def replace_marked_section(readme_text: str, table_markdown: str) -> str:
    start = readme_text.index(TABLE_START_MARKER) + len(TABLE_START_MARKER)
    end = readme_text.index(TABLE_END_MARKER)
    if end < start:
        raise ValueError("TABLE_END_MARKER appears before TABLE_START_MARKER")
    return readme_text[:start] + "\n\n" + table_markdown + "\n\n" + readme_text[end:]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    embodied_root = repo_root / "embodied_datasets"
    registry_path = embodied_root / "datasets_registry.yaml"
    configs_dir = embodied_root / "public_datasets_raw" / "convert_scripts" / "configs"
    readme_path = embodied_root / "README.md"

    entries = load_registry(registry_path)
    configs_by_id = {
        path.stem: load_dataset_config(path)
        for path in sorted(configs_dir.glob("*.yaml"))
    }
    table_markdown = render_overview_table(entries, configs_by_id)
    readme_text = readme_path.read_text(encoding="utf-8")
    updated_text = replace_marked_section(readme_text, table_markdown)
    readme_path.write_text(updated_text, encoding="utf-8")
    print(f"refreshed overview table for {len(entries)} datasets in {readme_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generate_overview_readme.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/generate_overview_readme.py tests/test_generate_overview_readme.py
git commit -m "feat: add embodied_datasets overview table generator"
```

---

### Task 6: Write embodied_datasets/README.md and generate the first table

**Files:**
- Create: `embodied_datasets/README.md`

**Interfaces:**
- Consumes: `TABLE_START_MARKER`, `TABLE_END_MARKER`, `main()` from Task 5's `common/generate_overview_readme.py`.
- Produces: nothing further consumed by other tasks — this is the final task in the plan.

- [ ] **Step 1: Write the hand-written README content**

Create `embodied_datasets/README.md`:

```markdown
# embodied_datasets

VLA（视觉-语言-动作）机器人操作数据集的统一注册、清洗、对齐流水线。所有数据集
先转换为 LeRobot v2.1 格式，再按论文方法论做五阶段数值清洗、三项跨模态质检和
跨本体维度统一。完整设计见：

- [2026-07-08 数据预处理与对齐流水线设计文档](../docs/superpowers/specs/2026-07-08-vla-data-pipeline-design.md)
- [2026-07-10 注册表 Schema 补全与数据根目录设计文档](../docs/superpowers/specs/2026-07-10-registry-schema-refinement-design.md)

## 目录结构

```
embodied_datasets/
├── datasets_registry.yaml          # 59个数据集的总览表（实测值，随流水线推进更新）
├── public_datasets_raw/
│   ├── <dataset_id>/raw/                    # 原始下载数据（重数据，见下方"数据根目录"）
│   ├── <dataset_id>/lerobot_v2_1_staging/   # 转换后未清洗的中间态（重数据）
│   ├── convert_scripts/
│   │   ├── configs/<dataset_id>.yaml   # 每个数据集的调研配置（声明值）
│   │   └── common/                     # 复用的 schema/io/onboarding 工具
│   ├── verify_scripts/             # 完整性校验（Plan B，未实现）
│   └── process_scripts/            # 清洗对齐流水线（Plan D，未实现）
├── urdf_assets/<robot_platform>/   # 按机器人型号共享的 URDF（重数据）
└── public_datasets/
    └── lerobot_v2_1/<dataset_id>/  # 清洗完成的最终数据（重数据）
```

## 数据根目录

`datasets_registry.yaml`、`convert_scripts/configs/*.yaml` 和所有脚本代码始终
留在仓库内，不受下面这条配置影响。只有实际的重数据目录
（`raw/`、`lerobot_v2_1_staging/`、`public_datasets/lerobot_v2_1/`、
`urdf_assets/`）可以指向仓库外任意路径，未来所有读写这些目录的脚本都会接受
一个 `--data-root` 参数：

```bash
python3 some_future_script.py --data-root /mnt/big_disk/vla_data
```

不传 `--data-root` 时默认使用仓库内的 `embodied_datasets/`。路径解析逻辑见
`public_datasets_raw/convert_scripts/common/paths.py`。

## 字段含义速查

`datasets_registry.yaml` 是"实测值"总览表（下载/转换/清洗进度），
`convert_scripts/configs/<id>.yaml` 是每个数据集的"声明值"详细配置（调研得到
的本体信息、数据表示方式等）。完整字段列表和取值范围见
`public_datasets_raw/convert_scripts/common/schema.py` 里的 pydantic 模型，
以及上面两份设计文档的字段表。

## 如何 onboard 新数据集

1. 在 `datasets_registry.yaml` 里加一条 `RegistryEntry`，在 `configs/` 下建一个
   同 id 的 stub `DatasetConfig`（只填 `id`/`name`/`source_url`）。
2. 用 `common/onboarding_agent.py` 的 `build_onboarding_prompt()` 生成调研任务
   的提示词，派给一个 Agent 去读官网/论文并填字段。
3. 用 `parse_and_validate_agent_output()` 校验 Agent 产出的 YAML 能通过
   schema 校验，写回 `configs/<id>.yaml`，`review_status` 保持
   `pending_human_review` 直到人工确认。

## 当前进度

<!-- AUTO-GENERATED TABLE START -->
<!-- AUTO-GENERATED TABLE END -->

上表由 `python3 public_datasets_raw/convert_scripts/common/generate_overview_readme.py`
生成，只更新 marker 之间的内容；手动新增数据集或更新状态后重新运行以刷新。
```

- [ ] **Step 2: Run the generator for real to fill in the table**

Run from repo root: `python3 embodied_datasets/public_datasets_raw/convert_scripts/common/generate_overview_readme.py`
Expected output: `refreshed overview table for 59 datasets in .../embodied_datasets/README.md`

- [ ] **Step 3: Verify the table landed correctly**

Run: `grep -c "^|" embodied_datasets/README.md` — expect 61 (1 header + 1 separator + 59 data rows).
Run: `head -20 embodied_datasets/README.md` and visually confirm the hand-written sections above the table are untouched.

- [ ] **Step 4: Run the full test suite one last time**

Run: `python3 -m pytest tests/ -v`
Expected: PASS (all tests, no regressions from any earlier task)

- [ ] **Step 5: Commit**

```bash
git add embodied_datasets/README.md
git commit -m "docs: add embodied_datasets top-level README with auto-generated overview table"
```
