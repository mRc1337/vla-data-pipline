# VLA 数据流水线 · 计划A（基础骨架）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 `embodied_datasets/` 目录骨架，把 `VLA Research.xlsx` 迁移为结构化、枚举校验的
`datasets_registry.yaml` + 每数据集配置 stub，并产出数据集 Onboarding Agent 所需的
prompt 构建与输出校验逻辑，供后续人工/agent 批量调研 33 个数据集时使用。

**Architecture:** 一个纯 Python 小型内部工具库 `common/`（`schema.py` 定义 pydantic 模型与
枚举，`io.py` 负责 YAML 读写，`migrate_xlsx_to_registry.py` 负责一次性迁移，
`onboarding_agent.py` 负责 prompt 构建与产出校验），配合一个一次性 CLI 脚本
`run_migration.py` 把真实的 `VLA Research.xlsx` 转换为仓库里的真实数据文件。

**Tech Stack:** Python 3.9 兼容语法、pydantic 2.x（枚举/校验）、PyYAML（YAML 读写）、
openpyxl（读取 xlsx）、pytest（测试）。

## Global Constraints

- Python 3.9 兼容语法（不使用 `X | Y` 联合类型的运行期求值、不使用 `match` 语句）；
  目标解释器 `/usr/bin/python3`（3.9.6）。
- 依赖版本锁定在 `requirements.txt`：`pydantic==2.10.3`、`PyYAML==6.0.2`、
  `openpyxl==3.1.5`、`pytest==8.3.4`。
- 所有枚举字段必须使用 `common/schema.py` 中定义的 `Enum` 类；模型开启
  `extra="forbid"`，未知取值或未知字段必须校验失败，不允许静默接受或强制转换
  （对应设计文档"完善字段，最好每个字段都是枚举或者纯数字"的决定）。
- 本计划只把 `VLA Research.xlsx` 里的"名称"与"官方下载链接"两列迁移进
  `source_url`；其余字段留空由后续 Onboarding Agent 调研阶段填充——本计划只产出
  prompt 构建函数与输出校验函数，**不在计划范围内实际批量调用 agent 调研 33 个
  数据集**（那是本计划完成后的下一步操作）。
- 目录结构、字段名称必须与 `docs/superpowers/specs/2026-07-08-vla-data-pipeline-design.md`
  第 3、5 节完全一致。

---

### Task 1: 目录骨架与项目配置

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/configs/.gitkeep`
- Create: `embodied_datasets/public_datasets_raw/process_scripts/configs/.gitkeep`
- Create: `embodied_datasets/public_datasets_raw/verify_scripts/logs/.gitkeep`
- Create: `embodied_datasets/urdf_assets/.gitkeep`
- Create: `embodied_datasets/public_datasets/lerobot_v2_1/.gitkeep`
- Create: `embodied_datasets/public_datasets/lerobot_v3_0/.gitkeep`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Consumes: 无（第一个任务）
- Produces: 目录树（见下方路径列表）；`pytest` 的 `pythonpath`/`testpaths` 配置，供后续
  所有任务的测试直接 `from common.xxx import ...`。

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_scaffolding.py`：

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "embodied_datasets/public_datasets_raw/convert_scripts/configs",
    "embodied_datasets/public_datasets_raw/process_scripts/configs",
    "embodied_datasets/public_datasets_raw/verify_scripts/logs",
    "embodied_datasets/urdf_assets",
    "embodied_datasets/public_datasets/lerobot_v2_1",
    "embodied_datasets/public_datasets/lerobot_v3_0",
]


def test_expected_directories_exist():
    for rel_dir in EXPECTED_DIRS:
        assert (REPO_ROOT / rel_dir).is_dir(), f"missing directory: {rel_dir}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/haoyuefan/Documents/github/vla_data_pipeline && python3 -m pytest tests/test_scaffolding.py -v`
Expected: FAIL（`embodied_datasets` 目录尚不存在）

- [ ] **Step 3: 创建目录骨架与项目配置文件**

```bash
mkdir -p embodied_datasets/public_datasets_raw/convert_scripts/configs
mkdir -p embodied_datasets/public_datasets_raw/convert_scripts/common
mkdir -p embodied_datasets/public_datasets_raw/process_scripts/configs
mkdir -p embodied_datasets/public_datasets_raw/verify_scripts/logs
mkdir -p embodied_datasets/urdf_assets
mkdir -p embodied_datasets/public_datasets/lerobot_v2_1
mkdir -p embodied_datasets/public_datasets/lerobot_v3_0
touch embodied_datasets/public_datasets_raw/convert_scripts/configs/.gitkeep
touch embodied_datasets/public_datasets_raw/process_scripts/configs/.gitkeep
touch embodied_datasets/public_datasets_raw/verify_scripts/logs/.gitkeep
touch embodied_datasets/urdf_assets/.gitkeep
touch embodied_datasets/public_datasets/lerobot_v2_1/.gitkeep
touch embodied_datasets/public_datasets/lerobot_v3_0/.gitkeep
```

创建 `requirements.txt`：

```
pydantic==2.10.3
PyYAML==6.0.2
openpyxl==3.1.5
pytest==8.3.4
```

创建 `pyproject.toml`：

```toml
[tool.pytest.ini_options]
pythonpath = ["embodied_datasets/public_datasets_raw/convert_scripts"]
testpaths = ["tests"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python3 -m pytest tests/test_scaffolding.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add requirements.txt pyproject.toml embodied_datasets tests/test_scaffolding.py
git commit -m "chore: scaffold embodied_datasets directory tree and pytest config"
```

---

### Task 2: 注册表与配置的数据模型（`common/schema.py`）

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/__init__.py`
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: Task1 的目录骨架与 pytest 配置
- Produces（供 Task3/4/5/6 使用）：
  - `common.schema.RegistryEntry`（字段：`id, name, priority, download_status,
    integrity_status, convert_status, process_status, raw_local_path,
    lerobot_v2_1_local_path, storage_size_gb, num_episodes, num_frames,
    duration_hours`）
  - `common.schema.DatasetConfig`（字段见设计文档 5.2，另加
    `field_sources: Dict[str,str]`、`suggested_new_enum_values: Dict[str,str]`）
  - 枚举类：`Priority, DownloadStatus, IntegrityStatus, ConvertStatus,
    ProcessStatus, LicenseEnum, RawFormat, CollectionMethod, EmbodimentClass,
    RobotPlatform, GripperType, ActionSpace, ActionFrame,
    RotationRepresentation, CameraView, DepthCoverage, UrdfSource,
    ReviewStatus`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_schema.py`：

```python
import pytest
from pydantic import ValidationError

from common.schema import DatasetConfig, RegistryEntry


def test_registry_entry_defaults():
    entry = RegistryEntry(id="droid", name="DROID")
    assert entry.priority.value == "P2"
    assert entry.download_status.value == "not_downloaded"
    assert entry.integrity_status.value == "not_verified"
    assert entry.convert_status.value == "not_converted"
    assert entry.process_status.value == "not_processed"


def test_registry_entry_rejects_invalid_enum():
    with pytest.raises(ValidationError):
        RegistryEntry(id="droid", name="DROID", download_status="downloaded_maybe")


def test_registry_entry_rejects_unknown_field():
    with pytest.raises(ValidationError):
        RegistryEntry(id="droid", name="DROID", not_a_real_field=1)


def test_dataset_config_defaults():
    config = DatasetConfig(id="droid", name="DROID")
    assert config.review_status.value == "pending_human_review"
    assert config.camera_views == []
    assert config.license is None
    assert config.field_sources == {}


def test_dataset_config_rejects_unknown_robot_platform():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", robot_platform="made_up_robot")


def test_dataset_config_accepts_known_robot_platform():
    config = DatasetConfig(id="droid", name="DROID", robot_platform="franka_panda")
    assert config.robot_platform.value == "franka_panda"


def test_dataset_config_num_arms_bounds():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", num_arms=3)


def test_dataset_config_camera_views_enum_checked():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", camera_views=["bird_eye_view"])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python3 -m pytest tests/test_schema.py -v`
Expected: FAIL（`common` 模块不存在）

- [ ] **Step 3: 实现 schema.py**

创建 `embodied_datasets/public_datasets_raw/convert_scripts/common/__init__.py`（空文件）。

创建 `embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python3 -m pytest tests/test_schema.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: 提交**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/__init__.py \
        embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py \
        tests/test_schema.py
git commit -m "feat: add registry and dataset config pydantic schema"
```

---

### Task 3: 注册表/配置的 YAML 读写（`common/io.py`）

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/io.py`
- Test: `tests/test_io.py`

**Interfaces:**
- Consumes: `common.schema.RegistryEntry`, `common.schema.DatasetConfig`（Task2）
- Produces（供 Task6 使用）：
  - `common.io.load_registry(path: Path) -> List[RegistryEntry]`
  - `common.io.save_registry(entries: List[RegistryEntry], path: Path) -> None`
  - `common.io.load_dataset_config(path: Path) -> DatasetConfig`
  - `common.io.save_dataset_config(config: DatasetConfig, path: Path) -> None`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_io.py`：

```python
from common.io import (
    load_dataset_config,
    load_registry,
    save_dataset_config,
    save_registry,
)
from common.schema import DatasetConfig, RegistryEntry


def test_registry_round_trip(tmp_path):
    path = tmp_path / "registry.yaml"
    entries = [
        RegistryEntry(id="droid", name="DROID"),
        RegistryEntry(id="bridgedata_v2", name="BridgeData V2"),
    ]
    save_registry(entries, path)
    loaded = load_registry(path)
    assert [e.id for e in loaded] == ["droid", "bridgedata_v2"]
    assert loaded[0].download_status.value == "not_downloaded"


def test_load_registry_missing_file_returns_empty(tmp_path):
    assert load_registry(tmp_path / "missing.yaml") == []


def test_dataset_config_round_trip(tmp_path):
    path = tmp_path / "droid.yaml"
    config = DatasetConfig(
        id="droid", name="DROID", source_url="https://a", license="MIT"
    )
    save_dataset_config(config, path)
    loaded = load_dataset_config(path)
    assert loaded.id == "droid"
    assert loaded.source_url == "https://a"
    assert loaded.license.value == "MIT"
    assert loaded.review_status.value == "pending_human_review"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python3 -m pytest tests/test_io.py -v`
Expected: FAIL（`common.io` 模块不存在）

- [ ] **Step 3: 实现 io.py**

创建 `embodied_datasets/public_datasets_raw/convert_scripts/common/io.py`：

```python
"""Load/save the dataset registry overview table and per-dataset onboarding
configs as YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from .schema import DatasetConfig, RegistryEntry


def load_registry(path: Path) -> List[RegistryEntry]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [RegistryEntry(**entry) for entry in raw]


def save_registry(entries: List[RegistryEntry], path: Path) -> None:
    data = [entry.model_dump(mode="json", exclude_none=True) for entry in entries]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def load_dataset_config(path: Path) -> DatasetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetConfig(**raw)


def save_dataset_config(config: DatasetConfig, path: Path) -> None:
    data = config.model_dump(mode="json", exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python3 -m pytest tests/test_io.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/io.py tests/test_io.py
git commit -m "feat: add YAML round-trip for registry entries and dataset configs"
```

---

### Task 4: xlsx 迁移脚本（`common/migrate_xlsx_to_registry.py`）

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/migrate_xlsx_to_registry.py`
- Test: `tests/test_migrate_xlsx_to_registry.py`

**Interfaces:**
- Consumes: `common.schema.RegistryEntry`, `common.schema.DatasetConfig`（Task2）
- Produces（供 Task6 使用）：
  - `common.migrate_xlsx_to_registry.slugify(name: str) -> str`
  - `common.migrate_xlsx_to_registry.read_name_and_link_rows(xlsx_path: Path,
    sheet_name: str = "VLA公开数据集") -> List[Tuple[str, Optional[str]]]`
  - `common.migrate_xlsx_to_registry.build_registry_and_configs(rows:
    List[Tuple[str, Optional[str]]]) -> Tuple[List[RegistryEntry], List[DatasetConfig]]`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_migrate_xlsx_to_registry.py`：

```python
import openpyxl
import pytest

from common.migrate_xlsx_to_registry import (
    build_registry_and_configs,
    read_name_and_link_rows,
    slugify,
)


@pytest.fixture
def sample_xlsx(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "VLA公开数据集"
    ws.append(["序号", "名称", "数据类型", "官方下载链接"])
    ws.append([1, "DROID", None, "https://droid-dataset.github.io/"])
    ws.append([2, "BridgeData V2", None, None])
    ws.append([3, "Open X-Embodiment", None, "https://example.com/oxe"])
    path = tmp_path / "sample.xlsx"
    wb.save(path)
    return path


def test_slugify_basic():
    assert slugify("DROID") == "droid"
    assert slugify("BridgeData V2") == "bridgedata_v2"
    assert slugify("Open X-Embodiment") == "open_x_embodiment"
    assert slugify("lerobot/ull_folding") == "lerobot_ull_folding"


def test_slugify_rejects_empty():
    with pytest.raises(ValueError):
        slugify("   ")


def test_read_name_and_link_rows(sample_xlsx):
    rows = read_name_and_link_rows(sample_xlsx)
    assert rows == [
        ("DROID", "https://droid-dataset.github.io/"),
        ("BridgeData V2", None),
        ("Open X-Embodiment", "https://example.com/oxe"),
    ]


def test_build_registry_and_configs(sample_xlsx):
    rows = read_name_and_link_rows(sample_xlsx)
    entries, configs = build_registry_and_configs(rows)
    assert [e.id for e in entries] == ["droid", "bridgedata_v2", "open_x_embodiment"]
    assert entries[0].name == "DROID"
    assert entries[0].download_status.value == "not_downloaded"
    assert configs[0].source_url == "https://droid-dataset.github.io/"
    assert configs[1].source_url is None
    assert configs[0].review_status.value == "pending_human_review"


def test_build_registry_and_configs_rejects_duplicate_ids():
    rows = [("DROID", "https://a"), ("droid", "https://b")]
    with pytest.raises(ValueError):
        build_registry_and_configs(rows)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python3 -m pytest tests/test_migrate_xlsx_to_registry.py -v`
Expected: FAIL（`common.migrate_xlsx_to_registry` 模块不存在）

- [ ] **Step 3: 实现 migrate_xlsx_to_registry.py**

创建 `embodied_datasets/public_datasets_raw/convert_scripts/common/migrate_xlsx_to_registry.py`：

```python
"""Migrate the legacy VLA Research.xlsx into datasets_registry.yaml and
per-dataset config stubs.

Per the 2026-07-08 decision, only two columns are trusted from the
spreadsheet: 名称 (name) and 官方下载链接 (official download link). Every
other column in the spreadsheet is sparse/inconsistent and is intentionally
dropped -- the rest of each DatasetConfig is filled in later by the
onboarding agent (see onboarding_agent.py).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import openpyxl

from .schema import DatasetConfig, RegistryEntry


def slugify(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    if not slug:
        raise ValueError(f"cannot derive a slug from name: {name!r}")
    return slug


def read_name_and_link_rows(
    xlsx_path: Path, sheet_name: str = "VLA公开数据集"
) -> List[Tuple[str, Optional[str]]]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    name_col = header.index("名称")
    link_col = header.index("官方下载链接")
    results: List[Tuple[str, Optional[str]]] = []
    for row in rows[1:]:
        if name_col >= len(row) or row[name_col] is None:
            continue
        name = str(row[name_col]).strip()
        if not name:
            continue
        link = row[link_col] if link_col < len(row) else None
        link = str(link).strip() if link else None
        results.append((name, link))
    return results


def build_registry_and_configs(
    rows: List[Tuple[str, Optional[str]]]
) -> Tuple[List[RegistryEntry], List[DatasetConfig]]:
    entries: List[RegistryEntry] = []
    configs: List[DatasetConfig] = []
    seen_ids = set()
    for name, link in rows:
        dataset_id = slugify(name)
        if dataset_id in seen_ids:
            raise ValueError(
                f"duplicate dataset id derived from name: {dataset_id!r} (name={name!r})"
            )
        seen_ids.add(dataset_id)
        entries.append(RegistryEntry(id=dataset_id, name=name))
        configs.append(DatasetConfig(id=dataset_id, name=name, source_url=link))
    return entries, configs
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python3 -m pytest tests/test_migrate_xlsx_to_registry.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/migrate_xlsx_to_registry.py \
        tests/test_migrate_xlsx_to_registry.py
git commit -m "feat: migrate xlsx name+official-link columns into registry/config builders"
```

---

### Task 5: Onboarding Agent 的 prompt 构建与产出校验（`common/onboarding_agent.py`）

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/common/onboarding_agent.py`
- Test: `tests/test_onboarding_agent.py`

**Interfaces:**
- Consumes: `common.schema.DatasetConfig` 及其全部枚举类（Task2）
- Produces（供未来批量调研执行时使用，本计划不调用）：
  - `common.onboarding_agent.build_onboarding_prompt(dataset_id: str, name: str,
    source_url: Optional[str]) -> str`
  - `common.onboarding_agent.parse_and_validate_agent_output(yaml_text: str) -> DatasetConfig`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_onboarding_agent.py`：

```python
import pytest

from common.onboarding_agent import (
    build_onboarding_prompt,
    parse_and_validate_agent_output,
)


def test_build_onboarding_prompt_includes_key_info():
    prompt = build_onboarding_prompt("droid", "DROID", "https://droid-dataset.github.io/")
    assert "droid" in prompt
    assert "DROID" in prompt
    assert "https://droid-dataset.github.io/" in prompt
    assert "franka_panda" in prompt
    assert "Apache-2.0" in prompt


def test_build_onboarding_prompt_handles_missing_source():
    prompt = build_onboarding_prompt("droid", "DROID", None)
    assert "自行搜索" in prompt


def test_parse_and_validate_agent_output_valid():
    yaml_text = """
id: droid
name: DROID
license: MIT
robot_platform: franka_panda
num_arms: 1
camera_views:
  - third_person
  - right_wrist
review_status: pending_human_review
field_sources:
  license: "https://droid-dataset.github.io/ - footer"
"""
    config = parse_and_validate_agent_output(yaml_text)
    assert config.id == "droid"
    assert config.license.value == "MIT"
    assert config.robot_platform.value == "franka_panda"
    assert config.camera_views[0].value == "third_person"


def test_parse_and_validate_agent_output_rejects_unknown_enum():
    yaml_text = """
id: droid
name: DROID
robot_platform: made_up_robot
"""
    with pytest.raises(ValueError):
        parse_and_validate_agent_output(yaml_text)


def test_parse_and_validate_agent_output_rejects_non_mapping():
    with pytest.raises(ValueError):
        parse_and_validate_agent_output("- just\n- a\n- list\n")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python3 -m pytest tests/test_onboarding_agent.py -v`
Expected: FAIL（`common.onboarding_agent` 模块不存在）

- [ ] **Step 3: 实现 onboarding_agent.py**

创建 `embodied_datasets/public_datasets_raw/convert_scripts/common/onboarding_agent.py`：

```python
"""Prompt template and output validator for the dataset onboarding research
agent (design doc section 8). This module does not call any LLM -- it only
(a) builds the research prompt an Agent-tool task should receive, and
(b) validates/parses whatever YAML that task returns before it is trusted
as a `DatasetConfig`.
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
    LicenseEnum,
    RawFormat,
    RobotPlatform,
    RotationRepresentation,
    UrdfSource,
)

_ENUM_FIELDS: Dict[str, Type[Enum]] = {
    "license": LicenseEnum,
    "raw_format": RawFormat,
    "collection_method": CollectionMethod,
    "embodiment_class": EmbodimentClass,
    "robot_platform": RobotPlatform,
    "gripper_type": GripperType,
    "action_space": ActionSpace,
    "action_frame": ActionFrame,
    "rotation_representation": RotationRepresentation,
    "depth_coverage": DepthCoverage,
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

其他字段：num_arms(0/1/2的整数), dof_per_arm(整数), has_mobile_base(布尔),
state_dim/action_dim(整数), fps(数值), fps_variable(布尔),
num_camera_views(整数), has_camera_calibration(布尔),
has_language_instruction(布尔), num_task_types(整数), urdf_available(布尔),
expected_size_gb(数值), expected_num_episodes(整数)。

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

- [ ] **Step 4: 运行测试确认通过**

Run: `python3 -m pytest tests/test_onboarding_agent.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/common/onboarding_agent.py \
        tests/test_onboarding_agent.py
git commit -m "feat: add onboarding agent prompt builder and output validator"
```

---

### Task 6: 真实迁移 —— 生成 `datasets_registry.yaml` 与 33 份配置 stub

**Files:**
- Create: `embodied_datasets/public_datasets_raw/convert_scripts/run_migration.py`
- Modify（生成，非手写）: `embodied_datasets/datasets_registry.yaml`
- Modify（生成，非手写）: `embodied_datasets/public_datasets_raw/convert_scripts/configs/*.yaml`（33 个文件）

**Interfaces:**
- Consumes: `common.migrate_xlsx_to_registry.read_name_and_link_rows/build_registry_and_configs`
  （Task4），`common.io.save_registry/save_dataset_config`（Task3）
- Produces: 仓库中真实的 `datasets_registry.yaml` 和 33 个 `configs/<id>.yaml`
  stub 文件，供后续 Onboarding Agent 批量调研阶段读取补全。

- [ ] **Step 1: 实现一次性迁移脚本**

创建 `embodied_datasets/public_datasets_raw/convert_scripts/run_migration.py`：

```python
"""One-off script: migrate VLA Research.xlsx into datasets_registry.yaml and
per-dataset onboarding config stubs.

Run from repo root:
    python3 embodied_datasets/public_datasets_raw/convert_scripts/run_migration.py
"""
from __future__ import annotations

from pathlib import Path

from common.io import save_dataset_config, save_registry
from common.migrate_xlsx_to_registry import (
    build_registry_and_configs,
    read_name_and_link_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
XLSX_PATH = REPO_ROOT / "VLA Research.xlsx"
REGISTRY_PATH = REPO_ROOT / "embodied_datasets" / "datasets_registry.yaml"
CONFIGS_DIR = (
    REPO_ROOT
    / "embodied_datasets"
    / "public_datasets_raw"
    / "convert_scripts"
    / "configs"
)


def main() -> None:
    rows = read_name_and_link_rows(XLSX_PATH)
    entries, configs = build_registry_and_configs(rows)
    save_registry(entries, REGISTRY_PATH)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    for config in configs:
        save_dataset_config(config, CONFIGS_DIR / f"{config.id}.yaml")
    print(f"wrote {len(entries)} registry entries to {REGISTRY_PATH}")
    print(f"wrote {len(configs)} config stubs to {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 安装依赖并运行迁移脚本**

Run:
```bash
cd /Users/haoyuefan/Documents/github/vla_data_pipeline
python3 -m pip install -r requirements.txt
python3 embodied_datasets/public_datasets_raw/convert_scripts/run_migration.py
```
Expected: 打印 `wrote 33 registry entries to .../datasets_registry.yaml` 和
`wrote 33 config stubs to .../configs`

- [ ] **Step 3: 人工核对生成结果**

Run: `cat embodied_datasets/datasets_registry.yaml | head -20`
确认：条目数为 33，每条包含 `id`/`name`/`priority: P2`/各状态字段默认值。

Run: `ls embodied_datasets/public_datasets_raw/convert_scripts/configs/ | wc -l`
Expected: `33`

Run: `cat embodied_datasets/public_datasets_raw/convert_scripts/configs/droid.yaml`
确认：包含 `source_url`（DROID 官方下载链接）与 `review_status: pending_human_review`，
其余字段为空，等待 Onboarding Agent 调研阶段填充。

- [ ] **Step 4: 全量跑一遍现有测试确保没有破坏之前的任务**

Run: `python3 -m pytest -v`
Expected: 全部 PASS（Task1-5 的测试 + 本任务无新增自动化测试，人工核对见Step3）

- [ ] **Step 5: 提交**

```bash
git add embodied_datasets/public_datasets_raw/convert_scripts/run_migration.py \
        embodied_datasets/datasets_registry.yaml \
        embodied_datasets/public_datasets_raw/convert_scripts/configs/
git commit -m "chore: migrate VLA Research.xlsx into datasets_registry.yaml and config stubs"
```

---

## 计划完成后的下一步（不在本计划范围内）

- 对 33 个 `configs/<id>.yaml` stub，逐个用 `build_onboarding_prompt()` 生成的
  prompt 派发调研 agent（Agent 工具，一个数据集一个任务，可并发），用
  `parse_and_validate_agent_output()` 校验产出后写回文件。
- 待 Onboarding Agent 调研完成、人工把 `review_status` 从
  `pending_human_review` 改为 `confirmed` 后，才能开始"计划B（完整性校验）"
  和"计划C（convert_scripts框架）"。
