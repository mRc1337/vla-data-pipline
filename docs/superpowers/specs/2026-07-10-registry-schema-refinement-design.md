# 注册表 Schema 补全 + 可配置数据根目录 + 顶层 README —— 设计文档

- 日期：2026-07-10
- 状态：草案，待用户审阅
- 前置文档：[2026-07-08 VLA 数据预处理与对齐流水线设计文档](2026-07-08-vla-data-pipeline-design.md)

## 1. 背景

Plan A（注册表基础设施）已完成并合并，59 个数据集（33 个存量 + 26 个新发现）已全部跑完
Onboarding Agent 调研，产出结构化的 `convert_scripts/configs/<id>.yaml`。调研过程中暴露
出三类问题：

1. **数据目录硬编码在仓库内**：`run_migration.py` 用 `Path(__file__).resolve().parents[3]`
   推导仓库根目录，未来的 `convert_scripts`/`verify_scripts`/`process_scripts` 如果延续这个
   写法，会把实际数据（可能几百 GB 到几 TB）的存放位置和仓库路径耦合死。
2. **Schema 字段覆盖不全**：Onboarding Agent 在 26 份调研报告里反复遇到"这个值不在枚举
   里"或"这个特性没有字段能装"的情况，被迫留空并写进 `suggested_new_enum_values`，累积了
   一批未处理的枚举缺口和结构性缺口。
3. **没有顶层说明文档**：`embodied_datasets/` 目录本身没有 README，不了解设计文档的人
   无法直接从目录里看出结构、字段含义和当前进度。

本设计一次性解决这三个问题。

## 2. 可配置数据根目录

### 2.1 范围

只挪动"重数据"目录，元数据永远留在仓库内：

| 内容 | 位置 |
|---|---|
| `datasets_registry.yaml`、`convert_scripts/configs/*.yaml`、所有脚本代码 | 始终在仓库内 `embodied_datasets/` 下，不受本设计影响 |
| `public_datasets_raw/<id>/raw/`、`public_datasets_raw/<id>/lerobot_v2_1_staging/`、`public_datasets/lerobot_v2_1/<id>/`、`urdf_assets/` | 受 `--data-root` 控制，可指向仓库外任意路径 |

### 2.2 新模块 `convert_scripts/common/paths.py`

```python
REPO_ROOT: Path                      # 仓库根目录（原 run_migration.py 里的推导逻辑迁移到这里）
DEFAULT_DATA_ROOT: Path = REPO_ROOT / "embodied_datasets"

def resolve_data_root(cli_value: Optional[str]) -> Path:
    """cli_value 非空则返回其绝对路径；为空则返回 DEFAULT_DATA_ROOT。"""

def raw_dir(data_root: Path, dataset_id: str) -> Path:
    """data_root / "public_datasets_raw" / dataset_id / "raw" """

def lerobot_v2_1_staging_dir(data_root: Path, dataset_id: str) -> Path:
    """data_root / "public_datasets_raw" / dataset_id / "lerobot_v2_1_staging" """

def lerobot_v2_1_final_dir(data_root: Path, dataset_id: str) -> Path:
    """data_root / "public_datasets" / "lerobot_v2_1" / dataset_id """

def urdf_assets_dir(data_root: Path, robot_platform: str) -> Path:
    """data_root / "urdf_assets" / robot_platform """
```

所有路径构造函数显式接收 `data_root: Path` 参数，不使用模块级全局状态，方便单测和多次
调用不同 `data_root` 场景。

### 2.3 CLI 约定

未来所有会读写重数据的脚本（`verify_scripts/verify_integrity.py`、
`convert_scripts/<id>.py`、`process_scripts/run_pipeline.py`，均为后续 Plan B/C/D 的产物，
本次不实现）必须接受一个可选的 `--data-root PATH` 参数；不传时用
`resolve_data_root(None)` 的默认值（即仓库内 `embodied_datasets/`），保持向后兼容。这条
写入下面的 Global Constraints，作为对后续实施计划的硬约束。

本次实现范围：只创建 `common/paths.py` 本身及其单测。当前没有任何已实现脚本读写重数据
目录，因此本次不需要改动 `run_migration.py`（它只读写仓库内的 registry/configs，不受
`--data-root` 影响）。

## 3. Schema 补全

### 3.1 新增枚举类型

```python
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
```

### 3.2 现有枚举新增值

| 枚举 | 新增成员 | 字符串值 | 依据（数据集） |
|---|---|---|---|
| `CameraView` | `WRIST` | `wrist` | fmb, furniturebench, the_colosseum, gensim2, fastumi, mv_umi, omniumi（单臂通用手腕相机） |
| `CameraView` | `BODY_WORN` | `body_worn` | dexcap（胸部佩戴相机） |
| `CameraView` | `WORMS_EYE` | `worms_eye` | aloha_unleashed（仰视相机） |
| `LicenseEnum` | `CC_BY_NC_SA_4_0` | `CC-BY-NC-SA-4.0` | galaxea_open_world_dataset, grutopia, airexo_2, yubi |
| `LicenseEnum` | `CC_BY_NC_ND_4_0` | `CC-BY-NC-ND-4.0` | egodex |
| `RobotPlatform` | `UNITREE_H1` | `unitree_h1` | bigym, humanoidbench |
| `RobotPlatform` | `FOURIER_GR1` | `fourier_gr1` | dexmimicgen |
| `RobotPlatform` | `FLEXIV_RIZON4` | `flexiv_rizon4` | airexo_2 |
| `RobotPlatform` | `GALAXEA_R1_LITE` | `galaxea_r1_lite` | galaxea_open_world_dataset |
| `RobotPlatform` | `TOYOTA_ELEY` | `toyota_eley` | yubi |
| `EmbodimentClass` | `QUADRUPED` | `quadruped` | robogen（locomotion 模块） |
| `CollectionMethod` | `AR_HAPTIC_GUIDED` | `ar_haptic_guided_synthesis` | arcap |
| `CollectionMethod` | `SCENE_ASSET_CURATION` | `scene_asset_curation` | grutopia |
| `RawFormat` | `VRS` | `VRS` | hot3d |
| `GripperType` | `THREE_JAW` | `three_jaw` | mv_umi |
| `ActionFrame` | `RELATIVE_TRAJECTORY` | `relative_trajectory` | mv_umi（SLAM 相对起点位姿） |
| `ActionFrame` | `MIXED_DELTA_ABSOLUTE` | `mixed_delta_absolute` | omniumi（平移 delta + 旋转 absolute 混合） |

### 3.3 `DatasetConfig` 新增字段

```python
# 溯源/分类
release_type: Optional[ReleaseType] = None
is_multi_embodiment: Optional[bool] = None
paper_url: Optional[str] = None
secondary_collection_methods: List[CollectionMethod] = Field(default_factory=list)

# 多模态感知（可扩展列表，新增模态只需加枚举值，不需要加字段）
additional_modalities: List[SensorModality] = Field(default_factory=list)

# 相机阵列拓扑（跟"有哪些视角"是不同维度，不放进 additional_modalities）
has_synchronized_multiview_rig: Optional[bool] = None

# 灵巧手/双手专属（跟 dof_per_arm/rotation_representation 分别针对手臂+手腕方向，
# 是互补而非替代关系：同一个数据集可以两组字段都填）
dof_per_hand: Optional[int] = None
hand_pose_representation: Optional[HandPoseRepresentation] = None

# 规模（声明值，跟 RegistryEntry 里的"实测值" duration_hours 对称）
expected_duration_hours: Optional[float] = None
num_subjects: Optional[int] = None
num_scenes: Optional[int] = None
num_objects: Optional[int] = None
```

字段设计原则（供后续扩展参考）：
- **能落到已有 list 字段里的特性用 enum 值扩展**（如相机新视角、gripper 新类型），
  **不落到已有 list 字段里的独立布尔特性才批量收进新 list**（如多模态传感器，四个独立
  布尔位收进 `additional_modalities: List[SensorModality]`，避免"每来一个新传感器就加
  一个新字段"）。
- **`embodiment_class`/`collection_method` 保持单值 scalar**，多值场景用配套的
  `is_multi_embodiment: bool` / `secondary_collection_methods: list` 表达——单数据集
  99% 场景是单值，让 onboarding agent 每次都写 `[x]` 而不是 `x` 是不必要的摩擦；真正
  异构的场景本来就是少数，用旁路字段表达即可。
- `hand_pose_representation` 与 `rotation_representation` 不是互斥关系：前者描述手指/
  手掌形状编码（MANO 参数化 / 3D 关键点 / 关节角），后者描述手腕/末端执行器朝向编码
  （欧拉角/四元数/6D/轴角/旋转矩阵）。带灵巧手的数据集两个字段通常都会填。

所有新字段均为 `Optional`/带默认值的 list，不影响现有 59 份配置文件的 schema 校验
（`extra="forbid"` 不受影响，因为是新增字段而非新增未声明字段）。

### 3.4 `onboarding_agent.py` 同步

`_ENUM_FIELDS` 字典补充：`release_type -> ReleaseType`、
`hand_pose_representation -> HandPoseRepresentation`、
`secondary_collection_methods -> CollectionMethod`、
`additional_modalities -> SensorModality`。沿用现有约定——list-of-enum 字段
（如已有的 `camera_views -> CameraView`）在字典里直接映射到成员枚举类型，不需要特殊
处理 `List[...]` 包装。

### 3.5 已知局限（不在本次方案里解决）

- HOT3D 的许可证是拆分的（序列数据 CC-BY-SA-4.0，手部标注 CC-BY-NC-SA-4.0，3D 模型
  修改版 CC-BY-SA-4.0），单一 `license` 枚举字段无法完整表达。本次不新增
  `CC-BY-SA-4.0`（不带 NC）这个值，`license` 字段继续留空，在 `field_sources` 里
  用自由文本说明拆分情况——这是枚举单值字段的固有局限，接受它比无限拆分枚举更划算。
- VITRA 主条目（VITRA-1M，egocentric 人类视频）不针对特定机器人平台；其配套的
  VITRA-TeleData 用 Realman 7-DoF 臂 + XHand 12-DoF 灵巧手做微调训练，不在这次
  robot_platform 枚举扩展范围内——如果以后 VITRA-TeleData 单独入库，再按需扩展。

## 4. 现有 59 份配置的回填范围

回填分两类，全部基于已完成的调研报告，不做新的调研：

**A. 具体枚举值修正**（直接应用调研报告里已经写好依据的 `suggested_new_enum_values`）：

| 数据集 | 字段 | 修正为 |
|---|---|---|
| functional_manipulation_benchmark_fmb, furniturebench, the_colosseum, gensim2, fastumi, mv_umi, omniumi | `camera_views` | 加入/替换为 `wrist` |
| dexcap | `camera_views` | `third_person` → `body_worn` |
| aloha_unleashed | `camera_views` | 加入 `worms_eye` |
| galaxea_open_world_dataset, grutopia, airexo_2, yubi | `license` | `CC-BY-NC-SA-4.0` |
| egodex | `license` | `CC-BY-NC-ND-4.0` |
| airexo_2 | `robot_platform` | `flexiv_rizon4` |
| bigym | `robot_platform` | `unitree_h1` |
| humanoidbench | `robot_platform` | `unitree_h1`（并设 `is_multi_embodiment: true`，因该 benchmark 覆盖多个具身形态，H1 是最主要的） |
| yubi | `robot_platform` | `toyota_eley`（并设 `is_multi_embodiment: true`，因实际部署在 UR/Franka/ELEY 三种平台） |
| mv_umi | `gripper_type` | `three_jaw` |
| mv_umi | `action_frame` | `relative_trajectory` |
| omniumi | `action_frame` | `mixed_delta_absolute` |
| arcap | `collection_method` | `umi`（此前权宜之计）→ `ar_haptic_guided_synthesis` |
| grutopia | `collection_method` | → `scene_asset_curation` |
| hot3d | `raw_format` | `Custom` → `VRS` |
| assembly101 | `has_synchronized_multiview_rig` | `true`（新字段赋值，不是修正旧字段） |

**B. `release_type` 批量分类**：

- `robogen`, `gensim2` → `generation_framework`
- `grutopia` → `scene_platform`
- `humanoidbench` → `rl_benchmark_env`
- 其余 55 个（已确认是有固定episode数据发布的真实数据集）→ `fixed_episode_dataset`

`robogen` 的 `embodiment_class` **不**改成 `quadruped`——它作为生成框架同时覆盖多种
本体（含 locomotion 用的四足），单一 scalar 值会误导，保持留空 + `is_multi_embodiment: true`
更准确；新增的 `quadruped` 枚举值留给未来真正的单一四足数据集使用。

回填通过一个一次性脚本实现（仿照 `run_migration.py` 的写法：加载 registry/configs，
按上表和上述分类规则原地修改后写回，不引入交互式或调研逻辑），由实现阶段的单测覆盖
"表里列出的每一条修正都被正确应用、未列出的字段不受影响"。

## 5. 顶层 README

新增 `embodied_datasets/README.md`，结构：

1. **手写说明部分**（本次直接写定，后续极少改动）：项目目的一段话、目录结构树状图
   （复用设计文档第3节的树，做适当精简）、字段含义速查（链接回本文档 + 08 号设计文档，
   不重复整张字段表）、如何 onboard 新数据集（引用 Onboarding Agent 流程一句话概述）、
   如何使用 `--data-root`（引用第2节）。
2. **自动生成部分**，用 HTML 注释 marker 包裹：
   ```
   <!-- AUTO-GENERATED TABLE START -->
   ...
   <!-- AUTO-GENERATED TABLE END -->
   ```
   表格列：`id, name, priority, download_status, convert_status, process_status,
   review_status`（来自 `datasets_registry.yaml`）+ `collection_method,
   embodiment_class`（来自对应的 `configs/<id>.yaml`，缺失显示为空），按 `priority`
   再按 `name` 排序。

新增生成脚本 `convert_scripts/common/generate_overview_readme.py`：读取 registry +
所有 configs，渲染表格，只替换 marker 之间的内容，不影响 marker 外的手写部分。手动运行
（`python -m common.generate_overview_readme` 或直接 `python common/generate_overview_readme.py`），
不接入 CI，与设计文档 §7 提到的"README.md 是自动生成的渲染视图"是同一原则在总览层面的
应用——但这里是半自动（手动触发刷新），跟 process_scripts 跑完自动生成单数据集 README
不同，因为总览表变化频率低（新增数据集才变），不需要每次 pipeline 跑完都刷新。

## 6. 范围与非目标

**包含**：
- `common/paths.py` 及其单测
- `schema.py` 的枚举/字段扩展及其单测
- `onboarding_agent.py` 的 `_ENUM_FIELDS` 同步
- 一次性回填脚本，修正第4节列出的具体数据集字段
- `common/generate_overview_readme.py` 及其单测，`embodied_datasets/README.md` 首版内容

**不包含**：
- `paper_url`/`citation_bibtex` 等新字段对现有59个配置的批量回填（只加字段，不强制
  填满——已有 `field_sources` 里的论文引用是自由文本，结构化提取留给未来需要时再做）
- `num_subjects`/`num_scenes`/`num_objects`/`additional_modalities`/`dof_per_hand`/
  `hand_pose_representation`/`has_synchronized_multiview_rig`（除 assembly101 外）
  对现有配置的批量回填——这些字段有具体数据集依据支撑其存在，但回填每一个需要重新
  读一遍对应论文，属于未来调研任务，不在本次范围
- Plan B（verify_scripts）/Plan C（convert_scripts 框架）/Plan D（process_scripts）
  ——`common/paths.py` 只是给它们预留接口约定，具体实现留在各自的计划里

## 7. Global Constraints（写入实施计划时原样带入）

- 所有 Python 代码遵循 Python 3.9 语法（不用 `X | Y`、不用 `match`）
- Pydantic 模型继续用 `ConfigDict(extra="forbid")`，新字段必须显式声明默认值
- 新枚举值/字段的命名和字符串值风格必须与 `schema.py` 现有风格一致（成员名
  `UPPER_SNAKE_CASE`，字符串值 `snake_case` 或已有格式如 `CC-BY-NC-4.0` 的连字符风格）
- 后续任何读写 `public_datasets_raw/`、`public_datasets/`、`urdf_assets/` 下重数据的
  脚本，必须通过 `common/paths.py` 的函数解析路径，并暴露 `--data-root` CLI 参数
- 回填脚本必须是纯规则驱动（表驱动），不引入新的调研/推断逻辑——所有回填值都来自
  第4节列出的、已经在调研报告里给出依据的具体值

## 8. 后续步骤

本设计确认后进入实施计划阶段（writing-plans），预计任务拆分：
1. `common/paths.py` + 单测
2. `schema.py` 枚举/字段扩展 + 单测
3. `onboarding_agent.py` 的 `_ENUM_FIELDS` 同步 + 单测
4. 回填脚本（第4节表驱动）+ 单测
5. `common/generate_overview_readme.py` + 单测
6. 手写 `embodied_datasets/README.md` 说明部分，运行生成器产出首版完整文件，提交
