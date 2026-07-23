# embodied_datasets

VLA（视觉-语言-动作）机器人操作数据集的统一注册、清洗、对齐流水线。所有数据集
先转换为 LeRobot v3.0 格式，再按论文方法论做五阶段数值清洗、三项跨模态质检和
跨本体维度统一。

## 目录结构

```
embodied_datasets/
├── datasets_registry.yaml          # 59个数据集的总览表（实测值，随流水线推进更新）
├── public_datasets_raw/
│   ├── <dataset_id>/raw/                    # 原始下载数据（重数据，见下方"数据根目录"）
│   ├── <dataset_id>/lerobot_v3_0_staging/   # 转换后未清洗的中间态（重数据）
│   ├── convert_scripts/
│   │   ├── configs/<dataset_id>.yaml   # 每个数据集的调研配置（声明值）
│   │   └── common/                     # 复用的 schema/io/onboarding 工具
│   ├── verify_scripts/             # 完整性校验（Plan B，未实现）
│   └── process_scripts/            # 清洗对齐流水线（9个stage/check模块 + 跨本体统一表示层，
│                                    # 见本文档"跨本体统一表示层"一节）
├── urdf_assets/<robot_platform>/   # 按机器人型号共享的 URDF（重数据）
└── public_datasets/
    └── lerobot_v3_0/<dataset_id>/  # 清洗完成的最终数据（重数据）
```

## 数据根目录

`datasets_registry.yaml`、`convert_scripts/configs/*.yaml` 和所有脚本代码始终
留在仓库内，不受下面这条配置影响。只有实际的重数据目录
（`raw/`、`lerobot_v3_0_staging/`、`public_datasets/lerobot_v3_0/`、
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
的本体信息、数据表示方式等）。下面整理两个 pydantic 模型的全部字段；权威定义
永远是 `public_datasets_raw/convert_scripts/common/schema.py` 里的代码，本节
如有出入以代码为准。

下面"当前进度"表格只展示9个核心字段，看不全；每个数据集**全部**约40个声明
字段（license、robot_platform、camera_views、num_subjects……）见文末"完整字段
总览"。

### RegistryEntry（`datasets_registry.yaml`）

随流水线推进自动更新，代表"实际发生了什么"。

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | str | 数据集唯一slug，如`droid` |
| `name` | str | 数据集展示名 |
| `priority` | enum | 见下方 `Priority` |
| `download_status` | enum | 见下方 `DownloadStatus` |
| `integrity_status` | enum | 见下方 `IntegrityStatus` |
| `convert_status` | enum | 见下方 `ConvertStatus` |
| `process_status` | enum | 见下方 `ProcessStatus` |
| `raw_local_path` | str，可空 | 原始数据相对`public_datasets_raw/`的本地路径 |
| `lerobot_v3_0_local_path` | str，可空 | 清洗完成数据相对`public_datasets/lerobot_v3_0/`的本地路径 |
| `storage_size_gb` | float，可空 | 实测占用空间（GB） |
| `num_episodes` | int，可空 | 实测episode数 |
| `num_frames` | int，可空 | 实测帧数 |
| `duration_hours` | float，可空 | 实测总时长（小时） |

#### `Priority`
`P0` / `P1` / `P2` —— 优先级，数字越小越优先。

#### `DownloadStatus`
`not_downloaded` 未下载 / `downloading` 下载中 / `completed` 已完成。

#### `IntegrityStatus`
`not_verified` 未校验 / `verified` 校验通过 / `failed` 校验失败——对应设计文档
第4节的完整性校验（体量核对+格式可解析性+视频抽样解码）。

#### `ConvertStatus`
`not_converted` / `converting` / `converted` / `failed` —— `convert_scripts`
（raw → lerobot_v3_0_staging）的执行状态。

#### `ProcessStatus`
`not_processed` / `processing` / `processed` / `failed` —— `process_scripts`
（五阶段清洗+三项质检+统一表示）的执行状态。

### DatasetConfig（`convert_scripts/configs/<id>.yaml`）

Onboarding 时调研得到，代表"声明的事实"。除 `id`/`name` 外全部可空——没有调研到
或不适用的字段应该留空，不应该编造。

#### 基础信息

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | str | 同 RegistryEntry |
| `name` | str | 同 RegistryEntry |
| `source_url` | str，可空 | 官方主页/下载页链接 |
| `paper_url` | str，可空 | 对应论文链接，跟`source_url`分开存 |

#### 许可与格式

| 字段 | 类型 | 含义 |
|---|---|---|
| `license` | enum | 见下方 `LicenseEnum` |
| `raw_format` | enum | 见下方 `RawFormat` |
| `release_type` | enum | 见下方 `ReleaseType` |

##### `LicenseEnum`
- `MIT` / `Apache-2.0` / `BSD-3-Clause` / `GPL-3.0` / `CC0-1.0` —— 常见开源协议
- `CC-BY-4.0` / `CC-BY-NC-4.0` / `CC-BY-NC-SA-4.0` / `CC-BY-NC-ND-4.0` / `CC-BY-SA-4.0` —— 知识共享协议家族，区别在"是否允许商用/是否要求署名/是否允许衍生/是否要求以同协议共享"
- `CDLA-Sharing-1.0` —— Community Data License Agreement，数据集专用的共享协议（跟代码用的开源协议不是一回事）
- `Proprietary` —— 私有/未公开协议
- `custom_research_eula` —— 各机构自己定制的门禁研究协议（如Ego4D、ETH各数据集的EULA），一次性法律文本不可复用，具体条款记在`field_sources`
- `Unknown` —— 确实没找到

##### `RawFormat`
- `RLDS` —— TensorFlow Datasets的RLDS封装
- `HDF5` —— HDF5文件
- `LeRobot` —— 已经是LeRobot格式
- `ROS_bag` —— ROS bag录制
- `MCAP` —— MCAP格式（ROS2常用）
- `TFRecord` —— 原生TFRecord（非RLDS封装）
- `VRS` —— Meta Project Aria的传感器容器格式
- `Custom` —— 各家自定义格式（npy/pkl/自定义目录结构等），目前占比最高

##### `ReleaseType`
- `fixed_episode_dataset` —— 固定episode数的常规数据集（绝大多数）
- `generation_framework` —— 生成框架，理论上能无限生成新任务/轨迹，没有固定语料（如RoboGen、GenSim2）
- `scene_platform` —— 场景/资产平台，提供可交互场景而非机器人轨迹本身（如GRUtopia）
- `rl_benchmark_env` —— 纯RL训练环境，rollout实时跑出来，没有预录制轨迹（如HumanoidBench）

#### 采集方式与本体分类

| 字段 | 类型 | 含义 |
|---|---|---|
| `collection_method` | enum | 见下方 `CollectionMethod`（主要采集方式，单值） |
| `secondary_collection_methods` | list[enum] | 次要/补充采集方式（同一数据集可以有多个） |
| `is_multi_embodiment` | bool，可空 | 是否同一份发布横跨多种具身形态——如果是，`embodiment_class`/`robot_platform`应该留空，不要强行选一个 |
| `embodiment_class` | enum | 见下方 `EmbodimentClass` |
| `robot_platform` | enum | 见下方 `RobotPlatform`（开放列表，会持续扩充） |

##### `CollectionMethod`
- `teleop` —— 远程遥操作（VR手柄/太空鼠/关节示教等接口都算）
- `kinesthetic` —— 物理引导示教（人手直接拉着机械臂走，跟"远程"操作是不同范畴）
- `autonomous_policy` —— 已训练策略自主采集
- `scripted` —— 预编程/随机化动作原语采集，既非遥操也非学习到的策略
- `umi` —— UMI（手持夹爪采集器）范式
- `egocentric_human` —— 纯第一视角人类活动录制
- `ego_exo_human` —— 同时录第一视角+第三方视角的人类活动（跟纯`egocentric_human`不是一回事）
- `mocap_multiview_human` —— 工作室多视角RGB+光学动捕（如TACO）
- `simulation` —— 仿真采集
- `human_to_robot_synthesis` —— 人手视频retarget合成机器人轨迹
- `synthetic_multimodal_augmentation` —— 对已有轨迹叠加合成的音频/对话等模态（如RoboOmni）
- `ar_haptic_guided_synthesis` —— AR头显+力反馈手套引导式合成（ARCap专用）
- `scene_asset_curation` —— 场景/资产整理，不是机器人轨迹采集

##### `EmbodimentClass`
- `single_arm` —— 单臂固定基座
- `dual_arm` —— 双臂固定基座（工业臂式布局，无躯干概念）
- `half_humanoid` —— 双臂+躯干无腿部（如AgiBot G1、Galbot G1）
- `humanoid` —— 完整人形（有腿/移动能力）
- `mobile_manipulator` —— 移动底盘+机械臂
- `human_hand` —— 裸手（只跟踪手部，无全身姿态）
- `human_full_body` —— 全身姿态（跟只覆盖手部的`human_hand`不是一回事）
- `quadruped` —— 四足

##### `RobotPlatform`
开放枚举，按具体机器人型号列出，目前包括：`franka_panda`、`franka_fr3`、`ur5`、
`ur5e`、`agilex_aloha`、`agilex_cobot_magic`、`xarm6`、`xarm7`、`kinova_gen3`、
`kuka_iiwa`、`sawyer`、`widowx`、`viperx`、`abb_yumi`、`hello_robot_stretch`、
`boston_dynamics_spot`、`everyday_robots_arm`、`agibot_g1`、`agibot_g2`、
`galbot_g1`、`tien_kung`、`arx5`、`unitree_g1`、`unitree_h1`、`unitree_h1_2`、
`unitree_aliengo`、`unitree_a1`、`anymal`、`atlas`、`fourier_gr1`、
`flexiv_rizon4`、`galaxea_r1`、`galaxea_r1_lite`、`r1pro`、`toyota_eley`、
`1x_eve`、`virtual_agent`（无实体硬件的虚拟具身，如AI2-THOR agent）、`other`
（未收录型号的兜底值，不允许静默把新型号硬塞进已有值——遇到新型号应该在
`suggested_new_enum_values`里提出）。

#### 机械结构

| 字段 | 类型 | 含义 |
|---|---|---|
| `num_arms` | int(0-2)，可空 | 手臂数 |
| `dof_per_arm` | int，可空 | 每臂自由度 |
| `dof_per_hand` | int，可空 | 每只灵巧手/末端自由度（跟`dof_per_arm`互补，不是替代） |
| `gripper_type` | enum | 见下方 `GripperType` |
| `hand_pose_representation` | enum | 见下方 `HandPoseRepresentation`——描述手指/手掌**形状**编码，跟描述手腕**朝向**编码的`rotation_representation`是互补关系，带灵巧手的数据集两个字段通常都会填 |
| `has_mobile_base` | bool，可空 | 是否有移动底盘 |

##### `GripperType`
- `parallel_jaw` —— 二指平行夹爪
- `dexterous_hand` —— 多指灵巧手
- `three_jaw` —— 三指夹爪
- `cage_pinch` —— 缆线操作专用非对称二指机制（一指"笼住"允许滑动，另一指"夹紧"）
- `suction` —— 吸盘
- `mixed` —— 同一发布内混用多种末端执行器（多本体聚合数据集）
- `none` —— 无末端执行器（裸手数据集）
- `unknown` —— 有末端执行器但型号不明

##### `HandPoseRepresentation`
- `mano` —— MANO参数化手部模型
- `keypoints_3d` —— 3D关键点
- `joint_angles` —— 关节角度
- `none` —— 不适用（非灵巧手/非手部追踪数据集）

#### 动作与坐标系表示

| 字段 | 类型 | 含义 |
|---|---|---|
| `action_space` | enum | 见下方 `ActionSpace` |
| `action_frame` | enum | 见下方 `ActionFrame` |
| `rotation_representation` | enum | 见下方 `RotationRepresentation` |
| `state_dim` | int，可空 | 状态向量维度 |
| `action_dim` | int，可空 | 动作向量维度 |

##### `ActionSpace`
- `joint_position` / `joint_velocity` / `joint_torque` —— 关节空间控制
- `eef_pose` —— 末端位姿控制
- `discrete_symbolic` —— 离散动作词表（AI2-THOR式，非连续控制）
- `mixed` —— 同一发布内混用多种动作空间
- `unknown` —— 有动作但空间类型不明

##### `ActionFrame`
- `delta` —— 相对上一步的增量
- `absolute` —— 绝对坐标
- `both` —— 数据里同时提供两种
- `relative_trajectory` —— 相对轨迹起点（如UMI的SLAM相对定位，跟逐步`delta`不同）
- `mixed_delta_absolute` —— 同一动作向量里部分维度delta、部分absolute（如平移delta+旋转absolute）
- `unknown`

##### `RotationRepresentation`
- `euler_xyz` / `quaternion` / `rotation_6d` / `axis_angle` / `rotation_matrix` —— 标准3D朝向编码
- `single_axis_angle` —— 只有单一旋转角（如仅yaw，不是完整3D朝向）
- `mixed` —— 同一发布内混用多种编码（多本体/多子集聚合数据集）
- `none` —— 无旋转编码（如纯关节控制、离散动作）
- `unknown`

#### 采集频率

| 字段 | 类型 | 含义 |
|---|---|---|
| `fps` | float，可空 | 采集帧率 |
| `fps_variable` | bool，可空 | 帧率是否逐数据集/逐子集变化 |

#### 视觉与相机

| 字段 | 类型 | 含义 |
|---|---|---|
| `num_camera_views` | int，可空 | 相机视角数 |
| `camera_views` | list[enum] | 见下方 `CameraView` |
| `has_synchronized_multiview_rig` | bool，可空 | 多相机是否同步组成一个阵列（跟"有哪些视角"是不同维度） |
| `has_camera_calibration` | bool，可空 | 是否提供相机标定参数 |
| `depth_coverage` | enum | 见下方 `DepthCoverage` |

##### `CameraView`
- `third_person` / `head` / `top` / `front` / `side` / `other` —— 常见外部视角
- `left_wrist` / `right_wrist` —— 双臂场景下分左右的手腕相机
- `wrist` —— 单臂通用手腕相机（非分手，跟`left_wrist`/`right_wrist`不同）
- `gripper_jaw` —— 爪部相机（跟臂部安装的`wrist`是不同安装位置，如Spot机器人）
- `body_worn` —— 佩戴在身体上（如胸部相机，跟头戴式`head`不同）
- `worms_eye` —— 仰视相机

##### `DepthCoverage`
`none` 无深度 / `partial` 部分视角有深度 / `full` 全部视角有深度。

#### 其他传感模态

| 字段 | 类型 | 含义 |
|---|---|---|
| `additional_modalities` | list[enum] | 见下方 `SensorModality`——可扩展列表，新增模态只需加枚举值，不需要新字段 |

##### `SensorModality`
`force_torque` 力/力矩 / `tactile` 触觉 / `audio` 音频 / `eye_gaze` 眼动 /
`imu` 惯性测量单元 / `semantic_segmentation` 语义分割视频流 /
`point_cloud_3d_scan` 持久3D场景重建点云（跟逐帧`depth_coverage`是不同维度）。

#### 语言与任务

| 字段 | 类型 | 含义 |
|---|---|---|
| `has_language_instruction` | bool，可空 | 是否有语言指令标注 |
| `num_task_types` | int，可空 | 任务类型数 |

#### URDF

| 字段 | 类型 | 含义 |
|---|---|---|
| `urdf_available` | bool，可空 | 是否能找到URDF |
| `urdf_source` | enum | 见下方 `UrdfSource` |

##### `UrdfSource`
`dataset_repo` 数据集自带 / `manufacturer_official` 厂商官方 /
`community_repo` 社区维护 / `not_found` 找不到。

#### 规模（声明值，对应RegistryEntry里的实测值）

| 字段 | 类型 | 含义 |
|---|---|---|
| `expected_size_gb` | float，可空 | 声明的数据体量（GB） |
| `expected_num_episodes` | int，可空 | 声明的episode数 |
| `expected_duration_hours` | float，可空 | 声明的总时长（小时） |
| `num_subjects` | int，可空 | 人类被试/操作员数 |
| `num_scenes` | int，可空 | 场景/环境数 |
| `num_objects` | int，可空 | 交互物体数 |

#### 审核与溯源

| 字段 | 类型 | 含义 |
|---|---|---|
| `review_status` | enum | 见下方 `ReviewStatus` |
| `field_sources` | dict[str,str] | 每个已填字段的信息来源（URL或论文章节），调研/复核时必须记录 |
| `suggested_new_enum_values` | dict[str,str] | 实际取值不在现有枚举里时，记录"建议新增什么值+理由"，而不是编造/硬塞进已有值 |

##### `ReviewStatus`
`pending_human_review` 待人工确认（默认值） / `confirmed` 已人工确认——
`convert_scripts`运行前应检查这个字段是`confirmed`，防止调研错误直接污染生产流水线。

如果只是想看"某个字段现在有哪些枚举值"，比这份手写文档更可靠的方式是直接跑：

```bash
python3 -c "
import sys; sys.path.insert(0, 'embodied_datasets/public_datasets_raw/convert_scripts')
from common.schema import <EnumName>
for m in <EnumName>: print(m.value)
"
```

## 如何 onboard 新数据集

1. 在 `datasets_registry.yaml` 里加一条 `RegistryEntry`，在 `configs/` 下建一个
   同 id 的 stub `DatasetConfig`（只填 `id`/`name`/`source_url`）。
2. 用 `common/onboarding_agent.py` 的 `build_onboarding_prompt()` 生成调研任务
   的提示词，派给一个 Agent 去读官网/论文并填字段。
3. 用 `parse_and_validate_agent_output()` 校验 Agent 产出的 YAML 能通过
   schema 校验，写回 `configs/<id>.yaml`，`review_status` 保持
   `pending_human_review` 直到人工确认。

## 跨本体统一表示层 —— 数据公共规范

本节描述 `process_scripts` 流水线产出的**最终 LeRobot v3.0 数据集**里，机器人本体
（robot-collected embodiment）的 `observation.state` 到底是什么格式。这是给下游训练/
评测脚本读取数据时依赖的公共契约，不是内部设计草稿——如果本节和 `unify_representation.py`
的实际代码不一致，代码是准的，请提 issue。

对应实现：`process_scripts/unify_representation.py`（计算逻辑）+
`process_scripts/run_pipeline.py`（把计算结果写进最终数据集）+
`process_scripts/common/io.py`（`write_lerobot_episodes` 的落盘细节）。

### 1. 谁会被统一表示，谁不会

Gate 条件（`unify_representation.py` 里的 `ROBOT_EMBODIMENT_CLASSES`）：只对以下
`embodiment_class` 生效——

```
single_arm, dual_arm, half_humanoid, humanoid, mobile_manipulator, quadruped
```

`human_hand`（第一/第三人称人手视频，如 H2O、OAKink2、TACO）和 `human_full_body`
（如 EgoAllo）**不经过这一层**，`observation.state` 保留原始 per-dataset 维度不变。
这些数据集的 `dof_per_hand` 常见值是 45/48（MANO 参数），如果强行塞进本规范的槎位
会严重失真——本规范从设计上就不覆盖它们，不是槎位不够宽的问题。

### 2. 80 维 canonical 向量布局

固定总维度 80，按下表切片。`JOINT_SLOT`（7）和 `GRIPPER_SLOT`（21）是从注册表实测数据
反推出来的（见第4节，`dof_per_arm` 实测最大值7、`dof_per_hand` 实测最大值21）；`EEF_SLOT`（7）
不是数据反推的，是固定的位姿表示惯例（3维位置+4维四元数），跟具体数据集无关：

| 子区间 | 维度 | 内容 | 常量名 |
|---|---|---|---|
| `[0:7]`（单臂）/ `[0:7]`（arm1） | 7 | 关节位置 | `JOINT_SLOT` |
| `[7:14]` | 7 | 末端位姿：3维位置 + 4维四元数 | `EEF_SLOT` |
| `[14:35]` | 21 | 夹爪/灵巧手槎位（见第3节分支规则） | `GRIPPER_SLOT` |
| `[35:70]`（仅双臂数据集） | 35 | arm2，结构与 `[0:35]` 完全相同 | `ARM_BLOCK_DIM` |
| `[70:73]` | 3 | 移动底盘 `vx/vy/yaw` 速度，仅 `has_mobile_base=true` 时填充 | `MOBILE_BASE_SLOT` |
| `[73:80]` | 7 | 预留，当前恒为0，给未来新模态（如躯干位姿）留空间 | — |

四元数的分量顺序（xyzw vs wxyz）是 `convert_scripts` onboarding 时应遵循的惯例，
`unify_representation.py` 本身不做任何顺序校验/转换——它只是把 `convert_scripts` 已经
排好序的4个数原样搬进 `[10:14]`，具体是不是xyzw取决于上游数据集转换代码是否遵循了这个
惯例，本节不能替代那边的校验。

单臂数据集：`[35:70]` 恒为0，对应的 `mask` 恒为 `False`（不是"第二臂静止"，是"没有第二臂"，
训练时应该按 mask 忽略这段，不要当成真实的零速度数据）。

`ARM_BLOCK_DIM = JOINT_SLOT + EEF_SLOT + GRIPPER_SLOT = 35`，`MOBILE_BASE_SLOT` 是从
`2 * ARM_BLOCK_DIM` 算出来的——这两个都是代码里的派生表达式，不是写死的数字，以后如果
再调整槎位宽度，这两处不用手动同步。

### 3. 夹爪槎位 `[14:35]` 的分支规则

槎位宽度固定21维，但不同 `gripper_type` 只用其中一部分：

- `parallel_jaw` / `three_jaw` / `cage_pinch` / `suction`（简单夹爪）→ 只用 **slot0**
  （1维开合宽度/吸附状态），剩余20维置0、mask=False
- `dexterous_hand`（灵巧手）→ 用实际列宽（最多21维，不足补0，超过截断），每一维单独
  设 mask
- 其他/未知 gripper_type → 全部置0，mask=False

**手腕不占用这个槎位**——手腕姿态已经在 `[7:14]` 的末端位姿里了（那是整条运动链末端的
笛卡尔位姿，含手腕）。这里的21维是手**自身**的执行器自由度（手指+虎口等），跟手腕解耦。

### 4. 为什么是21，不是15或其他数字

2026-07-20 之前的版本用的是15，来自参考论文（Qwen-RobotManip）的默认假设，没有拿本项目
自己的注册表数据验证过。后来查了 `datasets_registry.yaml`/`convert_scripts/configs/*.yaml`
里所有 `gripper_type=dexterous_hand` 的数据集，按 `embodiment_class` 分成"机器人采集"
（会走这一层）和"人手视频/MANO"（被第1节的gate排除，不会走这一层）两类。**机器人采集的
5个**（这是全部，不是抽样）：

| 数据集 | `embodiment_class` | `dof_per_hand` |
|---|---|---|
| `arcap` | single_arm | **16** |
| `nvidia_gr00t_teleop_g1` | humanoid | 7 |
| `nvidia_locomanipulation_grail` | humanoid | 7 |
| `dexmimicgen` | dual_arm | 6 |
| `gr00t_teleop_sim` | humanoid | 6 |

`dof_per_hand=21` 定这个宽度时，注册表里还有 `humanoidbench`（Unitree H1 + Shadow Hand，
`dof_per_hand=21`，来自其 `convert_scripts/configs/humanoidbench.yaml` 的
`field_sources.dof_per_hand` 标注：论文原文是"we normalize the action space to be
[-1,1]^||A||, where ||A||=61 (19 for the humanoid body and 21 for each hand)"，
"21 actuator DOF per Shadow Hand end effector" 是标注时对这句话的转述，不是论文原文
逐字引用），是当时6个机器人灵巧手数据集里的实测最大值，槎位宽度按它定为21——刚好
覆盖当时的数据，不多留冗余。

2026-07-21：`humanoidbench` 已按数据分类整理的结果从注册表移除（不在最终确认的66个
数据集范围内），当前注册表内机器人采集灵巧手数据集的实测最大值变成了16（`arcap`）。
**`GRIPPER_SLOT` 常量本身未跟着收缩，仍保持21**——是否要把21缩到16是一个独立的架构
决策（会牵动 `ARM_BLOCK_DIM`/`CANONICAL_DIM` 的派生值、已写好的测试、以及所有已按80维
写出的历史数据），这里先记录现状，不擅自改动，留给后续单独决定。如果以后有数据集实测
超过21维的灵巧手，或决定收缩槎位宽度，都需要按同样的方式（查注册表实测值，不是猜）
决定新宽度，并同步更新本节、`unify_representation.py`
的 `GRIPPER_SLOT` 常量、以及 `run_pipeline.py` 里 README 模板的"已知局限"措辞。

被排除的人手视频/MANO 数据集（`dexcap`=16, `ph2d`=6, `h2o`=48, `oakink2`=48, `taco`=48,
`vitra`=45, `hoi4d`=45）——不管它们的 `dof_per_hand` 多大或多小，排除的
原因始终是第1节的 `embodiment_class` gate（`human_hand`/`human_full_body`），不是因为
它们凑巧是45/48维；`dexcap`/`ph2d` 的 `dof_per_hand` 也不到21，但同样因为不是机器人采集
而被排除。

### 5. `episode.action` 不做统一化

本层**只处理 `observation.state`，不处理 `action`**。`action` 在最终输出里保持原始
per-dataset 维度不变。这是有意的范围收窄（不是遗漏）：`action` 的语义（joint velocity /
eef delta pose / 离散符号等）比 state 更多样，统一到同一套槎位需要额外设计，这轮没有做。

### 6. mask：怎么知道哪些维度是"真实数据"

`unify_representation.apply()` 除了算出80维向量，还会算一个80维的 bool mask（对同一个
数据集内所有帧、所有episode都是同一份，因为它只取决于 `dof_per_arm`/`num_arms`/
`gripper_type`/`has_mobile_base` 这些数据集级别的固定配置，不随帧变化）。

这个 mask 会作为**独立的 lerobot feature** 写进最终数据集：

```
observation.state_canonical_mask   # bool, shape (80,)，每帧都写，但整数据集内容完全相同
```

`mask[i]=False` 表示第 i 维是"这个本体本来就没有这个自由度"的零填充，不是"测量值恰好是0"。
训练时用这个 mask 过滤loss/attention，而不是直接假设全部80维都是有效信号。

### 7. 已知局限

- **`dof_per_arm` 配错了但没超列宽时，本层无法检测**：`_pack_arm` 完全信任
  `config.dof_per_arm` 划分关节/末端位姿的边界。如果这个值配错了（比如设置得比真实关节数
  大），但该臂的总列宽仍然够用，会把真实的末端位姿/夹爪数据错位塞进关节槎位，`mask` 还是
  `True`，没有任何报错信号。这跟 Stage4（有URDF可以交叉验证 `dof_per_arm`）不同，这一层
  没有独立的地面真值可以核对，正确性完全依赖上游 `DatasetConfig.dof_per_arm` 填得准。
- **只统一 state，不统一 action**（见第5节）。
- **不覆盖 MANO/人手视频**（见第1节）——这些数据的完整参数保留在
  `lerobot_v3_0_staging/` 原始数据里，本层完全不touch它们。
- **超过21维的灵巧手会被截断**——目前注册表里没有这种数据集，一旦出现需要重新评估
  槎位宽度（见第4节）。

### 8. 怎么验证

首次搭建需要先建独立的 `.venv-process`（仓库根 `README.md` 有完整说明）：

```bash
cd embodied_datasets/public_datasets_raw/process_scripts
python3.11 -m venv .venv-process
source .venv-process/bin/activate
pip install -r requirements.txt
```

之后每次只需要：

```bash
cd embodied_datasets/public_datasets_raw/process_scripts
source .venv-process/bin/activate   # 或 .venv-process/bin/pytest 直接调用
pytest tests/test_unify_representation.py -v   # 本层的单元测试：单臂/双臂/移动底盘/gate/21维灵巧手边界
pytest tests/test_run_pipeline.py -v           # 验证 canonical_state 真的替换了 episode.state 并写进最终数据集
```

`tests/test_unify_representation.py` 里有一个专门测试21维灵巧手的用例
（`test_dexterous_hand_with_21_dof_packs_without_truncation`），构造21维互不相同的数值，
断言全部21维都正确落进 `[14:35]` 且 `mask=True`——如果以后又把槎位宽度改窄了，这个测试
会先崩。

**注意：跑真实（非合成）数据集时需要系统装 `ffmpeg`**（不是pip依赖：
macOS 用 `brew install ffmpeg`，Ubuntu/Debian 用
`sudo apt update && sudo apt install ffmpeg`）。lerobot 的视频解码器（torchcodec）只检查库能不能
import，不检查能不能真正dlopen——没装系统ffmpeg时，所有用 `LeRobotDataset.create()`
构造的合成fixture测试照样全过（因为不涉及真实视频解码），只有读真实数据集的视频轨才会
在运行时崩。已经用 `lerobot/pusht`（HuggingFace上一个公开的小型v3.0格式数据集，206
episode/25650帧）验证过完整流程：真实下载、`load_lerobot_episodes` 原样读取、跑通全部
9个stage/check模块、最终写出的数据集里 `observation.state` 确认是80维、
`observation.state_canonical_mask` 确认写入且数值位置正确。

## 当前进度

<!-- AUTO-GENERATED TABLE START -->

| id | name | priority | download_status | convert_status | process_status | review_status | collection_method | embodiment_class |
|---|---|---|---|---|---|---|---|---|
| 1x_world_model_dataset | 1X World Model Dataset | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | humanoid |
| alfred | ALFRED | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | mobile_manipulator |
| arcap | ARCap | P2 | not_downloaded | not_converted | not_processed | pending_human_review | ar_haptic_guided_synthesis | single_arm |
| agibot_digital_world | AgiBot Digital World | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | dual_arm |
| agibot_world | AgiBot-World | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | dual_arm |
| assembly101 | Assembly101 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| behavior_robot_suite | BEHAVIOR Robot Suite | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| behavior_1k | BEHAVIOR-1K | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| brmdata | BRMData | P2 | not_downloaded | not_converted | not_processed | confirmed | teleop | mobile_manipulator |
| bridgedata_v2 | BridgeData V2 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| calvin | CALVIN | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| droid | DROID | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| dexcap | DexCap | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | human_hand |
| dexmimicgen | DexMimicGen | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | dual_arm |
| dobb_e | Dobb-E | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | mobile_manipulator |
| epic_kitchens_100 | EPIC-KITCHENS-100 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| ego_exo4d | Ego-Exo4D | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| ego4d | Ego4d | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_full_body |
| egodex | EgoDex | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| egoexolearn | EgoExoLearn | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| fastumi | FastUMI | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | single_arm |
| functional_manipulation_benchmark_fmb | Functional Manipulation Benchmark (FMB) | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| furniturebench | FurnitureBench | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| gr00t_teleop_sim | GR00T Teleop Sim | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | humanoid |
| galaxea_open_world_dataset | Galaxea Open-World Dataset | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| h2o | H2O | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| hd_epic | HD-EPIC | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| hoi4d | HOI4D | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| hot3d | HOT3D | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| libero | LIBERO | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| libero_plus | LIBERO-plus | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
| language_table | Language Table | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| mv_umi | MV-UMI | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | single_arm |
| meituan_libero_x | Meituan LIBERO-X | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| mimicgen | MimicGen | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | single_arm |
| mobile_aloha | Mobile ALOHA | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| nvidia_gr00t_teleop_g1 | NVIDIA GR00T-Teleop-G1 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | humanoid |
| nvidia_gr00t_x_embodiment_sim | NVIDIA GR00T-X-Embodiment-Sim | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation |  |
| nvidia_locomanipulation_grail | NVIDIA Locomanipulation-GRAIL | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | humanoid |
| nvidia_physicalai_robotics_manipulation_kitchen | NVIDIA PhysicalAI-Robotics-Manipulation-Kitchen | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | mobile_manipulator |
| nvidia_physicalai_robotics_manipulation_objects | NVIDIA PhysicalAI-Robotics-Manipulation-Objects | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | mobile_manipulator |
| nvidia_physicalai_robotics_manipulation_singlearm | NVIDIA PhysicalAI-Robotics-Manipulation-SingleArm | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
| oakink2 | OAKINK2 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| open_x_embodiment | Open X-Embodiment | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop |  |
| ph2d | PH2D | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| rh20t | RH20T | P2 | not_downloaded | not_converted | not_processed | confirmed | teleop | single_arm |
| rt_1 | RT-1 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| robocoin | RoboCOIN | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop |  |
| robocasa | RoboCasa | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | mobile_manipulator |
| robocook | RoboCook | P2 | not_downloaded | not_converted | not_processed | pending_human_review | autonomous_policy | single_arm |
| robodojo | RoboDojo | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation |  |
| robogene | RoboGene | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | dual_arm |
| robomind | RoboMind | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop |  |
| robonet | RoboNet | P2 | not_downloaded | not_converted | not_processed | pending_human_review | autonomous_policy |  |
| roboomni | RoboOmni | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| roboset | RoboSet | P2 | not_downloaded | not_converted | not_processed | pending_human_review | kinesthetic | single_arm |
| robotwin | RoboTwin | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | dual_arm |
| robovqa | RoboVQA | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human |  |
| roboverse | RoboVerse | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
| taco | TACO | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| teach | TEACh | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| the_colosseum | THE COLOSSEUM | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
| umi_datasets | UMI datasets | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | single_arm |
| vitra | VITRA | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| lerobot_full_folding | lerobot/full_folding | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation |  |
| tencent_hy_embodied_0_5_vla_data | tencent/Hy-Embodied-0.5-VLA-Data | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | dual_arm |

<!-- AUTO-GENERATED TABLE END -->

上表由 `python3 public_datasets_raw/convert_scripts/common/generate_overview_readme.py`
生成，只更新 marker 之间的内容；手动新增数据集或更新状态后重新运行以刷新。

## 完整字段总览

每个数据集在 `DatasetConfig` 里声明的全部字段（不含 `field_sources`/
`suggested_new_enum_values` ——这两个是逐字段的调研引用文本，经过多轮复核后
往往长达几句话，放进表格会让每一行宽到无法阅读；这两个字段仍然完整保存在
`convert_scripts/configs/<id>.yaml` 里，需要溯源时直接查那份 YAML）。

<!-- AUTO-GENERATED FULL TABLE START -->

| id | name | source_url | paper_url | license | raw_format | release_type | collection_method | secondary_collection_methods | is_multi_embodiment | embodiment_class | robot_platform | num_arms | dof_per_arm | dof_per_hand | gripper_type | hand_pose_representation | has_mobile_base | action_space | action_frame | rotation_representation | state_dim | action_dim | fps | fps_variable | num_camera_views | camera_views | has_synchronized_multiview_rig | has_camera_calibration | depth_coverage | additional_modalities | has_language_instruction | num_task_types | urdf_available | urdf_source | expected_size_gb | expected_num_episodes | expected_duration_hours | num_subjects | num_scenes | num_objects | review_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x_world_model_dataset | 1X World Model Dataset | https://huggingface.co/datasets/1x-technologies/world_model_raw_data | https://www.1x.tech/discover/1x-world-model | CC-BY-NC-SA-4.0 | Custom | fixed_episode_dataset | teleop |  | false | humanoid | 1x_eve | 2 | 7 |  | unknown | none | true | mixed |  | none | 25 |  | 30.0 | false | 1 | head | false | false | none |  | false |  | true | manufacturer_official | 23.97 |  | 104.67 |  |  |  | pending_human_review |
| alfred | ALFRED | https://askforalfred.com | https://arxiv.org/abs/1912.01734 | MIT | Custom | fixed_episode_dataset | simulation |  | false | mobile_manipulator | virtual_agent |  |  |  | none | none | true | discrete_symbolic |  | none |  |  | 5.0 |  | 1 | head | false |  | none |  | true | 7 | false | not_found | 109.0 | 8055 | 23.8 |  | 120 | 84 | pending_human_review |
| arcap | ARCap | https://huggingface.co/datasets/Ericcsr/ARCap | https://arxiv.org/abs/2410.08464 | CC-BY-4.0 | HDF5 | fixed_episode_dataset | ar_haptic_guided_synthesis |  | true | single_arm | franka_panda | 1 | 7 | 16 | dexterous_hand | joint_angles | false | mixed | absolute | quaternion |  |  |  |  | 1 | head | false | true | full |  | false | 4 | true | dataset_repo | 103.8 |  |  |  |  |  | pending_human_review |
| agibot_digital_world | AgiBot Digital World | https://huggingface.co/datasets/agibot-world/AgiBotDigitalWorld |  | CC-BY-NC-SA-4.0 | Custom | fixed_episode_dataset | simulation | human_to_robot_synthesis; scene_asset_curation | false | dual_arm | agibot_g1 | 2 | 7 | 2 | parallel_jaw | none | true | joint_position | delta | none | 22 | 22 | 30.0 | false | 8 | head; left_wrist; right_wrist; other | true | true | partial |  | true | 8 |  |  | 7635.47 | 48984 |  |  | 5 | 180 | pending_human_review |
| agibot_world | AgiBot-World | https://huggingface.co/datasets/agibot-world/AgiBotWorld2026 |  | CC-BY-NC-SA-4.0 | LeRobot | fixed_episode_dataset | teleop | simulation | false | dual_arm | agibot_g2 | 2 |  |  |  |  | true | mixed |  |  |  |  |  |  | 9 | head; left_wrist; right_wrist | true | true | partial |  | true | 29 |  |  | 9362.16 |  |  |  |  |  | pending_human_review |
| assembly101 | Assembly101 | https://assembly-101.github.io/ | https://arxiv.org/abs/2203.14712 | CC-BY-NC-4.0 | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  | 2 |  |  | none | keypoints_3d | false |  |  |  |  |  | 60.0 | false | 12 | head; top; side | true | true | none |  | true | 101 | false |  |  | 362 | 513.0 | 53 | 1 | 101 | pending_human_review |
| behavior_robot_suite | BEHAVIOR Robot Suite | https://behavior-robot-suite.github.io/ | https://arxiv.org/abs/2503.05652 | MIT | HDF5 | fixed_episode_dataset | teleop |  | false | mobile_manipulator | galaxea_r1 | 2 | 6 |  | parallel_jaw | none | true | mixed | absolute | none | 21 | 21 | 10.0 | false | 3 | head; left_wrist; right_wrist | true | true | full |  | false | 5 | true | dataset_repo | 232.68 | 561 | 20.82 |  |  |  | pending_human_review |
| behavior_1k | BEHAVIOR-1K | https://behavior.stanford.edu/ | https://arxiv.org/abs/2403.09227 | MIT | LeRobot | fixed_episode_dataset | teleop | simulation | false | mobile_manipulator | r1pro | 2 | 7 |  | parallel_jaw | none | true | mixed | absolute | quaternion | 61 | 23 | 30.0 | false | 3 | head; left_wrist; right_wrist |  | true | full |  | true | 100 |  |  | 3270.0 | 20000 | 1950.0 |  | 7 |  | pending_human_review |
| brmdata | BRMData | https://embodiedrobot.github.io/BRMData.html | https://arxiv.org/abs/2405.18860 | MIT | HDF5 | fixed_episode_dataset | teleop |  | false | mobile_manipulator | arx5 | 2 | 7 |  | parallel_jaw | none | true | joint_position | absolute | none | 14 | 14 | 60.0 | true | 3 | third_person; left_wrist; right_wrist | true |  | full | force_torque | false | 10 |  |  |  | 500 | 1.92 | 3 |  |  | confirmed |
| bridgedata_v2 | BridgeData V2 | https://rail-berkeley.github.io/bridgedata/ | https://arxiv.org/abs/2308.12952 | CC-BY-4.0 | Custom | fixed_episode_dataset | teleop | autonomous_policy | false | single_arm | widowx | 1 | 6 |  | parallel_jaw | none | false | eef_pose | delta | euler_xyz | 7 | 7 | 5.0 | false | 4 | third_person; wrist; other | false |  | partial |  | true | 13 | true | manufacturer_official | 441.0 | 60096 | 126.87 | 6 | 24 |  | pending_human_review |
| calvin | CALVIN | https://github.com/mees/calvin | https://arxiv.org/abs/2112.03227 | MIT | Custom | fixed_episode_dataset | teleop | simulation | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw |  | false | eef_pose | both | euler_xyz | 15 | 7 | 30.0 | false | 2 | third_person; wrist | true | true | full | tactile | true | 34 | true | dataset_repo | 656.0 | 20000 | 24.0 | 3 | 4 | 3 | pending_human_review |
| droid | DROID | https://droid-dataset.github.io/ | https://arxiv.org/abs/2403.12945 | CC-BY-4.0 | RLDS | fixed_episode_dataset | teleop |  | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw | none | false | mixed | both | euler_xyz |  | 7 | 15.0 | false | 3 | third_person; left_wrist | true | true | none |  | true | 86 | true | manufacturer_official | 1834.75 | 92233 | 350.0 | 50 | 564 |  | pending_human_review |
| dexcap | DexCap | https://huggingface.co/datasets/chenwangj/DexCap-Data | https://arxiv.org/abs/2403.07788 | CC-BY-4.0 | HDF5 | fixed_episode_dataset | human_to_robot_synthesis |  | false | human_hand | franka_panda | 2 | 7 | 16 | dexterous_hand | joint_angles | false | mixed | absolute | rotation_matrix |  |  | 20.0 | true | 1 | body_worn | false | true | full |  | false | 2 | true | dataset_repo | 107.83 | 34 | 1.5 |  |  |  | pending_human_review |
| dexmimicgen | DexMimicGen | https://dexmimicgen.github.io/ | https://arxiv.org/abs/2410.24185 | CC-BY-NC-SA-4.0 | HDF5 | fixed_episode_dataset | human_to_robot_synthesis |  | true | dual_arm | franka_panda | 2 |  | 6 | dexterous_hand | joint_angles | false | mixed | both | mixed |  |  | 20.0 | false | 3 | third_person; left_wrist; right_wrist | true | true | none |  | false | 9 | true | community_repo | 59.9 | 21000 | 155.6 |  | 9 |  | pending_human_review |
| dobb_e | Dobb-E | https://dobb-e.com | https://arxiv.org/abs/2311.16098 | Unknown | Custom | fixed_episode_dataset | umi |  | false | mobile_manipulator | hello_robot_stretch | 1 | 6 |  | parallel_jaw | none | true | eef_pose | delta | axis_angle |  | 7 | 30.0 | false | 1 | wrist | false | false | full |  | false | 8 | true | manufacturer_official | 77.0 | 5620 | 13.0 |  | 216 |  | pending_human_review |
| epic_kitchens_100 | EPIC-KITCHENS-100 | https://epic-kitchens.github.io/ | https://arxiv.org/abs/2006.13256 | custom_research_eula | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  |  |  |  |  | none |  |  |  |  |  |  | 50.0 | true | 1 | head | false | false | none | audio; imu | true |  | false | not_found | 1850.0 | 700 | 100.0 | 37 | 45 | 300 | pending_human_review |
| ego_exo4d | Ego-Exo4D | https://ego-exo4d-data.org/ | https://arxiv.org/abs/2311.18259 | custom_research_eula | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  | 2 |  |  | none | keypoints_3d | false |  |  |  |  |  |  | true |  | head; third_person | true | true | partial | audio; eye_gaze; imu; point_cloud_3d_scan | true | 8 | false | not_found |  | 5035 | 1286.3 | 740 | 123 |  | pending_human_review |
| ego4d | Ego4d | https://ego4d-data.org/ | https://arxiv.org/abs/2110.07058 | custom_research_eula | Custom | fixed_episode_dataset | egocentric_human |  | false | human_full_body |  |  |  |  | none | none |  |  |  |  |  |  | 30.0 | false | 1 | head | true | false | partial | audio; eye_gaze | true |  | false | not_found | 7100.0 |  | 3670.0 | 931 |  | 4336 | pending_human_review |
| egodex | EgoDex | https://github.com/apple/ml-egodex | https://arxiv.org/abs/2505.11709 | CC-BY-NC-ND-4.0 | HDF5 | fixed_episode_dataset | egocentric_human |  | false | human_hand |  | 2 |  |  | none | keypoints_3d | false | mixed | absolute | rotation_6d |  | 48 | 30.0 | false | 1 | head | false | true | none |  | true | 194 | false | not_found | 2000.0 | 338000 | 829.0 |  |  |  | pending_human_review |
| egoexolearn | EgoExoLearn | https://egoexolearn.github.io/ | https://arxiv.org/abs/2403.16182 | MIT | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  |  |  |  | none | none |  |  |  |  |  |  | 25.0 | false |  | head; third_person | false | false | none | eye_gaze | true | 8 | false | not_found | 142.7 | 432 | 96.5 |  | 7 | 254 | pending_human_review |
| fastumi | FastUMI | https://huggingface.co/IPEC-COMMUNITY/FastUMI-Data | https://arxiv.org/abs/2409.19499 | MIT | HDF5 | fixed_episode_dataset | umi |  | false | single_arm | other | 1 |  |  | parallel_jaw |  | false | eef_pose | absolute | quaternion | 7 | 7 | 60.0 | false | 1 | wrist | false | false | none |  | false | 22 | true | dataset_repo | 2372.24 | 10000 | 25.0 | 5 |  | 19 | pending_human_review |
| functional_manipulation_benchmark_fmb | Functional Manipulation Benchmark (FMB) | https://functional-manipulation-benchmark.github.io/ | https://arxiv.org/abs/2401.08553 | CC-BY-4.0 | Custom | fixed_episode_dataset | teleop | autonomous_policy | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw | none | false | eef_pose | delta | euler_xyz |  | 7 | 10.0 | false | 4 | side; wrist | true | true | full | force_torque | true | 2 | true | manufacturer_official | 778.0 | 22550 | 31.32 |  |  | 66 | pending_human_review |
| furniturebench | FurnitureBench | https://clvrai.github.io/furniture-bench/ | https://arxiv.org/abs/2305.12821 | MIT | Custom | fixed_episode_dataset | teleop |  | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw | none | false | eef_pose | delta | quaternion | 14 | 8 | 10.0 | false | 2 | front; wrist | true | true | none |  | false | 9 | true | dataset_repo | 1179.0 | 5100 | 219.6 | 2 |  | 8 | pending_human_review |
| gr00t_teleop_sim | GR00T Teleop Sim | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim |  | CC-BY-NC-4.0 | HDF5 | fixed_episode_dataset | teleop | simulation | false | humanoid | fourier_gr1 | 2 | 7 | 6 | dexterous_hand | joint_angles | false | joint_position | absolute | none | 44 | 44 | 20.0 | false | 1 | head | false |  | none |  | true | 24 |  |  | 14.0 | 24000 | 80.84 |  |  |  | pending_human_review |
| galaxea_open_world_dataset | Galaxea Open-World Dataset | https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset | https://arxiv.org/abs/2509.00576 | CC-BY-NC-SA-4.0 | LeRobot | fixed_episode_dataset | teleop |  | false | mobile_manipulator | galaxea_r1_lite | 2 | 6 |  | parallel_jaw | none | true | joint_position | absolute | none | 14 | 14 |  |  | 3 | head; left_wrist; right_wrist |  |  | partial |  | true | 150 | true | manufacturer_official | 13016.3 | 100000 | 500.0 |  | 50 | 1600 | pending_human_review |
| h2o | H2O | https://h2odataset.ethz.ch/ | https://arxiv.org/abs/2104.11181 | custom_research_eula | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  | 2 |  | 48 | none | mano | false | mixed |  | axis_angle |  |  | 30.0 | false | 5 | third_person; head | true | true | full |  | false | 36 | false | not_found |  | 24 | 1.06 | 4 | 3 | 8 | pending_human_review |
| hd_epic | HD-EPIC | https://hd-epic.github.io/ | https://arxiv.org/abs/2502.04144 | CC-BY-NC-4.0 | VRS | fixed_episode_dataset | egocentric_human |  | false | human_hand |  | 2 |  |  | none | none |  |  |  |  |  |  | 30.0 | false | 3 | head | false | true | partial | audio; eye_gaze | true |  | false | not_found | 2394.8 | 156 | 41.3 | 9 | 9 |  | pending_human_review |
| hoi4d | HOI4D | https://hoi4d.github.io/ | https://arxiv.org/abs/2203.01577 | CC-BY-NC-4.0 | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  |  |  | 45 | none | mano |  |  |  | axis_angle |  |  | 15.0 | false | 1 | head | false | true | full |  | true | 54 | false | not_found |  | 4000 | 22.2 | 9 | 610 | 800 | pending_human_review |
| hot3d | HOT3D | https://facebookresearch.github.io/hot3d/ | https://arxiv.org/abs/2411.19167 |  | VRS | fixed_episode_dataset | egocentric_human |  | false | human_hand |  |  |  |  | none | mano |  |  |  |  |  |  | 30.0 | false |  | head | true | true | none | eye_gaze; point_cloud_3d_scan | false | 4 | false |  |  | 3832 | 13.88 | 19 | 4 | 33 | pending_human_review |
| libero | LIBERO | https://libero-project.github.io/datasets | https://arxiv.org/abs/2306.03310 | CC-BY-4.0 | HDF5 | fixed_episode_dataset | teleop | simulation | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw |  | false | eef_pose | delta | axis_angle | 9 | 7 | 20.0 | false | 2 | third_person; wrist | true | true | none |  | true | 130 | false | not_found | 100.0 | 6500 | 14.0 |  |  |  | pending_human_review |
| libero_plus | LIBERO-plus | https://huggingface.co/datasets/Sylvest/LIBERO-plus | https://arxiv.org/abs/2510.13626 | MIT | RLDS | fixed_episode_dataset | simulation |  | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw |  | false | eef_pose | delta | axis_angle | 8 | 7 | 20.0 | false | 2 | front; wrist | true | true | none |  | true | 40 | true | manufacturer_official | 75.5 | 14347 | 31.1 |  |  |  | pending_human_review |
| language_table | Language Table | https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table/0.0.1;tab=objects?prefix=&forceOnObjectsSortingFiltering=false | https://arxiv.org/abs/2210.06407 | Apache-2.0 | RLDS | fixed_episode_dataset | teleop |  | false | single_arm | other | 1 | 6 |  | none | none | false | eef_pose | delta | none | 2 | 2 | 5.0 | false | 1 | third_person | false | false | none |  | true |  | true | manufacturer_official | 428.67 | 442226 | 3865.0 | 11 | 1 | 8 | pending_human_review |
| mv_umi | MV-UMI | https://huggingface.co/datasets/omarrayyann/mv-umi | https://arxiv.org/abs/2509.18757 | Unknown | Custom | fixed_episode_dataset | umi |  | false | single_arm | franka_panda | 1 | 7 |  | three_jaw |  | false | eef_pose | relative_trajectory | axis_angle | 7 | 7 | 59.94 | false | 2 | third_person; wrist | false | false | none |  | false | 3 | false | not_found | 58.6 | 916 | 2.4 |  |  |  | pending_human_review |
| meituan_libero_x | Meituan LIBERO-X | https://meituan.github.io/LIBERO-X/ | https://arxiv.org/abs/2602.06556 | CC-BY-4.0 | LeRobot | fixed_episode_dataset | teleop | simulation | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw |  | false | eef_pose | delta |  | 8 | 7 | 10.0 | false | 2 | third_person; wrist | true | true | none |  | true | 600 | false | not_found | 101.45 | 2520 | 24.7 |  | 100 |  | pending_human_review |
| mimicgen | MimicGen | https://github.com/NVlabs/mimicgen | https://arxiv.org/abs/2310.17596 | CC-BY-4.0 | HDF5 | fixed_episode_dataset | human_to_robot_synthesis | teleop | true | single_arm | franka_panda | 1 | 7 |  | parallel_jaw | none | false | eef_pose | delta | axis_angle |  | 7 | 20.0 | false | 2 | third_person; wrist | true | true | none |  | false | 12 | true | manufacturer_official | 149.0 | 50120 |  |  | 12 |  | pending_human_review |
| mobile_aloha | Mobile ALOHA | https://mobile-aloha.github.io/ | https://arxiv.org/abs/2401.02117 | MIT | HDF5 | fixed_episode_dataset | teleop |  | false | mobile_manipulator | viperx | 2 | 7 |  | parallel_jaw |  | true | mixed | absolute | none | 14 | 16 | 50.0 | false | 3 | top; left_wrist; right_wrist |  |  | none | force_torque | true | 6 | true | manufacturer_official | 30.7 | 278 | 2.61 | 2 |  |  | pending_human_review |
| nvidia_gr00t_teleop_g1 | NVIDIA GR00T-Teleop-G1 | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 |  | CC-BY-4.0 | LeRobot | fixed_episode_dataset | teleop |  | false | humanoid | unitree_g1 | 2 | 7 | 7 | dexterous_hand | joint_angles | false | joint_position | absolute | none | 43 | 43 | 20.0 | false | 1 | head | false | false | none |  | true | 4 |  |  | 0.4 | 1095 | 1.72 |  |  | 4 | pending_human_review |
| nvidia_gr00t_x_embodiment_sim | NVIDIA GR00T-X-Embodiment-Sim | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim | https://arxiv.org/abs/2503.14734 | CC-BY-4.0 | LeRobot | fixed_episode_dataset | simulation | teleop | true |  |  |  | 7 |  |  | joint_angles |  | mixed | mixed_delta_absolute |  |  |  |  | true |  | head; front; left_wrist; right_wrist; wrist; side |  |  |  |  | true | 82 |  |  |  | 345023 | 1236.67 |  |  |  | pending_human_review |
| nvidia_locomanipulation_grail | NVIDIA Locomanipulation-GRAIL | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL | https://arxiv.org/abs/2606.05160 | Apache-2.0 | Custom | fixed_episode_dataset | human_to_robot_synthesis | simulation | false | humanoid | unitree_g1 | 2 | 7 | 7 | dexterous_hand | joint_angles | false | joint_position | absolute | quaternion |  | 29 | 25.0 | false | 1 | third_person | false | true | none |  | false | 6 | false | not_found |  | 22189 | 61.52 |  |  | 6857 | pending_human_review |
| nvidia_physicalai_robotics_manipulation_kitchen | NVIDIA PhysicalAI-Robotics-Manipulation-Kitchen | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen |  | CC-BY-4.0 | LeRobot | fixed_episode_dataset | simulation |  | false | mobile_manipulator | kinova_gen3 | 2 | 7 |  | parallel_jaw |  | true | joint_position | absolute |  | 13 | 34 | 50.0 | false | 5 | third_person; third_person; left_wrist; right_wrist; head | true | true | full | semantic_segmentation | false | 8 | true | manufacturer_official | 11.4 | 874 | 2.25 |  |  |  | pending_human_review |
| nvidia_physicalai_robotics_manipulation_objects | NVIDIA PhysicalAI-Robotics-Manipulation-Objects | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects |  | custom_research_eula | LeRobot | fixed_episode_dataset | simulation |  | false | mobile_manipulator | kinova_gen3 | 2 | 7 |  | parallel_jaw |  | true | joint_position | absolute | none | 13 | 34 | 50.0 | false | 5 | third_person; third_person; head; left_wrist; right_wrist | true | true | full | semantic_segmentation | false | 3 | true | manufacturer_official | 4.0 | 540 | 0.72 |  |  |  | pending_human_review |
| nvidia_physicalai_robotics_manipulation_singlearm | NVIDIA PhysicalAI-Robotics-Manipulation-SingleArm | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm |  | CC-BY-4.0 | LeRobot | fixed_episode_dataset | simulation |  | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw |  | false | mixed | delta | mixed |  |  | 30.0 | false | 2 | third_person; wrist | true | true | partial |  | false | 6 | true | manufacturer_official | 15.2 | 38386 | 30.92 |  |  |  | pending_human_review |
| oakink2 | OAKINK2 | https://oakink.net/v2 | https://arxiv.org/abs/2403.19417 | CC-BY-SA-4.0 | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  | 2 |  | 48 | none | mano | false |  |  | quaternion |  |  | 30.0 | false | 4 | third_person; head | true | true | none |  | true | 60 | false | not_found | 2130.0 | 627 | 9.28 | 9 | 4 | 75 | pending_human_review |
| open_x_embodiment | Open X-Embodiment | https://github.com/google-deepmind/open_x_embodiment | https://arxiv.org/abs/2310.08864 | CC-BY-4.0 | RLDS | fixed_episode_dataset | teleop | autonomous_policy | true |  |  |  |  |  | mixed |  |  | mixed |  |  |  |  |  | true |  |  |  |  | partial |  | true | 527 |  |  | 8964.94 | 2419193 |  |  |  |  | pending_human_review |
| ph2d | PH2D | https://huggingface.co/datasets/RogerQi/PH2D | https://arxiv.org/abs/2503.13441 | MIT | HDF5 | fixed_episode_dataset | egocentric_human | teleop; simulation | true | human_hand | unitree_h1 | 2 | 7 | 6 | dexterous_hand | keypoints_3d | false | mixed | absolute | rotation_6d | 128 | 128 | 30.0 | false | 1 | head | false | true | none |  | true | 6 | true | dataset_repo | 15.83 | 28376 | 34.1 |  |  |  | pending_human_review |
| rh20t | RH20T | https://rh20t.github.io/#download | https://arxiv.org/abs/2307.00595 |  | Custom | fixed_episode_dataset | teleop | egocentric_human | true | single_arm |  | 1 |  |  | parallel_jaw |  | false |  |  | quaternion |  |  | 10.0 | true |  | third_person; wrist | true | true | partial | force_torque; tactile; audio | true | 147 | true | dataset_repo | 40000.0 | 110000 | 1111.11 |  | 10 |  | confirmed |
| rt_1 | RT-1 | https://console.cloud.google.com/storage/browser/gresearch/rt-1-data-release;tab=objects?prefix=&forceOnObjectsSortingFiltering=false | https://arxiv.org/abs/2212.06817 | CC-BY-4.0 | RLDS | fixed_episode_dataset | teleop |  | false | mobile_manipulator | everyday_robots_arm | 1 | 7 |  | parallel_jaw | none | true | mixed | delta | euler_xyz |  | 13 | 3.0 | false | 1 | head | false | false | none |  | true | 744 | false | not_found | 118.93 | 87212 |  |  | 3 | 17 | pending_human_review |
| robocoin | RoboCOIN | https://huggingface.co/RoboCOIN/datasets | https://arxiv.org/abs/2511.17441 | Apache-2.0 | LeRobot | fixed_episode_dataset | teleop |  | true |  |  | 2 |  |  | mixed |  |  | mixed |  | euler_xyz |  |  |  |  |  | head; left_wrist; right_wrist; third_person | true |  | none |  | true | 421 | false | not_found |  | 180000 |  |  | 16 | 432 | pending_human_review |
| robocasa | RoboCasa | https://robocasa.ai/docs/build/html/introduction/overview.html | https://arxiv.org/abs/2603.04356 | CC-BY-4.0 | LeRobot | fixed_episode_dataset | human_to_robot_synthesis | teleop | false | mobile_manipulator | franka_panda | 1 | 7 |  | parallel_jaw |  | true | mixed | delta | axis_angle | 16 | 12 | 20.0 | false | 3 | third_person; wrist | true | true | none |  | true | 365 | true | manufacturer_official |  | 655000 | 2227.0 |  | 2510 | 3200 | pending_human_review |
| robocook | RoboCook | https://hshi74.github.io/robocook/ | https://arxiv.org/abs/2306.14447 | MIT | ROS_bag | fixed_episode_dataset | autonomous_policy |  | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw | none | false | eef_pose |  | single_axis_angle |  |  |  |  | 4 | third_person | true | true | full |  | false | 2 | true | manufacturer_official |  |  | 4.7 |  | 1 | 1 | pending_human_review |
| robodojo | RoboDojo | https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo | https://arxiv.org/abs/2607.04434 | Apache-2.0 | HDF5 | fixed_episode_dataset | simulation | teleop; scripted | true |  |  | 2 | 6 | 1 | parallel_jaw | none | false | joint_position | absolute | quaternion | 28 | 14 | 25.0 | false | 3 | head; left_wrist; right_wrist | true | true | none |  | true | 60 | true | dataset_repo | 797.1 | 5320 | 38.57 | 4 |  |  | pending_human_review |
| robogene | RoboGene | https://huggingface.co/datasets/X-Humanoid/RoboGene | https://arxiv.org/abs/2602.16444 | Apache-2.0 | LeRobot | fixed_episode_dataset | teleop | scene_asset_curation | false | dual_arm | franka_fr3 |  |  | 1 | parallel_jaw | none | false | mixed | absolute | quaternion |  |  | 30.0 | false | 6 | front; side; top; left_wrist; right_wrist | true |  | full |  | true | 271 | false | not_found | 431.57 | 3752 | 13.95 |  | 8 |  | pending_human_review |
| robomind | RoboMIND | https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND | https://arxiv.org/abs/2412.13877 | Apache-2.0 | HDF5 | fixed_episode_dataset | teleop | simulation | true |  |  |  |  | 6 | mixed | joint_angles | false | joint_position |  |  |  |  |  | true |  | top; side; head; body_worn; left_wrist; right_wrist; front | true |  | partial |  | true | 6 | true | manufacturer_official | 12279.0 | 107000 | 305.5 |  |  | 96 | pending_human_review |
| robonet | RoboNet | https://www.robonet.wiki/ | https://arxiv.org/abs/1910.11215 | CC-BY-4.0 | HDF5 | fixed_episode_dataset | autonomous_policy |  | true |  |  |  |  |  | parallel_jaw | none | false | eef_pose | delta |  |  |  |  |  |  | third_person | true | false | none |  | false |  | false | not_found | 36.0 | 162000 |  |  |  |  | pending_human_review |
| roboomni | RoboOmni | https://huggingface.co/datasets/OpenMOSS-Team/OmniAction | https://arxiv.org/abs/2510.23763 | CC-BY-NC-4.0 | RLDS | fixed_episode_dataset | teleop | synthetic_multimodal_augmentation | true | single_arm |  | 1 |  |  | mixed | none | false | eef_pose | delta | euler_xyz |  | 7 |  |  |  |  |  |  |  | audio | true | 112 | false | not_found | 2710.9 | 141162 | 1764.53 |  |  | 748 | pending_human_review |
| roboset | RoboSet | https://robopen.github.io/roboset/ | https://arxiv.org/abs/2309.01918 | MIT | HDF5 | fixed_episode_dataset | kinesthetic | teleop; autonomous_policy | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw |  | false | joint_position | absolute | none | 8 | 8 | 5.0 | false | 4 | top; side; wrist | true |  | full | force_torque | true | 38 | true | manufacturer_official |  | 30050 | 70.12 |  | 4 |  | pending_human_review |
| robotwin | RoboTwin | https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset | https://arxiv.org/abs/2506.18088 | MIT | HDF5 | fixed_episode_dataset | simulation |  | true | dual_arm | agilex_aloha | 2 | 6 |  | parallel_jaw | none | false | mixed | absolute | quaternion |  |  | 16.67 | false | 3 | head; left_wrist; right_wrist | true | true | none |  | true | 50 | true | dataset_repo | 1360.0 | 100000 |  |  |  | 731 | pending_human_review |
| robovqa | RoboVQA | https://robovqa.github.io/ | https://arxiv.org/abs/2311.00899 | CC-BY-4.0 | TFRecord | fixed_episode_dataset | egocentric_human | teleop | true |  |  | 1 |  |  |  | none | true |  |  | none |  |  |  |  | 1 |  | false | false | none |  | true | 8 | false | not_found |  | 5246 | 238.0 |  | 3 | 2862 | pending_human_review |
| roboverse | RoboVerse | https://roboverseorg.github.io | https://arxiv.org/abs/2504.18904 | Apache-2.0 | Custom | generation_framework | simulation | teleop; autonomous_policy; scene_asset_curation | true | single_arm |  | 1 |  | 20 | parallel_jaw | joint_angles | false | mixed | both | mixed |  |  |  |  |  |  |  | true |  |  | true | 276 | true | dataset_repo | 44.9 | 510500 |  |  |  | 5500 | pending_human_review |
| taco | TACO | https://taco2024.github.io/ | https://arxiv.org/abs/2401.08399 | Unknown | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  | 2 |  | 48 | none | mano | false |  |  | axis_angle |  |  | 30.0 | false | 13 | third_person; head | true | true | partial |  | false | 15 | false | not_found |  | 2317 | 48.1 | 14 | 1 | 196 | pending_human_review |
| teach | TEACh | https://github.com/alexa/teach | https://arxiv.org/abs/2110.00534 | CDLA-Sharing-1.0 | Custom | fixed_episode_dataset | teleop | simulation | false | mobile_manipulator | virtual_agent |  |  |  | none | none | true | discrete_symbolic |  | none |  |  |  | true | 2 | head; other | false | false |  |  | true | 12 | false | not_found |  | 3047 |  |  | 109 |  | pending_human_review |
| the_colosseum | THE COLOSSEUM | https://robot-colosseum.github.io/ | https://arxiv.org/abs/2402.08191 | MIT | Custom | fixed_episode_dataset | simulation |  | false | single_arm | franka_panda | 1 | 7 |  | parallel_jaw |  | false | eef_pose | absolute | quaternion |  | 8 |  |  | 4 | front; third_person; wrist | true | true | full |  | true | 20 | true | manufacturer_official | 280.6 | 2000 |  |  |  |  | pending_human_review |
| umi_datasets | UMI datasets | https://umi-gripper.github.io | https://arxiv.org/abs/2402.10329 | MIT | Custom | fixed_episode_dataset | umi |  | false | single_arm | other | 1 |  |  | parallel_jaw | none | false | eef_pose | relative_trajectory | rotation_6d | 10 | 10 | 59.94 | true | 1 | wrist | false | true | none |  | false | 1 | false | not_found | 17.99 | 1400 |  | 3 | 30 | 15 | pending_human_review |
| vitra | VITRA | https://microsoft.github.io/VITRA/ | https://arxiv.org/abs/2510.21571 | MIT | Custom | fixed_episode_dataset | egocentric_human |  | false | human_hand |  | 2 |  | 45 | none | mano | false | mixed | mixed_delta_absolute | euler_xyz | 122 | 102 |  | true | 1 | head | false | true | none |  | true |  | false | not_found | 100.0 | 1222918 |  |  |  |  | pending_human_review |
| lerobot_full_folding | lerobot/full_folding |  |  |  |  | fixed_episode_dataset | simulation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 30.0 |  |  |  |  |  |  |  |  |  |  |  |  | 5688 | 130.82 |  |  |  | pending_human_review |
| tencent_hy_embodied_0_5_vla_data | tencent/Hy-Embodied-0.5-VLA-Data | https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data | https://arxiv.org/abs/2606.14409 | CC-BY-4.0 |  | fixed_episode_dataset | umi |  | false | dual_arm | other | 2 |  |  | parallel_jaw |  | false |  |  | quaternion | 16 | 2 | 30.0 | false | 3 | head; left_wrist; right_wrist |  |  | none |  | true | 70 | false | not_found | 18800.0 | 250304 | 2163.0 |  |  |  | pending_human_review |

<!-- AUTO-GENERATED FULL TABLE END -->

上表由 `python3 public_datasets_raw/convert_scripts/common/generate_full_export.py`
生成，只更新 marker 之间的内容；数据变化后重新运行即可刷新。
