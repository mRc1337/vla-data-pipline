# VLA 数据集注册表字段参考

本文档整理 `embodied_datasets/public_datasets_raw/convert_scripts/common/schema.py`
里两个 pydantic 模型的全部字段：`RegistryEntry`（`datasets_registry.yaml` 总览表，
实测值）和 `DatasetConfig`（`convert_scripts/configs/<id>.yaml` 每数据集详细配置，
声明值）。字段本身的权威定义永远是 `schema.py`；本文档是给人看的说明，如有出入
以代码为准。

## RegistryEntry（`datasets_registry.yaml`）

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
| `lerobot_v2_1_local_path` | str，可空 | 清洗完成数据相对`public_datasets/lerobot_v2_1/`的本地路径 |
| `storage_size_gb` | float，可空 | 实测占用空间（GB） |
| `num_episodes` | int，可空 | 实测episode数 |
| `num_frames` | int，可空 | 实测帧数 |
| `duration_hours` | float，可空 | 实测总时长（小时） |

### `Priority`
`P0` / `P1` / `P2` —— 优先级，数字越小越优先。

### `DownloadStatus`
`not_downloaded` 未下载 / `downloading` 下载中 / `completed` 已完成。

### `IntegrityStatus`
`not_verified` 未校验 / `verified` 校验通过 / `failed` 校验失败——对应设计文档
第4节的完整性校验（体量核对+格式可解析性+视频抽样解码）。

### `ConvertStatus`
`not_converted` / `converting` / `converted` / `failed` —— `convert_scripts`
（raw → lerobot_v2_1_staging）的执行状态。

### `ProcessStatus`
`not_processed` / `processing` / `processed` / `failed` —— `process_scripts`
（五阶段清洗+三项质检+统一表示）的执行状态。

---

## DatasetConfig（`convert_scripts/configs/<id>.yaml`）

Onboarding 时调研得到，代表"声明的事实"。除 `id`/`name` 外全部可空——没有调研到
或不适用的字段应该留空，不应该编造。

### 基础信息

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | str | 同 RegistryEntry |
| `name` | str | 同 RegistryEntry |
| `source_url` | str，可空 | 官方主页/下载页链接 |
| `paper_url` | str，可空 | 对应论文链接，跟`source_url`分开存 |

### 许可与格式

| 字段 | 类型 | 含义 |
|---|---|---|
| `license` | enum | 见下方 `LicenseEnum` |
| `raw_format` | enum | 见下方 `RawFormat` |
| `release_type` | enum | 见下方 `ReleaseType` |

#### `LicenseEnum`
- `MIT` / `Apache-2.0` / `BSD-3-Clause` / `GPL-3.0` / `CC0-1.0` —— 常见开源协议
- `CC-BY-4.0` / `CC-BY-NC-4.0` / `CC-BY-NC-SA-4.0` / `CC-BY-NC-ND-4.0` / `CC-BY-SA-4.0` —— 知识共享协议家族，区别在"是否允许商用/是否要求署名/是否允许衍生/是否要求以同协议共享"
- `CDLA-Sharing-1.0` —— Community Data License Agreement，数据集专用的共享协议（跟代码用的开源协议不是一回事）
- `Proprietary` —— 私有/未公开协议
- `custom_research_eula` —— 各机构自己定制的门禁研究协议（如Ego4D、ETH各数据集的EULA），一次性法律文本不可复用，具体条款记在`field_sources`
- `Unknown` —— 确实没找到

#### `RawFormat`
- `RLDS` —— TensorFlow Datasets的RLDS封装
- `HDF5` —— HDF5文件
- `LeRobot` —— 已经是LeRobot格式
- `ROS_bag` —— ROS bag录制
- `MCAP` —— MCAP格式（ROS2常用）
- `TFRecord` —— 原生TFRecord（非RLDS封装）
- `VRS` —— Meta Project Aria的传感器容器格式
- `Custom` —— 各家自定义格式（npy/pkl/自定义目录结构等），目前占比最高

#### `ReleaseType`
- `fixed_episode_dataset` —— 固定episode数的常规数据集（绝大多数）
- `generation_framework` —— 生成框架，理论上能无限生成新任务/轨迹，没有固定语料（如RoboGen、GenSim2）
- `scene_platform` —— 场景/资产平台，提供可交互场景而非机器人轨迹本身（如GRUtopia）
- `rl_benchmark_env` —— 纯RL训练环境，rollout实时跑出来，没有预录制轨迹（如HumanoidBench）

### 采集方式与本体分类

| 字段 | 类型 | 含义 |
|---|---|---|
| `collection_method` | enum | 见下方 `CollectionMethod`（主要采集方式，单值） |
| `secondary_collection_methods` | list[enum] | 次要/补充采集方式（同一数据集可以有多个） |
| `is_multi_embodiment` | bool，可空 | 是否同一份发布横跨多种具身形态——如果是，`embodiment_class`/`robot_platform`应该留空，不要强行选一个 |
| `embodiment_class` | enum | 见下方 `EmbodimentClass` |
| `robot_platform` | enum | 见下方 `RobotPlatform`（开放列表，会持续扩充） |

#### `CollectionMethod`
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

#### `EmbodimentClass`
- `single_arm` —— 单臂固定基座
- `dual_arm` —— 双臂固定基座（工业臂式布局，无躯干概念）
- `half_humanoid` —— 双臂+躯干无腿部（如AgiBot G1、Galbot G1）
- `humanoid` —— 完整人形（有腿/移动能力）
- `mobile_manipulator` —— 移动底盘+机械臂
- `human_hand` —— 裸手（只跟踪手部，无全身姿态）
- `human_full_body` —— 全身姿态（跟只覆盖手部的`human_hand`不是一回事）
- `quadruped` —— 四足

#### `RobotPlatform`
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

### 机械结构

| 字段 | 类型 | 含义 |
|---|---|---|
| `num_arms` | int(0-2)，可空 | 手臂数 |
| `dof_per_arm` | int，可空 | 每臂自由度 |
| `dof_per_hand` | int，可空 | 每只灵巧手/末端自由度（跟`dof_per_arm`互补，不是替代） |
| `gripper_type` | enum | 见下方 `GripperType` |
| `hand_pose_representation` | enum | 见下方 `HandPoseRepresentation`——描述手指/手掌**形状**编码，跟描述手腕**朝向**编码的`rotation_representation`是互补关系，带灵巧手的数据集两个字段通常都会填 |
| `has_mobile_base` | bool，可空 | 是否有移动底盘 |

#### `GripperType`
- `parallel_jaw` —— 二指平行夹爪
- `dexterous_hand` —— 多指灵巧手
- `three_jaw` —— 三指夹爪
- `cage_pinch` —— 缆线操作专用非对称二指机制（一指"笼住"允许滑动，另一指"夹紧"）
- `suction` —— 吸盘
- `mixed` —— 同一发布内混用多种末端执行器（多本体聚合数据集）
- `none` —— 无末端执行器（裸手数据集）
- `unknown` —— 有末端执行器但型号不明

#### `HandPoseRepresentation`
- `mano` —— MANO参数化手部模型
- `keypoints_3d` —— 3D关键点
- `joint_angles` —— 关节角度
- `none` —— 不适用（非灵巧手/非手部追踪数据集）

### 动作与坐标系表示

| 字段 | 类型 | 含义 |
|---|---|---|
| `action_space` | enum | 见下方 `ActionSpace` |
| `action_frame` | enum | 见下方 `ActionFrame` |
| `rotation_representation` | enum | 见下方 `RotationRepresentation` |
| `state_dim` | int，可空 | 状态向量维度 |
| `action_dim` | int，可空 | 动作向量维度 |

#### `ActionSpace`
- `joint_position` / `joint_velocity` / `joint_torque` —— 关节空间控制
- `eef_pose` —— 末端位姿控制
- `discrete_symbolic` —— 离散动作词表（AI2-THOR式，非连续控制）
- `mixed` —— 同一发布内混用多种动作空间
- `unknown` —— 有动作但空间类型不明

#### `ActionFrame`
- `delta` —— 相对上一步的增量
- `absolute` —— 绝对坐标
- `both` —— 数据里同时提供两种
- `relative_trajectory` —— 相对轨迹起点（如UMI的SLAM相对定位，跟逐步`delta`不同）
- `mixed_delta_absolute` —— 同一动作向量里部分维度delta、部分absolute（如平移delta+旋转absolute）
- `unknown`

#### `RotationRepresentation`
- `euler_xyz` / `quaternion` / `rotation_6d` / `axis_angle` / `rotation_matrix` —— 标准3D朝向编码
- `single_axis_angle` —— 只有单一旋转角（如仅yaw，不是完整3D朝向）
- `mixed` —— 同一发布内混用多种编码（多本体/多子集聚合数据集）
- `none` —— 无旋转编码（如纯关节控制、离散动作）
- `unknown`

### 采集频率

| 字段 | 类型 | 含义 |
|---|---|---|
| `fps` | float，可空 | 采集帧率 |
| `fps_variable` | bool，可空 | 帧率是否逐数据集/逐子集变化 |

### 视觉与相机

| 字段 | 类型 | 含义 |
|---|---|---|
| `num_camera_views` | int，可空 | 相机视角数 |
| `camera_views` | list[enum] | 见下方 `CameraView` |
| `has_synchronized_multiview_rig` | bool，可空 | 多相机是否同步组成一个阵列（跟"有哪些视角"是不同维度） |
| `has_camera_calibration` | bool，可空 | 是否提供相机标定参数 |
| `depth_coverage` | enum | 见下方 `DepthCoverage` |

#### `CameraView`
- `third_person` / `head` / `top` / `front` / `side` / `other` —— 常见外部视角
- `left_wrist` / `right_wrist` —— 双臂场景下分左右的手腕相机
- `wrist` —— 单臂通用手腕相机（非分手，跟`left_wrist`/`right_wrist`不同）
- `gripper_jaw` —— 爪部相机（跟臂部安装的`wrist`是不同安装位置，如Spot机器人）
- `body_worn` —— 佩戴在身体上（如胸部相机，跟头戴式`head`不同）
- `worms_eye` —— 仰视相机

#### `DepthCoverage`
`none` 无深度 / `partial` 部分视角有深度 / `full` 全部视角有深度。

### 其他传感模态

| 字段 | 类型 | 含义 |
|---|---|---|
| `additional_modalities` | list[enum] | 见下方 `SensorModality`——可扩展列表，新增模态只需加枚举值，不需要新字段 |

#### `SensorModality`
`force_torque` 力/力矩 / `tactile` 触觉 / `audio` 音频 / `eye_gaze` 眼动 /
`imu` 惯性测量单元 / `semantic_segmentation` 语义分割视频流 /
`point_cloud_3d_scan` 持久3D场景重建点云（跟逐帧`depth_coverage`是不同维度）。

### 语言与任务

| 字段 | 类型 | 含义 |
|---|---|---|
| `has_language_instruction` | bool，可空 | 是否有语言指令标注 |
| `num_task_types` | int，可空 | 任务类型数 |

### URDF

| 字段 | 类型 | 含义 |
|---|---|---|
| `urdf_available` | bool，可空 | 是否能找到URDF |
| `urdf_source` | enum | 见下方 `UrdfSource` |

#### `UrdfSource`
`dataset_repo` 数据集自带 / `manufacturer_official` 厂商官方 /
`community_repo` 社区维护 / `not_found` 找不到。

### 规模（声明值，对应RegistryEntry里的实测值）

| 字段 | 类型 | 含义 |
|---|---|---|
| `expected_size_gb` | float，可空 | 声明的数据体量（GB） |
| `expected_num_episodes` | int，可空 | 声明的episode数 |
| `expected_duration_hours` | float，可空 | 声明的总时长（小时） |
| `num_subjects` | int，可空 | 人类被试/操作员数 |
| `num_scenes` | int，可空 | 场景/环境数 |
| `num_objects` | int，可空 | 交互物体数 |

### 审核与溯源

| 字段 | 类型 | 含义 |
|---|---|---|
| `review_status` | enum | 见下方 `ReviewStatus` |
| `field_sources` | dict[str,str] | 每个已填字段的信息来源（URL或论文章节），调研/复核时必须记录 |
| `suggested_new_enum_values` | dict[str,str] | 实际取值不在现有枚举里时，记录"建议新增什么值+理由"，而不是编造/硬塞进已有值 |

#### `ReviewStatus`
`pending_human_review` 待人工确认（默认值） / `confirmed` 已人工确认——
`convert_scripts`运行前应检查这个字段是`confirmed`，防止调研错误直接污染生产流水线。

---

## 如何保持本文档不过期

本文档是手写的，`schema.py`改动后需要手动同步。如果只是想看"某个字段现在有哪些枚举
值"，比手动维护更新的文档更可靠的方式是直接跑：

```bash
python3 -c "
import sys; sys.path.insert(0, 'embodied_datasets/public_datasets_raw/convert_scripts')
from common.schema import <EnumName>
for m in <EnumName>: print(m.value)
"
```

或者查看 `embodied_datasets/datasets_full_export.csv` 看每个字段在全部82个数据集
里的实际取值分布。
