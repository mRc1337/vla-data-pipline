# VLA 数据预处理与对齐流水线 —— 设计文档

- 日期：2026-07-08
- 状态：草案，待用户审阅

## 1. 背景与目标

将多源公开 VLA / 机器人操作数据集统一转换为 LeRobot v2.1 格式，并按照
[Qwen-RobotManip Technical Report](../../../Qwen_RobotManip.pdf)（Qwen Team,
2026-06-16）描述的数据对齐（alignment）方法论，做清洗、跨本体对齐处理，
产出可直接用于 VLA 训练的高质量数据集，同时建立一套可持续维护、可扩展到
新数据集 / 新格式版本（如未来 lerobot v3.0）的工程规范。

现状：项目目录下已有一份 `VLA Research.xlsx`，记录了 33 个候选数据集的调研
信息（名称、官网/论文链接、下载状态等），但字段稀疏、非结构化，无法直接
驱动自动化脚本。本设计将把它升级为结构化、机器可读的注册体系。

## 2. 参考论文的关键设计（摘要）

论文中与"数据处理流程"直接相关的部分，是本设计的方法论依据：

- **数据来源分类**（Sec 2）：Robot（真机遥操/自主采集）、Human（egocentric
  人手视频，统一转为 MANO 表示）、Human-to-Robot（人手视频合成机器人轨迹）。
  本项目第一阶段只覆盖已有公开机器人数据集的清洗对齐，不做 Human-to-Robot
  合成（详见第 10 节"非目标"）。
- **五阶段状态-动作清洗流水线**（Sec 2.4，对所有数据集统一应用）：
  1. Stage1 突变检测：Savitzky-Golay 平滑 + 残差/加速度/jerk 三重阈值
  2. Stage2 状态-动作趋势对齐：互相关估计时延 + directional agreement 指标
     （阈值通常 0.6-0.7），用于发现时间戳不同步/丢包
  3. Stage3 极值过滤：按 embodiment 类型计算 q1/q99 分位数，裁剪异常帧
     （gripper 维度因双峰分布豁免）
  4. Stage4 关节-末端执行器正向运动学一致性校验：用 URDF + Pinocchio 算 FK，
     修正 TCP 偏移等系统性错误
  5. Stage5 Base 坐标系与末端朝向对齐：统一世界坐标系约定
- **三项跨模态质检**（Sec 2.4）：指令一致性（VLM 多阶段裁决）、视频-状态
  一致性（URDF 投影 vs SAM3 分割 IoU）、视频质量过滤（黑/糊/静止帧）
- **表示层**（Sec 3.2/3.3）：80 维跨本体统一状态-动作向量（每臂 29 维
  joint/eef/gripper/dexterous-hand 分区 + 22 维预留）；相机坐标系下的
  delta pose 动作表示（要求相机内外参标定可用）

## 3. 整体目录结构

```
embodied_datasets/
├── datasets_registry.yaml                # 总览表（取代现有xlsx）
├── public_datasets_raw/
│   ├── <dataset_id>/
│   │   ├── raw/                          # 原始下载数据，格式不动
│   │   └── lerobot_v2_1_staging/         # convert_scripts产出的"未清洗"中间态
│   ├── verify_scripts/
│   │   ├── verify_integrity.py           # 按raw_format分发的完整性校验器
│   │   └── logs/<dataset_id>.log         # 校验失败详情
│   ├── convert_scripts/
│   │   ├── configs/<dataset_id>.yaml     # 数据集描述性配置
│   │   ├── common/                       # 跨数据集复用的转换工具
│   │   └── <dataset_id>.py               # raw -> lerobot_v2_1_staging
│   └── process_scripts/
│       ├── configs/<dataset_id>.yaml     # 清洗对齐阈值配置
│       ├── stage1_sudden_change.py
│       ├── stage2_trend_alignment.py
│       ├── stage3_extreme_value.py
│       ├── stage4_fk_consistency.py
│       ├── stage5_orientation_alignment.py
│       ├── check1_instruction_consistency.py   # 调本地VLM服务
│       ├── check2_video_state_consistency.py   # 调本地SAM3服务 + URDF渲染
│       ├── check3_video_quality.py
│       ├── unify_representation.py       # 80维统一 + camera-frame delta pose
│       └── run_pipeline.py               # 编排以上步骤，生成README.md
├── urdf_assets/<robot_platform>/         # 按机器人型号共享存放
└── public_datasets/
    ├── lerobot_v2_1/<dataset_id>/
    │   ├── README.md                     # 自动生成
    │   ├── meta/   data/   videos/       # 标准LeRobot v2.1 schema
    └── lerobot_v3_0/                     # 预留，未来格式升级
```

设计原则：
- `public_datasets_raw/` 下既有原始数据也有处理代码（convert_scripts /
  process_scripts / verify_scripts），是"输入与工具"侧。
- `public_datasets/` 下只有最终干净、可直接使用的数据产物，不放任何脚本。
- `lerobot_v2_1_staging/` 作为转换与清洗之间的中间态，使 process_scripts
  可以反复调参重跑，无需每次重新跑 convert_scripts。
- `urdf_assets/` 按 `robot_platform` 而非 `dataset_id` 组织，因为同一机器人
  型号会被多个数据集复用。

## 4. 数据完整性校验（下载完成 → convert_scripts 之前）

`verify_scripts/verify_integrity.py` 按 `raw_format` 枚举分发，检查三层：

1. **体量核对**：实际下载的文件总大小 / episode 数，对照配置里的
   `expected_size_gb` / `expected_num_episodes`，抓"下载不全"。
2. **格式可解析性**（按格式通用校验，不用每数据集单写）：
   - RLDS/TFRecord：`tf.data.TFRecordDataset` 完整遍历不抛异常
   - HDF5：`h5py.File` 可打开，必需字段（`action`/`observation`）存在
   - LeRobot：官方 `lerobot` 库自带加载校验（episode索引与parquet/视频对齐）
   - ROS bag/MCAP：`rosbag info`/`mcap info` 验证头信息完整
3. **视频可解码抽样**：随机抽样部分 episode 的视频，用 `ffprobe`/`decord`
   探测解码，抓"体量对但内容损坏/截断"的情况。

结果写回 `datasets_registry.yaml` 的 `integrity_status` 字段；失败详情写入
独立日志文件，不写入 YAML。`convert_scripts` 运行前必须检查
`integrity_status == verified`，否则拒绝执行。

此环节只管"完不完整、能不能打开"，不做数据质量判断——质量清洗完全交给
process_scripts 的五阶段+三质检。

## 5. 数据集注册体系

分两层（总览 + 详细配置），字段尽量枚举化/数值化，避免自由文本导致脚本
无法解析。两层里都有规模字段，但含义不同：**总览表的规模字段是"实测值"**
（下载/转换完成后由脚本写回的实际数字，随流水线推进更新），**详细配置的
`expected_*` 字段是"声明值"**（onboarding 时从官方文档记录的期望值，作为
第 4 节完整性校验的比对基准，onboarding 后基本不变）。

### 5.1 总览表 `datasets_registry.yaml`（实测值，随流水线推进更新）

| 字段 | 类型 |
|---|---|
| `id` | string，唯一slug 如 `droid` |
| `name` | string |
| `priority` | enum: `P0,P1,P2` |
| `download_status` | enum: `not_downloaded,downloading,completed` |
| `integrity_status` | enum: `not_verified,verified,failed` |
| `convert_status` | enum: `not_converted,converting,converted,failed` |
| `process_status` | enum: `not_processed,processing,processed,failed` |
| `raw_local_path` | string，可空，相对 `public_datasets_raw/` |
| `lerobot_v2_1_local_path` | string，可空，相对 `public_datasets/lerobot_v2_1/` |
| `storage_size_gb` | number |
| `num_episodes` | int |
| `num_frames` | int |
| `duration_hours` | number |

### 5.2 详细配置 `convert_scripts/configs/<id>.yaml`（声明值，onboarding 时确定）

| 字段 | 类型 |
|---|---|
| `source_url` | string |
| `license` | enum: `MIT,Apache-2.0,BSD-3-Clause,CC-BY-4.0,CC-BY-NC-4.0,CC0-1.0,GPL-3.0,Proprietary,Unknown` |
| `raw_format` | enum: `RLDS,HDF5,LeRobot,ROS_bag,MCAP,TFRecord,Custom` |
| `collection_method` | enum: `teleop,autonomous_policy,umi,egocentric_human,simulation,human_to_robot_synthesis` |
| `embodiment_class` | enum: `single_arm,dual_arm,humanoid,mobile_manipulator,human_hand` |
| `robot_platform` | enum（开放列表，见 5.3） |
| `num_arms` | int: `0,1,2` |
| `dof_per_arm` | int，可空 |
| `gripper_type` | enum: `parallel_jaw,dexterous_hand,suction,none,unknown` |
| `has_mobile_base` | bool |
| `action_space` | enum: `joint_position,joint_velocity,joint_torque,eef_pose,mixed,unknown` |
| `action_frame` | enum: `delta,absolute,both,unknown` |
| `rotation_representation` | enum: `euler_xyz,quaternion,rotation_6d,axis_angle,rotation_matrix,none,unknown` |
| `state_dim` / `action_dim` | int，可空 |
| `fps` | number，可空（配合 `fps_variable`） |
| `fps_variable` | bool |
| `num_camera_views` | int |
| `camera_views` | list[enum]: `third_person,head,left_wrist,right_wrist,top,front,side,other` |
| `has_camera_calibration` | bool |
| `depth_coverage` | enum: `none,partial,full` |
| `has_language_instruction` | bool |
| `num_task_types` | int，可空 |
| `urdf_available` | bool |
| `urdf_source` | enum: `dataset_repo,manufacturer_official,community_repo,not_found` |
| `expected_size_gb` / `expected_num_episodes` | number/int |
| `review_status` | enum: `pending_human_review,confirmed`（见第 8 节） |

派生字段（脚本运行时计算，不落盘）：
`fk_check_feasible = urdf_available AND action_space in [joint_position, eef_pose]`

### 5.3 `robot_platform` 初始枚举

`franka_panda, ur5, ur5e, agilex_aloha, agilex_cobot_magic, xarm7,
kinova_gen3, sawyer, widowx, viperx, agibot_g1, tien_kung, arx5,
unitree_g1, other`

后续接入新机器人型号时追加，不允许脚本静默地把未知型号映射到已有枚举值。

## 6. convert_scripts 设计

职责：把 `public_datasets_raw/<id>/raw/` 里的原始格式数据，转换为标准
LeRobot v2.1 schema，落地到 `public_datasets_raw/<id>/lerobot_v2_1_staging/`。
不做清洗、不做跨本体维度统一——只做格式转换，保证输出能被官方 `lerobot`
工具链直接加载。

运行前置条件：`integrity_status == verified`。

顺手处理：转换过程中若在数据集官方 repo / 发布物中找到 URDF 文件，落地到
`urdf_assets/<robot_platform>/`，并将 `urdf_source` 标记为 `dataset_repo`；
找不到则标记 `not_found`，不阻断转换流程。

## 7. process_scripts 设计

`run_pipeline.py` 读取 `configs/<id>.yaml` 中的阈值参数，按顺序执行，每步
可能因注册表字段被 gate 掉（跳过原因写入日志，不静默）：

| 阶段 | 对应论文 | 关键参数 | Gate条件 |
|---|---|---|---|
| Stage1 突变检测 | Sec2.4 S1 | `residual_threshold, accel_threshold, jerk_threshold` | 始终执行 |
| Stage2 状态-动作趋势对齐 | Sec2.4 S2 | `da_threshold`（0.6-0.7） | 需 state+action 均存在 |
| Stage3 极值过滤 | Sec2.4 S3 | `quantile_low(q1), quantile_high(q99)` | 始终执行（gripper维度豁免） |
| Stage4 FK一致性校验 | Sec2.4 S4 | `tcp_offset_correction` | `fk_check_feasible=true` |
| Stage5 坐标系/朝向对齐 | Sec2.4 S5 | `world_frame_convention` | 始终执行 |
| Check1 指令一致性 | Sec2.4 C1 | 本地VLM服务地址、`subtask_segmentation` | `has_language_instruction=true` |
| Check2 视频-状态一致性 | Sec2.4 C2 | 本地SAM3服务地址、`iou_threshold` | `urdf_available=true` |
| Check3 视频质量过滤 | Sec2.4 C3 | 黑/糊/静止帧阈值 | 始终执行 |
| 统一表示层 | Sec3.2/3.3 | 80维slot映射表（按embodiment_class+gripper_type+num_arms定位子空间）+ `camera_frame_delta_pose_enabled` | camera-frame delta pose 需 `has_camera_calibration=true` |

VLM（指令一致性）与 SAM3（视频-状态一致性）均**本地部署**，`check1_*`/
`check2_*` 通过本地 HTTP/SDK 接口调用；具体模型权重与服务化方案留给实施
计划阶段确定，不在本设计文档中固化。

流水线运行结束后，`run_pipeline.py` 汇总过滤统计，自动生成/覆盖对应数据集
的 `README.md`（见第 9 节）。

## 8. 数据集 Onboarding Agent

用于自动化填充 `convert_scripts/configs/<id>.yaml` 中的调研型字段，替代
人工逐个翻阅文档。

**可信来源**：`datasets_registry.yaml` 迁移自现有 xlsx 的官网/官方下载链接，
以及按数据集名检索到的对应论文。

**流程**：
1. 每个数据集一次独立 agent 调研任务，读取官网 + 论文，抽取第 5.2 节定义
   的全部字段。
2. 字段值必须落在预定义 enum 里；遇到枚举表中没有的新值（如未知机器人
   型号），不得硬塞进已有枚举，而是单独标注"建议新增枚举值"，交由人工
   审核决定是否扩展枚举表。
3. 每个字段的调研结果附带来源引用（`_source`: URL / 论文章节），便于人工
   复核时直接定位出处。
4. 输出文件顶层标记 `review_status: pending_human_review`；人工确认无误后
   改为 `confirmed`。
5. **`convert_scripts` 运行前检查 `review_status == confirmed`**，否则拒绝
   执行——防止调研错误直接污染生产流水线。

**执行方式**：33 个候选数据集互相独立、无状态依赖，可一次性并发调研；也
可作为未来新增数据集时的可重复调用流程。首次批量执行的具体编排（人工触发
一次性运行 vs 固化为可重复脚本）留待落地阶段决定。

## 9. `lerobot_v2_1/<dataset_id>/README.md` 规范

**原则：README.md 是自动生成的渲染视图，不是手写文档**，内容来自
`convert_scripts/configs/<id>.yaml`（描述性字段）+ process_scripts 运行后
的统计结果，由 `run_pipeline.py` 在流水线跑完后自动生成/覆盖，避免文档与
实际配置漂移。

内容结构：
1. **基本信息**：名称、来源链接、License、引用 BibTeX
2. **本体信息**：embodiment_class、robot_platform、num_arms、gripper_type、DOF
3. **规模**：清洗后的最终 episode 数/帧数/时长（与 `datasets_registry.yaml`
   中"下载时"的规模对比，差异即为清洗流水线过滤掉的数据比例）
4. **表示层**：state/action 维度、坐标系约定、是否启用 camera-frame delta pose
5. **处理记录**（从运行日志汇总）：跑过哪些 stage/check、各自过滤比例、
   被跳过的 stage 及原因
6. **已知局限**：人工补充的自由文本定性说明

## 10. 范围与非目标（第一阶段）

**包含**：
- 已有公开机器人数据集的下载完整性校验、格式转换（→ 标准 LeRobot v2.1）、
  五阶段数值清洗、三项跨模态质检（VLM+SAM3 本地部署）、跨本体维度统一层
  （80维canonical向量 + 有标定数据时的camera-frame delta pose）
- 数据集注册体系与 Onboarding Agent 自动化调研

**不包含（后续阶段再评估）**：
- Human-to-Robot 数据合成流水线（论文 Sec 2.3：人手视频 retarget 到 15 种
  机器人形态）——独立的重 pipeline，依赖 SAM3/ProPainter/MuJoCo IK/深度补全，
  本阶段不做
- VLM/SAM3 具体模型选型与服务化部署细节——留给实施计划
- URDF 缺失时的人工补全/第三方 URDF 库检索——本阶段仅做"数据集自带资源
  中查找，找不到就跳过对应校验"，不做主动补全

## 11. 后续步骤

本设计文档确认后，进入实施计划阶段（writing-plans），细化：
- `datasets_registry.yaml` 初始化脚本（从现有 xlsx 迁移 + Onboarding Agent
  首批调研）
- 各 `raw_format` 的完整性校验器实现
- 五阶段清洗 + 三质检 + 统一表示层的具体代码实现与单测
- 本地 VLM / SAM3 服务化方案选型
