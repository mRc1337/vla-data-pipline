# 注册表 Schema 第二轮扩展 —— 设计文档

- 日期：2026-07-13
- 状态：草案，待用户审阅
- 前置文档：
  - [2026-07-08 VLA 数据预处理与对齐流水线设计文档](2026-07-08-vla-data-pipeline-design.md)
  - [2026-07-10 注册表 Schema 补全与数据根目录设计文档](2026-07-10-registry-schema-refinement-design.md)

## 1. 背景

第一轮 Schema 补全（2026-07-10）完成后，对全部 82 个数据集（59 个存量 + 25 个新发现，其中
2 个确认为重复已删除，故此时为 84；本文档写作时重复已清理，现为 82 个）做了一轮独立
高强度复核 + 新数据集调研。过程中，Onboarding Agent 们独立地在多个数据集上反复撞见了
同一批"枚举里没有这个值"的缺口——尤其是 `robot_platform`（具体机器人型号）、
`collection_method`（采集方式的细分类别）、`license`（非标准/定制许可证）。

与第一轮不同，本轮**不需要新增字段或新枚举类型**——第一轮引入的
`is_multi_embodiment`、`additional_modalities`、`secondary_collection_methods` 等字段
已经把"需要表达的维度"覆盖到位，本轮暴露的纯粹是"这些维度里合法取值不够多"。

## 2. 枚举值新增（37 个，全部来自已有数据集调研报告里的具体依据）

### 2.1 `RobotPlatform`（+18）

| 新增值 | 依据数据集 |
|---|---|
| `virtual_agent` | alfred, teach（无实体硬件的虚拟具身智能体） |
| `abb_yumi` | autobag, handloom |
| `r1pro` | behavior_1k |
| `galaxea_r1` | behavior_robot_suite（区别于已有的 `galaxea_r1_lite`） |
| `hello_robot_stretch` | dobb_e, ovmm |
| `franka_fr3` | gensim2, robogene（区别于已有的 `franka_panda`，Franka Research 3 型号） |
| `unitree_aliengo` | grutopia |
| `unitree_h1_2` | humanoid_x（27自由度变体，区别于已有的 `unitree_h1`） |
| `xarm6` | language_table（区别于已有的 `xarm7`） |
| `kuka_iiwa` | mimicgen, robonet |
| `boston_dynamics_spot` | partnr |
| `galbot_g1` | robocoin |
| `unitree_a1` | robogen（locomotion 模块） |
| `anymal` | robogen（locomotion 模块） |
| `atlas` | robogen（locomotion 模块，Boston Dynamics Atlas） |
| `everyday_robots_arm` | robovqa, rt_1 |
| `1x_eve`（Python 成员名 `X1_EVE`） | 1x_world_model_dataset |
| `agibot_g2` | agibot_world（2026年G2平台，区别于已有的 `agibot_g1`） |

### 2.2 `CollectionMethod`（+5）

| 新增值 | 依据数据集 |
|---|---|
| `ego_exo_human` | assembly101, ego_exo4d, h2o, oakink2（×4，本轮共识最强的一条：同时录制egocentric+exocentric的人类活动采集，跟现有的纯`egocentric_human`不是一回事） |
| `kinesthetic` | roboset（物理引导示教，跟远程`teleop`是不同的采集方式） |
| `scripted` | open_x_embodiment, autobag, pokeflex（×3，预编程/随机化动作原语，既非遥操也非学习到的策略） |
| `synthetic_multimodal_augmentation` | roboomni（对已有机器人轨迹叠加合成音频/对话标注） |
| `mocap_multiview_human` | taco（工作室多视角RGB+光学动捕，跟纯egocentric视频采集不同） |

### 2.3 `LicenseEnum`（+3）

| 新增值 | 依据数据集 |
|---|---|
| `CC-BY-SA-4.0` | oakink2（标准CC协议，遗漏） |
| `CDLA-Sharing-1.0` | teach（标准数据共享协议，遗漏） |
| `custom_research_eula` | ego4d, ego_exo4d, epic_kitchens_100, h2o, nvidia_physicalai_robotics_manipulation_objects, pokeflex（×6，见下方"设计取舍"） |

### 2.4 `EmbodimentClass`（+2）

| 新增值 | 依据数据集 |
|---|---|
| `human_full_body` | ego4d, egoallo（全身姿态估计，跟只覆盖手部的 `human_hand` 不是一回事） |
| `half_humanoid` | robocoin（双臂+躯干无腿部，论文称为该数据集49%的主流架构） |

### 2.5 `GripperType`（+2）

| 新增值 | 依据数据集 |
|---|---|
| `mixed` | open_x_embodiment, robocoin, robomind, roboomni（×4） |
| `cage_pinch` | handloom（缆线操作专用非对称二指夹爪机制） |

### 2.6 `RotationRepresentation`（+2）

| 新增值 | 依据数据集 |
|---|---|
| `mixed` | dexmimicgen, nvidia_physicalai_robotics_manipulation_singlearm, roboverse, xr_1_dataset（×4） |
| `single_axis_angle` | robocook, robonet（×2，动作空间只有单一旋转角，不是完整3D朝向） |

### 2.7 `SensorModality`（+3）

| 新增值 | 依据数据集 |
|---|---|
| `imu` | ego_exo4d, epic_kitchens_100（×2） |
| `semantic_segmentation` | nvidia_physicalai_robotics_manipulation_kitchen, nvidia_physicalai_robotics_manipulation_objects（×2） |
| `point_cloud_3d_scan` | ego_exo4d, hot3d（×2，持久3D场景重建，跟逐帧`depth_coverage`是不同维度） |

### 2.8 `ActionSpace`（+1）

| 新增值 | 依据数据集 |
|---|---|
| `discrete_symbolic` | alfred, teach（×2，AI2-THOR式离散动作词表，不是连续关节/末端控制） |

### 2.9 `CameraView`（+1）

| 新增值 | 依据数据集 |
|---|---|
| `gripper_jaw` | partnr（Spot机器人爪部相机，跟臂部wrist相机是不同安装位置） |

## 3. 设计取舍（明确不采纳的建议，避免枚举膨胀）

- **不单独区分 `vr_teleop`/具体teleop接口**：VR手柄/太空鼠/关节示教等都是"远程操作"的接口
  细节，不是采集方式本身的范畴差异，继续用现有 `teleop` 表达，接口细节写进
  `field_sources` 自由文本即可。（`kinesthetic` 例外收录——它不是"远程"操作，是物理引导，
  属于真正不同的采集范畴。）
- **不加 `action_frame` 的"不适用"哨兵值**：字段本身是 Optional，留空已经表达"不适用"，
  不需要专门造一个值。
- **不给每个机构的定制许可证单开一个枚举值**：ego4d/epic-kitchens/h2o/NVIDIA/ETH各自的
  门禁EULA都是一次性法律文本，不是可被未来新数据集复用的标准协议——跟 `CC-BY-SA-4.0`/
  `CDLA-Sharing-1.0` 这类真正的标准协议不同。合并成一个通用的 `custom_research_eula`，
  具体条款继续记在 `field_sources` 里，比给每家机构开一个枚举值更可持续。

## 4. 现有配置的回填范围

同第一轮模式：表驱动脚本，只应用数据集自己的调研报告里已经给出依据的具体值，不做新的
调研。多本体/多平台数据集（robocoin、robonet、roboverse、robogen、mimicgen、
gensim2 等）的相关字段本来就正确地留空（`is_multi_embodiment: true` + 字段留空），
本轮只是让枚举本身存在以备未来更细粒度的调研使用，**不**强行回填单一值。

| 数据集 | 字段 | 修正为 |
|---|---|---|
| 1x_world_model_dataset | `robot_platform` | `1x_eve` |
| agibot_world | `robot_platform` | `agibot_g2` |
| alfred | `robot_platform` | `virtual_agent` |
| alfred | `action_space` | `discrete_symbolic` |
| autobag | `robot_platform` | `abb_yumi` |
| autobag | `collection_method` | `scripted` |
| behavior_1k | `robot_platform` | `r1pro` |
| behavior_robot_suite | `robot_platform` | `galaxea_r1` |
| dobb_e | `robot_platform` | `hello_robot_stretch` |
| ego4d | `embodiment_class` | `human_full_body` |
| ego4d | `license` | `custom_research_eula` |
| ego_exo4d | `license` | `custom_research_eula` |
| ego_exo4d | `additional_modalities` | 追加 `imu`, `point_cloud_3d_scan` |
| egoallo | `embodiment_class` | `human_full_body` |
| epic_kitchens_100 | `license` | `custom_research_eula` |
| epic_kitchens_100 | `additional_modalities` | 追加 `imu` |
| h2o | `license` | `custom_research_eula` |
| h2o | `action_space` | `mixed` |
| handloom | `robot_platform` | `abb_yumi` |
| handloom | `gripper_type` | `cage_pinch` |
| hot3d | `additional_modalities` | 追加 `point_cloud_3d_scan` |
| humanoid_x | `robot_platform` | `unitree_h1` → `unitree_h1_2` |
| nvidia_physicalai_robotics_manipulation_kitchen | `additional_modalities` | `[semantic_segmentation]` |
| nvidia_physicalai_robotics_manipulation_objects | `additional_modalities` | `[semantic_segmentation]` |
| nvidia_physicalai_robotics_manipulation_objects | `license` | `custom_research_eula` |
| nvidia_physicalai_robotics_manipulation_singlearm | `rotation_representation` | `mixed` |
| oakink2 | `license` | `CC-BY-SA-4.0` |
| open_x_embodiment | `gripper_type` | `mixed` |
| ovmm | `robot_platform` | `other` → `hello_robot_stretch` |
| partnr | `robot_platform` | `boston_dynamics_spot` |
| partnr | `camera_views` | 区分 `wrist`（臂部）与 `gripper_jaw`（爪部） |
| pokeflex | `license` | `custom_research_eula` |
| robocoin | `gripper_type` | `mixed` |
| robocook | `rotation_representation` | `single_axis_angle` |
| robogene | `robot_platform` | `franka_fr3` |
| robomind | `gripper_type` | `mixed` |
| roboomni | `gripper_type` | `mixed` |
| roboomni | `secondary_collection_methods` | 追加 `synthetic_multimodal_augmentation` |
| roboverse | `rotation_representation` | `mixed` |
| rt_1 | `robot_platform` | `other` → `everyday_robots_arm` |
| teach | `robot_platform` | `other` → `virtual_agent` |
| teach | `license` | `CDLA-Sharing-1.0` |
| teach | `action_space` | `discrete_symbolic` |
| vitra | `gripper_type` | `dexterous_hand` → `none`（纠正：统一裸手数据集既有惯例，与h2o/hoi4d/ego_exo4d/taco/oakink2一致） |
| dexmimicgen | `rotation_representation` | `mixed` |
| xr_1_dataset | `rotation_representation` | `mixed` |

不回填（仅加枚举值供未来使用）：grutopia/mimicgen/gensim2/robogen/robonet/roboverse/
robocoin(embodiment_class)/robovqa 的 `robot_platform`/`embodiment_class` 保持留空
（多本体，无主导值）；rh20t 的 `license` 保持留空（拆分许可证，第一轮已确认这是枚举
单值字段的固有局限，不强行套用新值）。

## 5. 范围与非目标

**包含**：`schema.py` 的 37 个枚举值新增（不加新字段、不加新枚举类型）、表驱动回填
脚本（复用第一轮 `common/backfill_schema_refinement.py` 的模式，新建
`common/backfill_schema_round2.py`）、`onboarding_agent.py` 的 `_ENUM_FIELDS` 自动
拾取新枚举值（因为是同一批枚举类的成员扩展，不是新枚举类，`_ENUM_FIELDS` 字典本身
不需要改动——`_format_enum_choices` 遍历枚举类成员时会自动包含新增值）。

**不包含**：新字段、新枚举类型、`common/paths.py` 相关改动（本轮跟数据根目录无关）、
对多本体数据集的强行单值回填。

## 6. Global Constraints（写入实施计划时原样带入）

- 所有 Python 代码遵循 Python 3.9 语法
- 新枚举成员命名 `UPPER_SNAKE_CASE`，字符串值风格与 `schema.py` 现有同枚举内其他成员
  保持一致（如 `LicenseEnum` 用连字符大写风格 `CC-BY-SA-4.0`，其余多用小写下划线风格）
- 回填脚本必须是纯表驱动，不引入新的调研/推断逻辑——所有回填值都来自第4节列出的、
  已经在调研报告里给出依据的具体值
- `_ENUM_FIELDS` 字典本身不需要修改（本轮只增加枚举成员，不新增枚举类型或字段）；
  实施时验证这一点（写一个测试确认新枚举值出现在 `build_onboarding_prompt()` 的输出里）

## 7. 后续步骤

本设计确认后进入实施计划阶段，预计任务拆分：
1. `schema.py` 的 37 个枚举值新增 + 单测
2. 验证 `onboarding_agent.py` 无需改动即可拾取新值 + 单测
3. 回填脚本（第4节表驱动）+ 单测 + 真实运行
