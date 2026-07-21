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

| id | name | priority | download_status | convert_status | process_status | review_status | collection_method | embodiment_class |
|---|---|---|---|---|---|---|---|---|
| 1x_world_model_dataset | 1X World Model Dataset | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | humanoid |
| alfred | ALFRED | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | mobile_manipulator |
| aloha_unleashed | ALOHA Unleashed | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | dual_arm |
| arcap | ARCap | P2 | not_downloaded | not_converted | not_processed | pending_human_review | ar_haptic_guided_synthesis | single_arm |
| agibot_digital_world | AgiBot Digital World | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | dual_arm |
| agibot_world | AgiBot-World | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | dual_arm |
| airexo_2 | AirExo-2 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | dual_arm |
| assembly101 | Assembly101 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| autobag | AutoBag | P2 | not_downloaded | not_converted | not_processed | pending_human_review | scripted | dual_arm |
| behavior_robot_suite | BEHAVIOR Robot Suite | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| behavior_1k | BEHAVIOR-1K | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| brmdata | BRMData | P2 | not_downloaded | not_converted | not_processed | confirmed | teleop | mobile_manipulator |
| bigym | BiGym | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | humanoid |
| bridgedata_v2 | BridgeData V2 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| calvin | CALVIN | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| droid | DROID | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| dexcap | DexCap | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | human_hand |
| dexmimicgen | DexMimicGen | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | dual_arm |
| dobb_e | Dobb-E | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | mobile_manipulator |
| epic_kitchens_100 | EPIC-KITCHENS-100 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| ego_exo4d | Ego-Exo4D | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| ego4d | Ego4d | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_full_body |
| egoallo | EgoAllo | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_full_body |
| egodex | EgoDex | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| egoexolearn | EgoExoLearn | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| fastumi | FastUMI | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | single_arm |
| functional_manipulation_benchmark_fmb | Functional Manipulation Benchmark (FMB) | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| furniturebench | FurnitureBench | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| gr00t_teleop_sim | GR00T Teleop Sim | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | humanoid |
| grutopia | GRUtopia | P2 | not_downloaded | not_converted | not_processed | pending_human_review | scene_asset_curation |  |
| galaxea_open_world_dataset | Galaxea Open-World Dataset | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| gensim2 | GenSim2 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
| h2o | H2O | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| handloom | HANDLOOM | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | dual_arm |
| hd_epic | HD-EPIC | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| hoi4d | HOI4D | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| hot3d | HOT3D | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| humanoid_x | Humanoid-X | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | humanoid |
| humanoidbench | HumanoidBench | P2 | not_downloaded | not_converted | not_processed | confirmed | simulation | humanoid |
| libero | LIBERO | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| libero_plus | LIBERO-plus | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
| language_table | Language Table | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| mv_umi | MV-UMI | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | single_arm |
| maniskill3 | ManiSkill3 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
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
| ovmm | OVMM | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | mobile_manipulator |
| omniumi | OmniUMI | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | single_arm |
| open_x_embodiment | Open X-Embodiment | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop |  |
| partnr | PARTNR | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | mobile_manipulator |
| ph2d | PH2D | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| pokeflex | PokeFlex | P2 | not_downloaded | not_converted | not_processed | pending_human_review |  | single_arm |
| rh20t | RH20T | P2 | not_downloaded | not_converted | not_processed | confirmed | teleop | single_arm |
| rt_1 | RT-1 | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| robocoin | RoboCOIN | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop |  |
| robocasa | RoboCasa | P2 | not_downloaded | not_converted | not_processed | pending_human_review | human_to_robot_synthesis | mobile_manipulator |
| robocook | RoboCook | P2 | not_downloaded | not_converted | not_processed | pending_human_review | autonomous_policy | single_arm |
| robogen | RoboGen | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation |  |
| robogene | RoboGene | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | dual_arm |
| robomind | RoboMind | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop |  |
| robonet | RoboNet | P2 | not_downloaded | not_converted | not_processed | pending_human_review | autonomous_policy |  |
| roboomni | RoboOmni | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | single_arm |
| roboset | RoboSet | P2 | not_downloaded | not_converted | not_processed | pending_human_review |  | single_arm |
| robotwin | RoboTwin | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | dual_arm |
| robovqa | RoboVQA | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human |  |
| roboverse | RoboVerse | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
| taco | TACO | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| teach | TEACh | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop | mobile_manipulator |
| the_colosseum | THE COLOSSEUM | P2 | not_downloaded | not_converted | not_processed | pending_human_review | simulation | single_arm |
| umi_datasets | UMI datasets | P2 | not_downloaded | not_converted | not_processed | pending_human_review | umi | single_arm |
| vitra | VITRA | P2 | not_downloaded | not_converted | not_processed | pending_human_review | egocentric_human | human_hand |
| xr_1_dataset | XR-1 Dataset | P2 | not_downloaded | not_converted | not_processed | pending_human_review | teleop |  |
| yubi | YUBI | P2 | not_downloaded | not_converted | not_processed | confirmed | umi | dual_arm |
| lerobot_ull_folding | lerobot/ull_folding | P2 | not_downloaded | not_converted | not_processed | pending_human_review |  |  |

<!-- AUTO-GENERATED TABLE END -->

上表由 `python3 public_datasets_raw/convert_scripts/common/generate_overview_readme.py`
生成，只更新 marker 之间的内容；手动新增数据集或更新状态后重新运行以刷新。
