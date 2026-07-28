# embodied_datasets

VLA（视觉-语言-动作）机器人操作数据集的统一注册、清洗、对齐流水线。所有数据集
先转换为 LeRobot v3.0 格式，再进行五阶段数值清洗、三项跨模态质检和
跨本体维度统一（具体实现见"process_scripts 处理流程"一节）。

## 目录结构

```
embodied_datasets/
├── datasets_registry.yaml          # 66个数据集的总览表（实测值，随流水线推进更新）
├── scripts/
│   ├── convert_scripts/
│   │   ├── configs/<dataset_id>.yaml   # 每个数据集的调研配置（声明值）
│   │   └── common/                     # 复用的 schema/io/onboarding 工具
│   ├── verify_scripts/                 # 下载完整性校验
│   ├── process_scripts/                # 清洗对齐流水线（9个stage/check模块，见本文档
│   │                                    # "process_scripts 处理流程"/"跨本体统一表示层"两节）
│   └── shared/                         # convert_scripts/process_scripts 跨包复用（Episode/FkChain/
│                                        # lerobot读写封装）
└── data_root/                      # 全部重数据，--data-root 可整体指向仓库外任意路径
    ├── public_datasets_raw/<dataset_id>/       # 原始下载数据
    ├── public_datasets_staging/<dataset_id>/   # convert_scripts 产出，process_scripts 输入
    ├── public_datasets/lerobot_v3_0/<dataset_id>/  # process_scripts 最终产出
    └── urdf_assets/<robot_platform>/  # 按机器人型号共享的 URDF
```

## 数据根目录

`datasets_registry.yaml`、`convert_scripts/configs/*.yaml` 和所有脚本代码始终
留在仓库内，不受 `--data-root` 影响。只有重数据目录（`data_root/public_datasets_raw/`、
`data_root/public_datasets_staging/`、`data_root/public_datasets/lerobot_v3_0/`、
`data_root/urdf_assets/`）可以指向仓库外任意路径，读写这些目录的脚本都接受一个
`--data-root` 参数：

```bash
python3 run_pipeline.py --data-root /mnt/big_disk/vla_data --dataset-id droid
```

不传 `--data-root` 时默认使用仓库内的 `embodied_datasets/data_root/`。路径解析
逻辑见 `scripts/convert_scripts/common/paths.py`。

## process_scripts 处理流程

`run_pipeline.py` 按顺序对每个 episode 依次跑 Stage1-5，再跑 Check1-3（第三节
"跨本体统一表示层"另有独立说明，不在下表内）：

| 模块 | 实现 |
|---|---|
| `stage1_sudden_change` | Savitzky-Golay 平滑（`scipy.signal.savgol_filter`）后计算 residual/加速度/jerk，任一超过阈值的帧标记为异常并线性插值修复；标记帧占比超过 `episode_reject_threshold` 则整条 episode 拒绝 |
| `stage2_trend_alignment` | 对 state/action 的每个公共维度做互相关（`scipy.signal.correlate`）估计帧滞后（lag）和方向一致性；lag 绝对值超过 `max_lag_frames` 或方向一致性低于 `da_threshold` 则跳过该 episode，否则按估计的 lag 对齐并裁剪首尾帧 |
| `stage3_extreme_value` | 两遍处理：先对数据集全部 episode 的每一维算 `quantile_low`/`quantile_high` 分位数界（`gripper_dims_state`/`gripper_dims_action` 声明的夹爪维度豁免），再逐 episode 丢弃超界的帧 |
| `stage4_fk_consistency` | 仅当 `urdf_available` 且 `action_space ∈ {joint_position, eef_pose}` 时执行：用 `ikpy`（`shared/fk_backend.FkChain`）对关节角做正向运动学，与数据中报告的末端位置比对；系统性中位数偏移超过 `tcp_offset_tolerance` 时整 episode 做偏移修正，偏移方差过大（非系统性）时只标记待人工复核、不改数据 |
| `stage5_orientation_alignment` | 用配置的 4x4 base-to-world 变换矩阵对每帧末端位置和四元数朝向做坐标变换，统一各数据集的世界坐标系约定；未配置变换矩阵则跳过 |
| `check1_instruction_consistency` | 语言指令一致性检查，仅接入 `NullClient` 占位——真实VLM服务未实现，按 `NullClient` 给出的 `skip_reason` 直接跳过 |
| `check2_video_state_consistency` | FK投影位置与SAM3分割结果的IoU一致性检查，仅接入 `NullClient` 占位——真实SAM3服务未实现，直接跳过 |
| `check3_video_quality` | 三项质检中唯一已实现的一项：用 OpenCV 检测黑屏（平均亮度低于 `black_threshold`）、模糊（Laplacian方差低于 `blur_threshold`）、连续静止帧（帧间差低于 `still_threshold` 且持续帧数达到 `still_min_consecutive_frames`），命中的帧直接丢弃 |

## 跨本体统一表示层 —— 数据公共规范

本节定义 `process_scripts` 流水线产出的**最终 LeRobot v3.0 数据集**中，机器人本体
（robot-collected embodiment）`observation.state` 的格式。这是下游训练/评测脚本
读取数据时依赖的公共契约；如本节与 `unify_representation.py` 的实际代码不一致，
以代码为准，请提 issue。

对应实现：`process_scripts/unify_representation.py`（计算逻辑）+
`process_scripts/run_pipeline.py`（把计算结果写进最终数据集）+
`shared/lerobot_io.py`（`write_lerobot_episodes` 的落盘细节）。

### 1. 适用范围

Gate 条件（`unify_representation.py` 里的 `ROBOT_EMBODIMENT_CLASSES`）：只对以下
`embodiment_class` 生效——

```
single_arm, dual_arm, half_humanoid, humanoid, mobile_manipulator, quadruped
```

`human_hand`（第一/第三人称人手视频，如 H2O、OAKink2、TACO）和 `human_full_body`
（如 EgoAllo）不经过这一层，`observation.state` 保留原始 per-dataset 维度不变，
完整参数留在 `data_root/public_datasets_staging/` 原始数据中。

### 2. 128 维 canonical 向量布局

固定总维度 128，按下表切片。`JOINT_SLOT`（7）和 `GRIPPER_SLOT`（21）取自注册表内
`dof_per_arm`/`dof_per_hand` 的实测最大值（见第4节）；`EEF_SLOT`（7）是固定的位姿
表示惯例（3维位置 + 4维四元数），与具体数据集无关：

| 子区间 | 维度 | 内容 | 常量名 |
|---|---|---|---|
| `[0:7]`（arm1） | 7 | 关节位置 | `JOINT_SLOT` |
| `[7:14]` | 7 | 末端位姿：3维位置 + 4维四元数 | `EEF_SLOT` |
| `[14:35]` | 21 | 夹爪/灵巧手槎位（见第3节） | `GRIPPER_SLOT` |
| `[35:70]`（仅双臂数据集） | 35 | arm2，结构与 `[0:35]` 相同 | `ARM_BLOCK_DIM` |
| `[70:128]` | 58 | 预留，当前恒为0，给未来全身运控/其它传感器模态留空间 | `RESERVE_SLOT` |

四元数分量顺序（xyzw / wxyz）由 `convert_scripts` onboarding 时约定；
`unify_representation.py` 按该约定顺序原样写入 `[10:14]`，不做校验或转换。

单臂数据集的 `[35:70]` 恒为0，对应 `mask` 恒为 `False`（表示"没有第二臂"；训练时
应按 mask 忽略该区间，不应视为第二臂的零速度数据）。

`ARM_BLOCK_DIM = JOINT_SLOT + EEF_SLOT + GRIPPER_SLOT = 35`。移动底盘速度
（vx/vy/yaw）当前不写入 `[70:128]` 或任何其它槎位——`has_mobile_base=true` 的
数据集，该部分数值在打包时被排除以免污染臂部槎位，但直接丢弃，不被 canonical
向量捕获；全身运控/移动底盘的槎位设计留给后续单独决定。

### 3. 夹爪槎位 `[14:35]` 的分支规则

槎位宽度固定21维，不同 `gripper_type` 使用其中一部分：

- `parallel_jaw` / `three_jaw` / `cage_pinch` / `suction`（简单夹爪）→ 只用
  **slot0**（1维开合宽度/吸附状态），剩余20维置0、mask=False
- `dexterous_hand`（灵巧手）→ 使用实际列宽（最多21维，不足补0，超过截断），
  每一维单独设 mask
- 其他/未知 gripper_type → 全部置0，mask=False

手腕姿态记录在 `[7:14]` 的末端位姿中，不占用该槎位；这21维仅为手部自身的执行器
自由度（手指、虎口等）。

### 4. `GRIPPER_SLOT` 宽度

`GRIPPER_SLOT=21`，取自注册表内机器人采集类灵巧手数据集 `dof_per_hand` 实测最大值
（当前为16，`arcap`），预留余量。若未来出现实测超过21维的灵巧手数据集，需重新评估
该常量，并同步更新本节、`unify_representation.py` 的 `GRIPPER_SLOT`，以及
`run_pipeline.py` README 模板中的相应措辞。

人手视频/MANO 数据集（`dexcap`/`h2o`/`oakink2`/`taco`/`vitra`/`hoi4d`/`ph2d` 等，
`dof_per_hand` 常见15-48维）不受此宽度约束——它们由第1节的 gate 条件排除，不经过
本层。

### 5. `episode.action` 范围

本层只处理 `observation.state`，不处理 `action`。`action` 在最终输出中保持原始
per-dataset 维度不变。

### 6. mask 语义

`unify_representation.apply()` 除计算128维向量外，还计算一个128维 bool mask（同一
数据集内所有帧、所有episode共享同一份，仅取决于 `dof_per_arm`/`num_arms`/
`gripper_type`/`has_mobile_base` 等数据集级配置）。

该 mask 作为独立的 lerobot feature 写入最终数据集：

```
observation.state_canonical_mask   # bool, shape (128,)，每帧写入，整数据集内容相同
```

`mask[i]=False` 表示第 i 维是该本体不具备对应自由度的零填充，而非测量值为0。
训练时应使用该 mask 过滤loss/attention。

### 7. 已知局限

- `dof_per_arm` 配置错误但未超出列宽时无法检测：`_pack_arm` 依赖
  `config.dof_per_arm` 划分关节/末端位姿边界，配置错误但总列宽仍够用时会将末端
  位姿/夹爪数据错位写入关节槎位，`mask` 仍为 `True`，无报错信号。正确性依赖上游
  `DatasetConfig.dof_per_arm` 的准确性。
- 只统一 `state`，不统一 `action`（见第5节）。
- 不覆盖 MANO/人手视频（见第1节），完整参数保留在
  `data_root/public_datasets_staging/` 原始数据中。
- 超过21维的灵巧手会被截断（见第4节）。
- 移动底盘速度（vx/vy/yaw）当前不被任何槎位捕获（见第2节）：`has_mobile_base=true`
  的数据集，该部分数值在打包时被排除以免污染臂部槎位，但直接丢弃。
- 经 `convert_scripts` 转换的数据集，`data_root/public_datasets_staging/` 中不含
  视频：`shared/lerobot_io.py` 的 `write_lerobot_episodes` 目前硬编码
  `use_videos=False`，只写 `observation.state`/`action`/`task`。原始视频/图像观测
  在写入 staging 时被丢弃，`process_scripts` 的 `check2_video_state_consistency`/
  `check3_video_quality` 对经 `convert_scripts` 处理的数据集无可操作对象。修复需要
  为 `write_lerobot_episodes` 添加视频写入支持。

### 8. 验证

环境搭建见根目录 `README.md`。

```bash
cd embodied_datasets/scripts/process_scripts
source ../../../.venv/bin/activate
pytest tests/test_unify_representation.py -v   # 单臂/双臂/移动底盘/gate/21维灵巧手边界
pytest tests/test_run_pipeline.py -v           # canonical_state 替换 episode.state 并写入最终数据集
```

`test_dexterous_hand_with_21_dof_packs_without_truncation`：构造21维数值，断言全部落入 `[14:35]` 且 `mask=True`。

已用 `lerobot/pusht`（HuggingFace 公开的v3.0格式数据集，206 episode/25650帧）验证过完整流程：下载、`load_lerobot_episodes` 读取、9个stage/check模块、最终数据集写出、`observation.state_canonical_mask` 数值位置正确——该次验证在 `CANONICAL_DIM=80` 时进行；改成128维后的覆盖仅来自上面两个 pytest 命令（含合成数据的128维断言），未重新跑真实数据集下载验证。
