# vla_data_pipeline

VLA（视觉-语言-动作）机器人操作数据集的统一处理流水线。

## 项目结构

```
vla_data_pipeline/
├── embodied_datasets/     # 全部实际工作：数据集注册表、清洗流水线（唯一子目录 scripts/process_scripts/）
├── requirements.txt        # 全仓库共用的依赖清单
└── pyproject.toml          # pytest 配置（指向 process_scripts/tests）+ 项目元数据
```

## 环境搭建

```bash
sudo apt update && sudo apt install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 系统依赖

```bash
sudo apt update && sudo apt install ffmpeg
```

## 数据集目录结构

```
embodied_datasets/
└── scripts/
    └── process_scripts/
        ├── common/                                         # schema/io/paths（DatasetConfig/ProcessConfig）
        ├── configs/<dataset_id>.yaml                        # 每个数据集的清洗阈值配置（ProcessConfig）
        ├── registry_configs/<dataset_id>.yaml               # 每个数据集的调研配置（DatasetConfig）
        ├── episode.py / fk_backend.py / lerobot_io.py       # Episode/FkChain/lerobot读写
        └── stage1-5 / check1-3 / unify_representation.py / run_pipeline.py

embodied_datasets/data_root/                                # --data-root 可整体指向仓库外任意路径
├── public_datasets_raw/<dataset_id>/                       # 原始下载数据
├── public_datasets_staging/lerobot_v3_0/<dataset_id>/      # process_scripts 输入（外部转换产出）
├── public_datasets/lerobot_v3_0/<dataset_id>/              # process_scripts 最终产出
└── urdf_assets/<robot_platform>/                           # 按机器人型号共享的 URDF
```

```bash
python3 run_pipeline.py --data-root /mnt/big_disk/vla_data --dataset-id droid
```

## process_scripts 处理流程

`run_pipeline.py` 按顺序对每个 episode 依次跑 Stage1-5，再跑 Check1-3（下面
"跨本体统一表示"另有独立说明，不在下表内）：

| 模块 | 实现 |
|---|---|
| `stage1_sudden_change` | Savitzky-Golay 平滑（`scipy.signal.savgol_filter`）后计算 residual/加速度/jerk，任一超过阈值的帧标记为异常并线性插值修复；标记帧占比超过 `episode_reject_threshold` 则整条 episode 拒绝 |
| `stage2_trend_alignment` | 对 state/action 的每个公共维度做互相关（`scipy.signal.correlate`）估计帧滞后（lag）和方向一致性；lag 绝对值超过 `max_lag_frames` 或方向一致性低于 `da_threshold` 则跳过该 episode，否则按估计的 lag 对齐并裁剪首尾帧 |
| `stage3_extreme_value` | 两遍处理：先对数据集全部 episode 的每一维算 `quantile_low`/`quantile_high` 分位数界（`gripper_dims_state`/`gripper_dims_action` 声明的夹爪维度豁免），再逐 episode 丢弃超界的帧 |
| `stage4_fk_consistency` | 仅当 `urdf_available` 且 `action_space ∈ {joint_position, eef_pose}` 时执行：用 `ikpy`（`shared/fk_backend.FkChain`）对关节角做正向运动学，与数据中报告的末端位置比对；系统性中位数偏移超过 `tcp_offset_tolerance` 时整 episode 做偏移修正，偏移方差过大（非系统性）时只标记待人工复核、不改数据 |
| `stage5_orientation_alignment` | 用配置的 4x4 base-to-world 变换矩阵对每帧末端位置和四元数朝向做坐标变换，统一各数据集的世界坐标系约定；未配置变换矩阵则跳过 |
| `check1_instruction_consistency` | 语言指令一致性检查：通过OpenAI兼容接口调用VLM（默认模型 `qwen2.5-vl-7b-instruct`，API key经 `vlm_api_key_env` 指定的环境变量读取，从不写入配置文件），对语言指令与均匀采样的最多5帧图像做一致性判断，模型强制输出JSON；网络/解析失败时fail-safe为 `consistent=True`（不误伤好数据），`vlm_service_url` 已配置但密钥缺失时直接报错（不静默退化为跳过） |
| `check2_video_state_consistency` | FK投影位置与SAM3分割结果的IoU一致性检查，仅接入 `NullClient` 占位——真实SAM3服务未实现，直接跳过 |
| `check3_video_quality` | 三项质检中唯一已实现的一项：用 OpenCV 检测黑屏（平均亮度低于 `black_threshold`）、模糊（Laplacian方差低于 `blur_threshold`）、连续静止帧（帧间差低于 `still_threshold` 且持续帧数达到 `still_min_consecutive_frames`），命中的帧直接丢弃 |

## 跨本体统一表示

本节定义 `process_scripts` 流水线产出的**最终 LeRobot v3.0 数据集**中，机器人本体
（robot-collected embodiment）`observation.state` 的格式。这是下游训练/评测脚本
读取数据时依赖的公共契约。128 维 canonical 向量固定总维度 128，按下表切片：

| 子区间 | 维度 | 内容 | 常量名 |
|---|---|---|---|
| `[0:7]` | 7 | 关节位置 | `JOINT_SLOT` |
| `[7:14]` | 7 | 末端位姿：3维位置 + 4维四元数 | `EEF_SLOT` |
| `[14:35]` | 21 | 夹爪/灵巧手 | `GRIPPER_SLOT` |
| `[35:70]`（仅双臂数据集） | 35 | ARM2，结构与 `[0:35]` 相同 | `ARM_BLOCK_DIM` |
| `[70:128]` | 58 | 预留，当前恒为0，给未来全身运控/其它传感器模态留空间 | `RESERVE_SLOT` |

`unify_representation.apply()` 除计算128维向量外，还计算一个128维 bool mask。该
mask 作为独立的 lerobot feature 写入最终数据集：

```
observation.state_canonical_mask   # bool, shape (128,)，每帧写入，整数据集内容相同
```

`mask[i]=False` 表示第 i 维是该本体不具备对应自由度的零填充，而非测量值为0。训练时
应使用该 mask 过滤loss/attention。
