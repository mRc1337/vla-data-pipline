# 数据清洗使用指南

本文说明如何对已经转换为 LeRobot v3.0 的数据执行清洗，并将结果接入 Embodied Studio。当前推荐流程是 Qwen-RobotManip Stage 1–3 的不可变 Filter Manifest 模式：原始 Parquet 和视频始终只读，清洗程序只写标签、Episode 过滤结果和逐帧有效性 Mask。

> Stage 4–8 的既有产物可以在平台中查看，但当前仓库尚未提供统一的生产执行入口。`POST /api/pipelines/run` 是 Stage 插件调度骨架，默认处理器只生成占位结果，不应作为正式清洗命令。

## 1. 清洗阶段

| 阶段 | 名称 | 判定粒度 | 结果 |
|---|---|---|---|
| Stage 1 | Sudden Change Detection | 帧或 Episode | 检测 state/action 突变，按配置屏蔽异常帧或拒绝整条 Episode |
| Stage 2 | State-Action Trend Alignment | Episode | 比较 state/action 的时序和方向一致率；delta action 会先积分后比较 |
| Stage 3 | Extreme Value Filtering | 帧 | 使用同本体数据的分位数边界屏蔽极值帧，夹爪维度豁免范围判断 |

三个阶段按 `1 → 2 → 3` 串联：Stage 2 只使用 Stage 1 接受的数据，Stage 3 只使用前两阶段保留的数据。

## 2. 环境与输入要求

从仓库根目录执行：

```bash
cd /home/pai/zxw/vla-data-pipeline
```

安装依赖：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

输入数据集必须是完整的 LeRobot v3.0 目录，并至少包含：

```text
<dataset>/
├── meta/info.json
├── data/
└── videos/                 # 无视频数据集可缺省
```

`meta/info.json` 中必须声明 `codebase_version: "v3.0"`，数据 Parquet 必须包含 `episode_index`、`frame_index`、`observation.state` 和 `action`。

## 3. 配置文件

配置模板位于：

```text
embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json
```

配置分为两层：

- `defaults`：所有数据集的默认值。
- `datasets`：按数据集 ID 的 glob 模式覆盖默认值，例如 `libero_plus*`。

当多个模式同时匹配时，程序按配置文件中的顺序逐项合并。建议把本体、action 表示、维度映射和阈值写在具体数据集块中，不要依赖跨本体的统一默认值。

### 关键配置

| 参数 | 含义 |
|---|---|
| `state_key` / `action_key` | LeRobot 中 state/action 字段名 |
| `embodiment` | Stage 3 联合标定的本体分组名 |
| `action_mode` | `absolute` 或 `delta`；影响 Stage 2 的比较方式 |
| `stage1_exclusion` | `frame` 只屏蔽异常帧；`episode` 拒绝整条 Episode |
| `median_kernels` | Stage 1 级联中值滤波窗口 |
| `savgol_window` / `savgol_polyorder` | Stage 1 Savitzky–Golay 平滑参数 |
| `stage1_mad_scale` | Stage 1 MAD 阈值倍率 |
| `stage1_quantile_floor` | Stage 1 阈值的最低分位数 |
| `stage2_da_threshold` | Stage 2 最低方向一致率 |
| `stage2_max_lag_seconds` | Stage 2 最大因果时延，按数据集 FPS 换算为帧 |
| `stage2_min_active_steps` | 一个维度可评分所需的最少有效运动步数 |
| `stage3_alpha` | Stage 3 在 q1–q99 区间外增加的余量系数 |
| `calibration_samples` | 每个指标用于阈值标定的最大采样帧数 |
| `state_action_map` | Stage 2 的显式 `[[state_dim, action_dim], ...]` 映射 |
| `gripper_indices` | Stage 2 不比较、Stage 3 不做范围过滤的夹爪维度 |
| `angular_indices` | 需要执行角度 unwrap 的维度 |
| `quaternion_groups` | 需要消除四元数正负号跳变的维度组 |

论文没有公布所有工程阈值。正式运行前应根据数据 FPS、表示形式、真实/仿真来源和人工复核结果校准参数。

## 4. 预检

先扫描输入目录并检查 Stage 2 是否存在可信的 state/action 映射：

```bash
.venv/bin/python \
  embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json \
  inventory \
  --input-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0
```

重点检查输出中的：

- `status=ready`
- `state_dim` 和 `action_dim`
- `mapped_joint_dimensions` 大于 0
- `embodiment` 与预期一致

若出现 `stage2_mapping_required`，应在配置中补充 `state_action_map`，不要仅因为 state/action 数组等长就开启位置映射。

## 5. 小样本验证

先用独立输出目录验证 20 个 Episode。不要让小样本结果覆盖正式 Stage 目录：

```bash
.venv/bin/python \
  embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json \
  run \
  --dataset-path /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero_plus \
  --dataset-id libero_plus \
  --output-root /home/pai/zxw/qwen_robotmanip_smoke \
  --work-root /home/pai/zxw/qwen_robotmanip_work_smoke \
  --stages 1,2,3 \
  --max-episodes 20 \
  --overwrite
```

验证以下内容后再执行全量任务：

- 三个 Stage 都生成 `manifest.json` 和 `labels/*.parquet`。
- Stage 1 的突变比例合理，并确认命中维度不是姿态表示跳变或正常夹爪开合。
- Stage 2 没有大量 `unscored`，DA 失败比例符合人工抽检。
- Stage 3 的极值比例合理，夹爪维度没有被范围过滤。
- `source_dataset`、`parent_manifest`、`video_source` 均指向正确路径。

## 6. 全量运行

以 `libero_plus` 为例：

```bash
.venv/bin/python \
  embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json \
  run \
  --dataset-path /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero_plus \
  --dataset-id libero_plus \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/data_curation \
  --work-root /home/pai/zxw/qwen_robotmanip_work \
  --stages 1,2,3 \
  --overwrite
```

`--overwrite` 会替换对应数据集已有的 Stage 目录。程序先在 `work-root` 完成计算，再发布到 `output-root`；运行前仍应确认目标数据集和配置无误。

### 分阶段执行

长任务可以依次运行：

```bash
# Stage 1
.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json \
  run --dataset-path /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero_plus \
  --dataset-id libero_plus \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/data_curation \
  --work-root /home/pai/zxw/qwen_robotmanip_work --stages 1 --overwrite

# Stage 2
.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json \
  run --dataset-path /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero_plus \
  --dataset-id libero_plus \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/data_curation \
  --work-root /home/pai/zxw/qwen_robotmanip_work --stages 2 --overwrite

# Stage 3
.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json \
  run --dataset-path /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero_plus \
  --dataset-id libero_plus \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/data_curation \
  --work-root /home/pai/zxw/qwen_robotmanip_work --stages 3 --overwrite
```

后续阶段会从相同 `output-root` 自动读取前序 Manifest。必须保持相同的 `dataset-id`、配置和输出根目录。

## 7. 输出说明

```text
data_curation/
├── stage1/<dataset>/
│   ├── manifest.json
│   └── labels/
│       ├── frame_flags.parquet
│       ├── episode_summary.parquet
│       ├── episode_filter.parquet
│       ├── step_validity.parquet
│       └── thresholds.parquet
├── stage2/<dataset>/
│   ├── manifest.json
│   └── labels/
│       ├── dimension_metrics.parquet
│       ├── episode_flags.parquet
│       └── episode_filter.parquet
└── stage3/<dataset>/
    ├── manifest.json
    └── labels/
        ├── frame_flags.parquet
        ├── episode_summary.parquet
        ├── episode_filter.parquet
        ├── step_validity.parquet
        └── thresholds.parquet
```

核心语义：

- `episode_filter.parquet`：当前 Stage 是否保留 Episode。
- `step_validity.parquet`：逐帧有效性和原因码。
- `frame_flags.parquet`：异常帧、失败维度和相对阈值比值。
- `dimension_metrics.parquet`：Stage 2 的 lag、DA 和 active steps。
- `thresholds.parquet`：本次运行实际使用的标定阈值。
- `manifest.json`：配置、输入路径、父阶段、输出文件和汇总统计，是审计锚点。

这些结果是逻辑过滤，不是新的物理 LeRobot 数据集。训练读取器必须应用 Manifest 中的 Episode Filter 和逐帧 Mask；不要直接把原始目录当作已经物理删除异常帧的数据。

## 8. 在 Embodied Studio 中刷新

Stage 产物完成后启动平台：

```bash
./scripts/local_deploy.sh start --skip-install
```

在页面中依次执行：

1. 运行目录扫描，使 Catalog 发现最新 Manifest。
2. 点击“更新搜索索引”。
3. 等待索引状态变为 `succeeded`。
4. 使用 Stage 标签筛选并抽检 Episode。

也可以调用 API：

```bash
curl -X POST http://127.0.0.1:8000/api/catalog/scan \
  -H 'content-type: application/json' \
  -d '{"mode":"quick"}'

curl -X POST http://127.0.0.1:8000/api/search/index \
  -H 'content-type: application/json' \
  -d '{"datasets":["libero_plus"]}'
```

Stage 搜索状态含义：

- `available`：当前 Episode 有可读取产物。
- `episode_pending`：Stage 正在运行，但尚未处理当前 Episode。
- `upstream_filtered`：Episode 已被前序阶段过滤。
- `not_generated`：没有发现对应 Stage 产物。

例如 Stage 5 的 `not_candidate` 可能是 Stage 4 或更早阶段过滤后的传播结果，不等价于 Stage 5 算法主动判失败。

## 9. 阈值调参与人工复核

不要只根据总异常数量调整阈值。建议按以下顺序抽检：

1. 在平台筛选异常状态并随机选择 Episode。
2. 查看 `failed_state_dimensions`、`failed_action_dimensions` 和 threshold ratio。
3. 将曲线放大到异常帧前后 5–10 帧，并与视频时间轴联动。
4. 分别统计平移、旋转、夹爪和 delta action 维度的误报。
5. 先修正 `angular_indices`、`quaternion_groups`、`gripper_indices` 和 `state_action_map`，再调整全局阈值。
6. 使用独立 smoke 输出重跑并复核，确认后再覆盖正式产物。

Stage 1 对 state/action 的任一维度命中都会标记整帧，且不会豁免夹爪。Stage 2 才会排除夹爪趋势比较，Stage 3 才会豁免夹爪的范围检查。

## 10. 常见问题

### `output exists (use --overwrite)`

目标 Stage 已存在。确认可以替换后增加 `--overwrite`；若要做实验，改用独立的 `output-root`。

### `stage2_mapping_required`

state/action 字段名不足以证明物理维度一一对应。配置显式 `state_action_map`，不要盲目启用 `allow_positional_mapping`。

### Stage 2 大量 `unscored`

检查有效连续段长度、`stage2_min_active_steps`、Stage 1 Mask、FPS 和映射维度。静止维度不会被伪造为高 DA。

### Stage 3 报同本体 `stage3_alpha` 不一致

同一个 `embodiment` 分组必须使用相同的 `stage3_alpha`，因为它们联合标定边界。

### 页面仍显示旧标签

确认 Stage Manifest 已更新，然后重新执行 Catalog 扫描和搜索索引刷新。索引 `queued` 只表示等待执行，完成状态应为 `succeeded`。

### 预览正常但 Stage 详情较慢

Episode 摘要来自 SQLite；首次展开详情仍需按 Manifest 路径读取对应 Parquet/JSON。不要把 Catalog SQLite 放在远程 FUSE 挂载中。

## 11. 旧物化流水线

`embodied_datasets/scripts/process_scripts/run_pipeline.py` 是另一套旧式执行器，会依次运行 Stage 1–5、Check 1–3 和 canonical representation，并写出新的 LeRobot 数据集。它的 Stage 1–3 算法、配置格式和输出语义与本文推荐的 Filter Manifest 流程不同：

- 使用 YAML `ProcessConfig`，不是上述 JSON 配置。
- 可能插值、裁剪或物化 state/action。
- 不生成 `vla_curation_filter` schema v2 Manifest。
- 不应与新流程的 Stage 目录混用。

只有明确需要物理物化数据且下游已适配该格式时才使用旧执行器。新数据治理、平台审计和训练 Mask 流程统一使用本文的 `qwen_robotmanip_curation/curate.py`。

## 12. 进一步阅读

- [Qwen-RobotManip Stage 1–3 实现说明](../embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/README.md)
- [Embodied Studio 平台接口与索引说明](PLATFORM.md)
- [项目构建与问题复盘](PROJECT_BUILD_RETROSPECTIVE.md)
