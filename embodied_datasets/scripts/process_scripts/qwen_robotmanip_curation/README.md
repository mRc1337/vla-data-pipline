# Qwen-RobotManip Stage 1–3（LeRobot v3.0 Filter Manifest）

本目录实现 [Qwen-RobotManip Technical Report](https://arxiv.org/abs/2606.17846) 第 2.4 节前三阶段的数据过滤语义。输入保持 LeRobot v3.0，不使用 `/home/pai/zxw/公开数据集格式规范.docx`，也不生成跨本体 canonical 表示。

原始 `data/`、`videos/` 和 `meta/` 始终只读。Stage 只写标签、逐帧有效性 Mask、Episode Filter 和 Manifest；不复制或重编码视频，不重写源 Parquet，不插值或替换 state/action。新产物格式为 `vla_curation_filter` schema v2，不能与旧 `vla_curation_overlay` repair 产物混用；遇到旧 Manifest 时必须重跑前序 Stage。

## Stage 1：Sudden Change Detection

对 `observation.state` 和 `action` 的每一维执行：

1. NaN/Inf 仅在检测计算的临时数组中填充，源值不变；
2. 级联中值滤波和 Savitzky–Golay 平滑提取趋势；
3. 计算 raw 与趋势的 absolute residual、二阶有限差分 acceleration、三阶有限差分 jerk；
4. 当 `residual > residual_threshold AND (acceleration > acceleration_threshold OR jerk > jerk_threshold)` 时标记该维、该帧；
5. NaN/Inf 无条件标记异常。

阈值按数据集、feature、维度和指标分别标定。当前工程默认值为：

```text
threshold = max(median + 8 × 1.4826 × MAD, q99.9, numerical_floor)
```

过滤策略由数据集配置 `stage1_exclusion` 选择：

- `frame`：Episode 保留，只有异常帧的 `stage1_valid=False`；
- `episode`：任一异常帧导致 Episode 拒绝，同时该 Episode 所有帧 `stage1_valid=False`。

例如：

```json
{
  "defaults": {"stage1_exclusion": "frame"},
  "datasets": {
    "libero_plus*": {"stage1_exclusion": "frame"},
    "InternData-A1*": {"stage1_exclusion": "episode"}
  }
}
```

Stage 1 不豁免夹爪突变检测。`gripper_indices` 只用于 Stage 2 排除不可信的连续趋势比较，以及 Stage 3 豁免数值范围检查。

## Stage 2：State-Action Trend Alignment

Stage 2 只遍历 Stage 1 接受的 Episode，并只使用 Stage 1 有效帧：

1. 若 action 是 delta，先按原始物理时间顺序积分为 absolute action，再应用 Mask；
2. 将 Mask 切分成连续有效区间，各区间独立归一化、平滑、互相关、lag 估计和一阶差分，绝不跨无效区间拼接；
3. 在可靠映射的 state/action 维度上搜索非负因果 lag；
4. 各连续段的 directional agreement 按 active-step 数加权汇总；
5. 任一可评分维度 `DA < stage2_da_threshold` 时拒绝整个 Episode；无足够 active steps 的维度记录为 unscored，不伪造分数；
6. 不物理平移 action，不裁剪首尾，不修改任何数值。

可比较维度来自显式 `state_action_map`、可靠的同名字段，或主动设置 `allow_positional_mapping=true`。默认不会只因为两个数组等长就盲目配对。

## Stage 3：Extreme Value Filtering

Stage 3 只读取 Stage 1 有效帧和 Stage 2 接受的 Episode。对相同 `embodiment` 且 raw schema 相同的数据，按 feature 和 dimension 标定：

```text
q1 = percentile(values, 1)
q99 = percentile(values, 99)
lower = q1 - alpha × (q99 - q1)
upper = q99 + alpha × (q99 - q1)
```

超界值和 NaN/Inf 对应帧的 `stage3_valid=False`。夹爪维豁免上下界检查，但 NaN/Inf 仍无效。Stage 3 不因单个极值拒绝整个 Episode，也不截断、插值或修正数值。

## 论文要求与工程补全的边界

论文明确给出了前三阶段的目标语义：Stage 1 检测突变并按数据集进行 frame removal 或 trajectory/Episode removal；Stage 2 用 state/action 趋势方向一致性排除低 DA Episode；Stage 3 用本体内分位数范围过滤异常帧，夹爪范围例外；训练时需把逐帧有效性传播为 action chunk 的因果 Mask。

论文没有公布以下具体实现值：中值滤波窗口和级联次数、Savitzky–Golay 窗口及阶数、Stage 1 数值阈值公式、Stage 2 最大 lag 与 active-step 规则、Stage 3 `alpha`，以及 LIBERO-Plus 应采用 `frame` 还是 `episode`。本实现的选择及原因如下：

- 两次 5 帧中值滤波能抑制单点和短脉冲，11 帧三阶 Savitzky–Golay 在降噪时较好保留斜率与曲率；短 Episode 自动缩短窗口。窗口以帧计，不同 FPS 的数据应单独校准。
- `median + 8×MAD` 对少量离群点不敏感，q99.9 和数值下限避免静止维因 MAD 接近零而误报。这是保守起点，不是论文官方阈值。
- Stage 2 默认 `DA=0.65`，取论文典型区间 0.6–0.7 的中点。默认最大 lag 为 0.5 秒，并以 FPS 换算帧数；active step 排除双方近乎静止的差分，避免静止段获得虚假高分。
- 连续有效段独立计算可防止 Mask 两侧原本不相邻的帧被当作一步运动；按 active steps 加权可让长的有效运动段贡献与其证据量相称。
- Stage 3 默认 `alpha=0.1`，给中央 98% 区间留出有限样本误差和合法边界动作余量。它必须结合异常率和人工复核重新校准。
- 大规模标定默认采用固定 stride、最多 20 万帧，提供确定性和固定内存上界；周期数据可能采样混叠，可提高 `calibration_samples`。
- `libero_plus*` 默认 `stage1_exclusion=frame`，因为孤立数值异常不应连带丢弃整条多模态轨迹；这是本项目策略，不是论文指定值。若某数据集的突变意味着整次采集失真，应改成 `episode`。

逻辑删除而非物理删行的好处是：原始视频、timestamp、`frame_index`、多相机和 action/state 始终严格对齐；没有每阶段重复视频带来的存储和有损重编码；过滤决定可审计、可回滚。代价是训练读取器必须应用 Manifest/Mask，或在最终选定子集后只物化一次标准 LeRobot 数据集。

## 逐帧有效性与训练 Mask

Stage 1 和 Stage 3 的 `step_validity.parquet` 至少包含：

```text
episode_index, frame_index, index,
stage1_valid, stage2_valid, stage3_valid, valid, reason_codes
```

Stage 1 写入 `valid = stage1_valid`。Stage 3 继承 Stage 1 Mask，且只遍历 Stage 2 接受的 Episode，最终：

```python
valid = stage1_valid & stage2_valid & stage3_valid
```

训练 action chunk 使用：

```python
chunk_valid = validity[start:start + horizon]
causal_valid = np.logical_and.accumulate(chunk_valid)
loss_mask = causal_valid[:, None] & embodiment_slot_mask[None, :]
```

累计只作用于当前采样 chunk，不会把物理 Episode 中异常点之后的所有帧永久删除。代码提供 `causal_chunk_validity()` 和 `causal_loss_mask()`。

## 输出结构与 Manifest

```text
/mnt/data/embodied_datasets/public_datasets_staging/data_curation/
├── stage1/<dataset>/
│   ├── manifest.json
│   └── labels/{frame_flags,episode_summary,episode_filter,step_validity,thresholds}.parquet
├── stage2/<dataset>/
│   ├── manifest.json
│   └── labels/{dimension_metrics,episode_flags,episode_filter}.parquet
└── stage3/<dataset>/
    ├── manifest.json
    └── labels/{frame_flags,episode_summary,episode_filter,step_validity,thresholds}.parquet
```

Stage 目录中不会生成 `data/`、`videos/`、`repairs/` 或 `meta/`。Manifest 使用：

```text
format = vla_curation_filter
schema_version = 2
video_policy = reference_original
repair_files = []
output_format = lerobot_v3.0-filter-manifest
```

`parent_manifest` 串联前一阶段，`episode_filter` 和 `validity_files` 逐级生效；所有参数、策略、检测器版本、输入路径和输出汇总均写入 Manifest。

## 运行

先检查数据集与 Stage 2 映射：

```bash
./.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json inventory \
  --input-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0
```

LIBERO-Plus 全量运行：

```bash
./.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json run \
  --dataset-path /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero_plus \
  --dataset-id libero_plus \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/data_curation \
  --work-root /home/pai/zxw/qwen_robotmanip_work \
  --overwrite
```

小范围验证加 `--max-episodes 20`。也可用 `--stages 1`、`--stages 2`、`--stages 3` 分开运行，但必须保持相同 `output-root`；后续阶段会读取前序新格式 Manifest。旧正式结果必须先移走或使用 `--overwrite` 重跑，不能作为新流程的父节点。
