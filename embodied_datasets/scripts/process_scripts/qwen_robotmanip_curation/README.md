# Qwen-RobotManip Stage 1–3 清洗（LeRobot v3.0 → LeRobot v3.0）

本目录实现 [Qwen-RobotManip Technical Report](https://arxiv.org/abs/2606.17846) 第 2.4 节前三阶段。Stage 1–3 不使用 `/home/pai/zxw/公开数据集格式规范.docx`，不补齐 128 维，也不生成跨本体 canonical 表示。输入和每一阶段的主输出均为可由 `LeRobotDataset` 独立打开的 LeRobot v3.0 数据集；Parquet 审计表只是 sidecar。

每个阶段都保留输入数据的全部用户 feature（包括额外传感器字段、图像和所有视频视角），并由官方 LeRobot writer 重建 data、video、episode metadata、全局统计和连续索引。源数据始终只读。

## 三阶段分别处理什么

### Stage 1：突变检测与数值修复

对 `observation.state` 和 `action` 分别处理：

1. 对 NaN/Inf 做临时插值，仅用于稳定地计算检测指标；
2. 连续执行两次中值滤波，再执行 Savitzky–Golay 平滑，得到局部趋势；
3. 计算原信号相对趋势的 residual、二阶差分 acceleration、三阶差分 jerk；
4. 仅当 `residual` 超阈值并且 `acceleration` 或 `jerk` 也超阈值时，标记该维该帧；
5. `stage1_policy=frame` 时，仅对被标记的数值单元做同一 episode 内的线性时间插值，帧数、时间戳和视频不变；`episode` 时删除含异常帧的整个 episode。

默认阈值按数据集、feature、维度和指标分别计算：

```text
threshold = max(median + 8 × 1.4826 × MAD, q99.9, numerical_floor)
```

Stage 1 输出为 `data_curation/stage1/<dataset>/dataset/`，审计文件位于同级 `audit/`。

### Stage 2：action/state 趋势一致性与 episode 门控

Stage 2 读取 Stage 1 的 LeRobot v3.0 输出：

1. 对 state/action 做与 Stage 1 相同的平滑；
2. 若配置 `action_mode=delta`，先对 action 累加还原绝对趋势；
3. 在 `[0, stage2_max_lag_seconds]` 的因果窗口内用互相关寻找 action 领先 state 的最佳 lag；
4. 在双方都真实变化的 active steps 上计算方向一致率 DA；
5. 任一可评分的可比较维度 DA 低于阈值时，删除整个 episode；通过的 episode 不做时间平移。

可比较的 state/action 维只能通过以下方式确定：显式 `state_action_map`、state/action 的可靠同名字段，或明确设置 `allow_positional_mapping=true`。默认不会因为数组等长就盲目配对，也不依赖任何 128 维 canonical slot。

Stage 2 输出为 `data_curation/stage2/<dataset>/dataset/`，审计文件位于同级 `audit/`。

### Stage 3：本体内极值检测与数值修复

Stage 3 读取 Stage 2 的 LeRobot v3.0 输出。对于 `embodiment` 相同且 raw feature schema 相同的数据集，逐 feature、逐维汇总 q1/q99，并使用：

```text
lower = q1  - alpha × (q99 - q1)
upper = q99 + alpha × (q99 - q1)
```

默认 `alpha=0.1`。超界或非有限数值会被标记；夹爪维按论文要求豁免范围检查，但 NaN/Inf 仍会修复。被标记的数值单元在同一 episode 内做线性时间插值，帧数、时间戳和视频不变。

Stage 3 输出为 `data_curation/stage3/<dataset>/dataset/`，审计文件位于同级 `audit/`。

## 论文未公开部分：为什么这样补全

以下参数和物化策略是工程选择，不是论文作者公布的官方设置。每次运行都会在 `audit/run.json` 和阈值 Parquet 中记录最终值，便于追溯和重新标定。

### 两次 5 帧中值滤波 + 11 帧三阶 Savitzky–Golay

论文只给出“级联中值滤波 + Savitzky–Golay”，未给窗口和阶数。两次小窗口中值滤波能稳健抑制单点及短脉冲，又比一个大窗口更少损伤真实动作边缘；Savitzky–Golay 在降噪时保留局部斜率和曲率，适合随后计算高阶差分。三阶多项式足以描述短时位置、速度和加速度趋势，继续提高阶数更容易跟随噪声。短 episode 会自动缩短为合法奇数窗口。

窗口以帧计，因此不同 FPS 对应不同物理时间。默认值只是保守起点：低 FPS 或高速动作应减小窗口，高频强噪声可增大窗口。

### `median + 8×MAD` 与 q99.9 取最大值

论文要求分数据集/本体/旋转表示设置阈值，但未公开计算公式。median/MAD 不容易被少量坏点抬高；`1.4826×MAD` 在近似高斯分布下可解释为稳健标准差；8 倍尺度优先降低误杀。长期静止维的 MAD 可能接近零，因此再用 q99.9 和数值下限避免正常浮点波动被全部标记。逐维计算也避免弧度、米和归一化值共享不合理的绝对阈值。

联合条件 `residual AND (acceleration OR jerk)` 能同时排除两类误报：缓慢漂移可能 residual 大但没有突变，正常高速动作可能高阶差分大但仍贴合局部趋势。

### Stage 1/3 插值而不是删帧

论文允许排除异常 frame，但没有说明在带视频的 LeRobot 中如何物化。单独删除 Parquet 行会使视频帧、timestamp、frame index 和动作错位；每删一个帧就重编码并同步裁剪所有视角，成本高且会再次有损压缩。默认只替换被判异常的 state/action 单元，保留原时间网格和视频：

- 中间坏点由左右最近有效点线性插值；
- 边界坏点使用最近有效值；
- 某维整集均无有效值时填 0，同时审计表仍完整保留异常位置。

好处是多模态严格同步、修复范围最小、输出可以直接由标准 LeRobot loader 使用。代价是修复值是估计值，所以训练或复核时仍应保留 `audit/frame_flags.parquet`。若异常意味着整条轨迹不可信，应将 Stage 1 配成 `episode`，而不是删除孤立视频帧。

### Stage 1 的夹爪豁免

论文只明确提出 Stage 3 夹爪豁免；Stage 1 也沿用配置中的夹爪豁免，是额外的保守选择。开/关式夹爪本来就是双峰和离散跳变，连续轨迹突变检测容易产生大量假阳性。代价是夹爪传感器毛刺不会被 Stage 1 捕获；连续夹爪位置数据可按数据集清空该豁免配置。

### Stage 2 的 DA=0.65、0.5 秒 lag 与 active steps

论文给出的 DA 常用范围是 0.6–0.7，默认取中点 0.65，在明显错位拦截与控制死区/轻微回弹容忍之间折中。lag 用秒配置并按 FPS 换算，使不同帧率的物理窗口一致；只搜索非负 lag，落实“命令先于状态”的因果方向，同时把双向搜索结果写入审计用于发现反向时钟问题。

DA 只统计 state/action 一阶差分都超过各自 `q90(|Δ|)×1e-3` 的位置，且默认至少 10 个 active steps。否则大量静止的 0 符号会让完全不响应的关节获得虚假高分。短或几乎静止的 episode 会标记为不可评分，但不会仅因样本不足被拒绝。

Stage 2 不按最佳 lag 物理平移数据。论文这一阶段用于发现并排除不同步轨迹；平移会裁掉首尾并要求所有视频视角同步重写，而且单一 lag 未必适用于 episode 的每一段。把它作为 episode 质量门控可避免制造新的时序假设。

### Stage 3 的 `alpha=0.1` 与定步长采样

直接使用 `[q1,q99]` 即使数据正常也会按定义排除约 2% 尾部；向两侧扩展中央 98% 区间宽度的 10%，可以容纳有限样本误差、合法边界动作和轻微分布漂移，同时仍能拦截远离主体分布的飞点。alpha 太小会伤及少见但合法的极限姿态，太大会留下异常，应结合每维阈值、异常率和视频复核。

为限制数亿帧数据的内存占用，标定默认按固定 stride 确定性采样，每个数据集/本体组最多 20 万帧。它有固定内存上界且可复现，但周期数据可能发生采样混叠；对此可提高 `calibration_samples`，或改用分层/确定性哈希采样后对照。

## 配置与运行

复制 `config.example.json` 后按数据集覆盖参数。Stage 2 若字段名不能可靠配对，必须填写：

```json
{
  "datasets": {
    "my_dataset": {
      "embodiment": "my_robot",
      "state_action_map": [[0, 0], [1, 1], [2, 2]],
      "gripper_indices": {"observation.state": [6], "action": [6]}
    }
  }
}
```

`config.example.json` 已为 `libero_plus*` 配置专用语义：state 是绝对 EEF 位姿加双指夹爪，action 是 6D EEF 增量加夹爪命令，因此设为 `action_mode=delta`，只比较可逐分量可靠积分的 XYZ 平移 `[0,1,2]`。旋转增量必须在明确轴角尺度、坐标系和左/右乘约定后用 SO(3) 复合，不能直接逐分量累加；当前保守地不让旋转维产生错误的 episode 拒绝。state `[6,7]` 和 action `[6]` 作为夹爪维豁免。

清单检查：

```bash
./.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json inventory \
  --input-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0
```

小范围运行：

```bash
./.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json run \
  --dataset-path /path/to/lerobot_v3_dataset --dataset-id my_dataset \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/data_curation \
  --work-root /home/pai/zxw/qwen_robotmanip_work \
  --max-episodes 20 --overwrite
```

完整运行去掉 `--max-episodes`。`--output-root` 的默认值就是 `/mnt/data/embodied_datasets/public_datasets_staging/data_curation`，命令中可以省略；程序会分别写入该目录已有的 `stage1`、`stage2`、`stage3`。该区域与原始输入 `lerobot_v3_0` 隔离，不会被输入数据集发现逻辑误扫。按 `--stages 1`、`--stages 2`、`--stages 3` 分开执行时保持同一 `output-root`；程序会优先读取前一阶段的 `dataset/`。视频通过官方 writer 解码并重编码，以保证删除 episode 后索引、容器时间段和元数据一致，因此全量运行的主要成本是视频 I/O/编码，不是三个低维信号统计步骤。

## 输出结构

```text
/mnt/data/embodied_datasets/public_datasets_staging/data_curation/
  stage1/<dataset>/
    dataset/                 # 完整 LeRobot v3.0
    audit/
      frame_flags.parquet
      episode_summary.parquet
      thresholds.parquet
      run.json
  stage2/<dataset>/
    dataset/                 # 完整 LeRobot v3.0
    audit/
      dimension_metrics.parquet
      episode_flags.parquet
      run.json
  stage3/<dataset>/
    dataset/                 # 完整 LeRobot v3.0
    audit/
      frame_flags.parquet
      episode_summary.parquet
      thresholds.parquet
      run.json
```

`audit/` 用于解释、复核和重调参数；训练数据入口应指向各阶段的 `dataset/`，而不是 audit Parquet。
