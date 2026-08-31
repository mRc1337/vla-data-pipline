# Qwen-RobotManip 三阶段清洗（LeRobot v3）

对应论文 [Qwen-RobotManip Technical Report](https://arxiv.org/abs/2606.17846) 第 2.4 节的前三阶段。程序只读取 `observation.state`、`action` 和索引列，不解码视频，也不修改源数据。

论文没有公布 Stage 1 的窗口/数据集阈值和 Stage 3 的 `alpha`。本实现明确记录这些补充选择：Stage 1 使用两次中值滤波和 Savitzky–Golay 平滑，以 `max(median + 8*MAD, q99.9)` 做逐数据集、逐维阈值；Stage 2 默认 DA 阈值为 0.65；Stage 3 默认 `alpha=0.1`。所有参数均可在 JSON 配置中覆盖。

Stage 1 默认只生成帧级剔除。论文只在能够确认突变必然来自碰撞的 InternData-A1 上整集删除；若某个本地数据源也满足这一条件，再把该数据集的 `stage1_policy` 显式改为 `episode`，不要仅因它是真机数据就整集删除。

## 论文未公开部分的工程补全与原因

以下默认值是为了让论文描述能够在当前 LeRobot v3 数据上可复现地运行，并非论文作者公布的官方参数。选择原则是：先保证源数据只读、阈值可审计、不同 FPS/本体不会被一个绝对阈值混在一起，再通过真实数据 smoke 调整。每次运行都会把最终配置和阈值写入 `run.json`/`thresholds.parquet`，因此后续修改仍可追溯。

### Stage 1：平滑窗口和突变阈值

论文只说明“级联中值滤波 + Savitzky–Golay 平滑”，没有给出级数、窗口和多项式阶数。本实现默认使用两次 5 帧中值滤波，再使用 11 帧、3 阶 Savitzky–Golay：

- 两次小窗口中值滤波对单点/连续少量脉冲噪声稳健，但比一个很大的窗口更能保留真实动作边缘；
- 11 帧 Savitzky–Golay 能在平滑噪声的同时保留局部斜率和曲率，后续计算二阶/三阶差分时不会像普通移动平均那样明显削弱峰值；
- 3 阶多项式足以表达短时间内的位置、速度和加速度趋势，阶数继续增大容易追随噪声；
- 短 episode 会自动缩短为合法奇数窗口，避免因为长度不足直接丢数据。

代价是窗口以“帧”计而不是以秒计：3 FPS 与 60 FPS 对应的物理时间不同。因此它只是安全起点；高频强噪声数据可增大窗口，低 FPS 或高速动作数据应减小窗口。配置支持按 dataset pattern 覆盖 `median_kernels`、`savgol_window` 和 `savgol_polyorder`。

论文还说明阈值按数据集/本体/旋转表示等设置，但没有给出计算方法。本实现对 residual、二阶差分、三阶差分分别采用：

```text
threshold = max(median + 8 × 1.4826 × MAD, q99.9, numerical_floor)
```

这样处理的原因和收益是：

- `median + 8×MAD` 不容易被少量极端坏点反向拉高，比 mean/std 更适合先有污染的数据；
- `1.4826×MAD` 在近似高斯分布下可解释成稳健标准差，`8` 是偏保守的起点，优先降低误杀；
- 当某维长期不动导致 MAD 接近 0 时，`q99.9` 提供数据驱动的下限，避免任意微小浮点变化都被标记；
- `numerical_floor` 防止全零维阈值为 0；
- 阈值按数据集、信号类型和维度独立计算，不会把弧度、米、归一化夹爪值混成同一尺度。

Stage 1 使用论文给出的联合条件 `residual 超阈值 AND (acceleration 或 jerk 超阈值)`，而不是三项任一超阈值。好处是缓慢漂移可能 residual 大但导数不突变，正常快速运动可能导数大但仍贴合平滑趋势，两者都不会仅凭单一指标被误杀。

当前实现也豁免配置中的夹爪维。论文只在 Stage 3 明确要求夹爪豁免，因此这是额外的保守选择：夹爪命令常是开/关双峰信号，正常切换本来就是离散跳变，直接套连续轨迹检测会产生大量假阳性。代价是夹爪传感器的真实瞬时毛刺不会由 Stage 1 捕获；若某数据集记录的是连续夹爪位置并需要检查，应为它单独拆分 Stage 1/Stage 3 的豁免配置后再启用，而不是直接照搬默认列表。

### Stage 1：默认帧级清单而不是立即重写数据

论文允许从删帧到整 episode 删除的不同策略，但没有给通用规则。本实现默认 `stage1_policy=frame`，并输出稀疏 rejection manifest，不在原 Parquet 上插值或删除：

- LeRobot 的 Parquet 行、视频时间戳、episode/frame index 相互关联，单独删除 Parquet 行会破坏视频对齐；
- 稀疏清单可重复调阈值，不需要复制或重编码 TB 级视频；
- 可视化复核后，训练 loader 可以做 anti-join，最终物化流程也可以一次性应用所有阶段结果；
- 只有确认“任一突变都意味着整条轨迹不可信”时才配置 `stage1_policy=episode`。

Stage 2 仍使用完整连续 episode，不先删除 Stage 1 的局部帧，因为从中间挖掉帧会改变 lag。Stage 1 帧级异常会在 Stage 3 标定和最终输出中排除。

### Stage 2：DA、lag 和静止区间

论文给出的 DA 常用范围是 0.6～0.7。本实现取中点 `0.65`：相比 0.6 更能拦截明显错位，相比 0.7 对量化噪声、控制死区和轻微回弹更宽容。它不是普适常数；应先看 `dimension_metrics.parquet` 的 DA 分布，如果好/坏 episode 形成两个峰，应把阈值放在谷底，而不是机械保持 0.65。

论文没有公布 cross-correlation 的搜索窗口。本实现默认只在 action 领先或同时发生的 `[0, 0.5 秒]` 内选最优 lag，并按各数据集 FPS 换算成帧：

- 使用秒而不是固定帧数，使 5 FPS 和 30 FPS 数据具有可比较的物理窗口；
- 只用非负 lag 落实“命令先于状态”的因果约束，不会用一个相关性较高但物理上反向的 lag 修饰坏数据；
- 另外计算 `[-0.5, +0.5 秒]` 的非约束最优 lag 并写入诊断列，便于发现 state 反而领先 action 的时钟问题；
- 有明显慢执行器或低频控制链时，应根据实测响应延迟增大 `stage2_max_lag_seconds`，否则真实但缓慢的响应可能得到较低 DA。

DA 只统计 state 和 action 一阶差分都超过各自 `q90(|Δ|)×1e-3` 的 active steps，且默认至少需要 10 个 active steps。原因是大量静止值的符号都是 0；若把它们算作一致，完全不响应 action 的关节也可能获得虚假的高 DA。要求两边都实际变化并设置最小样本数，可以让 DA 表示“运动方向是否一致”，代价是很短或几乎静止的 episode 会成为不可评分而不是被拒绝。

Stage 2 只比较 canonical joint slots，不按相同数组下标盲配 state/action。这样能避免把末端位置、四元数、夹爪或环境状态当成关节进行相关分析。delta action 会先积分恢复绝对趋势；absolute action 直接比较。任一可评分关节低于 DA 阈值就删除整个 episode，因为时间戳错位或丢包通常是 episode 级记录问题，局部删帧无法恢复可靠因果关系。

### Stage 3：`alpha=0.1` 和分位数近似

论文公开了范围公式，但没有公布 `alpha`。本实现默认：

```text
lower = q1  - 0.1 × (q99 - q1)
upper = q99 + 0.1 × (q99 - q1)
```

选择 0.1 的原因是，如果直接使用 `[q1,q99]`，即使数据完全正常也会按定义删除约 2% 的尾部；向两侧各扩展中央 98% 区间宽度的 10%，能容纳有限样本误差、任务边界动作和轻微分布漂移，同时仍会排除远离主体分布的极值。好处是比固定物理上下限更容易扩展到多种机器人，又比无限放宽更能保护后续 q01/q99 normalization。

`alpha` 的主要权衡是：过小会把少见但合法的极限姿态删掉，过大会留下真正异常值。建议同时查看每维 `q01/q99/lower/upper`、Stage 3 剔除率和对应视频；若异常集中在任务的合法末端姿态，应增大 alpha 或按任务/本体细分统计，若明显传感器飞点仍落在范围内则减小 alpha。不要只根据“期望剔除百分比”调参。

为适应当前数亿帧数据和 OSSFS，本实现不把全部数值拼进内存，而是按固定 stride 确定性采样，默认每个数据集/本体组最多 20 万帧：

- 内存和本地临时空间有稳定上界；
- 同一输入和配置会得到相同样本及阈值，便于复现；
- 顺序扫描只读取 state/action 列，不解码视频，适合本机 OSSFS。

这是工程近似而非论文要求。固定 stride 在强周期数据上可能产生采样混叠；如果阈值对少量尾部样本非常敏感，应提高 `calibration_samples`，或改成确定性哈希/分层 episode 采样后做对照。当前 720 GiB 内存足够增大样本，但远端 I/O 仍是主要瓶颈。

Stage 3 按 `embodiment` 合并统计，且要求同组 raw schema 和 raw-to-canonical 映射一致。这样 Piper 与 Piper-X 可以共享本体范围，但不会把不同语义的第 3 列误合并。夹爪按论文要求豁免；NaN/Inf 无论分位数如何都会标记为异常。

### 为什么不先补成 128 维再计算阈值

三阶段在原始有效维度上计算，并在 manifest 中同时记录 raw 和 canonical 维号；最终才生成 128 维向量及 mask。这样做可避免大量 padding 0 进入中位数、MAD、相关性和分位数，导致不存在的自由度改变真实维度的统计结果。`canonical_indices` 只负责证明不同数据集的维度语义及 Stage 2 关节配对，padding 位始终 `mask=False`。

### 推荐的标定顺序

1. 先对每个 schema/本体抽 20～100 个 episode 做 smoke，不直接全量物化。
2. 查看 Stage 1 的阈值倍数和异常帧视频，先确定窗口，再调整 MAD scale/quantile floor。
3. 查看 Stage 2 各关节的 lag、active steps 和 DA 分布，确认 action mode 与映射无误后再定 DA 阈值。
4. Stage 3 必须在 Stage 1/2 参数冻结后重新标定，因为上游坏点会改变 q1/q99。
5. 固化配置、运行全量并保存 `run.json`、threshold parquet 和代码 commit；不同配置产物不要混用。

## 安装

项目已并入 `/home/pai/zxw/vla-data-pipeline/embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation`，依赖复用仓库根目录的 `.venv`，不再创建第二套虚拟环境。

当前机器有 40 个 CPU 核、720 GiB 内存和 4 张 A800 80GB；三个阶段都是低维时序统计，瓶颈是 `/mnt/data` 的 OSSFS I/O，GPU 不会带来收益。当前仍有多项转换任务向 staging 根目录写入，因此先按已完成子数据集串行运行。转换全部结束后可按 2～4 个互不重叠的 dataset selector 并行，但每个进程内部保持 Arrow `use_threads=False`，避免大量小对象请求拖慢 OSSFS。

## 先做清单检查

```bash
./.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.json inventory \
  --input-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --output /mnt/data/qwen_robotmanip_curation/inventory.json
```

默认发现两层内的数据集根目录（已经覆盖当前的顶层数据集和 LIBERO 子集）；若后续转换产物更深，可加 `--discovery-depth 3`。对单个已知路径优先使用 `run --dataset-path ...`，避免扫描仍在转换的大目录。

复制 `config.example.json` 为 `config.json`，补齐 inventory 中 `stage2_mapping_required` 的数据集。Stage 2 只能比较语义一致的关节状态和绝对关节动作；末端增量动作不能盲目按数组位置配到关节状态。增量关节动作须设 `action_mode: delta`，并配置 `state_action_map`。

本机当前 inventory 共发现 822 个 LeRobot 根：93 个可由字段名建立 Stage 2 映射，409 个需要人工补关节映射，320 个缺少默认 `observation.state` 或 `action`。只有配置中明确写入 `canonical_indices` 的数据集才算完成跨本体契约；`config.example.json` 已覆盖 4 个 RoboDojo 数据集。不能把“数组长度相同”当成语义相同后直接清洗。

## 小范围验证

```bash
./.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.json run \
  --dataset-path /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/robodojo_real_piper_lerobot_v30 \
  --dataset-id robodojo_real_piper_lerobot_v30 \
  --output-root /mnt/data/qwen_robotmanip_curation_smoke \
  --work-root /home/pai/zxw/qwen_robotmanip_work \
  --max-episodes 20 --overwrite
```

## 全量运行

确认所有转换进程结束、配置审核完毕后运行：

```bash
./.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.json run \
  --input-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --output-root /mnt/data/qwen_robotmanip_curation \
  --work-root /home/pai/zxw/qwen_robotmanip_work
```

在配置尚未覆盖全部 822 个根目录时，应使用一个或多个 `--select` 只运行已审核子集。例如当前已经全量验证的 Piper 本体：

```bash
./.venv/bin/python embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/curate.py \
  --config embodied_datasets/scripts/process_scripts/qwen_robotmanip_curation/config.example.json run \
  --input-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --select 'robodojo_real_piper*_lerobot_v30' \
  --output-root /mnt/data/qwen_robotmanip_curation \
  --work-root /home/pai/zxw/qwen_robotmanip_work
```

也可以用 `--select 'robodojo*'`（可重复）分批执行。阶段有依赖关系：Stage 2 会继承 Stage 1 的整集剔除，Stage 3 会继承前两阶段的帧/episode 剔除。若分开运行，保持同一个 `output-root` 并依次用 `--stages 1`、`--stages 2`、`--stages 3`。

## 输出

每个数据集会产生：

- `stage1_sudden_change/<dataset>/frame_flags.parquet`、`episode_summary.parquet`、`thresholds.parquet`；
- `stage2_trend_alignment/<dataset>/dimension_metrics.parquet`、`episode_flags.parquet`；
- `stage3_extreme_value/<dataset>/frame_flags.parquet`、`episode_summary.parquet`、`thresholds.parquet`；
- 每个阶段都有 `run.json`，记录参数、范围、计数和时间。

这些是稀疏 rejection manifests。训练时以 `(episode_index, frame_index)` 做反连接过滤；不要只删除 Parquet 行，否则会破坏 LeRobot 的视频时间戳和 episode 索引。Stage 2 的 `episode_flags.parquet` 应整集排除。

## 128 维跨本体表示

布局严格采用 `/home/pai/zxw/公开数据集格式规范.docx`：state 为 `[0:35]` 单臂、`[35:70]` ARM2、`[70:128]` 预留；action 为 `[0:34]` 单臂、`[34:69]` ARM2、`[69:128]` 预留。由于 ARM2 的有效结构与 ARM1 相同、只有 34 维，action 第 68 维定义为固定零且 `mask=False` 的 ARM2 padding。

每个数据集必须在 `canonical_indices` 中声明“原始列 -> 128 维 canonical 列”。清单中的阈值和异常帧同时记录原始维号与 canonical 维号；`output-root/canonical_layout.json` 是机器可读契约。最终 LeRobot 数据的 128 维物化继续复用上级目录的 `unify_representation.py`，并保留 `observation.state_canonical_mask` 与 `action_canonical_mask`。
