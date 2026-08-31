# Qwen-RobotManip 三阶段清洗（LeRobot v3）

对应论文 [Qwen-RobotManip Technical Report](https://arxiv.org/abs/2606.17846) 第 2.4 节的前三阶段。程序只读取 `observation.state`、`action` 和索引列，不解码视频，也不修改源数据。

论文没有公布 Stage 1 的窗口/数据集阈值和 Stage 3 的 `alpha`。本实现明确记录这些补充选择：Stage 1 使用两次中值滤波和 Savitzky–Golay 平滑，以 `max(median + 8*MAD, q99.9)` 做逐数据集、逐维阈值；Stage 2 默认 DA 阈值为 0.65；Stage 3 默认 `alpha=0.1`。所有参数均可在 JSON 配置中覆盖。

Stage 1 默认只生成帧级剔除。论文只在能够确认突变必然来自碰撞的 InternData-A1 上整集删除；若某个本地数据源也满足这一条件，再把该数据集的 `stage1_policy` 显式改为 `episode`，不要仅因它是真机数据就整集删除。

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
