# vla_data_pipeline

VLA（视觉-语言-动作）机器人操作数据集的清洗/对齐流水线：输入一份 LeRobot
v3.0 格式数据集 + 一份清洗/对齐参数文件，输出清洗对齐后的 LeRobot v3.0
数据集。不做数据下载、格式转换、下载完整性校验——只做"已经是 LeRobot
v3.0 格式的数据"到"清洗对齐后的 LeRobot v3.0 数据"这一步。

## 项目结构

```
vla_data_pipeline/
├── embodied_datasets/scripts/process_scripts/  # 全部实际代码
├── embodied_datasets/scripts/inspect_tool/     # 清洗前后对比的 Streamlit 可视化工具
├── requirements.txt        # 全仓库共用的依赖清单
└── pyproject.toml          # pytest 配置（指向 process_scripts/tests）+ 项目元数据
```

## 环境搭建

```bash
sudo apt update && sudo apt install python3.12 python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 系统依赖

```bash
sudo apt update && sudo apt install ffmpeg
```

## 用法

```bash
python3 run_pipeline.py \
    --input /any/path/to/lerobot_v3_0_dataset \
    --output /any/path/to/output_dataset \
    --config /any/path/to/process_config.yaml
```

## 通用数据集 schema-dump 工具

已下载的原始数据集格式不止一种（HDF5、RLDS/TFDS，以后还会有更多），在给
新格式写转换脚本之前，先用这个工具探测一遍目录结构和字段 schema——**只读
header/sidecar 元数据，从不读取完整数组、不解码视频、不读取 `.tfrecord`
二进制内容**，方便把探测结果整份复制回本地开发环境，而不用把原始数据本身
传出服务器。

单个数据集，直接把 JSON 打到 stdout：

```bash
python3 embodied_datasets/scripts/convert_scripts/dump_dataset_schema.py \
    --dataset-root /data/public_datasets_raw/<dataset_uid>
```

批量探测 `--raw-root` 下每个一级子目录（每个当作一个 dataset_uid），各写
一份 `<uid>.json`，stdout 额外打一行一个数据集的精简汇总，单个数据集探测
出错不中断整批：

```bash
python3 embodied_datasets/scripts/convert_scripts/dump_dataset_schema.py \
    --raw-root /data/public_datasets_raw --all \
    --output-dir /tmp/schema_reports
```

格式默认自动探测（有 `dataset_info.json` → RLDS；否则有 `.h5`/`.hdf5` →
HDF5；否则 `unknown`，仍会正常输出文件清单/目录树，不报错），也可以用
`--format hdf5|rlds` 显式指定。RLDS/TFDS 的完整 feature schema 解析依赖
`tensorflow_datasets`（本仓库不把它列为硬依赖），需要时再装：

```bash
pip install tensorflow-cpu==2.15.0 tensorflow-datasets==4.9.9
```

没装的话该工具仍会把 `dataset_info.json`/`features.json` 的原始内容收进
报告，只是不解码成结构化的 shape/dtype，报告里的 `fidelity` 字段会写清楚
当前处于哪种精度。

## ARCap 原始数据转换为 LeRobot v3.0

ARCap 的五分区、无视频、float64 点云 staging 转换也已实现；它使用 phase-group
粒度的 persistent-worker direct-commit、可验证恢复和 OSSFS marker 发布；四进程是代表性
多单元 warm 证据下的正式候选，短冷启动基准会按门禁拒绝无加速配置。调查、字段映射、容量计划、
正式命令与独立评估见
[`ARCAP_CONVERSION.md`](embodied_datasets/scripts/convert_scripts/ARCAP_CONVERSION.md)。全量转换未启动。

## Mobile ALOHA 原始数据转换为 LeRobot v3.0

转换入口只负责把原始 HDF5 搬运为 LeRobot v3.0，不执行清洗、重采样、归一化或
canonical 128 维映射。它会递归发现 episode，因此同时支持移动任务的一层目录和
静态 co-training 的两层目录：

```text
public_datasets_raw/<dataset_uid>/<language_instruction>/episode_*.hdf5
```

先执行预检：

```bash
cd /home/pai/zxw/vla-data-pipeline

env HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_mobile_aloha_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/mobile_aloha_staging \
  --dataset-uid mobile_aloha \
  --inspect-only
```

全量预检通过后，从项目根目录启动正式转换：

```bash
cd /home/pai/zxw/vla-data-pipeline

mkdir -p /home/pai/zxw/mobile_aloha_logs
mkdir -p /home/pai/zxw/.cache/huggingface

nohup env \
  HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_mobile_aloha_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/mobile_aloha_staging \
  --dataset-uid mobile_aloha \
  --resume \
  --streaming-encoding --video-codec h264 --video-preset fast \
  --eta-interval-seconds 10 \
  > /home/pai/zxw/mobile_aloha_logs/convert.log 2>&1 &

echo $! > /home/pai/zxw/mobile_aloha_logs/convert.pid
```

实时观察 ETA 和检查后台进程：

```bash
tail -f /home/pai/zxw/mobile_aloha_logs/convert.log
ps -fp "$(cat /home/pai/zxw/mobile_aloha_logs/convert.pid)"
```

ETA 日志包含全局已转换帧数、百分比、吞吐率、已用时间、剩余时间以及当前
partition/episode；默认每 10 秒刷新，并在每个 episode 保存后强制输出。最终输出位于
`/home/pai/zxw/mobile_aloha_staging/lerobot_v3_0/mobile_aloha`。只有进程返回 0，且日志末尾
出现以下两行，才能视为本地结构转换完成：

```text
wrote 3 partitions / 1103 episodes / 971850 frames
completed: converted=1, skipped=0
```

不要把正式转换的 `--staging-root` 直接指向 `/mnt/data`：该 OSSFS/FUSE 挂载不支持 MP4
muxer 关闭文件时所需的 seek-back/ftruncate。应先写服务器本地目录，完成验证后再同步到 OSS。

### 流式编码与 NVENC 小样本验证

`--streaming-encoding` 会直接把解码后的 RGB 帧送入视频编码器，跳过同步写临时 PNG 再读回的
中转。离线转换会自动把每相机队列设为“最长 episode 帧数 + 1”；手工指定更小的
`--encoder-queue-maxsize` 会被拒绝，因此不会触发 LeRobot 的队列满丢帧分支。转换完成后还会
逐相机核验 MP4 的总帧数、codec 和 FPS。

`--resume` 会在正式输出旁保留确定性 checkpoint，并只从下一个尚未完成的 episode 继续。
收到 `Ctrl-C` 或 `SIGTERM` 时，当前未完成 episode 会被丢弃，parquet/metadata 会关闭，重新执行
完全相同的命令即可续传。恢复前会核对源 HDF5 的路径、大小、mtime、schema 和全部视频编码参数；
任一项变化都会拒绝误续。`--resume` 与 `--overwrite`/`--skip-existing` 互斥。`kill -9` 或机器掉电
无法运行关闭逻辑，不保证最后一个仍打开的 parquet 文件可恢复。

先用一个真实 episode 做 CPU 同构基线：

```bash
env HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_mobile_aloha_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/mobile_aloha_streaming_test/cpu_h264 \
  --dataset-uid mobile_aloha --episode-limit 1 \
  --streaming-encoding --video-codec h264 --video-preset fast
```

在带 NVENC 引擎且容器已映射 GPU/video capability 的机器上，把编码器改为：

```bash
--streaming-encoding --video-codec h264_nvenc --video-preset p4
```

NVENC 路径启动前会真实打开与相机数相同的并发编码会话并编码一帧；不会把“FFmpeg 编译时包含
`h264_nvenc`”误当成硬件可用。A100/A800 没有 NVENC 编码引擎，需使用 L4、A10、RTX 等支持
NVENC 的 GPU。不要用 `--no-video-preflight` 绕过生产转换的探针。

转换后执行源数据对比评估：

```bash
env HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/evaluate_mobile_aloha_conversion.py \
  --partition-root /path/to/mobile_aloha/part-000-mobile-velocity-effort-3cams-50fps \
  --samples 20 --min-psnr-db 30 --output /path/to/evaluation.json
```

报告包含结构/逐路帧数、数值字段最大绝对误差、抽样 PSNR、codec、FPS、文件体积和转换吞吐；
任一结构、数值、帧数或画质门禁失败时命令返回非零。

由于 LeRobot 的单个 dataset 必须使用固定 feature schema，CLI 会自动把真实源数据分成
移动、静态有 effort、
静态无 effort 三个可独立加载的 LeRobot v3.0 子数据集，并写出
`collection_manifest.json`。它不会为静态数据伪造底盘动作，也不会丢弃静态数据独有的
`cam_low`。双臂动作写入 `action`，仅移动分区写入 `action.base`；只转换 RGB，相对父目录
按官方任务映射写入自然语言 LeRobot `task`。FPS 优先读取相机时间戳或 HDF5/sidecar 元数据，
缺失时默认使用该公开数据集的 50 FPS（可用 `--fps` 覆盖）。已有输出默认不会被覆盖。

三个参数都是任意路径，互相之间没有目录结构约定。`--config` 指向的 yaml
文件对应 `common/schema.py::ProcessConfig`——清洗/对齐阈值 + 该数据集的
本体信息（`embodiment_class`/`num_arms`/`dof_per_arm`/`gripper_type`/
`has_mobile_base` 等），全部直接手填在这一份文件里，只有 `id` 是必填
字段，其余都有默认值。

运行时会向 stderr 输出诊断信息：跑之前检查配置里是否有互相矛盾的开关
（如设了 `urdf_path` 却未打开 `fk_check_feasible`），跑完后按 stage 汇总
每类跳过/拒绝原因的出现次数，其中 check1/check2 因服务未配置或调用失败
而未真正执行检查的会标注 `[UNVERIFIED]`，避免和真实检查通过混淆。

## 模块化多格式转换（convert_dataset.py）

MimicGen 的 62 个 robomimic HDF5 容器使用专用集合入口
`convert_mimicgen_to_lerobot.py`：它复用通用 reader/writer/checkpoint，按真实固定 schema
一文件一分区，保留原生 dtype、split 和来源追踪，并支持可验证的 `--resume`。先阅读
[`MIMICGEN_CONVERSION.md`](embodied_datasets/scripts/convert_scripts/MIMICGEN_CONVERSION.md)，
其中包含完整字段映射、真实 smoke 结果、正式后台命令和只读 OSS dry-run 命令。

```bash
HF_HOME=/home/pai/zxw/.cache/huggingface .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_mimicgen_to_lerobot.py \
  --inspect-only --inspect-workers 4
```

DexMimicGen 使用 `convert_dexmimicgen_to_lerobot.py`。9 个官方 HDF5 的固定 schema
彼此不同，因此输出 9 个独立 LeRobot partition 和一个 collection manifest；数值字段保持原
dtype/shape/order，MJCF XML 以 SHA-256 去重 sidecar 保存。正式路径使用已实测加速的 4 个隔离
partition worker 与 CPU H.264，并拒绝本机真实预检失败的 NVENC。`--resume` 使用默认 64 episode
的 part 级 checkpoint；所有最终数据、work/cache、resume、
日志和锁都限制在固定 OSSFS staging 根目录。发布使用 `_INCOMPLETE`/`_SUCCESS`，不复制第二份
集合也不依赖完整目录 rename；容量由显式 staging/inflight 配额控制。完整证据、映射、
storage plan 与命令见
[`DEXMIMICGEN_CONVERSION.md`](embodied_datasets/scripts/convert_scripts/DEXMIMICGEN_CONVERSION.md)。

```bash
PYTHONPATH=embodied_datasets/scripts/convert_scripts .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_dexmimicgen_to_lerobot.py \
  --estimate-storage
```

RoboVerse v2 的发布文件是无图像、无时间戳且跨机器人/schema 异构的轨迹集合，使用专用入口
`convert_roboverse_to_lerobot.py`。它不会伪造相机或 FPS；默认因缺少物理时间基准而拒绝写入，
只有显式选择 `--allow-ordinal-timebase` 或提供有外部依据的 `--fps-override` 才转换。不同固定
schema、机器人和动作/状态对齐方式写为可追踪的独立 LeRobot part，支持校验后 part 级续传。
`--inspect-only` 会全量验证每帧 schema/dtype，并为每个动态 feature 汇总逐分量有限值范围和
NaN/+Inf/-Inf 计数；用 `--inspection-report` 可原子保存完整 JSON。真实 RLBench 对齐 episode、
ManiSkill 稳定混合 dtype action-only episode，以及保留 `[9,1]` 命名单元素数组轴的 LIBERO-90
episode 都已转换并通过独立 raw/Parquet/`LeRobotDataset` 首中末帧逐字段核对。2026-08-18
完整源扫描发现的 52 个截断 CALVIN 文件已从 pinned 官方 revision 恢复，52 文件定向只读
preflight 以 0 blocker 通过；修复后尚未重跑完整源扫描。仍明确阻止转换的是 1,003 个 episode
内 dtype 漂移的 ManiSkill 文件，未启动全量转换或 OSS 写入。
调查证据、字段映射、真实 smoke、风险和正式命令见
[`ROBOVERSE_CONVERSION.md`](embodied_datasets/scripts/convert_scripts/ROBOVERSE_CONVERSION.md)。

```bash
HF_HOME=/home/pai/zxw/.cache/huggingface .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_roboverse_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --source-directory roboverse --dataset-uid roboverse --inspect-only \
  --inspection-report /home/pai/zxw/roboverse_logs/preflight_summary.json
```


`convert_mobile_aloha_to_lerobot.py` 只覆盖 Mobile ALOHA 一种 HDF5 布局。其他
数据集（普通单臂 HDF5、RLDS/TFDS、"文件夹+图片+JSON" 三种格式）统一走
`convert_dataset.py`：按 `configs/<dataset_uid>.yaml` 里的 `format` 字段分发
到 `readers/` 下对应的 reader，再共用同一套写入/校验/发布逻辑
（`convert_core/lerobot_writer.py`）。新增一种格式只需要新写一个
`readers/<format>_reader.py` 并注册进 `readers/registry.py`，不需要改这个
CLI；新增一个数据集通常只需要新写一份 yaml，不需要新写 Python 脚本。

```bash
# 1. 先探测格式（复用上面的 dump_dataset_schema.py）
python3 embodied_datasets/scripts/convert_scripts/dump_dataset_schema.py \
    --dataset-root /data/public_datasets_raw/<dataset_uid>

# 2. 参照 configs/example_hdf5.yaml / example_rlds.yaml / example_raw_image_json.yaml
#    写一份 configs/<dataset_uid>.yaml，字段名/维度按第1步的报告填

# 3. 先 --dry-run 校验，不写任何输出
python3 embodied_datasets/scripts/convert_scripts/convert_dataset.py \
    --config embodied_datasets/scripts/convert_scripts/configs/<dataset_uid>.yaml \
    --raw-root /data/public_datasets_raw --staging-root /data/public_datasets_staging --dry-run

# 4. 确认后正式转换（或用 --configs-dir <dir> --all 批量跑一个目录下的所有 yaml）
python3 embodied_datasets/scripts/convert_scripts/convert_dataset.py \
    --config embodied_datasets/scripts/convert_scripts/configs/<dataset_uid>.yaml \
    --raw-root /data/public_datasets_raw --staging-root /data/public_datasets_staging
```

**部署到服务器前必读**：`embodied_datasets/scripts/convert_scripts/PIPELINE_STATUS.md`
记录了当前哪些格式路径已经端到端真实验证过（hdf5：已验证，含真实视频编码）、
哪些只验证了纯逻辑部分（rlds：本机没装 `tensorflow_datasets`，`tfds.builder_from_directory`
从未真正跑过一次；raw_image_json：约定是本项目自创的，还没拿真实下载的数据集核对过），
以及环境相关的几个坑（lerobot 实际安装版本、ffmpeg 在 conda 环境内外的可见性差异等）。

## 可视化检查工具（inspect_tool）

独立的 Streamlit 应用，对比同一份数据集清洗前后的差异（视频帧、
state/action 各 band 曲线、per-stage 处理记录），不依赖 `run_pipeline.py`
本身跑过：

```bash
streamlit run embodied_datasets/scripts/inspect_tool/app.py -- \
    --input /path/to/raw_lerobot_dataset \
    --output /path/to/output_dataset \
    --config /path/to/process_config.yaml
```

`embodied_datasets/scripts/process_scripts/` 目录下的其他文件：

```
common/                                         # schema（ProcessConfig）/ io（yaml读写）/ vlm_client / sam3_client / service_clients
episode.py / fk_backend.py / lerobot_io.py      # Episode/FkChain/lerobot读写
stage1-5 / check1-3 / unify_representation.py / run_pipeline.py
```

## process_scripts 处理流程

`run_pipeline.py` 按顺序对每个 episode 依次跑 Stage1-5，再跑 Check1-3（下面
"跨本体统一表示"另有独立说明，不在下表内）：

| 模块 | 实现 |
|---|---|
| `stage1_sudden_change` | Savitzky-Golay 平滑（`scipy.signal.savgol_filter`）后计算 residual/加速度/jerk，任一超过阈值的帧标记为异常并线性插值修复；标记帧占比超过 `episode_reject_threshold` 则整条 episode 拒绝 |
| `stage2_trend_alignment` | 对 state/action 的每个公共维度做互相关（`scipy.signal.correlate`）估计帧滞后（lag）和方向一致性；lag 绝对值超过 `max_lag_frames` 或方向一致性低于 `da_threshold` 则跳过该 episode，否则按估计的 lag 对齐并裁剪首尾帧 |
| `stage3_extreme_value` | 两遍处理：先对数据集全部 episode 的每一维算 `quantile_low`/`quantile_high` 分位数界（`gripper_dims_state`/`gripper_dims_action` 声明的夹爪维度豁免），再逐 episode 丢弃超界的帧 |
| `stage4_fk_consistency` | 仅当 `fk_check_feasible=true`（配置文件里直接手填，建议只在 `urdf_path` 已配置且该数据集action space是joint_position/eef_pose时打开）时执行：用 `ikpy`（`fk_backend.FkChain`）对关节角做正向运动学，与数据中报告的末端位置比对；系统性中位数偏移超过 `tcp_offset_tolerance` 时整 episode 做偏移修正，偏移方差过大（非系统性）时只标记待人工复核、不改数据 |
| `stage5_orientation_alignment` | 用配置的 4x4 base-to-world 变换矩阵对每帧末端位置和四元数朝向做坐标变换，统一各数据集的世界坐标系约定；未配置变换矩阵则跳过 |
| `check1_instruction_consistency` | 语言指令一致性检查：通过OpenAI兼容接口调用VLM（默认模型 `qwen2.5-vl-7b-instruct`，API key经 `vlm_api_key_env` 指定的环境变量读取，从不写入配置文件），对语言指令与均匀采样的最多5帧图像做一致性判断，模型强制输出JSON；网络/解析失败时fail-safe为 `consistent=True`（不误伤好数据），`vlm_service_url` 已配置但密钥缺失时直接报错（不静默退化为跳过） |
| `check2_video_state_consistency` | 视频-状态一致性检查：把FK算出的机械臂末端3D位置用相机外参变换、针孔投影成像素坐标，按物理半径 `gripper_radius_m` 换算出像素半径画一个圆盘作为"期望区域"，同时把投影点±半径围成的框、配合 `sam3_text_prompt`（默认`"robot gripper"`，可按数据集改）一起喂给HuggingFace transformers的`Sam3Model`（`LocalSam3Client`，需要 `sam3_model_id`=HF repo id 如`"facebook/sam3"` + `sam3_hf_token_env` 指定的环境变量里的gated-model访问token）拿到"实际分割mask"，两者算IoU；从episode均匀采样最多5帧取平均IoU，跟 `iou_threshold` 比，不达标只标记不丢弃 |
| `check3_video_quality` | 三项质检中唯一已实现的一项：用 OpenCV 检测黑屏（平均亮度低于 `black_threshold`）、模糊（Laplacian方差低于 `blur_threshold`）、连续静止帧（帧间差低于 `still_threshold` 且持续帧数达到 `still_min_consecutive_frames`），命中的帧直接丢弃 |

## 跨本体统一表示

本节定义 `process_scripts` 流水线产出的**最终 LeRobot v3.0 数据集**中，机器人本体
（robot-collected embodiment）`observation.state` 和 `action` 的格式。这是下游
训练/评测脚本读取数据时依赖的公共契约。

### state：128维canonical向量

128 维 canonical 向量固定总维度 128，按下表切片：

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

### action：128维canonical向量

128 维 canonical 向量固定总维度 128，按下表切片：

| 子区间 | 维度 | 内容 | 常量名 |
|---|---|---|---|
| `[0:7]` | 7 | joint delta/绝对值 | `ACTION_JOINT_SLOT` |
| `[7:10]` | 3 | 末端位置delta | `ACTION_EEF_POS_SLOT` |
| `[10:13]` | 3 | 末端旋转delta（axis-angle） | `ACTION_EEF_ROT_SLOT` |
| `[13:34]` | 21 | 夹爪/灵巧手 | `GRIPPER_SLOT` |
| `[34:68]`（仅双臂数据集） | 34 | ARM2，结构与 `[0:34]` 相同 | `ACTION_ARM_BLOCK_DIM` |
| `[68:128]` | 60 | 预留，当前恒为0 | 无命名 |

`unify_representation.apply_action()` 除计算128维向量外，还计算一个128维 bool
mask。该mask 作为独立的 lerobot feature 写入最终数据集：

```
action_canonical_mask   # bool, shape (128,)，每帧写入，整数据集内容相同
```

`mask[i]=False` 表示第 i 维是该本体不具备对应自由度的零填充，而非测量值为0。训练时
应使用该 mask 过滤loss/attention。
