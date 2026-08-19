# convert_scripts 流水线现状

初稿写于 2026-08-13，最近更新于 2026-08-19。本文同时记录本机（Windows，conda 环境
`vla_data_pipline`）和 PAI DSW 服务器
`/home/pai/zxw/vla-data-pipeline/.venv` 上**实际验证过**的内容，以及仍需拿真实数据完成的验证。
服务器环境和 Mobile ALOHA 当前状态以本文 2026-08-17 的记录为准。

## 现在有什么

```
convert_scripts/
├── dump_dataset_schema.py         # 格式探测：hdf5 / rlds / raw_image_json / unknown
├── convert_mobile_aloha_to_lerobot.py  # ALOHA移动/静态异构schema自动分区转换
├── convert_core/                  # 格式无关部分：plan数据类、pydantic配置、LeRobot写入器
│   ├── errors.py, path_utils.py, hdf5_common.py, episode_spec.py, dataset_config.py, lerobot_writer.py
├── readers/                       # 每种源格式一个文件，按格式名分发
│   ├── base.py, hdf5_reader.py, rlds_reader.py, raw_image_json_reader.py, registry.py
├── configs/                       # 每个dataset_uid一份yaml；附带三份带注释的模板
│   ├── example_hdf5.yaml, example_rlds.yaml, example_raw_image_json.yaml
├── convert_dataset.py             # 统一CLI：--config/--all, --dry-run, --overwrite/--skip-existing
└── tests/                         # 全仓库当前 455 passed / 1 skipped
```

`convert_mobile_aloha_to_lerobot.py` 里格式无关的通用 HDF5 逻辑（相机发现、FPS 推断、
数值矩阵校验、RGB 解码）已经提炼进 `convert_core.hdf5_common`，现在以原来的私有名字重新
导入回去。真正属于 ALOHA 专属的部分保留了下来：双臂/移动底盘 14+2 动作布局的歧义处理，
通过 `/base_action` 交叉校验；真实公开数据中的静态 arm-only episode 也会被识别。由于
LeRobot 的 feature schema 在单个 dataset 内必须固定，CLI 会按底盘动作、effort、相机集合
和 FPS 自动分区，并在集合根目录写 `collection_manifest.json`，不使用伪造的零值补齐。
`readers/hdf5_reader.py` 是给*其他所有* HDF5 数据集用的配置驱动 reader——单一无歧义的动作
向量，完全由 `configs/<uid>.yaml` 驱动，不需要新写 Python 文件。

专用转换器已经加入可用于 `nohup` 日志的实时 ETA：预检阶段按 episode 统计，正式转换阶段按
三个分区的全局 frame 数统计；日志包含完成量/总量、百分比、吞吐率、elapsed、ETA 和当前
partition/episode。刷新间隔由 `--eta-interval-seconds` 控制（默认 10 秒），每次
`save_episode()` 后无论是否达到间隔都会强制刷新，因此视频编码和 mux 时间会计入速度估算。
ETA 使用单调时钟并有确定性单元测试覆盖。

## 已验证 vs. 未验证

**ARCap：五个官方 HDF5 分区的 reader、phase-group work units、persistent-worker direct commit、
可恢复 checkpoint、RSS/存储门禁、marker 发布和独立 evaluator 已实现。** 固定 revision 的
1975 episodes / 231923 frames 已完成全量 metadata/schema/episode/phase reference 扫描，点云为
`[10000,6] float64` XYZ 米 + RGB `[0,1]`，无视频；open_bottle 因真实字段缺失保持独立 schema，
没有补零或 cast。最终真实两 work-unit smoke 覆盖 assemble 6 episodes / 779 frames，恢复精确复用
2/2 committed units；Parquet 九帧 540306 个值与 HDF5 bit-exact，真实 LeRobotDataset 重开后展示层
float32 最大误差 `2.98e-8`。四张 A800 的 NVENC 均不可用且本数据无视频；正式配置选择实测
145–147 frames/s、3.08–3.19× speedup 的四个持久 worker。估计全量输出 61,390,182,738 bytes，
正式上限为 staging 80 GiB / inflight 16 GiB，但 OSS 对象配额仍未获授权查询，因此全量正式转换
未启动。增强后的四 work-unit 冷启动 W1/W2/W4 端到端门禁为 11.30/10.81/7.71 frames/s，正确拒绝
把短样本回退称为加速；正式启动前仍须让代表性多单元增强基准再次通过。完整资源、I/O、语义
等价、checksum、清理证据和风险见 `ARCAP_CONVERSION.md`。

### 2026-08-18 GR00T Teleop Sim 恢复转换故障记录

GR00T 全量转换在本地 staging 的 part 级 checkpoint 上恢复时，先后暴露了两个容易被误判为
“卡住”的问题。第一处发生在预检摘要打印完成之后、worker 启动之前：恢复指纹原先对 24000 个
episode 的 Parquet 和视频路径逐一调用 `Path.resolve()`，在只读 `fuse.ossfs2` 源目录上形成约
48000 次远端路径解析且没有进度日志。现已改为纯词法绝对路径，并在指纹计算前后打印明确状态；
生产命令使用的绝对、非符号链接路径得到的指纹字符串不变，因此已有完成 marker 可以继续复用。

第二处发生在 `part-022-posttrainpnpnovelfromtraytotieredbasketsplita`：数据统计和 1000 个 H.264
视频 packet remux 均已完成，随后 Hugging Face `Dataset.from_list()` 在生成
`meta/episodes/chunk-000/file-000.parquet` 时失败，错误为
`arrays to be concatenated must be identically typed, but float and double were encountered`。
`Auto-inserting h264_mp4toannexb bitstream filter` 与移动 MP4 `moov atom` 都是正常的无重编码 remux
日志，不是失败原因。真实触发字段是 `next.done/q99`：episode 577（源 trajectory
`PosttrainPnPNovelFromTrayToTieredbasketSplitA-00190`）长度为 57，统计库返回 `float64`；其余长度
117～637 的 episode 返回 `float32`。此前只统一了视频统计 dtype，没有覆盖 bool/int 等非视频
feature 的边缘 quantile。

修复约定是统一所有 episode stats 的物理类型：`count` 固定为 `int64`，其余
`min/max/mean/std/q*` 固定为 `float64`；写视频前先校验所有 episode 的统计 key、shape 和 dtype，
使将来的 schema 漂移在昂贵的 remux 之前报告具体 feature/stat/episode。回归测试必须包含长度
57 与 117 的 bool 序列、Arrow/Hugging Face rows 构造、全局 stats 聚合，以及原有 GR00T
端到端转换/恢复测试。未写完成 marker 的 part 会在下一次 `--resume` 时安全重建；已有 marker
且重新打开校验通过的 part 不重算。

修复后的验证结果：由 `HEAD` 和本次拟提交文件组成的干净快照中，GR00T 测试为 23 passed；
另外使用 part-022 的真实 1000 条 `episodes.jsonl` 长度重建 `next.done` 统计，确认修复前同时
出现 `float32`/`float64`，修复后 1000 rows 均为 `float64`，Hugging Face Dataset 构造、
schema 校验和 `aggregate_stats()` 全部通过。全工作树回归为 409 passed / 1 skipped / 1 failed；
唯一失败来自尚未跟踪、也不属于本次提交的 1X World Model reader 测试，GR00T 修复的干净提交
快照不含该文件。

**RoboVerse v2：专用异构集合 reader/converter、三个真实 smoke、独立 evaluator 和 part 级 checkpoint 已实现。**
官方 `RoboVerseOrg/roboverse_data` release 固定到 revision
`fab63ccaaed54f413901f86edc3fa1ab77a96500`。下载内容是跨 suite/robot/schema 的
pickle/JSON 轨迹集合，没有 trajectory 图像、视频、时间戳或权威 FPS；转换器不补相机，且默认
拒绝发明物理时间。相等长度 action/state 按原 index 对齐；长度不等时拆成通过 source episode
UID 关联的完整 action-only/state-only part，不裁剪或填充。全量检查会统计每个固定 schema
动态 feature 的逐分量有限 min/max、NaN/+Inf/-Inf、action/state 总帧数、各 suite 长度范围、
对齐类型、robot/task/split，并可用 `--inspection-report` 原子写出完整 JSON。真实 RLBench
`franka` smoke 已完成 1 episode / 144 frames / 8 vector features；另一个真实 ManiSkill
`draw_triangle` smoke 完成 1 action-only episode / 249 frames，证明七个 float32 arm 和两个 int64
finger component 的稳定混合 dtype 可无 cast 拆为命名 scalar feature；LIBERO-90 smoke 完成
1 aligned episode / 197 frames / 15 features，并原样保留命名 component 的 `[9,1]` 与 `[3,1]`
单元素数组轴。独立 evaluator 不调用 converter reader，分别重开 raw、Parquet 与真实
`LeRobotDataset`，逐字段核对 0/72/143、0/124/248 和 0/98/196，并重新计算完整 episode
stats；三个 v9 JSON 报告均通过。LeRobot 0.6.0 的标准
PyTorch transform 会把声明/存储为 float64 的 Python float list 物化为 torch.float32；RLBench
抽样值仍逐值完全相等；LIBERO-90 的一般 float64 数值则与该运行时 float32 物化精确一致。
meta/Parquet dtype 和 byte-level 数值保持准确，此 consumer 行为已在报告
中明确记录。实际 task_index 映射和零视频也已核对。恢复测试覆盖已验证 part 复用、损坏重建、源/config 指纹变化
拒绝、并发锁和最终 checkpoint/cache 清理。完整证据、映射、命令和仍需用户决定的时间策略见
`ROBOVERSE_CONVERSION.md`。2026-08-18 全量只读 preflight 已完整扫描 10622 个候选源文件：
83959 个有效源 episode 可形成 11479 个固定 schema part、123549 个关联输出 episode、
12456493 帧；v9 扫描耗时 01:30:07，统计 10245438 个 action frame、5274419 个 state frame、
710 项 feature inventory，NaN/+Inf/-Inf 总数均为 0。该修复前扫描明确聚合出 1055 个硬阻塞文件：52 个本地为 0 字节但 pinned 官方对象非空的
CALVIN pickle，以及 1003 个 episode 内命名 finger target dtype 改变的官方 ManiSkill 文件。2026-08-19 已从 pinned revision 下载、
校验并逐文件原子替换全部 52 个 CALVIN 文件；定向只读 preflight 通过 52 files / 52 episodes /
17145 frames / 0 blockers。converter v10 新增显式 `--allow-lossless-dtype-promotion`：仅允许同一
component 的整数/布尔帧精确提升到该 component 已存在的唯一浮点 dtype，逐值执行 cast/restore
相等性检查，保持 shape、顺序、帧数和 episode 边界，并在 part/collection/source episode
provenance 中记录原始 dtype runs；不可精确表示的值仍阻塞。默认行为不变，仍拒绝漂移。2026-08-19
已用该 opt-in 完成整个 ManiSkill suite 的只读
preflight：1014 files / 8171 episodes / 1448392 action frames / 1015 output parts / 5432 个
component-episode promotion records / 0 source issues，覆盖全部 1003 个原阻塞文件；证据位于
`/home/pai/zxw/roboverse_logs/maniskill_dtype_promotion_preflight.{json,log}`。聚焦 reader/converter
测试 48 passed，并验证不可精确表示的 `16777217 -> float32` 仍被拒绝；
修复后的 v10 全量只读 preflight 随后完整扫描 10622 files 并以 0 退出：86727 source episodes、
10746462 action frames、5291564 state frames、12534 parts、126317 output episodes、12957517
output frames、5432 promotion records、711 feature inventory rows、0 blockers、0 NaN/+Inf/-Inf。
仅余 20 个官方 BiDex 零字节静态 sidecar，均为可显式确认的非阻塞问题。完整 JSON 报告和进度位于
`/home/pai/zxw/roboverse_logs/preflight_v10_post_repair.json` 与
`/home/pai/zxw/roboverse_logs/preflight_v10_post_repair.log`。全量正式转换未启动，也没有写 OSS。最终测试计数见
`ROBOVERSE_CONVERSION.md` 的 verification 记录。

**MimicGen：专用集合转换、真实 smoke 和可恢复 checkpoint 已实现。** 服务器下载目录实际拼成了
`public_datasets_raw/minicgen`，不是请求中的 `mimicgen`；内容已核对为官方
`amandlek/mimicgen_datasets` release（139 GB、62 个 robomimic HDF5：source 12、core 26、
object 2、robot 16、large_interpolation 6）。`readers/robomimic_hdf5_reader.py` 会遍历
`data/demo_N`，保留每个 leaf 的原 dtype/shape/value 和 mask split；
`convert_mimicgen_to_lerobot.py` 按一个源 HDF5 一个固定 schema 分区，使用
`convert_core/checkpoint.py` 的 fingerprint/原子 JSON/非阻塞锁，并在分区完成后重新打开和
FFprobe 视频才写 marker。真实 `core/square_d0` 1 episode / 136 帧 CPU H.264 smoke 已通过：
LeRobot 重开成功，Parquet 首/中/末帧 45 个数值样本逐值完全相等，6 个解码图像最低 PSNR
34.99 dB；报告在 `/home/pai/zxw/mimicgen_logs/smoke_square_1ep_evaluation.json`。新增测试 12 项
通过；最终当前工作区全仓库测试为 380 passed / 1 skipped。全量正式转换未启动、未写 OSS；
完整语义映射、resume 约定和命令见 `MIMICGEN_CONVERSION.md`。
源数据有一个已显式保留的 metadata 异常：`source/square.hdf5` 只有 10 个 demo，但八个旧
`mask/*` 列表含有不存在的 `demo_10`～`demo_199`；转换器不伪造 episode，在 manifest 原样
记录 mask 并列出 dangling references。其余 11 个 source 文件的 mask 引用均有效。
2026-08-18 的 4-worker 全量只读 preflight 已通过 62 partitions / 50120 episodes /
14875672 frames，日志为 `/home/pai/zxw/mimicgen_logs/full_preflight.log`。

**DexMimicGen：9 分区 reader/coordinator、配额门禁、集合 manifest、XML sidecar、part 级 checkpoint
与独立 evaluator 已实现，并通过真实 OSSFS smoke。** 官方 9 个 HDF5 共 9178 episodes / 2915177 frames，
每个容器内部 schema 固定但彼此不同，因此按源 HDF5 分区，不补零、不 cast、不做 128 维映射。
reader 全量核对 episode ID、schema、所有 leaf 第一维、`num_samples`、`data.attrs.total`、相机和
每 episode MJCF 引用；`actions` 顺序由 `action_dict` 重构抽样验证，state/joint/gripper 名称从
精确 XML 解析。XML 以 SHA-256 去重后保存为 zlib-9 sidecar。普通 MP4 在 OSSFS 上因 seek-back/
ftruncate 不可用，最终写入路径改为 fragmented MP4 顺序写；CPU H.264 使用 CRF 18、`fast`、
`yuv420p`、`tune=zerolatency`，首帧 PTS 为 0。本机四张 A800 的 NVENC 均真实失败，因此正式路径
固定为隔离 partition worker + CPU H.264。

2026-08-19 直接在固定 OSSFS staging 根完成 `threading/demo_0` 真实 smoke：1 episode / 196 frames /
3 cameras；独立 evaluator 从 Parquet 读取，35 个数值字段的 dtype/shape/value 逐项精确一致，
episode/frame/index/task 和 MJCF sidecar 均通过。三路视频均为 H.264/yuv420p/20 FPS/196 帧，最低
PSNR 36.94 dB。两 episode、每 part 一 episode 的 resume smoke 也通过：377 frames，每相机两个
video chunk；有效 `_SUCCESS` 默认拒绝覆盖，`--skip-existing` 验证后接受。最终 OSSFS-compatible
`threading/demo_0` 1/2/4 worker 基准分别为 14.9405/17.0237/17.2403 秒，对应 aggregate throughput
13.1187/23.0267/45.4748 frame/s（每 worker 重复一个 196-frame 真实 episode）。focused 测试
2026-08-19 最终审计又完成一次真实两 episode 受控中断与恢复：第一个 part 的 196 帧被精确复用，
最终 377 帧；增强 evaluator 证明 35/35 数值和 3/3 图像字段覆盖、partition/collection manifest、
`_SUCCESS` checksum 与 MJCF 引用均通过，最低 PSNR 36.80 dB。真实容量和锁冲突均在创建目标数据集前
非零退出。最终相关套件 60 passed，语法编译与 `git diff --check` 通过。原始
`two_arm_threading.hdf5` 的 size、mtime、SHA-256 在测试前后不变。全量正式转换仍未启动。
详见 `DEXMIMICGEN_CONVERSION.md`。

**hdf5（readers/hdf5_reader.py + convert_dataset.py）：端到端全链路已验证**，包括真实的
视频编码。`tests/test_convert_dataset.py` 用一份合成 HDF5 fixture 跑通完整流水线——
`LeRobotDataset.create` -> `add_frame` -> `save_episode`（真实 ffmpeg/SVT-AV1 视频编码）
-> `finalize` -> 重新打开并校验 -> 原子发布——最后还用一个真实的 `LeRobotDataset` 把结果
读回来验证。Mobile ALOHA 专用转换器现在也有混合移动/静态集合的真实编码、重新打开和集合
清单测试。

**Mobile ALOHA：真实源目录已完成只读全量预检。** 2026-08-17 对
`/mnt/data/embodied_datasets/public_datasets_raw/mobile_aloha` 的 1103 episodes / 971850 帧
运行专用 CLI `--inspect-only`，全部通过并得到三个分区：静态无 effort 289 episodes、静态有
effort 536 episodes、移动 278 episodes；三者均为 50 FPS。截至本文最近更新时，正式转换
已用原始非流式 AV1 路径启动；最近一次记录为 29300/971850 帧（3.0%，11.14 frames/s），
进程仍在运行，最终目录尚未原子发布，因此仍不能视为可验收输出。以
`/home/pai/zxw/mobile_aloha_logs/convert.log` 末尾和 PID 存活状态为准。

**流式编码小样本已用真实 Mobile ALOHA 验证。** 取
`aloha_mobile_cabinet/episode_0.hdf5`（1500 帧、3 路 640x480、50 FPS）运行 CPU H.264
流式编码，完成写入、重开和逐视频帧数校验，用时 20.51 秒、73.12 frames/s；三个视频均为
1500 帧/50 FPS，总数据集约 9.25 MiB。对源 HDF5 均匀抽样 20 帧评估：state/action/base/
velocity/effort 最大绝对误差全为 0，三路 PSNR 分别为 38.13/39.89/38.92 dB，30 dB 门禁通过。
同一 episode、同一 H.264 参数的非流式 PNG 中转对照用时 83.44 秒、17.98 frames/s；流式路径
实测加速 4.07 倍，且三路 MP4 字节数和抽样画质结果完全相同。机器可读报告分别位于
`/home/pai/zxw/mobile_aloha_streaming_test/cpu_h264/evaluation.json` 和
`/home/pai/zxw/mobile_aloha_streaming_test/baseline_h264/evaluation.json`。

NVENC 路径已经实现真实并发 session preflight、最长 episode + 1 的无丢帧队列下限，以及输出
MP4 frame/codec/FPS 校验。当前宿主机内核能看到 4 张 A800 80GB，但执行容器没有
`/dev/nvidia*`，而且 A800/A100 系列本身没有 NVENC 编码引擎；同一 1500 帧样本的
`h264_nvenc` 测试因此在写任何输出前按预期失败。真实 NVENC 吞吐和画质仍需在 L4/A10/RTX
等带 NVENC、且向容器开放 video capability 的实例上跑同一命令和评估工具，不能用软件 H.264
结果冒充。

正式转换不要直接把 `--staging-root` 指向 `/mnt/data`。该路径是 `ossfs2` FUSE 挂载，历史
实测不支持 MP4 muxer 关闭文件时需要的 seek-back/ftruncate；LeRobot 0.6.0 的流式编码临时
MP4 也建在 dataset root 下。服务器本地 overlay 盘约有 2 TB 可用空间，因此当前约定是先写：

```text
/home/pai/zxw/mobile_aloha_staging/lerobot_v3_0/mobile_aloha
```

脚本会先在同一父目录写 `.mobile_aloha.incomplete-<uuid>`，三个分区全部写完并逐一重新打开
验证后，才原子发布为 `mobile_aloha`。验证内容包括分区的 episode/frame 数、feature、FPS、
task 和逐 episode 长度；集合根目录另写 `collection_manifest.json`，每个分区写
`conversion_manifest.json`。只有进程返回 0，且日志末尾出现
`wrote 3 partitions / 1103 episodes / 971850 frames` 和 `completed: converted=1, skipped=0`，
才能视为本地结构转换完成。之后再用 `rsync` 顺序复制到 OSS，并做只读 checksum 对比。

**raw_image_json（readers/raw_image_json_reader.py）：reader 内部逻辑已充分验证，但文件
约定本身还没拿真实数据核对过。** 本仓库里没有现成的 raw_image_json 数据集可供核对——
`readers/raw_image_json_reader.py` 文档字符串和 `configs/example_raw_image_json.yaml` 里
"每个episode一个文件夹 + `metadata.json` + `frames[]`" 的结构，是这次实现自己发明的约定，
参照的是常见小型开源机器人学习数据集的大致样子。**在转换真实的 raw_image_json 数据集之前**，
先探测它的真实目录结构（如果需要可以进一步扩展 `dump_dataset_schema.py` 的 raw_image_json
探测器），把约定改成匹配真实数据的样子——不要反过来把真实数据硬凑成这个猜测的形状。

**rlds（readers/rlds_reader.py）：只验证了格式无关的辅助逻辑部分。** Windows conda 环境
和服务器项目 `.venv` 都没有安装 `tensorflow`/`tensorflow_datasets`，所以
`tests/test_rlds_reader.py` 只用普通
dict/numpy 数组覆盖了 `_get_by_path`/`_resolve_instruction`/`_validate_vector_step`/
`_camera_shape`/`_materialize_steps`，外加确认 `RldsReader.build_plan` 在库缺失时会抛出
清晰的 `RuntimeError`（这是真实执行过的断言，不是猜测）。真正的
`tfds.builder_from_directory(...).as_dataset(...)` 调用从没跑过一次。**在服务器上拿真实
RLDS 数据集验证之前，需要：**

1. 先解决 TensorFlow 版本策略：文档原先建议的 `tensorflow-cpu==2.15.0` 没有 Python 3.12
   wheel，而本项目要求 Python >=3.12；不要直接装进当前 `.venv`。可选方案是为 DROID/RLDS
   建 Python 3.11 独立环境，或先验证一个支持 Python 3.12 且与项目 NumPy/LeRobot 约束兼容
   的新版 TensorFlow，再把版本明确写入 optional requirements。
2. 安装并验证 `tensorflow_datasets`（原计划版本为 4.9.9）。
3. 先跑一遍 `dump_dataset_schema.py --dataset-root <path>` 拿到真实的带类型的 feature
   规格，据此填写 `configs/<uid>.yaml` 里 `vector_fields`/`cameras` 的 source_key 路径。
4. `convert_dataset.py --config ... --dry-run`，仔细看打印出来的 plan，再考虑真正跑一次。
5. 一个已知但还没修的扩展性局限：`RldsReader.build_plan` 会把请求的整个 split 一次性
   解码完，并把每个 episode 的 steps 全部缓存在内存里（见该模块的文档字符串），因为 RLDS
   不解码就拿不到每个 episode 的帧数。数据集不大不小时没问题；如果要处理几百 GB 的
   shard，就需要把 `iter_frames` 改成为每个 episode 重新打开一个独立的子迭代器，而不是
   重放缓存的 steps。

## 值得记住的环境相关发现

- **服务器项目环境已经建好并通过测试。** 2026-08-17 在项目根目录创建 Python 3.12.12 的
  `.venv`，按 `requirements.txt` 安装完成；关键版本为 `lerobot==0.6.0`、`numpy==2.2.6`、
  `h5py==3.12.1`、`opencv-python-headless==4.12.0.88`、`torch==2.11.0+cu130`、
  `transformers==5.14.1`。`pip check` 报告 `No broken requirements found`，环境约 5.8 GB。
  服务器 `/usr/bin/ffmpeg` 为 6.1.1，包含 libx264 和 libsvtav1 编码器。
- **服务器当前全仓库测试结果是 455 passed / 1 skipped / 4 warnings。** 第一次直接运行时有 46 项失败，全部源于受管环境中
  `/root/.cache/huggingface` 只读；把 `HF_HOME`/`HF_DATASETS_CACHE` 指到 `/tmp` 或
  `/home/pai/zxw/.cache/huggingface` 后重跑，全部通过。运行转换和测试前
  应显式设置一个可写的 `HF_HOME`，不要把首次失败误判成 LeRobot 或数据格式问题。
- **其他现成环境不能替代项目 `.venv`。** `/root/lerobot_v30_env` 已不存在；
  `/root/robotwin_env` 的包元数据是 LeRobot 0.3.3，缺少本项目写入器需要的 `finalize()`，也
  没有 TensorFlow/TFDS/pytest；`/mnt/data/embodied_datasets/.venvs/hf_download` 没有
  `bin/python`，不是完整虚拟环境。
- **lerobot 版本不一致的问题已通过项目独立环境解决。**
  `process_scripts/lerobot_io.py` 的文档字符串写的是针对 `lerobot==0.4.4` 验证过，但
  `requirements.txt` 锁定的是 `lerobot[dataset]==0.6.0`，`vla_data_pipline` 环境里实际
  装的也是这个版本，服务器项目 `.venv` 现在同样是 0.6.0。已经直接对照安装源码检查过
  `LeRobotDataset.create`/`add_frame`/`save_episode`/`finalize` 的签名——和本仓库里
  写入器假设一致。不要改用 `/root/robotwin_env` 运行本项目。
- **ffmpeg 在 Windows conda 环境里能找到，但不在裸机 Windows 的 PATH 里。**
  `shutil.which("ffmpeg")` 在 conda 环境外什么都找不到，但激活环境后
  `C:\...\envs\vla_data_pipline\Library\bin\ffmpeg.EXE` 确实存在且能正常工作（是 conda
  间接带进来的，不是通过 `requirements.txt`——正如 `requirements.txt` 里的注释所说，它
  确实是一个系统依赖，不能靠 pip 装）。在 Ubuntu/Debian 服务器上（该注释里提到的实际
  部署目标），跑任何视频编码路径之前先确认已经 `apt install ffmpeg`。
- **当前项目环境没有 `tensorflow_datasets`。** 只有 `dump_dataset_schema.py` 的软导入
  fallback 和缺依赖时报错路径实际跑过。详见上面的 RLDS 一节。
- **过小的合成视频帧会让编码器崩溃。** LeRobot 默认的视频编码器（SVT-AV1）在帧尺寸很小
  时内部会除零——一个 4x6px 的测试 fixture 直接让整个 Python 进程崩溃，报的是
  `encode_video_frames` 工作线程里的原生 `Windows fatal exception: int divide by zero`。
  已确认 64x64 可以正常工作。以后如果再写跑真实写入路径的测试，尺寸不要低于这个值。这也
  几乎可以肯定就是为什么 `convert_mobile_aloha_to_lerobot.py` 自己原有的测试套件
  （用的是 8x10 的fixture）从来没有调用过真实写入路径——一调用就会撞上同样的崩溃。

## 搭建过程中发现并修复的 bug（记录下来防止回归）

- `convert_dataset.py` 最初直接把 `reader.iter_frames`（一个接受 `(plan, episode)` 两个
  参数的 bound method）当作 `IterFrames` 回调传给
  `convert_core.lerobot_writer.write_dataset`，但写入器内部是按 `iter_frames(episode)`
  （只传1个参数）来调用的——`tests/test_convert_dataset.py` 当场抓到一个 `TypeError`。
  修复方式是包一层：`lambda episode: reader.iter_frames(plan, episode)`。
- `readers/raw_image_json_reader.py` 最初无条件要求必须提供 `config.fps`，但设计初衷
  （以及该模块自己的文档字符串）是每个 episode 自己的 `metadata.json["fps"]` 应该单独
  就够用。修复为：只有当某个具体 episode 的 metadata 里两者都没有时才报错要求兜底值。
  这是被 `tests/test_raw_image_json_reader.py` 里报错信息不对的失败用例抓到的。

## 快速用法参考

```bash
# 1. 先探测一份陌生的下载数据集——开销很小、只读，绝不触碰像素/张量数据本身。
python3 dump_dataset_schema.py --dataset-root /data/public_datasets_raw/<uid>

# 2. 参照三份 example_*.yaml 模板之一写一份 configs/<uid>.yaml，
#    字段名/维度按第1步报告里的真实取值填。

# 3. 先校验，不写任何东西。
python3 convert_dataset.py --config configs/<uid>.yaml \
    --raw-root /data/public_datasets_raw --staging-root /data/public_datasets_staging --dry-run

# 4. 确认后正式转换。
python3 convert_dataset.py --config configs/<uid>.yaml \
    --raw-root /data/public_datasets_raw --staging-root /data/public_datasets_staging

# 或者批量跑某个目录下的所有配置：
python3 convert_dataset.py --configs-dir configs/ --all \
    --raw-root /data/public_datasets_raw --staging-root /data/public_datasets_staging
```

Mobile ALOHA 继续直接用 `convert_mobile_aloha_to_lerobot.py`。CLI 会自动把移动和静态
co-training 数据按固定 schema 分区；它不需要 `configs/*.yaml`，因为动作布局歧义和分区规则
属于 ALOHA 领域逻辑。

服务器正式转换命令（从项目根目录执行）：

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

该命令启用 episode 边界断点续传。`Ctrl-C`/`SIGTERM` 后原样重跑命令即可；转换器会校验源文件
指纹和编码参数并跳过已持久化 episode。不要用 `kill -9` 停止，因为它无法关闭当前 parquet
footer，最后一个打开文件不保证可恢复。

实时观察 ETA 和检查后台进程：

```bash
tail -f /home/pai/zxw/mobile_aloha_logs/convert.log
ps -fp "$(cat /home/pai/zxw/mobile_aloha_logs/convert.pid)"
```

完成后除了依赖转换器内置的
重新打开校验，还应按 `collection_manifest.json` 再次独立加载三个分区，核对汇总值为
1103 episodes / 971850 frames，并抽样解码每个分区的首、中、末帧。需要最高强度检查时，对
所有 MP4 运行 `ffmpeg -v error -i <file> -f null -` 做完整 bitstream 解码。

本地验证通过后再发布到 OSS：

```bash
rsync -r --info=progress2 \
    /home/pai/zxw/mobile_aloha_staging/lerobot_v3_0/mobile_aloha/ \
    /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/mobile_aloha/

# 只读 checksum 复核；无输出表示源和目标中同名文件内容一致。
rsync -rcn --itemize-changes \
    /home/pai/zxw/mobile_aloha_staging/lerobot_v3_0/mobile_aloha/ \
    /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/mobile_aloha/
```

## 待办事项

- 启动 Mobile ALOHA 正式转换；完成内置校验、独立 manifest/LeRobot 加载校验和视频抽样解码
  后，再同步到 OSS。同步完成后保存 checksum 复核结果，并把本节的“尚未启动”更新为实际
  完成时间、三个分区的帧数、输出总字节数及验证结论。
- 目前还没有明确指定具体的 raw_image_json 或 rlds 数据集。一旦确定了具体已下载的
  dataset_uid，就对每一个先跑一遍 `dump_dataset_schema.py`，写好它的
  `configs/<uid>.yaml`，把每种格式的第一次真实转换当作这份文档一直推迟的那个"实际验证"
  来对待。
- 还没有 zarr/ROS-bag 的 reader。没做是因为目前已下载的数据集里还没有哪个被认定需要
  这种格式——如果/一旦需要，照着 `raw_image_json` 当初被加进来的方式，加一个
  `readers/<format>_reader.py` 并注册进 `READER_REGISTRY` 即可。
- `DatasetConversionConfig` 是一个所有三种格式共用的扁平 pydantic 模型；某个 reader
  会安静地忽略掉属于别的格式的字段，而不会报警告。如果这种"安静忽略"以后真的在实践中
  掩盖过一次真实的拼写错误，再回头处理这个问题。
