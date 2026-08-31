# VLA Data Pipeline 项目建设复盘与问题解决手册

> 更新日期：2026-08-31
>
> 代码基线：`856fc00`（分支 `Xuanwei-Zhang`）
>
> 适用范围：仓库初始化、数据集接入、LeRobot v3.0 转换、清洗/对齐、可视化检查、TB 级运行与故障恢复

## 1. 文档目的与证据边界

本文把项目从“数据集登记表和单机脚本”演进为“可验证、可恢复、可扩展的数据流水线”期间遇到的主要问题、根因、解决方法和复用经验整理到一个入口中。它不是某个数据集的运行说明，也不替代各数据集的专项转换文档。

内容来自以下可交叉验证的材料：

- 当前仓库代码、测试、README 和专项转换文档；
- Git 提交历史，时间跨度为 2026-07-21 至 2026-08-31；
- 与 Mobile ALOHA、GR00T、MimicGen、DexMimicGen、1X World Model、RoboVerse、AgiBot World 等任务有关的历史会话与运行记录；
- 已保存的 preflight、smoke、benchmark、evaluation 和 checkpoint 结果。

本文采用三种状态，避免把计划或局部测试写成完成事实：

- **已验证**：有代码测试、真实样本、独立评估器或产物重开结果支撑；
- **历史状态**：某次运行当时成立，但可能已被后续实现或运行状态取代；
- **待确认**：缺少上游语义、真实数据或完整运行证据，不作推断。

所有历史测试数量都只是对应提交时的快照。2026-08-31 对最新转换相关测试的抽查结果是 `71 passed, 1 failed`；唯一失败是上传失败测试仍期待异常立即抛出，而实现已加入自动重试并最终成功。它应作为测试契约待对齐项保留，不能用旧的“全绿”记录覆盖。

## 2. 项目的目标、边界与最终架构

### 2.1 两条职责明确的流水线

项目最终形成两条相互衔接、但职责不同的流水线：

1. **格式转换（convert）**：把 HDF5、旧版 LeRobot、TFRecord、token 化视频等源格式转换成 LeRobot v3.0。原则是保留原始语义、dtype、shape、episode 边界和媒体内容，只做目标格式必需的变换。
2. **清洗与对齐（process）**：只接收已经符合 LeRobot v3.0 的数据，执行异常帧修复、时序对齐、极值过滤、FK 一致性、坐标变换、视频/语言检查和跨本体 128 维表示。

早期把登记、转换、校验和清洗混在一起，导致职责边界模糊。最终采用“先忠实转换，再显式清洗”的分层方式：转换层不能偷偷归一化、重采样、补零或套用 canonical 表示；处理层也不负责猜测源格式。

### 2.2 三层数据区

```text
public_datasets_raw
        │  只读：检查、解码、转换，不原地修订
        ▼
public_datasets_staging/lerobot_v3_0
        │  标准格式：可重开、可审计、有 manifest/_SUCCESS
        ▼
public_datasets
           清洗、对齐、质检后的最终数据
```

原始区只读是贯穿整个项目的硬约束。发现下载损坏时，修复工具也采用“临时文件下载 → size/SHA-256 验证 → 同目录原子替换”，而不是边下载边覆盖原对象。

### 2.3 当前转换主链路

```mermaid
flowchart LR
    A[固定上游 revision 与来源] --> B[只读 schema/inventory preflight]
    B --> C[建立字段与语义契约]
    C --> D[按兼容 schema 分区]
    D --> E[冻结 plan 与 fingerprint]
    E --> F[有界 work unit 转换]
    F --> G[本地重开与校验]
    G --> H[复制到目标存储]
    H --> I[远端格式/大小/抽样校验]
    I --> J[原子写 unit/task marker]
    J --> K[删除本地 bulk]
    K --> L[聚合 metadata/manifest]
    L --> M[最后写入 _SUCCESS]
```

相应职责已沉淀到：

- `readers/`：源格式发现、schema 和逐帧读取；
- `convert_core/checkpoint.py`：恢复状态、fingerprint 和原子 JSON；
- `convert_core/parallel.py`：有界并行、失败停止派发；
- `convert_core/storage.py`：容量估算、预留和运行期门禁；
- `convert_core/lerobot_writer.py`：LeRobot 写入、统计和写后验证；
- `convert_core/direct_commit.py`：跨文件系统复制、远端验证和提交；
- `convert_core/runtime_layout.py`、`staging.py`：本地 work/cache/temp 与目标路径隔离；
- 专项 converter/evaluator：处理数据集特有语义，并提供独立验收。

## 3. 建设过程回顾

| 阶段 | 时间 | 当时的核心问题 | 形成的工程决策 | 代表提交 |
|---|---|---|---|---|
| 登记与盘点 | 07-21 | 数据集信息散落、字段口径不统一 | 建立 schema、YAML I/O、Excel 迁移与数据根目录扫描；多轮核验后将集合收敛为确认的 66 项 | `b72010c`～`c5d7ff1` |
| 通用清洗 | 07-21～07-30 | 处理步骤耦合、跨本体维度不统一 | 拆分 Stage1–5、Check1–3；定义 state/action 128 维表示及 mask | `00fea47`、`953b4fc`、`b0b8c61`、`e86cb95` |
| 工程结构收敛 | 07-23～07-28 | 目录层级、命名和环境重复演进 | 合并共享模块与虚拟环境；统一 `lerobot_v3_0` 路径；README 作为入口 | `5ca92c8`、`0dc7325`、`2178f5a` |
| 可视化检查 | 07-31～08-06 | 纯日志无法解释哪些帧被处理或丢弃 | 增加 sidecar、后台执行、前后对比图、Range 视频服务和同步播放器 | `c706b0b`～`ecabe72` |
| 模块化转换 | 08-14 | 每接一种格式都复制完整脚本 | 建立 reader registry、配置驱动 plan、统一 writer/schema dump | `642db54` |
| 真实数据校正 | 08-17～08-18 | 合成 fixture 无法暴露真实 schema、媒体和语义差异 | 对 Mobile ALOHA、GR00T、MimicGen、1X 做全量只读检查和真实 smoke | `1b5415a`～`1d7fd8b` |
| TB 级运行 | 08-18～08-21 | OSSFS、容量、恢复、并发和长任务可靠性不足 | work unit、背压、marker、fragmented MP4、本地 runtime、可迁移恢复 | `f581430`～`0386d01` |
| 多源扩展 | 08-19～08-31 | 更多数据集包含损坏文件、dtype 漂移和多 schema | 加入专用 reader/converter/evaluator，并持续复用通用事务协议 | `a83e4bc`、`264a2f9`、`856fc00` |

这段演进中最重要的变化不是“支持的数据集数量增加”，而是成功条件从“脚本能跑”升级为“输入契约清楚、过程可恢复、产物可独立验证、失败不会污染最终目录”。

## 4. 共性问题、根因与解决方案

### 4.1 问题：需求边界不断漂移

早期仓库同时承担数据登记、下载、格式转换、完整性校验、清洗和训练前表示统一，目录和接口因此反复调整。典型表现包括：

- `convert_scripts`、`registry`、`shared` 和 `process_scripts` 多次移动或合并；
- `lerobot_v2_1` 与 `lerobot_v3_0` 命名不一致；
- registry 中存在“收集到但尚未确认”的数据集；
- 转换逻辑可能误带入清洗、归一化或 canonical padding。

解决方案：

- 将仓库主目标明确为“LeRobot v3.0 转换 + 后续清洗/对齐”，把下载与原始完整性修复视为外部准备步骤；
- 将转换与处理拆成两个目录和两套明确契约；
- 删除未被代码实际消费的 registry 追踪层，避免双重真相源；
- 每份专项文档开头都声明转换“不做什么”；
- 所有有损或语义性变换必须在字段映射表中逐项列出。

经验：边界声明不是文档装饰。它决定一个“看似方便”的补零、重采样或字段改名到底是合法转换，还是未经授权的数据加工。

### 4.2 问题：把字段名当成语义证据

机器人数据中同名字段可能代表当前位置、下一状态目标、增量动作或归一化控制量；`quat` 也不能自动说明是 XYZW 还是 WXYZ。仅凭文件名和数组 shape 推断会生成结构正确但语义错误的数据。

解决方案形成了“证据优先级”：

1. 固定 revision 的官方 loader、builder 或控制器代码；
2. 官方数据卡、论文、README；
3. 本地全量 metadata/schema 检查；
4. 首/中/末帧及跨帧关系验证；
5. 仍不明确的单位、FPS 或语义显式标为 unknown，并要求调用者输入，绝不猜测。

例如 ARCap 通过官方 builder 的 `gap=3` 和全量相邻状态关系确认 10 Hz 及 following-state target；RoboOmni 没有 FPS/timestamp 证据，因此要求 `--fps` 显式给出；AgiBot 深度数据缺少物理单位和量化参数，所以只保留原媒体与 depth 标记。

### 4.3 问题：单一 LeRobot 数据集无法容纳异构 schema

多个源集合内部存在机器人、相机、动作维度或 dtype 差异。强行合并有三种坏结果：补伪造的零、丢字段、把不同含义塞进同一列。

解决方案：

- 以有序字段名、递归 Arrow 类型、shape、dtype、相机集合和必要元数据计算 schema fingerprint；
- 按固定 schema 拆为多个标准 LeRobot v3.0 partition；
- partition 内索引连续，collection manifest 记录全局范围和来源映射；
- mask 用来表达 canonical 表示中的“本体没有该自由度”，而不是在原始转换阶段伪造数据；
- 忽略不影响逻辑 schema 的噪声，如每 episode 变化的 RangeIndex `stop` 或 NumPy Unicode 固定宽度。

已遇到的实例包括：Mobile ALOHA 的移动/静态与 effort 差异、RoboCOIN 的 7 种 schema、RoboVerse 的 action-only/state-only 分裂、AgiBot 的真实/仿真/深度 patch 分区、ARCap `open_bottle` 缺少 `actions2`/EEF 字段。

### 4.4 问题：声明 schema 与物理 payload 不一致

真实数据会出现“`info.json` 声明 float32，但 Parquet 实际含 float64”“schema 声明 `[1]`，reader 返回 scalar”等情况。仅校验 metadata 会漏掉这些错误。

解决方案：

- preflight 同时读取 sidecar、Parquet footer 和代表性 payload；
- writer 写入前严格检查 Python/NumPy 类型、dtype 和 shape；
- 写后检查 Parquet 物理 schema，而不只看 `info.json`；
- 对允许的 dtype 提升采用显式 opt-in 策略，并做 cast/restore 精确相等检查；
- manifest 记录所有观察到的物理 schema 变体和处理策略；
- 禁止有损的 float64→float32 静默缩窄。

两个关键案例：

- 1X v1.1 的 closure 字段声明为 `[1] float32`，reader 却返回 `numpy.float32` 标量；修复为长度 1 的 ndarray，并让 evaluator 同时比较 shape、dtype 和数值。
- RoboVerse 的 ManiSkill finger target 在 episode 内发生整数/布尔到 float 的 dtype 漂移；新增 `--allow-lossless-dtype-promotion`，只有逐值可逆时才提升，不可精确表示的 `16777217 -> float32` 仍拒绝。

### 4.5 问题：统计值的 dtype 在边界长度下漂移

GR00T 的一个分区完成大量视频处理后，Hugging Face `Dataset.from_list()` 才报：

```text
arrays to be concatenated must be identically typed, but float and double were encountered
```

根因不是 FFmpeg 日志，而是长度为 57 的 bool 序列与长度为 117 以上的序列计算 `next.done/q99` 时得到不同浮点 dtype。之前只规范化了视频统计，没有规范化全部 feature stats。

解决方案：

- `count` 统一为 `int64`；
- `min/max/mean/std/q*` 统一为 `float64`；
- 在昂贵的视频 remux 之前检查每个 episode 的统计 key、shape 和 dtype；
- 使用真实 1,000 条 episode 长度重建统计并验证 Arrow/Hugging Face 拼接与全局聚合；
- 未写完成 marker 的 part 重建，已验证 part 原样复用。

经验：schema 稳定性必须覆盖“派生元数据”，不能只覆盖训练 feature。

### 4.6 问题：运行环境“能 import”不等于能生产运行

遇到过的环境差异包括：

- 旧环境中的 LeRobot 0.3.3 缺少当前 writer 使用的 `finalize()`；
- 项目最初的文档/代码曾混用 0.4.4 与 0.6.0 假设；
- Python 3.12 下，原建议的 TensorFlow 2.15 没有对应 wheel；
- `torchcodec` 可 import，但没有系统 FFmpeg 时真实视频仍会失败；
- 受管环境默认 Hugging Face cache 指向只读 `/root/.cache`，造成大量测试失败；
- Windows 裸环境找不到 FFmpeg，但激活 conda 后可以；
- 4 张 A800 可被 `nvidia-smi` 看见，不代表容器有 `/dev/nvidia*`，更不代表 GPU 带 NVENC。

解决方案：

- 仓库固定使用 Python 3.12 的独立 `.venv` 和 `requirements.txt`；
- 对照安装后的 LeRobot 源码核验关键 API，而不是依赖旧文档字符串；
- 测试和正式运行显式设置可写的 `HF_HOME`、`HF_DATASETS_CACHE`、`TMPDIR` 等；
- 把 FFmpeg 当作系统依赖做启动前检查；
- 编码器做真实 preflight，失败时明确回退 CPU H.264；
- RLDS/TensorFlow 作为尚未完成真实验证的可选环境，不能污染主环境。

### 4.7 问题：合成测试会掩盖真实媒体故障

早期测试使用 4×6 或 8×10 的微型 RGB fixture。LeRobot 默认 SVT-AV1 在过小尺寸上可能触发原生除零崩溃；另一方面，不包含真实解码和封装的测试即使全绿，也不能证明生产视频路径有效。

解决方案：

- 需要真实编码的 fixture 使用至少 64×64；
- 每种数据集至少做一个真实 episode smoke；
- 写入后通过 FFprobe/解码核验 codec、pix_fmt、FPS、首帧 PTS、分辨率和总帧数；
- 图像有损转换用抽样 PSNR 门禁，数值字段则要求精确或按明示的 dtype 策略比较；
- 能字节复制的旧版 MP4 不解码重编码，减少质量损失和成本。

### 4.8 问题：OSSFS 不是普通本地文件系统

这是项目中最集中的故障来源。实际症状包括：

- 普通 MP4 muxer 在关闭文件时需要 seek-back/ftruncate，OSSFS 返回 `EINVAL`；
- 遍历或 `Path.resolve()` 触发大量远端 metadata 请求，长时间没有日志，看起来像死锁；
- 容量扫描或临时文件 stat 遇到 `Stale file handle`；
- `tail -f`、正在写的日志、临时文件可见性与本地文件系统不同；
- 进程停在 `D` 状态，worker 尚未启动；
- 跨文件系统 `rename/Path.replace` 不具备预期语义。

解决方案分为四层：

1. **路径隔离**：编码 temp、work、cache 放本地磁盘；OSSFS 只放需持久化的最终 bulk、marker 和日志。
2. **顺序写媒体**：必须直接写 OSSFS 时采用 fragmented MP4，避免尾部 seek-back。
3. **禁止递归热路径扫描**：容量由本地 ledger、catalog 和已提交 inventory 计算；恢复时不对远端根做 `rglob`、`resolve` 或全量 stat。
4. **跨文件系统提交协议**：使用 copy → size/range/footer/media verify → marker → 删除本地 bulk，不把 rename 当原子提交。

相关修复集中在 `d5db706`、`d29b979`、`3649c3a`、`6ff1d01`、`7975afe`、`c1f6318`。

### 4.9 问题：TB 级任务不能依赖“磁盘大概够”

一个完整 task 可能大于本地剩余空间，例如 AgiBot 最大任务约 875 GB，而当时本地可用约 691–695 GB。另一类失败是配置的 8/32 GiB inflight 上限小于实际分区峰值，转换在正式开始前或运行中停止。

解决方案：

- preflight 输出源容器字节、逻辑输入字节、预计输出、保守上界、最大 unit 峰值；
- 把 task 切为 episode、phase group、archive 或固定帧数的 bounded unit；
- coordinator 在派发前预留估算空间，上传完成并删除本地 bulk 后释放；
- 同时设置 `max-local-inflight-bytes` 和 `min-local-free-bytes`；
- 单 unit 超限时在 materialize 前失败，要求继续拆分；
- 最终 metadata 从小型 unit manifest 聚合，避免第二份 TB 级临时副本。

容量门禁应在任何大文件写入前执行。`--max-inflight-bytes` 是磁盘预算，不是内存预分配值，这一点在操作说明中必须写清楚。

### 4.10 问题：断点续传存在，但恢复过程仍可能“卡住”

最初的 resume 只解决“已有 checkpoint”，没有解决恢复本身的成本和可靠性：

- 每次恢复重新 FFprobe 全部视频；
- 对数万个远端路径逐个 `resolve/stat/glob`；
- checkpoint 指纹包含 runtime 绝对路径，搬迁 work/temp 后被误判为语义变化；
- 旧参数后来变成非语义运行参数，却导致历史 checkpoint 不兼容；
- checkpoint 更新粒度较粗，短时间数字不变容易误判停滞；
- `kill -9` 留下未关闭 Parquet/footer，不能安全复用当前 unit。

解决方案：

- marker 保存已完成校验的大小、帧数、schema、哈希抽样和媒体摘要；
- `--resume-validation fast` 先校验 marker 与文件指纹，只有不一致才做完整 FFprobe；
- 探测增加 timeout 与明确进度，超时保留 checkpoint 并报错；
- fingerprint 区分语义参数与可迁移 runtime 路径，源根路径归一化为稳定相对标识；
- checkpoint 协议兼容历史的 deferred-video 选项；
- unit 完成后原子写 marker，当前未完成 unit 在恢复时清除重建；
- 使用非阻塞 `flock` 防止同一输出被两个进程并发写；
- `SIGINT/SIGTERM` 做受控收尾，明确告知不要依赖 `kill -9`。

### 4.11 问题：并行度越高不一定越快

并发受编码器启动、CPU 核数、内存、源 OSSFS I/O、目标上传和单元大小共同限制。真实结果中既有 4 worker 达到 3 倍以上加速，也有 RoboOmni 小样本 4 worker 比 1 worker 更慢。

解决方案：

- 分开 `conversion workers`、`encoder threads per worker` 和 `upload workers`；
- 禁止 OpenMP/BLAS/Arrow 隐式嵌套线程放大；
- 使用 1/2/4 worker × 1/2 uploader 的真实矩阵，不用纯计算 microbenchmark；
- 比较前先证明输出 schema、索引、数值和媒体语义等价；
- 记录 wall time、aggregate FPS、峰值 RSS、峰值临时空间、重试和错误；
- 小样本结果只用于判断启动开销，不能外推全量吞吐。

### 4.12 问题：日志静默导致“慢”和“死锁”难区分

多次会话中，日志停滞的真实原因各不相同：正常长视频编码、checkpoint 仅按 64 episode 刷新、OSSFS I/O 等待、VS Code `rg` 全盘扫描造成竞争、旧 PID 文件、进程实际退出，或主进程在 worker 启动前卡在路径扫描。

解决方案：

- preflight、fingerprint、worker dispatch、episode save、upload、validation、commit 都输出阶段事件；
- ETA 使用单调时钟，并在每个 episode/unit 边界强制刷新；
- coordinator 和 worker 分别写 JSONL 事件；
- 监控同时查看 PID 是否存在、进程状态、CPU time 是否增长、子进程、日志 mtime、checkpoint mtime 和输出字节增长；
- 不信任陈旧 PID 文件，使用命令模式再次确认；
- `D` 状态意味着内核 I/O 等待，信号可能要等系统调用返回；不能立即删除锁或启动第二个 writer。

推荐判断顺序：

```text
PID 是否存在？
  ├─ 否：查看最终 traceback / exit code / _SUCCESS
  └─ 是：CPU time、I/O bytes、日志或 checkpoint 是否增长？
       ├─ 是：仍在运行，结合当前阶段判断刷新粒度
       └─ 否：检查进程状态和 syscall
            ├─ D：优先调查 OSSFS/磁盘 I/O
            ├─ S：检查子进程、队列和锁等待
            └─ R：检查计算/编码耗时和系统争用
```

### 4.13 问题：成功产物与半成品难区分

仅仅存在目标目录或若干 Parquet/MP4，不代表转换完成。崩溃可能留下结构部分可读、但 metadata 或全局索引尚未提交的目录。

解决方案采用两阶段发布：

- 开始时写 `_INCOMPLETE`；
- 每个 unit/task 先在隔离位置生成并验证；
- 提交 bulk 后写 durable marker；
- 全部 partition 完成后聚合 metadata、stats 和 collection manifest；
- 最后写 `_SUCCESS`，再移除 `_INCOMPLETE`；
- `--skip-existing` 必须重核 fingerprint、manifest 和可读性，不能只看目录或 marker 名称；
- 默认不覆盖已有有效 `_SUCCESS`。

### 4.14 问题：测试通过仍不能证明数据转换正确

单元测试容易遗漏上游损坏、真实 codec、远端文件系统语义和库的展示层转换。因此逐步形成四级证据：

1. **单元测试**：边界 shape/dtype、索引、fingerprint、锁、失败注入；
2. **合成端到端**：真实 `LeRobotDataset.create/add_frame/save_episode/finalize`；
3. **真实 smoke**：至少一个真实 episode，输出重新打开、FFprobe、数值抽样、PSNR；
4. **独立 evaluator**：不调用 converter reader，分别重开 raw、Parquet、视频和 LeRobot API；
5. **全量只读 preflight**：覆盖所有文件/episode 的 schema、长度、有限值、引用和损坏项；
6. **正式运行验收**：`_SUCCESS`、manifest 汇总、全局索引、源文件不变和目标对象一致性。

“通过测试”必须同时说明测试覆盖了哪一级。旧测试计数不能作为当前 HEAD 的全量结论。

### 4.15 问题：清洗算法的“可执行”不等于语义正确

清洗/对齐流水线在早期实现后，又通过针对性审查发现了几类会改变训练语义的问题：

- 对绝对坐标 action 直接与 state 做互相关，会把共同趋势误认为正确的动作响应；
- Stage2 无法检查某条 episode 时，旧逻辑可能把“未知”当成“拒绝”；
- 视频静止检测可能删除夹爪正在切换、但画面变化很小的关键控制帧；
- action canonical layout 曾多出一个无语义 padding，导致双臂 block 边界错误；
- 检查模块的 fail-open/skip/reject 原因没有被完整暴露，用户只能看到最终行数变化。

解决方案：

- 对 absolute-frame action 先差分，再与状态变化做互相关；
- 将 `checked/rejected/skipped` 分开，只有完成检查且明确不通过时才丢 episode；
- 在静止帧过滤中保护夹爪状态发生转换的帧；
- 把 action 单臂块明确为 34 维、双臂占 `[0:68]`，保留独立 128 维 bool mask；
- 通过 `StageResult` 和 inspect sidecar 记录每阶段修改、拒绝和跳过原因；
- VLM/SAM3 的网络或解析失败采取文档化策略，密钥只从环境变量读取，不写入 YAML 或日志。

对应修复可追溯到 `4ff05ae`、`a173c67`、`cf5d802`、`47b5706` 和 `ad018be`。

### 4.16 问题：检查工具中的“原始序号”与“最终位置”不是一回事

清洗会删除 episode 或帧，因此 raw episode index 不能直接用来读取 final dataset。早期播放器还存在暂停只停 UI、不停实际视频，以及调速控件未同步到媒体元素的问题。大量视频在打开页面时全部解码，也造成不必要的启动开销。

解决方案：

- sidecar 保存 raw→final 的显式映射；
- final dataset 使用自身的 positional index 查找；
- 通过本地支持 HTTP Range 的静态服务按需读取 MP4；
- 播放器统一控制图表游标、多个视频的 currentTime、pause 和 playbackRate；
- 加载 final dataset 时跳过无必要的视频预解码；
- `_parse_range` 覆盖空值、非法值和超出 EOF 等边界测试。

这部分演进集中在 `c706b0b`～`ecabe72`，说明可视化工具同样需要把索引契约和媒体协议当成正式接口。

### 4.17 问题：迁移远程仓库时不能覆盖原有协作源

本地目录名是 `vla-data-pipeline`，目标 GitHub 仓库名是拼写不同的 `vla-data-pipline`；原仓库还保留公司 Bitbucket `origin`。若直接改写 `origin`，会丢失原有跟踪关系；若只推当前分支，空远端又会缺少本地 `master`。

解决方案：

- 先通过路径、当前分支、工作区和 `git ls-remote` 确认目标；
- 保留原 `origin`，新增名为 `github` 的远程；
- 提交工作区后先推当前 `Xuanwei-Zhang`，再同步全部本地分支和标签；
- 最后用 `git ls-remote --heads --tags github` 校验目标引用。

2026-08-31 的远端核验结果为：`Xuanwei-Zhang` 指向 `856fc00`，`master` 指向 `31760a3`。本文提交后应再次核验并以新的 HEAD 为准。

## 5. 数据集专项问题清单

### 5.1 Mobile ALOHA

主要问题：移动任务与静态 co-training 的目录层级、动作结构、effort 和相机集合不同；14+2 动作布局仅看 shape 存在歧义；早期输出使用 `arm_0` 等占位名称和目录名 task，而非真实关节名与自然语言指令；转换器虽然保存了 2 维 `action.base`，下游 processing loader 却只读取 14 维 `action`；旧 `has_mobile_base` 又错误假设 state/action 尾部有 3 维底盘量。非流式图片中转慢，正式输出若直接落到 OSSFS 还会触发 MP4 封装问题。

解决方案：

- 递归发现 episode，并用 `/base_action` 交叉验证移动底盘动作；
- 按移动/静态、effort、相机和 FPS 拆成固定 schema partition；
- 将 14 维紧凑双臂数据标成真实左右臂电机名称，把底盘两维标成线速度/角速度，并建立 21 个源目录到自然语言任务的固定映射；
- `Episode`、LeRobot loader/writer、裁帧与时序对齐全链路显式携带 `action.base`；canonical 逻辑不再从 arm state/action 尾部错误裁掉三维；
- 采用 CPU H.264 流式编码，本地原子发布后再同步 OSS；
- 用 1,500 帧、3 相机真实 episode 对比流式与 PNG 中转路径。

已验证结果：全量只读预检为 1,103 episodes / 971,850 frames / 3 partitions；真实流式样本约 73.12 frames/s，对照路径约 17.98 frames/s，数值字段精确一致，视频 PSNR 通过门禁。

### 5.2 GR00T Teleop Sim

主要问题：24 个旧版 LeRobot part 各自从零编号；恢复指纹对 OSSFS 做约 48,000 次路径解析；part-022 的 bool quantile dtype 漂移；日志中的 `h264_mp4toannexb` 和 `moov atom` 容易被误判为根因。

解决方案：

- 保留 24 个语义分区，并确定性重映射 episode/frame/task 索引；
- 指纹改用词法规范化路径，避免远端 `resolve()`；
- 统一全部 stats 物理类型，并把 schema 检查提前到 remux 前；
- marker 粒度恢复，未完成 part 重建，已完成 part 复用。

### 5.3 MimicGen

主要问题：实际源目录拼写为 `minicgen`；62 个 HDF5 跨集合异构；`source/square.hdf5` 的旧 mask 引用不存在的 demo；持久 worker 切换分区后缓存已删除的 `tempfile.tempdir`；远端视频/恢复扫描成本高。

解决方案：

- 按一个 HDF5 一个固定 schema partition；
- dangling mask 只记录、不伪造 episode；
- 每个 worker 切换分区时重置临时目录状态；
- 引入快速恢复校验、探测超时、本地 runtime 与 unit checkpoint；
- 用真实 HDF5 完成数值与视频 smoke，并做全量 metadata preflight。

### 5.4 DexMimicGen

主要问题：9 个 HDF5 schema 不同；每 episode 有 MJCF；普通 MP4 不能安全直写 OSSFS；A800 无 NVENC；本地 inflight 预算不足；递归扫描 OSSFS 导致协调器 `D` 状态；runtime 搬迁引起 fingerprint 不兼容。

解决方案：

- HDF5 级分区和隔离 worker；
- XML 按 SHA-256 去重并 zlib 压缩为 sidecar；
- 使用 fragmented MP4 + CPU H.264；
- work/temp/cache 移到本地，final/marker 保持在 staging；
- 将配额计算改为本地 ledger，消除远端递归扫描；
- fingerprint 排除可迁移 runtime 路径并兼容旧选项；
- 提升 inflight 上限前先用计划峰值证明需求。

### 5.5 1X World Model Dataset

主要问题：v1.1 与 v2.0 使用不同 token/decoder 和数值 schema；跨 shard segment；测试集存在 17 个 token frame 对 64 个 state row 的无权威对齐；closure scalar/`[1]` 不一致；多 worker 解码可能产生不等价像素；OSSFS 临时 MP4 不可见。

解决方案：

- 固定数据与官方 decoder revision/hash；
- v1.1、v2.0 分区，不把测试集错误对齐；
- 合并跨 shard segment，保留单帧 segment；
- closure 统一为 `[1] float32`，同时加强 reader、Parquet 和 evaluator 检查；
- worker 直接生成最终兼容 chunk，避免第二份 TB 级聚合副本；
- 多 worker 只有通过像素等价门禁才启用；实测 W2/W4 不满足时批准 W1；
- OSSFS 媒体使用顺序写的 fragmented MP4。

### 5.6 RoboVerse v2

主要问题：没有图像、视频、时间戳或权威 FPS；action/state 长度可不相等；52 个 CALVIN pickle 为零字节；1,003 个 ManiSkill 文件存在 episode 内 dtype 漂移；LeRobot 展示层可能把 float64 物化为 float32。

解决方案：

- 不发明时间，默认拒绝正式转换，等待用户明确时间策略；
- action/state 等长则按 index 对齐，不等长拆为由 source episode UID 关联的完整 action-only/state-only 输出；
- 从固定 revision 下载并按 hash 原子替换损坏文件；
- 使用显式、可逆的 lossless dtype promotion；
- evaluator 同时区分 raw/Parquet 的物理精度和 LeRobot API 的展示层行为。

### 5.7 ARCap

主要问题：110.45 GB HDF5 以 `[10000,6] float64` 点云为主要载荷；五个分区 schema 不完全相同；完整 point-cloud 扫描昂贵；单个最终输出约 57.2 GiB，需要控制本地峰值。

解决方案：

- 全量检查低维字段，点云采用全 schema + 分层抽样，完整扫描保持显式选项；
- 按完整 phase group 组成不超过约 8,000 帧的 work unit；
- 4 个持久 worker + 2 个 uploader，上传后立即释放本地 bulk；
- 对 Parquet 做 footer/schema 和首中末 byte-range 校验；
- `open_bottle` 缺失字段保持独立 schema，不补零。

### 5.8 DexCap

主要问题：43 GB 级源 HDF5 同时含 10,000×6 点云、RGB、46 维 action 和多类状态；源不含 FPS；本地内存和临时空间峰值较高。

解决方案：

- 使用官方约定的 10 Hz，并在文档中明确其证据来源；
- 所有数值字段保持 float64/int64，只有 RGB 转 H.264 是有损项；
- 预先冻结全局范围，使用 bounded unit、RSS watchdog 和双 uploader；
- 真实矩阵选择 4 conversion workers × 2 upload workers，而不是只看理论核数。

### 5.9 FMB

主要问题：NumPy archive 同时包含 BGR、uint16 depth、字符串 primitive、标量和多维 Jacobian；旧导出可能使用 `actions` 复数名；单/多对象 schema 不同。

解决方案：

- BGR 显式转 RGB，depth 作为精确 ArrayND Parquet 保存；
- `action/actions` 只做有证据的别名规范化；
- scalar 使用明确的一元素容器约定；
- 单对象与多对象分区，schema fingerprint 忽略帧数和 Unicode 存储宽度；
- archive 作为 checkpoint unit，上传失败保留本地 bulk 供重试。

### 5.10 RoboOmni

主要问题：TFRecord/sidecar 没有声明 FPS；同一 task 可能包含多个 schema；部分 `speech_conv` WAV 引用缺失；小样本并行受进程和编码器启动成本主导。

解决方案：

- `--fps` 设为必填，不把常用 10 Hz 当源事实；
- task 下继续按 exact feature signature 分区；
- task 串行、episode 有界并行，已有 task marker 恢复时不重扫源数据；
- 首次真实转换只做一次 warmup，并缓存 marker；
- 小样本实测 1 worker 快于 4 worker，因此当前建议 1×8，但不外推全量。

### 5.11 RoboCOIN

主要问题：14 个 task 形成 7 种不兼容 schema；部分物理 Parquet dtype 与声明不一致；Airbot 存在未声明的重复相机别名；episode tasks 文本有空值或标点漂移。

解决方案：

- 输出 7 个独立 LeRobot partition；
- 对 Agilex/ALOHA 的混合 float32/float64 采用无损共同超类型 float64；
- 重复相机必须通过 size 和首/中/末 64 KiB digest 一致性后才能省略；
- task 关系以 frame-level `task_index -> tasks.jsonl` 为准，并记录 sidecar 差异；
- 原 MP4 字节复制，不解码、不重编码。

### 5.12 AgiBot World

主要问题：源由大量 v2.1 archive 组成，102 个为零字节且另有截断项；最大 task 超过本地容量；真实与仿真 schema、RGB/depth codec 不同；深度物理单位未知。

解决方案：

- 一 source episode 一个 direct-commit unit，任务串行、任务内 episode 并行；
- 128 GiB inflight 上限 + 100 GiB free-space reserve，必要时仅把 archive 解压暂存在受控 OSS scratch，禁止在那里编码；
- MP4 字节复制并保留真实 codec/pix_fmt/depth 标记，不发明深度单位；
- 提供固定 revision 的 320 文件 manifest 修复工具，支持 HTTP Range 续传、size/SHA-256 和原子替换；
- 空 archive 不跳过，正式全量在遇到上游损坏时必须停止。

### 5.13 RoboGene

主要问题：旧版 LeRobot task 以 robot split 组织，split 内仍可能发生 schema 漂移；深度 image struct 和 RGB MP4 的存储形式不同。

解决方案：

- 先按 top-level robot split，再按 Parquet schema fingerprint 分区；
- RGB H.264 字节复制，depth image struct 原样留在 Parquet；
- 只重写生成索引，保持 action/state 字段名、shape、dtype 和值；
- committed task 通过小型 marker 恢复，不重新遍历 raw 目录。

### 5.14 通用 HDF5、RLDS 与 raw_image_json

通用 HDF5 路径已经通过真实写入链路验证。建设中修复过一个接口错配：`reader.iter_frames(plan, episode)` 被直接当成单参数 callback，最终由端到端测试捕获并改为闭包适配。

RLDS 当前只验证了不依赖 TensorFlow 的辅助逻辑和缺依赖错误路径，尚未运行真实 `tfds.builder_from_directory()`。需要单独解决 Python 3.12 与 TensorFlow 版本策略，并避免一次性缓存大 split 的所有 steps。

raw_image_json 的 reader 逻辑有测试，但目录/metadata 约定没有真实数据证据。曾修复“无条件要求全局 FPS”的问题：现在 episode metadata 自带 FPS 时无需配置兜底。真正接入数据前仍应先扩展 schema dump 并以真实结构修正约定。

## 6. 典型故障诊断表

| 表象 | 优先检查 | 常见根因 | 安全处理 |
|---|---|---|---|
| 启动后立即退出 | cwd、Python 路径、CLI 完整性 | 不在仓库根、漏脚本路径或参数 | 修正命令；若 Python 尚未启动通常无需 resume |
| 有 PID 但长时间无 worker | `ps` 状态、syscall、路径扫描 | coordinator 卡在 OSSFS `resolve/rglob/stat` | 不启动第二进程；修复扫描路径后受控停止并 resume |
| checkpoint 数字短时不变 | 当前 unit/episode、CPU time、输出增长 | marker 只在 unit 边界刷新 | 继续观察，不根据单一计数判断卡死 |
| `Stale file handle` | 报错路径是否在 OSSFS | 远端 temp/容量扫描/stat | temp/work 本地化；远端只保留最终对象和 marker |
| `Invalid argument` 写 MP4 | muxer 是否 seek-back | 普通 MP4 直写 OSSFS | 本地编码后复制，或 fragmented MP4 顺序写 |
| resume fingerprint mismatch | diff semantic 与 runtime 参数 | work/temp/raw 根搬迁或旧选项变化 | 对路径作稳定归一化，版本化兼容非语义参数 |
| inflight quota stopped | 计划 unit 峰值与限制 | 限制小于一个完整 unit | 提高经过证明的预算或进一步拆 unit，不删 checkpoint |
| Arrow mixed float/double | 具体 feature/stat/episode | quantile 或源 payload dtype 漂移 | 统一派生统计 dtype；源字段只做可逆显式提升 |
| schema `[1]` 但写成 scalar | reader 实际返回值和 Parquet | singleton 被 squeeze | 保持 ndarray `[1]`，加强物理 schema/evaluator |
| `moov atom` 或 bitstream-filter 日志 | 最后的 Python traceback | 正常 remux 日志被误判 | 从首个真正异常向上定位，不凭 FFmpeg 噪声判断 |
| 看到 GPU 但 NVENC 失败 | `/dev/nvidia*`、encoder capability | 容器未挂设备或 A800/A100 无 NVENC | preflight 后使用 CPU H.264；换 L4/A10/RTX 才测 NVENC |
| 大量测试突然失败 | cache 路径可写性 | Hugging Face 默认 cache 在只读目录 | 显式设置 `HF_HOME`/`HF_DATASETS_CACHE` |
| 输出目录存在但无法验收 | `_SUCCESS`、manifest、重开结果 | 半成品或 finalize 未完成 | 只把完整 marker 协议通过的目录视为成功 |
| 进程很慢且系统负载高 | 并发 `rg`、其他转换、system CPU | VS Code 全盘扫描 `/mnt/data` | 排除大数据挂载并错峰运行，不盲目加 worker |

## 7. 标准接入流程

以后新增数据集时，按以下顺序执行，可以复用本项目已经付出的排错成本。

### 第 1 步：冻结来源

- 记录数据集 revision、官方仓库 commit、论文/数据卡 URL；
- 保存文件相对路径、size、mtime，关键文件增加 SHA-256；
- 明确 raw tree 只读；
- 将下载损坏与转换逻辑问题分开处理。

### 第 2 步：只读探测

- 先运行通用 schema dump；
- 枚举所有 episode、字段、shape、dtype、相机、FPS、task、split；
- 检查零字节/截断文件、dangling reference、重复对象和 schema 漂移；
- 大数组先读 header，payload 检查分为抽样与显式全量两档。

### 第 3 步：建立语义映射

每个字段都回答：来源、目标名、shape、dtype、单位、坐标系、时间含义、变换、是否有损、证据。未知项写 unknown；需要用户策略的字段做成显式参数。

### 第 4 步：确定分区和索引

- 只有 schema 完全兼容的数据才能进入同一 partition；
- 冻结 source episode → partition episode、frame、task 的映射；
- partition 内索引连续；collection manifest 保存跨 partition 的稳定范围；
- subset smoke 使用独立 UID，不能污染正式输出的 index namespace。

### 第 5 步：设计资源上界

- 估算单 unit 峰值、总输出和安全余量；
- 明确本地/OSSFS 的 work、temp、cache、resume、logs、final 布局；
- 设置 inflight bytes、free-space reserve、RSS watchdog 和上传背压；
- 单 unit 必须能独立完成、验证、提交和释放。

### 第 6 步：实现恢复事务

- plan/fingerprint 必须覆盖源、selection、schema、task、FPS、codec 和输出身份；
- runtime 路径等非语义参数应可迁移；
- marker 原子写入且含足够的写后证据；
- 恢复只复用经过重新校验的 unit；
- 锁冲突、容量不足和参数不一致都在大写入前失败。

### 第 7 步：真实 smoke

- 覆盖至少一个真实 episode；异构集合至少覆盖每种关键 schema；
- 重开 raw、Parquet、视频和 `LeRobotDataset`；
- 数值字段检查 dtype/shape/value，视频检查结构和画质；
- 验证源文件 size/mtime/hash 未变化；
- 模拟中断、失败上传、损坏 marker 和并发启动。

### 第 8 步：并行基准

- 比较 1/2/4 worker 和必要的 uploader 组合；
- 所有候选先做语义等价检查；
- 记录吞吐、wall time、CPU、RSS、本地临时空间和错误；
- 选择真实瓶颈下的配置，不以 GPU 数或 CPU 核数直接推断。

### 第 9 步：正式运行与观察

- 使用 `tmux` 或 `nohup`，日志在稳定且可追踪的位置；
- 保存 PID 但不盲信 PID；
- 根据 phase、CPU/I/O、worker、日志、checkpoint 多信号判断状态；
- 受控中断后原命令 `--resume`，不随意删锁、marker 或已验证输出。

### 第 10 步：交付验收

- 进程退出码为 0；
- `_SUCCESS` 存在、`_INCOMPLETE` 不存在；
- 所有 partition 可被标准 LeRobot API 重开；
- manifest 汇总和实际文件/行数一致；
- episode/frame/index/task 连续且映射可追溯；
- 数值与媒体验证达到文档门禁；
- source 清单未变化；
- work/cache 中不再残留无需恢复的 bulk；
- 文档写明已验证范围、未验证范围和剩余风险。

## 8. 不应重复的做法

- 不根据字段名、shape 或“业内通常如此”猜单位、坐标系、FPS 和动作语义。
- 不为合并异构数据集而静默补零、裁剪、丢字段或缩窄 dtype。
- 不把测试 fixture 通过等同于真实数据、真实 codec 或真实 OSSFS 通过。
- 不在 OSSFS 上递归扫描容量、反复 resolve 全量路径或直接写普通 MP4 临时文件。
- 不因 checkpoint 一段时间没更新就判定进程死亡，也不因 PID 存在就判定任务健康。
- 不在旧进程仍持锁时启动第二个 writer，不直接删除锁文件。
- 不只看 `_SUCCESS` 文件名；必须验证 marker 内容、fingerprint、manifest 和数据可读性。
- 不用 `kill -9` 作为正常停止方式。
- 不把小样本并行加速比外推到全量任务。
- 不把历史测试数字当成当前 HEAD 的验证结果。

## 9. 当前已知缺口与后续优先级

### P0：保持验收可信

- 修正 2026-08-31 抽查中“上传失败会自动重试成功，而旧测试仍期待抛错”的契约分歧；明确期望是 retry exhaustion 才抛错，还是该路径禁用重试。
- 所有正式转换完成后，都必须补独立最终验收记录；“脚本与 smoke 已完成”不等于“全量数据已完成”。

### P1：补齐真实格式验证

- 为 RLDS 建兼容的隔离环境并运行真实 TFDS 数据；
- 在真实 raw_image_json 数据到位后修订目前仅由测试定义的文件约定；
- 对仍缺 FPS、单位或动作坐标系的数据集取得官方证据或明确用户策略。

### P2：继续降低长期运维成本

- 将各专项 converter 中已经稳定的 task catalog、direct commit、marker、空间 ledger 和 evaluator 约定继续收敛为共享 API；
- 给文档增加自动链接检查和“文档记录的命令是否仍被 CLI 接受”的轻量测试；
- 将历史运行状态从长期设计文档中分离为带日期的 run report，避免陈旧状态误导操作人员。

## 10. 文档与代码索引

### 总览

- [`README.md`](../README.md)：安装、通用用法、清洗/对齐和 canonical 表示；
- [`PIPELINE_STATUS.md`](../embodied_datasets/scripts/convert_scripts/PIPELINE_STATUS.md)：转换模块和历史验证状态；
- `embodied_datasets/scripts/convert_scripts/convert_core/`：通用转换、恢复、并行和提交组件；
- `embodied_datasets/scripts/process_scripts/`：清洗、对齐、质检和统一表示；
- `embodied_datasets/scripts/inspect_tool/`：前后对比及同步视频播放器。

### 专项转换文档

- [`ARCAP_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/ARCAP_CONVERSION.md)
- [`DEXCAP_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/DEXCAP_CONVERSION.md)
- [`GR00T_TELEOP_SIM_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/GR00T_TELEOP_SIM_CONVERSION.md)
- [`1X_WORLD_MODEL_DATASET_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/1X_WORLD_MODEL_DATASET_CONVERSION.md)
- [`MIMICGEN_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/MIMICGEN_CONVERSION.md)
- [`DEXMIMICGEN_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/DEXMIMICGEN_CONVERSION.md)
- [`ROBOVERSE_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/ROBOVERSE_CONVERSION.md)
- [`FMB_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/FMB_CONVERSION.md)
- [`ROBOOMNI_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/ROBOOMNI_CONVERSION.md)
- [`ROBOCOIN_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/ROBOCOIN_CONVERSION.md)
- [`AGIBOT_WORLD_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/AGIBOT_WORLD_CONVERSION.md)
- [`ROBOGENE_CONVERSION.md`](../embodied_datasets/scripts/convert_scripts/ROBOGENE_CONVERSION.md)

## 11. 一页式结论

这个项目最难的部分并不是把数组写进 Parquet，而是确保在多来源、异构 schema、远端 FUSE、长时间运行、有限本地空间和不完整上游数据同时存在时，仍然能够回答以下问题：

1. 输入到底是什么，语义证据在哪里？
2. 哪些变换发生了，哪些信息被保留或有损？
3. 中断后从哪里恢复，为什么该 checkpoint 可以信任？
4. 并行和存储峰值是否有实测依据？
5. 目标目录何时才算真正完成？
6. 另一个不依赖转换器内部实现的工具，能否重新证明结果？

最终形成的通用答案是：**固定证据、只读预检、按 schema 分区、有界 work unit、语义 fingerprint、独立验证、marker 驱动恢复、两阶段发布，以及对未知信息保持诚实。** 这套方法比任何单个 converter 更值得复用。
