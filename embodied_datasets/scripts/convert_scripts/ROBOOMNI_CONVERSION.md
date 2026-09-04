# RoboOmni / OmniAction 转换

`convert_roboomni_to_lerobot.py` 将挂载的 `fnlp/OmniAction` RLDS TFRecord 转为 LeRobot v3.0。挂载目录目前可能叫 `robomni`；脚本同时接受 `roboomni` 和 `robomni` 拼写，并且只读原始目录。

原数据的 sidecar 和 TFRecord 没有声明 FPS，也没有 timestamp feature。因此 `--fps` 是必须显式指定的输出时间基准，不是原始数据事实。下游训练路径常按 10 Hz 使用，但正式转换前仍需确认目标输出速率。

## 字段映射

| 原字段 | shape/dtype | 语义/单位 | LeRobot 字段 | 转换 | 依据 | 是否有损 |
|---|---|---|---|---|---|---|
| `steps/action/*` | component-defined numeric | 保留原始值；单位未在源数据声明 | `source.action.*` | identity | `features.json` | 否 |
| `steps/observation/*` numeric | component-defined numeric | 保留原始值；不标准化、不推断单位 | `source.observation.*` | identity | `features.json` | 否 |
| `steps/*` text | UTF-8 | 保留源文本 | `source.*` | UTF-8 decode | `features.json` | 否 |
| `steps/observation/image*`, `first_frame_image` | sidecar-defined RGB `uint8` | 保留分辨率 | `observation.images.*` | 流式 CPU 视频编码 | 视频编码有损 |
| `steps/speech_conv` | 源字符串 | `[UNK]` 分隔的外部 WAV 相对引用 | `source.speech_conv` | 保留引用；不静默补音频 | TFRecord 值及 `speech/` 树 | 缺失 WAV 不可恢复 |

相机命名固定为：`observation/image` → `observation.images.primary`，`observation/image_wrist` → `observation.images.wrist`，`first_frame_image` → `observation.images.first_frame`。不同 exact `features.json` signature 保持为独立 schema partition，不补零、不重排、不归一化、不丢弃字段。

## task、index 和 chunk

首次运行只做一次 TFRecord wire/index scan，并写入：

- `resume/task_catalog.json`：稳定 task 顺序、摘要和 plan 路径，不含 TB 级 bulk；
- `resume/tasks/<task_key>/task_plan.json`：该 task 的精确 source path/size/mtime、TFRecord offset、schema、字段映射、FPS、编码参数和 episode/frame 范围；文件带不可变 fingerprint。

`task_key` 来自真实 source instruction，按 UTF-8 字典序加 hash 稳定生成。一个 task 可以包含多个 schema partition；task 内每个 episode 是一个有界 unit，unit 的最终 chunk 编号按 task 顺序确定，因此 chunk 不跨 task。LeRobot partition 内使用连续的 partition-local `task_index`，collection manifest 同时记录稳定的 collection-global `task_index`。

每个 task 的流程严格串行：

```text
task plan → 当前 task preflight → task 内并行编码 → 本地验证
→ copy 到最终 chunk → size/footer/媒体抽样验证 → task commit marker
→ 清理该 task 本地 bulk → 下一个 task
```

不同 task 不重叠 preflight、编码或上传。上传采用“复制 → 验证 → marker → 删除本地 bulk”；不使用跨文件系统 `Path.replace`。上传失败不写 task marker，并保留当前 task 的本地 checkpoint。最终 `_SUCCESS` 最后写入；不存在 `_SUCCESS` 时集合视为未完成。

## resume 和空间保护

启动时只读取 task catalog、global state 和 task commit markers。已有 committed task 只读 marker，跳过 source preflight、转换和远端 bulk 校验；只有当前未提交 task 读取 task plan、校验引用的 source stat，并做首/中/末 payload 抽样。task marker 记录 source 引用、index range、目标文件大小/schema/媒体摘要和 unit marker 证据。

RoboOmni 不使用固定本地总量上限，也不要求 `--max-local-temp-bytes`。只使用 `--min-local-free-bytes`：task 开始和每次 dispatch 估算并检查峰值，运行中触及保留空间时抛错并保留 checkpoint。临时文件位于 `--local-work-root` 下，不在 OSSFS/FUSE 上编码。

首次实际转换在第一个未提交 task 的 preflight 后只做一次 30–60 帧 CPU warmup，写入 `resume/warmup.json` 后清理 warmup 目录；resume 只读该 marker，不重复预热。

## 运行命令

只检查 catalog 和两个 task 的最小 preflight（不启动转换）：

```bash
PYTHONPATH=embodied_datasets/scripts/convert_scripts .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_roboomni_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/robomni \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/roboomni \
  --local-work-root $HOME/roboomni_staging \
  --output-dataset-uid roboomni --fps 10 --inspect-only --max-shards 1
```

独立 smoke（两个 task，每 task 一个 episode；执行后应删除 output/work/resume/cache）：

```bash
PYTHONPATH=embodied_datasets/scripts/convert_scripts .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_roboomni_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/robomni \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --local-work-root $HOME/roboomni_staging \
  --output-dataset-uid roboomni_smoke_<run-id> --fps 10 --max-shards 1 \
  --max-tasks 2 --max-episodes 1 --workers 1 \
  --encoder-threads-per-worker 2 --upload-workers 1 \
  --min-local-free-bytes 200000000000
```

正式后台转换命令（只提供，不在本变更中执行）：

```bash
PYTHONPATH=embodied_datasets/scripts/convert_scripts .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_roboomni_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/robomni \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --local-work-root $HOME/roboomni_staging \
  --output-dataset-uid roboomni --fps <confirmed-output-fps> --resume \
  --workers 1 --encoder-threads-per-worker 8 --upload-workers 1 \
  --min-local-free-bytes 200000000000
```

查看状态：

```bash
cat $HOME/roboomni_staging/.conversion_resume/roboomni/global_state.json
find $HOME/roboomni_staging/.conversion_resume/roboomni/tasks \
  -name commit.json -print
```

恢复使用与原运行完全相同的参数并追加 `--resume`；不要修改 FPS、output UID、task selection 或 task plan 中的编码参数。

## 已完成的最小验证

真实挂载上完成了独立 `/tmp` inspect 和两 task smoke：catalog 选出 2 个 task，分别 1 episode/43 frames、1 episode/30 frames；CPU AV1 输出 640×480、10 FPS。两个 partition 均用 `LeRobotDataset` 重开成功，index 从 0 连续到末帧，task 文本可读，`_SUCCESS` 存在且 `_INCOMPLETE` 已移除。

新 task-scoped 编排还对同一 task（4 episodes、105 frames、两个 schema partition）做了同参数比较：1 worker × 8 encoder threads 为 21.685 s，4 workers × 8 encoder threads 为 37.381 s；两者输出 episode/frame/FPS/index/task 一致。这个小样本受编码器启动开销影响，当前正式建议 1×8，不据此推断全量吞吐。

另一次 recovery smoke 在第一个 task marker 写入后模拟第二 task 中断；恢复时禁止全量 `build_catalog` 调用，且只对第二 task 执行 preflight，最终成功发布。全量转换未执行，正式 output 未写入。

风险仍包括：源 FPS 未声明；numeric state/action 单位未声明；`speech_conv` 引用的部分 WAV 可能缺失；不同 schema partition 的字段集合不一致，不能合并为单一 schema。
