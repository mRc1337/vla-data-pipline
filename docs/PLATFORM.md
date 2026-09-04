# 本地数据治理平台

该平台把浏览器、FastAPI 和异步流水线部署在同一数据节点。原始数据和
LeRobot 3.0 文件始终留在本地挂载点；API 只返回元数据、曲线采样和视频
Range 响应。

## 快速启动

也可以直接使用一键脚本：

```bash
./scripts/local_deploy.sh start
./scripts/local_deploy.sh status
./scripts/local_deploy.sh logs api
./scripts/local_deploy.sh stop
```

首次启动会创建 `.venv`、安装 Python/npm 依赖，并把 PID/日志写入
`.local-run/`。若依赖已经准备好，可使用 `--skip-install`。Docker 模式：

```bash
./scripts/local_deploy.sh start --mode docker
./scripts/local_deploy.sh stop --mode docker
```

```bash
cd /home/pai/zxw/vla-data-pipeline
.venv/bin/pip install -r requirements.txt
VLA_DATA_ROOT=/mnt/data/embodied_datasets/public_datasets_staging \
  .venv/bin/uvicorn vla_platform.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/api/catalog/scan
curl http://localhost:8000/api/datasets
```

前端：`cd frontend && npm install && npm run dev`。生产环境先执行
`npm run build`，再用 `docker compose up --build` 启动 API、Redis、
PostgreSQL、Celery CPU worker 和 Nginx。GPU Stage 可复制 worker 服务并
设置 `CUDA_VISIBLE_DEVICES`，模型 Stage 仍使用同一个任务/manifest schema。

## API 闭环

- `POST /api/catalog/scan` 创建异步扫描任务，支持 `quick`、`standard`、`deep` 三种模式和限定 `root`。
- `GET /api/catalog/scans/{scan_id}` 查询进度；`GET /api/catalog/scans/{scan_id}/events` 通过 SSE 接收进度；`POST /api/catalog/scans/{scan_id}/cancel` 取消任务。
- 快速扫描只读取 `meta/info.json` 和目录标记，不递归统计视频大小；未变化数据集由指纹直接跳过。
- 扫描任务和指纹写入本地 SQLite；API 重启后未完成任务会标记为 `failed`，可以重新提交，已完成数据集仍可通过指纹复用。
- `standard`/`deep` 会把视频和 Parquet 文件的 `size+mtime` 作为文件级指纹；未变化文件直接复用已索引的 codec、分辨率、FPS、行数和完整性结果，新增、修改或删除的文件才会重新处理。
- `standard` 只读取视频容器头和 Parquet footer/schema/行数；`deep` 额外解码至少一帧并校验 Parquet schema。坏文件会落成 `fail` 记录，不会使整批扫描失败。
- 扫描范围默认只发现 staging 根下的标准 LeRobot 容器；`root` 可显式限定到嵌套目录或 `data_curation/stageN`，避免在 FUSE/海量目录中递归探测未知树。
- `GET /api/datasets/{uid}/episodes/{index}/series?view=raw|valid|repaired|diff` 返回有界
  Parquet state/action 采样。新 `vla_curation_filter` 使用 `valid`：从父 Manifest 链读取
  `step_validity.parquet`，无效帧以曲线断点显示；旧 overlay 的 `repaired/diff` 仅保留只读兼容。
- `GET /api/datasets/{uid}/tasks` 返回数据集的子任务名称、task index 和 episode 数量；`GET /api/datasets/{uid}/episodes?task_index=<id>` 按 task 筛选 episode。前端浏览顺序为“数据集 → Task → Episode”。
- `GET /api/datasets/{uid}/episodes/{index}/preview` 只从本地 SQLite 返回 Episode 元数据、
  多相机视频定位和 Stage 摘要，不扫描远端 Stage 目录。展开某个 Stage 时，前端再请求
  `GET /api/datasets/{uid}/episodes/{index}/stages/{stage_id}`；后端依据索引中的 Stage 根目录
  和 Manifest 明确列出的文件按需读取，不执行递归目录扫描。
- State/Action 曲线按数据集 `features[字段].names` 显示原始元素名称，并支持分别多选；平台不对固定索引赋予末端位置、姿态或关节语义，以兼容跨本体表示。
- 新格式曲线支持“原始数据 / 有效帧”切换，不显示虚假的修复值或差值；Stage 标签中的相邻异常帧会合并为连续红色区间，单帧异常至少显示一帧宽度。
- Episode 工作台将 Stage 1–3 异常位置与 Stage 4–8 专项内容合并为一个
  Stage 1–8 模块；各 Stage 默认展开，首屏摘要显示后异步加载详情，并可独立折叠。
  页面顺序为多相机视频、State/Action 曲线、Stage 1–8 模块。当前 Stage 6 原生读取
  `data_curation/stage6/<dataset>/episode_<index>.json`，展示任务计划、场景对象、
  语义子任务时间轴、置信度和可点击证据帧；Manifest 中的完成数同时显示为全量进度。
- Stage 1–3 与 Stage 4–8 使用一致的产物状态标签。Stage 目录不存在、运行存在但当前
  Episode 尚未产出、Episode 已被前序阶段过滤和产物可用是四种不同状态，缺失产物不会
  被误报为“无异常”。产物可用时，Stage 1/3 从 `frame_flags.parquet`、
  `episode_summary.parquet` 报告连续异常区间和帧统计；Stage 2 从
  `dimension_metrics.parquet`、`episode_flags.parquet` 报告方向一致率、时延、失败维度和
  Episode 过滤结论，并明确该阶段没有帧级异常区间。
- Stage 4 从运动学 Parquet 展示位置/姿态误差、软硬异常、Base 标定与异常区间；Stage 5
  展示逐 Episode 坐标变换和训练候选状态；Stage 7 展示 SAM2/FK 一致性结论、IoU、覆盖率、
  置信度及可跳转的采样帧；Stage 8 展示处置建议、无效区间和逐相机黑屏/模糊/损坏统计。
  Stage 4/5/7/8 没有当前 Episode 产物时才显示方法目标、预期可视化和所需字段的占位，
  “暂无产物”不解释为通过。Stage 6 若目录已存在但尚未生成选中 Episode，则显示“待处理”。
- Stage 任务完成一批结果后，应调用
  `vla_platform.stage_index.publish_stage_index_snapshot(stage_root, stage_id, payloads)`。
  它按稳定的 Episode 区间生成 `search_index/part-*.parquet`，最后原子发布
  `search_index_manifest.json`。Manifest 包含 generation、更新时间、文件数、变更 shard 和
  tombstone；搜索索引先比较 generation，未变化时不枚举 Episode JSON，变化时只批量读取
  新增或修改的 Parquet shard。主 `manifest.json` 的内容哈希也会写入索引 manifest；若 Stage
  继续产出导致主 manifest 变化，旧快照自动失效，不会掩盖新标签。旧的逐 Episode JSON
  目录仍可兼容读取；已有产物可执行
  `python -m vla_platform.stage_index <stage_root> --stage-id <1-8>` 一次性生成首个快照。
  对已经完整进入本地目录索引的已完成 Stage，可追加
  `--catalog-db <catalog.sqlite3> --dataset-uid <uid>`，直接从本地索引回填，避免再次读取海量 JSON。
- `POST /api/datasets/{uid}/episodes/{index}/video-proxies` 按需生成从零开始、帧数固定的 H.264 Episode 代理，`GET /api/video-proxy-jobs/{job_id}` 查询任务状态。前端默认使用代理，并可切回原始分片。
- 代理默认缓存在本地 `.local-run/video_proxy`，避免 FFmpeg 在 OSS/FUSE 挂载上随机写 MP4；可用 `VLA_VIDEO_PROXY_ROOT` 修改位置。默认单 Worker、每次编码 2 线程、缓存上限 20 GiB，可分别通过 `VLA_PROXY_WORKERS`、`VLA_PROXY_ENCODER_THREADS`、`VLA_PROXY_CACHE_BYTES` 调整。
- `GET /api/video-proxies/files/{relative_path}` 和 `GET /api/videos/{uid}/{relative_path}` 均支持 `Range: bytes=start-end`，不会把 MP4 上传到云端。
- `POST /api/pipelines/run` 创建 Stage 任务；`GET /api/tasks/{id}` 轮询状态、进度、错误和汇总。
- `POST /api/annotations` 新增人工标签版本，自动标签不会被覆盖。
- `POST /api/exports` 生成带筛选表达式、episode 列表和源数据版本的 manifest；`materialize=true` 才显式复制数据。

默认单机 profile 使用 SQLite + asyncio；Compose 中的 Redis/PostgreSQL/Celery
是可切换的分布式 profile。阶段产物写入
`data_curation/stage<N>/<dataset>/<run_id>/manifest.json` 和
`reports/summary.json`，以 manifest 引用原始数据，避免复制八份视频。
