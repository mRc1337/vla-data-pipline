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
- `GET /api/datasets/{uid}/episodes/{index}/series` 返回有界 Parquet state/action 采样。
- `GET /api/datasets/{uid}/tasks` 返回数据集的子任务名称、task index 和 episode 数量；`GET /api/datasets/{uid}/episodes?task_index=<id>` 按 task 筛选 episode。前端浏览顺序为“数据集 → Task → Episode”。
- `GET /api/datasets/{uid}/episodes/{index}/preview` 返回 Episode 元数据、instruction、按元数据定位的多相机视频 URL 和 Stage 产物对比；前端工作台提供 Episode 搜索选择、同步播放、当前帧/时间戳、state/action 及位置/姿态/关节曲线。视频通过 `/api/videos/...` 的 HTTP Range 响应播放。
- `GET /api/videos/{uid}/{relative_path}` 支持 `Range: bytes=start-end`，不会把 MP4 上传到云端。
- `POST /api/pipelines/run` 创建 Stage 任务；`GET /api/tasks/{id}` 轮询状态、进度、错误和汇总。
- `POST /api/annotations` 新增人工标签版本，自动标签不会被覆盖。
- `POST /api/exports` 生成带筛选表达式、episode 列表和源数据版本的 manifest；`materialize=true` 才显式复制数据。

默认单机 profile 使用 SQLite + asyncio；Compose 中的 Redis/PostgreSQL/Celery
是可切换的分布式 profile。阶段产物写入
`data_curation/stage<N>/<dataset>/<run_id>/manifest.json` 和
`reports/summary.json`，以 manifest 引用原始数据，避免复制八份视频。
