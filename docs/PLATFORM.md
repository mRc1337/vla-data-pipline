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
- `GET /api/datasets/{uid}/episodes/{index}/series` 返回有界 Parquet state/action 采样。
- `GET /api/videos/{uid}/{relative_path}` 支持 `Range: bytes=start-end`，不会把 MP4 上传到云端。
- `POST /api/pipelines/run` 创建 Stage 任务；`GET /api/tasks/{id}` 轮询状态、进度、错误和汇总。
- `POST /api/annotations` 新增人工标签版本，自动标签不会被覆盖。
- `POST /api/exports` 生成带筛选表达式、episode 列表和源数据版本的 manifest；`materialize=true` 才显式复制数据。

默认单机 profile 使用 SQLite + asyncio；Compose 中的 Redis/PostgreSQL/Celery
是可切换的分布式 profile。阶段产物写入
`data_curation/stage<N>/<dataset>/<run_id>/manifest.json` 和
`reports/summary.json`，以 manifest 引用原始数据，避免复制八份视频。
