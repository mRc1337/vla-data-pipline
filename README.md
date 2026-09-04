# Embodied Studio

Embodied Studio 是面向 VLA（Vision-Language-Action）机器人数据的本地治理与可视化平台。它在数据所在节点直接索引 LeRobot v3.0 数据和 Stage 1–8 质检产物，通过浏览器提供 Episode 搜索、标签筛选、多相机视频、State/Action 曲线以及阶段级审计视图。

平台默认不上传、不复制原始数据，也不会在预览请求中递归扫描 Stage 目录；摘要和搜索读取 SQLite 索引，展开某个阶段时才按 Manifest 中的确定路径加载详情。

## 主要能力

- 数据集、子数据集、Task 和 Episode 分层浏览。
- 按数据集、Instruction、Episode 编号及 Stage 1–8 标签组合搜索。
- 多相机 Episode 视频同步播放、时间轴定位和按需 H.264 代理缓存。
- State/Action 多维曲线、原始/有效帧视图及异常区间联动。
- Stage 1–8 统一状态展示，区分已有产物、待处理、前序过滤和未生成。
- Stage 详情按需加载，展示阈值、异常维度、运动学误差、语义分段和视频质量证据。
- 分层增量搜索索引：Stage Manifest 未变化时整阶段跳过，变化时只更新 Parquet 分片和 tombstone。
- 异步目录扫描、任务进度、人工标注和可复现导出 Manifest。

平台覆盖的八个阶段为：

1. Sudden Change Detection
2. State-Action Trend Alignment
3. Extreme Value Filtering
4. Joint-End-Effector Forward Kinematics Consistency
5. Base Frame and End-Effector Orientation Alignment
6. Instruction Consistency
7. Video-State Consistency
8. Video Quality Filtering

## 架构

```text
Browser / React + Vite
          │ /api
          ▼
FastAPI / vla_platform
    ├── SQLite catalog and search index
    ├── LeRobot metadata and Parquet readers
    ├── Stage manifest and detail readers
    ├── Thumbnail / H.264 proxy workers
    └── Async scan and pipeline tasks
          │
          ▼
local or mounted dataset storage
```

默认开发模式只监听 `127.0.0.1`。远程机器上运行时，可通过 SSH 端口转发或开发环境提供的端口代理访问。

## 数据目录约定

默认数据根目录为：

```text
/mnt/data/embodied_datasets/public_datasets_staging/
├── lerobot_v3_0/
│   └── <dataset>/
│       ├── meta/
│       ├── data/
│       └── videos/
└── data_curation/
    ├── stage1/<dataset>/
    ├── ...
    ├── stage8/<dataset>/
    └── _catalog/
```

也可以通过 `--data-root` 或 `VLA_DATA_ROOT` 指向其他目录。Stage 产物通过 Manifest 引用源数据，不需要为每个阶段复制视频。

## 快速启动

运行环境需要 Python 3.12+、Node.js/npm 和 FFmpeg。在 Ubuntu/Debian 上安装系统依赖：

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv ffmpeg
```

克隆仓库后运行：

```bash
python3.12 -m venv .venv
./scripts/local_deploy.sh start
```

脚本会安装 Python 和前端依赖并启动：

- Web：<http://127.0.0.1:5173>
- API 文档：<http://127.0.0.1:8000/docs>

常用管理命令：

```bash
./scripts/local_deploy.sh status
./scripts/local_deploy.sh logs api
./scripts/local_deploy.sh logs web
./scripts/local_deploy.sh stop
```

依赖已经安装时，可以跳过安装步骤：

```bash
./scripts/local_deploy.sh start --skip-install
```

指定数据目录：

```bash
./scripts/local_deploy.sh start \
  --data-root /path/to/public_datasets_staging
```

## 使用流程

1. 启动服务并打开 Web 页面。
2. 执行目录扫描，使数据集、Task、Episode 和视频元数据进入本地 Catalog。
3. 首次使用或 Stage 产物更新后，点击“更新搜索索引”。
4. 在 Episode 搜索中选择数据集和 Stage 标签，也可以输入 Task、Instruction 或 Episode 编号。
5. 点击搜索卡片进入 Episode 工作台，查看视频、State/Action 曲线和 Stage 1–8 详情。

索引刷新只读取紧凑元数据、Manifest 和标签分片，不解码视频。Episode 预览摘要直接读取 SQLite，展开阶段详情时才访问对应产物。

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VLA_DATA_ROOT` | `/mnt/data/embodied_datasets/public_datasets_staging` | 数据根目录 |
| `VLA_CURATION_ROOT` | `$VLA_DATA_ROOT/data_curation` | Stage 产物根目录 |
| `VLA_CATALOG_DB` | `.local-run/catalog.sqlite3`（启动脚本） | SQLite Catalog |
| `VLA_VIDEO_PROXY_ROOT` | `.local-run/video_proxy` | Episode 视频代理缓存 |
| `VLA_THUMBNAIL_ROOT` | `$VLA_CURATION_ROOT/_catalog/thumbnails` | 搜索卡片封面缓存 |
| `VLA_HOST` | `127.0.0.1` | 开发服务监听地址 |
| `VLA_API_PORT` | `8000` | API 端口 |
| `VLA_WEB_PORT` | `5173` | Web 端口 |
| `VLA_PROXY_WORKERS` | `1` | 视频代理并发数 |
| `VLA_PROXY_ENCODER_THREADS` | `2` | 单个代理编码线程数 |
| `VLA_PROXY_CACHE_BYTES` | `20 GiB` | 视频代理缓存上限 |

SQLite 和视频代理建议放在本地磁盘，不要放在 OSSFS/FUSE 挂载中。

## 手动开发

后端：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

VLA_DATA_ROOT=/path/to/public_datasets_staging \
  .venv/bin/uvicorn vla_platform.api:app \
  --host 127.0.0.1 --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev
```

Vite 会将 `/api` 转发到 `http://localhost:8000`。

## 测试与构建

```bash
.venv/bin/pytest vla_platform/tests
npm --prefix frontend run build
```

## Docker Compose

先构建前端静态资源，再启动 API、Web、Redis、PostgreSQL 和 CPU Worker：

```bash
npm --prefix frontend install
npm --prefix frontend run build
docker compose up --build -d
```

默认 Compose 文件绑定 `/mnt/data/embodied_datasets/public_datasets_staging`。在其他机器部署前，请调整 `docker-compose.yml` 中的 volume 路径和访问策略。

## 项目结构

```text
vla_platform/                 FastAPI、Catalog、搜索索引和异步任务
vla_platform/tests/           平台后端测试
frontend/                     React + TypeScript + Ant Design 前端
scripts/local_deploy.sh       本地一键启动和服务管理
deploy/nginx.conf             前端静态服务和 API 反向代理
docs/PLATFORM.md              API、索引和 Stage 产物说明
docs/DATA_CURATION.md         数据清洗配置、运行、验收和平台刷新指南
embodied_datasets/scripts/    数据转换、Stage 处理和检查工具
```

数据清洗操作见 [`docs/DATA_CURATION.md`](docs/DATA_CURATION.md)，平台接口与数据约定见 [`docs/PLATFORM.md`](docs/PLATFORM.md)。数据处理实现和项目演进记录见 [`docs/PROJECT_BUILD_RETROSPECTIVE.md`](docs/PROJECT_BUILD_RETROSPECTIVE.md)。

## 数据与安全边界

- 默认只读取指定数据根目录内的文件。
- API 返回元数据、曲线采样和支持 HTTP Range 的视频响应，不主动上传数据。
- `materialize=true` 的导出才会显式复制数据。
- 开发模式没有内置多用户认证，不应直接绑定公网地址；跨机器访问请使用 SSH 隧道或部署在受控网络并增加认证网关。

## License

见 [`LICENSE`](LICENSE)。
