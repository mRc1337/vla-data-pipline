# vla_data_pipeline

VLA（视觉-语言-动作）机器人操作数据集的统一注册、下载、格式转换、清洗对齐流水线。

## 项目结构

```
vla_data_pipeline/
├── embodied_datasets/     # 全部实际工作：数据集注册表、onboarding、转换、清洗流水线
│   └── README.md          # 详细文档：目录结构、字段参考、跨本体统一表示层规范、如何 onboard 新数据集
├── tests/                  # convert_scripts（注册表/onboarding工具）的测试套件
├── requirements.txt        # 根目录 Python 环境依赖（3.9 兼容）
└── pyproject.toml          # pytest 配置 + 项目元数据
```

详细文档见 [`embodied_datasets/README.md`](embodied_datasets/README.md)——包括完整的数据集注册表字段参考、`process_scripts` 清洗流水线的跨本体统一表示层规范、以及 onboarding 新数据集的流程。

## 两套独立的 Python 环境

这个仓库有两套互不依赖的 Python 环境，分别服务不同用途：

### 1. 根目录环境（Python 3.9+，注册表/onboarding 工具）

```bash
python3 -m venv .venv   # 或直接用系统 Python
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest       # 跑 convert_scripts 的测试套件
```

### 2. `process_scripts` 独立环境（Python 3.10+，清洗流水线）

`process_scripts`（LeRobot 数据清洗对齐流水线）依赖 `lerobot` 包，需要 Python ≥3.10，跟根目录环境隔离：

```bash
# Ubuntu 系统默认仓库通常没有 3.11，需要先装：
# sudo apt update && sudo apt install python3.11 python3.11-venv
cd embodied_datasets/process_scripts
python3.11 -m venv .venv-process
source .venv-process/bin/activate
pip install -r requirements.txt
pytest                  # 跑 process_scripts 的测试套件
```

**系统依赖**：处理真实（非合成）视频数据需要系统装 `ffmpeg`（不是 pip 依赖）：
- macOS：`brew install ffmpeg`
- Ubuntu/Debian：`sudo apt update && sudo apt install ffmpeg`

## License

内部专有代码，见 [`LICENSE`](LICENSE)。
