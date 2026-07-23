# vla_data_pipeline

VLA（视觉-语言-动作）机器人操作数据集的统一注册、下载、格式转换、清洗对齐流水线。

## 项目结构

```
vla_data_pipeline/
├── embodied_datasets/     # 全部实际工作：数据集注册表、onboarding、转换、清洗流水线
│   └── README.md          # 详细文档：目录结构、字段参考、跨本体统一表示层规范、如何 onboard 新数据集
├── tests/                  # convert_scripts（注册表/onboarding工具）的测试套件
├── requirements.txt        # 全仓库唯一的 Python 环境依赖
└── pyproject.toml          # pytest 配置 + 项目元数据
```

详细文档见 [`embodied_datasets/README.md`](embodied_datasets/README.md)——包括完整的数据集注册表字段参考、`process_scripts` 清洗流水线的跨本体统一表示层规范、以及 onboarding 新数据集的流程。

## 一套 Python 环境

整个仓库共用一套 venv（Python ≥3.10，`process_scripts` 依赖的 `lerobot`
包要求的下限）：

```bash
# Ubuntu 系统默认仓库通常没有 3.11，需要先装：
# sudo apt update && sudo apt install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**系统依赖**：处理真实（非合成）视频数据需要系统装 `ffmpeg`（不是 pip 依赖）：
- macOS：`brew install ffmpeg`
- Ubuntu/Debian：`sudo apt update && sudo apt install ffmpeg`

仓库里有 5 套各自独立的 pytest 运行（各有自己的 `pytest.ini`/`pyproject.toml`，
互不收集对方的测试），都激活同一个 `.venv` 即可，只是要先 `cd` 到对应目录：

```bash
python3 -m pytest                                    # 根目录 tests/：注册表/onboarding 工具
(cd embodied_datasets/scripts/process_scripts && pytest)  # 清洗流水线
(cd embodied_datasets/scripts/shared && pytest)
(cd embodied_datasets/scripts/convert_scripts && pytest)
(cd embodied_datasets/scripts/verify_scripts && pytest)
```

## License

内部专有代码，见 [`LICENSE`](LICENSE)。
