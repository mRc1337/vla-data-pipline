# vla_data_pipeline

VLA（视觉-语言-动作）机器人操作数据集的统一注册、下载、格式转换、清洗对齐流水线。

## 项目结构

```
vla_data_pipeline/
├── embodied_datasets/     # 全部实际工作：数据集注册表、onboarding、转换、清洗流水线
│   └── README.md          # 详细文档：目录结构、数据存储方式、process_scripts 处理流程、跨本体统一表示层规范
├── tests/                  # convert_scripts（注册表/onboarding工具）的测试套件
├── requirements.txt        # 全仓库共用的依赖清单
└── pyproject.toml          # pytest 配置 + 项目元数据
```

详细文档见 [`embodied_datasets/README.md`](embodied_datasets/README.md)——数据存储方式、`process_scripts` 清洗流水线的处理流程与跨本体统一表示层规范。

## 环境搭建

```bash
sudo apt update && sudo apt install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 系统依赖

```bash
sudo apt update && sudo apt install ffmpeg
```

## 测试

```bash
python3 -m pytest                                         # 根目录 tests/
(cd embodied_datasets/scripts/process_scripts && pytest)
(cd embodied_datasets/scripts/shared && pytest)
(cd embodied_datasets/scripts/convert_scripts && pytest)
(cd embodied_datasets/scripts/verify_scripts && pytest)
```

## License

内部专有代码，见 [`LICENSE`](LICENSE)。
