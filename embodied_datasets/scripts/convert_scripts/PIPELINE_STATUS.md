# convert_scripts 流水线现状

写于 2026-08-13，在 `embodied_datasets/scripts/convert_scripts/` 下搭建模块化多格式转换框架
的过程中记录。部署到服务器前请先读这份文档——它记录的是在本机（Windows，conda 环境
`vla_data_pipline`）上**实际验证过**的内容，以及哪些地方还需要拿真实数据/真实依赖在服务器
上重新核实。

## 现在有什么

```
convert_scripts/
├── dump_dataset_schema.py         # 格式探测：hdf5 / rlds / raw_image_json / unknown
├── convert_mobile_aloha_to_lerobot.py  # 行为不变；ALOHA双臂+底盘的动作逻辑仍留在这里
├── convert_core/                  # 格式无关部分：plan数据类、pydantic配置、LeRobot写入器
│   ├── errors.py, path_utils.py, hdf5_common.py, episode_spec.py, dataset_config.py, lerobot_writer.py
├── readers/                       # 每种源格式一个文件，按格式名分发
│   ├── base.py, hdf5_reader.py, rlds_reader.py, raw_image_json_reader.py, registry.py
├── configs/                       # 每个dataset_uid一份yaml；附带三份带注释的模板
│   ├── example_hdf5.yaml, example_rlds.yaml, example_raw_image_json.yaml
├── convert_dataset.py             # 统一CLI：--config/--all, --dry-run, --overwrite/--skip-existing
└── tests/                         # 原有25个 + 新增36个，全仓库共314个，全部通过
```

`convert_mobile_aloha_to_lerobot.py` 里格式无关的通用 HDF5 逻辑（相机发现、FPS 推断、
数值矩阵校验、RGB 解码）已经提炼进 `convert_core.hdf5_common`，现在以原来的私有名字重新
导入回去——该文件里每一处调用点都没有变化，原有25个测试逐字通过。真正属于 ALOHA 专属的
部分保留了下来：双臂/移动底盘 14+2 动作布局的歧义处理，通过 `/base_action` 交叉校验。
`readers/hdf5_reader.py` 是给*其他所有* HDF5 数据集用的配置驱动 reader——单一无歧义的动作
向量，完全由 `configs/<uid>.yaml` 驱动，不需要新写 Python 文件。

## 已验证 vs. 未验证

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

**hdf5（readers/hdf5_reader.py + convert_dataset.py）：端到端全链路已验证**，包括真实的
视频编码。`tests/test_convert_dataset.py` 用一份合成 HDF5 fixture 跑通完整流水线——
`LeRobotDataset.create` -> `add_frame` -> `save_episode`（真实 ffmpeg/SVT-AV1 视频编码）
-> `finalize` -> 重新打开并校验 -> 原子发布——最后还用一个真实的 `LeRobotDataset` 把结果
读回来验证。这一点值得特别指出，因为*原有*的 `convert_mobile_aloha_to_lerobot.py` 测试套件
从来没做过这件事（它的12个测试只跑了 `inspect_dataset`/`_episode_arrays`/`_read_rgb_frame`，
从没跑过真实的写入路径）——所以这是本仓库第一次真正确认写入侧的流水线在本机上确实能
产出一份合法的 LeRobot v3.0 数据集。

**raw_image_json（readers/raw_image_json_reader.py）：reader 内部逻辑已充分验证，但文件
约定本身还没拿真实数据核对过。** 本仓库里没有现成的 raw_image_json 数据集可供核对——
`readers/raw_image_json_reader.py` 文档字符串和 `configs/example_raw_image_json.yaml` 里
"每个episode一个文件夹 + `metadata.json` + `frames[]`" 的结构，是这次实现自己发明的约定，
参照的是常见小型开源机器人学习数据集的大致样子。**在转换真实的 raw_image_json 数据集之前**，
先探测它的真实目录结构（如果需要可以进一步扩展 `dump_dataset_schema.py` 的 raw_image_json
探测器），把约定改成匹配真实数据的样子——不要反过来把真实数据硬凑成这个猜测的形状。

**rlds（readers/rlds_reader.py）：只验证了格式无关的辅助逻辑部分。** 本环境里任何地方都
没装 `tensorflow_datasets`（连项目自己的 `vla_data_pipline` conda 环境里也没有——已用
`pip show tensorflow-datasets` 确认），所以 `tests/test_rlds_reader.py` 只用普通
dict/numpy 数组覆盖了 `_get_by_path`/`_resolve_instruction`/`_validate_vector_step`/
`_camera_shape`/`_materialize_steps`，外加确认 `RldsReader.build_plan` 在库缺失时会抛出
清晰的 `RuntimeError`（这是真实执行过的断言，不是猜测）。真正的
`tfds.builder_from_directory(...).as_dataset(...)` 调用从没跑过一次。**在服务器上拿真实
RLDS 数据集验证之前，需要：**
1. `pip install tensorflow-cpu==2.15.0 tensorflow-datasets==4.9.9`（版本号取自
   `dump_dataset_schema.py` 里已经用的那两个）。
2. 先跑一遍 `dump_dataset_schema.py --dataset-root <path>` 拿到真实的带类型的 feature
   规格，据此填写 `configs/<uid>.yaml` 里 `vector_fields`/`cameras` 的 source_key 路径。
3. `convert_dataset.py --config ... --dry-run`，仔细看打印出来的 plan，再考虑真正跑一次。
4. 一个已知但还没修的扩展性局限：`RldsReader.build_plan` 会把请求的整个 split 一次性
   解码完，并把每个 episode 的 steps 全部缓存在内存里（见该模块的文档字符串），因为 RLDS
   不解码就拿不到每个 episode 的帧数。数据集不大不小时没问题；如果要处理几百 GB 的
   shard，就需要把 `iter_frames` 改成为每个 episode 重新打开一个独立的子迭代器，而不是
   重放缓存的 steps。

## 值得记住的环境相关发现

- **lerobot 版本不一致，已解决但部署到服务器时值得再确认一次。**
  `process_scripts/lerobot_io.py` 的文档字符串写的是针对 `lerobot==0.4.4` 验证过，但
  `requirements.txt` 锁定的是 `lerobot[dataset]==0.6.0`，`vla_data_pipline` 环境里实际
  装的也是这个版本（已用 `pip show lerobot` 确认）。已经直接对照安装的 0.6.0 源码检查过
  `LeRobotDataset.create`/`add_frame`/`save_episode`/`finalize` 的签名——和本仓库里
  （新旧代码）所有写入器假设的一致。如果服务器环境最终用的是不同的 lerobot 版本，请重新
  确认这一点。
- **ffmpeg 在这个 conda 环境里能找到，但不在裸机 Windows 的 PATH 里。**
  `shutil.which("ffmpeg")` 在 conda 环境外什么都找不到，但激活环境后
  `C:\...\envs\vla_data_pipline\Library\bin\ffmpeg.EXE` 确实存在且能正常工作（是 conda
  间接带进来的，不是通过 `requirements.txt`——正如 `requirements.txt` 里的注释所说，它
  确实是一个系统依赖，不能靠 pip 装）。在 Ubuntu/Debian 服务器上（该注释里提到的实际
  部署目标），跑任何视频编码路径之前先确认已经 `apt install ffmpeg`。
- **本环境里任何地方都没有 `tensorflow_datasets`**，包括项目自己的 conda 环境——只有
  `dump_dataset_schema.py` 里那条软导入的 fallback 路径在这里真正跑过。详见上面 rlds
  那一节。
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

Mobile ALOHA 自己那个双臂/移动底盘数据集继续直接用 `convert_mobile_aloha_to_lerobot.py`
（它自己的CLI，没有变化）——它不应该、也不会变成一条 `configs/*.yaml` 记录，因为它的
动作布局歧义是领域逻辑，任何通用配置都表达不了。

## 待办事项

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
