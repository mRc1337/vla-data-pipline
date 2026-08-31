from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import convert_robocoin_to_lerobot as converter
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan, VectorFeatureSpec
from convert_core.errors import ConversionError
from convert_core.parallel import ParallelWorkUnit, verified_marker_path
import readers.robocoin_reader as robocoin_reader
from readers.robocoin_reader import build_robocoin_catalog
from readers.robogene_reader import RobogeneEpisodeSource, RobogenePartition


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _catalog_task(root: Path, name: str, lengths: list[int], *, width: int = 3) -> None:
    task = root / name
    (task / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "fixture",
        "fps": 30,
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [width], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (task / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(task / "meta" / "tasks.jsonl", [{"task_index": 0, "task": f"do {name}"}])
    _write_jsonl(
        task / "meta" / "episodes.jsonl",
        [
            {"episode_index": index, "tasks": [f"do {name}"], "length": length}
            for index, length in enumerate(lengths)
        ],
    )


def test_catalog_is_ordered_lightweight_and_limit_aware(tmp_path: Path):
    _catalog_task(tmp_path, "z_task", [5, 7])
    _catalog_task(tmp_path, "A_task", [3, 11])

    catalog = build_robocoin_catalog(tmp_path, limit_episodes_per_task=1)

    assert [task.key for task in catalog.tasks] == ["A_task", "z_task"]
    assert [task.total_episodes for task in catalog.tasks] == [1, 1]
    assert [task.total_frames for task in catalog.tasks] == [3, 5]
    assert [task.root_episode_start for task in catalog.tasks] == [0, 1]
    assert [task.root_frame_start for task in catalog.tasks] == [0, 3]


def test_catalog_partitions_incompatible_feature_shapes(tmp_path: Path):
    _catalog_task(tmp_path, "left", [2], width=3)
    _catalog_task(tmp_path, "right", [2], width=4)

    catalog = build_robocoin_catalog(tmp_path)

    assert catalog.tasks[0].partition != catalog.tasks[1].partition


def test_cached_catalog_restore_does_not_read_source(tmp_path: Path, monkeypatch):
    _catalog_task(tmp_path, "task", [2])
    catalog = build_robocoin_catalog(tmp_path)
    payload = robocoin_reader.catalog_to_payload(catalog)

    def fail_source_read(_path: Path):
        raise AssertionError("resume catalog must not reread source metadata")

    monkeypatch.setattr(robocoin_reader, "_read_json", fail_source_read)
    restored = robocoin_reader.catalog_from_payload(payload)

    assert restored.tasks[0].info == catalog.tasks[0].info
    assert restored.fingerprint_payload == payload


def test_catalog_and_task_plan_require_current_conversion_policy(tmp_path: Path):
    _catalog_task(tmp_path, "task", [2])
    catalog = build_robocoin_catalog(tmp_path)
    payload = robocoin_reader.catalog_to_payload(catalog)

    assert payload["schema_version"] == 2
    assert (
        payload["conversion_policy_version"]
        == robocoin_reader.ROBOCOIN_CONVERSION_POLICY_VERSION
    )
    stale = dict(payload)
    stale["conversion_policy_version"] -= 1
    with pytest.raises(ConversionError, match="unsupported RoboCOIN catalog state"):
        robocoin_reader.catalog_from_payload(stale)


def test_semantic_schema_ignores_top_level_pandas_metadata_only():
    field = pa.field(
        "observation.state",
        pa.list_(pa.field("element", pa.float32(), nullable=False)),
        nullable=False,
        metadata={b"unit": b"rad"},
    )
    first = pa.schema(
        [field],
        metadata={b"pandas": b'{"index_columns":[{"stop":10}]}'},
    )
    second = pa.schema(
        [field],
        metadata={b"pandas": b'{"index_columns":[{"stop":20}]}'},
    )

    assert robocoin_reader._semantic_schema_payload(first) == (
        robocoin_reader._semantic_schema_payload(second)
    )
    assert robocoin_reader._semantic_schema_payload(first) != (
        robocoin_reader._semantic_schema_payload(
            pa.schema([field.with_nullable(True)], metadata=first.metadata)
        )
    )
    assert robocoin_reader._semantic_schema_payload(first) != (
        robocoin_reader._semantic_schema_payload(
            pa.schema(
                [field.with_metadata({b"unit": b"degree"})],
                metadata=first.metadata,
            )
        )
    )


def test_semantic_schema_normalizes_only_allowed_agilex_float_promotion():
    float32_schema = pa.schema(
        [
            pa.field("action", pa.list_(pa.float32())),
            pa.field("observation.state", pa.list_(pa.float32())),
        ]
    )
    float64_schema = pa.schema(
        [
            pa.field("action", pa.list_(pa.float64())),
            pa.field("observation.state", pa.list_(pa.float64())),
        ]
    )

    assert robocoin_reader._semantic_schema_payload(float32_schema) != (
        robocoin_reader._semantic_schema_payload(float64_schema)
    )
    assert robocoin_reader._semantic_schema_payload(
        float32_schema,
        float64_fields=("action", "observation.state"),
    ) == robocoin_reader._semantic_schema_payload(
        float64_schema,
        float64_fields=("action", "observation.state"),
    )


def test_agilex_policy_uses_float64_common_supertype_without_mutating_source():
    info = {
        "robot_type": "aloha",
        "features": {
            "action": {"dtype": "float32", "shape": [26]},
            "observation.state": {"dtype": "float32", "shape": [26]},
            "gripper_open_scale_state": {"dtype": "float32", "shape": [2]},
            "gripper_open_scale_action": {"dtype": "float32", "shape": [2]},
        },
    }

    effective, policy = robocoin_reader._conversion_schema_policy("Agilex", info)

    assert info["features"]["action"]["dtype"] == "float32"
    assert effective["features"]["action"]["dtype"] == "float64"
    assert effective["features"]["observation.state"]["dtype"] == "float64"
    assert effective["features"]["gripper_open_scale_state"]["dtype"] == "float64"
    assert effective["features"]["gripper_open_scale_action"]["dtype"] == "float64"
    assert policy["dtype_resolution"]["action"]["lossy"] is False
    assert policy["version"] == robocoin_reader.ROBOCOIN_CONVERSION_POLICY_VERSION


def test_v21_lists_are_canonicalized_without_value_or_dtype_loss(tmp_path: Path):
    path = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32())),
                "scene_annotation": pa.array([[7], [9]], type=pa.list_(pa.int32())),
            }
        ),
        path,
    )
    plan = DatasetConversionPlan(
        dataset_uid="fixture",
        output_path=tmp_path,
        fps=30,
        measured_fps=30.0,
        robot_type="fixture",
        vector_features=(
            VectorFeatureSpec("observation.state", 2, dtype="float32", shape=(2,)),
            VectorFeatureSpec("scene_annotation", 1, dtype="int32", shape=(1,)),
        ),
        camera_features=(),
        episodes=(EpisodePlan("0", "0", "task", 2),),
    )

    converter._canonicalize_v3_parquet(path, plan)

    table = pq.read_table(path)
    assert table["observation.state"].type == pa.list_(pa.float32(), 2)
    assert table["observation.state"].to_pylist() == [[1.0, 2.0], [3.0, 4.0]]
    assert table["scene_annotation"].type == pa.int32()
    assert table["scene_annotation"].to_pylist() == [7, 9]


def test_v21_lists_allow_lossless_float32_to_float64_promotion(tmp_path: Path):
    path = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table(
            {
                "action": pa.array(
                    [[1.25, 2.5], [3.75, 4.0]], type=pa.list_(pa.float32())
                )
            }
        ),
        path,
    )
    plan = DatasetConversionPlan(
        dataset_uid="fixture",
        output_path=tmp_path,
        fps=30,
        measured_fps=30.0,
        robot_type="aloha",
        vector_features=(
            VectorFeatureSpec("action", 2, dtype="float64", shape=(2,)),
        ),
        camera_features=(),
        episodes=(EpisodePlan("0", "0", "task", 2),),
    )

    converter._canonicalize_v3_parquet(path, plan)

    table = pq.read_table(path)
    assert table["action"].type == pa.list_(pa.float64(), 2)
    assert table["action"].to_pylist() == [[1.25, 2.5], [3.75, 4.0]]

    double_path = tmp_path / "double-episode.parquet"
    double_values = [[1.0000000001, 2.0], [3.0, 4.0000000001]]
    pq.write_table(
        pa.table(
            {"action": pa.array(double_values, type=pa.list_(pa.float64()))}
        ),
        double_path,
    )
    converter._canonicalize_v3_parquet(double_path, plan)
    double_table = pq.read_table(double_path)
    assert double_table["action"].type == pa.list_(pa.float64(), 2)
    assert double_table["action"].to_pylist() == double_values


def test_v21_lists_reject_float64_to_float32_narrowing(tmp_path: Path):
    path = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table(
            {
                "action": pa.array(
                    [[1.0000000001, 2.0]], type=pa.list_(pa.float64())
                )
            }
        ),
        path,
    )
    plan = DatasetConversionPlan(
        dataset_uid="fixture",
        output_path=tmp_path,
        fps=30,
        measured_fps=30.0,
        robot_type="fixture",
        vector_features=(
            VectorFeatureSpec("action", 2, dtype="float32", shape=(2,)),
        ),
        camera_features=(),
        episodes=(EpisodePlan("0", "0", "task", 1),),
    )

    with pytest.raises(ConversionError, match="would narrow double to float"):
        converter._canonicalize_v3_parquet(path, plan)

    assert pq.read_table(path)["action"].type == pa.list_(pa.float64())


def test_cli_has_no_fixed_local_quota_and_defaults_to_measured_worker_count():
    parser = converter.build_parser()
    args = parser.parse_args([])

    assert args.workers == 1
    assert args.encoder_threads_per_worker == 8
    assert not hasattr(args, "max_local_temp_bytes")
    assert args.min_local_free_bytes == converter.DEFAULT_MIN_LOCAL_FREE_BYTES


def test_task_resume_reuses_completed_episode_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = tmp_path / "raw"
    work = tmp_path / "work" / "task"
    lengths = (2, 3)
    sources = []
    episodes = []
    for episode_index, length in enumerate(lengths):
        data_path = raw / f"episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "observation.state": pa.array(
                        [[float(episode_index), float(row)] for row in range(length)],
                        type=pa.list_(pa.float32()),
                    ),
                    "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                    "frame_index": pa.array(range(length), type=pa.int64()),
                    "index": pa.array(range(length), type=pa.int64()),
                    "task_index": pa.array([0] * length, type=pa.int64()),
                    "timestamp": pa.array(
                        [row / 30 for row in range(length)], type=pa.float32()
                    ),
                }
            ),
            data_path,
        )
        instruction = "place the object"
        episodes.append(EpisodePlan(str(episode_index), data_path.name, instruction, length))
        sources.append(
            RobogeneEpisodeSource(
                source_id=f"episode-{episode_index}",
                task_root=raw,
                split="train",
                task_name="fixture-task",
                source_episode_index=episode_index,
                instruction=instruction,
                stats={
                    "stats": {
                        "observation.state": {
                            "min": [0.0, 0.0],
                            "max": [1.0, float(length - 1)],
                            "mean": [0.5, float(length - 1) / 2],
                            "std": [0.5, 0.5],
                            "count": [length, length],
                        }
                    }
                },
                length=length,
                data_path=data_path,
                video_paths=(),
                data_bytes=data_path.stat().st_size,
                video_bytes=0,
            )
        )
    plan = DatasetConversionPlan(
        dataset_uid="fixture",
        output_path=tmp_path / "unused",
        fps=30,
        measured_fps=30.0,
        robot_type="fixture",
        vector_features=(
            VectorFeatureSpec("observation.state", 2, dtype="float32", shape=(2,)),
        ),
        camera_features=(),
        episodes=tuple(episodes),
        extra={
            "schema_policy": {
                "version": robocoin_reader.ROBOCOIN_CONVERSION_POLICY_VERSION,
                "dtype_resolution": {},
            },
            "source_parquet_schema_variants": [
                {"fingerprint": "fixture-source-schema", "episode_count": 2}
            ],
        },
    )
    partition = RobogenePartition(
        name="schema-fixture",
        split="train",
        schema_fingerprint="schema-fixture",
        plan=plan,
        source_info={
            "codebase_version": "v2.1",
            "robot_type": "fixture",
            "fps": 30,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": None,
                }
            },
        },
        episodes=tuple(sources),
        source_files=(),
        empty_tasks=(),
        sample_evidence=(),
    )
    serialized_plan = robocoin_reader.task_plan_to_payload(
        partition,
        task_fingerprint="fixture-fingerprint",
        evidence={"mapping": []},
    )
    assert serialized_plan["schema_version"] == 2
    assert serialized_plan["conversion_policy_version"] == (
        robocoin_reader.ROBOCOIN_CONVERSION_POLICY_VERSION
    )
    restored_partition, restored_fingerprint, _ = (
        robocoin_reader.task_plan_from_payload(serialized_plan)
    )
    assert restored_fingerprint == "fixture-fingerprint"
    assert restored_partition.plan.extra["schema_policy"] == plan.extra["schema_policy"]
    stale_plan = dict(serialized_plan)
    stale_plan["conversion_policy_version"] -= 1
    with pytest.raises(ConversionError, match="unsupported RoboCOIN task plan"):
        robocoin_reader.task_plan_from_payload(stale_plan)
    unit = ParallelWorkUnit(
        index=0,
        key="fixture-task",
        dataset_uid="fixture",
        target_path=str(work),
        episode_start=4,
        episode_end=6,
        frame_start=20,
        frame_end=25,
        task_indices=(9, 9),
        weight=5,
        estimated_memory_bytes=1,
        estimated_temp_bytes=1,
        fingerprint="fixture-fingerprint",
        payload=converter.UnitPayload(partition, tuple(sources)),
    )

    class Guard:
        def check(self, *_args, **_kwargs) -> None:
            return None

    actual_copy = converter._copy_episode
    copied: list[int] = []

    def fail_second_once(source, *, local_index, **kwargs):
        copied.append(local_index)
        if local_index == 1 and copied.count(1) == 1:
            raise RuntimeError("injected episode 2 interruption")
        return actual_copy(source, local_index=local_index, **kwargs)

    monkeypatch.setattr(converter, "_copy_episode", fail_second_once)

    with pytest.raises(RuntimeError, match="episode 2 interruption"):
        converter._build_task_unit(unit, workers=1, guard=Guard())

    first_data = work / "data/chunk-000/file-000.parquet"
    first_checkpoint = work / ".episode_checkpoints/episode-000000.json"
    assert first_data.is_file()
    assert first_checkpoint.is_file()
    assert not (work / ".episode_checkpoints/episode-000001.json").exists()
    assert not verified_marker_path(unit).exists()

    converter._build_task_unit(unit, workers=1, guard=Guard())

    assert copied == [0, 1, 1]
    assert first_data.is_file()
    assert first_checkpoint.is_file()
    assert (work / ".episode_checkpoints/episode-000001.json").is_file()
    assert verified_marker_path(unit).is_file()
    manifest = json.loads((work / "conversion_manifest.json").read_text())
    assert manifest["conversion_policy_version"] == (
        robocoin_reader.ROBOCOIN_CONVERSION_POLICY_VERSION
    )
    assert manifest["schema_policy"] == plan.extra["schema_policy"]
    assert manifest["source_parquet_schema_variants"] == (
        plan.extra["source_parquet_schema_variants"]
    )
