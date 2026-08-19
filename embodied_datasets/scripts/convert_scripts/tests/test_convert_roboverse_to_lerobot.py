from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

import convert_roboverse_to_lerobot as converter
import evaluate_roboverse_conversion as evaluator
from convert_core.checkpoint import exclusive_resume_lock, resume_paths
from convert_core.errors import ConversionError
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from readers.roboverse_v2_reader import inspect_collection


def _episode(
    offset: float = 0.0,
    *,
    states: int | None = 2,
    task: str = "pick the test object",
) -> dict:
    return {
        "actions": [
            {"dof_pos_target": {"joint_a": offset + index, "joint_b": offset + index + 1.0}}
            for index in range(2)
        ],
        "states": None
        if states is None
        else [
            {
                "robot": {
                    "pos": [offset + index, 0.0, 0.0],
                    "rot": [1.0, 0.0, 0.0, 0.0],
                    "dof_pos": {"joint_a": offset + index, "joint_b": offset + index + 1.0},
                }
            }
            for index in range(states)
        ],
        "init_state": {
            "robot": {
                "pos": np.asarray([offset, 0, 0], dtype=np.float32),
                "rot": np.asarray([1, 0, 0, 0], dtype=np.float32),
                "dof_pos": {"joint_a": offset, "joint_b": offset + 1.0},
            }
        },
        "extra": {"task_name": task, "seed": np.int32(offset)},
    }


def _write_source(raw_root: Path, filename: str = "robot_v2.pkl", episodes: list[dict] | None = None) -> Path:
    path = raw_root / "roboverse" / "trajs" / "fixture" / "task" / "v2" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"robot": episodes or [_episode()]}, handle)
    return path


def _args(raw_root: Path, staging_root: Path, *extra: str) -> list[str]:
    return [
        "--raw-root",
        str(raw_root),
        "--staging-root",
        str(staging_root),
        "--dataset-uid",
        "roboverse_fixture",
        "--source-path",
        "trajs/fixture/task/v2/robot_v2.pkl",
        "--eta-interval-seconds",
        "0.01",
        *extra,
    ]


def test_inspect_only_reads_real_schema_without_creating_staging(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    report = tmp_path / "inspection" / "summary.json"
    _write_source(raw)

    assert converter.main(
        _args(raw, staging, "--inspect-only", "--inspection-report", str(report))
    ) == 0

    assert not staging.exists()
    summary = converter.read_json_object(report, "inspection summary")
    assert summary["feature_inventory"][0]["finite_min"]
    output = capsys.readouterr().out
    assert '"source_episodes": 1' in output
    assert '"trajectory_images_or_videos_available": false' in output


def test_inspection_summary_reports_lengths_splits_and_numeric_ranges(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_source(raw)

    summary = converter._summary(inspect_collection(raw / "roboverse"))

    assert summary["source_action_frames"] == 2
    assert summary["source_state_frames"] == 2
    suite = summary["suites"]["fixture"]
    assert suite["source_episodes"] == 1
    assert suite["source_action_length_min"] == 2
    assert suite["source_action_length_max"] == 2
    assert suite["aligned_episodes"] == 1
    assert suite["source_splits"] == {"unspecified": 1}
    action = next(row for row in summary["feature_inventory"] if row["target_key"] == "action")
    assert action["finite_min"] == [0.0, 1.0]
    assert action["finite_max"] == [1.0, 2.0]
    assert action["frames"] == 2


def test_conversion_refuses_to_invent_physical_fps(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw)

    assert converter.main(_args(raw, staging)) == 1

    assert "no timestamps/FPS" in capsys.readouterr().err
    assert not staging.exists()


def test_conversion_rejects_ossfs_staging_root(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    _write_source(raw)

    assert converter.main(
        _args(
            raw,
            Path("/mnt/data/roboverse-converter-must-not-write-here"),
            "--allow-ordinal-timebase",
        )
    ) == 1

    assert "server-local storage" in capsys.readouterr().err


def test_real_lerobot_conversion_preserves_float64_values_and_static_sidecar(tmp_path: Path):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw)

    assert converter.main(_args(raw, staging, "--allow-ordinal-timebase")) == 0

    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    manifest = converter.read_json_object(output / "collection_manifest.json", "manifest")
    assert manifest["source_episode_count"] == 1
    assert manifest["output_frame_count"] == 2
    assert manifest["timebase"]["ordinal_timebase_enabled"] is True
    part = manifest["parts"][0]
    action_statistics = next(
        row for row in part["source_numeric_statistics"] if row["target_key"] == "action"
    )
    assert action_statistics["finite_min"] == [0.0, 1.0]
    assert action_statistics["finite_max"] == [1.0, 2.0]
    dataset = LeRobotDataset(repo_id=f"roboverse_fixture/{part['part_id']}", root=output / part["path"])
    assert dataset.num_episodes == 1
    assert len(dataset) == 2
    assert dataset.meta.features["action"]["dtype"] == "float64"
    assert dataset.meta.features["action"]["names"] == ["joint_a", "joint_b"]
    assert (output / "source_episodes.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert not list(output.rglob("*.mp4"))
    assert not list(output.rglob(converter.LEROBOT_CACHE_DIR))


def test_independent_evaluator_checks_source_storage_and_lerobot_samples(tmp_path: Path):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw)
    assert converter.main(_args(raw, staging, "--allow-ordinal-timebase")) == 0

    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    report = evaluator.evaluate_collection(output, raw / "roboverse")

    assert report["passed"] is True
    assert report["provenance_links_exact"] is True
    assert report["parts"][0]["sample_indices"] == [0, 1]
    assert report["parts"][0]["parquet_storage_dtype_and_values_preserved"] is True
    assert report["parts"][0]["lerobot_values_exact_at_samples"] is True


def test_conversion_preserves_stable_mixed_component_dtypes(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    episode = _episode(states=None)
    episode["actions"] = [
        {
            "dof_pos_target": {
                "joint_a": np.float32(index + 0.25),
                "joint_b": np.int64(index),
            }
        }
        for index in range(2)
    ]
    _write_source(raw, episodes=[episode])

    assert converter.main(_args(raw, staging, "--allow-ordinal-timebase")) == 0

    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    collection = converter.read_json_object(output / "collection_manifest.json", "manifest")
    part_root = output / collection["parts"][0]["path"]
    manifest = converter.read_json_object(part_root / "conversion_manifest.json", "manifest")
    parquet = next((part_root / "data").rglob("*.parquet"))
    schema = pq.read_schema(parquet)
    assert [(row["target_key"], row["dtype"]) for row in manifest["field_mapping"]] == [
        ("action.joint_a", "float32"),
        ("action.joint_b", "int64"),
    ]
    assert schema.field("action.joint_a").type == pa.float32()
    assert schema.field("action.joint_b").type == pa.int64()
    dataset = LeRobotDataset(repo_id="roboverse_fixture/mixed", root=part_root)
    assert dataset.meta.features["action.joint_a"]["shape"] == (1,)
    assert dataset.meta.features["action.joint_b"]["shape"] == (1,)
    assert dataset[1]["action.joint_a"].item() == pytest.approx(1.25)
    assert dataset[1]["action.joint_b"].item() == 1


def test_conversion_losslessly_promotes_dynamic_integer_component_with_provenance(
    tmp_path: Path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    episode = _episode(states=None)
    episode["actions"] = [
        {
            "dof_pos_target": {
                "arm": np.float32(0.25),
                "finger": np.float32(1.0),
            }
        },
        {
            "dof_pos_target": {
                "arm": np.float32(0.5),
                "finger": 0,
            }
        },
        {
            "dof_pos_target": {
                "arm": np.float32(0.75),
                "finger": 1,
            }
        },
    ]
    _write_source(raw, episodes=[episode])

    assert converter.main(
        _args(
            raw,
            staging,
            "--allow-ordinal-timebase",
            "--allow-lossless-dtype-promotion",
        )
    ) == 0

    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    collection = converter.read_json_object(output / "collection_manifest.json", "manifest")
    assert len(collection["semantic_changes"]["dtype_casts"]) == 1
    part_root = output / collection["parts"][0]["path"]
    manifest = converter.read_json_object(part_root / "conversion_manifest.json", "manifest")
    mapping = manifest["field_mapping"][0]
    assert mapping["target_key"] == "action"
    assert mapping["dtype"] == "float32"
    assert mapping["source_dtype_options"] == [
        ["float32"],
        ["float32", "int64"],
    ]
    cast = manifest["dtype_casts"][0]
    assert cast["source_component"] == "finger"
    assert cast["runs"] == [
        {"start": 0, "end_exclusive": 1, "dtype": "float32"},
        {"start": 1, "end_exclusive": 3, "dtype": "int64"},
    ]
    assert cast["numeric_values_exact"] is True
    assert cast["episode_boundary_preserved"] is True
    assert cast["lossy"] is False
    parquet = next((part_root / "data").rglob("*.parquet"))
    assert pq.read_schema(parquet).field("action").type.value_type == pa.float32()
    report = evaluator.evaluate_collection(output, raw / "roboverse")
    assert report["passed"] is True
    assert report["parts"][0]["lossless_dtype_promotions_verified"] == 1


def test_conversion_preserves_named_singleton_array_axes(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    episode = _episode()
    for index, state in enumerate(episode["states"]):
        state["robot"]["dof_pos"] = {
            "joint_a": np.asarray([index], dtype=np.float64),
            "joint_b": np.asarray([index + 1], dtype=np.float64),
        }
    _write_source(raw, episodes=[episode])

    assert converter.main(_args(raw, staging, "--allow-ordinal-timebase")) == 0

    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    collection = converter.read_json_object(output / "collection_manifest.json", "manifest")
    part_root = output / collection["parts"][0]["path"]
    manifest = converter.read_json_object(part_root / "conversion_manifest.json", "manifest")
    mapping = next(
        row for row in manifest["field_mapping"]
        if row["target_key"] == "observation.state.robot.dof_pos"
    )
    assert mapping["shape"] == [2, 1]
    assert mapping["source_component_shapes"] == [[1], [1]]
    parquet = next((part_root / "data").rglob("*.parquet"))
    arrow_type = pq.read_schema(parquet).field("observation.state.robot.dof_pos").type
    assert isinstance(arrow_type, pa.ExtensionType)
    assert tuple(arrow_type.shape) == (2, 1)
    stats = converter.read_json_object(part_root / "meta" / "stats.json", "stats")
    assert stats["observation.state.robot.dof_pos"]["min"] == [0.0]
    assert stats["observation.state.robot.dof_pos"]["max"] == [2.0]
    report = evaluator.evaluate_collection(output, raw / "roboverse")
    assert report["passed"] is True
    assert report["parts"][0]["parquet_storage_dtype_and_values_preserved"] is True


def test_resume_reuses_verified_part_and_cleans_checkpoint_after_publish(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw, episodes=[_episode(), _episode(10.0, states=None)])
    args = _args(raw, staging, "--allow-ordinal-timebase", "--resume")
    original = converter._convert_part
    calls = 0

    def interrupt_on_second(*call_args, **call_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*call_args, **call_kwargs)

    monkeypatch.setattr(converter, "_convert_part", interrupt_on_second)
    try:
        converter.main(args)
    except KeyboardInterrupt:
        pass

    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    checkpoint, state, _ = resume_paths(output)
    assert checkpoint.is_dir()
    assert len(list((state / converter.MARKERS_DIR).glob("*.json"))) == 1

    monkeypatch.setattr(converter, "_convert_part", original)
    assert converter.main(args) == 0
    assert output.is_dir()
    assert not checkpoint.exists()
    assert not state.exists()


def test_resume_cleans_state_left_after_atomic_publication(tmp_path: Path):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw)
    base_args = _args(raw, staging, "--allow-ordinal-timebase")
    assert converter.main(base_args) == 0
    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    checkpoint, state, _ = resume_paths(output)
    state.mkdir(parents=True)
    (state / converter.STATE_FILE).write_text("{}\n", encoding="utf-8")

    assert converter.main([*base_args, "--resume"]) == 0

    assert output.is_dir()
    assert not checkpoint.exists()
    assert not state.exists()


def test_resume_rejects_changed_source_fingerprint(tmp_path: Path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    source = _write_source(raw, episodes=[_episode(), _episode(10.0, states=None)])
    args = _args(raw, staging, "--allow-ordinal-timebase", "--resume")
    original = converter._convert_part
    calls = 0

    def interrupt_on_second(*call_args, **call_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*call_args, **call_kwargs)

    monkeypatch.setattr(converter, "_convert_part", interrupt_on_second)
    try:
        converter.main(args)
    except KeyboardInterrupt:
        pass
    monkeypatch.setattr(converter, "_convert_part", original)
    with source.open("wb") as handle:
        pickle.dump({"robot": [_episode(), _episode(20.0, states=None)]}, handle)

    assert converter.main(args) == 1
    assert "fingerprint mismatch" in capsys.readouterr().err


def test_part_manifest_records_lerobot_actual_task_indices(tmp_path: Path):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(
        raw,
        episodes=[
            _episode(task="z task encountered first"),
            _episode(10.0, task="a task encountered second"),
        ],
    )

    assert converter.main(_args(raw, staging, "--allow-ordinal-timebase")) == 0

    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    collection = converter.read_json_object(output / "collection_manifest.json", "manifest")
    part = collection["parts"][0]
    part_root = output / part["path"]
    manifest = converter.read_json_object(part_root / "conversion_manifest.json", "manifest")
    dataset = LeRobotDataset(repo_id="roboverse_fixture/part", root=part_root)
    actual = {
        str(int(row["task_index"])): str(task)
        for task, row in dataset.meta.tasks.iterrows()
    }
    assert manifest["task_index_mapping"] == actual
    assert {row["source_task_origin"] for row in manifest["episodes"]} == {
        "episode.extra.task_name"
    }


def test_resume_rebuilds_corrupt_verified_part(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw, episodes=[_episode(), _episode(10.0, states=None)])
    args = _args(raw, staging, "--allow-ordinal-timebase", "--resume")
    original = converter._convert_part
    calls = 0

    def interrupt_on_second(*call_args, **call_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*call_args, **call_kwargs)

    monkeypatch.setattr(converter, "_convert_part", interrupt_on_second)
    with pytest.raises(KeyboardInterrupt):
        converter.main(args)
    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    checkpoint, state, _ = resume_paths(output)
    completed_id = next((state / converter.MARKERS_DIR).glob("*.json")).stem
    (checkpoint / "parts" / completed_id / "conversion_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )

    rebuilt: list[str] = []

    def record_part(part, *call_args, **kwargs):
        rebuilt.append(part.part_id)
        return original(part, *call_args, **kwargs)

    monkeypatch.setattr(converter, "_convert_part", record_part)
    assert converter.main(args) == 0
    assert completed_id in rebuilt


def test_resume_rejects_timebase_configuration_change(tmp_path: Path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw, episodes=[_episode(), _episode(10.0, states=None)])
    args = _args(raw, staging, "--allow-ordinal-timebase", "--resume")
    original = converter._convert_part
    calls = 0

    def interrupt_on_second(*call_args, **call_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*call_args, **call_kwargs)

    monkeypatch.setattr(converter, "_convert_part", interrupt_on_second)
    with pytest.raises(KeyboardInterrupt):
        converter.main(args)
    monkeypatch.setattr(converter, "_convert_part", original)

    changed = _args(raw, staging, "--fps-override", "fixture=10", "--resume")
    assert converter.main(changed) == 1
    assert "timebase" in capsys.readouterr().err


def test_concurrent_resume_lock_is_rejected(tmp_path: Path):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw)
    collection = inspect_collection(raw / "roboverse")
    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    _, _, lock = resume_paths(output)

    with exclusive_resume_lock(lock):
        with pytest.raises(ConversionError, match="another resume process is using"):
            converter.convert_collection(
                collection,
                output_path=output,
                dataset_uid="roboverse_fixture",
                resume=True,
                overwrite=False,
                skip_existing=False,
                fps_overrides={},
                allow_ordinal=True,
                eta_interval_seconds=0.01,
                workers=1,
                selection={},
            )


def test_reported_source_sidecar_issue_requires_explicit_acknowledgement(
    tmp_path: Path, capsys
):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    source = _write_source(raw)
    (source.parents[2] / "initial_state_v2.json").touch()
    args = _args(raw, staging, "--allow-ordinal-timebase")
    source_arg = args.index("--source-path")
    del args[source_arg : source_arg + 2]

    assert converter.main(args) == 1
    assert "sidecar issue" in capsys.readouterr().err
    assert not staging.exists()

    assert converter.main([*args, "--allow-source-sidecar-issues"]) == 0
    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    manifest = converter.read_json_object(output / "collection_manifest.json", "manifest")
    assert manifest["source_issues"][0]["kind"] == "empty_source_sidecar"


def test_parallel_part_workers_publish_deterministic_collection(tmp_path: Path):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw, episodes=[_episode(), _episode(10.0, states=None)])

    assert converter.main(
        _args(raw, staging, "--allow-ordinal-timebase", "--workers", "2")
    ) == 0

    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    manifest = converter.read_json_object(output / "collection_manifest.json", "manifest")
    assert len(manifest["parts"]) == 2
    assert manifest["output_episode_count"] == 2
    assert manifest["output_frame_count"] == 4
    assert not list(output.rglob(converter.LEROBOT_CACHE_DIR))


def test_broken_trajectory_is_reported_and_cannot_be_acknowledged(
    tmp_path: Path, capsys
):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    source = _write_source(raw)
    (source.parent / "broken_v2.pkl").touch()
    args = _args(raw, staging, "--inspect-only")
    source_arg = args.index("--source-path")
    del args[source_arg : source_arg + 2]

    assert converter.main(args) == 1
    captured = capsys.readouterr()
    assert '"kind": "unreadable_or_invalid_trajectory"' in captured.out
    assert "blocking trajectory issue" in captured.err

    conversion_args = [
        value for value in args if value != "--inspect-only"
    ] + ["--allow-ordinal-timebase", "--allow-source-sidecar-issues"]
    assert converter.main(conversion_args) == 1
    assert "cannot be acknowledged or skipped" in capsys.readouterr().err
    assert not staging.exists()


def test_blocking_inspection_still_writes_complete_report(tmp_path: Path):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    source = _write_source(raw)
    broken = source.parent / "broken_v2.pkl"
    broken.touch()
    report = tmp_path / "inspection" / "summary.json"
    args = _args(raw, staging, "--inspect-only", "--inspection-report", str(report))
    source_arg = args.index("--source-path")
    del args[source_arg : source_arg + 2]

    assert converter.main(args) == 1

    summary = converter.read_json_object(report, "inspection report")
    assert summary["source_files"] == 2
    assert summary["source_episodes"] == 1
    assert summary["source_issues"][0]["blocking"] is True


def test_skip_existing_rejects_corrupt_source_provenance_index(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw)
    assert converter.main(_args(raw, staging, "--allow-ordinal-timebase")) == 0
    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    (output / "source_episodes.jsonl").write_text("{}\n", encoding="utf-8")

    assert converter.main(
        _args(raw, staging, "--allow-ordinal-timebase", "--skip-existing")
    ) == 1
    assert "source episode index does not match" in capsys.readouterr().err


def test_skip_existing_rejects_corrupt_full_source_statistics(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    staging = tmp_path / "staging"
    _write_source(raw)
    assert converter.main(_args(raw, staging, "--allow-ordinal-timebase")) == 0
    output = staging / "lerobot_v3_0" / "roboverse_fixture"
    collection = converter.read_json_object(output / "collection_manifest.json", "manifest")
    part_root = output / collection["parts"][0]["path"]
    statistics = converter.read_json_object(part_root / "meta" / "stats.json", "statistics")
    statistics["action"]["min"][0] = -999.0
    converter.atomic_write_json(part_root / "meta" / "stats.json", statistics)

    assert converter.main(
        _args(raw, staging, "--allow-ordinal-timebase", "--skip-existing")
    ) == 1
    assert "stats.json range" in capsys.readouterr().err


def test_overwrite_publication_restores_previous_output_on_rename_failure(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "output"
    temporary = tmp_path / "temporary"
    output.mkdir()
    temporary.mkdir()
    (output / "sentinel.txt").write_text("previous", encoding="utf-8")
    (temporary / "sentinel.txt").write_text("replacement", encoding="utf-8")
    original_rename = Path.rename

    def fail_publication(self: Path, target: Path):
        if self == temporary and target == output:
            raise OSError("injected publication failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_publication)

    with pytest.raises(OSError, match="injected publication failure"):
        converter.publish_temporary_output(temporary, output, overwrite=True)

    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "previous"
    assert temporary.is_dir()
