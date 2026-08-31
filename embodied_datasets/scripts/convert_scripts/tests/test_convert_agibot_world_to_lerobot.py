from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

import convert_core.direct_commit as direct_commit
from convert_agibot_world_to_lerobot import (
    TASK_MARKER_VERSION,
    UnitPayload,
    _fingerprint_without,
    _load_or_create_catalog,
    _read_task_marker,
    _bounded_worker_count,
    _rewrite_generated_columns,
    _run_task,
    _unit,
    _validate_local_unit,
)
from convert_core.checkpoint import atomic_write_json, canonical_fingerprint
from convert_core.direct_commit import commit_verified_unit, committed_marker_path
from convert_core.errors import ConversionError
from convert_core.parallel import ParallelWorkUnit, write_verified_unit_marker
from readers.agibot_world_reader import (
    ArchiveParts,
    EpisodeSource,
    _feature_specs,
    _normalized_video_info,
    build_lightweight_catalog,
    catalog_from_payload,
    catalog_payload,
    extract_members,
    validate_source_records,
)


def _write_tar_bytes(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_lightweight_catalog_is_ordered_and_pairs_depth_patch_data(tmp_path: Path):
    raw = tmp_path / "raw"
    complete = raw / "ImitationLearning" / "SceneB" / "task_0002" / "part.tar.gz"
    incomplete = raw / "ImitationLearning" / "SceneA" / "task_0001" / "part.tar.gz"
    complete.parent.mkdir(parents=True)
    incomplete.parent.mkdir(parents=True)
    complete.write_bytes(b"archive")
    incomplete.touch()

    lite = raw / "simulation" / "scenario" / "g2_swift_picker" / "lite"
    depth = lite.parent / "lite_depth_patch"
    lite.mkdir(parents=True)
    depth.mkdir()
    for path in (lite / "meta.tar.gz.000", lite / "data.tar.gz.000", lite / "videos.tar.gz.000", depth / "meta.tar.gz.000", depth / "videos.tar.gz.000"):
        path.write_bytes(b"part")

    catalog = build_lightweight_catalog(raw)
    assert [task.task_key for task in catalog.tasks] == sorted(task.task_key for task in catalog.tasks)
    empty = next(task for task in catalog.tasks if task.task_key.endswith("task_0001"))
    assert empty.unavailable_files == ("ImitationLearning/SceneA/task_0001/part.tar.gz",)
    patch = next(task for task in catalog.tasks if task.task_key.endswith("lite_depth_patch"))
    assert patch.shards[0].data_from_sibling_lite is True
    assert patch.shards[0].data.paths == (lite / "data.tar.gz.000",)
    assert patch.shards[0].sibling_metadata is not None
    assert patch.shards[0].sibling_metadata.paths == (lite / "meta.tar.gz.000",)
    with pytest.raises(ConversionError, match="empty/incomplete"):
        validate_source_records(empty, raw)
    payload = catalog_payload(catalog, output_uid="uid", selection={"tasks": []})
    restored = catalog_from_payload(payload)
    assert [task.task_key for task in restored.tasks] == [task.task_key for task in catalog.tasks]
    restored_patch = next(task for task in restored.tasks if task.task_key.endswith("lite_depth_patch"))
    assert restored_patch.shards[0].data_from_sibling_lite is True
    assert restored_patch.shards[0].sibling_metadata == patch.shards[0].sibling_metadata
    assert restored_patch.shards[0].source_files == patch.shards[0].source_files
    assert payload["catalog_fingerprint"] == _fingerprint_without(payload, "catalog_fingerprint")


def test_split_archive_reader_crosses_part_boundary(tmp_path: Path):
    payload = _write_tar_bytes({"data/value.bin": b"payload-across-split"})
    midpoint = len(payload) // 2
    first, second = tmp_path / "data.tar.gz.000", tmp_path / "data.tar.gz.001"
    first.write_bytes(payload[:midpoint])
    second.write_bytes(payload[midpoint:])
    destination = tmp_path / "out.bin"
    sizes = extract_members(ArchiveParts("data", (first, second)), {"data/value.bin": destination})
    assert sizes == {"data/value.bin": len(b"payload-across-split")}
    assert destination.read_bytes() == b"payload-across-split"


def test_resume_flag_can_create_then_reuse_first_catalog(tmp_path: Path):
    raw = tmp_path / "raw"
    archive = raw / "ImitationLearning" / "Scene" / "task_0001" / "part.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"catalog-only")
    args = SimpleNamespace(
        resume=True,
        raw_root=raw,
        output_dataset_uid="uid",
        task=[],
        start_task=None,
        max_tasks=None,
        max_shards_per_task=None,
        max_episodes_per_task=None,
    )
    layout = SimpleNamespace(resume=tmp_path / "resume")
    first_payload, first_catalog = _load_or_create_catalog(args, layout)
    second_payload, second_catalog = _load_or_create_catalog(args, layout)
    assert first_payload == second_payload
    assert first_catalog.tasks[0].task_key == second_catalog.tasks[0].task_key


def test_depth_camera_schema_preserves_one_channel_and_depth_flag():
    _vectors, cameras = _feature_specs(
        {
            "features": {
                "observation.images.depth": {
                    "dtype": "video",
                    "shape": [480, 640, 1],
                    "names": ["height", "width", "channel"],
                    "info": {"is_depth_map": True},
                }
            }
        }
    )
    assert (cameras[0].height, cameras[0].width) == (480, 640)
    info = _normalized_video_info(
        {"video_info": {"video.is_depth_map": True}},
        {"height": 480, "width": 640, "fps": 30.0, "codec": "png", "pix_fmt": "gray16be"},
    )
    assert info["source_video.codec"] == "png"
    assert info["source_video.pix_fmt"] == "gray16be"
    assert "video.codec" not in info and "video.pix_fmt" not in info


def test_generated_rebase_preserves_all_payload_columns(tmp_path: Path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "episode.parquet"
    table = pa.table(
        {
            "observation.state": pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32(), 2)),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "episode_index": pa.array([17, 17], type=pa.int64()),
            "index": pa.array([100, 101], type=pa.int64()),
            "task_index": pa.array([9, 9], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.1], type=pa.float32()),
        }
    )
    pq.write_table(table, path)
    source = EpisodeSource("s", "shard", 17, "task", 2, {}, "episode.parquet", (), 1, 0, {}, 42, 1000, 7)
    before = pq.read_table(path)
    _rewrite_generated_columns(path, source)
    after = pq.read_table(path)
    for name in set(before.column_names) - {"episode_index", "index", "task_index"}:
        assert before[name].equals(after[name])
    assert after["episode_index"].to_pylist() == [0, 0]
    assert after["index"].to_pylist() == [0, 1]
    assert after["task_index"].to_pylist() == [0, 0]


def test_marker_repair_rejects_complete_but_not_globalized_unit(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "unit"
    data = root / "data" / "chunk-000" / "file-000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "episode_index": pa.array([0, 0], type=pa.int64()),
                "index": pa.array([0, 1], type=pa.int64()),
                "task_index": pa.array([0, 0], type=pa.int64()),
                "timestamp": pa.array([0.0, 0.1], type=pa.float32()),
            }
        ),
        data,
    )
    meta = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta.parent.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0]}), meta)
    features = {
        name: {"dtype": dtype, "shape": [1], "names": None}
        for name, dtype in {
            "frame_index": "int64",
            "episode_index": "int64",
            "index": "int64",
            "task_index": "int64",
            "timestamp": "float32",
        }.items()
    }
    source = EpisodeSource("s", "shard", 0, "task", 2, {}, "data", (), 1, 0, {}, 5, 20, 3)
    task_plan = {
        "output_dataset_uid": "uid",
        "partition_name": "part",
        "task_key": "task",
        "unit_start": 0,
        "episode_start": 5,
        "fps": 10,
        "robot_type": "robot",
        "features": features,
        "mapping_table": [],
        "fingerprint": "fingerprint",
    }
    unit = ParallelWorkUnit(0, "unit", "uid", str(root), 5, 6, 20, 22, (3,), 2, 1, 1, "fingerprint", UnitPayload(task_plan, source, tmp_path))
    _validate_local_unit(unit, globalized=False)
    with pytest.raises(ConversionError, match="global generated indices"):
        _validate_local_unit(unit)


def test_unit_identity_includes_source_shard(tmp_path: Path):
    plan = {
        "task_key": "task",
        "output_dataset_uid": "uid",
        "unit_start": 0,
        "episode_start": 0,
        "fingerprint": "fingerprint",
    }
    first = EpisodeSource("task/shard-a/episode-000000", "shard-a", 0, "task", 1, {}, "data", (), 1, 0, {}, 0, 0, 0)
    second = EpisodeSource("task/shard-b/episode-000000", "shard-b", 0, "task", 1, {}, "data", (), 1, 0, {}, 1, 1, 0)
    layout = SimpleNamespace(work=tmp_path)
    assert _unit(plan, first, layout=layout, materialized_root=tmp_path).key != _unit(
        plan, second, layout=layout, materialized_root=tmp_path
    ).key


def test_inflight_budget_reduces_workers_and_counts_complete_unit_peaks():
    sources = tuple(
        EpisodeSource(
            f"source-{index}", "shard", index, "task", 1, {}, "data", (),
            data_bytes, video_bytes, {}, index, index, 0,
        )
        for index, (data_bytes, video_bytes) in enumerate(((10, 100), (20, 200), (30, 300)))
    )
    mib = 1024 * 1024
    two_worker_peak = 2 * 64 * mib + 600
    assert _bounded_worker_count(sources, 4, two_worker_peak) == (2, two_worker_peak)
    assert _bounded_worker_count(sources, 4, 3 * 64 * mib + 720) == (
        3,
        3 * 64 * mib + 720,
    )
    with pytest.raises(ConversionError, match="does not fit"):
        _bounded_worker_count(sources, 4, 64 * mib)


def test_whole_task_peak_is_checked_before_output_creation(tmp_path: Path):
    episodes = [
        {
            "source_id": f"source-{index}", "shard_id": "shard",
            "source_episode_index": index, "instruction": "task", "length": 1,
            "stats": {}, "data_member": "data", "video_members": [],
            "data_bytes": data_bytes, "video_bytes": video_bytes, "member_sizes": {},
            "global_episode_index": index, "global_frame_start": index,
            "partition_task_index": 0,
        }
        for index, (data_bytes, video_bytes) in enumerate(((10, 100), (20, 200), (30, 300)))
    ]
    task_plan = {
        "task_key": "task", "partition_name": "part", "episodes": episodes,
        "output_dataset_uid": "uid", "unit_start": 0, "episode_start": 0,
        "fingerprint": "fingerprint",
    }
    layout = SimpleNamespace(
        final=tmp_path / "remote", work=tmp_path / "work", resume=tmp_path / "resume"
    )
    args = SimpleNamespace(workers=2, max_local_inflight_bytes=1024**4)

    class RejectingCapacity:
        checked: tuple[str, int] | None = None

        def check(self, stage: str, *, required_additional_bytes: int = 0):
            self.checked = (stage, required_additional_bytes)
            raise ConversionError("insufficient capacity")

    capacity = RejectingCapacity()
    with pytest.raises(ConversionError, match="insufficient"):
        _run_task(
            task_plan,
            layout=layout,
            args=args,
            capacity=capacity,
            previously_committed_units=0,
        )
    assert capacity.checked == ("task in-flight peak task", 2 * 64 * 1024 * 1024 + 600)
    assert not layout.final.exists()


def test_task_marker_rejects_mutated_embedded_plan(tmp_path: Path):
    path = tmp_path / "commit.json"
    plan = {"task_key": "task", "partition_name": "part", "value": 1}
    plan["fingerprint"] = canonical_fingerprint(plan)
    marker = {"schema_version": TASK_MARKER_VERSION, "status": "committed", "task_key": "task", "fingerprint": plan["fingerprint"], "task_plan": plan}
    marker["marker_fingerprint"] = canonical_fingerprint(marker)
    atomic_write_json(path, marker)
    assert _read_task_marker(path, task_key="task") == marker
    marker["task_plan"]["value"] = 2
    marker["marker_fingerprint"] = _fingerprint_without(marker, "marker_fingerprint")
    atomic_write_json(path, marker)
    with pytest.raises(ConversionError, match="embedded"):
        _read_task_marker(path, task_key="task")


def test_low_space_upload_failure_retains_then_success_releases_local_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "unit"
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    table = pa.table(
        {
            "frame_index": pa.array([0], type=pa.int64()),
            "episode_index": pa.array([0], type=pa.int64()),
            "index": pa.array([0], type=pa.int64()),
            "task_index": pa.array([0], type=pa.int64()),
        }
    )
    pq.write_table(table, data / "file-000.parquet")
    pq.write_table(table, data / "file-001.parquet")
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    unit = ParallelWorkUnit(0, "task/unit", "uid", str(root), 0, 1, 0, 1, (0,), 1, 1, 1, "fingerprint", None)
    write_verified_unit_marker(unit)

    original_copy = direct_commit._copy_file
    calls = 0

    def fail_second(source: Path, destination: Path, *, block_bytes: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected upload failure")
        return original_copy(source, destination, block_bytes=block_bytes)

    monkeypatch.setattr(direct_commit, "_copy_file", fail_second)
    clock = iter((10.0, 12.0, 20.0, 23.0))
    monkeypatch.setattr(direct_commit.time, "monotonic", lambda: next(clock))
    with pytest.raises(OSError, match="injected"):
        commit_verified_unit(
            unit,
            partition_name="part",
            partition_root=tmp_path / "remote",
            resume_root=tmp_path / "resume",
            retain_local_after_commit=False,
        )
    assert len(list(data.glob("*.parquet"))) == 2
    marker = json.loads(committed_marker_path(tmp_path / "resume", "part", unit).read_text())
    assert marker["status"] == "committing"
    assert marker["upload_elapsed_seconds"] == 2.0

    monkeypatch.setattr(direct_commit, "_copy_file", original_copy)
    commit_verified_unit(
        unit,
        partition_name="part",
        partition_root=tmp_path / "remote",
        resume_root=tmp_path / "resume",
        retain_local_after_commit=False,
    )
    assert not root.exists()
    completed = json.loads(committed_marker_path(tmp_path / "resume", "part", unit).read_text())
    assert completed["status"] == "verified"
    assert completed["upload_elapsed_seconds"] == 5.0


def test_fingerprint_helper_excludes_only_requested_field():
    payload = {"schema_version": 1, "value": [1, 2]}
    payload["catalog_fingerprint"] = canonical_fingerprint(payload)
    assert payload["catalog_fingerprint"] == _fingerprint_without(payload, "catalog_fingerprint")


def test_zero_count_source_stats_remain_finite():
    from convert_core.direct_commit import _merge_stats_preserving_empty

    empty = {
        "camera": {
            "count": np.asarray([0]),
            "mean": np.asarray([[[0.0]]]),
            "std": np.asarray([[[0.0]]]),
        }
    }

    def must_not_aggregate(_values):
        raise AssertionError("zero-count source stats must not be divided")

    merged = _merge_stats_preserving_empty(empty, empty, must_not_aggregate)
    assert int(merged["camera"]["count"][0]) == 0
    assert np.isfinite(merged["camera"]["mean"]).all()
