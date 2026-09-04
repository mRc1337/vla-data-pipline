import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from convert_core.errors import ConversionError
from convert_core.parallel import ParallelWorkUnit
from convert_robogene_to_lerobot import LocalReservation
from convert_robogene_to_lerobot import (
    UnitPayload,
    _build_unit,
    _encoder_warmup_once,
    _load_catalog_for_run,
    _prepare_local_units,
    _resume_state_path,
)
from readers import robogene_reader
from readers.robogene_reader import (
    catalog_from_payload,
    catalog_to_payload,
    inspect_robogene,
    validate_catalog_source_files,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _task(root: Path, split: str, name: str, *, action_width: int, empty: bool = False) -> None:
    task = root / split / name
    info = {
        "codebase_version": "v2.1",
        "robot_type": split,
        "fps": 30,
        "chunks_size": 1000,
        "features": {
            "action": {"dtype": "float32", "shape": [action_width], "names": None},
            "observation.rgb": {"dtype": "video", "shape": [4, 5, 3], "info": {"video.codec": "h264"}},
        },
        "total_episodes": 0 if empty else 1,
    }
    (task / "meta").mkdir(parents=True, exist_ok=True)
    (task / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    if empty:
        return
    _write_jsonl(task / "meta" / "tasks.jsonl", [{"task_index": 0, "task": f"do {name}"}])
    _write_jsonl(task / "meta" / "episodes.jsonl", [{"episode_index": 0, "tasks": [f"do {name}"], "length": 2}])
    _write_jsonl(task / "meta" / "episodes_stats.jsonl", [{"episode_index": 0, "stats": {"action": {"min": [0.0] * action_width, "max": [1.0] * action_width, "mean": [0.5] * action_width, "std": [0.5] * action_width, "count": [2] * action_width}}}])
    data = task / "data" / "chunk-000" / "episode_000000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "action": pa.array([[0.0] * action_width, [1.0] * action_width], type=pa.list_(pa.float32(), action_width)),
                "episode_index": [0, 0],
                "frame_index": [0, 1],
                "index": [0, 1],
                "task_index": [0, 0],
                "timestamp": [0.0, 1 / 30],
            }
        ),
        data,
    )
    video = task / "videos" / "observation.rgb" / "chunk-000" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")


def test_reader_splits_schema_honors_exact_task_and_records_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _task(tmp_path, "dual_arm", "a", action_width=2)
    _task(tmp_path, "dual_arm", "b", action_width=3)
    _task(tmp_path, "dual_arm", "declared_empty", action_width=2, empty=True)
    monkeypatch.setattr(robogene_reader, "_inspect_payload_samples", lambda *_args: ())

    catalog = inspect_robogene(tmp_path, task_names={"a", "b", "declared_empty"})
    assert [part.name for part in catalog.partitions] == sorted(part.name for part in catalog.partitions)
    assert len(catalog.partitions) == 2
    assert all("--schema-" in part.name for part in catalog.partitions)
    assert all(part.empty_tasks == ("declared_empty",) for part in catalog.partitions)
    assert {source.task_name for part in catalog.partitions for source in part.episodes} == {"a", "b"}


def test_reader_limit_shards_and_catalog_resume_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _task(tmp_path, "single_arm", "a", action_width=2)
    _task(tmp_path, "single_arm", "b", action_width=2)
    monkeypatch.setattr(robogene_reader, "_inspect_payload_samples", lambda *_args: ())
    catalog = inspect_robogene(tmp_path, limit_shards=1)
    assert sum(len(part.episodes) for part in catalog.partitions) == 1
    restored = catalog_from_payload(catalog_to_payload(catalog))
    validate_catalog_source_files(restored, tmp_path)
    source = restored.partitions[0].episodes[0].data_path
    source.write_bytes(b"changed")
    with pytest.raises(ConversionError, match="fingerprint changed"):
        validate_catalog_source_files(restored, tmp_path)


def test_worker_rebases_indices_before_verified_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _task(tmp_path, "single_arm", "a", action_width=2)
    monkeypatch.setattr(robogene_reader, "_inspect_payload_samples", lambda *_args: ())
    catalog = inspect_robogene(tmp_path)
    source = catalog.partitions[0].episodes[0]
    monkeypatch.setattr("convert_robogene_to_lerobot._validate_local_unit", lambda _unit: None)
    monkeypatch.setattr("convert_robogene_to_lerobot.write_verified_unit_marker", lambda _unit: None)
    unit = ParallelWorkUnit(
        index=0,
        key="single_arm/unit-000000",
        dataset_uid="fixture",
        target_path=str(tmp_path / "work" / "unit-000000"),
        episode_start=3,
        episode_end=4,
        frame_start=10,
        frame_end=12,
        task_indices=(7,),
        weight=2,
        estimated_memory_bytes=1,
        estimated_temp_bytes=1,
        fingerprint="fixture",
        payload=UnitPayload(catalog.partitions[0], (source,)),
    )
    _build_unit(unit)
    table = pq.read_table(Path(unit.target_path) / "data/chunk-000/file-000.parquet")
    assert table["episode_index"].to_pylist() == [3, 3]
    assert table["index"].to_pylist() == [10, 11]
    assert table["task_index"].to_pylist() == [7, 7]


def test_encoder_warmup_is_persisted_and_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _task(tmp_path, "single_arm", "a", action_width=2)
    monkeypatch.setattr(robogene_reader, "_inspect_payload_samples", lambda *_args: ())
    catalog = inspect_robogene(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "convert_robogene_to_lerobot._encoder_warmup",
        lambda partition, local_root: calls.append((partition.name, str(local_root))),
    )
    _encoder_warmup_once(catalog, local_root=tmp_path, fingerprint="warmup-fingerprint", run_id="run-1")
    _encoder_warmup_once(catalog, local_root=tmp_path, fingerprint="warmup-fingerprint", run_id="run-1")
    assert len(calls) == 1


def test_resume_loads_frozen_catalog_without_rescanning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _task(tmp_path / "raw", "single_arm", "a", action_width=2)
    monkeypatch.setattr(robogene_reader, "_inspect_payload_samples", lambda *_args: ())
    local = tmp_path / "local"
    common = {
        "task": None,
        "limit_tasks": None,
        "limit_episodes": None,
        "limit_shards": None,
        "max_local_temp_bytes": 1000,
        "encoder_threads_per_worker": 8,
        "run_id": "run-a",
    }
    from argparse import Namespace

    first_args = Namespace(**common, resume=False)
    catalog, fingerprint, run_id = _load_catalog_for_run(first_args, tmp_path / "raw", state_path=_resume_state_path(local))
    assert catalog.partitions and fingerprint and run_id == "run-a"
    resume_args = Namespace(**common, resume=True)
    monkeypatch.setattr("convert_robogene_to_lerobot.inspect_robogene", lambda *_args, **_kwargs: pytest.fail("resume rescanned the raw tree"))
    restored, restored_fingerprint, restored_run_id = _load_catalog_for_run(resume_args, tmp_path / "raw", state_path=_resume_state_path(local))
    assert restored_fingerprint == fingerprint
    assert restored_run_id == run_id


def _unit(index: int, estimate: int) -> ParallelWorkUnit:
    return ParallelWorkUnit(index, f"unit-{index}", "fixture", "/tmp/unused", index, index + 1, index, index + 1, (0,), 1, 1, estimate, "fingerprint", None)


def test_local_reservation_accounts_for_existing_usage_and_reservations(tmp_path: Path):
    (tmp_path / "existing").write_bytes(b"x" * 10)
    class Guard:
        def check(self, *_args, **_kwargs) -> None:
            return None

    guard = Guard()
    ledger = LocalReservation(tmp_path, guard, 100, 3)
    first, second = _unit(0, 40), _unit(1, 40)
    ledger.reserve(first)
    ledger.reserve(second)
    # 10 current + 40 already reserved + 60 next would exceed the hard cap.
    with pytest.raises(ConversionError, match="exceeds local quota"):
        ledger.reserve(_unit(2, 101))
    ledger.release(first)
    ledger.release(second)


def test_local_reservation_try_reserve_is_nonblocking(tmp_path: Path):
    (tmp_path / "existing").write_bytes(b"x" * 10)

    class Guard:
        def check(self, *_args, **_kwargs) -> None:
            return None

    ledger = LocalReservation(tmp_path, Guard(), 100, 3)
    first = _unit(0, 40)
    blocked = _unit(1, 61)

    assert ledger.try_reserve(first)
    assert not ledger.try_reserve(blocked)
    ledger.release(first)
    assert ledger.try_reserve(blocked)


def test_resume_uploads_verified_local_units_and_discards_partial_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    verified = replace(_unit(0, 10), target_path=str(tmp_path / "unit-0"))
    partial = replace(_unit(1, 10), target_path=str(tmp_path / "unit-1"))
    verified_root = Path(verified.target_path)
    partial_root = Path(partial.target_path)
    verified_root.mkdir(parents=True)
    partial_root.mkdir(parents=True)
    (verified_root / "payload").write_bytes(b"verified")
    (partial_root / "payload").write_bytes(b"partial")
    verified_marker_path = verified_root.with_name("unit-0.verified.json")
    verified_marker_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "convert_robogene_to_lerobot.read_verified_unit_marker",
        lambda _unit: {"inventory": [{"size": 8}]},
    )

    ready, rebuild, verified_bytes = _prepare_local_units((verified, partial))

    assert ready == (verified,)
    assert rebuild == (partial,)
    assert verified_root.is_dir()
    assert partial_root.exists() is False
    assert verified_marker_path.is_file()
    assert verified_bytes == len(b"verified")
