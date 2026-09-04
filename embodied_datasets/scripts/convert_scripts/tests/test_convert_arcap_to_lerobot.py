from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

import convert_arcap_to_lerobot as converter
from convert_arcap_to_lerobot import (
    ARCapRuntimeLayout,
    _LocalReservationLedger,
    _RunProgress,
    _build_units,
    _cleanup_incomplete_local_units,
    _configure_runtime_environment,
    _estimate,
    _pending_units_for_partition,
    _phase_group_slices,
    _prepare_collection_state,
    _raw_source_inventory,
    _validate_published_collection,
)
from convert_core.checkpoint import atomic_write_json
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError
from convert_core.parallel import ParallelWorkUnit
from convert_core.staging import make_staging_layout, sha256_file
from readers.arcap_hdf5_reader import ARCapPartitionInfo, PARTITIONS_BY_NAME


def _plan(tmp_path: Path) -> DatasetConversionPlan:
    episodes = []
    for group, lengths in enumerate(((3, 4, 5), (2, 3, 4), (8, 8, 8))):
        for offset, length in enumerate(lengths):
            episodes.append(
                EpisodePlan(
                    episode_uid=f"episode-{len(episodes)}",
                    source_relative_path=f"demo_{len(episodes)}",
                    instruction="task",
                    num_frames=length,
                    extra={
                        "phase_group_index": group,
                        "phase_group_size": 3,
                        "phase_offset": offset,
                    },
                )
            )
    return DatasetConversionPlan(
        dataset_uid="arcap_test",
        output_path=tmp_path / "assemble",
        fps=10,
        measured_fps=10.0,
        robot_type="test",
        vector_features=(),
        camera_features=(),
        episodes=tuple(episodes),
    )


def test_work_units_batch_whole_phase_groups_deterministically(tmp_path: Path) -> None:
    units = _phase_group_slices(_plan(tmp_path), max_frames=22)
    assert [(unit.episode_start, unit.episode_end) for unit in units] == [(0, 6), (6, 9)]
    assert [(unit.frame_start, unit.frame_end) for unit in units] == [(0, 21), (21, 45)]
    assert [unit.index for unit in units] == [0, 1]


def test_work_units_reject_truncated_phase_group(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(ConversionError, match="incomplete"):
        _phase_group_slices(replace(plan, episodes=plan.episodes[:-1]), 100)


def test_runtime_layout_rejects_home_escape(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    with pytest.raises(ConversionError, match="inside staging root"):
        make_staging_layout(
            output_root=root,
            dataset_uid="arcap",
            run_id="run",
            work_dir=Path("/home/arcap-work"),
        )


def _unit(tmp_path: Path, index: int, estimate: int) -> ParallelWorkUnit:
    return ParallelWorkUnit(
        index=index,
        key=f"unit-{index}",
        dataset_uid=f"unit-{index}",
        target_path=str(tmp_path / f"unit-{index}"),
        episode_start=index,
        episode_end=index + 1,
        frame_start=index,
        frame_end=index + 1,
        task_indices=(0,),
        weight=1,
        estimated_memory_bytes=1,
        estimated_temp_bytes=estimate,
        fingerprint=f"fingerprint-{index}",
        payload=None,
    )


class _FakeGuard:
    def check(self, _stage: str, **_values: int):
        return SimpleNamespace(as_dict=lambda: dict(_values))


def test_local_reservation_rejects_single_unit_over_global_limit(tmp_path: Path) -> None:
    ledger = _LocalReservationLedger(
        local_root=tmp_path,
        guard=_FakeGuard(),  # type: ignore[arg-type]
        max_bytes=10,
        max_units=2,
    )
    with pytest.raises(ConversionError, match="split the unit"):
        ledger.reserve(_unit(tmp_path, 0, 11))


def test_local_reservation_backpressures_until_uploader_releases(tmp_path: Path) -> None:
    ledger = _LocalReservationLedger(
        local_root=tmp_path,
        guard=_FakeGuard(),  # type: ignore[arg-type]
        max_bytes=10,
        max_units=1,
    )
    first = _unit(tmp_path, 0, 6)
    second = _unit(tmp_path, 1, 6)
    ledger.reserve(first)
    acquired = threading.Event()

    def reserve_second() -> None:
        ledger.reserve(second)
        acquired.set()

    thread = threading.Thread(target=reserve_second)
    thread.start()
    time.sleep(0.05)
    assert not acquired.is_set()
    ledger.release(first)
    assert acquired.wait(1.0)
    ledger.release(second)
    thread.join(timeout=1.0)


def test_interruption_cleanup_removes_only_unverified_local_units(tmp_path: Path) -> None:
    incomplete = _unit(tmp_path, 0, 1)
    verified = _unit(tmp_path, 1, 1)
    incomplete_path = Path(incomplete.target_path)
    verified_path = Path(verified.target_path)
    incomplete_path.mkdir()
    verified_path.mkdir()
    hidden = incomplete_path.with_name(f".{incomplete_path.name}.incomplete-dead")
    hidden.mkdir()
    cache = incomplete_path.with_name(f".{incomplete_path.name}.datasets-cache")
    cache.mkdir()
    verified_marker = verified_path.with_name(f"{verified_path.name}.verified.json")
    verified_marker.write_text("{}", encoding="utf-8")

    removed = _cleanup_incomplete_local_units((incomplete, verified))

    assert str(incomplete_path) in removed
    assert str(hidden) in removed
    assert str(cache) in removed
    assert not incomplete_path.exists()
    assert verified_path.is_dir()
    assert verified_marker.is_file()


def test_raw_source_inventory_records_size_and_mtime(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    source.write_bytes(b"read-only-source")
    spec = SimpleNamespace(filename=source.name)

    inventory = _raw_source_inventory(tmp_path, (spec,))

    assert inventory == [
        {
            "relative_path": source.name,
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
        }
    ]


def test_arcap_runtime_environment_is_entirely_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "arcap_staging"
    layout = ARCapRuntimeLayout(
        root=tmp_path / "oss",
        local_root=local,
        dataset_uid="arcap",
        run_id="run",
        final=tmp_path / "oss" / "arcap",
        work=local / "work" / "run",
        resume=local / "resume" / "arcap",
        logs=local / "logs" / "arcap",
        cache=local / "cache" / "run",
        temp=local / "work" / "run" / "tmp",
        lock=local / "resume" / "arcap.lock",
    )
    for key in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "MPLCONFIGDIR",
        "VLA_DATASETS_CACHE_ROOT",
        "CUDA_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "NUMBA_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    values = _configure_runtime_environment(layout, create=False)
    assert "CUDA_CACHE_PATH" in values
    assert all(Path(value).is_relative_to(local) for value in values.values())


def test_collection_resume_starts_fresh_and_rejects_changed_fingerprint(
    tmp_path: Path,
) -> None:
    layout = make_staging_layout(
        output_root=tmp_path / "staging",
        dataset_uid="arcap-test",
        run_id="run",
    )
    layout.create_runtime_directories()
    payload = {"source": "official", "workers": 2}
    first = _prepare_collection_state(layout, payload, require_existing=False)
    assert _prepare_collection_state(layout, payload, require_existing=True) == first
    with pytest.raises(ConversionError, match="fingerprint changed"):
        _prepare_collection_state(
            layout,
            {"source": "official", "workers": 4},
            require_existing=True,
        )


def _published_info(tmp_path: Path) -> tuple[Path, ARCapPartitionInfo]:
    final = tmp_path / "arcap-published"
    partition = final / "assemble"
    partition.mkdir(parents=True)
    plan = replace(
        _plan(tmp_path),
        dataset_uid="arcap_assemble",
        output_path=partition,
    )
    spec = PARTITIONS_BY_NAME["assemble"]
    info = ARCapPartitionInfo(
        spec=spec,
        source_path=tmp_path / spec.filename,
        source_relative_path=spec.filename,
        plan=plan,
        all_episode_count=len(plan.episodes),
        all_frame_count=plan.num_frames,
        selected_logical_bytes=1,
        source_schema=(),
        schema_fingerprint="schema",
        episode_length_summary={},
        payload_scan={},
    )
    record = {
        "name": spec.name,
        "relative_path": spec.name,
        "episodes": len(plan.episodes),
        "frames": plan.num_frames,
        "source_relative_path": spec.filename,
        "source_schema_fingerprint": "schema",
        "source_sha256": spec.sha256,
        "phase_group_size": spec.phase_group_size,
        "instruction": spec.instruction,
    }
    atomic_write_json(final / "collection_manifest.json", {"partitions": [record]})
    atomic_write_json(
        final / "_SUCCESS",
        {
            "fingerprint": "expected",
            "collection_manifest_sha256": sha256_file(
                final / "collection_manifest.json"
            ),
        },
    )
    atomic_write_json(
        partition / "conversion_manifest.json",
        {
            "dataset_uid": plan.dataset_uid,
            "source_format": "arcap_hdf5",
            "robot_type": plan.robot_type,
            "fps": plan.fps,
            "num_episodes": len(plan.episodes),
            "num_frames": plan.num_frames,
            "num_video_features": 0,
        },
    )
    return final, info


def test_build_units_uses_global_flat_work_paths_and_preflight_is_complete(
    tmp_path: Path,
) -> None:
    final, info = _published_info(tmp_path)
    second_spec = PARTITIONS_BY_NAME["clutter"]
    second_info = replace(
        info,
        spec=second_spec,
        source_path=tmp_path / second_spec.filename,
        source_relative_path=second_spec.filename,
        plan=replace(
            info.plan,
            dataset_uid="arcap_clutter",
            output_path=final / "clutter",
        ),
        selected_logical_bytes=2,
    )
    local = tmp_path / "arcap_staging"
    layout = ARCapRuntimeLayout(
        root=final.parent,
        local_root=local,
        dataset_uid="arcap-test",
        run_id="run",
        final=final,
        work=local / "work" / "run",
        resume=local / "resume" / "arcap-test",
        logs=local / "logs" / "arcap-test",
        cache=local / "cache" / "run",
        temp=local / "work" / "run" / "tmp",
        lock=local / "resume" / "arcap-test.lock",
    )
    args = SimpleNamespace(
        max_frames_per_unit=22,
        acceleration_mode="parallel",
        workers=4,
        max_inflight_units=4,
        upload_workers=2,
        encoder_threads_per_worker=8,
        worker_memory_limit_bytes=1024,
        output_dataset_uid="arcap-test",
        eta_interval_seconds=10.0,
    )

    units, local = _build_units((info, second_info), layout, args)
    estimate = _estimate((info, second_info), units)

    assert [Path(unit.target_path) for unit in units] == [
        layout.work / f"unit-{index:06d}" for index in range(len(units))
    ]
    assert len(_pending_units_for_partition(units, "assemble", {0, 1})) == 2
    assert len(_pending_units_for_partition(units, "clutter", {0, 1})) == 2
    assert len(local) == 2
    assert estimate["source_container_bytes"] == (
        info.spec.source_bytes + second_info.spec.source_bytes
    )
    assert estimate["selected_logical_input_bytes"] == (
        info.selected_logical_bytes + second_info.selected_logical_bytes
    )
    assert estimate["planned_work_units"] == len(units)
    assert estimate["maximum_unit_estimated_temp_bytes"] == max(
        unit.estimated_temp_bytes for unit in units
    )
    assert estimate["estimated_wall_seconds"] > 0
    assert estimate["reference_end_to_end_frames_per_second"] > 0


def test_skip_validation_reopens_every_partition_and_rejects_manifest_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final, info = _published_info(tmp_path)
    reopened: list[Path] = []
    monkeypatch.setattr(
        converter,
        "validate_written_dataset",
        lambda _plan, path: reopened.append(path),
    )
    _validate_published_collection(
        final,
        [info],
        expected_fingerprint="expected",
        cache_root=tmp_path / "cache",
    )
    assert reopened == [final / "assemble"]

    manifest = json.loads((final / "collection_manifest.json").read_text())
    manifest["partitions"][0]["frames"] += 1
    atomic_write_json(final / "collection_manifest.json", manifest)
    marker = json.loads((final / "_SUCCESS").read_text())
    marker["collection_manifest_sha256"] = sha256_file(
        final / "collection_manifest.json"
    )
    atomic_write_json(final / "_SUCCESS", marker)
    with pytest.raises(ConversionError, match="changed frames"):
        _validate_published_collection(
            final,
            [info],
            expected_fingerprint="expected",
            cache_root=tmp_path / "cache",
        )


def test_resume_progress_counts_reused_frames_and_writes_jsonl(tmp_path: Path) -> None:
    layout = make_staging_layout(
        output_root=tmp_path / "staging",
        dataset_uid="arcap-test",
        run_id="run",
    )
    layout.create_runtime_directories()
    units = tuple(
        ParallelWorkUnit(
            index=index,
            key=f"unit-{index}",
            dataset_uid=f"unit-{index}",
            target_path=str(layout.work / f"unit-{index}"),
            episode_start=index,
            episode_end=index + 1,
            frame_start=index * 10,
            frame_end=(index + 1) * 10,
            task_indices=(0,),
            weight=10,
            estimated_memory_bytes=1,
            estimated_temp_bytes=1,
            fingerprint=f"fingerprint-{index}",
            payload=None,
        )
        for index in range(2)
    )
    args = SimpleNamespace(
        eta_interval_seconds=1.0,
        acceleration_mode="parallel",
        encoder_threads_per_worker=1,
        workers=2,
    )
    progress = _RunProgress(
        layout=layout,
        args=args,
        units=units,
        pending=(units[1],),
        reused=1,
        estimate={"expected_output_bytes": 1_000},
        active_workers=1,
    )
    snapshot = progress._snapshot("resume", None, inventory=False)
    assert snapshot["completed_units"] == 1
    assert snapshot["completed_frames"] == 10
    assert snapshot["percent"] == 50.0
    assert snapshot["reused_verified_units"] == 1
    progress.emit("resume", force=True)
    events = list(layout.logs.glob("*.coordinator.jsonl"))
    assert len(events) == 1
    assert json.loads(events[0].read_text().splitlines()[-1])["event"] == "progress"
