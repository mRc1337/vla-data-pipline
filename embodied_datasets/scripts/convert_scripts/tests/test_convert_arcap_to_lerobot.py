from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import convert_arcap_to_lerobot as converter
from convert_arcap_to_lerobot import (
    _RunProgress,
    _phase_group_slices,
    _prepare_collection_state,
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
