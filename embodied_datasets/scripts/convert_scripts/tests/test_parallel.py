from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time

import numpy as np
import pytest

from convert_core.equivalence import verify_lerobot_equivalence
from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import convert_dataset
from convert_core.parallel import (
    ParallelWorkError,
    ParallelWorkUnit,
    aggregate_lerobot_work_units,
    prepare_work_units,
    run_parallel_work_units,
    split_plan_into_units,
    validate_inflight_budget,
    validate_verified_unit_marker,
    write_verified_unit_marker,
)


def _unit(
    root: Path,
    index: int,
    *,
    weight: int = 1,
    payload: object = None,
) -> ParallelWorkUnit:
    return ParallelWorkUnit(
        index=index,
        key=f"unit-{index}",
        dataset_uid=f"dataset-unit-{index}",
        target_path=str(root / f"unit-{index}"),
        episode_start=index,
        episode_end=index + 1,
        frame_start=sum(range(index + 1)) if weight == index + 1 else index,
        frame_end=(sum(range(index + 1)) + weight) if weight == index + 1 else index + 1,
        task_indices=(index,),
        weight=weight,
        estimated_memory_bytes=weight * 10,
        estimated_temp_bytes=weight * 20,
        fingerprint=f"fingerprint-{index}",
        payload=payload,
    )


def _contiguous_units(
    root: Path, weights: list[int], payloads: list[object]
) -> tuple[ParallelWorkUnit, ...]:
    frame_start = 0
    units = []
    for index, (weight, payload) in enumerate(zip(weights, payloads, strict=True)):
        units.append(
            ParallelWorkUnit(
                index=index,
                key=f"unit-{index}",
                dataset_uid=f"dataset-unit-{index}",
                target_path=str(root / f"unit-{index}"),
                episode_start=index,
                episode_end=index + 1,
                frame_start=frame_start,
                frame_end=frame_start + weight,
                task_indices=(index,),
                weight=weight,
                estimated_memory_bytes=weight * 10,
                estimated_temp_bytes=weight * 20,
                fingerprint=f"fingerprint-{index}",
                payload=payload,
            )
        )
        frame_start += weight
    return tuple(units)


def _process_worker(unit: ParallelWorkUnit) -> str:
    payload = dict(unit.payload)
    path = Path(unit.target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("started\n", encoding="utf-8")
    if payload.get("hard_exit"):
        os._exit(17)
    time.sleep(float(payload.get("delay", 0)))
    if payload.get("fail"):
        raise RuntimeError("synthetic worker failure")
    return unit.key


def test_split_plan_preassigns_episode_frame_task_and_part_order(tmp_path: Path):
    episodes = tuple(
        EpisodePlan(
            f"episode-{index}",
            f"source/{index}",
            "task-a" if index != 1 else "task-b",
            index + 1,
            {"checkpoint_unit": "shard-0" if index < 3 else "shard-1"},
        )
        for index in range(4)
    )
    plan = DatasetConversionPlan(
        "split-test",
        tmp_path / "out",
        30,
        30.0,
        "robot",
        (),
        (),
        episodes,
    )
    units = split_plan_into_units(plan, max_episodes_per_unit=2)
    assert [unit.key for unit in units] == [
        "shard-0/part_00000",
        "shard-0/part_00001",
        "shard-1",
    ]
    assert [(unit.episode_start, unit.episode_end) for unit in units] == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]
    assert [(unit.frame_start, unit.frame_end) for unit in units] == [
        (0, 3),
        (3, 6),
        (6, 10),
    ]
    assert [unit.task_indices for unit in units] == [(0, 1), (0,), (0,)]


def test_completion_order_does_not_change_plan_order(tmp_path: Path):
    units = _contiguous_units(
        tmp_path,
        [3, 2, 1],
        [
            {"delay": 0.25},
            {"delay": 0.01},
            {"delay": 0.01},
        ],
    )
    result = run_parallel_work_units(units, _process_worker, workers=2)
    assert [item.key for item in result.results] == ["unit-0", "unit-1", "unit-2"]
    assert result.completion_order != tuple(item.key for item in result.results)


def test_worker_failure_stops_new_dispatch(tmp_path: Path):
    units = _contiguous_units(
        tmp_path,
        [4, 3, 2, 1],
        [
            {"fail": True},
            {"delay": 0.2},
            {},
            {},
        ],
    )
    with pytest.raises(ParallelWorkError, match="unit-0"):
        run_parallel_work_units(units, _process_worker, workers=2)
    assert Path(units[0].target_path).exists()
    assert Path(units[1].target_path).exists()
    assert not Path(units[2].target_path).exists()
    assert not Path(units[3].target_path).exists()


def test_worker_hard_exit_is_reported_and_stops_new_dispatch(tmp_path: Path):
    units = _contiguous_units(
        tmp_path,
        [4, 3, 2],
        [
            {"hard_exit": True},
            {"delay": 0.5},
            {},
        ],
    )
    with pytest.raises(ParallelWorkError, match="unit-0"):
        run_parallel_work_units(units, _process_worker, workers=2)
    assert Path(units[0].target_path).exists()
    assert Path(units[1].target_path).exists()
    assert not Path(units[2].target_path).exists()


def test_running_health_failure_stops_new_dispatch(tmp_path: Path):
    units = _contiguous_units(
        tmp_path,
        [4, 3, 2],
        [
            {"delay": 0.5},
            {"delay": 0.5},
            {},
        ],
    )

    def fail_health_check() -> None:
        raise ConversionError("synthetic periodic capacity failure")

    with pytest.raises(ConversionError, match="periodic capacity failure"):
        run_parallel_work_units(
            units,
            _process_worker,
            workers=2,
            health_check=fail_health_check,
            health_check_interval_seconds=0.01,
        )
    assert Path(units[0].target_path).exists()
    assert Path(units[1].target_path).exists()
    assert not Path(units[2].target_path).exists()


def test_before_dispatch_failure_stops_new_frontier_work(tmp_path: Path):
    units = _contiguous_units(
        tmp_path,
        [4, 3, 2, 1],
        [
            {"delay": 0.01},
            {"delay": 0.5},
            {},
            {},
        ],
    )
    checked: list[str] = []

    def reject_third(
        unit: ParallelWorkUnit, active: tuple[ParallelWorkUnit, ...]
    ) -> None:
        checked.append(unit.key)
        if unit.key == "unit-2":
            assert [item.key for item in active] == ["unit-1"]
            raise ConversionError("synthetic dispatch capacity failure")

    with pytest.raises(ConversionError, match="dispatch capacity failure"):
        run_parallel_work_units(
            units,
            _process_worker,
            workers=2,
            before_dispatch=reject_third,
        )

    assert checked == ["unit-0", "unit-1", "unit-2"]
    assert Path(units[0].target_path).exists()
    assert Path(units[1].target_path).exists()
    assert not Path(units[2].target_path).exists()
    assert not Path(units[3].target_path).exists()


def test_prepare_units_reuses_verified_and_rebuilds_corrupt_marker(tmp_path: Path):
    units = _contiguous_units(tmp_path, [1, 1], [None, None])
    for unit in units:
        target = Path(unit.target_path)
        target.mkdir()
        (target / "payload.bin").write_bytes(bytes([unit.index]))
        write_verified_unit_marker(unit)
    marker = Path(units[1].target_path).with_name("unit-1.verified.json")
    marker.write_text("{}\n", encoding="utf-8")

    prepared = prepare_work_units(units, lambda _unit: None)
    assert [unit.key for unit in prepared.reusable] == ["unit-0"]
    assert [unit.key for unit in prepared.pending] == ["unit-1"]
    assert prepared.discarded_corrupt == ("unit-1",)
    assert Path(units[0].target_path).is_dir()
    assert not Path(units[1].target_path).exists()


def test_prepare_units_repairs_missing_marker_after_publication(tmp_path: Path):
    units = _contiguous_units(tmp_path, [1], [None])
    target = Path(units[0].target_path)
    target.mkdir()
    (target / "payload.bin").write_bytes(b"complete")
    prepared = prepare_work_units(units, lambda _unit: None)
    assert prepared.reusable == units
    assert prepared.pending == ()
    assert prepared.repaired_markers == ("unit-0",)
    validate_verified_unit_marker(units[0])


def test_inflight_budget_counts_all_workers(tmp_path: Path):
    units = _contiguous_units(tmp_path, [3, 2, 1], [None, None, None])
    estimate = validate_inflight_budget(
        units,
        2,
        memory_budget_bytes=50,
        temp_budget_bytes=100,
    )
    assert estimate.memory_bytes == 50
    assert estimate.temp_bytes == 100
    with pytest.raises(ConversionError, match="memory estimate"):
        validate_inflight_budget(
            units,
            2,
            memory_budget_bytes=49,
            temp_budget_bytes=100,
        )


def test_ordered_aggregation_is_serially_equivalent(tmp_path: Path):
    pytest.importorskip("lerobot")
    from lerobot.configs.video import RGBEncoderConfig

    episodes = tuple(
        EpisodePlan(
            f"episode-{index}",
            f"source/{index}",
            "task-a" if index == 0 else "task-b",
            3,
            {"checkpoint_unit": f"unit-{index}"},
        )
        for index in range(2)
    )
    plan = DatasetConversionPlan(
        "parallel-equivalence",
        tmp_path / "parallel",
        10,
        10.0,
        "robot",
        (VectorFeatureSpec("observation.state", 2),),
        (CameraFeatureSpec("observation.images.head", 16, 16),),
        episodes,
    )
    encoder = RGBEncoderConfig(vcodec="h264", crf=18, preset="medium")

    def frames(episode: EpisodePlan):
        episode_index = int(episode.episode_uid.rsplit("-", 1)[1])
        for frame_index in range(3):
            yield {
                "observation.state": np.array(
                    [episode_index, frame_index], dtype=np.float32
                ),
                "observation.images.head": np.full(
                    (16, 16, 3), episode_index * 80 + frame_index * 10, dtype=np.uint8
                ),
                "task": episode.instruction,
            }

    serial = tmp_path / "serial"
    convert_dataset(
        replace(plan, output_path=serial),
        frames,
        reader_format="synthetic",
        rgb_encoder=encoder,
        streaming_encoding=True,
        blocking_streaming_encoding=True,
        encoder_queue_maxsize=1,
        encoder_threads=1,
    )
    units = []
    frame_start = 0
    # Materialize in reverse completion order, then aggregate by fixed indices.
    for index in (1, 0):
        target = tmp_path / f"unit-{index}"
        unit_plan = replace(
            plan,
            dataset_uid=f"parallel-equivalence-unit-{index}",
            output_path=target,
            episodes=(episodes[index],),
        )
        convert_dataset(
            unit_plan,
            frames,
            reader_format="synthetic",
            rgb_encoder=encoder,
            streaming_encoding=True,
            blocking_streaming_encoding=True,
            encoder_queue_maxsize=1,
            encoder_threads=1,
        )
        unit = ParallelWorkUnit(
            index=index,
            key=f"unit-{index}",
            dataset_uid=unit_plan.dataset_uid,
            target_path=str(target),
            episode_start=index,
            episode_end=index + 1,
            frame_start=index * 3,
            frame_end=(index + 1) * 3,
            task_indices=(index,),
            weight=3,
            estimated_memory_bytes=1,
            estimated_temp_bytes=1,
            fingerprint=f"fingerprint-{index}",
            payload=unit_plan,
        )
        write_verified_unit_marker(unit)
        units.append(unit)
    units.sort(key=lambda unit: unit.index)
    aggregate_lerobot_work_units(
        plan,
        units,
        plan.output_path,
        reader_format="synthetic",
        parallel_evidence={"workers": 2, "worker_completion_order": ["unit-1", "unit-0"]},
    )
    report = verify_lerobot_equivalence(serial, plan.output_path)
    assert report.total_frames == 6
    assert report.video_frames_compared == 6
