from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from convert_core.direct_commit import (
    commit_verified_unit,
    committed_marker_path,
    finalize_direct_partition,
    prepare_direct_commits,
    validate_committed_unit,
)
from convert_core.equivalence import verify_lerobot_equivalence
from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.lerobot_writer import convert_dataset
from convert_core.parallel import (
    ParallelWorkUnit,
    split_plan_into_units,
    write_verified_unit_marker,
)


def _fixture(tmp_path: Path):
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
        "direct-commit-fixture",
        tmp_path / "final" / "v1_1",
        10,
        10.0,
        "robot",
        (VectorFeatureSpec("observation.state", 2),),
        (CameraFeatureSpec("observation.images.head", 16, 16),),
        episodes,
    )

    def frames(episode: EpisodePlan):
        episode_index = int(episode.episode_uid.rsplit("-", 1)[1])
        for frame_index in range(3):
            yield {
                "observation.state": np.asarray(
                    [episode_index, frame_index], dtype=np.float32
                ),
                "observation.images.head": np.full(
                    (16, 16, 3), episode_index * 80 + frame_index * 10, dtype=np.uint8
                ),
                "task": episode.instruction,
            }

    return plan, frames


def _materialize_units(plan, frames, tmp_path: Path):
    from lerobot.configs.video import RGBEncoderConfig

    encoder = RGBEncoderConfig(vcodec="h264", crf=18, preset="medium")
    units = []
    for item in split_plan_into_units(plan):
        target = tmp_path / "work" / f"unit-{item.index:06d}"
        unit_plan = replace(
            plan,
            dataset_uid=f"{plan.dataset_uid}-unit-{item.index}",
            output_path=target,
            episodes=plan.episodes[item.episode_start : item.episode_end],
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
            index=item.index,
            key=item.key,
            dataset_uid=unit_plan.dataset_uid,
            target_path=str(target),
            episode_start=item.episode_start,
            episode_end=item.episode_end,
            frame_start=item.frame_start,
            frame_end=item.frame_end,
            task_indices=item.task_indices,
            weight=item.frame_end - item.frame_start,
            estimated_memory_bytes=1,
            estimated_temp_bytes=1,
            fingerprint=f"fingerprint-{item.index}",
            payload=unit_plan,
        )
        write_verified_unit_marker(unit)
        units.append(unit)
    return tuple(units)


def test_direct_commit_is_equivalent_without_bulk_aggregation_copy(tmp_path: Path):
    pytest.importorskip("lerobot")
    from lerobot.configs.video import RGBEncoderConfig

    plan, frames = _fixture(tmp_path)
    serial = tmp_path / "serial"
    convert_dataset(
        replace(plan, output_path=serial),
        frames,
        reader_format="synthetic",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="medium"),
        streaming_encoding=True,
        blocking_streaming_encoding=True,
        encoder_queue_maxsize=1,
        encoder_threads=1,
    )
    units = _materialize_units(plan, frames, tmp_path)
    resume = tmp_path / "resume"
    plan.output_path.mkdir(parents=True)
    for unit in reversed(units):
        commit_verified_unit(
            unit,
            partition_name=plan.output_path.name,
            partition_root=plan.output_path,
            resume_root=resume,
        )
    finalize_direct_partition(
        plan,
        units,
        plan.output_path,
        resume_root=resume,
        reader_format="synthetic",
        parallel_evidence={"workers": 2},
        metadata_batch_bytes=1,
        metadata_batch_episodes=1,
    )
    report = verify_lerobot_equivalence(
        serial, plan.output_path, storage_layout_independent=True
    )
    assert report.total_frames == 6
    assert report.video_frames_compared == 6
    manifest = (plan.output_path / "conversion_manifest.json").read_text()
    assert '"bulk_aggregation_copy": false' in manifest
    assert not list((tmp_path / "work").rglob("*.mp4"))
    assert not list((tmp_path / "work").rglob("data/**/*.parquet"))


def test_resume_revalidates_committed_files_and_rebuilds_only_corrupt_unit(tmp_path: Path):
    pytest.importorskip("lerobot")
    plan, frames = _fixture(tmp_path)
    units = _materialize_units(plan, frames, tmp_path)
    resume = tmp_path / "resume"
    plan.output_path.mkdir(parents=True)
    for unit in units:
        commit_verified_unit(
            unit,
            partition_name=plan.output_path.name,
            partition_root=plan.output_path,
            resume_root=resume,
        )
    validate_committed_unit(
        units[0],
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume,
    )
    marker = committed_marker_path(resume, plan.output_path.name, units[1])
    import json

    state = json.loads(marker.read_text())
    damaged = plan.output_path / state["bulk"][0]["destination"]
    damaged.write_bytes(b"damaged")
    prepared = prepare_direct_commits(
        units,
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume,
    )
    assert prepared.committed == (units[0],)
    assert prepared.uncommitted == (units[1],)
    assert prepared.discarded_corrupt == (units[1].key,)
    assert not marker.exists()
    assert not Path(units[1].target_path).exists()
