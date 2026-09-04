from dataclasses import replace
from pathlib import Path
import threading

import numpy as np
import pytest

from convert_core.direct_commit import (
    DirectCommitUploader,
    commit_verified_unit,
    committed_marker_path,
    finalize_direct_partition,
    globalize_unit_data_files,
    prepare_direct_commits,
    unit_metadata_root,
    validate_committed_unit,
)
from convert_core.errors import ConversionError
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


def _fixture(tmp_path: Path, lengths: tuple[int, int] = (3, 3)):
    episodes = tuple(
        EpisodePlan(
            f"episode-{index}",
            f"source/{index}",
            "task-a" if index == 0 else "task-b",
            lengths[index],
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
        for frame_index in range(episode.num_frames):
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
        globalize_unit_data_files(unit)
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


def test_direct_commit_index_stats_keep_one_schema_with_single_frame_unit(
    tmp_path: Path,
):
    pytest.importorskip("lerobot")

    plan, frames = _fixture(tmp_path, lengths=(2, 1))
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

    finalize_direct_partition(
        plan,
        units,
        plan.output_path,
        resume_root=resume,
        reader_format="synthetic",
        parallel_evidence={"workers": 1},
        metadata_batch_bytes=1,
        metadata_batch_episodes=1,
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(next((plan.output_path / "meta" / "episodes").rglob("*.parquet")))
    for feature in ("frame_index", "episode_index", "index", "task_index"):
        for stat in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            assert schema.field(f"stats/{feature}/{stat}").type == pa.list_(pa.float64())
        assert schema.field(f"stats/{feature}/count").type == pa.list_(pa.int64())
    for stat in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
        assert schema.field(f"stats/observation.images.head/{stat}").type == pa.list_(
            pa.list_(pa.list_(pa.float64()))
        )
    assert schema.field("stats/observation.images.head/count").type == pa.list_(
        pa.int64()
    )


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


def test_cross_filesystem_commit_records_remote_evidence_and_deletes_local_bulk(
    tmp_path: Path,
):
    pytest.importorskip("lerobot")
    plan, frames = _fixture(tmp_path)
    unit = _materialize_units(plan, frames, tmp_path)[0]
    resume = tmp_path / "resume"
    plan.output_path.mkdir(parents=True)

    marker = commit_verified_unit(
        unit,
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume,
        copy_block_bytes=17,
    )

    assert marker["status"] == "verified"
    assert marker["bulk"]
    for record in marker["bulk"]:
        assert record["remote_validation"]["size_verified"] == record["size"]
        assert record["remote_validation"]["samples"] == record["samples"]
        assert record["remote_validation"]["format"] == record["format"]
        assert record["remote_validation"]["remote_sha256_verified"] is False
        assert not (Path(unit.target_path) / record["relative_path"]).exists()
        assert (plan.output_path / record["destination"]).is_file()
    assert not Path(unit.target_path).exists()
    assert (unit_metadata_root(resume, plan.output_path.name, unit) / "meta/info.json").is_file()


def test_failed_upload_keeps_local_bulk_removes_remote_and_does_not_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("lerobot")
    plan, frames = _fixture(tmp_path)
    unit = _materialize_units(plan, frames, tmp_path)[0]
    resume = tmp_path / "resume"
    plan.output_path.mkdir(parents=True)

    def interrupted_copy(source: Path, destination: Path, *, block_bytes: int) -> None:
        del source, block_bytes
        destination.write_bytes(b"partial")
        raise OSError("simulated interrupted upload")

    monkeypatch.setattr("convert_core.direct_commit._copy_file", interrupted_copy)
    with pytest.raises(OSError, match="simulated interrupted upload"):
        commit_verified_unit(
            unit,
            partition_name=plan.output_path.name,
            partition_root=plan.output_path,
            resume_root=resume,
        )

    marker_path = committed_marker_path(resume, plan.output_path.name, unit)
    state = __import__("json").loads(marker_path.read_text())
    assert state["status"] == "committing"
    for record in state["bulk"]:
        assert (Path(unit.target_path) / record["relative_path"]).is_file()
        assert not (plan.output_path / record["destination"]).exists()

    monkeypatch.undo()
    resumed = commit_verified_unit(
        unit,
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume,
    )
    assert resumed["status"] == "verified"
    assert not Path(unit.target_path).exists()


def test_resume_after_upload_interruption_retransmits_only_failed_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("lerobot")
    import convert_core.direct_commit as direct_commit

    plan, frames = _fixture(tmp_path)
    units = _materialize_units(plan, frames, tmp_path)
    resume = tmp_path / "resume"
    plan.output_path.mkdir(parents=True)
    commit_verified_unit(
        units[0],
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume,
    )

    real_copy = direct_commit._copy_file

    def interrupted_copy(source: Path, destination: Path, *, block_bytes: int) -> None:
        del source, block_bytes
        destination.write_bytes(b"partial")
        raise OSError("simulated second-unit upload interruption")

    monkeypatch.setattr(direct_commit, "_copy_file", interrupted_copy)
    with pytest.raises(OSError, match="second-unit upload interruption"):
        commit_verified_unit(
            units[1],
            partition_name=plan.output_path.name,
            partition_root=plan.output_path,
            resume_root=resume,
        )

    retransmitted: list[Path] = []

    def recording_copy(source: Path, destination: Path, *, block_bytes: int) -> None:
        retransmitted.append(source)
        real_copy(source, destination, block_bytes=block_bytes)

    monkeypatch.setattr(direct_commit, "_copy_file", recording_copy)
    prepared = prepare_direct_commits(
        units,
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume,
    )

    assert prepared.committed == units
    assert prepared.uncommitted == ()
    assert retransmitted
    assert all(Path(units[1].target_path) in path.parents for path in retransmitted)
    assert not Path(units[0].target_path).exists()
    assert not Path(units[1].target_path).exists()


def test_remote_sample_corruption_is_rejected_without_full_remote_hash(tmp_path: Path):
    pytest.importorskip("lerobot")
    plan, frames = _fixture(tmp_path)
    unit = _materialize_units(plan, frames, tmp_path)[0]
    resume = tmp_path / "resume"
    plan.output_path.mkdir(parents=True)
    marker = commit_verified_unit(
        unit,
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=resume,
    )
    record = marker["bulk"][0]
    destination = plan.output_path / record["destination"]
    with destination.open("r+b") as stream:
        stream.seek(record["samples"][0]["offset"])
        original = stream.read(1)
        stream.seek(record["samples"][0]["offset"])
        stream.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(ConversionError, match="sample"):
        validate_committed_unit(
            unit,
            partition_name=plan.output_path.name,
            partition_root=plan.output_path,
            resume_root=resume,
        )


def test_fresh_commit_reuses_final_inventory_without_rehashing_bulk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("lerobot")
    import convert_core.direct_commit as direct_commit

    plan, frames = _fixture(tmp_path)
    unit = _materialize_units(plan, frames, tmp_path)[0]
    plan.output_path.mkdir(parents=True)
    real_sha256 = direct_commit.sha256_file

    def reject_bulk_rehash(path: Path) -> str:
        target = Path(unit.target_path)
        if path.is_relative_to(target):
            relative = path.relative_to(target)
            if relative.parts[0] in {"data", "videos"}:
                raise AssertionError(f"bulk file was redundantly rehashed: {path}")
        return real_sha256(path)

    monkeypatch.setattr(direct_commit, "sha256_file", reject_bulk_rehash)
    commit_verified_unit(
        unit,
        partition_name=plan.output_path.name,
        partition_root=plan.output_path,
        resume_root=tmp_path / "resume",
        trust_verified_marker=True,
    )


def test_uploader_enqueues_next_unit_while_upload_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import convert_core.direct_commit as direct_commit

    started = threading.Event()
    release = threading.Event()
    completed: list[str] = []

    def blocking_commit(unit, **_kwargs):
        started.set()
        assert release.wait(timeout=5)
        completed.append(unit.key)
        return {}

    monkeypatch.setattr(direct_commit, "commit_verified_unit", blocking_commit)
    units = tuple(
        ParallelWorkUnit(
            index=index,
            key=f"unit-{index}",
            dataset_uid=f"dataset-{index}",
            target_path=str(tmp_path / f"unit-{index}"),
            episode_start=index,
            episode_end=index + 1,
            frame_start=index,
            frame_end=index + 1,
            task_indices=(0,),
            weight=1,
            estimated_memory_bytes=1,
            estimated_temp_bytes=1,
            fingerprint=f"fingerprint-{index}",
            payload=None,
        )
        for index in range(2)
    )
    uploader = DirectCommitUploader(
        partition_name="v1_1",
        partition_root=tmp_path / "remote",
        resume_root=tmp_path / "resume",
        workers=1,
        max_queue_units=2,
    )
    uploader.submit(units[0], trust_verified_marker=True)
    assert started.wait(timeout=5)
    uploader.submit(units[1], trust_verified_marker=True)
    assert completed == []
    release.set()
    stats = uploader.close()
    assert completed == ["unit-0", "unit-1"]
    assert stats["submitted_units"] == stats["completed_units"] == 2


def test_uploader_runs_post_commit_hook_only_after_commit_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import convert_core.direct_commit as direct_commit

    events: list[tuple[str, str]] = []

    def fake_commit(unit, **_kwargs):
        events.append(("commit", unit.key))
        return {}

    monkeypatch.setattr(direct_commit, "commit_verified_unit", fake_commit)
    unit = ParallelWorkUnit(
        index=0,
        key="unit-0",
        dataset_uid="dataset-0",
        target_path=str(tmp_path / "unit-0"),
        episode_start=0,
        episode_end=1,
        frame_start=0,
        frame_end=1,
        task_indices=(0,),
        weight=1,
        estimated_memory_bytes=1,
        estimated_temp_bytes=1,
        fingerprint="fingerprint-0",
        payload=None,
    )
    uploader = DirectCommitUploader(
        partition_name="v1_1",
        partition_root=tmp_path / "remote",
        resume_root=tmp_path / "resume",
        workers=1,
        max_queue_units=1,
        on_committed=lambda value: events.append(("cleanup", value.key)),
    )
    uploader.submit(unit, trust_verified_marker=True)
    uploader.close()

    assert events == [("commit", "unit-0"), ("cleanup", "unit-0")]
