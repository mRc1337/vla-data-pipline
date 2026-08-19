from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

import convert_mimicgen_to_lerobot as cm
from convert_core.checkpoint import exclusive_resume_lock
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import convert_dataset
from readers.robomimic_hdf5_reader import inspect_partition, iter_frames


def _write_source(
    path: Path,
    *,
    mismatched_second: bool = False,
    lengths: tuple[int, ...] = (3, 2),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env_args = {
        "env_name": "Square_D0",
        "env_version": "1.4.1",
        "type": 1,
        "env_kwargs": {
            "robots": ["Panda"],
            "control_freq": 20,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "controller_configs": {"type": "OSC_POSE", "control_delta": True},
        },
    }
    model = """<mujoco><worldbody>
    <joint name="robot0_joint1"/><joint name="robot0_joint2"/>
    <joint name="gripper0_finger_joint1"/><joint name="gripper0_finger_joint2"/>
    </worldbody></mujoco>"""
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["total"] = sum(lengths)
        for demo_index, length in enumerate(lengths):
            demo = data.create_group(f"demo_{demo_index}")
            demo.attrs["num_samples"] = length
            demo.attrs["model_file"] = model
            width = 8 if mismatched_second and demo_index == 1 else 7
            demo.create_dataset("actions", data=np.arange(length * width, dtype=np.float64).reshape(length, width))
            demo.create_dataset("states", data=np.arange(length * 4, dtype=np.float64).reshape(length, 4))
            demo.create_dataset("rewards", data=np.arange(length, dtype=np.float64))
            demo.create_dataset("dones", data=np.arange(length, dtype=np.int64))
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_joint_pos", data=np.ones((length, 2), dtype=np.float64))
            obs.create_dataset("robot0_gripper_qpos", data=np.ones((length, 2), dtype=np.float64))
            obs.create_dataset("robot0_contact", data=np.zeros(length, dtype=bool))
            image = np.zeros((length, 64, 64, 3), dtype=np.uint8)
            image[..., 0] = 40 + demo_index
            image[..., 1] = 80
            image[..., 2] = 120
            obs.create_dataset("agentview_image", data=image)
            obs.create_dataset("robot0_eye_in_hand_image", data=image)
        mask = h5_file.create_group("mask")
        mask.create_dataset("train", data=np.asarray([b"demo_0"]))
        mask.create_dataset("valid", data=np.asarray([b"demo_1"]))


def _inspect(tmp_path: Path, *, mismatched_second: bool = False):
    source_root = tmp_path / "raw" / "minicgen"
    source = source_root / "core" / "square_d0.hdf5"
    _write_source(source, mismatched_second=mismatched_second)
    return inspect_partition(
        source,
        raw_dataset_root=source_root,
        collection_output=tmp_path / "staging" / "lerobot_v3_0" / "mimicgen",
    )


def _collection_args(*, resume: bool, workers: int) -> Namespace:
    return Namespace(
        skip_existing=False,
        resume=resume,
        overwrite=False,
        category=[],
        partition=[],
        max_partitions=None,
        max_episodes=None,
        video_codec="h264",
        video_quality=18,
        video_preset="fast",
        encoder_threads=2,
        workers=workers,
        eta_interval_seconds=60.0,
        resume_validation="fast",
        resume_probe_timeout=30.0,
        memory_budget_gib=None,
        temp_budget_gib=None,
    )


def _staged_args(*, resume: bool, workers: int = 1) -> Namespace:
    args = _collection_args(resume=resume, workers=workers)
    args.max_staging_bytes = None
    args.max_inflight_bytes = None
    args.max_inflight_units = workers
    args.storage_check_interval_seconds = 60.0
    return args


def test_reader_preserves_native_schema_values_names_and_splits(tmp_path: Path):
    info = _inspect(tmp_path)
    plan = info.plan

    assert plan.fps == 20
    assert plan.robot_type == "Panda"
    assert [episode.num_frames for episode in plan.episodes] == [3, 2]
    assert plan.episodes[0].extra["source_splits"] == ("train",)
    assert plan.episodes[1].extra["source_splits"] == ("valid",)
    schema = plan.feature_schema()
    assert schema["action"]["dtype"] == "float64"
    assert schema["source.done"]["dtype"] == "int64"
    assert schema["observation.robot0_contact"]["dtype"] == "bool"
    assert schema["observation.robot0_joint_pos"]["names"] == ["robot0_joint1", "robot0_joint2"]

    frames = list(iter_frames(plan, plan.episodes[0]))
    assert frames[0]["action"].dtype == np.float64
    assert frames[0]["source.done"].dtype == np.int64
    assert frames[0]["observation.robot0_contact"].dtype == np.bool_
    assert frames[0]["observation.images.agentview"].shape == (64, 64, 3)


def test_reader_rejects_episode_schema_drift(tmp_path: Path):
    with pytest.raises(ConversionError, match="schema differs"):
        _inspect(tmp_path, mismatched_second=True)


def test_resume_lock_is_non_blocking(tmp_path: Path):
    lock = tmp_path / ".mimicgen.resume.lock"
    with exclusive_resume_lock(lock):
        with pytest.raises(ConversionError, match="another resume process"):
            with exclusive_resume_lock(lock):
                pass


@pytest.mark.parametrize(
    "flags",
    [
        ["--resume", "--overwrite"],
        ["--resume", "--skip-existing"],
        ["--overwrite", "--skip-existing"],
    ],
)
def test_destructive_modes_are_mutually_exclusive(flags: list[str]):
    with pytest.raises(SystemExit, match="2"):
        cm.main(flags)


def test_parallel_cli_and_legacy_encoder_alias_are_compatible():
    parser = cm._build_parser()
    modern = parser.parse_args(
        ["--workers", "4", "--encoder-threads-per-worker", "8", "--benchmark-workers", "1", "2", "4"]
    )
    legacy = parser.parse_args(["--encoder-threads", "7"])

    assert modern.workers == 4
    assert modern.encoder_threads == 8
    assert modern.benchmark_workers == [1, 2, 4]
    assert legacy.encoder_threads == 7


@pytest.mark.parametrize("codec", ["h264_nvenc", "hevc_nvenc", "av1_nvenc"])
def test_cli_rejects_known_unavailable_nvenc_codecs(codec: str):
    with pytest.raises(SystemExit, match="2"):
        cm.main(["--video-codec", codec])


def test_resume_removes_known_unfinished_partition(tmp_path: Path):
    info = _inspect(tmp_path)
    data_root = tmp_path / ".mimicgen.resume"
    state_root = tmp_path / ".mimicgen.resume-state"
    payload = {"source_revision": "test"}
    fingerprint = cm.canonical_fingerprint(payload)
    state_root.mkdir()
    cm.atomic_write_json(
        state_root / "state.json",
        {
            "resume_schema_version": cm.RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": payload,
        },
    )
    unfinished = data_root / f".{info.partition_name}.incomplete-deadbeef"
    unfinished.mkdir(parents=True)
    (unfinished / "partial.mp4").write_bytes(b"partial")

    pending, reused, frames = cm._prepare_resume(
        [info], data_root, state_root, payload, fingerprint, "h264"
    )

    assert pending == [info]
    assert reused == 0
    assert frames == 0
    assert not unfinished.exists()


def test_resume_reports_changed_fingerprint_section(tmp_path: Path):
    info = _inspect(tmp_path)
    data_root = tmp_path / ".mimicgen.resume"
    state_root = tmp_path / ".mimicgen.resume-state"
    state_root.mkdir()
    cm.atomic_write_json(
        state_root / "state.json",
        {
            "resume_schema_version": cm.RESUME_SCHEMA_VERSION,
            "fingerprint": "old",
            "fingerprint_payload": {"video_encoding": {"codec": "libsvtav1"}},
        },
    )
    with pytest.raises(ConversionError, match="video_encoding"):
        cm._prepare_resume(
            [info],
            data_root,
            state_root,
            {"video_encoding": {"codec": "h264"}},
            "new",
            "h264",
        )


def test_resume_rebuilds_corrupt_marked_partition(tmp_path: Path):
    info = _inspect(tmp_path)
    data_root = tmp_path / ".mimicgen.resume"
    state_root = tmp_path / ".mimicgen.resume-state"
    payload = {"source_stat": {"size": info.source_path.stat().st_size}}
    fingerprint = cm.canonical_fingerprint(payload)
    state_root.mkdir()
    cm.atomic_write_json(
        state_root / "state.json",
        {
            "resume_schema_version": cm.RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": payload,
        },
    )
    corrupt_root = data_root / info.partition_name
    corrupt_root.mkdir(parents=True)
    (corrupt_root / "partial.parquet").write_bytes(b"not parquet")
    cm.atomic_write_json(
        state_root / "partitions" / f"{info.partition_name}.json",
        {
            "resume_schema_version": cm.RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "partition": info.partition_name,
            "episodes": len(info.plan.episodes),
            "frames": info.plan.num_frames,
        },
    )

    pending, reused, _frames = cm._prepare_resume(
        [info], data_root, state_root, payload, fingerprint, "h264"
    )

    assert pending == [info]
    assert reused == 0
    assert not corrupt_root.exists()


def test_resume_fast_validation_uses_persisted_file_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    info = _inspect(tmp_path)
    data_root = tmp_path / ".mimicgen.resume"
    state_root = tmp_path / ".mimicgen.resume-state"
    payload = {"source_revision": "test"}
    fingerprint = cm.canonical_fingerprint(payload)
    state_root.mkdir()
    cm.atomic_write_json(
        state_root / "state.json",
        {
            "resume_schema_version": cm.RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": payload,
        },
    )
    checkpoint = data_root / info.partition_name
    checkpoint.mkdir(parents=True)
    (checkpoint / "validated.bin").write_bytes(b"unchanged")
    validation = cm._checkpoint_validation(checkpoint, [])
    marker = state_root / "partitions" / f"{info.partition_name}.json"
    cm.atomic_write_json(
        marker,
        {
            "resume_schema_version": cm.RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "partition": info.partition_name,
            "episodes": len(info.plan.episodes),
            "frames": info.plan.num_frames,
            "validation": validation,
        },
    )

    def unexpected_full_validation(*_args, **_kwargs):
        raise AssertionError("fast resume must not run full dataset or video validation")

    monkeypatch.setattr(cm, "validate_written_dataset", unexpected_full_validation)
    monkeypatch.setattr(cm, "_validate_video_streams", unexpected_full_validation)

    pending, reused, frames = cm._prepare_resume(
        [info], data_root, state_root, payload, fingerprint, "h264"
    )

    assert pending == []
    assert reused == 1
    assert frames == info.plan.num_frames


def test_ffprobe_timeout_is_reported_as_preserved_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test")

    def time_out(command, **kwargs):
        assert kwargs["timeout"] == 0.25
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(cm.subprocess, "run", time_out)
    with pytest.raises(cm.ResumeValidationUnavailable, match="checkpoint was preserved"):
        cm._ffprobe(video, timeout_seconds=0.25)


def test_resume_validation_timeout_keeps_partition_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    info = _inspect(tmp_path)
    data_root = tmp_path / ".mimicgen.resume"
    state_root = tmp_path / ".mimicgen.resume-state"
    payload = {"source_revision": "test"}
    fingerprint = cm.canonical_fingerprint(payload)
    state_root.mkdir()
    cm.atomic_write_json(
        state_root / "state.json",
        {
            "resume_schema_version": cm.RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": payload,
        },
    )
    checkpoint = data_root / info.partition_name
    checkpoint.mkdir(parents=True)
    (checkpoint / "validated.bin").write_bytes(b"unchanged")
    marker = state_root / "partitions" / f"{info.partition_name}.json"
    cm.atomic_write_json(
        marker,
        {
            "resume_schema_version": cm.RESUME_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "partition": info.partition_name,
            "episodes": len(info.plan.episodes),
            "frames": info.plan.num_frames,
            "validation": cm._checkpoint_validation(checkpoint, []),
        },
    )
    monkeypatch.setattr(cm, "validate_written_dataset", lambda *_args, **_kwargs: None)

    def unavailable(*_args, **_kwargs):
        raise cm.ResumeValidationUnavailable("probe timeout; checkpoint was preserved")

    monkeypatch.setattr(cm, "_validate_video_streams", unavailable)

    with pytest.raises(cm.ResumeValidationUnavailable, match="checkpoint was preserved"):
        cm._prepare_resume(
            [info],
            data_root,
            state_root,
            payload,
            fingerprint,
            "h264",
            validation_mode="full",
            probe_timeout_seconds=0.01,
        )

    assert checkpoint.is_dir()
    assert marker.is_file()


def test_source_stat_change_changes_fingerprint(tmp_path: Path):
    info = _inspect(tmp_path)
    args = Namespace(
        category=[],
        partition=[],
        max_partitions=None,
        max_episodes=None,
        video_codec="h264",
        video_quality=18,
        video_preset="fast",
        encoder_threads=2,
    )
    output = tmp_path / "out"
    before = cm.canonical_fingerprint(cm._fingerprint_payload([info], tmp_path, output, args))
    stat = info.source_path.stat()
    info.source_path.touch()
    if info.source_path.stat().st_mtime_ns == stat.st_mtime_ns:
        pytest.skip("filesystem did not update mtime")
    after = cm.canonical_fingerprint(cm._fingerprint_payload([info], tmp_path, output, args))
    assert before != after


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_real_writer_preserves_parquet_dtypes(tmp_path: Path):
    info = _inspect(tmp_path)
    plan = info.plan
    output = convert_dataset(
        plan,
        lambda episode: iter_frames(plan, episode),
        reader_format="robomimic_hdf5",
    )
    parquet_path = next((output / "data").rglob("*.parquet"))
    schema = pq.read_schema(parquet_path)
    assert str(schema.field("action").type) == "fixed_size_list<element: double>[7]"
    assert str(schema.field("source.done").type) == "int64"
    assert str(schema.field("observation.robot0_contact").type) == "bool"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_real_collection_interrupt_resume_reuses_partition_and_cleans_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "raw" / "minicgen"
    first_source = source_root / "core" / "square_d0.hdf5"
    second_source = source_root / "core" / "square_d1.hdf5"
    _write_source(first_source)
    _write_source(second_source)
    output = tmp_path / "staging" / "lerobot_v3_0" / "mimicgen_resume_test"
    infos = [
        inspect_partition(
            path,
            raw_dataset_root=source_root,
            collection_output=output,
        )
        for path in (first_source, second_source)
    ]
    args = Namespace(
        skip_existing=False,
        resume=True,
        overwrite=False,
        category=[],
        partition=[],
        max_partitions=None,
        max_episodes=None,
        video_codec="h264",
        video_quality=18,
        video_preset="fast",
        encoder_threads=2,
        eta_interval_seconds=60.0,
    )
    real_convert = cm.convert_dataset
    calls = 0

    def interrupt_second(*call_args, **call_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated interruption")
        return real_convert(*call_args, **call_kwargs)

    monkeypatch.setattr(cm, "convert_dataset", interrupt_second)
    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        cm.convert_collection(infos, output, args, source_root)

    resume_data, resume_state, resume_lock = cm.resume_paths(output)
    assert (resume_data / infos[0].partition_name).is_dir()
    first_marker = resume_state / "partitions" / f"{infos[0].partition_name}.json"
    assert first_marker.is_file()
    assert cm.read_json_object(first_marker, "partition marker")["validation"]["file_fingerprint"]
    assert not (resume_data / infos[1].partition_name).exists()

    monkeypatch.setattr(cm, "convert_dataset", real_convert)
    cm.convert_collection(infos, output, args, source_root)

    assert output.is_dir()
    assert all((output / info.partition_name / "meta" / "info.json").is_file() for info in infos)
    assert not resume_data.exists()
    assert not resume_state.exists()
    assert not resume_lock.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_parallel_collection_is_semantically_equivalent_and_plan_ordered(tmp_path: Path):
    source_root = tmp_path / "raw" / "minicgen"
    sources = [
        source_root / "core" / "square_d0.hdf5",
        source_root / "core" / "square_d1.hdf5",
    ]
    _write_source(sources[0])
    _write_source(sources[1])
    plan_root = tmp_path / "plans" / "mimicgen_equivalence"
    infos = [
        inspect_partition(path, raw_dataset_root=source_root, collection_output=plan_root)
        for path in sources
    ]
    serial = tmp_path / "serial" / "mimicgen_equivalence"
    parallel = tmp_path / "parallel" / "mimicgen_equivalence"

    cm.convert_collection(infos, serial, _collection_args(resume=False, workers=1), source_root)
    cm.convert_collection(infos, parallel, _collection_args(resume=False, workers=2), source_root)

    evidence = cm.verify_collection_equivalence(
        infos, serial, parallel, video_codec="h264"
    )
    manifest = cm.read_json_object(parallel / "collection_manifest.json", "manifest")
    assert evidence == {"partitions": 2, "episodes": 4, "frames": 10, "video_files": 4}
    assert [row["path"] for row in manifest["partitions"]] == [
        info.partition_name for info in infos
    ]
    assert not list(tmp_path.rglob("*.worker-cache"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_parallel_worker_failure_preserves_validated_marker_and_resume_rebuilds_only_failure(
    tmp_path: Path,
):
    source_root = tmp_path / "raw" / "minicgen"
    sources = [
        source_root / "core" / "square_d0.hdf5",
        source_root / "core" / "square_d1.hdf5",
    ]
    for source in sources:
        _write_source(source)
    output = tmp_path / "staging" / "mimicgen_parallel_resume"
    good_infos = [
        inspect_partition(path, raw_dataset_root=source_root, collection_output=output)
        for path in sources
    ]
    missing_source = tmp_path / "missing.hdf5"
    bad_episodes = tuple(
        replace(
            episode,
            extra={**episode.extra, "source_path": missing_source},
        )
        for episode in good_infos[1].plan.episodes
    )
    bad_info = replace(
        good_infos[1],
        plan=replace(good_infos[1].plan, episodes=bad_episodes),
    )
    # Use a single deterministic dispatch slot so the valid first unit is
    # durably checkpointed before the injected second-unit failure.
    args = _collection_args(resume=True, workers=1)

    with pytest.raises((ConversionError, FileNotFoundError)):
        cm.convert_collection([good_infos[0], bad_info], output, args, source_root)

    resume_data, resume_state, resume_lock = cm.resume_paths(output)
    first_marker = resume_state / "partitions" / f"{good_infos[0].partition_name}.json"
    second_marker = resume_state / "partitions" / f"{good_infos[1].partition_name}.json"
    assert (resume_data / good_infos[0].partition_name).is_dir()
    assert first_marker.is_file()
    assert not second_marker.exists()

    cm.convert_collection(good_infos, output, args, source_root)

    assert output.is_dir()
    assert not resume_data.exists()
    assert not resume_state.exists()
    assert not resume_lock.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_direct_staged_conversion_publishes_sentinels_and_refuses_success_overwrite(
    tmp_path: Path,
):
    source_root = tmp_path / "raw" / "minicgen"
    source = source_root / "core" / "square_d0.hdf5"
    _write_source(source, lengths=(3, 2))
    root = tmp_path / "staging" / "lerobot_v3_0"
    layout = cm.build_runtime_layout(
        output_root=root,
        dataset_uid="mimicgen",
        run_id="sentinel-test",
    )
    info = inspect_partition(
        source,
        raw_dataset_root=source_root,
        collection_output=layout.output_path,
    )
    args = _staged_args(resume=False)

    cm.convert_collection_staged([info], layout, args, source_root)

    assert (layout.output_path / cm.SUCCESS_SENTINEL).is_file()
    assert not (layout.output_path / cm.INCOMPLETE_SENTINEL).exists()
    assert not layout.resume_dir.exists()
    assert not layout.work_dir.exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cm.convert_collection_staged([info], layout, args, source_root)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_direct_staged_capacity_failure_retains_incomplete_and_checkpoint_root(
    tmp_path: Path,
):
    source_root = tmp_path / "raw" / "minicgen"
    source = source_root / "core" / "square_d0.hdf5"
    _write_source(source, lengths=(2,))
    root = tmp_path / "staging" / "lerobot_v3_0"
    layout = cm.build_runtime_layout(
        output_root=root,
        dataset_uid="mimicgen",
        run_id="capacity-test",
    )
    info = inspect_partition(
        source,
        raw_dataset_root=source_root,
        collection_output=layout.output_path,
    )
    args = _staged_args(resume=False)
    args.max_staging_bytes = 1

    with pytest.raises(ConversionError, match="capacity guard stopped"):
        cm.convert_collection_staged([info], layout, args, source_root)

    assert (layout.output_path / cm.INCOMPLETE_SENTINEL).is_file()
    assert not (layout.output_path / cm.SUCCESS_SENTINEL).exists()
    assert (layout.resume_dir / "state.json").is_file()
    assert not layout.work_dir.exists()


def test_direct_staged_lock_conflict_cannot_mutate_active_run(tmp_path: Path):
    source_root = tmp_path / "raw" / "minicgen"
    source = source_root / "core" / "square_d0.hdf5"
    _write_source(source, lengths=(2,))
    root = tmp_path / "staging" / "lerobot_v3_0"
    layout = cm.build_runtime_layout(
        output_root=root,
        dataset_uid="mimicgen",
        work_dir=root / ".conversion_work" / "mimicgen" / "formal",
        run_id="lock-conflict",
    )
    info = inspect_partition(
        source,
        raw_dataset_root=source_root,
        collection_output=layout.output_path,
    )
    args = _staged_args(resume=True)
    layout.work_dir.mkdir(parents=True)
    active_work = layout.work_dir / "active-worker.bin"
    active_work.write_bytes(b"still active")
    layout.output_path.mkdir(parents=True)
    incomplete = layout.output_path / cm.INCOMPLETE_SENTINEL
    incomplete.write_text('{"owner":"first"}\n', encoding="utf-8")

    with exclusive_resume_lock(layout.lock_path):
        with pytest.raises(ConversionError, match="another resume process"):
            cm.convert_collection_staged([info], layout, args, source_root)

    assert active_work.read_bytes() == b"still active"
    assert incomplete.read_text(encoding="utf-8") == '{"owner":"first"}\n'
    assert not layout.logs_dir.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_direct_staged_worker_failure_resumes_only_invalid_partition(tmp_path: Path):
    source_root = tmp_path / "raw" / "minicgen"
    sources = [
        source_root / "core" / "square_d0.hdf5",
        source_root / "core" / "square_d1.hdf5",
    ]
    for source in sources:
        _write_source(source, lengths=(2,))
    root = tmp_path / "staging" / "lerobot_v3_0"
    first_layout = cm.build_runtime_layout(
        output_root=root,
        dataset_uid="mimicgen",
        run_id="failed-run",
    )
    infos = [
        inspect_partition(
            source,
            raw_dataset_root=source_root,
            collection_output=first_layout.output_path,
        )
        for source in sources
    ]
    missing_source = tmp_path / "missing.hdf5"
    bad_episodes = tuple(
        replace(
            episode,
            extra={**episode.extra, "source_path": missing_source},
        )
        for episode in infos[1].plan.episodes
    )
    bad_info = replace(infos[1], plan=replace(infos[1].plan, episodes=bad_episodes))
    args = _staged_args(resume=False, workers=1)

    with pytest.raises(ConversionError, match="parallel work unit"):
        cm.convert_collection_staged(
            [infos[0], bad_info], first_layout, args, source_root
        )

    first_marker = (
        first_layout.resume_dir / "partitions" / f"{infos[0].partition_name}.json"
    )
    assert first_marker.is_file()
    assert (first_layout.output_path / cm.INCOMPLETE_SENTINEL).is_file()
    reusable_file = next(
        path
        for path in (first_layout.output_path / infos[0].partition_name).rglob("*")
        if path.is_file()
    )
    reusable_mtime = reusable_file.stat().st_mtime_ns

    resume_layout = cm.build_runtime_layout(
        output_root=root,
        dataset_uid="mimicgen",
        run_id="resume-run",
    )
    resume_args = _staged_args(resume=True, workers=1)
    cm.convert_collection_staged(infos, resume_layout, resume_args, source_root)

    assert reusable_file.stat().st_mtime_ns == reusable_mtime
    assert (resume_layout.output_path / cm.SUCCESS_SENTINEL).is_file()
    assert not (resume_layout.output_path / cm.INCOMPLETE_SENTINEL).exists()
    assert not resume_layout.resume_dir.exists()
