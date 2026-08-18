from __future__ import annotations

import json
from argparse import Namespace
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


def _write_source(path: Path, *, mismatched_second: bool = False) -> None:
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
    lengths = [3, 2]
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
