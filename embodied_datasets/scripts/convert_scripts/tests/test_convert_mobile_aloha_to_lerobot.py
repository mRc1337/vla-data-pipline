import io
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from evaluate_mobile_aloha_conversion import evaluate
from convert_mobile_aloha_to_lerobot import (
    ARM_NAMES,
    BASE_ACTION_NAMES,
    ConversionError,
    EtaProgress,
    VideoEncodingConfig,
    _episode_arrays,
    _format_duration,
    _preflight_video_encoder,
    _read_rgb_frame,
    _resume_lock,
    _resume_paths,
    _streaming_queue_size,
    collection_summary,
    convert_collection,
    inspect_dataset,
    inspect_dataset_collection,
    main,
    plan_summary,
)


def test_eta_progress_reports_rate_elapsed_and_remaining_time():
    now = [100.0]
    stream = io.StringIO()
    progress = EtaProgress(
        "[mobile_aloha] convert",
        100,
        "frames",
        interval_seconds=5.0,
        clock=lambda: now[0],
        stream=stream,
    )

    now[0] = 110.0
    progress.update(20)
    line = stream.getvalue()
    assert "20/100 frames ( 20.0%)" in line
    assert "2.00 frames/s" in line
    assert "elapsed 00:00:10" in line
    assert "ETA 00:00:40" in line

    now[0] = 130.0
    progress.finish(context="complete")
    assert "100/100 frames (100.0%)" in stream.getvalue()
    assert "ETA 00:00:00 | complete" in stream.getvalue()


def test_eta_duration_formats_multi_day_runs():
    assert _format_duration(2 * 86400 + 3 * 3600 + 4 * 60 + 5) == "2d 03:04:05"


def _write_episode(
    path: Path,
    *,
    num_frames: int = 4,
    combined_action: bool = False,
    include_separate_base: bool = True,
    include_depth: bool = True,
    include_velocity: bool = True,
    include_effort: bool = False,
    camera_names: tuple[str, ...] = ("cam_high",),
    image_size: tuple[int, int] = (8, 10),
    fps: float = 20.0,
) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = np.arange(num_frames * 14, dtype=np.float32).reshape(num_frames, 14)
    arm_action = state + 100.0
    base_action = np.arange(num_frames * 2, dtype=np.float32).reshape(num_frames, 2) + 500.0
    action = np.concatenate([arm_action, base_action], axis=1) if combined_action else arm_action
    # Deliberately BGR: after conversion the first RGB pixel is [30, 20, 10].
    height, width = image_size
    rgb_bgr = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    rgb_bgr[..., 0] = 10
    rgb_bgr[..., 1] = 20
    rgb_bgr[..., 2] = 30

    with h5py.File(path, "w") as h5_file:
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=state)
        if include_velocity:
            observations.create_dataset("qvel", data=state / 10.0)
        if include_effort:
            observations.create_dataset("effort", data=state / 100.0)
        images = observations.create_group("images")
        for camera_name in camera_names:
            camera = images.create_dataset(camera_name, data=rgb_bgr)
            camera.attrs["fps"] = fps
        if include_depth:
            images.create_dataset(
                "cam_high_depth", data=np.zeros((num_frames, height, width), dtype=np.uint16)
            )
        h5_file.create_dataset("action", data=action)
        if include_separate_base:
            h5_file.create_dataset("base_action", data=base_action)
    return {"state": state, "arm_action": arm_action, "base_action": base_action, "rgb_bgr": rgb_bgr}


def _inspect(tmp_path: Path, uid: str = "mobile_aloha_test", **kwargs):
    return inspect_dataset(
        raw_root=tmp_path / "public_datasets_raw",
        staging_root=tmp_path / "public_datasets_staging",
        dataset_uid=uid,
        **kwargs,
    )


def test_inspect_preserves_instruction_filters_depth_and_uses_camera_fps(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "pick up the cup" / "episode_0.hdf5"
    _write_episode(episode_path, fps=30.0)

    plan = _inspect(tmp_path)

    assert plan.fps == 30
    assert plan.measured_fps == pytest.approx(30.0)
    assert plan.episodes[0].instruction == "pick up the cup"
    assert [camera.feature_key for camera in plan.episodes[0].cameras] == ["observation.images.cam_high"]
    assert plan.episodes[0].skipped_depth_keys == ("/observations/images/cam_high_depth",)
    assert plan.episodes[0].has_velocity is True
    assert plan.episodes[0].has_effort is False
    assert plan.output_path == tmp_path / "public_datasets_staging" / "lerobot_v3_0" / "mobile_aloha_test"


def test_public_release_directory_is_mapped_to_natural_language_instruction(tmp_path: Path):
    episode_path = (
        tmp_path
        / "public_datasets_raw"
        / "mobile_aloha_test"
        / "aloha_mobile_elevator_truncated"
        / "episode_0.hdf5"
    )
    _write_episode(episode_path)

    plan = _inspect(tmp_path)

    assert plan.episodes[0].instruction == "Take the elevator to the 1st floor."


def test_separate_action_is_loaded_as_independent_arm_and_base_features(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    expected = _write_episode(episode_path)
    plan = _inspect(tmp_path)

    with h5py.File(episode_path, "r") as h5_file:
        arrays = _episode_arrays(h5_file, plan, plan.episodes[0])

    assert plan.episodes[0].action_layout == "separate_14_plus_2"
    assert np.array_equal(arrays["action"], expected["arm_action"])
    assert np.array_equal(arrays["action.base"], expected["base_action"])


def test_combined_16d_action_is_split_without_separate_base(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    expected = _write_episode(episode_path, combined_action=True, include_separate_base=False)
    plan = _inspect(tmp_path)

    with h5py.File(episode_path, "r") as h5_file:
        arrays = _episode_arrays(h5_file, plan, plan.episodes[0])

    assert plan.episodes[0].action_layout == "combined_16"
    assert np.array_equal(arrays["action"], expected["arm_action"])
    assert np.array_equal(arrays["action.base"], expected["base_action"])


def test_combined_and_separate_base_mismatch_fails(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    _write_episode(episode_path, combined_action=True, include_separate_base=True)
    with h5py.File(episode_path, "r+") as h5_file:
        h5_file["/base_action"][0, 0] += 1.0

    with pytest.raises(ConversionError, match="disagree"):
        _inspect(tmp_path)


def test_uncompressed_bgr_is_converted_to_rgb(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    _write_episode(episode_path)
    plan = _inspect(tmp_path)
    camera = plan.episodes[0].cameras[0]

    with h5py.File(episode_path, "r") as h5_file:
        image = _read_rgb_frame(
            h5_file[camera.source_key],
            0,
            camera,
            source_path=episode_path,
            uncompressed_color_order="bgr",
        )

    assert image.shape == (8, 10, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [30, 20, 10]


def test_compressed_bgr_jpeg_is_decoded_as_rgb(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    _write_episode(episode_path)
    bgr = np.zeros((8, 10, 3), dtype=np.uint8)
    bgr[..., 0] = 10
    bgr[..., 1] = 20
    bgr[..., 2] = 30
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    with h5py.File(episode_path, "r+") as h5_file:
        images = h5_file["/observations/images"]
        del images["cam_high"]
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        camera = images.create_dataset("cam_high", shape=(4,), dtype=dtype)
        for index in range(4):
            camera[index] = encoded
        camera.attrs["fps"] = 20.0

    plan = _inspect(tmp_path)
    camera = plan.episodes[0].cameras[0]
    with h5py.File(episode_path, "r") as h5_file:
        image = _read_rgb_frame(
            h5_file[camera.source_key],
            0,
            camera,
            source_path=episode_path,
            uncompressed_color_order="rgb",
        )

    assert camera.storage == "compressed"
    assert image[0, 0].tolist() == [30, 20, 10]


def test_timestamp_fps_takes_precedence_over_attribute(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    _write_episode(episode_path, num_frames=4, fps=10.0)
    with h5py.File(episode_path, "r+") as h5_file:
        images = h5_file["/observations/images"]
        timestamps = images.create_group("timestamps")
        timestamps.create_dataset("cam_high", data=np.arange(4, dtype=np.float64) / 20.0)

    plan = _inspect(tmp_path)

    assert plan.fps == 20
    assert plan.episodes[0].fps_evidence[0].source.startswith("HDF5 timestamps")


def test_explicit_fps_is_only_a_fallback(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    _write_episode(episode_path, fps=25.0)

    plan = _inspect(tmp_path, fps=50.0)

    assert plan.fps == 25
    assert plan.episodes[0].fps_evidence[0].source.startswith("HDF5 attribute")


def test_missing_fps_uses_mobile_aloha_default_and_can_be_made_strict(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    _write_episode(episode_path)
    with h5py.File(episode_path, "r+") as h5_file:
        del h5_file["/observations/images/cam_high"].attrs["fps"]

    assert _inspect(tmp_path).fps == 50
    with pytest.raises(ConversionError, match="no FPS metadata"):
        _inspect(tmp_path, fps=None)


def test_episode_at_dataset_root_is_rejected_without_instruction(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "episode_0.hdf5"
    _write_episode(episode_path)

    with pytest.raises(ConversionError, match="no instruction subdirectory"):
        _inspect(tmp_path)


def test_camera_schema_must_be_identical_across_episodes(tmp_path: Path):
    root = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task"
    _write_episode(root / "episode_0.hdf5")
    _write_episode(root / "episode_1.hdf5")
    with h5py.File(root / "episode_1.hdf5", "r+") as h5_file:
        h5_file["/observations/images"].move("cam_high", "cam_other")

    with pytest.raises(ConversionError, match="camera schema"):
        _inspect(tmp_path)


def test_camera_frame_count_must_match_state(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    _write_episode(episode_path)
    with h5py.File(episode_path, "r+") as h5_file:
        images = h5_file["/observations/images"]
        del images["cam_high"]
        camera = images.create_dataset("cam_high", data=np.zeros((3, 8, 10, 3), dtype=np.uint8))
        camera.attrs["fps"] = 20.0

    with pytest.raises(ConversionError, match="has 3 frames, expected 4"):
        _inspect(tmp_path)


def test_plan_summary_exposes_independent_base_action_feature(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "nested" / "task" / "episode_0.hdf5"
    _write_episode(episode_path)

    summary = plan_summary(_inspect(tmp_path))

    assert summary["tasks"] == ["nested/task"]
    assert summary["features"]["action"]["shape"] == (14,)
    assert summary["features"]["action.base"]["shape"] == (2,)
    assert summary["features"]["observation.state"]["names"] == list(ARM_NAMES)
    assert summary["features"]["action"]["names"] == list(ARM_NAMES)
    assert summary["features"]["action.base"]["names"] == list(BASE_ACTION_NAMES)
    assert "observation.images.cam_high_depth" not in summary["features"]


def test_static_arm_only_episode_omits_base_action_feature(tmp_path: Path):
    episode_path = (
        tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "static task" / "episode_0.hdf5"
    )
    expected = _write_episode(episode_path, include_separate_base=False)

    plan = _inspect(tmp_path)
    with h5py.File(episode_path, "r") as h5_file:
        arrays = _episode_arrays(h5_file, plan, plan.episodes[0])

    assert plan.robot_type == "aloha_static"
    assert plan.episodes[0].action_layout == "arm_only_14"
    assert plan.episodes[0].has_base_action is False
    assert np.array_equal(arrays["action"], expected["arm_action"])
    assert "action.base" not in arrays
    assert "action.base" not in plan_summary(plan)["features"]


def test_collection_partitions_real_mobile_aloha_schema_variants(tmp_path: Path):
    root = tmp_path / "public_datasets_raw" / "mobile_aloha_test"
    mobile_cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    static_cameras = mobile_cameras + ("cam_low",)
    _write_episode(
        root / "aloha_mobile_cabinet" / "episode_0.hdf5",
        include_effort=True,
        camera_names=mobile_cameras,
    )
    _write_episode(
        root / "aloha_static_cotraining_datasets" / "with_effort" / "episode_0.hdf5",
        include_separate_base=False,
        include_effort=True,
        camera_names=static_cameras,
    )
    _write_episode(
        root / "aloha_static_cotraining_datasets" / "without_effort" / "episode_0.hdf5",
        include_separate_base=False,
        include_effort=False,
        camera_names=static_cameras,
    )

    collection = inspect_dataset_collection(
        raw_root=tmp_path / "public_datasets_raw",
        staging_root=tmp_path / "public_datasets_staging",
        dataset_uid="mobile_aloha_test",
    )

    assert len(collection.partitions) == 3
    assert collection.num_episodes == 3
    assert collection.num_frames == 12
    assert collection.output_path == (
        tmp_path / "public_datasets_staging" / "lerobot_v3_0" / "mobile_aloha_test"
    )
    schemas = [plan_summary(partition)["features"] for partition in collection.partitions]
    assert sum("action.base" in schema for schema in schemas) == 1
    assert sum("observation.effort" in schema for schema in schemas) == 2
    image_counts = [
        len([key for key in schema if key.startswith("observation.images.")])
        for schema in schemas
    ]
    assert sorted(image_counts) == [3, 4, 4]
    assert {partition.robot_type for partition in collection.partitions} == {
        "mobile_aloha",
        "aloha_static",
    }
    assert collection_summary(collection)["episodes"] == 3


def test_cli_inspect_only_partitions_without_writing(tmp_path: Path, capsys):
    root = tmp_path / "raw" / "mobile_aloha"
    _write_episode(root / "mobile" / "episode_0.hdf5")
    _write_episode(root / "static" / "episode_0.hdf5", include_separate_base=False)

    exit_code = main(
        [
            "--raw-root", str(tmp_path / "raw"),
            "--staging-root", str(tmp_path / "staging"),
            "--dataset-uid", "mobile_aloha",
            "--inspect-only",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"partitions": 2' in captured.out
    assert "ETA 00:00:00" in captured.err
    assert not (tmp_path / "staging").exists()


def test_episode_limit_builds_a_real_subset_plan(tmp_path: Path):
    root = tmp_path / "raw" / "mobile_aloha" / "task"
    for index in range(3):
        _write_episode(root / f"episode_{index}.hdf5")

    collection = inspect_dataset_collection(
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
        dataset_uid="mobile_aloha",
        episode_limit=1,
    )

    assert collection.num_episodes == 1
    assert collection.num_frames == 4
    assert collection.partitions[0].episodes[0].source_relative_path == "task/episode_0.hdf5"


def test_streaming_queue_must_hold_the_longest_complete_episode(tmp_path: Path):
    episode_path = tmp_path / "raw" / "mobile_aloha" / "task" / "episode_0.hdf5"
    _write_episode(episode_path, num_frames=4)
    plan = inspect_dataset(
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
        dataset_uid="mobile_aloha",
    )

    assert _streaming_queue_size(plan, VideoEncodingConfig(streaming=True)) == 5
    with pytest.raises(ConversionError, match="too small"):
        _streaming_queue_size(
            plan,
            VideoEncodingConfig(streaming=True, encoder_queue_maxsize=4),
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video encoding")
def test_mixed_collection_writes_and_reopens_real_lerobot_v3_datasets(tmp_path: Path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "raw" / "mobile_aloha"
    common = {"include_depth": False, "image_size": (64, 64)}
    _write_episode(root / "aloha_mobile_cabinet" / "episode_0.hdf5", include_effort=True, **common)
    _write_episode(
        root / "static_effort" / "episode_0.hdf5",
        include_separate_base=False,
        include_effort=True,
        **common,
    )
    _write_episode(
        root / "static_no_effort" / "episode_0.hdf5",
        include_separate_base=False,
        include_effort=False,
        **common,
    )
    collection = inspect_dataset_collection(
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
        dataset_uid="mobile_aloha",
    )

    output = convert_collection(collection)

    manifest = json.loads((output / "collection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "lerobot_v3_0_collection"
    assert manifest["num_partitions"] == 3
    assert manifest["num_episodes"] == 3
    assert manifest["num_frames"] == 12
    assert {partition.output_path.name for partition in collection.partitions} == {
        item["path"] for item in manifest["partitions"]
    }
    for partition in collection.partitions:
        dataset = LeRobotDataset(
            repo_id=partition.dataset_uid,
            root=output / partition.output_path.name,
        )
        assert dataset.num_episodes == 1
        assert len(dataset) == 4
        assert dataset.meta.features["observation.state"]["names"] == list(ARM_NAMES)
        assert dataset.meta.features["action"]["names"] == list(ARM_NAMES)
        if partition.robot_type == "mobile_aloha":
            assert dataset.meta.features["action.base"]["names"] == list(BASE_ACTION_NAMES)
            assert dataset.meta.tasks.index.tolist() == [
                "Open the top cabinet, store the pot inside it, then close the cabinet."
            ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video encoding")
def test_lossless_queue_streaming_bypasses_png_and_preserves_every_frame(tmp_path: Path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "raw" / "mobile_aloha"
    _write_episode(
        root / "aloha_mobile_cabinet" / "episode_0.hdf5",
        num_frames=8,
        include_depth=False,
        camera_names=("cam_high", "cam_left_wrist"),
        image_size=(64, 64),
    )
    collection = inspect_dataset_collection(
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
        dataset_uid="mobile_aloha",
    )
    encoding = VideoEncodingConfig(
        streaming=True,
        codec="h264",
        preset="fast",
    )

    output = convert_collection(collection, video_encoding=encoding)

    partition = collection.partitions[0]
    partition_root = output / partition.output_path.name
    assert not list(partition_root.rglob("*.png"))
    manifest = json.loads((partition_root / "conversion_manifest.json").read_text())
    assert manifest["video_encoding"]["streaming"] is True
    assert manifest["video_encoding"]["codec"] == "h264"
    assert manifest["conversion_metrics"]["frames_per_second"] > 0
    dataset = LeRobotDataset(repo_id=partition.dataset_uid, root=partition_root)
    assert len(dataset) == 8
    for key in dataset.meta.video_keys:
        assert dataset.meta.features[key]["info"]["video.codec"] == "h264"
    report = evaluate(partition_root, samples=4, min_psnr_db=30.0)
    assert report["pass"] is True
    assert report["numeric_pass"] is True
    assert all(item["frames_match"] for item in report["videos"].values())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video encoding")
def test_resume_continues_after_last_durable_episode(tmp_path: Path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "raw" / "mobile_aloha" / "task"
    for episode_index in range(3):
        _write_episode(
            root / f"episode_{episode_index}.hdf5",
            num_frames=5,
            include_depth=False,
            image_size=(64, 64),
        )
    collection = inspect_dataset_collection(
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
        dataset_uid="mobile_aloha",
    )
    encoding = VideoEncodingConfig(streaming=True, codec="h264", preset="fast")
    completed_calls: list[int] = []

    def interrupt_after_first_episode(plan, episode_index):
        completed_calls.append(episode_index)
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        convert_collection(
            collection,
            resume=True,
            video_encoding=encoding,
            episode_completed_hook=interrupt_after_first_episode,
        )

    collection_checkpoint, collection_state, _ = _resume_paths(collection.output_path)
    partition = collection.partitions[0]
    partition_output = collection_checkpoint / partition.output_path.name
    partition_checkpoint, partition_state, _ = _resume_paths(partition_output)
    assert collection_checkpoint.is_dir()
    assert collection_state.is_file()
    assert partition_checkpoint.is_dir()
    assert partition_state.is_file()
    checkpoint_dataset = LeRobotDataset(
        repo_id=partition.dataset_uid,
        root=partition_checkpoint,
    )
    assert checkpoint_dataset.num_episodes == 1
    assert len(checkpoint_dataset) == 5

    output = convert_collection(collection, resume=True, video_encoding=encoding)

    assert completed_calls == [0]
    assert output == collection.output_path
    assert not collection_checkpoint.exists()
    assert not collection_state.exists()
    dataset = LeRobotDataset(
        repo_id=partition.dataset_uid,
        root=output / partition.output_path.name,
    )
    assert dataset.num_episodes == 3
    assert len(dataset) == 15
    assert [int(row["length"]) for row in dataset.meta.episodes] == [5, 5, 5]
    expected_state = np.concatenate(
        [
            np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
            for _ in range(3)
        ]
    )
    assert np.array_equal(dataset.hf_dataset["observation.state"], expected_state)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video encoding")
def test_resume_rejects_changed_source_file(tmp_path: Path):
    pytest.importorskip("lerobot")
    episode_path = tmp_path / "raw" / "mobile_aloha" / "task" / "episode_0.hdf5"
    _write_episode(episode_path, include_depth=False, image_size=(64, 64))
    collection = inspect_dataset_collection(
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
        dataset_uid="mobile_aloha",
    )
    encoding = VideoEncodingConfig(streaming=True, codec="h264", preset="fast")

    def interrupt(plan, episode_index):
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError):
        convert_collection(
            collection,
            resume=True,
            video_encoding=encoding,
            episode_completed_hook=interrupt,
        )
    changed_encoding = VideoEncodingConfig(
        streaming=True,
        codec="h264",
        quality=31,
        preset="fast",
    )
    with pytest.raises(ConversionError, match="does not match"):
        convert_collection(collection, resume=True, video_encoding=changed_encoding)

    stat = episode_path.stat()
    os.utime(episode_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    changed_collection = inspect_dataset_collection(
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
        dataset_uid="mobile_aloha",
    )

    with pytest.raises(ConversionError, match="does not match"):
        convert_collection(changed_collection, resume=True, video_encoding=encoding)


def test_resume_lock_rejects_a_second_writer(tmp_path: Path):
    lock_path = tmp_path / ".dataset.resume.lock"
    with _resume_lock(lock_path):
        with pytest.raises(ConversionError, match="another conversion process"):
            with _resume_lock(lock_path):
                pass


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video encoding")
def test_real_h264_nvenc_streaming_when_hardware_is_available(tmp_path: Path):
    """Hardware gate: skipped on CPU/A100/A800, fully exercised on NVENC hosts."""
    pytest.importorskip("lerobot")
    root = tmp_path / "raw" / "mobile_aloha"
    _write_episode(
        root / "aloha_mobile_cabinet" / "episode_0.hdf5",
        num_frames=8,
        include_depth=False,
        camera_names=("cam_high", "cam_left_wrist"),
        image_size=(64, 64),
    )
    collection = inspect_dataset_collection(
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
        dataset_uid="mobile_aloha",
    )
    encoding = VideoEncodingConfig(streaming=True, codec="h264_nvenc")
    try:
        _preflight_video_encoder(collection.partitions[0], encoding)
    except ConversionError as exc:
        pytest.skip(f"NVENC runtime unavailable: {exc}")

    output = convert_collection(collection, video_encoding=encoding)
    partition_root = output / collection.partitions[0].output_path.name
    manifest = json.loads((partition_root / "conversion_manifest.json").read_text())
    assert manifest["video_encoding"]["codec"] == "h264_nvenc"
    assert not list(partition_root.rglob("*.png"))
