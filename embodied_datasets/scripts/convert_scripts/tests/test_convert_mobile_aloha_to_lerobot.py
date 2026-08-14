from pathlib import Path

import h5py
import numpy as np
import pytest

from convert_mobile_aloha_to_lerobot import (
    ConversionError,
    _episode_arrays,
    _read_rgb_frame,
    inspect_dataset,
    plan_summary,
)


def _write_episode(
    path: Path,
    *,
    num_frames: int = 4,
    combined_action: bool = False,
    include_separate_base: bool = True,
    include_depth: bool = True,
    fps: float = 20.0,
) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = np.arange(num_frames * 14, dtype=np.float32).reshape(num_frames, 14)
    arm_action = state + 100.0
    base_action = np.arange(num_frames * 2, dtype=np.float32).reshape(num_frames, 2) + 500.0
    action = np.concatenate([arm_action, base_action], axis=1) if combined_action else arm_action
    # Deliberately BGR: after conversion the first RGB pixel is [30, 20, 10].
    rgb_bgr = np.zeros((num_frames, 8, 10, 3), dtype=np.uint8)
    rgb_bgr[..., 0] = 10
    rgb_bgr[..., 1] = 20
    rgb_bgr[..., 2] = 30

    with h5py.File(path, "w") as h5_file:
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=state)
        observations.create_dataset("qvel", data=state / 10.0)
        images = observations.create_group("images")
        camera = images.create_dataset("cam_high", data=rgb_bgr)
        camera.attrs["fps"] = fps
        if include_depth:
            images.create_dataset("cam_high_depth", data=np.zeros((num_frames, 8, 10), dtype=np.uint16))
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


def test_missing_fps_requires_explicit_fallback(tmp_path: Path):
    episode_path = tmp_path / "public_datasets_raw" / "mobile_aloha_test" / "task" / "episode_0.hdf5"
    _write_episode(episode_path)
    with h5py.File(episode_path, "r+") as h5_file:
        del h5_file["/observations/images/cam_high"].attrs["fps"]

    with pytest.raises(ConversionError, match="no FPS metadata"):
        _inspect(tmp_path)
    assert _inspect(tmp_path, fps=50.0).fps == 50


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
    assert "observation.images.cam_high_depth" not in summary["features"]
