from pathlib import Path

import h5py
import numpy as np
import pytest

from convert_core.dataset_config import CameraFieldConfig, DatasetConversionConfig, VectorFieldConfig
from convert_core.errors import ConversionError
from readers.hdf5_reader import Hdf5Reader


def _write_episode(path: Path, *, num_frames: int = 4, fps: float = 20.0) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = np.arange(num_frames * 7, dtype=np.float32).reshape(num_frames, 7)
    action = state + 100.0
    # Deliberately BGR: after conversion the first RGB pixel is [30, 20, 10].
    rgb_bgr = np.zeros((num_frames, 8, 10, 3), dtype=np.uint8)
    rgb_bgr[..., 0] = 10
    rgb_bgr[..., 1] = 20
    rgb_bgr[..., 2] = 30

    with h5py.File(path, "w") as h5_file:
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=state)
        images = observations.create_group("images")
        camera = images.create_dataset("cam_high", data=rgb_bgr)
        camera.attrs["fps"] = fps
        h5_file.create_dataset("action", data=action)
    return {"state": state, "action": action, "rgb_bgr": rgb_bgr}


def _config(**overrides) -> DatasetConversionConfig:
    defaults: dict = dict(
        dataset_uid="single_arm_test",
        format="hdf5",
        robot_type="test_robot",
        vector_fields=[
            VectorFieldConfig(feature_key="observation.state", source_key="/observations/qpos", dim=7),
            VectorFieldConfig(feature_key="action", source_key="/action", dim=7),
        ],
    )
    defaults.update(overrides)
    return DatasetConversionConfig(**defaults)


def test_build_plan_discovers_camera_and_measures_fps(tmp_path: Path):
    episode_path = tmp_path / "raw" / "single_arm_test" / "pick up the cup" / "episode_0.h5"
    _write_episode(episode_path, fps=30.0)

    plan = Hdf5Reader().build_plan(_config(), tmp_path / "raw", tmp_path / "staging")

    assert plan.fps == 30
    assert plan.measured_fps == pytest.approx(30.0)
    assert plan.episodes[0].instruction == "pick up the cup"
    assert plan.episodes[0].num_frames == 4
    assert [camera.feature_key for camera in plan.camera_features] == ["observation.images.cam_high"]
    assert plan.output_path == tmp_path / "staging" / "lerobot_v3_0" / "single_arm_test"


def test_feature_schema_reflects_configured_vector_fields_and_camera(tmp_path: Path):
    episode_path = tmp_path / "raw" / "single_arm_test" / "task" / "episode_0.h5"
    _write_episode(episode_path)

    plan = Hdf5Reader().build_plan(_config(), tmp_path / "raw", tmp_path / "staging")
    features = plan.feature_schema()

    assert features["observation.state"]["shape"] == (7,)
    assert features["action"]["shape"] == (7,)
    assert features["observation.images.cam_high"]["dtype"] == "video"
    assert features["observation.images.cam_high"]["shape"] == (8, 10, 3)


def test_iter_frames_yields_expected_arrays_and_converts_bgr_to_rgb(tmp_path: Path):
    episode_path = tmp_path / "raw" / "single_arm_test" / "task" / "episode_0.h5"
    expected = _write_episode(episode_path)
    reader = Hdf5Reader()
    plan = reader.build_plan(_config(), tmp_path / "raw", tmp_path / "staging")

    frames = list(reader.iter_frames(plan, plan.episodes[0]))

    assert len(frames) == 4
    assert np.array_equal(frames[0]["observation.state"], expected["state"][0])
    assert np.array_equal(frames[0]["action"], expected["action"][0])
    assert frames[0]["task"] == "task"
    assert frames[0]["observation.images.cam_high"].shape == (8, 10, 3)
    assert frames[0]["observation.images.cam_high"][0, 0].tolist() == [30, 20, 10]


def test_vector_field_dimension_mismatch_raises(tmp_path: Path):
    episode_path = tmp_path / "raw" / "single_arm_test" / "task" / "episode_0.h5"
    _write_episode(episode_path)

    config = _config(
        vector_fields=[VectorFieldConfig(feature_key="observation.state", source_key="/observations/qpos", dim=99)]
    )
    with pytest.raises(ConversionError, match="must have shape"):
        Hdf5Reader().build_plan(config, tmp_path / "raw", tmp_path / "staging")


def test_missing_episodes_raises(tmp_path: Path):
    (tmp_path / "raw" / "single_arm_test").mkdir(parents=True)
    with pytest.raises(ConversionError, match="no episodes matched"):
        Hdf5Reader().build_plan(_config(), tmp_path / "raw", tmp_path / "staging")


def test_no_vector_fields_configured_raises(tmp_path: Path):
    episode_path = tmp_path / "raw" / "single_arm_test" / "task" / "episode_0.h5"
    _write_episode(episode_path)

    config = _config(vector_fields=[])
    with pytest.raises(ConversionError, match="vector_fields"):
        Hdf5Reader().build_plan(config, tmp_path / "raw", tmp_path / "staging")


def test_explicit_camera_config_overrides_auto_discovery(tmp_path: Path):
    episode_path = tmp_path / "raw" / "single_arm_test" / "task" / "episode_0.h5"
    _write_episode(episode_path)
    with h5py.File(episode_path, "r+") as h5_file:
        images = h5_file["/observations/images"]
        images.create_dataset("cam_low", data=np.zeros((4, 8, 10, 3), dtype=np.uint8)).attrs["fps"] = 20.0

    config = _config(cameras=[CameraFieldConfig(feature_key="observation.images.cam_high", source_key="/observations/images/cam_high")])
    plan = Hdf5Reader().build_plan(config, tmp_path / "raw", tmp_path / "staging")

    assert [camera.feature_key for camera in plan.camera_features] == ["observation.images.cam_high"]


def test_instruction_source_constant(tmp_path: Path):
    episode_path = tmp_path / "raw" / "single_arm_test" / "whatever" / "episode_0.h5"
    _write_episode(episode_path)

    config = _config(instruction_source="constant", instruction_constant="do the thing")
    plan = Hdf5Reader().build_plan(config, tmp_path / "raw", tmp_path / "staging")

    assert plan.episodes[0].instruction == "do the thing"
