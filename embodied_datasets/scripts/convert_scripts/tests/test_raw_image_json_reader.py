import json
from pathlib import Path

import numpy as np
import pytest

from convert_core.dataset_config import CameraFieldConfig, DatasetConversionConfig, VectorFieldConfig
from convert_core.errors import ConversionError
from readers.raw_image_json_reader import RawImageJsonReader


def _write_episode(episode_dir: Path, *, num_frames: int = 3, instruction: str = "pick up") -> None:
    cv2 = pytest.importorskip("cv2")
    episode_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(num_frames):
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[..., 0] = 10  # R
        image[..., 1] = 20  # G
        image[..., 2] = 30  # B
        bgr = image[..., ::-1]
        filename = f"frame_{index:04d}.png"
        cv2.imwrite(str(episode_dir / filename), bgr)
        frames.append({"state": [float(index)] * 4, "action": [float(index)] * 4, "image": filename})
    metadata = {"language_instruction": instruction, "fps": 10, "frames": frames}
    (episode_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _config(**overrides) -> DatasetConversionConfig:
    defaults: dict = dict(
        dataset_uid="raw_test",
        format="raw_image_json",
        robot_type="test_robot",
        instruction_source="field",
        vector_fields=[
            VectorFieldConfig(feature_key="observation.state", source_key="state", dim=4),
            VectorFieldConfig(feature_key="action", source_key="action", dim=4),
        ],
        cameras=[CameraFieldConfig(feature_key="observation.images.primary", source_key="image")],
    )
    defaults.update(overrides)
    return DatasetConversionConfig(**defaults)


def test_build_plan_reads_metadata_and_discovers_camera_shape(tmp_path: Path):
    episode_dir = tmp_path / "raw" / "raw_test" / "episode_0"
    _write_episode(episode_dir)

    plan = RawImageJsonReader().build_plan(_config(), tmp_path / "raw", tmp_path / "staging")

    assert plan.fps == 10
    assert plan.episodes[0].instruction == "pick up"
    assert plan.episodes[0].num_frames == 3
    assert plan.camera_features[0].height == 6
    assert plan.camera_features[0].width == 8
    assert plan.output_path == tmp_path / "staging" / "lerobot_v3_0" / "raw_test"


def test_iter_frames_yields_expected_vectors_task_and_image(tmp_path: Path):
    episode_dir = tmp_path / "raw" / "raw_test" / "episode_0"
    _write_episode(episode_dir)
    reader = RawImageJsonReader()
    plan = reader.build_plan(_config(), tmp_path / "raw", tmp_path / "staging")

    frames = list(reader.iter_frames(plan, plan.episodes[0]))

    assert len(frames) == 3
    assert np.array_equal(frames[1]["observation.state"], np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    assert frames[0]["task"] == "pick up"
    assert frames[0]["observation.images.primary"].shape == (6, 8, 3)
    assert frames[0]["observation.images.primary"][0, 0].tolist() == [10, 20, 30]


def test_glob_sentinel_uses_natural_sorted_files_without_per_frame_filenames(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    episode_dir = tmp_path / "raw" / "raw_test" / "episode_0"
    episode_dir.mkdir(parents=True)
    for index in range(3):
        image = np.full((4, 4, 3), index * 10, dtype=np.uint8)
        cv2.imwrite(str(episode_dir / f"frame_{index:04d}.png"), image)
    metadata = {
        "language_instruction": "task",
        "fps": 5,
        "frames": [{"state": [0.0], "action": [0.0]} for _ in range(3)],
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    config = _config(
        vector_fields=[
            VectorFieldConfig(feature_key="observation.state", source_key="state", dim=1),
            VectorFieldConfig(feature_key="action", source_key="action", dim=1),
        ],
        cameras=[CameraFieldConfig(feature_key="observation.images.primary", source_key="$glob")],
        image_glob="*.png",
    )
    reader = RawImageJsonReader()
    plan = reader.build_plan(config, tmp_path / "raw", tmp_path / "staging")
    frames = list(reader.iter_frames(plan, plan.episodes[0]))

    assert len(frames) == 3
    assert frames[1]["observation.images.primary"][0, 0].tolist() == [10, 10, 10]


def test_missing_metadata_field_raises(tmp_path: Path):
    episode_dir = tmp_path / "raw" / "raw_test" / "episode_0"
    _write_episode(episode_dir)

    config = _config(
        vector_fields=[VectorFieldConfig(feature_key="observation.state", source_key="not_a_field", dim=4)]
    )
    with pytest.raises(ConversionError, match="missing field"):
        RawImageJsonReader().build_plan(config, tmp_path / "raw", tmp_path / "staging")


def test_camera_shape_mismatch_across_episodes_raises(tmp_path: Path):
    _write_episode(tmp_path / "raw" / "raw_test" / "episode_0")
    cv2 = pytest.importorskip("cv2")
    episode_1 = tmp_path / "raw" / "raw_test" / "episode_1"
    episode_1.mkdir(parents=True)
    # Different resolution than episode_0's 6x8 camera.
    bigger = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(episode_1 / "frame_0000.png"), bigger)
    (episode_1 / "metadata.json").write_text(
        json.dumps(
            {
                "language_instruction": "task",
                "fps": 10,
                "frames": [{"state": [0.0] * 4, "action": [0.0] * 4, "image": "frame_0000.png"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConversionError, match="camera shapes"):
        RawImageJsonReader().build_plan(_config(), tmp_path / "raw", tmp_path / "staging")


def test_no_metadata_json_raises(tmp_path: Path):
    (tmp_path / "raw" / "raw_test").mkdir(parents=True)
    with pytest.raises(ConversionError, match="metadata.json"):
        RawImageJsonReader().build_plan(_config(), tmp_path / "raw", tmp_path / "staging")


def test_instruction_source_defaults_to_path_parent(tmp_path: Path):
    episode_dir = tmp_path / "raw" / "raw_test" / "pick up the cup" / "episode_0"
    _write_episode(episode_dir)

    config = _config(instruction_source="path_parent")
    plan = RawImageJsonReader().build_plan(config, tmp_path / "raw", tmp_path / "staging")

    assert plan.episodes[0].instruction == "pick up the cup/episode_0"
