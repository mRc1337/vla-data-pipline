import json
from pathlib import Path

import numpy as np
import pytest

from convert_core.dataset_config import DatasetConversionConfig
from convert_core.errors import ConversionError
from readers.one_x_world_model_reader import (
    TASK_PLACEHOLDER,
    V2_STATE_NAMES,
    OneXWorldModelReader,
)


def _config(**updates) -> DatasetConversionConfig:
    values = {
        "dataset_uid": "one_x_test",
        "format": "one_x_world_model",
        "source_directory": "1x_world_model_dataset",
        "robot_type": "eve",
        "one_x_version": "v2.0",
        "one_x_splits": ["train_v2.0"],
    }
    values.update(updates)
    return DatasetConversionConfig(**values)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_v2_shard(root: Path, index: int, segment_ids: list[int]) -> None:
    frames = len(segment_ids)
    _write_json(
        root / "metadata" / f"metadata_{index}.json",
        {"shard_num_frames": frames, "shard_ind": index},
    )
    for directory in ("robot_states", "segment_indices", "videos"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    np.arange(frames * 25, dtype=np.float32).reshape(frames, 25).tofile(
        root / "robot_states" / f"states_{index}.bin"
    )
    np.asarray(segment_ids, dtype=np.int32).tofile(
        root / "segment_indices" / f"segment_idx_{index}.bin"
    )
    blocks = (frames + 16) // 17
    np.zeros((blocks, 3, 32, 32), dtype=np.int32).tofile(
        root / "videos" / f"video_{index}.bin"
    )


def _write_revision(root: Path) -> None:
    metadata = root / ".cache" / "huggingface" / "download" / "README.md.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("42e3e12fff6848b511583ba6e8afa7f82ef9014e\netag\n0\n", encoding="utf-8")


def test_decoder_preflight_consumes_at_most_60_real_frames(monkeypatch):
    reader = OneXWorldModelReader()
    calls = []

    class Episode:
        num_frames = 100

    class Plan:
        episodes = (Episode(),)

    def frames(_plan, _episode):
        for index in range(100):
            calls.append(index)
            yield index

    monkeypatch.setattr(reader, "iter_frames", frames)
    reader.preflight_decoder(Plan())

    assert calls == list(range(60))


def test_v2_plan_merges_segment_across_shards_and_preserves_provenance(tmp_path: Path):
    source = tmp_path / "raw" / "1x_world_model_dataset"
    split = source / "train_v2.0"
    _write_revision(source)
    _write_json(split / "metadata.json", {"num_shards": 2, "query": None, "hz": 30, "num_images": 8})
    _write_v2_shard(split, 0, [0, 0, 1, 1])
    _write_v2_shard(split, 1, [1, 1, 2, 2])

    reader = OneXWorldModelReader()
    plan = reader.build_plan(_config(), tmp_path / "raw", tmp_path / "stage")

    assert [episode.episode_uid for episode in plan.episodes] == [
        "train_v2.0:segment:0",
        "train_v2.0:segment:1",
        "train_v2.0:segment:2",
    ]
    assert [episode.num_frames for episode in plan.episodes] == [2, 4, 2]
    assert len(plan.episodes[1].extra["spans"]) == 2
    assert plan.episodes[1].extra["checkpoint_unit"] == "train_v2.0/shard_00001"
    assert plan.episodes[0].instruction == TASK_PLACEHOLDER
    assert plan.vector_features[0].names == V2_STATE_NAMES
    assert plan.extra["source_splits"] == ["train_v2.0"]
    assert plan.extra["task_provenance"]["source_has_instruction"] is False
    assert plan.extra["partition_rules"]["partition_value"] == "v2.0"
    assert plan.extra["field_mapping"][0]["dtype_cast"] is None
    assert plan.extra["field_mapping"][0]["field_reordered"] is False
    assert plan.episodes[1].extra["source_spans"] == (
        {
            "start": 2,
            "end": 4,
            "shard_index": 0,
            "video": "train_v2.0/videos/video_0.bin",
            "state": "train_v2.0/robot_states/states_0.bin",
        },
        {
            "start": 0,
            "end": 2,
            "shard_index": 1,
            "video": "train_v2.0/videos/video_1.bin",
            "state": "train_v2.0/robot_states/states_1.bin",
        },
    )
    reader._v2_decoder = lambda tokens: np.zeros(
        (len(tokens), 17, 256, 256, 3), dtype=np.uint8
    )
    frame = next(reader.iter_frames(plan, plan.episodes[0]))
    assert isinstance(frame["observation.state"], np.ndarray)
    assert frame["observation.state"].shape == (25,)
    assert frame["observation.state"].dtype == np.float32
    assert frame["observation.images.head"].shape == (256, 256, 3)
    assert frame["observation.images.head"].dtype == np.uint8


def test_v2_plan_rejects_wrong_binary_size(tmp_path: Path):
    source = tmp_path / "raw" / "1x_world_model_dataset"
    split = source / "train_v2.0"
    _write_json(split / "metadata.json", {"num_shards": 1, "query": None, "hz": 30, "num_images": 4})
    _write_v2_shard(split, 0, [0, 0, 0, 0])
    (split / "robot_states" / "states_0.bin").write_bytes(b"broken")

    with pytest.raises(ConversionError, match="expected exactly"):
        OneXWorldModelReader().build_plan(_config(), tmp_path / "raw", tmp_path / "stage")


def test_v2_plan_rejects_nonfinite_state_and_unreferenced_file(tmp_path: Path):
    source = tmp_path / "raw" / "1x_world_model_dataset"
    split = source / "train_v2.0"
    _write_json(split / "metadata.json", {"num_shards": 1, "query": None, "hz": 30, "num_images": 4})
    _write_v2_shard(split, 0, [0, 0, 0, 0])
    states = np.memmap(
        split / "robot_states" / "states_0.bin", dtype=np.float32, mode="r+", shape=(4, 25)
    )
    states[2, 3] = np.nan
    states.flush()

    with pytest.raises(ConversionError, match="non-finite float at row 2, column 3"):
        OneXWorldModelReader().build_plan(_config(), tmp_path / "raw", tmp_path / "stage")

    states[2, 3] = 0.0
    states.flush()
    (split / "unreferenced.bin").write_bytes(b"ignored")
    with pytest.raises(ConversionError, match=r"unreferenced=\['unreferenced.bin'\]"):
        OneXWorldModelReader().build_plan(_config(), tmp_path / "raw", tmp_path / "stage")


def test_test_split_is_explicitly_rejected(tmp_path: Path):
    source = tmp_path / "raw" / "1x_world_model_dataset"
    source.mkdir(parents=True)

    with pytest.raises(ConversionError, match="17 decoded frames.*64 state rows"):
        OneXWorldModelReader().build_plan(
            _config(one_x_include_test=True), tmp_path / "raw", tmp_path / "stage"
        )


def test_duplicate_split_selection_is_rejected_before_source_is_read(tmp_path: Path):
    source = tmp_path / "raw" / "1x_world_model_dataset"
    source.mkdir(parents=True)

    with pytest.raises(ConversionError, match="duplicate v2.0 split selection"):
        OneXWorldModelReader().build_plan(
            _config(one_x_splits=["train_v2.0", "train_v2.0"]),
            tmp_path / "raw",
            tmp_path / "stage",
        )


def test_v1_plan_keeps_source_arrays_separate(tmp_path: Path):
    source = tmp_path / "raw" / "1x_world_model_dataset"
    split = source / "val_v1.1"
    source.mkdir(parents=True)
    (source / "magvit2.ckpt").write_bytes(b"fixture")
    _write_json(
        split / "metadata.json",
        {
            "token_dtype": "uint32",
            "s": 16,
            "h": 16,
            "w": 16,
            "vocab_size": 262144,
            "hz": 30,
            "num_images": 4,
        },
    )
    split.mkdir(parents=True, exist_ok=True)
    np.zeros((4, 16, 16), dtype=np.uint32).tofile(split / "video.bin")
    np.asarray([0, 0, 1, 1], dtype=np.int32).tofile(split / "segment_ids.bin")
    actions = split / "actions"
    actions.mkdir()
    for name, width in (
        ("joint_pos", 21),
        ("neck_desired", 3),
        ("driving_command", 2),
        ("l_hand_closure", 1),
        ("r_hand_closure", 1),
    ):
        np.zeros((4, width), dtype=np.float32).tofile(actions / f"{name}.bin")

    reader = OneXWorldModelReader()
    plan = reader.build_plan(
        _config(one_x_version="v1.1", one_x_splits=["val_v1.1"]),
        tmp_path / "raw",
        tmp_path / "stage",
    )

    assert len(plan.episodes) == 2
    assert set(plan.feature_schema()) == {
        "action.joint_position",
        "action.neck_desired",
        "action.driving_command",
        "action.left_hand_closure",
        "action.right_hand_closure",
        "observation.images.head",
    }
    assert plan.feature_schema()["action.neck_desired"]["names"] is None
    transforms = {row["lerobot"]: row["transform"] for row in plan.extra["field_mapping"]}
    assert transforms["action.joint_position"] == "none"
    assert transforms["action.neck_desired"] == "none"
    assert transforms["action.driving_command"] == "none"
    assert "length-one LeRobot array" in transforms["action.left_hand_closure"]
    assert "length-one LeRobot array" in transforms["action.right_hand_closure"]
    assert plan.extra["partition_rules"]["partition_value"] == "v1.1"
    assert all(row["dtype_cast"] is None for row in plan.extra["field_mapping"])
    assert all(row["field_reordered"] is False for row in plan.extra["field_mapping"])
    assert "command/target/measured status is unpublished" in plan.extra["field_mapping"][0][
        "role_evidence"
    ]

    reader._v1_decoder = lambda tokens: np.zeros(
        (len(tokens), 256, 256, 3), dtype=np.uint8
    )
    frame = next(reader.iter_frames(plan, plan.episodes[0]))
    for feature in plan.vector_features:
        value = frame[feature.feature_key]
        assert isinstance(value, np.ndarray)
        assert value.shape == feature.resolved_shape
        assert value.dtype == np.dtype(feature.dtype)
    assert frame["action.left_hand_closure"].shape == (1,)
    assert frame["action.right_hand_closure"].shape == (1,)

    invalid = dict(frame)
    invalid["action.left_hand_closure"] = np.float32(0)
    with pytest.raises(ConversionError, match="must be np.ndarray"):
        reader._validate_frame_contract(plan, plan.episodes[0], invalid)
