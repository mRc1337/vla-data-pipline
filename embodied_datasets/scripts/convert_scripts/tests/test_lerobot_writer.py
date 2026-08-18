import copy
import json
from pathlib import Path

import pytest

from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import build_manifest, validate_info_json


def _plan(root: Path) -> DatasetConversionPlan:
    return DatasetConversionPlan(
        dataset_uid="metadata_test",
        output_path=root,
        fps=30,
        measured_fps=30.0,
        robot_type="eve",
        vector_features=(
            VectorFeatureSpec(
                feature_key="observation.state",
                dim=2,
                names=("first", "second"),
            ),
        ),
        camera_features=(
            CameraFeatureSpec(
                feature_key="observation.images.head",
                height=256,
                width=256,
            ),
        ),
        episodes=(
            EpisodePlan(
                episode_uid="source:7",
                source_relative_path="split/segment:7",
                instruction="second task encountered first",
                num_frames=2,
                extra={"source_split": "split", "source_segment_id": 7},
            ),
            EpisodePlan(
                episode_uid="source:8",
                source_relative_path="split/segment:8",
                instruction="alphabetically first",
                num_frames=3,
                extra={"source_split": "split", "source_segment_id": 8},
            ),
        ),
    )


def _info(plan: DatasetConversionPlan) -> dict:
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["first", "second"],
            },
            "observation.images.head": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 256,
                    "video.width": 256,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": 30,
                    "video.channels": 3,
                    "has_audio": False,
                    "is_depth_map": False,
                },
            },
        },
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 2,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": "eve",
        "splits": {"train": "0:2"},
    }


def _write_info(root: Path, value: dict) -> None:
    path = root / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_info_json_validator_covers_counts_templates_names_and_video_metadata(
    tmp_path: Path,
):
    plan = _plan(tmp_path)
    valid = _info(plan)
    _write_info(tmp_path, valid)

    assert validate_info_json(plan, tmp_path, 2) == valid

    corruptions = [
        ("codebase_version", lambda value: value.__setitem__("codebase_version", "v2.1")),
        ("total_frames", lambda value: value.__setitem__("total_frames", 4)),
        ("data_path", lambda value: value.__setitem__("data_path", "wrong/{index}")),
        (
            "names",
            lambda value: value["features"]["observation.state"].__setitem__(
                "names", ["second", "first"]
            ),
        ),
        (
            "video.width",
            lambda value: value["features"]["observation.images.head"]["info"].__setitem__(
                "video.width", 128
            ),
        ),
    ]
    for expected_error, corrupt in corruptions:
        value = copy.deepcopy(valid)
        corrupt(value)
        _write_info(tmp_path, value)
        with pytest.raises(ConversionError, match=expected_error):
            validate_info_json(plan, tmp_path, 2)


def test_manifest_records_explicit_output_indices_and_source_task(tmp_path: Path):
    manifest = build_manifest(_plan(tmp_path), reader_format="fixture")

    assert manifest["task_index_mapping"] == {
        "0": "second task encountered first",
        "1": "alphabetically first",
    }
    assert manifest["episodes"][0]["lerobot_episode_index"] == 0
    assert manifest["episodes"][0]["lerobot_task_index"] == 0
    assert manifest["episodes"][0]["source_task"] is None
    assert manifest["episodes"][1]["lerobot_episode_index"] == 1
    assert manifest["episodes"][1]["lerobot_task_index"] == 1
