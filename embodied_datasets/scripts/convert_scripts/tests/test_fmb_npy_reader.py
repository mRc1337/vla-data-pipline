from __future__ import annotations

from argparse import Namespace
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np

from convert_core.dataset_config import DatasetConversionConfig
from convert_core.errors import ConversionError
from convert_core.dataset_config import load_dataset_config
from convert_core.lerobot_writer import build_manifest
from convert_fmb_to_lerobot import _catalog_for_run
from readers.fmb_npy_reader import inspect_fmb, iter_fmb_frames


def _trajectory(*, multi: bool, frames: int = 3) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        **{f"obs/{name}": np.full((frames, 256, 256, 3), [1, 2, 3], dtype=np.uint8) for name in ("side_1", "side_2", "wrist_1", "wrist_2")},
        **{f"obs/{name}_depth": np.arange(frames * 256 * 256, dtype=np.uint16).reshape(frames, 256, 256) for name in ("side_1", "side_2", "wrist_1", "wrist_2")},
        "obs/tcp_pose": np.ones((frames, 7), dtype=np.float64),
        "obs/tcp_vel": np.ones((frames, 6), dtype=np.float64),
        "obs/tcp_force": np.ones((frames, 3), dtype=np.float64),
        "obs/tcp_torque": np.ones((frames, 3), dtype=np.float64),
        "obs/q": np.ones((frames, 7), dtype=np.float64),
        "obs/dq": np.ones((frames, 7), dtype=np.float64),
        "obs/jacobian": np.ones((frames, 6, 7), dtype=np.float64),
        "obs/gripper_pose": np.arange(frames, dtype=np.int64),
        "action": np.ones((frames, 7), dtype=np.float64),
        "primitive": np.asarray(["grasp", "insert", "release"][:frames]),
    }
    if multi:
        result["object_id"] = np.full((frames,), 4, dtype=np.int64)
    else:
        result["object_info"] = {
            "length": "S",
            "size": "M",
            "shape": "1",
            "color": "2",
            "angle": "horizontal",
            "distractor": "n",
        }
    return result


def _write_archive(root: Path, name: str, member: str, payload: dict[str, np.ndarray]) -> None:
    buffer = BytesIO()
    np.save(buffer, payload, allow_pickle=True)
    with zipfile.ZipFile(root / name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, buffer.getvalue())


def _config() -> DatasetConversionConfig:
    return DatasetConversionConfig(
        dataset_uid="functional_manipulation_benchmark_fmb",
        format="fmb_npy",
        robot_type="franka_panda",
        fps=10,
    )


def test_inspect_partitions_and_preserves_schema(tmp_path: Path) -> None:
    _write_archive(
        tmp_path,
        "single_object_manipulation.zip",
        "media/fmb/np_release/single_object_manipulation/insert_only_1_S_S_1_n_7.npy",
        _trajectory(multi=False),
    )
    _write_archive(
        tmp_path,
        "multi_object_manipulation_assembly_2.zip",
        "media/fmb/np_release/multi_object_manipulation/board_2/trajectory_4_8.npy",
        _trajectory(multi=True),
    )

    catalog = inspect_fmb(_config(), tmp_path)

    assert {partition.plan.extra["source_kind"] for partition in catalog.partitions} == {"single_object", "multi_object"}
    multi = next(partition for partition in catalog.partitions if partition.plan.extra["source_kind"] == "multi_object")
    single = next(partition for partition in catalog.partitions if partition.plan.extra["source_kind"] == "single_object")
    assert multi.plan.fps == 10
    assert multi.plan.num_frames == 3
    assert multi.plan.episodes[0].instruction == "Pick up the red object and insert it."
    assert single.plan.episodes[0].instruction == "Insert the rectangle object."
    assert single.plan.episodes[0].extra["object_info"]["shape"] == "1"
    assert "observation.depth.side_1" in multi.plan.feature_schema()
    assert multi.plan.feature_schema()["observation.primitive"]["dtype"] == "string"
    assert multi.plan.feature_schema()["observation.object_id"]["dtype"] == "int64"
    assert catalog.sample_evidence[0]["frames"] == 3
    assert catalog.fingerprint_payload["task_mapping"][0]["task"]
    camera_mapping = next(row for row in catalog.mapping_table if row["source_field"] == "obs/side_1")
    assert camera_mapping["lossy"] is True
    manifest = build_manifest(multi.plan, reader_format="fmb_npy")
    assert manifest["episodes"][0]["source_provenance"]["member"].endswith("trajectory_4_8.npy")
    assert "timestamp_provenance" in multi.plan.extra


def test_iter_frames_reverses_bgr_only_and_keeps_numeric_values(tmp_path: Path) -> None:
    member = "media/fmb/np_release/multi_object_manipulation/board_2/trajectory_4_8.npy"
    _write_archive(tmp_path, "multi_object_manipulation_assembly_2.zip", member, _trajectory(multi=True))
    catalog = inspect_fmb(_config(), tmp_path)
    partition = catalog.partitions[0]
    episode = partition.plan.episodes[0]
    frame = next(iter_fmb_frames(partition.plan, episode, tmp_path))

    assert frame["observation.images.side_1"][0, 0].tolist() == [3, 2, 1]
    assert frame["observation.depth.side_1"].dtype == np.uint16
    assert frame["action"].dtype == np.float64
    assert frame["observation.primitive"] == "grasp"
    assert frame["observation.gripper_pose"].shape == (1,)
    assert frame["observation.object_id"].tolist() == [4]


def test_plural_action_alias_is_canonicalized(tmp_path: Path) -> None:
    payload = _trajectory(multi=True)
    payload["actions"] = payload.pop("action")
    member = "media/fmb/np_release/multi_object_manipulation/board_2/trajectory_4_8.npy"
    _write_archive(tmp_path, "multi_object_manipulation_assembly_2.zip", member, payload)
    catalog = inspect_fmb(_config(), tmp_path)
    assert "action" in catalog.partitions[0].plan.feature_schema()
    assert "actions" not in catalog.partitions[0].plan.feature_schema()


def test_single_object_filename_is_retained_when_object_info_is_absent(tmp_path: Path) -> None:
    payload = _trajectory(multi=False)
    payload.pop("object_info")
    member = "media/fmb/np_release/single_object_manipulation/insert_only_1_S_S_1_horizontal_n_7.npy"
    _write_archive(tmp_path, "single_object_manipulation.zip", member, payload)
    catalog = inspect_fmb(_config(), tmp_path)
    provenance = catalog.partitions[0].plan.episodes[0].extra["object_info"]
    assert provenance == {
        "shape": "1",
        "size": "S",
        "length": "S",
        "color": "1",
        "angle": "horizontal",
        "distractor": "n",
    }


def test_incomplete_zip_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "broken.zip").write_bytes(b"PK\x03\x04partial")
    try:
        inspect_fmb(_config(), tmp_path)
    except ConversionError as exc:
        assert "incomplete or invalid" in str(exc)
    else:
        raise AssertionError("incomplete archive was accepted")


def test_schema_validation_covers_all_cameras_and_multi_object_id(tmp_path: Path) -> None:
    payload = _trajectory(multi=True)
    payload["obs/side_2"] = payload["obs/side_2"][:, :128]
    _write_archive(
        tmp_path,
        "multi_object_manipulation_assembly_2.zip",
        "media/fmb/np_release/multi_object_manipulation/board_2/trajectory_4_8.npy",
        payload,
    )
    try:
        inspect_fmb(_config(), tmp_path)
    except ConversionError as exc:
        assert "obs/side_2" in str(exc)
    else:
        raise AssertionError("invalid camera schema was accepted")

    missing_id_root = tmp_path / "missing-id"
    missing_id_root.mkdir()
    payload = _trajectory(multi=True)
    payload.pop("object_id")
    _write_archive(
        missing_id_root,
        "multi_object_manipulation_assembly_2.zip",
        "media/fmb/np_release/multi_object_manipulation/board_2/trajectory_4_8.npy",
        payload,
    )
    try:
        inspect_fmb(_config(), missing_id_root)
    except ConversionError as exc:
        assert "object_id" in str(exc)
    else:
        raise AssertionError("multi-object trajectory without object_id was accepted")


def test_unrecognized_payload_field_is_not_silently_dropped(tmp_path: Path) -> None:
    payload = _trajectory(multi=False)
    payload["obs/undocumented"] = np.zeros((3, 1), dtype=np.float64)
    _write_archive(
        tmp_path,
        "single_object_manipulation.zip",
        "media/fmb/np_release/single_object_manipulation/insert_only_1_S_S_1_n_7.npy",
        payload,
    )
    try:
        inspect_fmb(_config(), tmp_path)
    except ConversionError as exc:
        assert "would be dropped" in str(exc)
    else:
        raise AssertionError("unrecognized payload field was silently dropped")


def test_partial_shard_cache_is_not_reused_for_full_preflight(tmp_path: Path) -> None:
    for index in (1, 2):
        _write_archive(
            tmp_path,
            f"{index:02d}.zip",
            f"media/fmb/np_release/single_object_manipulation/task_{index}_1.npy",
            _trajectory(multi=False),
        )
    config_path = Path(__file__).parents[1] / "configs" / "functional_manipulation_benchmark_fmb.yaml"
    config = load_dataset_config(config_path)
    cache_root = tmp_path / "cache"
    partial_args = Namespace(config=config_path, max_shards=1)
    partial = _catalog_for_run(config, tmp_path, cache_root, partial_args)
    assert len(partial.partitions[0].entries) == 1

    full_args = Namespace(config=config_path, max_shards=None)
    full = _catalog_for_run(config, tmp_path, cache_root, full_args)
    assert len(full.partitions[0].entries) == 2


def test_schema_fingerprint_ignores_episode_length_and_unicode_width(tmp_path: Path) -> None:
    first = _trajectory(multi=False)
    second = _trajectory(multi=False, frames=2)
    second["primitive"] = np.asarray(["a", "bb"])
    member_prefix = "media/fmb/np_release/single_object_manipulation/"
    _write_archive(
        tmp_path,
        "01.zip",
        member_prefix + "task_1_1.npy",
        first,
    )
    _write_archive(
        tmp_path,
        "02.zip",
        member_prefix + "task_2_1.npy",
        second,
    )

    catalog = inspect_fmb(_config(), tmp_path)

    assert len(catalog.partitions) == 1
    assert len(catalog.partitions[0].entries) == 2


def test_worker_payload_is_revalidated_against_preflight(tmp_path: Path) -> None:
    member = "media/fmb/np_release/multi_object_manipulation/board_2/trajectory_4_8.npy"
    _write_archive(tmp_path, "multi_object_manipulation_assembly_2.zip", member, _trajectory(multi=True))
    catalog = inspect_fmb(_config(), tmp_path)
    partition = catalog.partitions[0]
    episode = partition.plan.episodes[0]
    from dataclasses import replace

    stale_episode = replace(episode, num_frames=episode.num_frames - 1)
    try:
        next(iter_fmb_frames(partition.plan, stale_episode, tmp_path))
    except ConversionError as exc:
        assert "payload has" in str(exc)
    else:
        raise AssertionError("changed payload length was not rejected")
