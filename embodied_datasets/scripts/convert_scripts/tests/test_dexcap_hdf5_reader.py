from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from readers.dexcap_hdf5_reader import (  # noqa: E402
    PARTITION_SPECS,
    inspect_partition,
    iter_frames,
)


def _write_fixture(root: Path) -> Path:
    path = root / PARTITION_SPECS[0].filename
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for episode_index, frames in enumerate((4, 3)):
            demo = data.create_group(f"demo_{episode_index}")
            pos = np.arange(frames * 6, dtype=np.float64).reshape(frames, 6)
            quat = np.arange(frames * 8, dtype=np.float64).reshape(frames, 8)
            hand = np.arange(frames * 32, dtype=np.float64).reshape(frames, 32)
            target = np.concatenate((pos, quat, hand), axis=1)
            demo.create_dataset("actions", data=np.vstack((target[1:], target[-1])))
            demo.create_dataset("glove_states", data=np.zeros((frames, 63), dtype=np.float64))
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_image", data=np.zeros((frames, 84, 84, 3), dtype=np.uint8))
            obs.create_dataset("label", data=np.arange(frames, dtype=np.int64))
            obs.create_dataset("pointcloud", data=np.zeros((frames, 10_000, 6), dtype=np.float64))
            obs.create_dataset("robot0_eef_hand", data=hand)
            obs.create_dataset("robot0_eef_pos", data=pos)
            obs.create_dataset("robot0_eef_quat", data=quat)
            demo.create_dataset("rewards", data=np.zeros(frames, dtype=np.float64))
            demo.create_dataset("states", data=np.zeros((frames, 16), dtype=np.float64))
            demo.create_dataset("dones", data=np.array([0] * (frames - 1) + [1], dtype=np.int64))
            demo.attrs["num_samples"] = frames
        data.attrs["total"] = 2
    return path


def test_inspect_checks_all_metadata_and_limits_selected_plan(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path)
    info = inspect_partition(
        source,
        raw_dataset_root=tmp_path,
        collection_output=tmp_path / "out",
        max_phase_groups=1,
    )
    assert info.all_episode_count == 2
    assert info.all_frame_count == 7
    assert len(info.plan.episodes) == 1
    assert info.plan.episodes[0].num_frames == 4
    assert info.plan.vector_features[-1].feature_key == "source.label"
    assert info.plan.camera_features[0].feature_key == "observation.images.agentview"


def test_iter_frames_preserves_source_arrays_and_action_alignment(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path)
    info = inspect_partition(
        source,
        raw_dataset_root=tmp_path,
        collection_output=tmp_path / "out",
        max_phase_groups=1,
    )
    plan = info.plan
    episode = plan.episodes[0]
    frames = list(iter_frames(plan, episode))
    assert len(frames) == 4
    assert frames[0]["action"].shape == (46,)
    assert frames[0]["observation.pointcloud"].shape == (10_000, 6)
    assert frames[0]["observation.images.agentview"].shape == (84, 84, 3)
    assert frames[0]["source.reward"].shape == (1,)
    np.testing.assert_array_equal(
        frames[0]["action"],
        np.concatenate(
            (
                frames[1]["observation.eef_position"],
                frames[1]["observation.eef_quaternion"],
                frames[1]["observation.eef_hand"],
            )
        ),
    )
