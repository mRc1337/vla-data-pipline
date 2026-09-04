from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from convert_core.errors import ConversionError
from readers import arcap_hdf5_reader as arcap


def _write_source(path: Path, *, invalid_done: bool = False) -> None:
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["mean_init_arm"] = np.zeros(7)
        data.attrs["mean_init_hand"] = np.zeros(1)
        data.attrs["mean_init_pos"] = np.zeros(3)
        data.attrs["mean_init_quat"] = np.array([0.0, 0.0, 0.0, 1.0])
        data.attrs["total"] = 12
        for episode_index in range(6):
            demo = data.create_group(f"demo_{episode_index}")
            demo.attrs["num_samples"] = 2
            obs = demo.create_group("obs")
            arm = rng.normal(size=(2, 7))
            hand = rng.normal(size=(2, 1))
            eef_pos = rng.normal(size=(2, 3))
            eef_quat = rng.normal(size=(2, 4))
            pointcloud = rng.random(size=(2, arcap.POINT_COUNT, 6))
            pointcloud[..., :3] -= 0.5
            obs.create_dataset("robot0_arm_joints", data=arm)
            obs.create_dataset("robot0_hand_joints", data=hand)
            obs.create_dataset("robot0_eef_pos", data=eef_pos)
            obs.create_dataset("robot0_eef_quat", data=eef_quat)
            obs.create_dataset("pointcloud", data=pointcloud)
            joint = np.concatenate((arm, hand), axis=1)
            eef = np.concatenate((eef_pos, eef_quat, hand), axis=1)
            demo.create_dataset("actions", data=np.stack((joint[1], joint[1])))
            demo.create_dataset("actions2", data=np.stack((eef[1], eef[1])))
            dones = np.zeros(2, dtype=np.int64)
            if episode_index % 3 == 2 or (invalid_done and episode_index == 0):
                dones[-1] = 1
            demo.create_dataset("dones", data=dones)
            demo.create_dataset("rewards", data=np.zeros(2, dtype=np.float64))
            demo.create_dataset("states", data=np.zeros(2, dtype=np.float64))


def _patch_spec(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    spec = replace(
        arcap.PARTITIONS_BY_NAME["assemble"],
        filename=path.name,
        source_bytes=path.stat().st_size,
    )
    monkeypatch.setitem(arcap.PARTITIONS_BY_FILE, path.name, spec)


def test_inspect_and_iter_preserve_float64(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "assemble_test.hdf5"
    _write_source(source)
    _patch_spec(monkeypatch, source)
    info = arcap.inspect_partition(
        source,
        raw_dataset_root=tmp_path,
        collection_output=tmp_path / "out",
        full_lowdim_scan=True,
    )
    assert len(info.plan.episodes) == 6
    assert info.plan.num_frames == 12
    assert info.payload_scan["phase_groups"] == 2
    frames = list(arcap.iter_frames(info.plan, info.plan.episodes[0]))
    assert len(frames) == 2
    assert frames[0]["observation.pointcloud"].shape == (10_000, 6)
    assert frames[0]["observation.pointcloud"].dtype == np.float64
    assert frames[0]["task"] == info.spec.instruction


def test_full_scan_rejects_two_done_episodes_in_phase_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "assemble_bad.hdf5"
    _write_source(source, invalid_done=True)
    _patch_spec(monkeypatch, source)
    with pytest.raises(ConversionError, match="done flags"):
        arcap.inspect_partition(
            source,
            raw_dataset_root=tmp_path,
            collection_output=tmp_path / "out",
            full_lowdim_scan=True,
        )
