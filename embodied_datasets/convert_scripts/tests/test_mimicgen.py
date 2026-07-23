import pytest

pytest.importorskip("lerobot")

from pathlib import Path

import h5py
import numpy as np

from common.schema import DatasetConfig
from mimicgen import convert
from shared.lerobot_io import load_lerobot_episodes

URDF_PATH = str(Path(__file__).parent / "fixtures" / "simple_arm.urdf")


def _make_synthetic_robomimic_hdf5(path: Path, num_demos: int = 2, num_frames: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        data_group = f.create_group("data")
        rng = np.random.RandomState(0)
        for demo_index in range(num_demos):
            demo_group = data_group.create_group(f"demo_{demo_index}")
            obs_group = demo_group.create_group("obs")
            obs_group.create_dataset("robot0_joint_pos", data=rng.uniform(-0.1, 0.1, size=(num_frames, 2)))
            obs_group.create_dataset("robot0_gripper_qpos", data=rng.uniform(0, 1, size=(num_frames, 2)))
            demo_group.create_dataset("actions", data=rng.uniform(-1, 1, size=(num_frames, 7)))


def test_convert_reads_hdf5_and_writes_staging_dataset(tmp_path):
    raw_path = tmp_path / "raw" / "mimicgen"
    _make_synthetic_robomimic_hdf5(raw_path / "demo.hdf5", num_demos=2, num_frames=5)
    output_path = tmp_path / "staging" / "mimicgen"

    config = DatasetConfig(id="mimicgen", name="MimicGen Test", dof_per_arm=2, fps=20.0, urdf_available=False)

    report = convert(raw_path, output_path, config)

    assert report.num_episodes == 2
    assert report.num_frames == 10
    assert report.urdf_path is None

    episodes = load_lerobot_episodes(output_path)
    assert len(episodes) == 2
    # [joint(2) | eef_pos(3) + eef_quat(4) | gripper(2)]
    assert episodes[0].state.shape == (5, 2 + 3 + 4 + 2)
    assert episodes[0].action.shape == (5, 7)


def test_convert_computes_eef_pos_via_fk_when_urdf_available(tmp_path):
    data_root = tmp_path / "data_root"
    raw_path = data_root / "raw" / "mimicgen"
    _make_synthetic_robomimic_hdf5(raw_path / "demo.hdf5", num_demos=1, num_frames=3)
    with h5py.File(raw_path / "demo.hdf5", "a") as f:
        # Zero every joint for this test so FK's expected position is the
        # fixture URDF's fully-extended [1.5, 0, 0] (see
        # shared/tests/fixtures/simple_arm.urdf).
        f["data"]["demo_0"]["obs"]["robot0_joint_pos"][...] = 0.0
    output_path = data_root / "staging" / "mimicgen"

    urdf_dir = data_root / "urdf_assets" / "franka_panda"
    urdf_dir.mkdir(parents=True)
    (urdf_dir / "franka_panda.urdf").write_text(Path(URDF_PATH).read_text())

    config = DatasetConfig(
        id="mimicgen", name="MimicGen Test", dof_per_arm=2, fps=20.0, urdf_available=True, robot_platform="franka_panda"
    )

    report = convert(raw_path, output_path, config)

    assert report.urdf_path == str(urdf_dir / "franka_panda.urdf")
    episodes = load_lerobot_episodes(output_path)
    assert np.allclose(episodes[0].state[:, 2:5], [1.5, 0.0, 0.0], atol=1e-6)


def test_convert_warns_when_raw_joint_width_mismatches_declared_dof_per_arm(tmp_path):
    raw_path = tmp_path / "raw" / "mimicgen"
    _make_synthetic_robomimic_hdf5(raw_path / "demo.hdf5", num_demos=1, num_frames=3)
    output_path = tmp_path / "staging" / "mimicgen"

    # Fixture writes 2-wide robot0_joint_pos, declare dof_per_arm=7 (mimicgen's
    # real value) to trigger the mismatch warning.
    config = DatasetConfig(id="mimicgen", name="MimicGen Test", dof_per_arm=7, fps=20.0, urdf_available=False)

    report = convert(raw_path, output_path, config)

    assert any("!= declared dof_per_arm 7" in warning for warning in report.warnings)
