import pytest

pytest.importorskip("lerobot")

from common.schema import DatasetConfig
from nvidia_physicalai_robotics_manipulation_singlearm import convert
from shared.lerobot_io import load_lerobot_episodes
from tests.fixtures import make_synthetic_dataset


def test_convert_relayouts_raw_state_into_canonical_column_order(tmp_path):
    data_root = tmp_path / "data_root"
    raw_path = data_root / "raw" / "nvidia_physicalai_robotics_manipulation_singlearm"
    # Raw state = [joint(2) | gripper(1)] per this reference's documented
    # assumption (see module docstring) -- state_dim=3 for this synthetic test.
    make_synthetic_dataset(raw_path, repo_id="test/singlearm", num_episodes=2, num_frames=4, state_dim=3, action_dim=3, fps=30.0)
    output_path = data_root / "staging" / "nvidia_physicalai_robotics_manipulation_singlearm"

    config = DatasetConfig(
        id="nvidia_physicalai_robotics_manipulation_singlearm",
        name="Test",
        dof_per_arm=2,
        fps=30.0,
        urdf_available=False,
    )

    report = convert(raw_path, output_path, config)

    assert report.num_episodes == 2
    assert report.num_frames == 8
    assert report.urdf_path is None

    episodes = load_lerobot_episodes(output_path)
    # [joint(2) | eef_pos(3) + eef_quat(4) | gripper(1)]
    assert episodes[0].state.shape == (4, 2 + 3 + 4 + 1)
    assert episodes[0].action.shape == (4, 3)


def test_convert_skips_episodes_narrower_than_dof_per_arm_with_warning(tmp_path):
    data_root = tmp_path / "data_root"
    raw_path = data_root / "raw" / "nvidia_physicalai_robotics_manipulation_singlearm"
    # state_dim=2 (not 1): a width-1 observation.state feature round-trips
    # through the installed lerobot/datasets as a 0-d scalar instead of a
    # (1,)-shaped array (verified empirically), which crashes
    # load_lerobot_episodes before this test's actual narrower-than-
    # dof_per_arm path is even reached. 2 < dof_per_arm=3 below still
    # exercises the same skip-with-warning behavior without hitting that.
    make_synthetic_dataset(raw_path, repo_id="test/singlearm_narrow", num_episodes=1, num_frames=4, state_dim=2, action_dim=3, fps=30.0)
    output_path = data_root / "staging" / "nvidia_physicalai_robotics_manipulation_singlearm"

    config = DatasetConfig(
        id="nvidia_physicalai_robotics_manipulation_singlearm",
        name="Test",
        dof_per_arm=3,
        fps=30.0,
        urdf_available=False,
    )

    report = convert(raw_path, output_path, config)

    assert report.num_episodes == 0
    assert any("skipping" in warning for warning in report.warnings)
