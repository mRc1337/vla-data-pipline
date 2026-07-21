import pytest

lerobot = pytest.importorskip("lerobot")

from pathlib import Path

import numpy as np

from common.io import load_process_config, save_process_config
from common.schema import ProcessConfig


def test_save_then_load_round_trip(tmp_path: Path):
    config = ProcessConfig(id="droid", residual_threshold=0.1, quantile_low=0.02)
    path = tmp_path / "droid.yaml"
    save_process_config(config, path)

    loaded = load_process_config(path)
    assert loaded.id == "droid"
    assert loaded.residual_threshold == 0.1
    assert loaded.quantile_low == 0.02


def test_save_creates_parent_dirs(tmp_path: Path):
    config = ProcessConfig(id="droid")
    path = tmp_path / "nested" / "droid.yaml"
    save_process_config(config, path)
    assert path.exists()


def test_round_trip_preserves_optional_scalar_field(tmp_path: Path):
    """Test that Optional scalar fields (urdf_path) round-trip correctly."""
    config = ProcessConfig(id="droid", urdf_path="/path/to/robot.urdf")
    path = tmp_path / "droid.yaml"
    save_process_config(config, path)

    loaded = load_process_config(path)
    assert loaded.urdf_path == "/path/to/robot.urdf"


def test_round_trip_preserves_optional_nested_dict_field(tmp_path: Path):
    """Test that Optional nested dict fields with int keys round-trip correctly.

    This exercises the trickiest case: extreme_value_bounds has type
    Optional[Dict[str, Dict[int, List[float]]]], which means the inner dict
    has int keys. YAML/JSON force these to strings on dump, so pydantic must
    coerce them back to int on load.
    """
    config = ProcessConfig(
        id="droid",
        extreme_value_bounds={
            "state": {0: [-1.0, 1.0], 3: [0.0, 255.0]},
            "action": {1: [-0.5, 0.5]},
        },
    )
    path = tmp_path / "droid.yaml"
    save_process_config(config, path)

    loaded = load_process_config(path)
    # Verify the nested dict with int keys round-trips correctly
    assert loaded.extreme_value_bounds is not None
    assert loaded.extreme_value_bounds == {
        "state": {0: [-1.0, 1.0], 3: [0.0, 255.0]},
        "action": {1: [-0.5, 0.5]},
    }
    # Verify inner keys are actually int, not str
    assert all(isinstance(k, int) for k in loaded.extreme_value_bounds["state"].keys())
    assert all(isinstance(k, int) for k in loaded.extreme_value_bounds["action"].keys())


def test_load_lerobot_episodes_round_trips_synthetic_dataset(tmp_path: Path):
    from tests.fixtures import make_synthetic_dataset
    from common.io import load_lerobot_episodes

    dataset_root = tmp_path / "synthetic_ds"
    make_synthetic_dataset(dataset_root, repo_id="test/synthetic", num_episodes=2, num_frames=5, state_dim=3, action_dim=2)

    episodes = load_lerobot_episodes(dataset_root)
    assert len(episodes) == 2
    assert episodes[0].state.shape == (5, 3)
    assert episodes[0].action.shape == (5, 2)
    assert episodes[0].episode_index == 0


def test_write_lerobot_episodes_produces_loadable_dataset(tmp_path: Path):
    from common.episode import Episode
    from common.io import load_lerobot_episodes, write_lerobot_episodes

    episodes = [
        Episode(
            episode_index=0,
            timestamps=np.arange(4, dtype=np.float64) / 10.0,
            state=np.zeros((4, 2), dtype=np.float32),
            action=np.zeros((4, 2), dtype=np.float32),
        )
    ]
    output_path = tmp_path / "written_ds"
    write_lerobot_episodes(episodes, output_path, fps=10.0, robot_type="test_robot")

    reloaded = load_lerobot_episodes(output_path)
    assert len(reloaded) == 1
    assert reloaded[0].state.shape == (4, 2)


def test_write_lerobot_episodes_without_canonical_mask_has_no_mask_feature(tmp_path: Path):
    """Explicit regression coverage for the canonical_mask=None default: no
    caller of write_lerobot_episodes existing before this feature was added
    should see a new feature appear in the written dataset."""
    from common.episode import Episode
    from common.io import write_lerobot_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = [
        Episode(
            episode_index=0,
            timestamps=np.arange(3, dtype=np.float64) / 10.0,
            state=np.zeros((3, 2), dtype=np.float32),
            action=np.zeros((3, 2), dtype=np.float32),
        )
    ]
    output_path = tmp_path / "written_no_mask_ds"
    write_lerobot_episodes(episodes, output_path, fps=10.0, robot_type="test_robot")

    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    row = reloaded[0]
    assert "observation.state_canonical_mask" not in row


def test_write_lerobot_episodes_with_canonical_mask_adds_mask_feature(tmp_path: Path):
    from common.episode import Episode
    from common.io import write_lerobot_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    mask = np.zeros(80, dtype=bool)
    mask[:10] = True
    mask[58:61] = True
    episodes = [
        Episode(
            episode_index=0,
            timestamps=np.arange(4, dtype=np.float64) / 10.0,
            state=np.zeros((4, 80), dtype=np.float32),
            action=np.zeros((4, 2), dtype=np.float32),
        ),
        Episode(
            episode_index=1,
            timestamps=np.arange(3, dtype=np.float64) / 10.0,
            state=np.ones((3, 80), dtype=np.float32),
            action=np.zeros((3, 2), dtype=np.float32),
        ),
    ]
    output_path = tmp_path / "written_mask_ds"
    write_lerobot_episodes(episodes, output_path, fps=10.0, robot_type="test_robot", canonical_mask=mask)

    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    assert len(reloaded) == 4 + 3
    for row in (reloaded[0], reloaded[4]):  # one frame from each episode
        assert "observation.state_canonical_mask" in row
        written_mask = row["observation.state_canonical_mask"].numpy().astype(bool)
        assert written_mask.shape == (80,)
        assert np.array_equal(written_mask, mask)
