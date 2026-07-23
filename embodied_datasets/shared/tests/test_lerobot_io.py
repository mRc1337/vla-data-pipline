import pytest

lerobot = pytest.importorskip("lerobot")

from pathlib import Path

import numpy as np


def test_load_lerobot_episodes_round_trips_synthetic_dataset(tmp_path: Path):
    from shared.tests.fixtures import make_synthetic_dataset
    from shared.lerobot_io import load_lerobot_episodes

    dataset_root = tmp_path / "synthetic_ds"
    make_synthetic_dataset(dataset_root, repo_id="test/synthetic", num_episodes=2, num_frames=5, state_dim=3, action_dim=2)

    episodes = load_lerobot_episodes(dataset_root)
    assert len(episodes) == 2
    assert episodes[0].state.shape == (5, 3)
    assert episodes[0].action.shape == (5, 2)
    assert episodes[0].episode_index == 0


def test_load_lerobot_episodes_populates_language_instruction_from_task(tmp_path: Path):
    from shared.tests.fixtures import make_synthetic_dataset
    from shared.lerobot_io import load_lerobot_episodes

    dataset_root = tmp_path / "synthetic_ds_with_task"
    make_synthetic_dataset(
        dataset_root, repo_id="test/synthetic_task", num_episodes=1, num_frames=3, task="pick up the cup"
    )

    episodes = load_lerobot_episodes(dataset_root)
    assert len(episodes) == 1
    assert episodes[0].language_instruction == "pick up the cup"


def test_load_lerobot_episodes_maps_empty_task_to_none_instruction(tmp_path: Path):
    """The empty-string task is the fallback write_lerobot_episodes/run_pipeline.py
    use when Episode.language_instruction was originally None -- it must load back
    as None, not "", so check1_instruction_consistency.py's `not
    episode.language_instruction` gate treats it as absent."""
    from shared.tests.fixtures import make_synthetic_dataset
    from shared.lerobot_io import load_lerobot_episodes

    dataset_root = tmp_path / "synthetic_ds_empty_task"
    make_synthetic_dataset(dataset_root, repo_id="test/synthetic_empty_task", num_episodes=1, num_frames=3, task="")

    episodes = load_lerobot_episodes(dataset_root)
    assert len(episodes) == 1
    assert episodes[0].language_instruction is None


def test_load_lerobot_episodes_raises_on_inconsistent_per_frame_task(tmp_path: Path):
    """If a dataset's frames within one episode disagree on task text (a
    labeling artifact, partial re-label, etc.), load_lerobot_episodes must
    fail loudly with a ValueError identifying the mismatch rather than
    silently picking frame 0's task for the whole episode -- the existing
    make_synthetic_dataset fixture always writes one fixed task string per
    whole dataset, so this builds the dataset directly to vary the task
    across frames of the same episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from shared.lerobot_io import load_lerobot_episodes

    dataset_root = tmp_path / "synthetic_ds_inconsistent_task"
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id="test/synthetic_inconsistent_task", fps=10.0, root=dataset_root, features=features
    )
    tasks = ["pick up the cup", "pick up the cup", "put down the plate"]
    for task in tasks:
        dataset.add_frame(
            {
                "observation.state": np.zeros(2, dtype=np.float32),
                "action": np.zeros(2, dtype=np.float32),
                "task": task,
            }
        )
    dataset.save_episode()
    dataset.finalize()

    with pytest.raises(ValueError, match="inconsistent per-frame task text"):
        load_lerobot_episodes(dataset_root)


def test_load_lerobot_episodes_populates_video_frames(tmp_path: Path):
    from shared.tests.fixtures import make_synthetic_dataset
    from shared.lerobot_io import load_lerobot_episodes

    dataset_root = tmp_path / "synthetic_ds_with_video"
    make_synthetic_dataset(
        dataset_root, repo_id="test/synthetic_video", num_episodes=1, num_frames=4, include_video=True
    )

    episodes = load_lerobot_episodes(dataset_root)
    assert len(episodes) == 1
    episode = episodes[0]
    assert "observation.image" in episode.frames
    frames = episode.frames["observation.image"]
    assert frames.shape == (4, 32, 32, 3)
    assert frames.dtype == np.uint8
    # Video is lossy-compressed (confirmed empirically: random-noise frames
    # can shift by tens of levels per pixel after AV1 encoding), so a tight
    # per-pixel comparison against the written frames is not a meaningful
    # assertion here. Instead assert a coarse, genuinely non-vacuous
    # property that would fail if decoding silently produced garbage (e.g.
    # an all-zero or perfectly flat buffer): the decoded frames actually
    # vary across pixels/frames, i.e. real image content came back.
    assert frames.std() > 1.0
    assert not np.all(frames == frames[0, 0, 0])


def test_write_lerobot_episodes_produces_loadable_dataset(tmp_path: Path):
    from shared.episode import Episode
    from shared.lerobot_io import load_lerobot_episodes, write_lerobot_episodes

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
    from shared.episode import Episode
    from shared.lerobot_io import write_lerobot_episodes
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
    from shared.episode import Episode
    from shared.lerobot_io import write_lerobot_episodes
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
