import pytest

lerobot = pytest.importorskip("lerobot")

from pathlib import Path


def test_video_clip_info_returns_distinct_windows_for_episodes_sharing_one_file(tmp_path: Path):
    from tests.fixtures import make_synthetic_dataset
    from video_locator import video_clip_info

    dataset_root = tmp_path / "synthetic_video_ds"
    make_synthetic_dataset(
        dataset_root, repo_id="test/locator", num_episodes=3, num_frames=5, include_video=True, fps=10.0,
    )

    clips = [video_clip_info(dataset_root, i, "observation.image") for i in range(3)]
    for clip in clips:
        assert clip is not None
    # All three tiny synthetic episodes are small enough to land in the same
    # physical mp4 file -- lerobot packs episodes into files up to
    # video_files_size_in_mb (default 200MB), so "one file per episode" is
    # not guaranteed. from_timestamp/to_timestamp, not the file path, is
    # what actually distinguishes one episode's clip from another's here.
    assert clips[0]["relative_path"] == clips[1]["relative_path"] == clips[2]["relative_path"]
    assert [c["from_timestamp"] for c in clips] == [0.0, 0.5, 1.0]
    assert [c["to_timestamp"] for c in clips] == [0.5, 1.0, 1.5]


def test_video_clip_info_returns_none_for_non_video_feature(tmp_path: Path):
    from tests.fixtures import make_synthetic_dataset
    from video_locator import video_clip_info

    dataset_root = tmp_path / "synthetic_no_video_ds"
    make_synthetic_dataset(dataset_root, repo_id="test/locator_novideo", num_episodes=1, num_frames=5)

    assert video_clip_info(dataset_root, 0, "observation.state") is None
    assert video_clip_info(dataset_root, 0, "observation.image") is None


def test_video_clip_info_returns_none_for_out_of_range_episode(tmp_path: Path):
    from tests.fixtures import make_synthetic_dataset
    from video_locator import video_clip_info

    dataset_root = tmp_path / "synthetic_video_ds_range"
    make_synthetic_dataset(
        dataset_root, repo_id="test/locator_range", num_episodes=1, num_frames=5, include_video=True,
    )

    assert video_clip_info(dataset_root, 5, "observation.image") is None
    assert video_clip_info(dataset_root, -1, "observation.image") is None
