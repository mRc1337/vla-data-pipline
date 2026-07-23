import pytest

pytest.importorskip("lerobot")

from pathlib import Path

from common.format_checkers import (
    CheckOutcome,
    check_custom,
    check_format,
    check_hdf5,
    check_lerobot,
    check_scale,
    check_video_decodable,
    find_video_files,
)


def test_check_hdf5_passes_when_action_and_obs_keys_present(tmp_path):
    import h5py

    raw_path = tmp_path / "raw"
    raw_path.mkdir(parents=True)
    with h5py.File(raw_path / "demo.hdf5", "w") as f:
        demo = f.create_group("data").create_group("demo_0")
        demo.create_group("obs").create_dataset("robot0_joint_pos", data=[[0.0]])
        demo.create_dataset("actions", data=[[0.0]])

    result = check_hdf5(raw_path)
    assert result.outcome == CheckOutcome.PASSED
    assert result.episode_count == 1


def test_check_hdf5_fails_when_required_keys_missing(tmp_path):
    import h5py

    raw_path = tmp_path / "raw"
    raw_path.mkdir(parents=True)
    with h5py.File(raw_path / "demo.hdf5", "w") as f:
        f.create_group("data").create_group("demo_0").create_dataset("nothing_useful", data=[0])

    result = check_hdf5(raw_path)
    assert result.outcome == CheckOutcome.FAILED


def test_check_hdf5_fails_when_file_cannot_be_opened(tmp_path):
    raw_path = tmp_path / "raw"
    raw_path.mkdir(parents=True)
    (raw_path / "demo.hdf5").write_bytes(b"not a real hdf5 file")

    result = check_hdf5(raw_path)
    assert result.outcome == CheckOutcome.FAILED


def test_check_hdf5_fails_when_no_files_found(tmp_path):
    raw_path = tmp_path / "raw"
    raw_path.mkdir(parents=True)

    result = check_hdf5(raw_path)
    assert result.outcome == CheckOutcome.FAILED


def test_check_lerobot_passes_on_valid_dataset(tmp_path):
    from tests.fixtures import make_synthetic_dataset

    raw_path = tmp_path / "raw"
    make_synthetic_dataset(raw_path, repo_id="test/verify_lerobot", num_episodes=3, num_frames=5)

    result = check_lerobot(raw_path)
    assert result.outcome == CheckOutcome.PASSED
    assert result.episode_count == 3


def test_check_lerobot_fails_when_directory_is_not_a_lerobot_dataset(tmp_path):
    raw_path = tmp_path / "raw"
    raw_path.mkdir(parents=True)
    (raw_path / "not_lerobot.txt").write_text("nope")

    result = check_lerobot(raw_path)
    assert result.outcome == CheckOutcome.FAILED


def test_check_custom_reports_no_checker_when_none_registered(tmp_path):
    result = check_custom(tmp_path, "some_unregistered_dataset")
    assert result.outcome == CheckOutcome.NO_CHECKER


def test_check_scale_passes_within_tolerance():
    result = check_scale(actual_size_gb=95.0, actual_episode_count=48, expected_size_gb=100.0, expected_num_episodes=50)
    assert result.outcome == CheckOutcome.PASSED


def test_check_scale_fails_when_size_way_off():
    result = check_scale(actual_size_gb=10.0, actual_episode_count=48, expected_size_gb=100.0, expected_num_episodes=50)
    assert result.outcome == CheckOutcome.FAILED


def test_check_scale_fails_when_episode_count_way_off():
    result = check_scale(actual_size_gb=95.0, actual_episode_count=5, expected_size_gb=100.0, expected_num_episodes=50)
    assert result.outcome == CheckOutcome.FAILED


def test_check_scale_skips_dims_not_declared():
    result = check_scale(actual_size_gb=95.0, actual_episode_count=None, expected_size_gb=None, expected_num_episodes=None)
    assert result.outcome == CheckOutcome.PASSED


def test_find_video_files_matches_known_extensions_only(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    result = find_video_files(tmp_path)
    assert result == [tmp_path / "clip.mp4"]


def test_check_video_decodable_reports_no_checker_when_ffprobe_missing(monkeypatch):
    import shutil as shutil_module

    monkeypatch.setattr(shutil_module, "which", lambda tool: None)
    result = check_video_decodable([Path("/nonexistent/video.mp4")])
    assert result.outcome == CheckOutcome.NO_CHECKER


def test_check_video_decodable_passes_when_ffprobe_succeeds(monkeypatch):
    import shutil as shutil_module
    import subprocess as subprocess_module

    monkeypatch.setattr(shutil_module, "which", lambda tool: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        subprocess_module,
        "run",
        lambda *a, **k: subprocess_module.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
    )
    result = check_video_decodable([Path("/fake/video.mp4")])
    assert result.outcome == CheckOutcome.PASSED


def test_check_video_decodable_fails_when_ffprobe_reports_error(monkeypatch):
    import shutil as shutil_module
    import subprocess as subprocess_module

    monkeypatch.setattr(shutil_module, "which", lambda tool: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        subprocess_module,
        "run",
        lambda *a, **k: subprocess_module.CompletedProcess(args=a, returncode=1, stdout="", stderr="moov atom not found"),
    )
    result = check_video_decodable([Path("/fake/corrupt.mp4")])
    assert result.outcome == CheckOutcome.FAILED


def test_check_format_dispatches_hdf5_to_check_hdf5(tmp_path):
    import h5py

    raw_path = tmp_path / "raw"
    raw_path.mkdir(parents=True)
    with h5py.File(raw_path / "demo.hdf5", "w") as f:
        demo = f.create_group("data").create_group("demo_0")
        demo.create_group("obs")
        demo.create_dataset("actions", data=[[0.0]])

    result = check_format(raw_path, "HDF5", "test_id")
    assert result.outcome == CheckOutcome.PASSED


def test_check_format_dispatches_custom_to_no_checker(tmp_path):
    result = check_format(tmp_path, "Custom", "unregistered")
    assert result.outcome == CheckOutcome.NO_CHECKER


def test_check_format_falls_back_to_no_checker_for_unimplemented_format(tmp_path):
    result = check_format(tmp_path, "VRS", "some_id")
    assert result.outcome == CheckOutcome.NO_CHECKER
