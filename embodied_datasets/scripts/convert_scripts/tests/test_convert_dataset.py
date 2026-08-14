from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import yaml

import convert_dataset as cd

# LeRobotDataset.create(..., use_videos=True) encodes every episode's camera
# frames as an actual video via PyAV/ffmpeg during save_episode() -- unlike
# convert_mobile_aloha_to_lerobot.py's own test suite, which only exercises
# inspect_dataset()/_episode_arrays()/_read_rgb_frame() and never calls the
# real write path, these tests are (as far as this repo goes) the first ones
# to actually invoke it. requirements.txt already documents that system
# ffmpeg is required for real video encode/decode and is not pip-installable
# -- skip rather than fail on a dev machine that doesn't have it (this one
# doesn't; see PIPELINE_STATUS.md), and rely on the server (the documented
# real deployment target) to actually exercise this path.
_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(
    not _FFMPEG_AVAILABLE, reason="ffmpeg is not installed; LeRobotDataset video encoding requires it"
)


def _write_episode(path: Path, *, num_frames: int = 4, fps: float = 20.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = np.arange(num_frames * 5, dtype=np.float32).reshape(num_frames, 5)
    action = state + 10.0
    # LeRobot's default video encoder (SVT-AV1) divides by zero internally on
    # very small frame sizes -- 64x64 is the smallest size confirmed to work
    # (see PIPELINE_STATUS.md); the convert_mobile_aloha_to_lerobot.py test
    # suite's 8x10 fixtures never hit this because none of its tests call
    # the real write path.
    rgb = np.zeros((num_frames, 64, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 1
    rgb[..., 1] = 2
    rgb[..., 2] = 3
    with h5py.File(path, "w") as h5_file:
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=state)
        images = observations.create_group("images")
        camera = images.create_dataset("cam", data=rgb)
        camera.attrs["fps"] = fps
        h5_file.create_dataset("action", data=action)


def _write_config(path: Path, *, dataset_uid: str) -> None:
    config = {
        "dataset_uid": dataset_uid,
        "format": "hdf5",
        "robot_type": "test_robot",
        "vector_fields": [
            {"feature_key": "observation.state", "source_key": "/observations/qpos", "dim": 5},
            {"feature_key": "action", "source_key": "/action", "dim": 5},
        ],
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_dry_run_validates_without_writing_output(tmp_path: Path, capsys):
    _write_episode(tmp_path / "raw" / "cli_test" / "task" / "episode_0.h5")
    config_path = tmp_path / "cli_test.yaml"
    _write_config(config_path, dataset_uid="cli_test")

    exit_code = cd.main(
        [
            "--config", str(config_path),
            "--raw-root", str(tmp_path / "raw"),
            "--staging-root", str(tmp_path / "staging"),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "staging").exists()
    assert "validated 1 dataset" in capsys.readouterr().out


@requires_ffmpeg
def test_full_run_writes_a_real_lerobot_dataset(tmp_path: Path):
    _write_episode(tmp_path / "raw" / "cli_test" / "task" / "episode_0.h5")
    config_path = tmp_path / "cli_test.yaml"
    _write_config(config_path, dataset_uid="cli_test")

    exit_code = cd.main(
        [
            "--config", str(config_path),
            "--raw-root", str(tmp_path / "raw"),
            "--staging-root", str(tmp_path / "staging"),
        ]
    )

    assert exit_code == 0
    output_path = tmp_path / "staging" / "lerobot_v3_0" / "cli_test"
    assert output_path.is_dir()
    assert (output_path / "conversion_manifest.json").is_file()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="cli_test", root=output_path)
    assert dataset.num_episodes == 1
    assert len(dataset) == 4
    assert int(dataset.meta.fps) == 20


@requires_ffmpeg
def test_skip_existing_does_not_reconvert(tmp_path: Path):
    _write_episode(tmp_path / "raw" / "cli_test" / "task" / "episode_0.h5")
    config_path = tmp_path / "cli_test.yaml"
    _write_config(config_path, dataset_uid="cli_test")
    common_args = [
        "--config", str(config_path),
        "--raw-root", str(tmp_path / "raw"),
        "--staging-root", str(tmp_path / "staging"),
    ]

    assert cd.main(common_args) == 0
    output_path = tmp_path / "staging" / "lerobot_v3_0" / "cli_test"
    written_at = output_path.stat().st_mtime

    assert cd.main(common_args + ["--skip-existing"]) == 0
    assert output_path.stat().st_mtime == written_at


@requires_ffmpeg
def test_overwrite_requires_flag_and_then_replaces_output(tmp_path: Path):
    _write_episode(tmp_path / "raw" / "cli_test" / "task" / "episode_0.h5", num_frames=4)
    config_path = tmp_path / "cli_test.yaml"
    _write_config(config_path, dataset_uid="cli_test")
    common_args = [
        "--config", str(config_path),
        "--raw-root", str(tmp_path / "raw"),
        "--staging-root", str(tmp_path / "staging"),
    ]
    assert cd.main(common_args) == 0

    assert cd.main(common_args) == 1  # output already exists, no --overwrite

    _write_episode(tmp_path / "raw" / "cli_test" / "task" / "episode_1.h5", num_frames=4)
    assert cd.main(common_args + ["--overwrite"]) == 0
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id="cli_test", root=tmp_path / "staging" / "lerobot_v3_0" / "cli_test"
    )
    assert dataset.num_episodes == 2


@requires_ffmpeg
def test_all_mode_continues_past_one_bad_config_and_reports_nonzero_exit(tmp_path: Path, capsys):
    _write_episode(tmp_path / "raw" / "good_uid" / "task" / "episode_0.h5")
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    _write_config(configs_dir / "good.yaml", dataset_uid="good_uid")
    _write_config(configs_dir / "bad.yaml", dataset_uid="missing_uid")

    exit_code = cd.main(
        [
            "--configs-dir", str(configs_dir),
            "--all",
            "--raw-root", str(tmp_path / "raw"),
            "--staging-root", str(tmp_path / "staging"),
        ]
    )

    assert exit_code == 1
    assert (tmp_path / "staging" / "lerobot_v3_0" / "good_uid").is_dir()
    assert "bad.yaml" in capsys.readouterr().err
