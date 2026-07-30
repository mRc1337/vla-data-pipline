"""Tests for run_pipeline.py: run_dataset orchestration and main(). See
docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
section 10.
"""
import pytest

lerobot = pytest.importorskip("lerobot")


def test_run_dataset_end_to_end_with_synthetic_data(tmp_path):
    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/e2e", num_episodes=3, num_frames=30, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    # Stage1's savgol residual/accel/jerk thresholds default to values tuned
    # for real (smooth-ish) robot trajectories; make_synthetic_dataset's
    # per-frame uniform(-1, 1) noise has no such smoothness, so residual/
    # accel/jerk on this data run up to ~1-2 in magnitude (measured
    # empirically) -- well past the tight defaults (0.05/0.5/5.0 -- residual
    # in particular). Loosen all three well past that so stage1 doesn't
    # reject every synthetic episode outright, letting this smoke test
    # actually exercise stage2 onward rather than trivially passing on an
    # empty survivors list. State and action are independently random in
    # this fixture, so stage2's cross-correlation lag estimate is
    # essentially uniform noise too -- max_lag_frames=30 (>= num_frames)
    # keeps the lag gate from rejecting on that noise. quantile_low/high=0/1
    # similarly disable stage3's extreme-value filtering (which needs real
    # distributional structure, not pure per-frame noise, to be meaningful)
    # and da_threshold=0.0 disables stage2's directional-agreement gate for
    # the same reason.
    save_process_config(
        ProcessConfig(
            id="e2e_test",
            episode_reject_threshold=0.9,
            residual_threshold=5.0,
            accel_threshold=5.0,
            jerk_threshold=5.0,
            da_threshold=0.0,
            max_lag_frames=30,
            quantile_low=0.0,
            quantile_high=1.0,
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    stats = run_dataset(staging_path, output_path, config_path)

    assert stats["input_episodes"] == 3
    assert stats["output_episodes"] == 3
    assert stats["output_frames"] > 0
    assert len(stats["log"]) > 0
    assert output_path.exists()


def test_run_dataset_writes_configured_fps_not_hardcoded_default(tmp_path):
    """write_lerobot_episodes derives every frame's timestamp from
    frame_index / fps, so a hardcoded fps=1.0 (the pre-fix behavior)
    corrupts the written dataset's temporal metadata. run_dataset must
    thread config.fps through instead.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/fps", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="fps_test",
            fps=25.0,
            episode_reject_threshold=0.9,
            residual_threshold=5.0,
            accel_threshold=5.0,
            jerk_threshold=5.0,
            da_threshold=0.0,
            max_lag_frames=10,
            quantile_low=0.0,
            quantile_high=1.0,
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    stats = run_dataset(staging_path, output_path, config_path)

    assert stats["fps"] == 25.0
    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    assert reloaded.meta.info.fps == 25.0
    # frame_index / fps: frame 1's timestamp should be 1/25 = 0.04s, not the
    # pre-fix hardcoded 1/1.0 = 1.0s.
    assert reloaded[1]["timestamp"].item() == pytest.approx(1 / 25.0)


def test_run_dataset_falls_back_to_default_fps_when_unset(tmp_path):
    """`ProcessConfig.fps` is Optional -- a config author may not set it.
    run_dataset must fall back to a sensible default rather than crashing
    (None/0 would break frame_index / fps in lerobot)."""
    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/nofps", num_episodes=1, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    config = ProcessConfig(
        id="nofps_test",
        episode_reject_threshold=0.9,
        residual_threshold=5.0,
        accel_threshold=5.0,
        jerk_threshold=5.0,
        da_threshold=0.0,
        max_lag_frames=10,
        quantile_low=0.0,
        quantile_high=1.0,
    )
    assert config.fps is None
    save_process_config(config, config_path)

    output_path = tmp_path / "output"
    stats = run_dataset(staging_path, output_path, config_path)  # must not raise

    assert stats["fps"] == 1.0


def test_run_dataset_filters_episodes_with_all_frames_dropped(tmp_path):
    """If stage3's extreme-value bounds drop every frame of an episode, that
    episode must never reach write_lerobot_episodes (which raises a
    lerobot-level ValueError on any zero-frame episode) -- it must be
    filtered out of final_episodes and recorded in the log instead.
    """
    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/allbad", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    # Loosen stage1 the same way as the end-to-end test above, so episodes
    # actually reach stage3 instead of being rejected outright by the
    # sudden-change gate. quantile_low == quantile_high == 0.5 then
    # collapses every per-dim bound to a single point, so essentially
    # every frame of every episode (aside from the rare exact-median
    # value) is flagged out-of-bounds by stage3 -- deliberately engineered
    # to reproduce the all-frames-dropped case without touching
    # run_dataset's own logic.
    save_process_config(
        ProcessConfig(
            id="allbad_test",
            episode_reject_threshold=0.9,
            residual_threshold=5.0,
            accel_threshold=5.0,
            jerk_threshold=5.0,
            da_threshold=0.0,
            max_lag_frames=10,
            quantile_low=0.5,
            quantile_high=0.5,
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    stats = run_dataset(staging_path, output_path, config_path)

    assert stats["output_episodes"] == 0
    assert not output_path.exists()
    assert any(reason == "all_frames_dropped" for _stage, _idx, reason, _rejected in stats["log"])


def test_run_dataset_replaces_state_with_canonical_128dim_for_robot_embodiment(tmp_path):
    """Task 15's unify_representation.apply computes a canonical 128-dim
    projection of episode.state into result.stats, but does not itself
    mutate episode.state -- run_dataset must swap it in for robot-collected
    embodiment classes (result.skip_reason is None) before writing the
    output dataset, and write the dataset-constant canonical_mask as an
    extra per-frame feature."""
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset
    from common.io import save_process_config, load_process_config
    from common.schema import ProcessConfig
    from episode import Episode
    import unify_representation

    staging_path = tmp_path / "staging"
    # state_dim=14 matches unify_representation's single-arm layout: 6 joint
    # | 3 eef_pos + 4 eef_quat | 1 gripper.
    make_synthetic_dataset(staging_path, repo_id="test/canon", num_episodes=2, num_frames=10, state_dim=14, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="canon_test",
            episode_reject_threshold=0.9,
            residual_threshold=5.0,
            accel_threshold=5.0,
            jerk_threshold=5.0,
            da_threshold=0.0,
            max_lag_frames=10,
            quantile_low=0.0,
            quantile_high=1.0,
            embodiment_class="single_arm",
            num_arms=1,
            dof_per_arm=6,
            gripper_type="parallel_jaw",
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    stats = run_dataset(staging_path, output_path, config_path)

    assert stats["output_episodes"] == 2

    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    row0 = reloaded[0]
    assert tuple(row0["observation.state"].shape) == (128,)
    assert "observation.state_canonical_mask" in row0
    written_mask = row0["observation.state_canonical_mask"].numpy().astype(bool)

    # Independently recompute the expected mask via unify_representation for
    # the same config, rather than hard-coding the packing layout here.
    config = load_process_config(config_path)
    probe_episode = Episode(episode_index=0, timestamps=np.zeros(1), state=np.zeros((1, 14)), action=np.zeros((1, 4)))
    expected_mask = unify_representation.apply(probe_episode, config).stats["canonical_mask"]
    assert np.array_equal(written_mask, expected_mask)


def test_run_dataset_leaves_state_unchanged_for_non_robot_embodiment(tmp_path):
    """Non-robot-collected embodiment classes (e.g. human_hand) must be left
    at their original per-dataset dimensionality -- unify_representation
    skips them (skip_reason='embodiment_not_robot_collected'), so
    run_dataset must not replace episode.state, and no mask feature should
    be written."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/nonrobot", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="nonrobot_test",
            episode_reject_threshold=0.9,
            residual_threshold=5.0,
            accel_threshold=5.0,
            jerk_threshold=5.0,
            da_threshold=0.0,
            max_lag_frames=10,
            quantile_low=0.0,
            quantile_high=1.0,
            embodiment_class="human_hand",
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    stats = run_dataset(staging_path, output_path, config_path)

    assert stats["output_episodes"] == 2

    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    row0 = reloaded[0]
    assert tuple(row0["observation.state"].shape) == (4,)
    assert "observation.state_canonical_mask" not in row0


def test_run_dataset_replaces_action_with_canonical_54dim_when_configured(tmp_path):
    """apply_action's canonical action (when not skipped) must replace
    episode.action before writing, and its mask must be written as an
    independent action_canonical_mask feature -- mirroring how
    test_run_dataset_replaces_state_with_canonical_128dim_for_robot_embodiment
    already covers the state side."""
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    # action_dim=8 matches single-arm eef_pose layout: 3 eef_pos + 4 eef_quat + 1 gripper.
    make_synthetic_dataset(staging_path, repo_id="test/canon_action", num_episodes=2, num_frames=10, state_dim=14, action_dim=8, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="canon_action_test",
            episode_reject_threshold=0.9,
            residual_threshold=5.0,
            accel_threshold=5.0,
            jerk_threshold=5.0,
            da_threshold=0.0,
            max_lag_frames=10,
            quantile_low=0.0,
            quantile_high=1.0,
            embodiment_class="single_arm",
            num_arms=1,
            dof_per_arm=6,
            gripper_type="parallel_jaw",
            action_space="eef_pose",
            action_frame="delta",
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    stats = run_dataset(staging_path, output_path, config_path)

    assert stats["output_episodes"] == 2

    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    row0 = reloaded[0]
    assert tuple(row0["action"].shape) == (54,)
    assert "action_canonical_mask" in row0
    mask = row0["action_canonical_mask"].numpy().astype(bool)
    assert np.all(mask[0:7])
    assert not np.any(mask[27:54])


def test_main_does_not_crash_and_marks_failed_when_zero_episodes_survive(tmp_path):
    """Reproduces the original crash: main() used to unconditionally write
    `output_path / "README.md"` even though run_dataset() never created
    output_path when every episode is rejected/emptied (write_lerobot_episodes
    early-returns on an empty list) -- raising FileNotFoundError.
    """
    from tests.fixtures import make_synthetic_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig
    import run_pipeline

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/zero_survivors", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    # Same trick as test_run_dataset_filters_episodes_with_all_frames_dropped:
    # quantile_low == quantile_high collapses stage3's bounds to a single
    # point, dropping essentially every frame of every episode.
    save_process_config(
        ProcessConfig(
            id="zero_survivors",
            episode_reject_threshold=0.9,
            residual_threshold=5.0,
            accel_threshold=5.0,
            jerk_threshold=5.0,
            da_threshold=0.0,
            max_lag_frames=10,
            quantile_low=0.5,
            quantile_high=0.5,
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    exit_code = run_pipeline.main(
        ["--input", str(staging_path), "--output", str(output_path), "--config", str(config_path)]
    )  # must not raise

    assert exit_code == 1
    assert not output_path.exists()


def test_main_succeeds_and_writes_output(tmp_path):
    from tests.fixtures import make_synthetic_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig
    import run_pipeline

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/main_success", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="main_success",
            episode_reject_threshold=0.9,
            residual_threshold=5.0,
            accel_threshold=5.0,
            jerk_threshold=5.0,
            da_threshold=0.0,
            max_lag_frames=10,
            quantile_low=0.0,
            quantile_high=1.0,
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    exit_code = run_pipeline.main(
        ["--input", str(staging_path), "--output", str(output_path), "--config", str(config_path)]
    )

    assert exit_code == 0
    assert output_path.exists()


def test_main_returns_error_when_config_missing(tmp_path):
    import run_pipeline

    staging_path = tmp_path / "staging"
    output_path = tmp_path / "output"
    config_path = tmp_path / "missing_config.yaml"

    exit_code = run_pipeline.main(
        ["--input", str(staging_path), "--output", str(output_path), "--config", str(config_path)]
    )

    assert exit_code == 1
