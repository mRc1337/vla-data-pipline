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


def test_run_dataset_passes_through_episodes_stage2_could_not_check(tmp_path):
    """Regression: stage2 used to be gated on `skip_reason` in run_dataset,
    so its "couldn't run the check at all" skip_reasons
    (insufficient_frames_for_trend_alignment) were treated identically to a
    real rejection and silently dropped the episode. A 1-frame episode
    (too short for stage2's cross-correlation, but otherwise perfectly
    valid) must survive to the output instead of vanishing.
    """
    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/too_short_for_stage2", num_episodes=2, num_frames=1, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    # quantile_low/high=0/1 disables stage3's extreme-value filtering, which
    # otherwise computes far-too-tight bounds from just 2 total data points
    # (the default 0.01/0.99 quantiles) and would drop the single frame this
    # test is specifically trying to keep alive -- same loosening convention
    # used by the other synthetic-data tests in this file.
    save_process_config(ProcessConfig(id="too_short_for_stage2_test", quantile_low=0.0, quantile_high=1.0), config_path)

    output_path = tmp_path / "output"
    stats = run_dataset(staging_path, output_path, config_path)

    assert stats["output_episodes"] == 2
    assert any(
        stage == "stage2_trend_alignment" and reason == "insufficient_frames_for_trend_alignment" and not rejected
        for stage, _idx, reason, rejected in stats["log"]
    )


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


def test_run_dataset_replaces_action_with_canonical_128dim_when_configured(tmp_path):
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
    assert tuple(row0["action"].shape) == (128,)
    assert "action_canonical_mask" in row0
    mask = row0["action_canonical_mask"].numpy().astype(bool)
    assert not np.any(mask[0:7])  # eef_pose: no joint data
    assert np.all(mask[7:13])
    assert not np.any(mask[34:128])  # single arm: arm2 block + reserve untouched


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


def test_config_consistency_warnings_flags_urdf_path_without_fk_check_feasible():
    from run_pipeline import config_consistency_warnings
    from common.schema import ProcessConfig

    config = ProcessConfig(id="t", urdf_path="/robot.urdf", fk_check_feasible=False)
    warnings = config_consistency_warnings(config)
    assert any("fk_check_feasible=False" in w for w in warnings)


def test_config_consistency_warnings_flags_fk_check_feasible_without_urdf_path():
    from run_pipeline import config_consistency_warnings
    from common.schema import ProcessConfig

    config = ProcessConfig(id="t", fk_check_feasible=True, urdf_path=None)
    warnings = config_consistency_warnings(config)
    assert any("urdf_path is unset" in w for w in warnings)


def test_config_consistency_warnings_flags_language_instruction_without_vlm_service():
    from run_pipeline import config_consistency_warnings
    from common.schema import ProcessConfig

    config = ProcessConfig(id="t", has_language_instruction=True, vlm_service_url=None)
    warnings = config_consistency_warnings(config)
    assert any("fail-open mode (vlm_service_not_configured)" in w for w in warnings)


def test_config_consistency_warnings_flags_camera_calibration_without_sam3_model():
    from run_pipeline import config_consistency_warnings
    from common.schema import ProcessConfig

    config = ProcessConfig(id="t", urdf_available=True, has_camera_calibration=True, sam3_model_id=None)
    warnings = config_consistency_warnings(config)
    assert any("fail-open mode (sam3_service_not_configured)" in w for w in warnings)


def test_config_consistency_warnings_empty_for_consistent_config():
    from run_pipeline import config_consistency_warnings
    from common.schema import ProcessConfig

    config = ProcessConfig(id="t")
    assert config_consistency_warnings(config) == []


def test_summarize_log_reports_rejection_and_skip_counts():
    from run_pipeline import summarize_log

    log = [
        ("stage1_sudden_change", 0, None, False),
        ("stage1_sudden_change", 1, None, True),
        ("check1_instruction_consistency", 0, "vlm_call_failed", False),
        ("check1_instruction_consistency", 1, "vlm_call_failed", False),
    ]
    lines = summarize_log(log)
    assert "stage1_sudden_change: rejected 1 episode(s)" in lines
    assert any("vlm_call_failed x2" in line and "UNVERIFIED" in line for line in lines)


def test_summarize_log_does_not_tag_real_skip_reasons_as_unverified():
    from run_pipeline import summarize_log

    log = [("stage2_trend_alignment", 0, "insufficient_frames_for_trend_alignment", False)]
    lines = summarize_log(log)
    assert any(
        "insufficient_frames_for_trend_alignment" in line and "UNVERIFIED" not in line for line in lines
    )


def test_main_prints_config_consistency_warning_to_stderr(tmp_path, capsys):
    from tests.fixtures import make_synthetic_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig
    import run_pipeline

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/warn", num_episodes=1, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="warn_test",
            episode_reject_threshold=0.9, residual_threshold=5.0, accel_threshold=5.0,
            jerk_threshold=5.0, da_threshold=0.0, max_lag_frames=10, quantile_low=0.0, quantile_high=1.0,
            fk_check_feasible=True, urdf_path=None,
        ),
        config_path,
    )

    output_path = tmp_path / "output"
    run_pipeline.main(["--input", str(staging_path), "--output", str(output_path), "--config", str(config_path)])

    captured = capsys.readouterr()
    assert "fk_check_feasible=True but urdf_path is unset" in captured.err


def test_main_prints_pipeline_diagnostics_summary(tmp_path, capsys):
    from tests.fixtures import make_synthetic_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig
    import run_pipeline

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/diag", num_episodes=2, num_frames=1, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(ProcessConfig(id="diag_test", quantile_low=0.0, quantile_high=1.0), config_path)

    output_path = tmp_path / "output"
    run_pipeline.main(["--input", str(staging_path), "--output", str(output_path), "--config", str(config_path)])

    captured = capsys.readouterr()
    assert "diagnostic: stage2_trend_alignment: insufficient_frames_for_trend_alignment x2" in captured.err
