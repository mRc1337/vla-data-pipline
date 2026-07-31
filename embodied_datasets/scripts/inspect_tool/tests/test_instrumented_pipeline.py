import pytest

lerobot = pytest.importorskip("lerobot")

from instrumented_pipeline import run_dataset_instrumented
from metadata_io import load_metadata, metadata_exists


def test_run_dataset_instrumented_writes_metadata_and_matches_run_pipeline_counts(tmp_path):
    from tests.fixtures import make_synthetic_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/inspect", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="inspect_test",
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
    metadata = run_dataset_instrumented(staging_path, output_path, config_path)

    assert len(metadata["episodes"]) == 2
    assert all(ep["survived"] for ep in metadata["episodes"])
    assert all(ep["output_frame_count"] > 0 for ep in metadata["episodes"])
    # Every stage this pipeline runs must have produced exactly one record
    # per episode, in the same order run_pipeline.py calls them.
    expected_stage_order = [
        "stage1_sudden_change", "stage2_trend_alignment", "stage3_extreme_value",
        "stage4_fk_consistency", "stage5_orientation_alignment",
        "check1_instruction_consistency", "check2_video_state_consistency",
        "check3_video_quality", "unify_representation", "unify_representation_action",
    ]
    for ep in metadata["episodes"]:
        assert [s["stage"] for s in ep["stages"]] == expected_stage_order

    assert metadata_exists(output_path)
    assert load_metadata(output_path) == metadata
    assert output_path.exists()


def test_run_dataset_instrumented_records_all_frames_dropped(tmp_path):
    from tests.fixtures import make_synthetic_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/inspect_allbad", num_episodes=1, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    # quantile_low == quantile_high collapses stage3's bounds to a point,
    # dropping essentially every frame -- same trick test_run_pipeline.py's
    # own all-frames-dropped test uses.
    save_process_config(
        ProcessConfig(
            id="inspect_allbad_test",
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
    metadata = run_dataset_instrumented(staging_path, output_path, config_path)

    assert len(metadata["episodes"]) == 1
    assert metadata["episodes"][0]["survived"] is False
    assert any(s["stage"] == "run_pipeline" and s["skip_reason"] == "all_frames_dropped" for s in metadata["episodes"][0]["stages"])
