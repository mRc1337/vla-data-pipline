"""Tests for run_pipeline.py: the sibling-package registry loader,
run_dataset orchestration, and generate_dataset_readme. See
docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
section 10.
"""
import pytest

lerobot = pytest.importorskip("lerobot")

from pathlib import Path

from run_pipeline import _load_registry_common, compute_final_local_path


def test_compute_final_local_path_is_relative_to_final_dir(tmp_path):
    """Design doc (docs/superpowers/specs/2026-07-21-convert-scripts-verify-scripts-design.md
    section 9) documents `final_local_path` as relative to `data_root/final/`,
    matching how `raw_local_path` is relative to `data_root/raw/` -- so the
    stored value must be just `<dataset_id>`, not `final/<dataset_id>`.
    """
    data_root = tmp_path
    output_path = data_root / "final" / "droid"

    result = compute_final_local_path(output_path, data_root)

    assert result == "droid"


def test_load_registry_common_exposes_expected_symbols():
    registry_common = _load_registry_common()
    assert hasattr(registry_common.schema, "RegistryEntry")
    assert hasattr(registry_common.schema, "DatasetConfig")
    assert hasattr(registry_common.io, "load_registry")
    assert hasattr(registry_common.io, "save_registry")
    assert hasattr(registry_common.paths, "staging_dir")
    assert hasattr(registry_common.paths, "final_dir")


def test_load_registry_common_does_not_shadow_process_scripts_common():
    from common.schema import ProcessConfig

    _load_registry_common()
    from common.schema import ProcessConfig as ProcessConfigAfter

    assert ProcessConfig is ProcessConfigAfter


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
    stats = run_dataset("e2e_test", staging_path, output_path, config_path)

    assert stats["input_episodes"] == 3
    assert stats["output_episodes"] == 3
    assert stats["output_frames"] > 0
    assert len(stats["log"]) > 0
    assert output_path.exists()


def test_run_dataset_writes_real_fps_from_dataset_config_not_hardcoded_one(tmp_path):
    """write_lerobot_episodes derives every frame's timestamp from
    frame_index / fps, so a hardcoded fps=1.0 (the pre-fix behavior)
    corrupts the written dataset's temporal metadata. run_dataset must
    thread dataset_config.fps through instead.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset, _load_registry_common
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/fps", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="fps_test",
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

    registry_common = _load_registry_common()
    dataset_config = registry_common.schema.DatasetConfig(id="fps_test", name="FPS Test", fps=25.0)

    output_path = tmp_path / "output"
    stats = run_dataset("fps_test", staging_path, output_path, config_path, dataset_config=dataset_config)

    assert stats["fps"] == 25.0
    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    assert reloaded.meta.info["fps"] == 25.0
    # frame_index / fps: frame 1's timestamp should be 1/25 = 0.04s, not the
    # pre-fix hardcoded 1/1.0 = 1.0s.
    assert reloaded[1]["timestamp"].item() == pytest.approx(1 / 25.0)


def test_run_dataset_falls_back_to_default_fps_when_dataset_config_fps_unset(tmp_path):
    """`DatasetConfig.fps` is Optional -- some registry entries may not have
    it populated. run_dataset must fall back to a sensible default rather
    than crashing (None/0 would break frame_index / fps in lerobot)."""
    from tests.fixtures import make_synthetic_dataset
    from run_pipeline import run_dataset, _load_registry_common
    from common.io import save_process_config
    from common.schema import ProcessConfig

    staging_path = tmp_path / "staging"
    make_synthetic_dataset(staging_path, repo_id="test/nofps", num_episodes=1, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    config_path = tmp_path / "config.yaml"
    save_process_config(
        ProcessConfig(
            id="nofps_test",
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

    registry_common = _load_registry_common()
    dataset_config = registry_common.schema.DatasetConfig(id="nofps_test", name="No FPS Test")
    assert dataset_config.fps is None

    output_path = tmp_path / "output"
    stats = run_dataset("nofps_test", staging_path, output_path, config_path, dataset_config=dataset_config)  # must not raise

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
    stats = run_dataset("allbad_test", staging_path, output_path, config_path)

    assert stats["output_episodes"] == 0
    assert not output_path.exists()
    assert any(reason == "all_frames_dropped" for _stage, _idx, reason, _rejected in stats["log"])


def test_run_dataset_replaces_state_with_canonical_80dim_for_robot_embodiment(tmp_path):
    """Task 15's unify_representation.apply computes a canonical 80-dim
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
    from shared.episode import Episode
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
    stats = run_dataset("canon_test", staging_path, output_path, config_path)

    assert stats["output_episodes"] == 2

    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    row0 = reloaded[0]
    assert tuple(row0["observation.state"].shape) == (80,)
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
    stats = run_dataset("nonrobot_test", staging_path, output_path, config_path)

    assert stats["output_episodes"] == 2

    reloaded = LeRobotDataset(repo_id=output_path.name, root=output_path)
    row0 = reloaded[0]
    assert tuple(row0["observation.state"].shape) == (4,)
    assert "observation.state_canonical_mask" not in row0


def test_generate_dataset_readme_includes_key_sections():
    from types import SimpleNamespace

    from run_pipeline import generate_dataset_readme
    from common.schema import ProcessConfig

    dataset_config = SimpleNamespace(
        name="DROID",
        source_url="https://droid-dataset.github.io/",
        license="CC-BY-4.0",
        robot_platform="franka_panda",
        state_dim=15,
        action_dim=7,
    )
    config = ProcessConfig(id="droid", embodiment_class="single_arm", num_arms=1, gripper_type="parallel_jaw")
    stats = {
        "input_episodes": 10,
        "output_episodes": 8,
        "output_frames": 4000,
        "log": [("stage1_sudden_change", 0, None, False)],
    }

    readme = generate_dataset_readme(dataset_config, config, stats)
    assert "DROID" in readme
    assert "清洗前 episode 数: 10" in readme
    assert "清洗后 episode 数: 8" in readme
    assert "已知局限" in readme


class _TempDatasetConfigs:
    """main() resolves `process_scripts/configs/<id>.yaml`,
    `convert_scripts/configs/<id>.yaml`, and `datasets_registry.yaml` all
    relative to run_pipeline.py's own location, never relative to
    `--data-root` -- the registry and onboarding configs always live in
    the repo, even when `--data-root` points somewhere else entirely for
    the heavy staging/final data. So exercising main() end-to-end means
    appending a throwaway uuid'd RegistryEntry to the real registry and
    writing (then cleaning up all three) real files under the repo's
    config directories and registry file, scoped to that same uuid'd
    dataset id so nothing collides with or corrupts real onboarded
    datasets.
    """

    def __init__(self, registry_entry):
        import run_pipeline

        self.dataset_id = registry_entry.id
        self.registry_common = run_pipeline._load_registry_common()
        self.registry_path = Path(run_pipeline.__file__).resolve().parents[1] / "datasets_registry.yaml"
        self.process_config_path = Path(run_pipeline.__file__).resolve().parent / "configs" / f"{self.dataset_id}.yaml"
        self.dataset_config_path = (
            Path(run_pipeline.__file__).resolve().parents[1] / "convert_scripts" / "configs" / f"{self.dataset_id}.yaml"
        )
        entries = self.registry_common.io.load_registry(self.registry_path)
        entries.append(registry_entry)
        self.registry_common.io.save_registry(entries, self.registry_path)

    def reload_entry(self):
        entries = self.registry_common.io.load_registry(self.registry_path)
        return next(e for e in entries if e.id == self.dataset_id)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.process_config_path.unlink(missing_ok=True)
        self.dataset_config_path.unlink(missing_ok=True)
        entries = self.registry_common.io.load_registry(self.registry_path)
        entries = [e for e in entries if e.id != self.dataset_id]
        self.registry_common.io.save_registry(entries, self.registry_path)


def test_main_does_not_crash_and_marks_failed_when_zero_episodes_survive(tmp_path):
    """Reproduces the original crash: main() used to unconditionally write
    `output_path / "README.md"` even though run_dataset() never created
    output_path when every episode is rejected/emptied (write_lerobot_episodes
    early-returns on an empty list) -- raising FileNotFoundError. It must
    also not mark the registry PROCESSED (with num_episodes=0) for a run
    that persisted nothing to disk; FAILED (convert_scripts/common/schema.py's
    ProcessStatus) is the accurate status here.
    """
    import uuid

    from tests.fixtures import make_synthetic_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig
    import run_pipeline

    dataset_id = f"zero_survivors_{uuid.uuid4().hex[:8]}"
    data_root = tmp_path / "data_root"
    staging_path = data_root / "staging" / dataset_id
    make_synthetic_dataset(staging_path, repo_id=f"test/{dataset_id}", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    registry_common = run_pipeline._load_registry_common()
    initial_entry = registry_common.schema.RegistryEntry(
        id=dataset_id, name="Zero Survivors Test", convert_status=registry_common.schema.ConvertStatus.CONVERTED
    )

    with _TempDatasetConfigs(initial_entry) as cfg:
        # Same trick as test_run_dataset_filters_episodes_with_all_frames_dropped:
        # quantile_low == quantile_high collapses stage3's bounds to a single
        # point, dropping essentially every frame of every episode.
        save_process_config(
            ProcessConfig(
                id=dataset_id,
                episode_reject_threshold=0.9,
                residual_threshold=5.0,
                accel_threshold=5.0,
                jerk_threshold=5.0,
                da_threshold=0.0,
                max_lag_frames=10,
                quantile_low=0.5,
                quantile_high=0.5,
            ),
            cfg.process_config_path,
        )
        registry_common.io.save_dataset_config(
            registry_common.schema.DatasetConfig(id=dataset_id, name="Zero Survivors Test", fps=10.0),
            cfg.dataset_config_path,
        )

        output_path = data_root / "final" / dataset_id
        exit_code = run_pipeline.main(["--dataset-id", dataset_id, "--data-root", str(data_root)])  # must not raise

        assert exit_code == 1
        assert not output_path.exists()

        # Read back while still inside the `with` block -- __exit__ removes
        # this throwaway entry from the real registry, so it must be
        # inspected before that happens.
        reloaded_entry = cfg.reload_entry()
        assert reloaded_entry.process_status == registry_common.schema.ProcessStatus.FAILED
        assert reloaded_entry.num_episodes == 0
        assert reloaded_entry.num_frames == 0
        assert reloaded_entry.duration_hours == 0.0
        # Nothing was ever written to disk under this run, so storage_size_gb/
        # final_local_path must not be fabricated for a path that doesn't
        # exist -- they stay at the registry's pre-run default.
        assert reloaded_entry.storage_size_gb is None
        assert reloaded_entry.final_local_path is None


def test_main_updates_duration_hours_and_storage_size_gb_on_success(tmp_path):
    """Design doc section 10 step 5 lists duration_hours/storage_size_gb
    alongside process_status/num_episodes/num_frames/final_local_path
    as registry fields main() must write back after a successful run."""
    import uuid

    from tests.fixtures import make_synthetic_dataset
    from common.io import save_process_config
    from common.schema import ProcessConfig
    import run_pipeline

    dataset_id = f"main_success_{uuid.uuid4().hex[:8]}"
    data_root = tmp_path / "data_root"
    staging_path = data_root / "staging" / dataset_id
    make_synthetic_dataset(staging_path, repo_id=f"test/{dataset_id}", num_episodes=2, num_frames=10, state_dim=4, action_dim=4, fps=10.0)

    registry_common = run_pipeline._load_registry_common()
    initial_entry = registry_common.schema.RegistryEntry(
        id=dataset_id, name="Main Success Test", convert_status=registry_common.schema.ConvertStatus.CONVERTED
    )

    with _TempDatasetConfigs(initial_entry) as cfg:
        save_process_config(
            ProcessConfig(
                id=dataset_id,
                episode_reject_threshold=0.9,
                residual_threshold=5.0,
                accel_threshold=5.0,
                jerk_threshold=5.0,
                da_threshold=0.0,
                max_lag_frames=10,
                quantile_low=0.0,
                quantile_high=1.0,
            ),
            cfg.process_config_path,
        )
        registry_common.io.save_dataset_config(
            registry_common.schema.DatasetConfig(id=dataset_id, name="Main Success Test", fps=10.0),
            cfg.dataset_config_path,
        )

        output_path = data_root / "final" / dataset_id
        exit_code = run_pipeline.main(["--dataset-id", dataset_id, "--data-root", str(data_root)])

        assert exit_code == 0
        assert (output_path / "README.md").exists()
        # Independently recompute size from the actual written directory
        # (rather than hardcoding a byte count) to check the registry value
        # against reality, not against a copy of the same formula.
        expected_size_bytes = run_pipeline._dir_size_bytes(output_path)

        # Read back while still inside the `with` block -- __exit__ removes
        # this throwaway entry from the real registry, so it must be
        # inspected before that happens.
        reloaded_entry = cfg.reload_entry()
        assert reloaded_entry.process_status == registry_common.schema.ProcessStatus.PROCESSED
        assert reloaded_entry.num_episodes == 2
        assert reloaded_entry.num_frames > 0
        assert reloaded_entry.final_local_path == dataset_id

        expected_duration_hours = reloaded_entry.num_frames / 10.0 / 3600.0
        assert reloaded_entry.duration_hours == pytest.approx(expected_duration_hours)

        assert reloaded_entry.storage_size_gb is not None
        # The synthetic fixture dataset is tiny (a handful of parquet rows), so
        # its rounded GB value can legitimately be 0.0 -- assert against actual
        # on-disk bytes (which must be nonzero, README.md alone guarantees that)
        # rather than the rounded GB figure.
        assert expected_size_bytes > 0
        assert reloaded_entry.storage_size_gb == round(expected_size_bytes / 1e9, 2)
