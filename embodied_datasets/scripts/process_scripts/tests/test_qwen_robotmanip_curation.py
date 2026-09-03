import importlib.util
import json
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

MODULE = pathlib.Path(__file__).parents[1] / "qwen_robotmanip_curation" / "curate.py"
sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("curate", MODULE)
curate = importlib.util.module_from_spec(SPEC)
sys.modules["curate"] = curate
SPEC.loader.exec_module(curate)


def make_manual_dataset(tmp_path, episodes, state_dim=2, action_dim=2):
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    rows = []
    global_index = 0
    for episode_index, (states, actions) in enumerate(episodes):
        for frame_index, (state, action) in enumerate(zip(states, actions)):
            rows.append({
                "episode_index": episode_index, "frame_index": frame_index, "index": global_index,
                "observation.state": list(state), "action": list(action),
            })
            global_index += 1
    info = {
        "codebase_version": "v3.0", "fps": 10,
        "total_episodes": len(episodes), "total_frames": len(rows),
        "features": {
            "observation.state": {"dtype": "float64", "shape": [state_dim],
                                  "names": [f"joint_{i}" for i in range(state_dim)]},
            "action": {"dtype": "float64", "shape": [action_dim],
                       "names": [f"joint_{i}" for i in range(action_dim)]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    schema = pa.schema([
        ("episode_index", pa.int64()), ("frame_index", pa.int64()), ("index", pa.int64()),
        ("observation.state", pa.list_(pa.float64(), state_dim)),
        ("action", pa.list_(pa.float64(), action_dim)),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def config_for(**overrides):
    config = curate.load_config(None)
    config["defaults"].update({
        "median_kernels": [], "savgol_window": 1, "savgol_polyorder": 0,
        "allow_positional_mapping": True, "stage2_min_active_steps": 1, **overrides,
    })
    return config


def force_frame_one_anomaly(monkeypatch):
    def metrics(values, cfg, key):
        shape = np.asarray(values).shape
        residual = np.zeros(shape)
        acceleration = np.zeros(shape)
        jerk = np.zeros(shape)
        residual[1, 0] = acceleration[1, 0] = jerk[1, 0] = 10.0
        return residual, acceleration, jerk

    monkeypatch.setattr(curate, "finite_metrics", metrics)
    monkeypatch.setattr(
        curate, "robust_threshold",
        lambda values, mad_scale, quantile_floor: np.ones(values.shape[1]),
    )


def test_sudden_change_requires_residual_and_derivative():
    x = np.linspace(0, 1, 101)[:, None]
    x[50, 0] += 10
    residual, acceleration, jerk = curate.finite_metrics(x, dict(curate.DEFAULTS), "observation.state")
    assert residual[50, 0] > 1
    assert acceleration[50, 0] > 1 or jerk[50, 0] > 1


def test_stage1_frame_exclusion_marks_only_anomaly_and_writes_no_repairs(tmp_path, monkeypatch):
    source = make_manual_dataset(tmp_path, [(
        [[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [2, 0]],
    )])
    ds = curate.load_dataset(source, config_for(stage1_exclusion="frame"), "demo")
    force_frame_one_anomaly(monkeypatch)
    out = tmp_path / "out"
    result = curate.run_stage1(ds, out, None)
    validity = pq.read_table(out / "labels" / "step_validity.parquet").to_pylist()
    episode_filter = pq.read_table(out / "labels" / "episode_filter.parquet").to_pylist()
    assert [row["valid"] for row in validity] == [True, False, True]
    assert episode_filter[0]["accepted"] is True
    assert result["flagged_frames"] == 1 and result["rejected_episodes"] == 0
    assert not (out / "repairs").exists()


def test_stage1_episode_exclusion_rejects_all_frames(tmp_path, monkeypatch):
    source = make_manual_dataset(tmp_path, [(
        [[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [2, 0]],
    )])
    ds = curate.load_dataset(source, config_for(stage1_exclusion="episode"), "demo")
    force_frame_one_anomaly(monkeypatch)
    out = tmp_path / "out"
    curate.run_stage1(ds, out, None)
    validity = pq.read_table(out / "labels" / "step_validity.parquet").to_pylist()
    episode_filter = pq.read_table(out / "labels" / "episode_filter.parquet").to_pylist()
    assert all(not row["valid"] for row in validity)
    assert episode_filter[0] == {
        "episode_index": 0, "num_frames": 3, "accepted": False,
        "reason_code": "stage1_sudden_change",
    }


def test_stage2_integrates_delta_before_mask_and_keeps_valid_runs_separate():
    cfg = config_for(action_mode="delta")["defaults"]
    state = np.arange(7, dtype=float)[:, None]
    delta = np.ones((7, 1), dtype=float)
    valid = np.array([True, True, True, False, True, True, True])
    segments = curate.stage2_signal_segments(
        state, delta, valid, cfg, "observation.state", "action"
    )
    assert len(segments) == 2
    assert segments[0][1][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert segments[1][1][:, 0].tolist() == [5.0, 6.0, 7.0]


def test_stage2_low_da_rejects_episode(tmp_path, monkeypatch):
    values = np.arange(8, dtype=float)
    source = make_manual_dataset(tmp_path, [(np.c_[values, values], np.c_[values, values])])
    ds = curate.load_dataset(source, config_for(stage2_da_threshold=0.65), "demo")
    monkeypatch.setattr(curate, "trend_metric_segments", lambda *args, **kwargs: {
        "lag": 0, "correlation": 0.1, "unconstrained_lag": 0,
        "active_steps": 7, "directional_agreement": 0.2,
    })
    out = tmp_path / "stage2"
    curate.run_stage2(ds, out, None)
    episode_filter = pq.read_table(out / "labels" / "episode_filter.parquet").to_pylist()
    assert episode_filter[0]["accepted"] is False
    assert episode_filter[0]["reason_code"] == "state_action_trend_mismatch"


def test_stage3_calibration_excludes_invalid_frames_and_rejected_episodes(tmp_path):
    source = make_manual_dataset(tmp_path, [
        ([[1, 1], [999, 999], [3, 3]], [[1, 1], [999, 999], [3, 3]]),
        ([[1000, 1000]] * 3, [[1000, 1000]] * 3),
    ])
    base = curate.load_dataset(source, config_for(), "demo")
    ds = curate.Dataset(base.path, base.dataset_id, base.info, base.cfg,
                        frozenset({0}), frozenset({(0, 1)}))
    thresholds = curate.calibrate_stage3([ds], None)
    q01, q99, _, _ = thresholds[ds.state_key]
    assert np.allclose(q01, [1.02, 1.02])
    assert np.allclose(q99, [2.98, 2.98])


def test_stage3_gripper_range_is_exempt_but_nonfinite_is_invalid(tmp_path):
    source = make_manual_dataset(tmp_path, [(
        [[1, 999], [2, np.nan], [3, 500]], [[1, 999], [2, 500], [3, 500]],
    )])
    ds = curate.load_dataset(
        source, config_for(gripper_indices={"observation.state": [1], "action": [1]}), "demo"
    )
    thresholds = {
        ds.state_key: (np.zeros(2), np.ones(2), np.zeros(2), np.full(2, 10.0)),
        ds.action_key: (np.zeros(2), np.ones(2), np.zeros(2), np.full(2, 10.0)),
    }
    out = tmp_path / "stage3"
    curate.run_stage3_dataset(ds, thresholds, out, None)
    flags = pq.read_table(out / "labels" / "frame_flags.parquet").to_pylist()
    validity = pq.read_table(out / "labels" / "step_validity.parquet").to_pylist()
    assert [row["frame_index"] for row in flags] == [1]
    assert [row["valid"] for row in validity] == [True, False, True]
    assert pq.read_table(out / "labels" / "episode_filter.parquet").to_pylist()[0]["accepted"] is True


def test_filter_manifest_chain_and_separate_stage_execution(tmp_path):
    values = np.arange(20, dtype=float)
    source_path = make_manual_dataset(tmp_path, [
        (np.c_[values, values], np.c_[values, values]),
        (np.c_[values + 1, values + 1], np.c_[values + 1, values + 1]),
    ])
    config = config_for(stage2_da_threshold=-1.0)
    source = curate.load_dataset(source_path, config, "synthetic")
    output = tmp_path / "outputs"
    work = tmp_path / "work"
    for stage in (1, 2, 3):
        curate.execute([source], [stage], output, work, None, stage != 1)

    previous_manifest = None
    for stage in (1, 2, 3):
        root = output / f"stage{stage}" / "synthetic"
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["format"] == "vla_curation_filter"
        assert manifest["schema_version"] == 2
        assert manifest["video_policy"] == "reference_original"
        assert pathlib.Path(manifest["video_source"]) == (source_path / "videos").resolve()
        assert manifest["parent_manifest"] == previous_manifest
        assert manifest["repair_files"] == []
        assert manifest["output_format"] == "lerobot_v3.0-filter-manifest"
        assert (root / "labels" / "episode_filter.parquet").is_file()
        assert not any((root / name).exists() for name in ("data", "videos", "repairs", "meta"))
        if stage in (1, 3):
            validity_path = root / "labels" / "step_validity.parquet"
            assert manifest["validity_files"] == ["labels/step_validity.parquet"]
            assert pq.read_table(validity_path).num_rows == 40
        else:
            assert manifest["validity_files"] == []
        previous_manifest = str((root / "manifest.json").resolve())


def test_legacy_repair_overlay_is_not_silently_mixed(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": "vla_curation_overlay", "repair_files": ["repairs/state_action.parquet"]
    }))
    with pytest.raises(ValueError, match="rerun the preceding stages"):
        curate.dataset_from_manifest(manifest, curate.load_config(None))


def test_causal_masks_are_chunk_local():
    step_valid = np.array([True, True, False, True])
    assert curate.causal_chunk_validity(step_valid).tolist() == [True, True, False, False]
    mask = curate.causal_loss_mask(step_valid, np.array([True, False, True]))
    assert mask.tolist() == [
        [True, False, True], [True, False, True],
        [False, False, False], [False, False, False],
    ]


def test_default_output_and_config_use_filter_semantics():
    assert curate.DEFAULT_OUTPUT_ROOT == pathlib.Path(
        "/mnt/data/embodied_datasets/public_datasets_staging/data_curation"
    )
    config = json.loads((MODULE.parent / "config.example.json").read_text())
    assert config["defaults"]["stage1_exclusion"] == "frame"
    assert "stage1_policy" not in config["defaults"]
