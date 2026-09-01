import importlib.util
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


def test_sudden_change_requires_residual_and_derivative():
    cfg = dict(curate.DEFAULTS)
    x = np.linspace(0, 1, 101)[:, None]
    x[50, 0] += 10
    residual, acceleration, jerk = curate.finite_metrics(x, cfg, "observation.state")
    assert residual[50, 0] > 1
    assert acceleration[50, 0] > 1 or jerk[50, 0] > 1


def test_positive_lag_and_directional_agreement():
    rng = np.random.default_rng(4)
    action = np.cumsum(rng.normal(size=200))
    state = np.r_[action[:1].repeat(3), action[:-3]]
    metric = curate.trend_metric(state, action, max_lag=8, min_active=10)
    assert metric["lag"] == 3
    assert metric["directional_agreement"] > 0.95


def test_robust_threshold_ignores_single_spike():
    values = np.zeros((10_000, 1))
    values[-1] = 100
    threshold = curate.robust_threshold(values, 8.0, 0.999)
    assert threshold[0] < 100


def test_stage2_mapping_uses_matching_lerobot_feature_names():
    cfg = dict(curate.DEFAULTS)
    info = {
        "features": {
            "observation.state": {"shape": [3], "names": ["joint_a", "joint_b", "gripper"]},
            "action": {"shape": [3], "names": ["joint_a", "joint_b", "gripper"]},
        }
    }
    ds = curate.Dataset(pathlib.Path("/tmp/example"), "example", info, cfg)
    assert curate.state_action_map(ds, 3, 3) == [(0, 0), (1, 1)]


def test_stage2_requires_names_explicit_map_or_opt_in_positional_map():
    cfg = dict(curate.DEFAULTS)
    info = {
        "features": {
            "observation.state": {"shape": [2], "names": None},
            "action": {"shape": [2], "names": None},
        }
    }
    ds = curate.Dataset(pathlib.Path("/tmp/example"), "example", info, cfg)
    assert curate.state_action_map(ds, 2, 2) == []
    cfg["allow_positional_mapping"] = True
    assert curate.state_action_map(ds, 2, 2) == [(0, 0), (1, 1)]


def test_stage2_does_not_treat_generic_state_action_labels_as_semantic_mapping():
    cfg = dict(curate.DEFAULTS)
    info = {
        "features": {
            "observation.state": {"shape": [2], "names": ["state_0", "state_1"]},
            "action": {"shape": [2], "names": ["action_0", "action_1"]},
        }
    }
    ds = curate.Dataset(pathlib.Path("/tmp/example"), "example", info, cfg)
    assert curate.state_action_map(ds, 2, 2) == []


def test_interpolate_dimensions_repairs_only_flagged_cells():
    values = np.array([[0.0, 10.0], [99.0, 11.0], [2.0, 12.0]], dtype=np.float32)
    bad = np.zeros_like(values, dtype=bool)
    bad[1, 0] = True
    repaired = curate.interpolate_dimensions(values, bad)
    assert repaired.tolist() == [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]


def test_default_output_root_is_staging_sibling_of_lerobot_input():
    assert curate.DEFAULT_OUTPUT_ROOT == pathlib.Path(
        "/mnt/data/embodied_datasets/public_datasets_staging/data_curation"
    )
    assert curate.STAGE_NAMES == {1: "stage1", 2: "stage2", 3: "stage3"}
    args = curate.build_parser().parse_args([
        "run", "--dataset-path", "/tmp/input", "--work-root", "/tmp/work",
    ])
    assert args.output_root == curate.DEFAULT_OUTPUT_ROOT


def test_materializer_outputs_loadable_lerobot_v3_and_removes_episode(tmp_path):
    process_scripts = pathlib.Path(__file__).parents[1]
    sys.path.insert(0, str(process_scripts))
    from tests.fixtures import make_synthetic_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_path = make_synthetic_dataset(
        tmp_path / "source", "test/source", num_episodes=2, num_frames=3, state_dim=2, action_dim=2
    )
    config = curate.load_config(None)
    ds = curate.load_dataset(source_path, config, "synthetic")
    source = LeRobotDataset(repo_id="source", root=source_path)
    left = source[0]["observation.state"].numpy()[0]
    right = source[2]["observation.state"].numpy()[0]

    flags = tmp_path / "flags.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "episode_index": 0,
        "frame_index": 1,
        "failed_state_dimensions": [0],
        "failed_action_dimensions": [],
    }]), flags)
    rejected = tmp_path / "rejected.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "episode_index": 1,
        "reject_episode": True,
    }]), rejected)

    output = tmp_path / "output"
    counts = curate.materialize_lerobot_v3(ds, output, flags, rejected)
    result = LeRobotDataset(repo_id="output", root=output)
    assert result.meta.info.codebase_version == "v3.0"
    assert result.num_episodes == 1
    assert len(result) == 3
    assert counts == {
        "output_episodes": 1,
        "output_frames": 3,
        "repaired_frames": 1,
        "removed_episodes": 1,
    }
    assert result[1]["observation.state"].numpy()[0] == pytest.approx((left + right) / 2)


def test_execute_chains_three_lerobot_v3_datasets(tmp_path):
    process_scripts = pathlib.Path(__file__).parents[1]
    sys.path.insert(0, str(process_scripts))
    from tests.fixtures import make_synthetic_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_path = make_synthetic_dataset(
        tmp_path / "source", "test/pipeline", num_episodes=2, num_frames=20,
        state_dim=2, action_dim=2,
    )
    config = curate.load_config(None)
    config["defaults"]["allow_positional_mapping"] = True
    config["defaults"]["stage2_da_threshold"] = -1.0
    source = curate.load_dataset(source_path, config, "synthetic")

    results = curate.execute(
        [source], [1, 2, 3], tmp_path / "outputs", tmp_path / "work", None, False
    )

    assert len(results) == 3
    previous = source_path.resolve()
    for stage in (1, 2, 3):
        root = tmp_path / "outputs" / curate.STAGE_NAMES[stage] / "synthetic"
        dataset = LeRobotDataset(repo_id=f"stage{stage}", root=root / "dataset")
        assert dataset.meta.info.codebase_version == "v3.0"
        assert dataset.num_episodes == 2
        assert (root / "audit" / "run.json").is_file()
        assert pathlib.Path(results[stage - 1]["source"]).resolve() == previous
        previous = (root / "dataset").resolve()
