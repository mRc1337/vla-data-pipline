import importlib.util
import pathlib
import sys

import numpy as np

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


def test_docx_action_arm2_padding_contract():
    layout = curate.LAYOUT["action"]
    assert layout["arm2"] == [34, 69]
    assert layout["arm2_payload"] == [34, 68]
    assert layout["arm2_padding_zero_mask_false"] == 68
    assert layout["reserve"] == [69, 128]


def test_stage2_mapping_is_restricted_to_canonical_joint_slots():
    cfg = dict(curate.DEFAULTS)
    cfg["canonical_indices"] = {
        "observation.state": [0, 7, 14],
        "action": [0, 7, 13],
    }
    info = {
        "features": {
            "observation.state": {"shape": [3], "names": ["joint", "eef", "gripper"]},
            "action": {"shape": [3], "names": ["joint", "eef", "gripper"]},
        }
    }
    ds = curate.Dataset(pathlib.Path("/tmp/example"), "example", info, cfg)
    assert curate.state_action_map(ds, 3, 3) == [(0, 0)]


def test_action_padding_cannot_be_mapped():
    cfg = dict(curate.DEFAULTS)
    cfg["canonical_indices"] = {"observation.state": [0], "action": [68]}
    info = {
        "features": {
            "observation.state": {"shape": [1], "names": ["joint"]},
            "action": {"shape": [1], "names": ["joint"]},
        }
    }
    ds = curate.Dataset(pathlib.Path("/tmp/example"), "example", info, cfg)
    import pytest
    with pytest.raises(ValueError, match="padding"):
        curate.canonical_indices(ds, "action", 1)
