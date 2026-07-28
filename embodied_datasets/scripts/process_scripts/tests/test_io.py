import pytest

lerobot = pytest.importorskip("lerobot")

from pathlib import Path

from common.io import (
    load_dataset_config,
    load_process_config,
    save_dataset_config,
    save_process_config,
)
from common.schema import DatasetConfig, ProcessConfig


def test_save_then_load_round_trip(tmp_path: Path):
    config = ProcessConfig(id="droid", residual_threshold=0.1, quantile_low=0.02)
    path = tmp_path / "droid.yaml"
    save_process_config(config, path)

    loaded = load_process_config(path)
    assert loaded.id == "droid"
    assert loaded.residual_threshold == 0.1
    assert loaded.quantile_low == 0.02


def test_save_creates_parent_dirs(tmp_path: Path):
    config = ProcessConfig(id="droid")
    path = tmp_path / "nested" / "droid.yaml"
    save_process_config(config, path)
    assert path.exists()


def test_round_trip_preserves_optional_scalar_field(tmp_path: Path):
    """Test that Optional scalar fields (urdf_path) round-trip correctly."""
    config = ProcessConfig(id="droid", urdf_path="/path/to/robot.urdf")
    path = tmp_path / "droid.yaml"
    save_process_config(config, path)

    loaded = load_process_config(path)
    assert loaded.urdf_path == "/path/to/robot.urdf"


def test_round_trip_preserves_optional_nested_dict_field(tmp_path: Path):
    """Test that Optional nested dict fields with int keys round-trip correctly.

    This exercises the trickiest case: extreme_value_bounds has type
    Optional[Dict[str, Dict[int, List[float]]]], which means the inner dict
    has int keys. YAML/JSON force these to strings on dump, so pydantic must
    coerce them back to int on load.
    """
    config = ProcessConfig(
        id="droid",
        extreme_value_bounds={
            "state": {0: [-1.0, 1.0], 3: [0.0, 255.0]},
            "action": {1: [-0.5, 0.5]},
        },
    )
    path = tmp_path / "droid.yaml"
    save_process_config(config, path)

    loaded = load_process_config(path)
    # Verify the nested dict with int keys round-trips correctly
    assert loaded.extreme_value_bounds is not None
    assert loaded.extreme_value_bounds == {
        "state": {0: [-1.0, 1.0], 3: [0.0, 255.0]},
        "action": {1: [-0.5, 0.5]},
    }
    # Verify inner keys are actually int, not str
    assert all(isinstance(k, int) for k in loaded.extreme_value_bounds["state"].keys())
    assert all(isinstance(k, int) for k in loaded.extreme_value_bounds["action"].keys())


def test_dataset_config_round_trip(tmp_path):
    path = tmp_path / "droid.yaml"
    config = DatasetConfig(
        id="droid", name="DROID", source_url="https://a", license="MIT"
    )
    save_dataset_config(config, path)
    loaded = load_dataset_config(path)
    assert loaded.id == "droid"
    assert loaded.source_url == "https://a"
    assert loaded.license.value == "MIT"
    assert loaded.review_status.value == "pending_human_review"
