from common.io import (
    load_dataset_config,
    load_registry,
    save_dataset_config,
    save_registry,
)
from common.schema import DatasetConfig, RegistryEntry


def test_registry_round_trip(tmp_path):
    path = tmp_path / "registry.yaml"
    entries = [
        RegistryEntry(id="droid", name="DROID"),
        RegistryEntry(id="bridgedata_v2", name="BridgeData V2"),
    ]
    save_registry(entries, path)
    loaded = load_registry(path)
    assert [e.id for e in loaded] == ["droid", "bridgedata_v2"]
    assert loaded[0].download_status.value == "not_downloaded"


def test_load_registry_missing_file_returns_empty(tmp_path):
    assert load_registry(tmp_path / "missing.yaml") == []


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
