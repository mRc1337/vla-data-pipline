from pathlib import Path

from common.paths import (
    DEFAULT_DATA_ROOT,
    REPO_ROOT,
    final_dir,
    raw_dir,
    resolve_data_root,
    staging_dir,
    urdf_assets_dir,
)


def test_default_data_root_is_repo_embodied_datasets_data():
    assert DEFAULT_DATA_ROOT == REPO_ROOT / "embodied_datasets" / "data_root"


def test_resolve_data_root_defaults_when_none():
    assert resolve_data_root(None) == DEFAULT_DATA_ROOT


def test_resolve_data_root_defaults_when_empty_string():
    assert resolve_data_root("") == DEFAULT_DATA_ROOT


def test_resolve_data_root_uses_explicit_value(tmp_path):
    custom = tmp_path / "my_data"
    assert resolve_data_root(str(custom)) == custom.resolve()


def test_raw_dir():
    data_root = Path("/mnt/big_disk")
    assert raw_dir(data_root, "droid") == Path("/mnt/big_disk/public_datasets_raw/droid")


def test_staging_dir():
    data_root = Path("/mnt/big_disk")
    assert staging_dir(data_root, "droid") == Path("/mnt/big_disk/public_datasets_staging/lerobot_v3_0/droid")


def test_final_dir():
    data_root = Path("/mnt/big_disk")
    assert final_dir(data_root, "droid") == Path("/mnt/big_disk/public_datasets/lerobot_v3_0/droid")


def test_urdf_assets_dir():
    data_root = Path("/mnt/big_disk")
    assert urdf_assets_dir(data_root, "franka_panda") == Path(
        "/mnt/big_disk/urdf_assets/franka_panda"
    )
