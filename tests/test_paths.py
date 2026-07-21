from pathlib import Path

from common.paths import (
    DEFAULT_DATA_ROOT,
    REPO_ROOT,
    lerobot_v2_1_final_dir,
    lerobot_v2_1_staging_dir,
    raw_dir,
    resolve_data_root,
    urdf_assets_dir,
)


def test_default_data_root_is_repo_embodied_datasets():
    assert DEFAULT_DATA_ROOT == REPO_ROOT / "embodied_datasets"


def test_resolve_data_root_defaults_when_none():
    assert resolve_data_root(None) == DEFAULT_DATA_ROOT


def test_resolve_data_root_defaults_when_empty_string():
    assert resolve_data_root("") == DEFAULT_DATA_ROOT


def test_resolve_data_root_uses_explicit_value(tmp_path):
    custom = tmp_path / "my_data"
    assert resolve_data_root(str(custom)) == custom.resolve()


def test_raw_dir():
    data_root = Path("/mnt/big_disk")
    assert raw_dir(data_root, "droid") == Path(
        "/mnt/big_disk/public_datasets_raw/droid/raw"
    )


def test_lerobot_v2_1_staging_dir():
    data_root = Path("/mnt/big_disk")
    assert lerobot_v2_1_staging_dir(data_root, "droid") == Path(
        "/mnt/big_disk/public_datasets_raw/droid/lerobot_v2_1_staging"
    )


def test_lerobot_v2_1_final_dir():
    data_root = Path("/mnt/big_disk")
    assert lerobot_v2_1_final_dir(data_root, "droid") == Path(
        "/mnt/big_disk/public_datasets/lerobot_v2_1/droid"
    )


def test_urdf_assets_dir():
    data_root = Path("/mnt/big_disk")
    assert urdf_assets_dir(data_root, "franka_panda") == Path(
        "/mnt/big_disk/urdf_assets/franka_panda"
    )
