import pytest
from pydantic import ValidationError

from common.schema import DatasetConfig, RegistryEntry


def test_registry_entry_defaults():
    entry = RegistryEntry(id="droid", name="DROID")
    assert entry.priority.value == "P2"
    assert entry.download_status.value == "not_downloaded"
    assert entry.integrity_status.value == "not_verified"
    assert entry.convert_status.value == "not_converted"
    assert entry.process_status.value == "not_processed"


def test_registry_entry_rejects_invalid_enum():
    with pytest.raises(ValidationError):
        RegistryEntry(id="droid", name="DROID", download_status="downloaded_maybe")


def test_registry_entry_rejects_unknown_field():
    with pytest.raises(ValidationError):
        RegistryEntry(id="droid", name="DROID", not_a_real_field=1)


def test_dataset_config_defaults():
    config = DatasetConfig(id="droid", name="DROID")
    assert config.review_status.value == "pending_human_review"
    assert config.camera_views == []
    assert config.license is None
    assert config.field_sources == {}


def test_dataset_config_rejects_unknown_robot_platform():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", robot_platform="made_up_robot")


def test_dataset_config_accepts_known_robot_platform():
    config = DatasetConfig(id="droid", name="DROID", robot_platform="franka_panda")
    assert config.robot_platform.value == "franka_panda"


def test_dataset_config_num_arms_bounds():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", num_arms=3)


def test_dataset_config_camera_views_enum_checked():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", camera_views=["bird_eye_view"])
