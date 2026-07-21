from common.backfill_schema_round2 import apply_round2_corrections
from common.schema import DatasetConfig


def test_apply_round2_corrections_sets_robot_platform():
    config = DatasetConfig(id="rt_1", name="RT-1", robot_platform="other")
    updated = apply_round2_corrections(config)
    assert updated.robot_platform.value == "everyday_robots_arm"


def test_apply_round2_corrections_sets_multiple_fields_for_one_dataset():
    config = DatasetConfig(id="teach", name="TEACh", robot_platform="other")
    updated = apply_round2_corrections(config)
    assert updated.robot_platform.value == "virtual_agent"
    assert updated.license.value == "CDLA-Sharing-1.0"
    assert updated.action_space.value == "discrete_symbolic"


def test_apply_round2_corrections_overwrites_additional_modalities_list():
    config = DatasetConfig(
        id="ego_exo4d", name="Ego-Exo4D", additional_modalities=["audio", "eye_gaze"]
    )
    updated = apply_round2_corrections(config)
    assert [m.value for m in updated.additional_modalities] == [
        "audio",
        "eye_gaze",
        "imu",
        "point_cloud_3d_scan",
    ]


def test_apply_round2_corrections_splits_wrist_and_gripper_jaw_cameras():
    config = DatasetConfig(
        id="partnr", name="PARTNR", camera_views=["third_person", "head", "wrist"]
    )
    updated = apply_round2_corrections(config)
    assert [v.value for v in updated.camera_views] == [
        "third_person",
        "head",
        "wrist",
        "gripper_jaw",
    ]


def test_apply_round2_corrections_fixes_vitra_gripper_type():
    config = DatasetConfig(id="vitra", name="VITRA", gripper_type="dexterous_hand")
    updated = apply_round2_corrections(config)
    assert updated.gripper_type.value == "none"


def test_apply_round2_corrections_leaves_untouched_datasets_unchanged():
    config = DatasetConfig(
        id="droid", name="DROID", source_url="https://a", license="MIT"
    )
    updated = apply_round2_corrections(config)
    assert updated.source_url == "https://a"
    assert updated.license.value == "MIT"
