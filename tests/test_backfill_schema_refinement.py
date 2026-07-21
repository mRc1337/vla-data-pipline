from common.backfill_schema_refinement import apply_corrections
from common.schema import DatasetConfig


def test_apply_corrections_overwrites_listed_camera_views():
    config = DatasetConfig(id="fastumi", name="FastUMI", camera_views=["other"])
    updated = apply_corrections(config)
    assert [v.value for v in updated.camera_views] == ["wrist"]


def test_apply_corrections_sets_fixed_final_camera_views_for_aloha_unleashed():
    config = DatasetConfig(
        id="aloha_unleashed", name="ALOHA Unleashed", camera_views=["top"]
    )
    updated = apply_corrections(config)
    assert [v.value for v in updated.camera_views] == [
        "top",
        "left_wrist",
        "right_wrist",
        "other",
        "worms_eye",
    ]


def test_apply_corrections_sets_multiple_fields_for_one_dataset():
    config = DatasetConfig(id="mv_umi", name="MV-UMI")
    updated = apply_corrections(config)
    assert [v.value for v in updated.camera_views] == ["third_person", "wrist"]
    assert updated.gripper_type.value == "three_jaw"
    assert updated.action_frame.value == "relative_trajectory"


def test_apply_corrections_body_worn_for_dexcap():
    config = DatasetConfig(id="dexcap", name="DexCap", camera_views=["third_person"])
    updated = apply_corrections(config)
    assert [v.value for v in updated.camera_views] == ["body_worn"]


def test_apply_corrections_sets_default_release_type():
    config = DatasetConfig(id="droid", name="DROID")
    updated = apply_corrections(config)
    assert updated.release_type.value == "fixed_episode_dataset"


def test_apply_corrections_sets_framework_release_type_and_multi_embodiment():
    config = DatasetConfig(id="robogen", name="RoboGen")
    updated = apply_corrections(config)
    assert updated.release_type.value == "generation_framework"
    assert updated.is_multi_embodiment is True


def test_apply_corrections_sets_scene_platform_release_type():
    config = DatasetConfig(id="grutopia", name="GRUtopia")
    updated = apply_corrections(config)
    assert updated.release_type.value == "scene_platform"
    assert updated.license.value == "CC-BY-NC-SA-4.0"
    assert updated.collection_method.value == "scene_asset_curation"


def test_apply_corrections_sets_rl_benchmark_release_type():
    config = DatasetConfig(id="humanoidbench", name="HumanoidBench")
    updated = apply_corrections(config)
    assert updated.release_type.value == "rl_benchmark_env"
    assert updated.robot_platform.value == "unitree_h1"
    assert updated.is_multi_embodiment is True


def test_apply_corrections_recolors_arcap_collection_method():
    config = DatasetConfig(id="arcap", name="ARCap", collection_method="umi")
    updated = apply_corrections(config)
    assert updated.collection_method.value == "ar_haptic_guided_synthesis"


def test_apply_corrections_preserves_fields_not_in_table():
    config = DatasetConfig(
        id="droid", name="DROID", source_url="https://a", license="MIT"
    )
    updated = apply_corrections(config)
    assert updated.source_url == "https://a"
    assert updated.license.value == "MIT"
