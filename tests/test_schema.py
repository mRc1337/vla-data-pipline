import pytest
from pydantic import ValidationError

from common.schema import (
    DatasetConfig,
    HandPoseRepresentation,
    ReleaseType,
    RegistryEntry,
    SensorModality,
)


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


def test_camera_view_new_members():
    from common.schema import CameraView

    assert CameraView.WRIST.value == "wrist"
    assert CameraView.BODY_WORN.value == "body_worn"
    assert CameraView.WORMS_EYE.value == "worms_eye"


def test_license_new_members():
    from common.schema import LicenseEnum

    assert LicenseEnum.CC_BY_NC_SA_4_0.value == "CC-BY-NC-SA-4.0"
    assert LicenseEnum.CC_BY_NC_ND_4_0.value == "CC-BY-NC-ND-4.0"


def test_robot_platform_new_members():
    from common.schema import RobotPlatform

    assert RobotPlatform.UNITREE_H1.value == "unitree_h1"
    assert RobotPlatform.FOURIER_GR1.value == "fourier_gr1"
    assert RobotPlatform.FLEXIV_RIZON4.value == "flexiv_rizon4"
    assert RobotPlatform.GALAXEA_R1_LITE.value == "galaxea_r1_lite"
    assert RobotPlatform.TOYOTA_ELEY.value == "toyota_eley"


def test_embodiment_class_quadruped():
    from common.schema import EmbodimentClass

    assert EmbodimentClass.QUADRUPED.value == "quadruped"


def test_collection_method_new_members():
    from common.schema import CollectionMethod

    assert CollectionMethod.AR_HAPTIC_GUIDED.value == "ar_haptic_guided_synthesis"
    assert CollectionMethod.SCENE_ASSET_CURATION.value == "scene_asset_curation"


def test_raw_format_vrs():
    from common.schema import RawFormat

    assert RawFormat.VRS.value == "VRS"


def test_gripper_type_three_jaw():
    from common.schema import GripperType

    assert GripperType.THREE_JAW.value == "three_jaw"


def test_action_frame_new_members():
    from common.schema import ActionFrame

    assert ActionFrame.RELATIVE_TRAJECTORY.value == "relative_trajectory"
    assert ActionFrame.MIXED_DELTA_ABSOLUTE.value == "mixed_delta_absolute"


def test_release_type_members():
    assert ReleaseType.FIXED_EPISODE_DATASET.value == "fixed_episode_dataset"
    assert ReleaseType.GENERATION_FRAMEWORK.value == "generation_framework"
    assert ReleaseType.SCENE_PLATFORM.value == "scene_platform"
    assert ReleaseType.RL_BENCHMARK_ENV.value == "rl_benchmark_env"


def test_hand_pose_representation_members():
    assert HandPoseRepresentation.MANO.value == "mano"
    assert HandPoseRepresentation.KEYPOINTS_3D.value == "keypoints_3d"
    assert HandPoseRepresentation.JOINT_ANGLES.value == "joint_angles"
    assert HandPoseRepresentation.NONE.value == "none"


def test_sensor_modality_members():
    assert SensorModality.FORCE_TORQUE.value == "force_torque"
    assert SensorModality.TACTILE.value == "tactile"
    assert SensorModality.AUDIO.value == "audio"
    assert SensorModality.EYE_GAZE.value == "eye_gaze"


def test_dataset_config_new_fields_default():
    config = DatasetConfig(id="droid", name="DROID")
    assert config.release_type is None
    assert config.is_multi_embodiment is None
    assert config.paper_url is None
    assert config.secondary_collection_methods == []
    assert config.additional_modalities == []
    assert config.has_synchronized_multiview_rig is None
    assert config.dof_per_hand is None
    assert config.hand_pose_representation is None
    assert config.expected_duration_hours is None
    assert config.num_subjects is None
    assert config.num_scenes is None
    assert config.num_objects is None


def test_dataset_config_new_fields_accept_valid_values():
    config = DatasetConfig(
        id="droid",
        name="DROID",
        release_type="generation_framework",
        is_multi_embodiment=True,
        paper_url="https://arxiv.org/abs/1234.5678",
        secondary_collection_methods=["simulation"],
        additional_modalities=["force_torque", "tactile"],
        has_synchronized_multiview_rig=True,
        dof_per_hand=16,
        hand_pose_representation="mano",
        expected_duration_hours=41.3,
        num_subjects=19,
        num_scenes=100,
        num_objects=50,
    )
    assert config.release_type.value == "generation_framework"
    assert config.secondary_collection_methods[0].value == "simulation"
    assert [m.value for m in config.additional_modalities] == [
        "force_torque",
        "tactile",
    ]
    assert config.hand_pose_representation.value == "mano"


def test_dataset_config_rejects_invalid_release_type():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", release_type="not_a_real_type")


def test_dataset_config_rejects_invalid_modality():
    with pytest.raises(ValidationError):
        DatasetConfig(id="droid", name="DROID", additional_modalities=["smell"])


def test_robot_platform_round2_new_members():
    from common.schema import RobotPlatform

    assert RobotPlatform.FRANKA_FR3.value == "franka_fr3"
    assert RobotPlatform.XARM6.value == "xarm6"
    assert RobotPlatform.KUKA_IIWA.value == "kuka_iiwa"
    assert RobotPlatform.ABB_YUMI.value == "abb_yumi"
    assert RobotPlatform.HELLO_ROBOT_STRETCH.value == "hello_robot_stretch"
    assert RobotPlatform.BOSTON_DYNAMICS_SPOT.value == "boston_dynamics_spot"
    assert RobotPlatform.EVERYDAY_ROBOTS_ARM.value == "everyday_robots_arm"
    assert RobotPlatform.AGIBOT_G2.value == "agibot_g2"
    assert RobotPlatform.GALBOT_G1.value == "galbot_g1"
    assert RobotPlatform.UNITREE_H1_2.value == "unitree_h1_2"
    assert RobotPlatform.UNITREE_ALIENGO.value == "unitree_aliengo"
    assert RobotPlatform.UNITREE_A1.value == "unitree_a1"
    assert RobotPlatform.ANYMAL.value == "anymal"
    assert RobotPlatform.ATLAS.value == "atlas"
    assert RobotPlatform.GALAXEA_R1.value == "galaxea_r1"
    assert RobotPlatform.R1PRO.value == "r1pro"
    assert RobotPlatform.X1_EVE.value == "1x_eve"
    assert RobotPlatform.VIRTUAL_AGENT.value == "virtual_agent"


def test_collection_method_round2_new_members():
    from common.schema import CollectionMethod

    assert CollectionMethod.KINESTHETIC.value == "kinesthetic"
    assert CollectionMethod.SCRIPTED.value == "scripted"
    assert CollectionMethod.EGO_EXO_HUMAN.value == "ego_exo_human"
    assert CollectionMethod.MOCAP_MULTIVIEW_HUMAN.value == "mocap_multiview_human"
    assert (
        CollectionMethod.SYNTHETIC_MULTIMODAL_AUGMENTATION.value
        == "synthetic_multimodal_augmentation"
    )


def test_license_round2_new_members():
    from common.schema import LicenseEnum

    assert LicenseEnum.CC_BY_SA_4_0.value == "CC-BY-SA-4.0"
    assert LicenseEnum.CDLA_SHARING_1_0.value == "CDLA-Sharing-1.0"
    assert LicenseEnum.CUSTOM_RESEARCH_EULA.value == "custom_research_eula"


def test_embodiment_class_round2_new_members():
    from common.schema import EmbodimentClass

    assert EmbodimentClass.HUMAN_FULL_BODY.value == "human_full_body"
    assert EmbodimentClass.HALF_HUMANOID.value == "half_humanoid"


def test_gripper_type_round2_new_members():
    from common.schema import GripperType

    assert GripperType.MIXED.value == "mixed"
    assert GripperType.CAGE_PINCH.value == "cage_pinch"


def test_rotation_representation_round2_new_members():
    from common.schema import RotationRepresentation

    assert RotationRepresentation.MIXED.value == "mixed"
    assert RotationRepresentation.SINGLE_AXIS_ANGLE.value == "single_axis_angle"


def test_sensor_modality_round2_new_members():
    from common.schema import SensorModality

    assert SensorModality.IMU.value == "imu"
    assert SensorModality.SEMANTIC_SEGMENTATION.value == "semantic_segmentation"
    assert SensorModality.POINT_CLOUD_3D_SCAN.value == "point_cloud_3d_scan"


def test_action_space_round2_new_members():
    from common.schema import ActionSpace

    assert ActionSpace.DISCRETE_SYMBOLIC.value == "discrete_symbolic"


def test_camera_view_round2_new_members():
    from common.schema import CameraView

    assert CameraView.GRIPPER_JAW.value == "gripper_jaw"


def test_dataset_config_accepts_round2_enum_values():
    config = DatasetConfig(
        id="droid",
        name="DROID",
        robot_platform="abb_yumi",
        collection_method="ego_exo_human",
        license="custom_research_eula",
        embodiment_class="half_humanoid",
        gripper_type="mixed",
        rotation_representation="single_axis_angle",
        additional_modalities=["imu", "semantic_segmentation"],
        action_space="discrete_symbolic",
        camera_views=["gripper_jaw"],
    )
    assert config.robot_platform.value == "abb_yumi"
    assert config.collection_method.value == "ego_exo_human"
    assert config.license.value == "custom_research_eula"
    assert config.embodiment_class.value == "half_humanoid"
    assert config.gripper_type.value == "mixed"
    assert config.rotation_representation.value == "single_axis_angle"
    assert [m.value for m in config.additional_modalities] == [
        "imu",
        "semantic_segmentation",
    ]
    assert config.action_space.value == "discrete_symbolic"
    assert config.camera_views[0].value == "gripper_jaw"
