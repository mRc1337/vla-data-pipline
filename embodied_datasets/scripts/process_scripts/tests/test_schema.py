import pytest

lerobot = pytest.importorskip("lerobot")

from pydantic import ValidationError

from common.schema import (
    DatasetConfig,
    HandPoseRepresentation,
    ProcessConfig,
    ReleaseType,
    SensorModality,
)


def test_process_config_requires_id():
    config = ProcessConfig(id="droid")
    assert config.savgol_window == 11
    assert config.savgol_polyorder == 3
    assert config.residual_threshold == 0.05
    assert config.accel_threshold == 0.5
    assert config.jerk_threshold == 5.0
    assert config.episode_reject_threshold == 0.3
    assert config.da_threshold == 0.65
    assert config.max_lag_frames == 5
    assert config.quantile_low == 0.01
    assert config.quantile_high == 0.99
    assert config.tcp_offset_tolerance == 0.02
    assert config.world_frame_convention == "robot_base"
    assert config.base_to_world_transform is None
    assert config.vlm_service_url is None
    assert config.sam3_service_url is None
    assert config.iou_threshold == 0.5
    assert config.camera_frame_delta_pose_enabled is False
    assert config.gripper_dims_state == []
    assert config.gripper_dims_action == []
    assert config.per_dim_thresholds == {}
    assert config.extreme_value_bounds is None
    assert config.fk_check_feasible is False
    assert config.urdf_path is None
    assert config.has_language_instruction is False
    assert config.urdf_available is False
    assert config.has_camera_calibration is False
    assert config.vlm_model_name == "qwen2.5-vl-7b-instruct"
    assert config.vlm_api_key_env is None
    assert config.embodiment_class is None
    assert config.num_arms == 1
    assert config.dof_per_arm is None
    assert config.gripper_type == "unknown"
    assert config.has_mobile_base is False


def test_process_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        ProcessConfig(id="droid", not_a_real_field=1)


def test_process_config_per_dim_thresholds_override():
    config = ProcessConfig(
        id="droid",
        per_dim_thresholds={"state": {"0": {"residual": 0.1, "accel": 1.0, "jerk": 10.0}}},
    )
    assert config.per_dim_thresholds["state"]["0"]["residual"] == 0.1


def test_process_config_rejects_negative_savgol_polyorder():
    with pytest.raises(ValidationError):
        ProcessConfig(id="x", savgol_polyorder=-1)


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


def test_process_config_accepts_vlm_fields():
    config = ProcessConfig(
        id="x",
        vlm_service_url="http://localhost:9000/v1",
        vlm_model_name="custom-model",
        vlm_api_key_env="MY_VLM_KEY",
    )
    assert config.vlm_service_url == "http://localhost:9000/v1"
    assert config.vlm_model_name == "custom-model"
    assert config.vlm_api_key_env == "MY_VLM_KEY"
