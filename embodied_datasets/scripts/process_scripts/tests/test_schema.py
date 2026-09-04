import pytest

lerobot = pytest.importorskip("lerobot")

from pydantic import ValidationError

from common.schema import ProcessConfig


def test_process_config_requires_id():
    config = ProcessConfig(id="droid")
    assert config.fps is None
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
    assert config.sam3_model_id is None
    assert config.sam3_text_prompt == "robot gripper"
    assert config.sam3_hf_token_env is None
    assert config.gripper_radius_m is None
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
    assert config.action_space is None
    assert config.action_frame is None


def test_process_config_accepts_fps():
    config = ProcessConfig(id="droid", fps=15.0)
    assert config.fps == 15.0


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


def test_process_config_accepts_sam3_fields():
    config = ProcessConfig(
        id="x",
        sam3_model_id="facebook/sam3",
        sam3_text_prompt="robot hand",
        sam3_hf_token_env="MY_HF_TOKEN",
        gripper_radius_m=0.04,
    )
    assert config.sam3_model_id == "facebook/sam3"
    assert config.sam3_text_prompt == "robot hand"
    assert config.sam3_hf_token_env == "MY_HF_TOKEN"
    assert config.gripper_radius_m == 0.04


def test_process_config_sam3_text_prompt_defaults_to_robot_gripper():
    config = ProcessConfig(id="x")
    assert config.sam3_text_prompt == "robot gripper"
    assert config.sam3_model_id is None
    assert config.sam3_hf_token_env is None


def test_process_config_accepts_action_fields():
    config = ProcessConfig(id="x", action_space="eef_pose", action_frame="delta")
    assert config.action_space == "eef_pose"
    assert config.action_frame == "delta"
