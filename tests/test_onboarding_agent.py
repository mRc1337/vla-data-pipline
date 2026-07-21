import pytest

from common.onboarding_agent import (
    build_onboarding_prompt,
    parse_and_validate_agent_output,
)


def test_build_onboarding_prompt_includes_key_info():
    prompt = build_onboarding_prompt("droid", "DROID", "https://droid-dataset.github.io/")
    assert "droid" in prompt
    assert "DROID" in prompt
    assert "https://droid-dataset.github.io/" in prompt
    assert "franka_panda" in prompt
    assert "Apache-2.0" in prompt


def test_build_onboarding_prompt_handles_missing_source():
    prompt = build_onboarding_prompt("droid", "DROID", None)
    assert "自行搜索" in prompt


def test_parse_and_validate_agent_output_valid():
    yaml_text = """
id: droid
name: DROID
license: MIT
robot_platform: franka_panda
num_arms: 1
camera_views:
  - third_person
  - right_wrist
review_status: pending_human_review
field_sources:
  license: "https://droid-dataset.github.io/ - footer"
"""
    config = parse_and_validate_agent_output(yaml_text)
    assert config.id == "droid"
    assert config.license.value == "MIT"
    assert config.robot_platform.value == "franka_panda"
    assert config.camera_views[0].value == "third_person"


def test_parse_and_validate_agent_output_rejects_unknown_enum():
    yaml_text = """
id: droid
name: DROID
robot_platform: made_up_robot
"""
    with pytest.raises(ValueError):
        parse_and_validate_agent_output(yaml_text)


def test_parse_and_validate_agent_output_rejects_non_mapping():
    with pytest.raises(ValueError):
        parse_and_validate_agent_output("- just\n- a\n- list\n")


def test_build_onboarding_prompt_includes_new_enum_fields():
    prompt = build_onboarding_prompt("droid", "DROID", "https://droid-dataset.github.io/")
    assert "generation_framework" in prompt
    assert "mano" in prompt
    assert "force_torque" in prompt
    assert "three_jaw" in prompt


def test_build_onboarding_prompt_includes_new_scalar_field_descriptions():
    prompt = build_onboarding_prompt("droid", "DROID", "https://droid-dataset.github.io/")
    assert "dof_per_hand" in prompt
    assert "expected_duration_hours" in prompt
    assert "num_subjects" in prompt
    assert "is_multi_embodiment" in prompt
    assert "paper_url" in prompt


def test_parse_and_validate_agent_output_accepts_new_fields():
    yaml_text = """
id: droid
name: DROID
release_type: fixed_episode_dataset
is_multi_embodiment: false
paper_url: "https://arxiv.org/abs/1234.5678"
additional_modalities:
  - force_torque
hand_pose_representation: joint_angles
num_subjects: 5
review_status: pending_human_review
"""
    config = parse_and_validate_agent_output(yaml_text)
    assert config.release_type.value == "fixed_episode_dataset"
    assert config.additional_modalities[0].value == "force_torque"
    assert config.hand_pose_representation.value == "joint_angles"


def test_build_onboarding_prompt_includes_round2_enum_values():
    prompt = build_onboarding_prompt("droid", "DROID", "https://droid-dataset.github.io/")
    assert "abb_yumi" in prompt
    assert "ego_exo_human" in prompt
    assert "custom_research_eula" in prompt
    assert "half_humanoid" in prompt
    assert "cage_pinch" in prompt
    assert "single_axis_angle" in prompt
    assert "imu" in prompt
    assert "discrete_symbolic" in prompt
    assert "gripper_jaw" in prompt
