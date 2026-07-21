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
