"""Prompt template and output validator for the dataset onboarding research
agent (design doc section 8, and the 2026-07-10 refinement section 3.4).
This module does not call any LLM -- it only (a) builds the research
prompt an Agent-tool task should receive, and (b) validates/parses
whatever YAML that task returns before it is trusted as a `DatasetConfig`.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Type

import yaml
from pydantic import ValidationError

from .schema import (
    ActionFrame,
    ActionSpace,
    CameraView,
    CollectionMethod,
    DatasetConfig,
    DepthCoverage,
    EmbodimentClass,
    GripperType,
    HandPoseRepresentation,
    LicenseEnum,
    RawFormat,
    ReleaseType,
    RobotPlatform,
    RotationRepresentation,
    SensorModality,
    UrdfSource,
)

_ENUM_FIELDS: Dict[str, Type[Enum]] = {
    "license": LicenseEnum,
    "raw_format": RawFormat,
    "release_type": ReleaseType,
    "collection_method": CollectionMethod,
    "secondary_collection_methods": CollectionMethod,
    "embodiment_class": EmbodimentClass,
    "robot_platform": RobotPlatform,
    "gripper_type": GripperType,
    "hand_pose_representation": HandPoseRepresentation,
    "action_space": ActionSpace,
    "action_frame": ActionFrame,
    "rotation_representation": RotationRepresentation,
    "depth_coverage": DepthCoverage,
    "additional_modalities": SensorModality,
    "urdf_source": UrdfSource,
    "camera_views": CameraView,
}


def _format_enum_choices(enum_cls: Type[Enum]) -> str:
    return ", ".join(member.value for member in enum_cls)


def build_onboarding_prompt(dataset_id: str, name: str, source_url: Optional[str]) -> str:
    enum_lines = "\n".join(
        f"- {field}: {_format_enum_choices(enum_cls)}"
        for field, enum_cls in _ENUM_FIELDS.items()
    )
    source_line = source_url or "(未提供，请自行搜索该数据集官网/论文)"
    return f"""\
你正在为数据集 "{name}"（id: {dataset_id}）调研结构化元数据。

可信来源（优先使用，找不到再自行搜索官网/论文）：
{source_line}

请仔细阅读官网和对应论文，为下列每个字段给出取值。每个枚举字段的值必须
严格从给定选项中选择；如果实际情况不在选项里，不要编造，而是把这个字段
留空，并在 suggested_new_enum_values 里写下建议新增的枚举值和理由。

枚举字段及可选值：
{enum_lines}

其他字段：num_arms(0/1/2的整数), dof_per_arm(整数), dof_per_hand(整数：灵巧
手/手指自由度，跟dof_per_arm是互补关系，不是替代), has_mobile_base(布尔),
state_dim/action_dim(整数), fps(数值), fps_variable(布尔),
num_camera_views(整数), has_camera_calibration(布尔),
has_synchronized_multiview_rig(布尔：多相机是否同步组成一个阵列，跟
camera_views里有哪些视角是不同维度), has_language_instruction(布尔),
num_task_types(整数), urdf_available(布尔), expected_size_gb(数值),
expected_num_episodes(整数), expected_duration_hours(数值：声明的总时长小时数),
num_subjects(整数：人类被试数量，仅适用于人类采集的数据集),
num_scenes(整数：场景/环境数量), num_objects(整数：交互物体数量),
is_multi_embodiment(布尔：是否同一份发布同时横跨多种具身形态，如果是就把
embodiment_class/robot_platform留空，不要强行选一个), paper_url(字符串：
对应论文链接，跟source_url分开存)。

对你填写的每一个字段，在 field_sources 里记录信息来源（URL 或论文章节）。

输出必须是可以直接解析为以下YAML结构的文本（不要用markdown代码块包裹），
顶层字段名必须与上面列出的字段名完全一致，另外必须包含:
id: {dataset_id}
name: {name}
review_status: pending_human_review
"""


def parse_and_validate_agent_output(yaml_text: str) -> DatasetConfig:
    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError("agent output must parse to a YAML mapping")
    try:
        return DatasetConfig(**raw)
    except ValidationError as exc:
        raise ValueError(f"agent output failed schema validation: {exc}") from exc
