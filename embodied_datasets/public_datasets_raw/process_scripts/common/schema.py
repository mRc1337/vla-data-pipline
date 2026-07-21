"""Pydantic model for process_scripts/configs/<id>.yaml cleaning-threshold
parameters, plus runtime gate fields that run_pipeline.py populates from
convert_scripts' DatasetConfig/RegistryEntry before calling each stage/check
module (these are NOT hand-authored in the yaml file). Field names must
stay in sync with
docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
section 5.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProcessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str

    # Stage1: sudden-change detection
    savgol_window: int = 11
    savgol_polyorder: int = Field(default=3, ge=0)
    residual_threshold: float = 0.05
    accel_threshold: float = 0.5
    jerk_threshold: float = 5.0
    per_dim_thresholds: Dict[str, Dict[str, Dict[str, float]]] = Field(default_factory=dict)
    episode_reject_threshold: float = 0.3

    # Stage2: state-action trend alignment
    da_threshold: float = 0.65
    max_lag_frames: int = 5

    # Stage3: extreme-value filtering
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    gripper_dims_state: List[int] = Field(default_factory=list)
    gripper_dims_action: List[int] = Field(default_factory=list)
    extreme_value_bounds: Optional[Dict[str, Dict[int, List[float]]]] = None

    # Stage4: FK consistency
    tcp_offset_tolerance: float = 0.02
    fk_check_feasible: bool = False
    urdf_path: Optional[str] = None

    # Stage5: orientation/world-frame alignment
    world_frame_convention: str = "robot_base"
    base_to_world_transform: Optional[List[float]] = None

    # Check1: instruction consistency
    vlm_service_url: Optional[str] = None
    has_language_instruction: bool = False

    # Check2: video-state consistency
    sam3_service_url: Optional[str] = None
    iou_threshold: float = 0.5
    urdf_available: bool = False
    has_camera_calibration: bool = False

    # Check3: video quality
    black_threshold: float = 10.0
    blur_threshold: float = 100.0
    still_threshold: float = 1.0
    still_min_consecutive_frames: int = 30

    # Unify representation
    camera_frame_delta_pose_enabled: bool = False
    embodiment_class: Optional[str] = None
    num_arms: int = 1
    dof_per_arm: Optional[int] = None
    gripper_type: str = "unknown"
    has_mobile_base: bool = False
