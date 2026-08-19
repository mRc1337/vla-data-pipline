"""In-memory data structures every stage/check module operates on. See
docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
section 6 and
docs/superpowers/specs/2026-07-28-sam3-check2-camera-calibration-design.md
section 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    extrinsics: np.ndarray  # (4,4), camera_from_base: p_cam = extrinsics @ [*p_base, 1]


@dataclass
class Episode:
    episode_index: int
    timestamps: np.ndarray
    state: np.ndarray
    action: np.ndarray
    # Optional mobile-base command kept separate from the arm action.  Mobile
    # ALOHA stores [linear_velocity, angular_velocity] in the independent
    # LeRobot feature ``action.base``; it is not appended to state or action.
    base_action: Optional[np.ndarray] = None
    frames: Dict[str, np.ndarray] = field(default_factory=dict)
    language_instruction: Optional[str] = None
    camera_calibration: Dict[str, CameraCalibration] = field(default_factory=dict)


@dataclass
class StageResult:
    episode: Episode
    dropped_frame_indices: List[int] = field(default_factory=list)
    rejected: bool = False
    skip_reason: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
