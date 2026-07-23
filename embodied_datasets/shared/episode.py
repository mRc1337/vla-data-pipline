"""In-memory data structures every stage/check module operates on. See
docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
section 6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Episode:
    episode_index: int
    timestamps: np.ndarray
    state: np.ndarray
    action: np.ndarray
    frames: Dict[str, np.ndarray] = field(default_factory=dict)
    language_instruction: Optional[str] = None


@dataclass
class StageResult:
    episode: Episode
    dropped_frame_indices: List[int] = field(default_factory=list)
    rejected: bool = False
    skip_reason: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
