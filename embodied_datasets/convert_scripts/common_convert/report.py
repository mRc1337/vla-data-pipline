"""ConversionReport: what every convert_scripts/<dataset_id>.py convert()
must return. See
docs/superpowers/specs/2026-07-21-convert-scripts-verify-scripts-design.md
section 7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConversionReport:
    num_episodes: int
    num_frames: int
    warnings: List[str] = field(default_factory=list)
    urdf_path: Optional[str] = None
