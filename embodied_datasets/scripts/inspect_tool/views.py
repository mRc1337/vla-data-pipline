"""Pure helper functions for inspect_tool/app.py's display logic --
grouping canonical 128-dim state/action vectors into named bands
(joint/eef/gripper/reserve) for plotting, and summarizing an episode's
per-stage processing record for display. No streamlit imports here, so
these stay independently unit-testable (app.py is not automated-tested,
per the design doc's testing plan).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_PROCESS_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "process_scripts"
if str(_PROCESS_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESS_SCRIPTS_DIR))

import numpy as np

import unify_representation as ur


@dataclass
class Band:
    label: str
    start: int
    end: int  # exclusive


def episode_status_label(episode_record: dict) -> str:
    """Returns e.g. "kept", or "rejected at stage2_trend_alignment" for the
    first stage whose `rejected` is True (there is at most one rejecting
    stage per episode, since run_pipeline.py's -- and this tool's own --
    orchestration stops handing a rejected episode to the next stage)."""
    for stage in episode_record["stages"]:
        if stage["rejected"]:
            return f"rejected at {stage['stage']}"
    if episode_record["survived"]:
        return "kept"
    return "dropped (all frames removed)"


def state_bands(num_arms: int) -> List[Band]:
    num_arms = max(1, min(num_arms, 2))
    bands = []
    for arm_idx in range(num_arms):
        offset = arm_idx * ur.ARM_BLOCK_DIM
        bands.append(Band(f"arm{arm_idx + 1}_joint", offset, offset + ur.JOINT_SLOT))
        bands.append(Band(f"arm{arm_idx + 1}_eef", offset + ur.JOINT_SLOT, offset + ur.JOINT_SLOT + ur.EEF_SLOT))
        bands.append(Band(
            f"arm{arm_idx + 1}_gripper",
            offset + ur.JOINT_SLOT + ur.EEF_SLOT,
            offset + ur.ARM_BLOCK_DIM,
        ))
    bands.append(Band("reserve", num_arms * ur.ARM_BLOCK_DIM, ur.CANONICAL_DIM))
    return bands


def action_bands(num_arms: int) -> List[Band]:
    num_arms = max(1, min(num_arms, 2))
    bands = []
    for arm_idx in range(num_arms):
        offset = arm_idx * ur.ACTION_ARM_BLOCK_DIM
        eef_offset = offset + ur.ACTION_JOINT_SLOT
        gripper_offset = eef_offset + ur.ACTION_EEF_POS_SLOT + ur.ACTION_EEF_ROT_SLOT + 1
        bands.append(Band(f"arm{arm_idx + 1}_joint", offset, eef_offset))
        bands.append(Band(f"arm{arm_idx + 1}_eef_pos", eef_offset, eef_offset + ur.ACTION_EEF_POS_SLOT))
        bands.append(Band(
            f"arm{arm_idx + 1}_eef_rot",
            eef_offset + ur.ACTION_EEF_POS_SLOT,
            eef_offset + ur.ACTION_EEF_POS_SLOT + ur.ACTION_EEF_ROT_SLOT,
        ))
        bands.append(Band(f"arm{arm_idx + 1}_gripper", gripper_offset, offset + ur.ACTION_ARM_BLOCK_DIM))
    bands.append(Band("reserve", num_arms * ur.ACTION_ARM_BLOCK_DIM, ur.ACTION_CANONICAL_DIM))
    return bands


def slice_band(vector: np.ndarray, mask: Optional[np.ndarray], band: Band) -> dict:
    """vector: shape (num_frames, dim). Returns the band's columns plus,
    when a mask is given, whether every dim in the band is unpopulated
    (all mask[start:end] False) -- used to grey out bands that are
    entirely zero-padding for this embodiment. Without a mask (e.g. slicing
    the RAW, pre-canonicalization array), the band is always reported
    populated -- there is no mask concept for raw per-dataset data."""
    values = vector[:, band.start:band.end]
    band_mask = None if mask is None else mask[band.start:band.end]
    populated = True if band_mask is None else bool(np.any(band_mask))
    return {"label": band.label, "values": values, "mask": band_mask, "populated": populated}
