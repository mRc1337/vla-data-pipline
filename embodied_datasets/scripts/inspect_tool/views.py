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
import plotly.graph_objects as go

import unify_representation as ur

# Cycled by dimension index so a dimension's raw (dotted) and final (solid)
# traces share a color -- that's what makes the overlay readable.
_DIM_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


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


def band_overlay_figure(label: str, raw_slice: dict, final_slice: Optional[dict]) -> go.Figure:
    """One figure per band with raw and final overlaid per-dimension (same
    color, dotted vs solid) instead of two separate charts, so drift from
    cleaning/canonicalization is visible directly instead of eyeballed
    across two plots. Raw and final frame counts can differ (episodes lose
    frames during cleaning) -- both are plotted against their own frame
    index, not a shared/aligned x-axis."""
    fig = go.Figure()
    raw_values = raw_slice["values"]
    for dim in range(raw_values.shape[1]):
        color = _DIM_COLORS[dim % len(_DIM_COLORS)]
        fig.add_trace(go.Scatter(
            y=raw_values[:, dim], mode="lines", name=f"dim{dim} (raw)",
            line=dict(color=color, dash="dot"),
        ))
    if final_slice is not None:
        final_values = final_slice["values"]
        for dim in range(final_values.shape[1]):
            color = _DIM_COLORS[dim % len(_DIM_COLORS)]
            fig.add_trace(go.Scatter(
                y=final_values[:, dim], mode="lines", name=f"dim{dim} (final)",
                line=dict(color=color),
            ))
    fig.update_layout(
        title=label, height=300, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h"),
    )
    return fig
