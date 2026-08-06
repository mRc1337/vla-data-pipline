"""Builds the single JSON payload inspect_tool's synced player component
needs for one episode -- video clip locations/offsets plus ready-to-render
Plotly figure specs for every state/action band. All numeric-array/Plotly
work is delegated to views.py's existing state_bands/action_bands/
slice_band/band_overlay_figure so this module owns exactly one thing:
assembling those pieces (plus video_locator's clip lookups) into the shape
player_component.py's JS expects.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from video_locator import video_clip_info
from views import action_bands, band_overlay_figure, slice_band, state_bands


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", value)


def build_video_urls(
    raw_root: Path,
    final_root: Optional[Path],
    raw_port: int,
    final_port: Optional[int],
    episode_index: int,
    view_keys: List[str],
    final_episode_index: Optional[int] = None,
) -> Dict[str, dict]:
    """One entry per view key: {"raw": clip_or_None, "final": clip_or_None},
    where each clip is {"url", "from_timestamp", "to_timestamp",
    "element_id"}. `final_root`/`final_port` are None when the final dataset
    isn't available yet (pipeline still running) -- every view's "final"
    comes back None in that case, same as a view whose video just hasn't
    been written for some other reason.

    `episode_index` addresses the RAW dataset. `final_episode_index`, when
    given, addresses the FINAL dataset instead -- the final dataset only
    contains surviving episodes, renumbered positionally (0..M-1) in write
    order, so a raw episode's final positional index can differ from its raw
    episode_index whenever an earlier raw episode was rejected (see
    views.raw_to_final_episode_index_map). Defaults to `episode_index` when
    not given, so callers that never see raw/final index drift (e.g. a
    single-episode dataset with nothing rejected) don't need to pass it."""
    resolved_final_index = episode_index if final_episode_index is None else final_episode_index
    urls: Dict[str, dict] = {}
    for view in view_keys:
        raw_clip = _clip_or_none(raw_root, raw_port, episode_index, view, "raw")
        final_clip = (
            _clip_or_none(final_root, final_port, resolved_final_index, view, "final")
            if final_root is not None and final_port is not None
            else None
        )
        urls[view] = {"raw": raw_clip, "final": final_clip}
    return urls


def _clip_or_none(root: Path, port: int, episode_index: int, view_key: str, side: str) -> Optional[dict]:
    info = video_clip_info(root, episode_index, view_key)
    if info is None:
        return None
    return {
        "url": f"http://127.0.0.1:{port}/{info['relative_path']}",
        "from_timestamp": info["from_timestamp"],
        "to_timestamp": info["to_timestamp"],
        "element_id": f"video-{_safe_id(view_key)}-{side}",
    }


def _band_group_payload(group_name: str, bands, raw_array, final_array, mask) -> List[dict]:
    group = []
    for band in bands:
        raw_slice = slice_band(raw_array, None, band)
        final_slice = slice_band(final_array, mask, band) if final_array is not None else None
        fig = band_overlay_figure(band.label, raw_slice, final_slice)
        group.append({
            "label": band.label,
            "populated": True if final_slice is None else final_slice["populated"],
            "figure": fig.to_plotly_json(),
            "div_id": f"chart-{group_name}-{_safe_id(band.label)}",
        })
    return group


def build_payload(raw_episode, final_episode, config, masks: dict, video_urls: Dict[str, dict], fps: float) -> dict:
    """`masks` is {"state": array_or_None, "action": array_or_None}, as
    already loaded by app.py's _load_canonical_masks. `video_urls` is
    build_video_urls()'s return value for this same episode."""
    final_state = final_episode.state if final_episode is not None else None
    final_action = final_episode.action if final_episode is not None else None

    return {
        "fps": float(fps),
        "views": sorted(raw_episode.frames.keys()),
        "videos": video_urls,
        "raw_frame_count": int(raw_episode.state.shape[0]),
        "final_frame_count": int(final_episode.state.shape[0]) if final_episode is not None else 0,
        "charts": {
            "state": _band_group_payload(
                "state", state_bands(config.num_arms), raw_episode.state, final_state, masks["state"],
            ),
            "action": _band_group_payload(
                "action", action_bands(config.num_arms), raw_episode.action, final_action, masks["action"],
            ),
        },
    }
