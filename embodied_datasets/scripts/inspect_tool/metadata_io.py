"""Save/load the inspect_tool sidecar file (`_inspect_metadata.json`) that
records why each episode's frames were processed the way they were --
skip_reason/rejected/dropped_frame_indices/stats per stage, per episode.
Deliberately excludes large arrays (canonical_state/canonical_mask/
action_canonical/action_canonical_mask and their _dim siblings) since
those already live in the final lerobot dataset's own features; this file
only carries the in-between "why" that isn't recoverable from the two
datasets alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

_PROCESS_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "process_scripts"
if str(_PROCESS_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESS_SCRIPTS_DIR))

import numpy as np

from episode import StageResult

METADATA_FILENAME = "_inspect_metadata.json"

_EXCLUDED_STATS_KEYS = frozenset({
    "canonical_state", "canonical_mask", "canonical_dim",
    "action_canonical", "action_canonical_mask", "action_canonical_dim",
})


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def stage_result_to_record(stage_name: str, result: StageResult) -> dict:
    """Converts one stage's StageResult into a JSON-safe record, dropping
    large array fields that are redundant with the final lerobot dataset's
    own features."""
    filtered_stats = {k: v for k, v in result.stats.items() if k not in _EXCLUDED_STATS_KEYS}
    return {
        "stage": stage_name,
        "skip_reason": result.skip_reason,
        "rejected": bool(result.rejected),
        "dropped_frame_indices": list(result.dropped_frame_indices),
        "stats": _json_safe(filtered_stats),
    }


def metadata_path(output_path: Path) -> Path:
    return output_path / METADATA_FILENAME


def metadata_exists(output_path: Path) -> bool:
    return metadata_path(output_path).exists()


def save_metadata(metadata: dict, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path(output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(output_path: Path) -> Optional[dict]:
    path = metadata_path(output_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
