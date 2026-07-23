"""Resolve the configurable data root and build paths under it for the
heavy data directories (raw/, staging/, final/, urdf_assets/).
datasets_registry.yaml and convert_scripts/configs/ always stay inside the
repo and are unaffected by this module -- see
docs/superpowers/specs/2026-07-21-convert-scripts-verify-scripts-design.md
section 4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "embodied_datasets" / "data"


def resolve_data_root(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).resolve()
    return DEFAULT_DATA_ROOT


def raw_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "raw" / dataset_id


def staging_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "staging" / dataset_id


def final_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "final" / dataset_id


def urdf_assets_dir(data_root: Path, robot_platform: str) -> Path:
    return data_root / "urdf_assets" / robot_platform
