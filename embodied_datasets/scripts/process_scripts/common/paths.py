"""Resolve the configurable data root and build paths under it for the
heavy data directories used by process_scripts (public_datasets_staging/,
public_datasets/lerobot_v3_0/). configs/ and registry_configs/ always stay
inside the repo and are unaffected by this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPO_ROOT / "embodied_datasets" / "data_root"


def resolve_data_root(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).resolve()
    return DEFAULT_DATA_ROOT


def staging_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "public_datasets_staging" / "lerobot_v3_0" / dataset_id


def final_dir(data_root: Path, dataset_id: str) -> Path:
    return data_root / "public_datasets" / "lerobot_v3_0" / dataset_id
