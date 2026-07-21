"""One-off script: apply the 2026-07-10 registry schema refinement's
table-driven corrections (common/backfill_schema_refinement.py) to every
existing dataset config.

Run from repo root:
    python3 embodied_datasets/public_datasets_raw/convert_scripts/run_backfill_schema_refinement.py
"""
from __future__ import annotations

from pathlib import Path

from common.backfill_schema_refinement import apply_corrections
from common.io import load_dataset_config, save_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = (
    REPO_ROOT
    / "embodied_datasets"
    / "public_datasets_raw"
    / "convert_scripts"
    / "configs"
)


def main() -> None:
    paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    for path in paths:
        config = load_dataset_config(path)
        updated = apply_corrections(config)
        save_dataset_config(updated, path)
    print(f"backfilled {len(paths)} dataset configs in {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
