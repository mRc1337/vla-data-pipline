"""One-off script: apply the 2026-07-13 registry schema round-2 enum
extension's table-driven corrections
(common/backfill_schema_round2.py) to every existing dataset config.

Run from repo root:
    python3 embodied_datasets/convert_scripts/run_backfill_schema_round2.py
"""
from __future__ import annotations

from pathlib import Path

from common.backfill_schema_round2 import apply_round2_corrections
from common.io import load_dataset_config, save_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = (
    REPO_ROOT
    / "embodied_datasets"
    / "convert_scripts"
    / "configs"
)


def main() -> None:
    paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    for path in paths:
        config = load_dataset_config(path)
        updated = apply_round2_corrections(config)
        save_dataset_config(updated, path)
    print(f"backfilled {len(paths)} dataset configs in {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
