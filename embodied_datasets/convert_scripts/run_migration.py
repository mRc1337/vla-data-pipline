"""One-off script: migrate VLA Research.xlsx into datasets_registry.yaml and
per-dataset onboarding config stubs.

Run from repo root:
    python3 embodied_datasets/public_datasets_raw/convert_scripts/run_migration.py
"""
from __future__ import annotations

from pathlib import Path

from common.io import save_dataset_config, save_registry
from common.migrate_xlsx_to_registry import (
    build_registry_and_configs,
    read_name_and_link_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
XLSX_PATH = REPO_ROOT / "VLA Research.xlsx"
REGISTRY_PATH = REPO_ROOT / "embodied_datasets" / "datasets_registry.yaml"
CONFIGS_DIR = (
    REPO_ROOT
    / "embodied_datasets"
    / "convert_scripts"
    / "configs"
)


def main() -> None:
    rows = read_name_and_link_rows(XLSX_PATH)
    entries, configs = build_registry_and_configs(rows)
    save_registry(entries, REGISTRY_PATH)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    for config in configs:
        save_dataset_config(config, CONFIGS_DIR / f"{config.id}.yaml")
    print(f"wrote {len(entries)} registry entries to {REGISTRY_PATH}")
    print(f"wrote {len(configs)} config stubs to {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
