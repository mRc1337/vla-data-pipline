"""Render a wide CSV export containing every DatasetConfig field for every
dataset in the registry. Complements generate_overview_readme.py's
9-column progress table in embodied_datasets/README.md -- this is the
"see everything" view, meant to be opened in a spreadsheet rather than
read in a terminal.

Run from repo root:
    python3 embodied_datasets/public_datasets_raw/convert_scripts/common/generate_full_export.py
"""
from __future__ import annotations

import csv
import io as io_module
from pathlib import Path
from typing import Any, List

from .io import load_dataset_config
from .schema import DatasetConfig

_COLUMNS = list(DatasetConfig.model_fields.keys())


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "; ".join(_format_value(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={v}" for k, v in value.items())
    if hasattr(value, "value"):
        return value.value
    return str(value)


def render_full_csv(configs: List[DatasetConfig]) -> str:
    buffer = io_module.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_COLUMNS)
    for config in configs:
        writer.writerow([_format_value(getattr(config, col)) for col in _COLUMNS])
    return buffer.getvalue()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    embodied_root = repo_root / "embodied_datasets"
    configs_dir = embodied_root / "public_datasets_raw" / "convert_scripts" / "configs"
    output_path = embodied_root / "datasets_full_export.csv"

    configs = [
        load_dataset_config(path) for path in sorted(configs_dir.glob("*.yaml"))
    ]
    output_path.write_text(render_full_csv(configs), encoding="utf-8")
    print(f"wrote full export for {len(configs)} datasets to {output_path}")


if __name__ == "__main__":
    main()
