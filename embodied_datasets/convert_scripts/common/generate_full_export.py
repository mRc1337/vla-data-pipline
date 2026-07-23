"""Render the full DatasetConfig field table for every dataset into
embodied_datasets/README.md (see generate_overview_readme.py for the
9-column progress table -- this is the "see every declared field" view).
field_sources/suggested_new_enum_values are excluded: they're per-field
provenance/audit text, often several sentences long once a dataset has
been through re-verification, and would make the table unreadable.
The YAML configs remain the source of truth for those two fields.

Run from repo root:
    python3 embodied_datasets/convert_scripts/common/generate_full_export.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .io import load_dataset_config, load_registry
from .readme_sections import replace_marked_section
from .schema import DatasetConfig, RegistryEntry

FULL_TABLE_START_MARKER = "<!-- AUTO-GENERATED FULL TABLE START -->"
FULL_TABLE_END_MARKER = "<!-- AUTO-GENERATED FULL TABLE END -->"

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}

_EXCLUDED_COLUMNS = {"field_sources", "suggested_new_enum_values"}
_COLUMNS = [
    name for name in DatasetConfig.model_fields.keys() if name not in _EXCLUDED_COLUMNS
]


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


def render_full_table_markdown(
    entries: List[RegistryEntry], configs_by_id: Dict[str, DatasetConfig]
) -> str:
    sorted_entries = sorted(
        entries, key=lambda e: (_PRIORITY_ORDER[e.priority.value], e.name)
    )
    lines = [
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "---|" * len(_COLUMNS),
    ]
    for entry in sorted_entries:
        config = configs_by_id[entry.id]
        row = [_format_value(getattr(config, col)) for col in _COLUMNS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    embodied_root = repo_root / "embodied_datasets"
    registry_path = embodied_root / "datasets_registry.yaml"
    configs_dir = embodied_root / "convert_scripts" / "configs"
    readme_path = embodied_root / "README.md"

    entries = load_registry(registry_path)
    configs_by_id = {
        path.stem: load_dataset_config(path)
        for path in sorted(configs_dir.glob("*.yaml"))
    }
    table_markdown = render_full_table_markdown(entries, configs_by_id)
    readme_text = readme_path.read_text(encoding="utf-8")
    updated_text = replace_marked_section(
        readme_text, FULL_TABLE_START_MARKER, FULL_TABLE_END_MARKER, table_markdown
    )
    readme_path.write_text(updated_text, encoding="utf-8")
    print(f"refreshed full field table for {len(entries)} datasets in {readme_path}")


if __name__ == "__main__":
    main()
