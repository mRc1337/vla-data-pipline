"""Render the auto-generated dataset overview table for
embodied_datasets/README.md (see
docs/superpowers/specs/2026-07-10-registry-schema-refinement-design.md
section 5). Reads datasets_registry.yaml + convert_scripts/configs/*.yaml
and replaces only the content between the AUTO-GENERATED TABLE markers in
the README -- everything else in the file is hand-written and left as-is.

Run from repo root:
    python3 embodied_datasets/convert_scripts/common/generate_overview_readme.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .io import load_dataset_config, load_registry
from .readme_sections import replace_marked_section as _replace_marked_section
from .schema import DatasetConfig, RegistryEntry

TABLE_START_MARKER = "<!-- AUTO-GENERATED TABLE START -->"
TABLE_END_MARKER = "<!-- AUTO-GENERATED TABLE END -->"

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}

_COLUMNS = [
    "id",
    "name",
    "priority",
    "download_status",
    "convert_status",
    "process_status",
    "review_status",
    "collection_method",
    "embodiment_class",
]


def render_overview_table(
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
        config = configs_by_id.get(entry.id)
        review_status = config.review_status.value if config else ""
        collection_method = (
            config.collection_method.value
            if config and config.collection_method
            else ""
        )
        embodiment_class = (
            config.embodiment_class.value
            if config and config.embodiment_class
            else ""
        )
        row = [
            entry.id,
            entry.name,
            entry.priority.value,
            entry.download_status.value,
            entry.convert_status.value,
            entry.process_status.value,
            review_status,
            collection_method,
            embodiment_class,
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def replace_marked_section(readme_text: str, table_markdown: str) -> str:
    return _replace_marked_section(
        readme_text, TABLE_START_MARKER, TABLE_END_MARKER, table_markdown
    )


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
    table_markdown = render_overview_table(entries, configs_by_id)
    readme_text = readme_path.read_text(encoding="utf-8")
    updated_text = replace_marked_section(readme_text, table_markdown)
    readme_path.write_text(updated_text, encoding="utf-8")
    print(f"refreshed overview table for {len(entries)} datasets in {readme_path}")


if __name__ == "__main__":
    main()
