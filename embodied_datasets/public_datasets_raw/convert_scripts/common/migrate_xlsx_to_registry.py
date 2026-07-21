"""Migrate the legacy VLA Research.xlsx into datasets_registry.yaml and
per-dataset config stubs.

Per the 2026-07-08 decision, only two columns are trusted from the
spreadsheet: 名称 (name) and 官方下载链接 (official download link). Every
other column in the spreadsheet is sparse/inconsistent and is intentionally
dropped -- the rest of each DatasetConfig is filled in later by the
onboarding agent (see onboarding_agent.py).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import openpyxl

from .schema import DatasetConfig, RegistryEntry


def slugify(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    if not slug:
        raise ValueError(f"cannot derive a slug from name: {name!r}")
    return slug


def read_name_and_link_rows(
    xlsx_path: Path, sheet_name: str = "VLA公开数据集"
) -> List[Tuple[str, Optional[str]]]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    name_col = header.index("名称")
    link_col = header.index("官方下载链接")
    results: List[Tuple[str, Optional[str]]] = []
    for row in rows[1:]:
        if name_col >= len(row) or row[name_col] is None:
            continue
        name = str(row[name_col]).strip()
        if not name:
            continue
        link = row[link_col] if link_col < len(row) else None
        link = str(link).strip() if link else None
        results.append((name, link))
    return results


def build_registry_and_configs(
    rows: List[Tuple[str, Optional[str]]]
) -> Tuple[List[RegistryEntry], List[DatasetConfig]]:
    entries: List[RegistryEntry] = []
    configs: List[DatasetConfig] = []
    seen_ids = set()
    for name, link in rows:
        dataset_id = slugify(name)
        if dataset_id in seen_ids:
            raise ValueError(
                f"duplicate dataset id derived from name: {dataset_id!r} (name={name!r})"
            )
        seen_ids.add(dataset_id)
        entries.append(RegistryEntry(id=dataset_id, name=name))
        configs.append(DatasetConfig(id=dataset_id, name=name, source_url=link))
    return entries, configs
