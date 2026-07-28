"""Load/save the dataset registry overview table, per-dataset onboarding
configs, and process_scripts/configs/<id>.yaml as YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from .schema import DatasetConfig, ProcessConfig, RegistryEntry


def load_registry(path: Path) -> List[RegistryEntry]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [RegistryEntry(**entry) for entry in raw]


def save_registry(entries: List[RegistryEntry], path: Path) -> None:
    data = [entry.model_dump(mode="json", exclude_none=True) for entry in entries]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def load_dataset_config(path: Path) -> DatasetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetConfig(**raw)


def save_dataset_config(config: DatasetConfig, path: Path) -> None:
    data = config.model_dump(mode="json", exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def load_process_config(path: Path) -> ProcessConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProcessConfig(**raw)


def save_process_config(config: ProcessConfig, path: Path) -> None:
    data = config.model_dump(mode="json", exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
