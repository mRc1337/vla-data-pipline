"""Scan a data root for already-downloaded dataset directories and report
their presence and size. Read-only against the filesystem -- does not
itself decide whether to mutate datasets_registry.yaml (see
run_scan_downloaded_datasets.py for the CLI wrapper that optionally does).

Real download directories are rarely named after our registry's slug
(e.g. a user names their download folder 'AgiBot-World', not
'agibot_world') -- this module matches folder names against dataset ids
and registry names after normalizing away case/hyphen/underscore/dot
differences, plus a small table of aliases for folders whose name is a
genuinely different label (e.g. a dataset's actual HF/paper name) rather
than just a formatting variant.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

# Folder names that don't normalize-match their dataset id or registry
# name -- add an entry here when a real download directory's name is a
# genuinely different label for a dataset already in the registry.
# Keys are already normalized (see _normalize below).
FOLDER_NAME_ALIASES: Dict[str, str] = {
    "omniaction": "roboomni",
    "robotset": "roboset",
    "robotwin20": "robotwin",
    "lerobot": "lerobot_full_folding",
}


class FoundDataset(NamedTuple):
    dataset_id: str
    folder_name: str
    size_bytes: int


def _normalize(name: str) -> str:
    return re.sub(r"[-_.\s]", "", name).lower()


def _dir_size_bytes(path: Path) -> int:
    """os.scandir-based walk instead of Path.rglob(): DirEntry.is_file()/
    stat() reuse the lstat info the OS already returned from the directory
    read, where Path.rglob() re-stats every entry from scratch -- on raw
    download directories with large file counts (many datasets ship as
    thousands of per-frame images or shards) the difference is the
    dominant cost of a full registry scan.
    """
    total = 0
    stack = [path]
    while stack:
        with os.scandir(stack.pop()) as it:
            for entry in it:
                if entry.is_dir():
                    stack.append(entry.path)
                elif entry.is_file():
                    total += entry.stat().st_size
    return total


def scan_raw_directory(
    data_root: Path, datasets: List[Tuple[str, str]]
) -> Tuple[List[FoundDataset], List[str]]:
    """datasets: list of (dataset_id, name) pairs, e.g. from the
    registry. Lists every subdirectory directly under
    data_root/public_datasets_raw/, matches each one to a dataset_id by
    normalized id, normalized name, or FOLDER_NAME_ALIASES, and reports
    its total size. Returns (found, unmatched_folder_names) -- unmatched
    non-empty folders are reported by name so nothing is silently
    skipped."""
    lookup: Dict[str, str] = {}
    known_ids = set()
    for dataset_id, name in datasets:
        lookup[_normalize(dataset_id)] = dataset_id
        lookup[_normalize(name)] = dataset_id
        known_ids.add(dataset_id)
    for folder_key, dataset_id in FOLDER_NAME_ALIASES.items():
        if dataset_id in known_ids:
            lookup[folder_key] = dataset_id

    # Mirrors common/paths.py's raw_dir() (data_root/public_datasets_raw/
    # <dataset_id>) -- keep both in sync if that layout ever changes.
    raw_root = data_root / "public_datasets_raw"
    found: List[FoundDataset] = []
    unmatched: List[str] = []
    if not raw_root.is_dir():
        return found, unmatched

    for entry in sorted(raw_root.iterdir()):
        if not entry.is_dir():
            continue
        size = _dir_size_bytes(entry)
        if size == 0:
            continue
        dataset_id = lookup.get(_normalize(entry.name))
        if dataset_id is None:
            unmatched.append(entry.name)
        else:
            found.append(FoundDataset(dataset_id, entry.name, size))
    return found, unmatched
