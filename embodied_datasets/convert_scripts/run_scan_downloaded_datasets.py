"""CLI wrapper: scan a data root for already-downloaded datasets and
report their presence and size. With --update-registry, also writes
download_status/storage_size_gb/raw_local_path into
datasets_registry.yaml for every matched dataset.

Folder names are matched to a dataset id/name after normalizing away
case/hyphen/underscore/dot differences (see
common/scan_downloaded_datasets.py). Any non-empty folder under
public_datasets_raw/ that doesn't match anything is printed as
"unmatched" rather than silently skipped -- check
FOLDER_NAME_ALIASES in that module if a real dataset folder ends up
there.

Run from repo root:
    python3 embodied_datasets/convert_scripts/run_scan_downloaded_datasets.py --data-root /path/to/data
    python3 embodied_datasets/convert_scripts/run_scan_downloaded_datasets.py --data-root /path/to/data --update-registry
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.io import load_registry, save_registry
from common.paths import resolve_data_root
from common.scan_downloaded_datasets import scan_raw_directory
from common.schema import DownloadStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "embodied_datasets" / "datasets_registry.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory containing public_datasets_raw/. Defaults to "
        "the in-repo embodied_datasets/ directory.",
    )
    parser.add_argument(
        "--update-registry",
        action="store_true",
        help="Write download_status/storage_size_gb/raw_local_path into "
        "datasets_registry.yaml for every dataset found.",
    )
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    entries = load_registry(REGISTRY_PATH)
    datasets = [(entry.id, entry.name) for entry in entries]

    found, unmatched = scan_raw_directory(data_root, datasets)

    print(f"Scanned data root: {data_root}")
    print(f"Found {len(found)}/{len(datasets)} datasets with downloaded data:")
    for item in sorted(found, key=lambda f: f.dataset_id):
        print(
            f"  {item.dataset_id} (folder: {item.folder_name}): "
            f"{item.size_bytes / 1e9:.2f} GB"
        )
    if unmatched:
        print(f"Unmatched non-empty folders ({len(unmatched)}) -- not in registry "
              "or no id/name/alias match:")
        for folder_name in unmatched:
            print(f"  {folder_name}")

    if args.update_registry:
        found_by_id = {item.dataset_id: item for item in found}
        for entry in entries:
            item = found_by_id.get(entry.id)
            if item is not None:
                entry.download_status = DownloadStatus.COMPLETED
                entry.storage_size_gb = round(item.size_bytes / 1e9, 2)
                entry.raw_local_path = item.folder_name
        save_registry(entries, REGISTRY_PATH)
        print(f"Updated datasets_registry.yaml for {len(found)} datasets.")


if __name__ == "__main__":
    main()
