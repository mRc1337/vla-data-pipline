#!/usr/bin/env python3
"""Rebuild a compact SQLite catalog without mutating the live catalog.

Example:
  python scripts/rebuild_compact_catalog.py \
    --source .local-run/catalog.sqlite3 \
    --target .local-run/catalog.compact.sqlite3 \
    --data-root /mnt/data/embodied_datasets/public_datasets_staging

After validating the target, point VLA_CATALOG_DB at it and restart the API.
The source database is never deleted, vacuumed, or opened for writing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_platform.catalog import Catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.absolute()
    target = args.target.absolute()
    if source == target:
        parser.error("--target must differ from --source")
    if not source.is_file():
        parser.error(f"source catalog does not exist: {source}")
    result = Catalog(source, args.data_root).rebuild_compact_database(target)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
