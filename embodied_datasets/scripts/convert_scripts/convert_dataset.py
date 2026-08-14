"""Unified CLI entry point for the modular convert_scripts framework.

Dispatches one ``configs/<dataset-uid>.yaml`` (see
``convert_core.dataset_config.DatasetConversionConfig``) to the reader
registered for its ``format`` (see ``readers/registry.py``), then writes
through the same ``convert_core.lerobot_writer`` every reader shares --
create -> add_frame -> save_episode -> finalize -> re-open and validate ->
atomically publish. Adding a new source format never touches this file:
write one ``readers/<format>_reader.py``, register it, done.

Usage::

    # one dataset, validate only -- never writes output
    python3 convert_dataset.py --config configs/<uid>.yaml \\
        --raw-root /data/public_datasets_raw --staging-root /data/public_datasets_staging --dry-run

    # one dataset, convert
    python3 convert_dataset.py --config configs/<uid>.yaml \\
        --raw-root /data/public_datasets_raw --staging-root /data/public_datasets_staging

    # every *.yaml in a directory
    python3 convert_dataset.py --configs-dir configs/ --all \\
        --raw-root /data/public_datasets_raw --staging-root /data/public_datasets_staging
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from pydantic import ValidationError

from convert_core.dataset_config import DatasetConversionConfig, load_dataset_config
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import convert_dataset, plan_summary
from readers.registry import get_reader


def _run_one(config_path: Path, *, raw_root: Path, staging_root: Path, dry_run: bool, skip_existing: bool, overwrite: bool) -> str:
    config: DatasetConversionConfig = load_dataset_config(config_path)
    reader = get_reader(config.format)

    expected_output = staging_root / "lerobot_v3_0" / config.dataset_uid
    if expected_output.exists() and skip_existing and not dry_run:
        print(f"[{config.dataset_uid}] skipped existing output: {expected_output}")
        return "skipped"

    plan = reader.build_plan(config, raw_root, staging_root)
    print(json.dumps(plan_summary(plan), ensure_ascii=False, indent=2, default=str))
    if dry_run:
        return "validated"

    output = convert_dataset(
        plan,
        lambda episode: reader.iter_frames(plan, episode),
        reader_format=config.format,
        overwrite=overwrite,
    )
    print(f"[{config.dataset_uid}] wrote {len(plan.episodes)} episodes / {plan.num_frames} frames to {output}")
    return "converted"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True, type=Path, help="Path to public_datasets_raw.")
    parser.add_argument("--staging-root", required=True, type=Path, help="Path to public_datasets_staging.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", type=Path, help="Path to one configs/<dataset-uid>.yaml.")
    selection.add_argument("--all", action="store_true", help="Convert every *.yaml file in --configs-dir.")
    parser.add_argument("--configs-dir", type=Path, help="Directory of per-dataset yaml configs, used with --all.")
    parser.add_argument(
        "--dry-run",
        "--inspect-only",
        dest="dry_run",
        action="store_true",
        help="Validate and print the plan without writing any output.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip dataset UIDs whose v3 output already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing UID only after new output validates.")
    return parser


def _resolve_config_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    if args.all:
        if args.configs_dir is None:
            parser.error("--all requires --configs-dir")
        paths = sorted(args.configs_dir.glob("*.yaml"), key=lambda path: path.name.casefold())
        if not paths:
            parser.error(f"no *.yaml files found in {args.configs_dir}")
        return paths
    if args.configs_dir is not None:
        parser.error("--config and --configs-dir are mutually exclusive; use --all --configs-dir to batch")
    return [args.config]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing and --overwrite cannot be used together")
    config_paths = _resolve_config_paths(args, parser)

    converted = skipped = validated = 0
    had_error = False
    for config_path in config_paths:
        try:
            status = _run_one(
                config_path,
                raw_root=args.raw_root,
                staging_root=args.staging_root,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                overwrite=args.overwrite,
            )
        except (ConversionError, FileExistsError, OSError, RuntimeError, ValidationError, ValueError) as exc:
            print(f"error: {config_path}: {exc}", file=sys.stderr)
            had_error = True
            if not args.all:
                return 1
            continue
        if status == "converted":
            converted += 1
        elif status == "skipped":
            skipped += 1
        elif status == "validated":
            validated += 1

    if args.dry_run:
        print(f"validated {validated} dataset(s); no output written")
    else:
        print(f"completed: converted={converted}, skipped={skipped}")
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
