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
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys
from typing import Sequence

from pydantic import ValidationError

from convert_core.dataset_config import DatasetConversionConfig, load_dataset_config
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import convert_dataset, plan_summary
from readers.registry import get_reader


def _select_episodes(
    plan,
    *,
    tasks: list[str],
    max_episodes: int | None,
    max_checkpoint_units: int | None,
    output_dataset_uid: str | None,
):
    episodes = list(plan.episodes)
    if tasks:
        requested = set(tasks)
        episodes = [episode for episode in episodes if episode.instruction in requested]
        missing = requested - {episode.instruction for episode in episodes}
        if missing:
            raise ConversionError(f"requested tasks are absent: {sorted(missing)}")
    if max_checkpoint_units is not None:
        selected_units: list[str] = []
        kept = []
        for episode in episodes:
            unit = str(episode.extra.get("checkpoint_unit", episode.episode_uid))
            if unit not in selected_units:
                if len(selected_units) >= max_checkpoint_units:
                    break
                selected_units.append(unit)
            kept.append(episode)
        episodes = kept
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ConversionError("episode selection is empty")
    if output_dataset_uid is not None:
        if Path(output_dataset_uid).name != output_dataset_uid or output_dataset_uid in {"", ".", ".."}:
            raise ConversionError(f"output dataset UID must be one path component: {output_dataset_uid!r}")
        output = plan.output_path.with_name(output_dataset_uid)
        return replace(plan, dataset_uid=output_dataset_uid, output_path=output, episodes=tuple(episodes))
    return replace(plan, episodes=tuple(episodes))


def _run_one(
    config_path: Path,
    *,
    raw_root: Path,
    staging_root: Path,
    dry_run: bool,
    skip_existing: bool,
    overwrite: bool,
    resume: bool,
    eta_interval_seconds: float,
    tasks: list[str],
    max_episodes: int | None,
    max_checkpoint_units: int | None,
    output_dataset_uid: str | None,
) -> str:
    config: DatasetConversionConfig = load_dataset_config(config_path)
    reader = get_reader(config.format)

    expected_uid = output_dataset_uid or config.dataset_uid
    expected_output = staging_root / "lerobot_v3_0" / expected_uid
    if expected_output.exists() and skip_existing and not dry_run:
        print(f"[{config.dataset_uid}] skipped existing output: {expected_output}")
        return "skipped"

    plan = reader.build_plan(config, raw_root, staging_root)
    plan = _select_episodes(
        plan,
        tasks=tasks,
        max_episodes=max_episodes,
        max_checkpoint_units=max_checkpoint_units,
        output_dataset_uid=output_dataset_uid,
    )
    print(json.dumps(plan_summary(plan), ensure_ascii=False, indent=2, default=str))
    if dry_run:
        return "validated"

    output = convert_dataset(
        plan,
        lambda episode: reader.iter_frames(plan, episode),
        reader_format=config.format,
        overwrite=overwrite,
        resume=resume,
        eta_interval_seconds=eta_interval_seconds,
        conversion_options={"config_path": str(config_path)},
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from verified reader-defined checkpoints beside the final output.",
    )
    parser.add_argument(
        "--eta-interval-seconds",
        type=float,
        default=10.0,
        help="Seconds between newline progress/ETA records.",
    )
    parser.add_argument("--task", action="append", default=[], help="Keep one exact task string; repeatable.")
    parser.add_argument("--max-episodes", type=int, help="Keep only the first N selected episodes.")
    parser.add_argument(
        "--max-checkpoint-units",
        type=int,
        help="Keep only the first N reader-defined parts/shards/checkpoint units.",
    )
    parser.add_argument(
        "--output-dataset-uid",
        help="Use an independent UID/output path, required for non-dry-run subset conversions.",
    )
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
    selected_modes = sum(bool(value) for value in (args.skip_existing, args.overwrite, args.resume))
    if selected_modes > 1:
        parser.error("--skip-existing, --overwrite, and --resume are mutually exclusive")
    for name in ("max_episodes", "max_checkpoint_units"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.eta_interval_seconds <= 0:
        parser.error("--eta-interval-seconds must be positive")
    selection_active = bool(args.task or args.max_episodes or args.max_checkpoint_units)
    if selection_active and not args.dry_run and args.output_dataset_uid is None:
        parser.error("subset conversion requires --output-dataset-uid to protect the full dataset UID")
    config_paths = _resolve_config_paths(args, parser)

    converted = skipped = validated = 0
    had_error = False
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        for config_path in config_paths:
            try:
                status = _run_one(
                    config_path,
                    raw_root=args.raw_root,
                    staging_root=args.staging_root,
                    dry_run=args.dry_run,
                    skip_existing=args.skip_existing,
                    overwrite=args.overwrite,
                    resume=args.resume,
                    eta_interval_seconds=args.eta_interval_seconds,
                    tasks=args.task,
                    max_episodes=args.max_episodes,
                    max_checkpoint_units=args.max_checkpoint_units,
                    output_dataset_uid=args.output_dataset_uid,
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
    except KeyboardInterrupt:
        print("interrupted; --resume retained only the last verified checkpoint unit", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    if args.dry_run:
        print(f"validated {validated} dataset(s); no output written")
    else:
        print(f"completed: converted={converted}, skipped={skipped}")
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
