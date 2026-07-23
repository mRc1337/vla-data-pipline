"""Orchestrates one dataset's convert() + post-convert self-check, then
updates datasets_registry.yaml. See
docs/superpowers/specs/2026-07-21-convert-scripts-verify-scripts-design.md
section 7.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_dataset_config, load_registry, save_registry  # noqa: E402
from common.paths import raw_dir, resolve_data_root, staging_dir  # noqa: E402
from common.schema import ConvertStatus, IntegrityStatus  # noqa: E402
from common_convert.self_check import run_self_check  # noqa: E402
from shared.lerobot_io import load_lerobot_episodes  # noqa: E402

EMBODIED_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EMBODIED_ROOT / "datasets_registry.yaml"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _write_log(dataset_id: str, lines: List[str]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{dataset_id}.log"
    log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert one dataset's raw data into staging LeRobot format.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)

    entries = load_registry(REGISTRY_PATH)
    entry = next((e for e in entries if e.id == args.dataset_id), None)
    if entry is None:
        print(f"error: {args.dataset_id!r} not found in {REGISTRY_PATH}", file=sys.stderr)
        return 1
    if entry.integrity_status != IntegrityStatus.VERIFIED:
        print(
            f"error: {args.dataset_id!r} has integrity_status={entry.integrity_status!r}, "
            "expected 'verified' -- run verify_scripts first",
            file=sys.stderr,
        )
        return 1

    data_root = resolve_data_root(args.data_root)
    dataset_config = load_dataset_config(CONFIGS_DIR / f"{args.dataset_id}.yaml")
    raw_path = raw_dir(data_root, args.dataset_id)
    output_path = staging_dir(data_root, args.dataset_id)

    convert_module = importlib.import_module(args.dataset_id)
    report = convert_module.convert(raw_path, output_path, dataset_config)

    episodes = load_lerobot_episodes(output_path)
    check_result = run_self_check(
        episodes,
        report,
        state_dim=dataset_config.state_dim,
        action_dim=dataset_config.action_dim,
        expected_num_episodes=dataset_config.expected_num_episodes,
        urdf_path=report.urdf_path,
        dof_per_arm=dataset_config.dof_per_arm,
    )

    _write_log(args.dataset_id, list(report.warnings) + check_result.reasons)

    if not check_result.passed:
        entry.convert_status = ConvertStatus.FAILED
        save_registry(entries, REGISTRY_PATH)
        print(f"convert failed for {args.dataset_id}: see {LOGS_DIR / f'{args.dataset_id}.log'}", file=sys.stderr)
        return 1

    entry.convert_status = ConvertStatus.CONVERTED
    entry.num_episodes = report.num_episodes
    entry.num_frames = report.num_frames
    entry.storage_size_gb = round(_dir_size_bytes(output_path) / 1e9, 2)
    save_registry(entries, REGISTRY_PATH)
    print(f"converted {args.dataset_id}: {report.num_episodes} episodes, {report.num_frames} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
