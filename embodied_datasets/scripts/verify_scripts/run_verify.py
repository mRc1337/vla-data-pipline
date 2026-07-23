"""Orchestrates download-integrity verification: for every dataset with
download_status=completed and integrity_status!=verified (or a single
`--dataset-id`), dispatches to a format-specific checker by raw_format,
runs the size/episode-count scale check and a video-sample-decode check,
and writes integrity_status back to the registry. See
docs/superpowers/specs/2026-07-21-convert-scripts-verify-scripts-design.md
section 6.

Loads convert_scripts/common (RegistryEntry/DatasetConfig/paths/etc.)
under the alias "registry_common" via importlib.util instead of sys.path,
because verify_scripts/common (this package's own format_checkers.py) and
convert_scripts/common are both literally named "common" -- putting both
directories on sys.path would make whichever imports first win for every
subsequent `import common` in the process. Mirrors
process_scripts/run_pipeline.py's identical _load_registry_common()
helper.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Tuple

# NOTE: this puts verify_scripts/ on sys.path so bare `import common` below
# resolves to verify_scripts/common/ (format_checkers.py's package).
# convert_scripts/run_convert.py does the analogous thing for its own
# (differently-shaped) "common" package. Both work correctly in isolation,
# but if a future script ever imports both run_convert and run_verify into
# the same process, whichever's `import common` runs first wins
# (sys.modules caching) and the other's `from common.X import ...` would
# silently resolve against the wrong package. No current caller does this --
# just don't be the one who does it silently.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # verify_scripts/: common (format_checkers)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: shared

from common.format_checkers import CheckOutcome, check_format, check_scale, check_video_decodable, find_video_files  # noqa: E402

CONVERT_SCRIPTS_COMMON_DIR = Path(__file__).resolve().parents[1] / "convert_scripts" / "common"
LOGS_DIR = Path(__file__).resolve().parent / "logs"
VIDEO_SAMPLE_SIZE = 3


def _load_registry_common() -> ModuleType:
    alias = "registry_common"
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            alias, CONVERT_SCRIPTS_COMMON_DIR / "__init__.py", submodule_search_locations=[str(CONVERT_SCRIPTS_COMMON_DIR)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    module = sys.modules[alias]
    module.schema = importlib.import_module(f"{alias}.schema")
    module.io = importlib.import_module(f"{alias}.io")
    module.paths = importlib.import_module(f"{alias}.paths")
    return module


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _write_log(dataset_id: str, lines: List[str]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"{dataset_id}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_one(dataset_id: str, raw_path: Path, dataset_config) -> Tuple[str, List[str]]:
    log_lines = []
    format_result = check_format(raw_path, dataset_config.raw_format, dataset_id)
    log_lines.append(
        f"format check ({dataset_config.raw_format}): {format_result.outcome.value}"
        + (f" -- {format_result.reason}" if format_result.reason else "")
    )

    if format_result.outcome == CheckOutcome.NO_CHECKER:
        return "skipped_no_checker", log_lines
    if format_result.outcome == CheckOutcome.FAILED:
        return "failed", log_lines

    actual_size_gb = _dir_size_bytes(raw_path) / 1e9
    scale_result = check_scale(
        actual_size_gb, format_result.episode_count, dataset_config.expected_size_gb, dataset_config.expected_num_episodes
    )
    log_lines.append(
        f"scale check: {scale_result.outcome.value}" + (f" -- {scale_result.reason}" if scale_result.reason else "")
    )
    if scale_result.outcome == CheckOutcome.FAILED:
        return "failed", log_lines

    video_paths = find_video_files(raw_path)[:VIDEO_SAMPLE_SIZE]
    if video_paths:
        decode_result = check_video_decodable(video_paths)
        log_lines.append(
            f"video sample decode check ({len(video_paths)} sampled): {decode_result.outcome.value}"
            + (f" -- {decode_result.reason}" if decode_result.reason else "")
        )
        if decode_result.outcome == CheckOutcome.FAILED:
            return "failed", log_lines
        # NO_CHECKER here (e.g. ffprobe not installed on this machine) is
        # logged but doesn't block verification -- unlike the raw-format
        # check, a missing video-probe tool is an environment limitation,
        # not evidence the dataset itself might be silently broken.

    return "verified", log_lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify download integrity for datasets pending integrity_status=verified.")
    parser.add_argument("--dataset-id", default=None, help="Verify only this dataset; default verifies every eligible dataset.")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)

    registry_common = _load_registry_common()
    data_root = registry_common.paths.resolve_data_root(args.data_root)
    registry_path = Path(__file__).resolve().parents[2] / "datasets_registry.yaml"
    configs_dir = Path(__file__).resolve().parents[1] / "convert_scripts" / "configs"

    entries = registry_common.io.load_registry(registry_path)
    eligible = [
        e
        for e in entries
        if e.download_status == registry_common.schema.DownloadStatus.COMPLETED
        and e.integrity_status != registry_common.schema.IntegrityStatus.VERIFIED
        and (args.dataset_id is None or e.id == args.dataset_id)
    ]
    if args.dataset_id and not eligible:
        print(
            f"error: {args.dataset_id!r} not found, or not eligible "
            "(download_status != completed, or already verified)",
            file=sys.stderr,
        )
        return 1

    exit_code = 0
    for entry in eligible:
        dataset_config = registry_common.io.load_dataset_config(configs_dir / f"{entry.id}.yaml")
        raw_path = registry_common.paths.raw_dir(data_root, entry.id)
        status, log_lines = _verify_one(entry.id, raw_path, dataset_config)
        _write_log(entry.id, log_lines)
        entry.integrity_status = registry_common.schema.IntegrityStatus(status)
        if status != "verified":
            exit_code = 1
            print(f"{entry.id}: {status} -- see {LOGS_DIR / f'{entry.id}.log'}", file=sys.stderr)
        else:
            print(f"{entry.id}: verified")

    registry_common.io.save_registry(entries, registry_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
