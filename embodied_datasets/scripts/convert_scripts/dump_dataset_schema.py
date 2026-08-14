"""Dump a read-only structural schema report for a raw robotics dataset.

Supports HDF5, RLDS/TFDS, and raw_image_json (image files + a JSON metadata
sidecar per episode, see readers/raw_image_json_reader.py) today; anything
else falls back to a directory tree and file inventory. This never reads
array/tensor payloads, decodes video, or touches ``.tfrecord`` binary
content -- only directory structure, file sizes, and per-format
header/sidecar metadata (shapes, dtypes, attrs, feature specs, JSON
structure). It exists so a new raw format can be inspected on the server
while bringing back only a small JSON report, not the dataset itself.

Usage::

    python3 dump_dataset_schema.py --dataset-root /data/public_datasets_raw/<uid>
    python3 dump_dataset_schema.py --raw-root /data/public_datasets_raw --all \
        --output-dir /tmp/schema_reports

RLDS/TFDS datasets are detected by the presence of a TFDS ``dataset_info.json``
sidecar. Decoding it into typed feature specs uses ``tensorflow_datasets``,
which is intentionally not a hard dependency of this repo (see
``requirements.txt``); install it only when you need to inspect an RLDS
dataset::

    pip install tensorflow-cpu==2.15.0 tensorflow-datasets==4.9.9

(pins matched to the sibling openpi repo). Without it, the raw
``dataset_info.json``/``features.json`` text is still included in the
report -- just undecoded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Sequence


HDF5_SUFFIXES = {".h5", ".hdf5"}
TFDS_INFO_FILENAME = "dataset_info.json"
TFDS_FEATURES_FILENAME = "features.json"
TFDS_SIDECAR_FILENAMES = {TFDS_INFO_FILENAME, TFDS_FEATURES_FILENAME}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
JSON_SIDECAR_SUFFIXES = {".json", ".jsonl"}
DEFAULT_MAX_TREE_DEPTH = 4
DEFAULT_MAX_TREE_ENTRIES = 50
DEFAULT_HDF5_SAMPLE_FILES = 3
TFDS_INSTALL_HINT = "tensorflow-cpu==2.15.0 tensorflow-datasets==4.9.9"


class SchemaDumpError(ValueError):
    """Raised when a dataset root cannot be scanned at all."""


def _natural_sort_key(path: Path) -> list[tuple[int, str | int]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.as_posix())
    ]


# ---------------------------------------------------------------------------
# Single-pass directory walk: file inventory (unbounded, for correct totals)
# + a bounded directory tree (for display) + candidate marker files for
# format detection. One os.walk instead of a separate pass per concern.
# ---------------------------------------------------------------------------


def _scan_root(root: Path, *, max_tree_depth: int, max_tree_entries: int) -> dict[str, Any]:
    if not root.is_dir():
        raise SchemaDumpError(f"dataset root does not exist or is not a directory: {root}")

    by_extension: dict[str, dict[str, int]] = {}
    total_files = 0
    total_bytes = 0
    hdf5_paths: list[Path] = []
    tfds_info_paths: list[Path] = []
    raw_image_json_dirs: list[tuple[Path, list[str]]] = []
    tree_root: dict[str, Any] = {"name": root.name or str(root), "type": "dir", "children": []}
    nodes_by_path: dict[Path, dict[str, Any]] = {root: tree_root}

    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        dirnames.sort(key=str.casefold)
        filenames.sort(key=str.casefold)
        depth = len(dirpath.relative_to(root).parts)
        node = nodes_by_path.get(dirpath)
        can_record = node is not None and depth < max_tree_depth

        if node is not None and not can_record and (dirnames or filenames):
            node["children"].append({"note": f"depth limit ({max_tree_depth}) reached; contents omitted"})

        recorded_dirs = 0
        for name in dirnames:
            child_path = dirpath / name
            if can_record and recorded_dirs < max_tree_entries:
                child_node: dict[str, Any] = {"name": name, "type": "dir", "children": []}
                node["children"].append(child_node)
                nodes_by_path[child_path] = child_node
                recorded_dirs += 1
        if can_record and len(dirnames) > recorded_dirs:
            node["children"].append({"note": f"... {len(dirnames) - recorded_dirs} more subdirectories omitted"})

        recorded_files = 0
        image_count = 0
        json_sidecar_names: list[str] = []
        for name in filenames:
            file_path = dirpath / name
            try:
                size = file_path.stat().st_size
            except OSError:
                size = 0
            suffix = file_path.suffix.casefold() or "<none>"
            bucket = by_extension.setdefault(suffix, {"count": 0, "bytes": 0})
            bucket["count"] += 1
            bucket["bytes"] += size
            total_files += 1
            total_bytes += size
            if suffix in HDF5_SUFFIXES:
                hdf5_paths.append(file_path)
            if name == TFDS_INFO_FILENAME:
                tfds_info_paths.append(file_path)
            if suffix in IMAGE_SUFFIXES:
                image_count += 1
            if suffix in JSON_SIDECAR_SUFFIXES and name not in TFDS_SIDECAR_FILENAMES:
                json_sidecar_names.append(name)

            if can_record and recorded_files < max_tree_entries:
                node["children"].append({"name": name, "type": "file", "bytes": size})
                recorded_files += 1
        if can_record and len(filenames) > recorded_files:
            node["children"].append({"note": f"... {len(filenames) - recorded_files} more files omitted"})
        if image_count > 0 and json_sidecar_names:
            raw_image_json_dirs.append((dirpath, sorted(json_sidecar_names, key=str.casefold)))

    hdf5_paths.sort(key=_natural_sort_key)
    tfds_info_paths.sort(key=_natural_sort_key)
    raw_image_json_dirs.sort(key=lambda item: _natural_sort_key(item[0]))

    return {
        "directory_tree": tree_root,
        "file_inventory": {
            "total_files": total_files,
            "total_bytes": total_bytes,
            "by_extension": dict(sorted(by_extension.items())),
        },
        "hdf5_paths": hdf5_paths,
        "tfds_info_paths": tfds_info_paths,
        "raw_image_json_dirs": raw_image_json_dirs,
    }


def detect_format(scan: dict[str, Any]) -> tuple[str, str]:
    if scan["tfds_info_paths"]:
        return "rlds", f"found {TFDS_INFO_FILENAME} at {scan['tfds_info_paths'][0]}"
    if scan["hdf5_paths"]:
        return "hdf5", f"found HDF5 file at {scan['hdf5_paths'][0]}"
    if scan["raw_image_json_dirs"]:
        first_dir, json_names = scan["raw_image_json_dirs"][0]
        return "raw_image_json", f"found image file(s) alongside JSON sidecar {json_names[0]!r} at {first_dir}"
    return "unknown", "no dataset_info.json, .h5/.hdf5 files, or image+JSON sidecar directories found under root"


# ---------------------------------------------------------------------------
# HDF5 inspector: header-only (shape/dtype/attrs), never dataset[:] values.
# ---------------------------------------------------------------------------


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("h5py is required; install the project's requirements.txt") from exc
    return h5py


def _serialize_attrs(attrs: Any) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, (bytes, bytearray)):
            serialized[key] = value.decode("utf-8", errors="replace")
        elif hasattr(value, "tolist"):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized


def _hdf5_header(path: Path) -> dict[str, Any]:
    h5py = _require_h5py()
    datasets: dict[str, dict[str, Any]] = {}
    groups: list[str] = []

    def _visit(name: str, obj: Any) -> None:
        key = "/" + name
        if isinstance(obj, h5py.Dataset):
            datasets[key] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "attrs": _serialize_attrs(obj.attrs),
            }
        elif isinstance(obj, h5py.Group):
            groups.append(key)

    with h5py.File(path, "r") as h5_file:
        root_attrs = _serialize_attrs(h5_file.attrs)
        h5_file.visititems(_visit)
    return {"groups": sorted(groups), "datasets": datasets, "root_attrs": root_attrs}


def _hdf5_signature(header: dict[str, Any]) -> dict[str, tuple[str, int, tuple[int, ...]]]:
    # Drop axis 0: it almost always encodes per-episode frame count, which
    # varies by design and is not a schema difference (matches the frame-count
    # vs. feature-width distinction already made in convert_mobile_aloha_to_lerobot.py).
    return {
        key: (value["dtype"], len(value["shape"]), tuple(value["shape"][1:]))
        for key, value in header["datasets"].items()
    }


def _compare_hdf5_headers(headers: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    if len(headers) <= 1:
        return True, []
    reference = _hdf5_signature(headers[0])
    diffs: list[str] = []
    for index, header in enumerate(headers[1:], start=1):
        signature = _hdf5_signature(header)
        if signature == reference:
            continue
        missing = sorted(set(reference) - set(signature))
        extra = sorted(set(signature) - set(reference))
        changed = sorted(key for key in set(reference) & set(signature) if reference[key] != signature[key])
        diffs.append(f"sample[{index}] differs from sample[0]: missing={missing} extra={extra} changed={changed}")
    return not diffs, diffs


def _dump_hdf5_schema(scan: dict[str, Any], root: Path, *, sample_files: int = DEFAULT_HDF5_SAMPLE_FILES) -> dict[str, Any]:
    hdf5_paths: list[Path] = scan["hdf5_paths"]
    if not hdf5_paths:
        return {"error": "no .h5/.hdf5 files found"}
    sample = hdf5_paths[:sample_files]

    per_file: dict[str, Any] = {}
    headers: list[dict[str, Any]] = []
    for path in sample:
        relative = str(path.relative_to(root))
        try:
            header = _hdf5_header(path)
        except OSError as exc:
            per_file[relative] = {"error": str(exc)}
            continue
        per_file[relative] = header
        headers.append(header)

    consistent, diff = _compare_hdf5_headers(headers)
    return {
        "sampled_files": [str(path.relative_to(root)) for path in sample],
        "total_hdf5_files": len(hdf5_paths),
        "per_file": per_file,
        "consistent_across_sample": consistent,
        "schema_diff": diff,
    }


# ---------------------------------------------------------------------------
# RLDS/TFDS inspector: sidecar JSON metadata only, never .tfrecord payloads.
# ---------------------------------------------------------------------------


def _require_tfds():
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        return None
    return tfds


def _describe_features(features: Any) -> Any:
    if features is None:
        return None
    for attr in ("to_json", "to_json_content"):
        method = getattr(features, attr, None)
        if not callable(method):
            continue
        try:
            result = method()
        except Exception:  # pragma: no cover - depends on installed tfds version
            continue
        try:
            return json.loads(result) if isinstance(result, str) else result
        except (TypeError, ValueError):
            continue
    return repr(features)


def _raw_sidecar_json(version_dir: Path, relative: str) -> dict[str, Any]:
    sidecars: dict[str, Any] = {}
    for filename in (TFDS_INFO_FILENAME, TFDS_FEATURES_FILENAME):
        path = version_dir / filename
        if not path.is_file():
            continue
        try:
            sidecars[filename] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            sidecars[filename] = {"error": str(exc)}
    return {
        "version_dir": relative,
        "fidelity": f"raw_sidecar_json (install {TFDS_INSTALL_HINT} for decoded feature shapes/dtypes)",
        "raw_sidecar_json": sidecars,
    }


def _tfds_builder_info(tfds: Any, version_dir: Path, relative: str) -> dict[str, Any]:
    try:
        builder = tfds.builder_from_directory(str(version_dir))
        info = builder.info
    except Exception as exc:  # pragma: no cover - depends on real tfds metadata
        fallback = _raw_sidecar_json(version_dir, relative)
        fallback["fidelity"] = f"tfds_builder_from_directory failed ({exc}); raw sidecar JSON only"
        return fallback

    result: dict[str, Any] = {
        "version_dir": relative,
        "fidelity": "tfds_builder_info",
        "name": getattr(info, "name", None),
        "version": str(getattr(info, "version", "")),
        "splits": {
            name: {
                "num_examples": getattr(split, "num_examples", None),
                "num_shards": getattr(split, "num_shards", None),
            }
            for name, split in info.splits.items()
        },
        "dataset_size_bytes": getattr(info, "dataset_size", None),
        "features": _describe_features(getattr(info, "features", None)),
    }
    try:
        result["dataset_info_as_json"] = json.loads(info.as_json)
    except Exception:  # pragma: no cover - depends on installed tfds version
        pass
    return result


def _dump_rlds_schema(scan: dict[str, Any], root: Path, *, sample_files: int = DEFAULT_HDF5_SAMPLE_FILES) -> dict[str, Any]:
    tfds_info_paths: list[Path] = scan["tfds_info_paths"]
    if not tfds_info_paths:
        return {"error": f"no {TFDS_INFO_FILENAME} found"}

    tfds = _require_tfds()
    datasets: list[dict[str, Any]] = []
    for info_path in tfds_info_paths:
        version_dir = info_path.parent
        relative = str(version_dir.relative_to(root))
        if tfds is not None:
            datasets.append(_tfds_builder_info(tfds, version_dir, relative))
        else:
            datasets.append(_raw_sidecar_json(version_dir, relative))
    return {"datasets": datasets}


def _dump_generic_schema(scan: dict[str, Any], root: Path, *, sample_files: int = DEFAULT_HDF5_SAMPLE_FILES) -> dict[str, Any]:
    return {
        "note": (
            "no structured inspector matched this dataset (no dataset_info.json, .h5/.hdf5 files, "
            "or image+JSON sidecar directories found); see file_inventory/directory_tree for a manual look"
        ),
    }


# ---------------------------------------------------------------------------
# raw_image_json inspector: one directory per episode with image files plus
# a JSON metadata sidecar (see readers/raw_image_json_reader.py). Only lists
# filenames and summarizes JSON *structure* (keys, list lengths) -- never
# decodes an image or dumps a JSON sidecar's full content.
# ---------------------------------------------------------------------------


def _summarize_json_structure(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        summary: dict[str, Any] = {"type": "object", "keys": sorted(data.keys())}
        frames = data.get("frames")
        if isinstance(frames, list):
            summary["frames_count"] = len(frames)
            if frames and isinstance(frames[0], dict):
                summary["frame_0_keys"] = sorted(frames[0].keys())
        return summary
    if isinstance(data, list):
        summary = {"type": "array", "length": len(data)}
        if data and isinstance(data[0], dict):
            summary["item_0_keys"] = sorted(data[0].keys())
        return summary
    return {"type": type(data).__name__}


def _dump_raw_image_json_schema(scan: dict[str, Any], root: Path, *, sample_files: int = DEFAULT_HDF5_SAMPLE_FILES) -> dict[str, Any]:
    candidates: list[tuple[Path, list[str]]] = scan["raw_image_json_dirs"]
    if not candidates:
        return {"error": "no directory with both image files and a JSON sidecar found"}

    sampled_episodes: list[dict[str, Any]] = []
    for episode_dir, json_names in candidates[:sample_files]:
        entry: dict[str, Any] = {
            "episode_dir": str(episode_dir.relative_to(root)),
            "json_sidecars": json_names,
        }
        try:
            image_count = sum(
                1
                for path in episode_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            )
        except OSError as exc:
            image_count = None
            entry["image_count_error"] = str(exc)
        entry["num_image_files"] = image_count

        sidecars: dict[str, Any] = {}
        for json_name in json_names:
            json_path = episode_dir / json_name
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                sidecars[json_name] = {"error": str(exc)}
                continue
            sidecars[json_name] = _summarize_json_structure(data)
        entry["sidecars"] = sidecars
        sampled_episodes.append(entry)

    return {
        "total_candidate_episode_dirs": len(candidates),
        "sampled_episodes": sampled_episodes,
    }


FORMAT_INSPECTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "hdf5": _dump_hdf5_schema,
    "rlds": _dump_rlds_schema,
    "raw_image_json": _dump_raw_image_json_schema,
}


def dump_schema(
    dataset_root: Path,
    *,
    format_override: str = "auto",
    max_tree_depth: int = DEFAULT_MAX_TREE_DEPTH,
    max_tree_entries: int = DEFAULT_MAX_TREE_ENTRIES,
    hdf5_sample_files: int = DEFAULT_HDF5_SAMPLE_FILES,
) -> dict[str, Any]:
    root = dataset_root.resolve()
    scan = _scan_root(root, max_tree_depth=max_tree_depth, max_tree_entries=max_tree_entries)

    if format_override != "auto":
        detected_format, evidence = format_override, "explicit --format override"
    else:
        detected_format, evidence = detect_format(scan)

    schema = FORMAT_INSPECTORS.get(detected_format, _dump_generic_schema)(
        scan, root, sample_files=hdf5_sample_files
    )

    return {
        "dataset_root": str(root),
        "detected_format": detected_format,
        "format_detection_evidence": evidence,
        "file_inventory": scan["file_inventory"],
        "directory_tree": scan["directory_tree"],
        "schema": schema,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset-root", type=Path, help="Inspect a single dataset directory.")
    target.add_argument("--raw-root", type=Path, help="Root containing one dataset-uid directory per subdirectory.")
    parser.add_argument("--all", action="store_true", help="With --raw-root, inspect every immediate subdirectory.")
    parser.add_argument("--output", type=Path, help="With --dataset-root, also write the report JSON here.")
    parser.add_argument("--output-dir", type=Path, help="With --raw-root --all, write one <uid>.json per dataset here.")
    parser.add_argument("--format", choices=("auto", "hdf5", "rlds", "raw_image_json"), default="auto")
    parser.add_argument("--max-tree-depth", type=int, default=DEFAULT_MAX_TREE_DEPTH)
    parser.add_argument("--max-tree-entries", type=int, default=DEFAULT_MAX_TREE_ENTRIES)
    parser.add_argument("--hdf5-sample-files", type=int, default=DEFAULT_HDF5_SAMPLE_FILES)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.raw_root is not None and not args.all:
        parser.error("--raw-root requires --all")
    if args.all and args.raw_root is None:
        parser.error("--all requires --raw-root")
    if args.all and args.output_dir is None:
        parser.error("--all requires --output-dir")
    if args.dataset_root is not None and args.output_dir is not None:
        parser.error("--output-dir is only used with --raw-root --all; use --output for a single dataset")


def _dump_one(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    return dump_schema(
        root,
        format_override=args.format,
        max_tree_depth=args.max_tree_depth,
        max_tree_entries=args.max_tree_entries,
        hdf5_sample_files=args.hdf5_sample_files,
    )


def _dump_single_dataset(args: argparse.Namespace) -> int:
    try:
        report = _dump_one(args.dataset_root, args)
    except (SchemaDumpError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


def _dump_all_datasets(args: argparse.Namespace) -> int:
    if not args.raw_root.is_dir():
        print(f"error: raw root does not exist: {args.raw_root}", file=sys.stderr)
        return 1
    uids = sorted((path.name for path in args.raw_root.iterdir() if path.is_dir()), key=str.casefold)
    if not uids:
        print(f"error: raw root contains no dataset directories: {args.raw_root}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for uid in uids:
        error: str | None = None
        try:
            report = _dump_one(args.raw_root / uid, args)
        except (SchemaDumpError, OSError, RuntimeError) as exc:
            error = str(exc)
            report = {"dataset_root": str((args.raw_root / uid).resolve()), "error": error}
        output_path = args.output_dir / f"{uid}.json"
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        summary_rows.append(
            {
                "dataset_uid": uid,
                "detected_format": report.get("detected_format", "error"),
                "total_files": report.get("file_inventory", {}).get("total_files"),
                "total_bytes": report.get("file_inventory", {}).get("total_bytes"),
                "error": error,
            }
        )

    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"wrote {len(summary_rows)} report(s) to {args.output_dir}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args.dataset_root is not None:
        return _dump_single_dataset(args)
    return _dump_all_datasets(args)


if __name__ == "__main__":
    sys.exit(main())
