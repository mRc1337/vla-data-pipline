"""Independently compare a DexMimicGen LeRobot collection with source HDF5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence
import zlib

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from convert_core.lerobot_writer import validate_video_files, validate_written_dataset
from readers.dexmimicgen_hdf5_reader import inspect_partition


DEFAULT_SOURCE = Path("/mnt/data/embodied_datasets/public_datasets_raw/dexmimicgen")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb(value: Any) -> np.ndarray:
    image = _numpy(value)
    if image.shape == (3, 84, 84):
        image = image.transpose(1, 2, 0)
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.rint(image)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape != (84, 84, 3):
        raise ValueError(f"unexpected decoded image shape {image.shape}")
    return image


def _sample_indices(length: int, count: int) -> list[int]:
    return sorted(set(np.linspace(0, length - 1, min(length, count), dtype=int).tolist()))


def _episode_at(episodes: Sequence[dict[str, Any]], index: int) -> tuple[dict[str, Any], int]:
    cursor = 0
    for episode in episodes:
        length = int(episode["num_frames"])
        if index < cursor + length:
            return episode, index - cursor
        cursor += length
    raise IndexError(index)


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def _source_location(source: str) -> tuple[str, str]:
    relative, separator, group = source.partition("::data/")
    if not separator or not group:
        raise ValueError(f"invalid DexMimicGen episode source {source!r}")
    return relative, group


def _json_normalized(value: Any) -> Any:
    """Normalize tuples and numpy scalars to their JSON representation."""

    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _partition_manifest_evidence(
    manifest: dict[str, Any], plan: Any
) -> dict[str, Any]:
    tasks = list(dict.fromkeys(episode.instruction for episode in plan.episodes))
    task_indices = {task: index for index, task in enumerate(tasks)}
    episode_keys = (
        "lerobot_episode_index",
        "lerobot_task_index",
        "episode_uid",
        "source",
        "source_task",
        "source_episode_id",
        "source_model_sha256",
        "source_model_uncompressed_bytes",
        "instruction",
        "num_frames",
        "source_splits",
        "source_split",
        "source_segment_id",
        "source_spans",
    )
    expected_episodes = []
    for index, episode in enumerate(plan.episodes):
        expected_episodes.append(
            {
                "lerobot_episode_index": index,
                "lerobot_task_index": task_indices[episode.instruction],
                "episode_uid": episode.episode_uid,
                "source": episode.source_relative_path,
                "source_task": episode.extra.get("source_task"),
                "source_episode_id": episode.extra.get("source_episode_id"),
                "source_model_sha256": episode.extra.get("source_model_sha256"),
                "source_model_uncompressed_bytes": episode.extra.get(
                    "source_model_uncompressed_bytes"
                ),
                "instruction": episode.instruction,
                "num_frames": episode.num_frames,
                "source_splits": list(episode.extra.get("source_splits", ())),
                "source_split": episode.extra.get("source_split"),
                "source_segment_id": episode.extra.get("source_segment_id"),
                "source_spans": list(episode.extra.get("source_spans", ())),
            }
        )
    actual_episodes = [
        {key: row.get(key) for key in episode_keys}
        for row in manifest.get("episodes", [])
        if isinstance(row, dict)
    ]
    features = _json_normalized(plan.feature_schema())
    field_mapping = _json_normalized(plan.extra.get("field_mapping", []))
    mapped_features = [
        row.get("lerobot_key")
        for row in manifest.get("field_mapping", [])
        if isinstance(row, dict)
    ]
    checkpoint_units = [
        row.get("checkpoint_unit")
        for row in manifest.get("episodes", [])
        if isinstance(row, dict)
    ]
    closed_units: set[str] = set()
    previous_unit: str | None = None
    checkpoint_units_valid = True
    for unit in checkpoint_units:
        if not isinstance(unit, str) or not unit:
            checkpoint_units_valid = False
            break
        if unit != previous_unit:
            if unit in closed_units:
                checkpoint_units_valid = False
                break
            if previous_unit is not None:
                closed_units.add(previous_unit)
            previous_unit = unit
    checks = {
        "dataset_uid": manifest.get("dataset_uid") == plan.dataset_uid,
        "robot_type": manifest.get("robot_type") == plan.robot_type,
        "fps": manifest.get("fps") == plan.fps,
        "measured_fps": manifest.get("measured_fps") == plan.measured_fps,
        "num_episodes": manifest.get("num_episodes") == len(plan.episodes),
        "num_frames": manifest.get("num_frames") == plan.num_frames,
        "num_video_features": manifest.get("num_video_features")
        == len(plan.camera_features),
        "features": manifest.get("features") == features,
        "field_mapping": manifest.get("field_mapping") == field_mapping,
        "field_mapping_coverage": (
            len(mapped_features) == len(features)
            and len(set(mapped_features)) == len(features)
            and set(mapped_features) == set(features)
        ),
        "episodes": actual_episodes == expected_episodes,
        "checkpoint_units": (
            len(checkpoint_units) == len(plan.episodes) and checkpoint_units_valid
        ),
        "task_index_mapping": manifest.get("task_index_mapping")
        == {str(index): task for index, task in enumerate(tasks)},
        "source_dataset": manifest.get("source_dataset")
        == plan.extra.get("source_dataset"),
        "source_revision": manifest.get("source_revision")
        == plan.extra.get("source_revision"),
        "source_relative_path": manifest.get("source_relative_path")
        == plan.extra.get("source_relative_path"),
        "source_env_name": manifest.get("source_env_name")
        == plan.extra.get("source_env_name"),
    }
    return {
        "checks": checks,
        "expected_feature_count": len(features),
        "mapped_feature_count": len(mapped_features),
        "passed": all(checks.values()),
    }


def evaluate_partition(
    partition_root: Path,
    source_root: Path,
    *,
    samples: int,
    min_psnr_db: float,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = _read_json(partition_root / "conversion_manifest.json")
    source_path = source_root / manifest["source_relative_path"]
    info = inspect_partition(
        source_path,
        raw_dataset_root=source_root,
        collection_output=partition_root.parent,
        max_episodes=int(manifest["num_episodes"]),
    )
    plan = replace_output(info.plan, partition_root)
    manifest_evidence = _partition_manifest_evidence(manifest, plan)
    validate_written_dataset(plan, partition_root)
    video_validation = validate_video_files(
        plan, partition_root, expected_frames=int(manifest["num_frames"])
    )
    dataset = LeRobotDataset(
        repo_id=manifest["dataset_uid"], root=partition_root, video_backend="pyav"
    )
    numeric_results: dict[str, dict[str, Any]] = {}
    image_psnr: dict[str, list[float]] = {
        key: []
        for key, feature in manifest["features"].items()
        if feature["dtype"] == "video"
    }
    sampled = _sample_indices(len(dataset), samples)
    parquet_paths = sorted((partition_root / "data").rglob("*.parquet"))
    stored_samples = pads.dataset(
        [str(path) for path in parquet_paths], format="parquet"
    ).take(pa.array(sampled, type=pa.int64()))
    index_checks = {
        "index": True,
        "frame_index": True,
        "episode_index": True,
        "task_index": True,
    }
    with h5py.File(source_path, "r") as source_file:
        for sample_position, global_index in enumerate(sampled):
            episode, local_index = _episode_at(manifest["episodes"], global_index)
            _relative, demo_name = _source_location(episode["source"])
            demo = source_file[f"data/{demo_name}"]
            output = dataset[global_index]
            index_checks["index"] &= int(stored_samples["index"][sample_position].as_py()) == global_index
            index_checks["frame_index"] &= (
                int(stored_samples["frame_index"][sample_position].as_py()) == local_index
            )
            index_checks["episode_index"] &= (
                int(stored_samples["episode_index"][sample_position].as_py())
                == int(episode["lerobot_episode_index"])
            )
            index_checks["task_index"] &= (
                int(stored_samples["task_index"][sample_position].as_py())
                == int(episode["lerobot_task_index"])
            )
            for mapping in manifest["field_mapping"]:
                source_key = mapping["source_key"].removeprefix("data/<demo>/")
                feature_key = mapping["lerobot_key"]
                source_value = np.asarray(demo[source_key][local_index])
                if mapping["lossy"]:
                    image_psnr[feature_key].append(
                        _psnr(source_value, _rgb(output[feature_key]))
                    )
                    continue
                if source_value.ndim == 0:
                    source_value = source_value.reshape(1)
                feature = manifest["features"][feature_key]
                converted = np.asarray(
                    stored_samples[feature_key][sample_position].as_py(),
                    dtype=np.dtype(feature["dtype"]),
                ).reshape(tuple(feature["shape"]))
                exact = bool(
                    converted.shape == source_value.shape
                    and converted.dtype == source_value.dtype
                    and np.array_equal(converted, source_value)
                )
                error = None
                if converted.shape == source_value.shape:
                    error = float(
                        np.max(
                            np.abs(
                                converted.astype(np.float64)
                                - source_value.astype(np.float64)
                            ),
                            initial=0.0,
                        )
                    )
                row = numeric_results.setdefault(
                    feature_key,
                    {
                        "exact": True,
                        "samples": 0,
                        "max_abs_error": 0.0,
                        "source_dtype": str(source_value.dtype),
                        "output_dtype": str(converted.dtype),
                        "parquet_type": str(stored_samples.schema.field(feature_key).type),
                    },
                )
                row["exact"] = bool(row["exact"] and exact)
                row["samples"] += 1
                if error is not None:
                    row["max_abs_error"] = max(float(row["max_abs_error"]), error)
    quality = {
        key: {
            "samples": len(values),
            "minimum_psnr_db": min(values),
            "required_psnr_db": min_psnr_db,
            "pass": bool(values and min(values) >= min_psnr_db),
        }
        for key, values in image_psnr.items()
    }
    numeric_pass = bool(numeric_results) and all(
        row["exact"] for row in numeric_results.values()
    )
    images_pass = bool(quality) and all(row["pass"] for row in quality.values())
    indices_pass = all(index_checks.values())
    expected_numeric = {
        key for key, feature in manifest["features"].items() if feature["dtype"] != "video"
    }
    expected_images = {
        key for key, feature in manifest["features"].items() if feature["dtype"] == "video"
    }
    numeric_coverage = {
        "expected": len(expected_numeric),
        "evaluated": len(numeric_results),
        "missing": sorted(expected_numeric - set(numeric_results)),
        "unexpected": sorted(set(numeric_results) - expected_numeric),
    }
    numeric_coverage["passed"] = not (
        numeric_coverage["missing"] or numeric_coverage["unexpected"]
    )
    image_coverage = {
        "expected": len(expected_images),
        "evaluated": len(quality),
        "missing": sorted(expected_images - set(quality)),
        "unexpected": sorted(set(quality) - expected_images),
    }
    image_coverage["passed"] = not (
        image_coverage["missing"] or image_coverage["unexpected"]
    )
    passed = bool(
        manifest_evidence["passed"]
        and numeric_pass
        and numeric_coverage["passed"]
        and images_pass
        and image_coverage["passed"]
        and indices_pass
    )
    return {
        "partition": partition_root.name,
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "sampled_frame_indices": sampled,
        "numeric": numeric_results,
        "numeric_exact": numeric_pass,
        "numeric_coverage": numeric_coverage,
        "indices": index_checks,
        "indices_exact": indices_pass,
        "image_quality": quality,
        "image_coverage": image_coverage,
        "videos": video_validation,
        "manifest": manifest_evidence,
        "passed": passed,
    }


def replace_output(plan: Any, output: Path) -> Any:
    from dataclasses import replace

    return replace(plan, output_path=output)


def _validate_sidecars(collection_root: Path, collection: dict[str, Any]) -> dict[str, Any]:
    rows = []
    passed = True
    for item in collection["model_sidecars"]:
        path = collection_root / item["relative_path"]
        try:
            raw = zlib.decompress(path.read_bytes())
            actual = hashlib.sha256(raw).hexdigest()
            valid = (
                actual == item["sha256"]
                and len(raw) == int(item["uncompressed_bytes"])
                and path.stat().st_size == int(item["compressed_bytes"])
            )
        except (OSError, zlib.error):
            actual = None
            valid = False
        passed &= valid
        rows.append({"relative_path": item["relative_path"], "sha256": actual, "valid": valid})
    referenced = {
        episode["source_model_sha256"] for episode in collection["episodes"]
    }
    declared = {item["sha256"] for item in collection["model_sidecars"]}
    coverage = referenced == declared
    return {"files": rows, "reference_coverage": coverage, "passed": passed and coverage}


def _validate_publication(
    collection_root: Path, collection_path: Path
) -> dict[str, Any]:
    marker = _read_json(collection_root / "_SUCCESS")
    actual_sha256 = hashlib.sha256(collection_path.read_bytes()).hexdigest()
    checks = {
        "status": marker.get("status") == "success",
        "manifest_sha256": marker.get("collection_manifest_sha256")
        == actual_sha256,
        "incomplete_absent": not (collection_root / "_INCOMPLETE").exists(),
    }
    return {
        "checks": checks,
        "collection_manifest_sha256": actual_sha256,
        "passed": all(checks.values()),
    }


def _validate_collection_manifest(
    collection_root: Path, collection: dict[str, Any]
) -> dict[str, Any]:
    partition_rows = collection.get("partitions", [])
    partition_manifests: dict[str, dict[str, Any]] = {}
    partition_checks = []
    for row in partition_rows:
        name = str(row.get("name"))
        manifest = _read_json(collection_root / name / "conversion_manifest.json")
        partition_manifests[name] = manifest
        checks = {
            "dataset_uid": row.get("dataset_uid") == manifest.get("dataset_uid"),
            "episodes": row.get("episodes") == manifest.get("num_episodes"),
            "frames": row.get("frames") == manifest.get("num_frames"),
            "features": row.get("features") == manifest.get("features"),
            "source_relative_path": row.get("source_relative_path")
            == manifest.get("source_relative_path"),
        }
        partition_checks.append(
            {"partition": name, "checks": checks, "passed": all(checks.values())}
        )

    episode_keys = (
        "collection_episode_index",
        "partition",
        "lerobot_episode_index",
        "lerobot_task_index",
        "source_episode_id",
        "source",
        "source_task",
        "instruction",
        "source_model_sha256",
        "num_frames",
    )
    expected_episodes = []
    global_index = 0
    for row in partition_rows:
        name = str(row.get("name"))
        for episode in partition_manifests[name]["episodes"]:
            expected_episodes.append(
                {
                    "collection_episode_index": global_index,
                    "partition": name,
                    "lerobot_episode_index": episode["lerobot_episode_index"],
                    "lerobot_task_index": episode["lerobot_task_index"],
                    "source_episode_id": episode["source_episode_id"],
                    "source": episode["source"],
                    "source_task": episode["source_task"],
                    "instruction": episode["instruction"],
                    "source_model_sha256": episode["source_model_sha256"],
                    "num_frames": episode["num_frames"],
                }
            )
            global_index += 1
    actual_episodes = [
        {key: row.get(key) for key in episode_keys}
        for row in collection.get("episodes", [])
        if isinstance(row, dict)
    ]
    expected_tasks = {
        name: {
            **manifest["task_index_mapping"],
            "source_env_name": manifest["source_env_name"],
        }
        for name, manifest in partition_manifests.items()
    }
    discovered = sorted(
        path.parent.name
        for path in collection_root.glob("*/conversion_manifest.json")
    )
    declared = [str(row.get("name")) for row in partition_rows]
    checks = {
        "format": collection.get("format") == "lerobot_v3_0_collection",
        "dataset_uid": collection.get("dataset_uid") == collection_root.name,
        "partition_names_unique": len(declared) == len(set(declared)),
        "partition_directories": sorted(declared) == discovered,
        "partition_records": all(row["passed"] for row in partition_checks),
        "episodes": actual_episodes == expected_episodes,
        "task_index_mapping": collection.get("task_index_mapping") == expected_tasks,
        "source_revision": all(
            manifest.get("source_revision") == collection.get("source_revision")
            for manifest in partition_manifests.values()
        ),
    }
    return {
        "checks": checks,
        "partitions": partition_checks,
        "episode_count": len(expected_episodes),
        "passed": all(checks.values()),
    }


def evaluate(
    collection_root: Path,
    source_root: Path,
    *,
    samples: int,
    min_psnr_db: float,
) -> dict[str, Any]:
    collection_path = collection_root / "collection_manifest.json"
    collection = _read_json(collection_path)
    partitions = [
        evaluate_partition(
            collection_root / item["name"],
            source_root,
            samples=samples,
            min_psnr_db=min_psnr_db,
        )
        for item in collection["partitions"]
    ]
    sidecars = _validate_sidecars(collection_root, collection)
    publication = _validate_publication(collection_root, collection_path)
    collection_manifest = _validate_collection_manifest(collection_root, collection)
    return {
        "collection_root": str(collection_root.resolve()),
        "source_root": str(source_root.resolve()),
        "source_revision": collection["source_revision"],
        "partitions": partitions,
        "model_sidecars": sidecars,
        "publication": publication,
        "collection_manifest": collection_manifest,
        "passed": bool(
            sidecars["passed"]
            and publication["passed"]
            and collection_manifest["passed"]
            and all(row["passed"] for row in partitions)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--min-psnr-db", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    report = evaluate(
        args.collection_root,
        args.source_root,
        samples=args.samples,
        min_psnr_db=args.min_psnr_db,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
