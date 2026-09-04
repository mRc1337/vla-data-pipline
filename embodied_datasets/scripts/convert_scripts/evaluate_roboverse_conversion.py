#!/usr/bin/env python3
"""Independently audit a converted RoboVerse collection against raw v2 files."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import pickle
from typing import Any
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_core.errors import ConversionError


GENERATED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"expected a JSON object: {path}")
    return value


def _load_source(path: Path) -> Any:
    try:
        if path.name.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as handle:
                return pickle.load(handle)
        if path.suffix == ".pkl":
            with path.open("rb") as handle:
                return pickle.load(handle)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConversionError(f"cannot independently load source {path}: {exc}") from exc
    raise ConversionError(f"unsupported RoboVerse source file: {path}")


def _raw_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _numpy(value: Any) -> np.ndarray:
    array = _raw_numpy(value)
    return array.reshape(1) if array.ndim == 0 else array


def _lossless_cast(array: np.ndarray, target_dtype: str, description: str) -> np.ndarray:
    cast = array.astype(np.dtype(target_dtype), casting="unsafe", copy=False)
    restored = cast.astype(array.dtype, casting="unsafe", copy=False)
    if not np.array_equal(restored, array):
        raise ConversionError(
            f"{description} is not exactly representable as {target_dtype}"
        )
    return cast


def _mapped_source_value(container: Any, mapping: dict[str, Any]) -> np.ndarray:
    value = container
    for component in mapping["source_path"]:
        if not isinstance(value, dict) or component not in value:
            raise ConversionError(
                f"source value for {mapping['target_key']} lacks component {component!r}"
            )
        value = value[component]
    components = mapping.get("source_components")
    dtype_options = mapping.get("source_dtype_options")
    if components is not None:
        if not isinstance(value, dict):
            raise ConversionError(f"source value for {mapping['target_key']} is not a mapping")
        items = []
        component_shapes = mapping.get("source_component_shapes")
        for index, component in enumerate(components):
            item = _raw_numpy(value[component])
            if item.size != 1:
                raise ConversionError(
                    f"source component {component!r} for {mapping['target_key']} is not scalar"
                )
            if component_shapes is not None and list(item.shape) != component_shapes[index]:
                raise ConversionError(
                    f"source component {component!r} for {mapping['target_key']} has shape "
                    f"{item.shape}, expected {tuple(component_shapes[index])}"
                )
            if dtype_options is not None:
                if str(item.dtype) not in dtype_options[index]:
                    raise ConversionError(
                        f"source component {component!r} for {mapping['target_key']} has "
                        f"unrecorded dtype {item.dtype}"
                    )
                if str(item.dtype) != mapping["dtype"]:
                    item = _lossless_cast(
                        item,
                        mapping["dtype"],
                        f"source component {component!r} for {mapping['target_key']}",
                    )
            items.append(item)
        value = np.stack(items, axis=0)
    array = _numpy(value)
    if components is None and dtype_options is not None:
        if len(dtype_options) != 1 or str(array.dtype) not in dtype_options[0]:
            raise ConversionError(
                f"source {mapping['target_key']} has unrecorded dtype {array.dtype}"
            )
        if str(array.dtype) != mapping["dtype"]:
            array = _lossless_cast(array, mapping["dtype"], mapping["target_key"])
    expected_shape = tuple(mapping["shape"])
    if tuple(array.shape) != expected_shape or str(array.dtype) != mapping["dtype"]:
        raise ConversionError(
            f"source {mapping['target_key']} is {array.dtype}{array.shape}, "
            f"expected {mapping['dtype']}{expected_shape}"
        )
    return array


def _promotion_source_array(container: Any, promotion: dict[str, Any]) -> np.ndarray:
    value = container
    for component in promotion["source_path"]:
        if not isinstance(value, dict) or component not in value:
            raise ConversionError(
                f"dtype-promotion provenance lacks source component {component!r}"
            )
        value = value[component]
    source_component = promotion.get("source_component")
    if source_component is not None:
        if not isinstance(value, dict) or source_component not in value:
            raise ConversionError(
                f"dtype-promotion provenance lacks named component {source_component!r}"
            )
        value = value[source_component]
    array = _raw_numpy(value)
    if source_component is None and array.ndim == 0:
        array = array.reshape(1)
    return array


def _verify_dtype_promotion_frame(
    raw_episode: dict[str, Any],
    promotion: dict[str, Any],
    frame_index: int,
) -> None:
    stream = promotion["stream"]
    container = raw_episode["actions" if stream == "action" else "states"][frame_index]
    array = _promotion_source_array(container, promotion)
    runs = [
        run
        for run in promotion["runs"]
        if int(run["start"]) <= frame_index < int(run["end_exclusive"])
    ]
    if len(runs) != 1:
        raise ConversionError(
            f"dtype-promotion provenance has no unique run at frame {frame_index}"
        )
    if str(array.dtype) != runs[0]["dtype"] or list(array.shape) != promotion["source_shape"]:
        raise ConversionError(
            f"dtype-promotion provenance differs from raw frame {frame_index}: "
            f"{array.dtype}{array.shape}"
        )
    if str(array.dtype) not in promotion["source_dtypes"]:
        raise ConversionError("dtype-promotion run uses an unrecorded source dtype")
    if str(array.dtype) != promotion["target_dtype"]:
        _lossless_cast(array, promotion["target_dtype"], "dtype-promotion source value")


def _source_container(raw_episode: dict[str, Any], stream_kind: str, key: str, frame: int) -> Any:
    if stream_kind == "state":
        return raw_episode["states"][frame]
    if stream_kind == "action":
        return raw_episode["actions"][frame]
    return raw_episode["actions"][frame] if key.startswith("action") else raw_episode["states"][frame]


def _parquet_table(part_root: Path) -> pa.Table:
    paths = sorted((part_root / "data").rglob("*.parquet"))
    if not paths:
        raise ConversionError(f"no Parquet data in {part_root}")
    tables = [pq.read_table(path) for path in paths]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def _primitive_arrow_type(value: pa.DataType) -> pa.DataType:
    if isinstance(value, pa.ExtensionType):
        value = value.storage_type
    while pa.types.is_list(value) or pa.types.is_large_list(value) or pa.types.is_fixed_size_list(value):
        value = value.value_type
    return value


def _expected_arrow_type(dtype: str) -> pa.DataType:
    mapping = {
        "bool": pa.bool_(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "float16": pa.float16(),
        "float32": pa.float32(),
        "float64": pa.float64(),
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ConversionError(f"unsupported audit dtype {dtype!r}") from exc


def _json_scalar(value: Any) -> int | float | bool:
    return np.asarray(value).item()


def _numeric_statistics(values: list[np.ndarray]) -> dict[str, Any]:
    matrix = np.stack([value.reshape(-1) for value in values])
    finite_min: list[int | float | bool | None] = []
    finite_max: list[int | float | bool | None] = []
    nan_count: list[int] = []
    positive_infinity_count: list[int] = []
    negative_infinity_count: list[int] = []
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        if np.issubdtype(column.dtype, np.inexact):
            nan = np.isnan(column)
            positive = np.isposinf(column)
            negative = np.isneginf(column)
            finite = np.isfinite(column)
        else:
            nan = positive = negative = np.zeros(column.shape, dtype=bool)
            finite = np.ones(column.shape, dtype=bool)
        finite_values = column[finite]
        finite_min.append(_json_scalar(finite_values.min()) if finite_values.size else None)
        finite_max.append(_json_scalar(finite_values.max()) if finite_values.size else None)
        nan_count.append(int(nan.sum()))
        positive_infinity_count.append(int(positive.sum()))
        negative_infinity_count.append(int(negative.sum()))
    return {
        "frames": int(matrix.shape[0]),
        "finite_min": finite_min,
        "finite_max": finite_max,
        "nan_count": nan_count,
        "positive_infinity_count": positive_infinity_count,
        "negative_infinity_count": negative_infinity_count,
    }


def _assert_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ConversionError(f"{description}: {actual!r} != {expected!r}")


def _evaluate_part(
    collection_root: Path,
    raw_root: Path,
    collection: dict[str, Any],
    part_entry: dict[str, Any],
    source_cache: dict[str, Any],
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    part_root = collection_root / part_entry["path"]
    manifest = _read_json(part_root / "conversion_manifest.json")
    info = _read_json(part_root / "meta" / "info.json")
    written_stats = _read_json(part_root / "meta" / "stats.json")
    dataset_uid = f"{collection['dataset_uid']}/{part_entry['part_id']}"
    dataset = LeRobotDataset(repo_id=dataset_uid, root=part_root)
    table = _parquet_table(part_root)

    expected_frames = int(manifest["num_frames"])
    expected_episodes = int(manifest["num_episodes"])
    _assert_equal(len(dataset), expected_frames, f"{part_entry['part_id']} LeRobot frame count")
    _assert_equal(dataset.num_episodes, expected_episodes, f"{part_entry['part_id']} episode count")
    _assert_equal(table.num_rows, expected_frames, f"{part_entry['part_id']} Parquet row count")
    _assert_equal(info["codebase_version"], "v3.0", "LeRobot codebase version")
    _assert_equal(info["total_frames"], expected_frames, "info.json total_frames")
    _assert_equal(info["total_episodes"], expected_episodes, "info.json total_episodes")
    _assert_equal(info["robot_type"], manifest["robot_type"], "robot_type")
    _assert_equal(info["fps"], manifest["fps"], "fps")

    actual_task_mapping = {
        str(int(row["task_index"])): str(task) for task, row in dataset.meta.tasks.iterrows()
    }
    _assert_equal(actual_task_mapping, manifest["task_index_mapping"], "task_index mapping")

    source_values: dict[str, list[np.ndarray]] = {
        mapping["target_key"]: [] for mapping in manifest["field_mapping"]
    }
    sample_rows: list[dict[str, Any]] = []
    all_lerobot_values_exact = True
    cursor = 0
    for episode in sorted(manifest["episodes"], key=lambda row: int(row["target_episode_index"])):
        relative_path = episode["source_relative_path"]
        if relative_path not in source_cache:
            # Keep only one raw container resident. A full collection can be
            # hundreds of gigabytes, while adjacent action/state parts often
            # share the same source file and still benefit from this cache.
            source_cache.clear()
            source_cache[relative_path] = _load_source(raw_root / relative_path)
        raw_episode = source_cache[relative_path][episode["source_robot"]][
            int(episode["source_episode_index"])
        ]
        episode_length = int(episode["target_length"])
        sample_local_indices = sorted({0, episode_length // 2, episode_length - 1})
        for local_index in range(episode_length):
            for promotion in episode.get("lossless_dtype_promotions", []):
                _verify_dtype_promotion_frame(raw_episode, promotion, local_index)
            for mapping in manifest["field_mapping"]:
                container = _source_container(
                    raw_episode, manifest["stream_kind"], mapping["target_key"], local_index
                )
                source_values[mapping["target_key"]].append(
                    _mapped_source_value(container, mapping)
                )
        for local_index in sample_local_indices:
            global_index = cursor + local_index
            output = dataset[global_index]
            feature_rows: dict[str, Any] = {}
            for mapping in manifest["field_mapping"]:
                key = mapping["target_key"]
                source = source_values[key][global_index]
                parquet_value = table[key][global_index].as_py()
                parquet_physical_shape = list(np.asarray(parquet_value).shape)
                lerobot_tensor_shape = list(output[key].shape)
                parquet = _numpy(parquet_value)
                lerobot = _numpy(output[key])
                parquet_exact = bool(np.array_equal(source.reshape(-1), parquet.reshape(-1)))
                lerobot_exact = bool(np.array_equal(source.reshape(-1), lerobot.reshape(-1)))
                lerobot_matches_float32_materialization = bool(
                    source.dtype == np.dtype("float64")
                    and str(output[key].dtype) == "torch.float32"
                    and np.array_equal(
                        source.astype(np.float32).reshape(-1),
                        lerobot.reshape(-1),
                    )
                )
                if not parquet_exact or (
                    not lerobot_exact and not lerobot_matches_float32_materialization
                ):
                    raise ConversionError(
                        f"{part_entry['part_id']} frame {global_index} feature {key} differs from source"
                    )
                all_lerobot_values_exact = all_lerobot_values_exact and lerobot_exact
                feature_rows[key] = {
                    "source_dtype": str(source.dtype),
                    "parquet_dtype": str(_primitive_arrow_type(table.schema.field(key).type)),
                    "lerobot_tensor_dtype": str(output[key].dtype),
                    "declared_shape": list(mapping["shape"]),
                    "source_shape": list(source.shape),
                    "parquet_physical_shape": parquet_physical_shape,
                    "lerobot_tensor_shape": lerobot_tensor_shape,
                    "parquet_exact": True,
                    "lerobot_value_exact": lerobot_exact,
                    "lerobot_matches_float32_materialization": (
                        lerobot_matches_float32_materialization
                    ),
                }
            expected_timestamp = local_index / float(manifest["fps"])
            timestamp = float(_numpy(output["timestamp"]).item())
            if not math.isclose(timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1e-6):
                raise ConversionError(
                    f"{part_entry['part_id']} frame {global_index} timestamp {timestamp} "
                    f"!= {expected_timestamp}"
                )
            _assert_equal(int(_numpy(output["frame_index"]).item()), local_index, "frame_index")
            _assert_equal(
                int(_numpy(output["episode_index"]).item()),
                int(episode["target_episode_index"]),
                "episode_index",
            )
            _assert_equal(int(_numpy(output["index"]).item()), global_index, "index")
            _assert_equal(str(output["task"]), episode["source_task"], "task text")
            expected_task_index = next(
                int(index)
                for index, task in manifest["task_index_mapping"].items()
                if task == episode["source_task"]
            )
            _assert_equal(
                int(_numpy(output["task_index"]).item()), expected_task_index, "task_index"
            )
            sample_rows.append(
                {
                    "global_index": global_index,
                    "episode_index": int(episode["target_episode_index"]),
                    "source_frame_index": local_index,
                    "timestamp": timestamp,
                    "features": feature_rows,
                }
            )
        cursor += episode_length

    statistics_by_key = {row["target_key"]: row for row in manifest["source_numeric_statistics"]}
    collection_statistics = {
        row["target_key"]: row for row in part_entry["source_numeric_statistics"]
    }
    statistics_evidence: dict[str, Any] = {}
    for mapping in manifest["field_mapping"]:
        key = mapping["target_key"]
        meta_feature = dataset.meta.features[key]
        _assert_equal(meta_feature["dtype"], mapping["dtype"], f"{key} metadata dtype")
        _assert_equal(list(meta_feature["shape"]), mapping["shape"], f"{key} metadata shape")
        _assert_equal(meta_feature.get("names"), mapping.get("names"), f"{key} metadata names")
        primitive = _primitive_arrow_type(table.schema.field(key).type)
        _assert_equal(primitive, _expected_arrow_type(mapping["dtype"]), f"{key} Parquet dtype")
        computed = _numeric_statistics(source_values[key])
        for recorded in (statistics_by_key[key], collection_statistics[key]):
            for stat_name, value in computed.items():
                _assert_equal(recorded[stat_name], value, f"{key} source {stat_name}")
        _assert_equal(written_stats[key]["count"], [computed["frames"]], f"{key} stats count")
        written_min = np.asarray(written_stats[key]["min"]).reshape(-1)
        written_max = np.asarray(written_stats[key]["max"]).reshape(-1)
        if len(mapping["shape"]) >= 2:
            nonfinite = sum(computed["nan_count"]) + sum(
                computed["positive_infinity_count"]
            ) + sum(computed["negative_infinity_count"])
            finite_min = [value for value in computed["finite_min"] if value is not None]
            finite_max = [value for value in computed["finite_max"] if value is not None]
            if written_min.size != 1 or written_max.size != 1:
                raise ConversionError(f"{key} ArrayND stats.json range is not aggregate")
            if nonfinite == 0 and finite_min and finite_max and (
                not np.array_equal(written_min[0], min(finite_min))
                or not np.array_equal(written_max[0], max(finite_max))
            ):
                raise ConversionError(f"{key} aggregate stats.json range differs from source")
            statistics_evidence[key] = computed
            continue
        for component_index in range(len(computed["finite_min"])):
            nonfinite = sum(
                computed[name][component_index]
                for name in (
                    "nan_count",
                    "positive_infinity_count",
                    "negative_infinity_count",
                )
            )
            if nonfinite == 0:
                if not np.array_equal(
                    written_min[component_index],
                    computed["finite_min"][component_index],
                ):
                    raise ConversionError(
                        f"{key} stats.json min component {component_index} differs from source"
                    )
                if not np.array_equal(
                    written_max[component_index],
                    computed["finite_max"][component_index],
                ):
                    raise ConversionError(
                        f"{key} stats.json max component {component_index} differs from source"
                    )
        statistics_evidence[key] = computed

    unexpected_features = set(dataset.meta.features) - set(source_values) - GENERATED_FEATURES
    if unexpected_features:
        raise ConversionError(f"unexpected LeRobot features: {sorted(unexpected_features)}")
    video_files = sorted(part_root.rglob("*.mp4"))
    if video_files:
        raise ConversionError(f"source has no cameras but output has videos: {video_files}")
    invalid_casts = [
        cast
        for cast in manifest["dtype_casts"]
        if cast.get("lossy") is not False
        or cast.get("numeric_values_exact") is not True
        or cast.get("episode_boundary_preserved") is not True
    ]
    if invalid_casts or manifest["dropped_fields"] or manifest["reordered_fields"]:
        raise ConversionError(f"part manifest declares a lossy transformation: {part_entry['part_id']}")
    episode_casts = [
        {
            "source_episode_uid": episode["source_episode_uid"],
            **promotion,
        }
        for episode in manifest["episodes"]
        for promotion in episode.get("lossless_dtype_promotions", [])
    ]
    _assert_equal(episode_casts, manifest["dtype_casts"], "episode/part dtype promotion provenance")

    return {
        "part_id": part_entry["part_id"],
        "stream_kind": manifest["stream_kind"],
        "episodes": expected_episodes,
        "frames": expected_frames,
        "features": len(manifest["field_mapping"]),
        "robot_type": manifest["robot_type"],
        "fps": manifest["fps"],
        "time_basis": manifest["time_basis"],
        "task_index_mapping": actual_task_mapping,
        "sample_indices": [row["global_index"] for row in sample_rows],
        "samples": sample_rows,
        "source_statistics": statistics_evidence,
        "parquet_storage_dtype_and_values_preserved": not manifest["dtype_casts"],
        "parquet_storage_schema_and_numeric_values_verified": True,
        "lossless_dtype_promotions_verified": len(manifest["dtype_casts"]),
        "lerobot_values_exact_at_samples": all_lerobot_values_exact,
        "lerobot_values_match_documented_runtime_materialization": True,
        "lerobot_metadata_schema_exact": True,
        "video_files": 0,
    }


def evaluate_collection(collection_root: Path, raw_root: Path) -> dict[str, Any]:
    collection_root = collection_root.expanduser().resolve()
    raw_root = raw_root.expanduser().resolve()
    collection = _read_json(collection_root / "collection_manifest.json")
    if collection.get("converter") != "convert_roboverse_to_lerobot.py":
        raise ConversionError(f"not a RoboVerse conversion: {collection_root}")
    source_cache: dict[str, Any] = {}
    parts = [
        _evaluate_part(collection_root, raw_root, collection, part, source_cache)
        for part in collection["parts"]
    ]

    index_rows = [
        json.loads(line)
        for line in (collection_root / "source_episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    _assert_equal(len(index_rows), collection["source_episode_count"], "source episode index size")
    indexed_parts = {
        (linked["part_id"], int(linked["target_episode_index"]))
        for row in index_rows
        for linked in row["output_parts"]
    }
    manifested_parts = {
        (part["part_id"], int(episode["target_episode_index"]))
        for part in collection["parts"]
        for episode in _read_json(
            collection_root / part["path"] / "conversion_manifest.json"
        )["episodes"]
    }
    _assert_equal(indexed_parts, manifested_parts, "source/output provenance links")

    internal_caches = [path.as_posix() for path in collection_root.rglob(".lerobot-datasets-cache")]
    resume_paths = [
        path.as_posix()
        for suffix in ("resume", "resume-state")
        if (path := collection_root.parent / f".{collection_root.name}.{suffix}").exists()
    ]
    if internal_caches or resume_paths:
        raise ConversionError(
            f"published collection contains validation/checkpoint residue: {internal_caches + resume_paths}"
        )
    return {
        "collection_root": str(collection_root),
        "source_root": str(raw_root),
        "converter_version": collection["converter_version"],
        "source_dataset": collection["source_dataset"],
        "source_revision": collection["source_revision"],
        "source_episodes": collection["source_episode_count"],
        "output_episodes": collection["output_episode_count"],
        "output_frames": collection["output_frame_count"],
        "parts": parts,
        "provenance_links_exact": True,
        "published_internal_cache_directories": 0,
        "resume_data_or_state_directories_after_publish": 0,
        "note": (
            "LeRobot's stock PyTorch transform materializes Python floating-point lists as "
            "torch.float32. Unchanged source dtypes are verified in meta/info.json and Parquet; "
            "opt-in exact promotions are verified against raw dtype runs and cast/restore equality. "
            "Sampled values must remain exact or equal the documented float32 runtime materialization."
        ),
        "passed": True,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_collection(args.collection_root, args.raw_root)
    if args.report is not None:
        _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
