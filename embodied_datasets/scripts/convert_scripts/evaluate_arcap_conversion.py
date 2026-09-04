"""Independently validate an ARCap staging collection against official HDF5."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Sequence

sys.dont_write_bytecode = True

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_core.checkpoint import read_json_object
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import validate_written_dataset
from convert_core.staging import sha256_file
from convert_core.staging import validate_contained_path
from readers.arcap_hdf5_reader import (
    PARTITIONS_BY_NAME,
    inspect_partition,
)


DEFAULT_RAW_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw/arcap")
DEFAULT_OUTPUT = Path(
    "/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/arcap"
)
APPROVED_ROOT = DEFAULT_OUTPUT.parent


def _remove_tree_with_retries(path: Path, *, attempts: int = 8) -> None:
    for attempt in range(attempts):
        if not path.exists():
            return
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        time.sleep(0.1 * (attempt + 1))
    raise OSError(f"could not remove evaluation cache after {attempts} attempts: {path}")


@contextmanager
def _evaluation_environment(cache: Path):
    cache = validate_contained_path(cache, APPROVED_ROOT, label="evaluation cache")
    values = {
        "TMPDIR": cache / "tmp",
        "TMP": cache / "tmp",
        "TEMP": cache / "tmp",
        "XDG_CACHE_HOME": cache / "xdg",
        "HF_HOME": cache / "huggingface",
        "HF_DATASETS_CACHE": cache / "datasets",
        "VLA_DATASETS_CACHE_ROOT": cache / "datasets",
        "TORCH_HOME": cache / "torch",
        "MPLCONFIGDIR": cache / "matplotlib",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
    }
    previous = {key: os.environ.get(key) for key in values}
    previous_tempdir = tempfile.tempdir
    for path in set(values.values()):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({key: str(value) for key, value in values.items()})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(values["TMPDIR"])
    try:
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _remove_tree_with_retries(cache)


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    tables = [pq.read_table(path) for path in sorted((root / "meta/episodes").rglob("*.parquet"))]
    if not tables:
        raise ConversionError(f"missing episode metadata: {root}")
    table = pa.concat_tables(tables)
    return table.to_pylist()


def _find_rows(
    root: Path, global_indices: Sequence[int], columns: list[str]
) -> dict[int, dict[str, Any]]:
    wanted = set(global_indices)
    if not wanted:
        return {}
    found: dict[int, dict[str, Any]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for group_index in range(parquet.num_row_groups):
            row_group = parquet.metadata.row_group(group_index)
            index_column = next(
                (
                    row_group.column(index)
                    for index in range(row_group.num_columns)
                    if row_group.column(index).path_in_schema == "index"
                ),
                None,
            )
            statistics = index_column.statistics if index_column is not None else None
            if (
                statistics is not None
                and statistics.has_min_max
                and not any(statistics.min <= value <= statistics.max for value in wanted)
            ):
                continue
            indices = parquet.read_row_group(group_index, columns=["index"])["index"].to_numpy()
            positions = {
                int(value): row
                for row, value in enumerate(indices)
                if int(value) in wanted
            }
            if not positions:
                continue
            table = parquet.read_row_group(group_index, columns=columns)
            for value, row in positions.items():
                if value in found:
                    raise ConversionError(f"duplicate global frame index {value} below {root}")
                found[value] = {key: table[key][row].as_py() for key in columns}
                wanted.remove(value)
            if not wanted:
                return found
    raise ConversionError(
        f"cannot locate global frame indices {sorted(wanted)} below {root}"
    )


def _samples(plan: Any, sample_episodes: int) -> list[tuple[int, int, int]]:
    episode_ids = sorted(
        {0, len(plan.episodes) // 2, len(plan.episodes) - 1}
        if sample_episodes >= 3
        else np.linspace(0, len(plan.episodes) - 1, sample_episodes, dtype=int).tolist()
    )
    starts: list[int] = []
    cursor = 0
    for episode in plan.episodes:
        starts.append(cursor)
        cursor += episode.num_frames
    result = []
    for episode_index in episode_ids:
        length = plan.episodes[episode_index].num_frames
        for local_frame in sorted({0, length // 2, length - 1}):
            result.append((episode_index, local_frame, starts[episode_index] + local_frame))
    return result


def evaluate_partition(
    raw_root: Path,
    collection: Path,
    partition_record: dict[str, Any],
    *,
    sample_episodes: int,
) -> dict[str, Any]:
    name = str(partition_record["name"])
    spec = PARTITIONS_BY_NAME[name]
    expected_episodes = int(partition_record["episodes"])
    if expected_episodes % spec.phase_group_size:
        raise ConversionError(f"manifest partition {name} cuts a phase group")
    info = inspect_partition(
        raw_root / spec.filename,
        raw_dataset_root=raw_root,
        collection_output=collection,
        max_phase_groups=expected_episodes // spec.phase_group_size,
    )
    plan = info.plan
    output = collection / name
    validate_written_dataset(plan, output)
    rows = _episode_rows(output)
    if len(rows) != len(plan.episodes):
        raise ConversionError(f"{name}: episode metadata count changed")
    cursor = 0
    for index, (row, episode) in enumerate(zip(rows, plan.episodes, strict=True)):
        expected = {
            "episode_index": index,
            "length": episode.num_frames,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + episode.num_frames,
        }
        for key, value in expected.items():
            if int(row[key]) != value:
                raise ConversionError(f"{name}: episode boundary {key} changed at {index}")
        cursor += episode.num_frames

    mapping = plan.extra["field_mapping"]
    source_by_output = {
        item["lerobot_key"]: item["source_key"].removeprefix("data/<demo>/")
        for item in mapping
    }
    columns = [*source_by_output, "index", "episode_index", "task_index", "timestamp"]
    samples = _samples(plan, sample_episodes)
    written_rows = _find_rows(
        output, [global_frame for _, _, global_frame in samples], columns
    )
    exact_values = 0
    timestamp_error = 0.0
    with h5py.File(info.source_path, "r") as source:
        for episode_index, local_frame, global_frame in samples:
            episode = plan.episodes[episode_index]
            demo = source[f"data/{episode.extra['demo_name']}"]
            written = written_rows[global_frame]
            if int(written["episode_index"]) != episode_index:
                raise ConversionError(f"{name}: frame episode_index changed at {global_frame}")
            for output_key, source_key in source_by_output.items():
                expected = np.asarray(demo[source_key][local_frame])
                actual = np.asarray(written[output_key], dtype=expected.dtype)
                if expected.shape == ():
                    actual = actual.reshape(())
                else:
                    actual = actual.reshape(expected.shape)
                if not np.array_equal(actual, expected, equal_nan=True):
                    raise ConversionError(
                        f"{name}: Parquet value differs bit-exactly for {output_key} "
                        f"at episode {episode_index} frame {local_frame}"
                    )
                exact_values += expected.size
            timestamp_error = max(
                timestamp_error, abs(float(written["timestamp"]) - local_frame / 10.0)
            )

    # Reopen through the public loader as a separate presentation-layer check.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.dataset_uid, root=output)
    loader_max_error = 0.0
    loader_dtypes: set[str] = set()
    with h5py.File(info.source_path, "r") as source:
        for episode_index, local_frame, global_frame in _samples(plan, min(sample_episodes, 3)):
            item = dataset[global_frame]
            expected = np.asarray(
                source[
                    f"data/{plan.episodes[episode_index].extra['demo_name']}/obs/pointcloud"
                ][local_frame]
            )
            shown = np.asarray(item["observation.pointcloud"])
            loader_dtypes.add(str(shown.dtype))
            loader_max_error = max(loader_max_error, float(np.max(np.abs(shown - expected))))
    return {
        "partition": name,
        "episodes": len(plan.episodes),
        "frames": plan.num_frames,
        "sampled_frames": len(samples),
        "bit_exact_parquet_scalar_values": exact_values,
        "parquet_feature_dtype": "double/float64 preserved",
        "synthetic_timestamp_max_abs_error": timestamp_error,
        "loader_presentation_dtypes": sorted(loader_dtypes),
        "loader_presentation_max_abs_error": loader_max_error,
        "loader_note": "presentation conversion is distinct from raw Parquet fidelity",
    }


def evaluate(raw_root: Path, output: Path, sample_episodes: int) -> dict[str, Any]:
    success = read_json_object(output / "_SUCCESS", "ARCap success marker")
    if (output / "_INCOMPLETE").exists():
        raise ConversionError("collection has both _SUCCESS and _INCOMPLETE")
    manifest_path = output / "collection_manifest.json"
    if success.get("collection_manifest_sha256") != sha256_file(manifest_path):
        raise ConversionError("collection manifest digest differs from _SUCCESS")
    manifest = read_json_object(manifest_path, "ARCap collection manifest")
    partitions = [
        evaluate_partition(
            raw_root,
            output,
            record,
            sample_episodes=sample_episodes,
        )
        for record in manifest["partitions"]
    ]
    return {
        "status": "valid",
        "output": str(output),
        "partitions": partitions,
        "totals": {
            "episodes": sum(row["episodes"] for row in partitions),
            "frames": sum(row["frames"] for row in partitions),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-episodes", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.sample_episodes <= 0:
        parser.error("--sample-episodes must be positive")
    try:
        cache = args.cache_dir or (
            APPROVED_ROOT / ".conversion_work" / "arcap-evaluation" / args.output.name
        )
        with _evaluation_environment(cache):
            result = evaluate(args.raw_root, args.output, args.sample_episodes)
        print(json.dumps(result, indent=2))
        return 0
    except (ConversionError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
