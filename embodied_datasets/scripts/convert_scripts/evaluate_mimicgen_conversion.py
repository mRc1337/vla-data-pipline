#!/usr/bin/env python3
"""Sample exact numeric storage and decoded video quality for MimicGen output."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq

from convert_core.errors import ConversionError


def _parquet_value(root: Path, key: str, global_index: int) -> tuple[Any, str]:
    remaining = global_index
    for path in sorted((root / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        if remaining >= parquet.metadata.num_rows:
            remaining -= parquet.metadata.num_rows
            continue
        for row_group in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(row_group).num_rows
            if remaining >= rows:
                remaining -= rows
                continue
            table = parquet.read_row_group(row_group, columns=[key])
            return table[key][remaining].as_py(), str(table[key].type)
    raise ConversionError(f"global frame index {global_index} is outside parquet data in {root}")


def evaluate_partition(root: Path, source_root: Path, psnr_threshold: float) -> dict[str, Any]:
    manifest = json.loads((root / "conversion_manifest.json").read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    total_frames = sum(int(episode["num_frames"]) for episode in episodes)
    sample_indices = sorted({0, total_frames // 2, total_frames - 1})
    starts: list[int] = []
    cursor = 0
    for episode in episodes:
        starts.append(cursor)
        cursor += int(episode["num_frames"])

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=manifest["dataset_uid"], root=root, return_uint8=True)
    numeric_checked = image_checked = 0
    minimum_psnr = math.inf
    for global_index in sample_indices:
        episode_index = max(index for index, start in enumerate(starts) if start <= global_index)
        frame_index = global_index - starts[episode_index]
        source_ref = episodes[episode_index]["source"]
        relative_file, demo_path = source_ref.split("::", 1)
        with h5py.File(source_root / relative_file, "r") as h5_file:
            demo = h5_file[demo_path]
            output_row = dataset[global_index]
            for mapping in manifest["field_mapping"]:
                source_key = mapping["source_key"].replace("data/<demo>/", "")
                lerobot_key = mapping["lerobot_key"]
                source = np.asarray(demo[source_key][frame_index])
                if source_key.endswith("_image"):
                    decoded = np.asarray(output_row[lerobot_key])
                    if decoded.shape[0] == 3:
                        decoded = np.moveaxis(decoded, 0, -1)
                    mse = float(np.mean((source.astype(np.float64) - decoded.astype(np.float64)) ** 2))
                    psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
                    minimum_psnr = min(minimum_psnr, psnr)
                    image_checked += 1
                    continue
                stored, arrow_type = _parquet_value(root, lerobot_key, global_index)
                stored_array = np.asarray(stored)
                if not np.array_equal(source.reshape(-1), stored_array.reshape(-1)):
                    raise ConversionError(
                        f"{root.name} frame {global_index} feature {lerobot_key}: stored value differs from source"
                    )
                expected_dtype = mapping["source_dtype"]
                arrow_expected = {
                    "float64": "double",
                    "float32": "float",
                    "int64": "int64",
                    "int32": "int32",
                    "bool": "bool",
                }.get(expected_dtype)
                if arrow_expected is not None and arrow_expected not in arrow_type:
                    raise ConversionError(
                        f"{root.name} feature {lerobot_key}: parquet type {arrow_type}, expected {expected_dtype}"
                    )
                numeric_checked += 1
    if minimum_psnr < psnr_threshold:
        raise ConversionError(
            f"{root.name}: minimum sampled PSNR {minimum_psnr:.2f} dB is below {psnr_threshold:.2f} dB"
        )
    return {
        "partition": root.name,
        "sample_indices": sample_indices,
        "numeric_values_exact": numeric_checked,
        "decoded_images_checked": image_checked,
        "minimum_psnr_db": minimum_psnr,
        "lerobot_reopened": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--psnr-threshold", type=float, default=30.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    collection = json.loads((args.collection_root / "collection_manifest.json").read_text(encoding="utf-8"))
    results = [
        evaluate_partition(args.collection_root / item["path"], args.source_root, args.psnr_threshold)
        for item in collection["partitions"]
    ]
    report = {
        "collection_root": str(args.collection_root),
        "source_root": str(args.source_root),
        "partitions": results,
        "passed": True,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
