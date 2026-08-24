"""Bounded DexCap → LeRobot v3.0 conversion entry point.

DexCap uses the same generic LeRobot work-unit and direct-commit machinery as
the already validated ARCap converter, but has its own reader and field
contract.  The small adapter below changes only the reader/configuration
surface; it does not route DexCap files through ARCap parsing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import convert_arcap_to_lerobot as _pipeline
from readers import dexcap_hdf5_reader as _reader


_pipeline.DEFAULT_RAW_ROOT = Path(
    "/mnt/data/embodied_datasets/public_datasets_raw/dexcap"
)
_pipeline.DEFAULT_LOCAL_WORK_ROOT = Path("/home/pai/zxw/dexcap_staging")
_pipeline.PARTITION_SPECS = _reader.PARTITION_SPECS
_pipeline.OFFICIAL_PARTITIONS = tuple(item.name for item in _reader.PARTITION_SPECS)
_pipeline.READER_FORMAT = "dexcap_hdf5"
_pipeline.PIPELINE_LABEL = "DexCap"
_pipeline.SOURCE_DATASET = _reader.SOURCE_DATASET
_pipeline.COLLECTION_KIND = "dexcap_collection"
_pipeline.__file__ = __file__
_pipeline.inspect_partition = _reader.inspect_partition
_pipeline.iter_frames = _reader.iter_frames


_original_conversion_options = _pipeline._conversion_options
_original_estimate = _pipeline._estimate
_original_manifest = _pipeline._manifest


def _conversion_options(args):
    values = dict(_original_conversion_options(args))
    values.pop("arcap_converter_schema_version", None)
    values.update(
        {
            "dexcap_converter_schema_version": 1,
            "video_features": 1,
            "timestamp_expression": "frame_index / 10",
            "action_gap": "stored one-frame-ahead target; no shift applied",
        }
    )
    return values


# The shared orchestrator's historical estimate is ARCap-specific.  DexCap's
# real bounded W4/U2 run produced 365,991,792 bytes for 1,108 packaging frames;
# use that measured ratio and the reader's all-episode metadata to estimate the
# selected scope and the complete selected-partition output.  This remains a
# planning estimate, not a claim that the full conversion has run.
_BENCHMARK_FRAMES = 1_108
_BENCHMARK_OUTPUT_BYTES = 365_991_792
_BENCHMARK_END_TO_END_FPS = 23.753111497349607


def _estimate(infos, units):
    estimate = dict(_original_estimate(infos, units))
    selected_frames = sum(info.plan.num_frames for info in infos)
    full_frames = sum(info.all_frame_count for info in infos)
    if full_frames <= 0:
        return estimate
    expected_full = math.ceil(
        _BENCHMARK_OUTPUT_BYTES * full_frames / _BENCHMARK_FRAMES
    )
    expected_selected = math.ceil(expected_full * selected_frames / full_frames)
    estimate.update(
        {
            "method": (
                "real DexCap local-to-OSS W4/U2 benchmark scaled by selected frames; "
                "15% conservative output margin; planning estimate only"
            ),
            "expected_output_bytes": expected_selected,
            "conservative_output_bytes": math.ceil(expected_selected * 1.15),
            "reference_end_to_end_frames_per_second": _BENCHMARK_END_TO_END_FPS,
            "estimated_wall_seconds": selected_frames / _BENCHMARK_END_TO_END_FPS,
            "estimated_wall_time_basis": (
                "selected real DexCap W4/U2 end-to-end benchmark; planning projection only"
            ),
            "expected_full_output_bytes": expected_full,
            "full_dataset_frames": full_frames,
            "benchmark_reference_frames": _BENCHMARK_FRAMES,
            "benchmark_reference_output_bytes": _BENCHMARK_OUTPUT_BYTES,
        }
    )
    return estimate


def _manifest(infos, local, args):
    manifest = _original_manifest(infos, local, args)
    manifest.update(
        {
            "source_dataset": _reader.SOURCE_DATASET,
            "source_revision": _reader.SOURCE_REVISION,
            "video_features": 1,
            "partition_rule": "one deterministic partition per released DexCap HDF5 file",
            "cleaning_or_128d_mapping_applied": False,
        }
    )
    return manifest


_pipeline._conversion_options = _conversion_options
_pipeline._estimate = _estimate
_pipeline._manifest = _manifest


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not any(value == "--output-dataset-uid" or value.startswith("--output-dataset-uid=") for value in values):
        values.extend(("--output-dataset-uid", "dexcap"))
    return _pipeline.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
