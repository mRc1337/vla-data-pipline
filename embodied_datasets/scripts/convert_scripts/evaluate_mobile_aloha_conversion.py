"""Evaluate a converted Mobile ALOHA partition against its source HDF5 files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from convert_core.hdf5_common import CameraSpec, read_rgb_frame
from convert_mobile_aloha_to_lerobot import _video_stream_summary


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_uint8(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.rint(array)
    return np.clip(array, 0, 255).astype(np.uint8)


def _source_numeric(
    h5_file: h5py.File,
    manifest: dict[str, Any],
    episode: dict[str, Any],
    frame_index: int,
) -> dict[str, np.ndarray]:
    keys = manifest["hdf5_keys"]
    action = np.asarray(h5_file[keys["action"]][frame_index], dtype=np.float32).reshape(-1)
    values = {
        "observation.state": np.asarray(
            h5_file[keys["state"]][frame_index], dtype=np.float32
        ).reshape(-1),
        "action": action[:14],
    }
    if "action.base" in manifest["features"]:
        if action.size >= 16:
            values["action.base"] = action[14:16]
        else:
            values["action.base"] = np.asarray(
                h5_file[keys["base_action"]][frame_index], dtype=np.float32
            ).reshape(-1)
    if "observation.velocity" in manifest["features"]:
        values["observation.velocity"] = np.asarray(
            h5_file[keys["velocity"]][frame_index], dtype=np.float32
        ).reshape(-1)
    if "observation.effort" in manifest["features"]:
        values["observation.effort"] = np.asarray(
            h5_file[keys["effort"]][frame_index], dtype=np.float32
        ).reshape(-1)
    return values


def _sample_indices(length: int, count: int) -> list[int]:
    return sorted(set(np.linspace(0, length - 1, min(length, count), dtype=int).tolist()))


def evaluate(
    partition_root: Path,
    *,
    samples: int = 20,
    min_psnr_db: float = 30.0,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = json.loads((partition_root / "conversion_manifest.json").read_text(encoding="utf-8"))
    dataset = LeRobotDataset(
        repo_id=manifest["dataset_uid"],
        root=partition_root,
        video_backend="pyav",
    )
    expected_frames = int(manifest["num_frames"])
    video_results: dict[str, Any] = {}
    all_video_frames_match = True
    for key, feature in manifest["features"].items():
        if feature["dtype"] != "video":
            continue
        paths = sorted((partition_root / "videos" / key).rglob("*.mp4"))
        frame_count, codecs, frame_rates = _video_stream_summary(paths)
        size_bytes = sum(path.stat().st_size for path in paths)
        video_results[key] = {
            "files": len(paths),
            "frames": frame_count,
            "expected_frames": expected_frames,
            "frames_match": frame_count == expected_frames,
            "codecs": sorted(codecs),
            "frame_rates": sorted(frame_rates),
            "size_bytes": size_bytes,
        }
        all_video_frames_match &= frame_count == expected_frames

    episode_starts: list[int] = []
    running = 0
    for episode in manifest["episodes"]:
        episode_starts.append(running)
        running += int(episode["num_frames"])

    camera_error: dict[str, list[float]] = {
        key: [] for key, feature in manifest["features"].items() if feature["dtype"] == "video"
    }
    numeric_max_abs_error: dict[str, float] = {}
    for global_index in _sample_indices(expected_frames, samples):
        episode_index = max(
            index for index, start in enumerate(episode_starts) if start <= global_index
        )
        episode = manifest["episodes"][episode_index]
        local_index = global_index - episode_starts[episode_index]
        item = dataset[global_index]
        source_path = Path(manifest["raw_dataset_root"]) / episode["source"]
        with h5py.File(source_path, "r") as h5_file:
            for key, expected in _source_numeric(
                h5_file, manifest, episode, local_index
            ).items():
                error = float(np.max(np.abs(_as_numpy(item[key]).reshape(-1) - expected)))
                numeric_max_abs_error[key] = max(numeric_max_abs_error.get(key, 0.0), error)

            for camera_dict in episode["cameras"]:
                camera = CameraSpec(**camera_dict)
                reference = read_rgb_frame(
                    h5_file[camera.source_key],
                    local_index,
                    camera,
                    source_path=source_path,
                    uncompressed_color_order=manifest["uncompressed_color_order"],
                )
                converted = _rgb_uint8(item[camera.feature_key])
                mse = float(
                    np.mean(
                        (
                            converted.astype(np.float32)
                            - reference.astype(np.float32)
                        )
                        ** 2
                    )
                )
                camera_error[camera.feature_key].append(mse)

    quality: dict[str, Any] = {}
    all_quality_pass = True
    for key, errors in camera_error.items():
        mean_mse = float(np.mean(errors))
        psnr = math.inf if mean_mse == 0 else 10.0 * math.log10((255.0**2) / mean_mse)
        passed = psnr >= min_psnr_db
        all_quality_pass &= passed
        quality[key] = {
            "samples": len(errors),
            "mean_mse": mean_mse,
            "psnr_db": "inf" if math.isinf(psnr) else psnr,
            "minimum_psnr_db": min_psnr_db,
            "pass": passed,
        }

    numeric_match = all(error <= 1e-6 for error in numeric_max_abs_error.values())
    structure_match = len(dataset) == expected_frames and dataset.num_episodes == manifest["num_episodes"]
    result = {
        "partition_root": str(partition_root.resolve()),
        "dataset_uid": manifest["dataset_uid"],
        "video_encoding": manifest.get("video_encoding"),
        "conversion_metrics": manifest.get("conversion_metrics"),
        "structure": {
            "frames": len(dataset),
            "expected_frames": expected_frames,
            "episodes": dataset.num_episodes,
            "expected_episodes": manifest["num_episodes"],
            "pass": structure_match,
        },
        "videos": video_results,
        "numeric_max_abs_error": numeric_max_abs_error,
        "numeric_pass": numeric_match,
        "image_quality": quality,
        "pass": structure_match and all_video_frames_match and numeric_match and all_quality_pass,
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-root", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--min-psnr-db", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    report = evaluate(
        args.partition_root,
        samples=args.samples,
        min_psnr_db=args.min_psnr_db,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
