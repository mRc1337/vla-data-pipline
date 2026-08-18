"""Independently compare a converted 1X World Model collection with source data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import av
import numpy as np

from readers.one_x_world_model_reader import OneXWorldModelReader


CAMERA_KEY = "observation.images.head"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_uint8_rgb(value: Any) -> np.ndarray:
    image = _as_numpy(value)
    if image.shape == (3, 256, 256):
        image = image.transpose(1, 2, 0)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError(f"unexpected output image {image.shape}/{image.dtype}")
    return image


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))
    # Use a finite sentinel so the machine-readable JSON remains RFC-compliant.
    return 100.0 if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def _vector_result(
    output_value: Any,
    source_value: np.ndarray,
    feature_schema: dict[str, Any],
) -> dict[str, Any]:
    output = _as_numpy(output_value)
    source = np.asarray(source_value)
    expected_shape = tuple(int(value) for value in feature_schema["shape"])
    expected_dtype = np.dtype(feature_schema["dtype"])
    shape_matches = output.shape == source.shape == expected_shape
    dtype_matches = output.dtype == source.dtype == expected_dtype
    exact = bool(shape_matches and dtype_matches and np.array_equal(output, source))
    max_abs_error = None
    if output.shape == source.shape:
        max_abs_error = float(
            np.max(
                np.abs(output.astype(np.float64) - source.astype(np.float64))
            )
        )
    return {
        "exact": exact,
        "shape_matches": shape_matches,
        "dtype_matches": dtype_matches,
        "expected_dtype": str(expected_dtype),
        "output_dtype": str(output.dtype),
        "source_dtype": str(source.dtype),
        "expected_shape": list(expected_shape),
        "source_shape": list(source.shape),
        "output_shape": list(output.shape),
        "max_abs_error": max_abs_error,
    }


def _episode_at(manifest: dict[str, Any], index: int) -> tuple[dict[str, Any], int]:
    cursor = 0
    for episode in manifest["episodes"]:
        length = int(episode["num_frames"])
        if index < cursor + length:
            return episode, index - cursor
        cursor += length
    raise IndexError(index)


def _span_at(episode: dict[str, Any], local_index: int) -> tuple[dict[str, Any], int]:
    cursor = 0
    for span in episode["source_spans"]:
        length = int(span["end"]) - int(span["start"])
        if local_index < cursor + length:
            return span, int(span["start"]) + local_index - cursor
        cursor += length
    raise IndexError(local_index)


def _video_evidence(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = int(stream.frames or 0)
        if frames <= 0:
            frames = sum(1 for _ in container.decode(stream))
        rate = stream.average_rate or stream.base_rate
        return {
            "path": str(path),
            "frames": frames,
            "fps": float(rate) if rate is not None else None,
            "width": int(stream.width),
            "height": int(stream.height),
            "codec": stream.codec.canonical_name,
        }


def _source_vectors(
    source_root: Path,
    version: str,
    span: dict[str, Any],
    source_index: int,
    manifest: dict[str, Any],
) -> dict[str, np.ndarray]:
    if version == "v2.0":
        values = np.memmap(source_root / span["state"], dtype=np.float32, mode="r").reshape(-1, 25)
        return {"observation.state": np.array(values[source_index], copy=True)}
    result = {}
    for mapping in manifest["field_mapping"]:
        source_name = Path(mapping["source"]).stem
        feature = mapping["lerobot"]
        width = int(manifest["features"][feature]["shape"][0])
        values = np.memmap(
            source_root / span["actions"][source_name], dtype=np.float32, mode="r"
        ).reshape(-1, width)
        result[feature] = np.array(values[source_index], copy=True)
    return result


def _source_image(
    reader: OneXWorldModelReader,
    decoder_plan: Any,
    source_root: Path,
    version: str,
    span: dict[str, Any],
    source_index: int,
) -> np.ndarray:
    if version == "v1.1":
        tokens = np.memmap(source_root / span["video"], dtype=np.uint32, mode="r").reshape(-1, 16, 16)
        return reader._get_v1_decoder(decoder_plan)(np.array(tokens[source_index : source_index + 1]))[0]
    tokens = np.memmap(source_root / span["video"], dtype=np.int32, mode="r").reshape(-1, 3, 32, 32)
    block_index, offset = divmod(source_index, 17)
    return reader._get_v2_decoder(decoder_plan)(np.array(tokens[block_index : block_index + 1]))[0, offset]


def evaluate_partition(
    partition_root: Path,
    *,
    version: str,
    source_root: Path,
    v1_decoder_repo: Path | None,
    cosmos_decoder_path: Path | None,
    min_psnr_db: float,
    skip_pixels: bool,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = _read_json(partition_root / "conversion_manifest.json")
    dataset = LeRobotDataset(repo_id=manifest["dataset_uid"], root=partition_root)
    if len(dataset) != int(manifest["num_frames"]):
        raise ValueError("LeRobot frame count differs from conversion manifest")
    if dataset.num_episodes != int(manifest["num_episodes"]):
        raise ValueError("LeRobot episode count differs from conversion manifest")

    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    reader = OneXWorldModelReader()
    decoder_plan = SimpleNamespace(
        extra={
            "source_root": source_root,
            "decoder": {
                "v1_decoder_repo": str(v1_decoder_repo) if v1_decoder_repo else None,
                "v2_decoder_path": str(cosmos_decoder_path) if cosmos_decoder_path else None,
                "batch_size": 1,
            },
        }
    )
    samples = []
    all_exact = True
    pixel_pass = True
    for index in indices:
        episode, local_index = _episode_at(manifest, index)
        span, source_index = _span_at(episode, local_index)
        output = dataset[index]
        vector_results = {}
        for key, source_value in _source_vectors(
            source_root, version, span, source_index, manifest
        ).items():
            result = _vector_result(output[key], source_value, manifest["features"][key])
            all_exact &= bool(result["exact"])
            vector_results[key] = result
        expected_timestamp = local_index / float(manifest["fps"])
        timestamp = float(_as_numpy(output["timestamp"]).item())
        if not math.isclose(timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(
                f"frame {index}: timestamp {timestamp} != expected {expected_timestamp}"
            )
        sample = {
            "output_index": index,
            "source_split": episode["source_split"],
            "source_segment_id": episode["source_segment_id"],
            "source_index": source_index,
            "vector_features": vector_results,
            "timestamp": timestamp,
        }
        if not skip_pixels:
            source_image = _source_image(
                reader, decoder_plan, source_root, version, span, source_index
            )
            output_image = _as_uint8_rgb(output[CAMERA_KEY])
            psnr = _psnr(source_image, output_image)
            sample["image_psnr_db"] = psnr
            pixel_pass &= psnr >= min_psnr_db
        samples.append(sample)

    video_paths = sorted((partition_root / "videos" / CAMERA_KEY).rglob("*.mp4"))
    videos = [_video_evidence(path) for path in video_paths]
    video_frames = sum(int(row["frames"]) for row in videos)
    video_pass = (
        video_frames == len(dataset)
        and all(row["width"] == 256 and row["height"] == 256 for row in videos)
        and all(math.isclose(float(row["fps"]), 30.0, abs_tol=1e-9) for row in videos)
    )
    return {
        "partition": partition_root.name,
        "version": version,
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "samples": samples,
        "vectors_exact": all_exact,
        "pixels_checked": not skip_pixels,
        "pixels_pass": pixel_pass,
        "videos": videos,
        "videos_pass": video_pass,
        "passed": all_exact and pixel_pass and video_pass,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/data/embodied_datasets/public_datasets_raw/1x_world_model_dataset"),
    )
    parser.add_argument("--v1-decoder-repo", type=Path)
    parser.add_argument("--cosmos-decoder-path", type=Path)
    parser.add_argument("--min-psnr-db", type=float, default=30.0)
    parser.add_argument("--skip-pixels", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    collection = _read_json(args.collection_root / "collection_manifest.json")
    results = []
    for partition in collection["partitions"]:
        results.append(
            evaluate_partition(
                args.collection_root / partition["name"],
                version=partition["source_version"],
                source_root=args.source_root,
                v1_decoder_repo=args.v1_decoder_repo,
                cosmos_decoder_path=args.cosmos_decoder_path,
                min_psnr_db=args.min_psnr_db,
                skip_pixels=args.skip_pixels,
            )
        )
    report = {
        "collection": str(args.collection_root),
        "source_revision": collection["source_revision"],
        "min_psnr_db": args.min_psnr_db,
        "partitions": results,
        "passed": all(row["passed"] for row in results),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
