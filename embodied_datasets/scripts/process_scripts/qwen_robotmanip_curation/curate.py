#!/usr/bin/env python3
"""Qwen-RobotManip stages 1-3 as validity masks over immutable LeRobot v3 data.

Stages write labels, step-validity masks, and episode filters.  Videos and
source Parquet files always remain immutable and are referenced by manifest.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fnmatch
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
PAPER = "https://arxiv.org/abs/2606.17846"
DEFAULT_OUTPUT_ROOT = pathlib.Path(
    "/mnt/data/embodied_datasets/public_datasets_staging/data_curation"
)
STAGE_NAMES = {
    1: "stage1",
    2: "stage2",
    3: "stage3",
}
DEFAULTS: dict[str, Any] = {
    "state_key": "observation.state",
    "action_key": "action",
    "embodiment": None,
    "action_mode": "absolute",
    "stage1_exclusion": "frame",
    "median_kernels": [5, 5],
    "savgol_window": 11,
    "savgol_polyorder": 3,
    "stage1_mad_scale": 8.0,
    "stage1_quantile_floor": 0.999,
    "stage2_da_threshold": 0.65,
    "stage2_max_lag_seconds": 0.5,
    "stage2_min_active_steps": 10,
    "stage3_alpha": 0.1,
    "calibration_samples": 200_000,
    "allow_positional_mapping": False,
    "gripper_indices": {"observation.state": [], "action": []},
    "angular_indices": {"observation.state": [], "action": []},
    "quaternion_groups": {"observation.state": [], "action": []},
    "state_action_map": None,
}


@dataclass(frozen=True)
class Dataset:
    path: pathlib.Path
    dataset_id: str
    info: dict[str, Any]
    cfg: dict[str, Any]
    accepted_episodes: frozenset[int] | None = None
    invalid_frames: frozenset[tuple[int, int]] = frozenset()
    manifest_path: pathlib.Path | None = None

    @property
    def fps(self) -> float:
        return float(self.info.get("fps") or 1.0)

    @property
    def state_key(self) -> str:
        return str(self.cfg["state_key"])

    @property
    def action_key(self) -> str:
        return str(self.cfg["action_key"])


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: pathlib.Path | None) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if path:
        raw = json.loads(path.read_text())
    return {
        "defaults": deep_merge(DEFAULTS, raw.get("defaults", {})),
        "datasets": raw.get("datasets", {}),
    }


def dataset_config(config: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    cfg = copy.deepcopy(config["defaults"])
    for pattern, override in config["datasets"].items():
        if fnmatch.fnmatch(dataset_id, pattern):
            cfg = deep_merge(cfg, override)
    if cfg["embodiment"] is None:
        cfg["embodiment"] = dataset_id
    if cfg["action_mode"] not in {"absolute", "delta"}:
        raise ValueError(f"{dataset_id}: action_mode must be absolute or delta")
    if cfg["stage1_exclusion"] not in {"frame", "episode"}:
        raise ValueError(f"{dataset_id}: stage1_exclusion must be frame or episode")
    return cfg


def discover(root: pathlib.Path, config: dict[str, Any], max_depth: int = 2) -> list[Dataset]:
    """Discover completed datasets without descending into data/video payloads."""
    root = root.resolve()
    found: list[Dataset] = []
    stack: list[tuple[pathlib.Path, int]] = [(root, 0)]
    skip = {"data", "videos", "images", "meta", "source_metadata"}
    while stack:
        current, depth = stack.pop()
        if current.name.startswith(".") or current.name.endswith(".incomplete"):
            continue
        info_path = current / "meta" / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            dataset_id = current.relative_to(root).as_posix()
            found.append(Dataset(current, dataset_id, info, dataset_config(config, dataset_id)))
            continue
        if depth >= max_depth:
            continue
        try:
            children = list(os.scandir(current))
        except OSError as exc:
            print(f"warning: cannot scan {current}: {exc}", file=sys.stderr)
            continue
        for entry in reversed(children):
            if (
                entry.is_dir(follow_symlinks=False)
                and not entry.name.startswith(".")
                and entry.name not in skip
                and not entry.name.endswith(".incomplete")
            ):
                stack.append((pathlib.Path(entry.path), depth + 1))
    return sorted(found, key=lambda item: item.dataset_id)


def load_dataset(path: pathlib.Path, config: dict[str, Any], dataset_id: str | None = None) -> Dataset:
    path = path.resolve()
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"completed LeRobot metadata not found: {info_path}")
    dataset_id = dataset_id or path.name
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"{dataset_id}: expected LeRobot v3.0 input, got {info.get('codebase_version')!r}"
        )
    return Dataset(path, dataset_id, info, dataset_config(config, dataset_id))


def flattened_names(feature: dict[str, Any] | None, dim: int) -> list[str | None]:
    names = None if feature is None else feature.get("names")
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list) or len(names) != dim:
        return [None] * dim
    return [None if x is None else str(x) for x in names]


def feature_dim(info: dict[str, Any], key: str) -> int:
    feature = info.get("features", {}).get(key)
    if not feature:
        raise KeyError(f"missing numeric feature {key!r}")
    shape = feature.get("shape") or []
    if not shape:
        raise ValueError(f"feature {key!r} has no shape")
    return int(math.prod(shape))


def parquet_files(ds: Dataset) -> list[pathlib.Path]:
    files = sorted(ds.path.glob("data/chunk-*/*.parquet"))
    if not files:
        files = sorted(ds.path.glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet files under {ds.path / 'data'}")
    return files


def read_episode_filter(path: pathlib.Path) -> frozenset[int]:
    if not path.is_file():
        raise FileNotFoundError(f"episode filter not found: {path}")
    table = pq.read_table(path, columns=["episode_index", "accepted"])
    return frozenset(
        int(row["episode_index"])
        for row in table.to_pylist()
        if bool(row["accepted"])
    )


def read_invalid_frames(path: pathlib.Path) -> frozenset[tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"step validity file not found: {path}")
    table = pq.read_table(
        path,
        columns=["episode_index", "frame_index", "valid"],
        filters=[("valid", "=", False)],
    )
    return frozenset(
        (int(row["episode_index"]), int(row["frame_index"]))
        for row in table.to_pylist()
    )


def _resolve_manifest_path(manifest_dir: pathlib.Path, value: str | None) -> pathlib.Path | None:
    if not value:
        return None
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (manifest_dir / path).resolve()


def dataset_from_manifest(
    manifest_path: pathlib.Path,
    config: dict[str, Any],
    dataset_id: str | None = None,
    _seen: frozenset[pathlib.Path] = frozenset(),
) -> Dataset:
    """Open a logical stage view while keeping the original dataset immutable."""
    manifest_path = manifest_path.resolve()
    if manifest_path in _seen:
        raise ValueError(f"cyclic parent_manifest chain at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "vla_curation_filter":
        legacy = manifest.get("format") == "vla_curation_overlay" or bool(manifest.get("repair_files"))
        detail = "; rerun the preceding stages to create vla_curation_filter schema v2" if legacy else ""
        raise ValueError(f"unsupported curation manifest: {manifest_path}{detail}")
    manifest_dir = manifest_path.parent
    parent_path = _resolve_manifest_path(manifest_dir, manifest.get("parent_manifest"))
    if parent_path:
        parent = dataset_from_manifest(
            parent_path, config, dataset_id or manifest.get("dataset_id"), _seen | {manifest_path}
        )
        source_path = parent.path
        invalid_frames = set(parent.invalid_frames)
    else:
        source_path = pathlib.Path(manifest["source_dataset"]).resolve()
        parent = load_dataset(source_path, config, dataset_id or manifest.get("dataset_id"))
        invalid_frames = set()
    for value in manifest.get("validity_files", []):
        validity_path = _resolve_manifest_path(manifest_dir, value)
        if validity_path is not None:
            invalid_frames.update(read_invalid_frames(validity_path))
    filter_path = _resolve_manifest_path(manifest_dir, manifest.get("episode_filter"))
    accepted = read_episode_filter(filter_path) if filter_path else parent.accepted_episodes
    if accepted is not None and parent.accepted_episodes is not None:
        accepted = frozenset(accepted & parent.accepted_episodes)
    return Dataset(
        source_path,
        dataset_id or str(manifest.get("dataset_id") or parent.dataset_id),
        parent.info,
        parent.cfg,
        accepted,
        frozenset(invalid_frames),
        manifest_path,
    )


def arrow_numpy(array: pa.Array) -> np.ndarray:
    if pa.types.is_fixed_size_list(array.type):
        size = array.type.list_size
        values = array.values.to_numpy(zero_copy_only=False)
        start = array.offset * size
        return np.asarray(values[start : start + len(array) * size], dtype=np.float64).reshape(len(array), size)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        return np.asarray(array.to_pylist(), dtype=np.float64)
    return np.asarray(array.to_numpy(zero_copy_only=False))


def iter_episodes(
    ds: Dataset,
    signal_keys: Iterable[str],
    max_episodes: int | None = None,
    batch_size: int = 65_536,
) -> Iterator[dict[str, np.ndarray]]:
    signal_keys = list(signal_keys)
    requested = ["episode_index", "frame_index", "index", *signal_keys]
    pending: dict[str, list[np.ndarray]] | None = None
    pending_episode: int | None = None
    yielded = 0
    last_episode: int | None = None

    def emit(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        episode = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
        episode_index = int(episode["episode_index"][0])
        episode["_step_valid"] = np.asarray([
            (episode_index, int(frame_index)) not in ds.invalid_frames
            for frame_index in episode["frame_index"]
        ], dtype=bool)
        return episode

    def accepted(episode_index: int) -> bool:
        return ds.accepted_episodes is None or episode_index in ds.accepted_episodes

    def maybe_emit(parts: dict[str, list[np.ndarray]], episode_index: int) -> dict[str, np.ndarray] | None:
        return emit(parts) if accepted(episode_index) else None

    for file in parquet_files(ds):
        pf = pq.ParquetFile(file)
        available = set(pf.schema_arrow.names)
        columns = [key for key in requested if key in available]
        required = {"episode_index", "frame_index", *signal_keys}
        missing = required - set(columns)
        if missing:
            raise KeyError(f"{file}: missing columns {sorted(missing)}")
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
            arrays = {name: arrow_numpy(batch.column(i)) for i, name in enumerate(batch.schema.names)}
            if "index" not in arrays:
                arrays["index"] = np.full(len(batch), -1, dtype=np.int64)
            episodes = arrays["episode_index"].astype(np.int64, copy=False)
            if len(episodes) == 0:
                continue
            boundaries = np.flatnonzero(np.diff(episodes) != 0) + 1
            starts = np.r_[0, boundaries]
            stops = np.r_[boundaries, len(episodes)]
            for start, stop in zip(starts, stops):
                episode = int(episodes[start])
                if last_episode is not None and episode < last_episode:
                    raise ValueError(f"episode_index is not monotonic near {file}: {episode} < {last_episode}")
                last_episode = episode
                piece = {key: value[start:stop] for key, value in arrays.items()}
                if pending_episode is None:
                    pending_episode = episode
                    pending = {key: [value] for key, value in piece.items()}
                elif episode == pending_episode:
                    assert pending is not None
                    for key, value in piece.items():
                        pending[key].append(value)
                else:
                    assert pending is not None
                    emitted = maybe_emit(pending, pending_episode)
                    if emitted is not None:
                        yield emitted
                        yielded += 1
                        if max_episodes is not None and yielded >= max_episodes:
                            return
                    pending_episode = episode
                    pending = {key: [value] for key, value in piece.items()}
    if pending is not None and (max_episodes is None or yielded < max_episodes):
        emitted = maybe_emit(pending, int(pending_episode))
        if emitted is not None:
            yield emitted


def normalized_signal(x: np.ndarray, cfg: dict[str, Any], key: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    for dim in cfg.get("angular_indices", {}).get(key, []):
        x[:, int(dim)] = np.unwrap(x[:, int(dim)])
    for group in cfg.get("quaternion_groups", {}).get(key, []):
        idx = np.asarray(group, dtype=int)
        for row in range(1, len(x)):
            if np.dot(x[row - 1, idx], x[row, idx]) < 0:
                x[row, idx] *= -1
    return x


def fill_nonfinite(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    for dim in range(out.shape[1]):
        y = out[:, dim]
        ok = np.isfinite(y)
        if ok.all():
            continue
        if not ok.any():
            y[:] = 0.0
        else:
            positions = np.arange(len(y))
            y[~ok] = np.interp(positions[~ok], positions[ok], y[ok])
    return out


def odd_window(requested: int, length: int, minimum: int = 3) -> int | None:
    value = min(int(requested), length if length % 2 else length - 1)
    if value % 2 == 0:
        value -= 1
    return value if value >= minimum else None


def smooth(x: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    y = fill_nonfinite(x)
    for requested in cfg["median_kernels"]:
        kernel = odd_window(int(requested), len(y))
        if kernel:
            y = median_filter(y, size=(kernel, 1), mode="nearest")
    polyorder = int(cfg["savgol_polyorder"])
    window = odd_window(int(cfg["savgol_window"]), len(y), polyorder + 2)
    if window:
        y = savgol_filter(y, window_length=window, polyorder=min(polyorder, window - 1), axis=0, mode="interp")
    return y


def finite_metrics(x: np.ndarray, cfg: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = normalized_signal(x, cfg, key)
    raw = fill_nonfinite(raw)
    trend = smooth(raw, cfg)
    residual = np.abs(raw - trend)
    acceleration = np.abs(np.diff(raw, n=2, axis=0, prepend=np.repeat(raw[:1], 2, axis=0)))
    jerk = np.abs(np.diff(raw, n=3, axis=0, prepend=np.repeat(raw[:1], 3, axis=0)))
    return residual, acceleration, jerk


class StrideSamples:
    def __init__(self, total_rows: int, capacity: int):
        self.stride = max(1, math.ceil(max(total_rows, 1) / max(capacity, 1)))
        self.position = 0
        self.parts: list[np.ndarray] = []
        self.capacity = capacity

    def add(self, values: np.ndarray) -> None:
        positions = np.arange(self.position, self.position + len(values))
        selected = values[positions % self.stride == 0]
        self.position += len(values)
        if len(selected):
            self.parts.append(np.asarray(selected, dtype=np.float64))

    def array(self) -> np.ndarray:
        if not self.parts:
            raise ValueError("no calibration samples collected")
        return np.concatenate(self.parts, axis=0)[: self.capacity]


def gripper_indices(ds: Dataset, key: str, dim: int) -> set[int]:
    explicit = {int(x) for x in ds.cfg.get("gripper_indices", {}).get(key, [])}
    names = flattened_names(ds.info.get("features", {}).get(key), dim)
    detected = {
        i
        for i, name in enumerate(names)
        if name and any(token in name.lower() for token in ("gripper", "finger", "jaw"))
    }
    return explicit | detected


def robust_threshold(values: np.ndarray, mad_scale: float, quantile_floor: float) -> np.ndarray:
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0)
    robust = med + mad_scale * 1.4826 * mad
    quantile = np.nanquantile(values, quantile_floor, axis=0)
    magnitude = np.maximum(np.nanquantile(np.abs(values), 0.99, axis=0), 1.0)
    return np.maximum.reduce([robust, quantile, magnitude * 1e-10])


class ParquetRows:
    def __init__(self, path: pathlib.Path, schema: pa.Schema, flush_rows: int = 20_000):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.flush_rows = flush_rows
        self.rows: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()

    def __enter__(self) -> "ParquetRows":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def publish(local_dir: pathlib.Path, output_dir: pathlib.Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_dir, output_dir)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


STAGE1_FRAME_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("frame_index", pa.int64()), ("index", pa.int64()),
    ("failed_state_dimensions", pa.list_(pa.int32())), ("failed_action_dimensions", pa.list_(pa.int32())),
    ("max_residual_ratio", pa.float64()), ("max_acceleration_ratio", pa.float64()),
    ("max_jerk_ratio", pa.float64()), ("flagged_frame", pa.bool_()),
])
STAGE1_EPISODE_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("num_frames", pa.int64()), ("flagged_frames", pa.int64()),
    ("valid_frames", pa.int64()), ("flag_fraction", pa.float64()),
    ("exclusion_policy", pa.string()), ("reject_episode", pa.bool_()),
])
THRESHOLD_SCHEMA = pa.schema([
    ("feature", pa.string()), ("dimension", pa.int32()), ("name", pa.string()),
    ("metric", pa.string()), ("threshold", pa.float64()), ("exempt_gripper", pa.bool_()),
])
EPISODE_FILTER_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("num_frames", pa.int64()),
    ("accepted", pa.bool_()), ("reason_code", pa.string()),
])
STEP_VALIDITY_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("frame_index", pa.int64()), ("index", pa.int64()),
    ("stage1_valid", pa.bool_()), ("stage2_valid", pa.bool_()),
    ("stage3_valid", pa.bool_()), ("valid", pa.bool_()),
    ("reason_codes", pa.list_(pa.string())),
])


def causal_chunk_validity(step_valid: np.ndarray) -> np.ndarray:
    """Mask every action-chunk step after the first invalid supervision step."""
    return np.logical_and.accumulate(np.asarray(step_valid, dtype=bool))


def causal_loss_mask(step_valid: np.ndarray, embodiment_slot_mask: np.ndarray) -> np.ndarray:
    """Compose the paper's causal step mask with the per-dimension slot mask."""
    causal = causal_chunk_validity(step_valid)
    slots = np.asarray(embodiment_slot_mask, dtype=bool)
    if slots.ndim != 1:
        raise ValueError("embodiment_slot_mask must be one-dimensional")
    return causal[:, None] & slots[None, :]


def run_stage1(ds: Dataset, local_dir: pathlib.Path, max_episodes: int | None) -> dict[str, Any]:
    started = utc_now()
    labels_dir = local_dir / "labels"
    state_dim = feature_dim(ds.info, ds.state_key)
    action_dim = feature_dim(ds.info, ds.action_key)
    total = int(ds.info.get("total_frames") or 0)
    capacity = int(ds.cfg["calibration_samples"])
    collectors = {
        (key, metric): StrideSamples(total, capacity)
        for key in (ds.state_key, ds.action_key)
        for metric in ("residual", "acceleration", "jerk")
    }
    calibration_episodes = 0
    for episode in iter_episodes(ds, [ds.state_key, ds.action_key], max_episodes):
        calibration_episodes += 1
        for key in (ds.state_key, ds.action_key):
            for metric, values in zip(("residual", "acceleration", "jerk"), finite_metrics(episode[key], ds.cfg, key)):
                collectors[(key, metric)].add(values)
    thresholds: dict[tuple[str, str], np.ndarray] = {}
    for key in (ds.state_key, ds.action_key):
        for metric in ("residual", "acceleration", "jerk"):
            thresholds[(key, metric)] = robust_threshold(
                collectors[(key, metric)].array(),
                float(ds.cfg["stage1_mad_scale"]),
                float(ds.cfg["stage1_quantile_floor"]),
            )

    labels_dir.mkdir(parents=True, exist_ok=True)
    flagged_frames = 0
    rejected_episodes = 0
    frame_count = 0
    output_episodes = output_frames = 0
    with ParquetRows(labels_dir / "frame_flags.parquet", STAGE1_FRAME_SCHEMA) as fw, ParquetRows(
        labels_dir / "episode_summary.parquet", STAGE1_EPISODE_SCHEMA
    ) as ew, ParquetRows(labels_dir / "episode_filter.parquet", EPISODE_FILTER_SCHEMA) as filter_writer, ParquetRows(
        labels_dir / "step_validity.parquet", STEP_VALIDITY_SCHEMA
    ) as validity_writer:
        for episode in iter_episodes(ds, [ds.state_key, ds.action_key], max_episodes):
            eid = int(episode["episode_index"][0])
            n = len(episode["frame_index"])
            frame_count += n
            feature_flags: dict[str, np.ndarray] = {}
            metric_values: dict[tuple[str, str], np.ndarray] = {}
            for key, dim in ((ds.state_key, state_dim), (ds.action_key, action_dim)):
                residual, acceleration, jerk = finite_metrics(episode[key], ds.cfg, key)
                metric_values[(key, "residual")] = residual
                metric_values[(key, "acceleration")] = acceleration
                metric_values[(key, "jerk")] = jerk
                flags = (residual > thresholds[(key, "residual")]) & (
                    (acceleration > thresholds[(key, "acceleration")]) | (jerk > thresholds[(key, "jerk")])
                )
                flags |= ~np.isfinite(episode[key])
                feature_flags[key] = flags
            union = feature_flags[ds.state_key].any(axis=1) | feature_flags[ds.action_key].any(axis=1)
            indices = np.flatnonzero(union)
            flagged_frames += len(indices)
            exclusion = str(ds.cfg["stage1_exclusion"])
            reject_episode = bool(len(indices) and exclusion == "episode")
            rejected_episodes += int(reject_episode)
            frame_valid = ~union
            if reject_episode:
                frame_valid[:] = False
            for row in indices:
                failed_state = np.flatnonzero(feature_flags[ds.state_key][row]).astype(int).tolist()
                failed_action = np.flatnonzero(feature_flags[ds.action_key][row]).astype(int).tolist()
                ratios: dict[str, float] = {}
                for metric in ("residual", "acceleration", "jerk"):
                    values = []
                    for key in (ds.state_key, ds.action_key):
                        denominator = np.maximum(thresholds[(key, metric)], np.finfo(np.float64).tiny)
                        values.append(metric_values[(key, metric)][row] / denominator)
                    ratios[metric] = float(np.nanmax(np.concatenate(values)))
                fw.append({
                    "episode_index": eid,
                    "frame_index": int(episode["frame_index"][row]),
                    "index": int(episode["index"][row]),
                    "failed_state_dimensions": failed_state,
                    "failed_action_dimensions": failed_action,
                    "max_residual_ratio": ratios["residual"],
                    "max_acceleration_ratio": ratios["acceleration"],
                    "max_jerk_ratio": ratios["jerk"],
                    "flagged_frame": True,
                })
            for row in range(n):
                reasons: list[str] = []
                if union[row]:
                    reasons.append("stage1_sudden_change")
                if reject_episode:
                    reasons.append("stage1_episode_rejected")
                validity_writer.append({
                    "episode_index": eid,
                    "frame_index": int(episode["frame_index"][row]),
                    "index": int(episode["index"][row]),
                    "stage1_valid": bool(frame_valid[row]),
                    "stage2_valid": True,
                    "stage3_valid": True,
                    "valid": bool(frame_valid[row]),
                    "reason_codes": reasons,
                })
            ew.append({
                "episode_index": eid, "num_frames": n, "flagged_frames": len(indices),
                "valid_frames": int(frame_valid.sum()),
                "flag_fraction": float(len(indices) / max(n, 1)),
                "exclusion_policy": exclusion, "reject_episode": reject_episode,
            })
            filter_writer.append({
                "episode_index": eid, "num_frames": n, "accepted": not reject_episode,
                "reason_code": "stage1_sudden_change" if reject_episode else None,
            })
            if not reject_episode:
                output_episodes += 1
                output_frames += int(frame_valid.sum())
    with ParquetRows(labels_dir / "thresholds.parquet", THRESHOLD_SCHEMA) as tw:
        for key, dim in ((ds.state_key, state_dim), (ds.action_key, action_dim)):
            names = flattened_names(ds.info["features"].get(key), dim)
            for metric in ("residual", "acceleration", "jerk"):
                for index, value in enumerate(thresholds[(key, metric)]):
                    tw.append({
                        "feature": key, "dimension": index,
                        "name": names[index], "metric": metric,
                        "threshold": float(value), "exempt_gripper": False,
                    })
    result = {
        "paper": PAPER, "stage": 1, "dataset_id": ds.dataset_id, "source": str(ds.path),
        "started_at": started, "finished_at": utc_now(), "max_episodes": max_episodes,
        "calibration_episodes": calibration_episodes, "processed_frames": frame_count,
        "flagged_frames": flagged_frames, "rejected_episodes": rejected_episodes,
        "output_episodes": output_episodes, "output_frames": output_frames,
        "invalid_frames": frame_count - output_frames,
        "exclusion_policy": ds.cfg["stage1_exclusion"], "config": ds.cfg,
    }
    return result


def normalize_joint_name(name: str | None) -> str | None:
    if not name:
        return None
    value = name.lower()
    compact = "".join(char for char in value if char.isalnum())
    if compact.startswith(("state", "action")) and compact.removeprefix("state").removeprefix("action").isdigit():
        # Generic positional labels such as state_0/action_0 carry no semantic
        # evidence that both columns describe the same physical quantity.
        return None
    for token in ("observation", "state", "action", "target", "command", "position", "pos"):
        value = value.replace(token, "")
    value = "".join(char for char in value if char.isalnum())
    if value.isdigit():
        return None
    return value or None


def state_action_map(ds: Dataset, state_dim: int, action_dim: int) -> list[tuple[int, int]]:
    explicit = ds.cfg.get("state_action_map")
    if explicit is not None:
        pairs = [(int(pair[0]), int(pair[1])) for pair in explicit]
        if any(s < 0 or s >= state_dim or a < 0 or a >= action_dim for s, a in pairs):
            raise ValueError(f"{ds.dataset_id}: state_action_map index is outside feature dimensions")
        if len({s for s, _ in pairs}) != len(pairs) or len({a for _, a in pairs}) != len(pairs):
            raise ValueError(f"{ds.dataset_id}: state_action_map must be one-to-one")
    else:
        state_names = flattened_names(ds.info["features"].get(ds.state_key), state_dim)
        action_names = flattened_names(ds.info["features"].get(ds.action_key), action_dim)
        action_candidates: dict[str, list[int]] = {}
        for i, name in enumerate(action_names):
            normalized = normalize_joint_name(name)
            if normalized:
                action_candidates.setdefault(normalized, []).append(i)
        state_counts: dict[str, int] = {}
        for name in state_names:
            normalized = normalize_joint_name(name)
            if normalized:
                state_counts[normalized] = state_counts.get(normalized, 0) + 1
        pairs = []
        for i, name in enumerate(state_names):
            normalized = normalize_joint_name(name)
            if normalized and state_counts[normalized] == 1 and len(action_candidates.get(normalized, [])) == 1:
                pairs.append((i, action_candidates[normalized][0]))
        if not pairs and ds.cfg.get("allow_positional_mapping"):
            pairs = [(i, i) for i in range(min(state_dim, action_dim))]
    state_gripper = gripper_indices(ds, ds.state_key, state_dim)
    action_gripper = gripper_indices(ds, ds.action_key, action_dim)
    return [(s, a) for s, a in pairs if s not in state_gripper and a not in action_gripper]


def aligned(a: np.ndarray, s: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return a[:-lag], s[lag:]
    if lag < 0:
        return a[-lag:], s[:lag]
    return a, s


def correlation(a: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(s)
    if ok.sum() < 3:
        return float("-inf")
    a = a[ok] - np.mean(a[ok])
    s = s[ok] - np.mean(s[ok])
    denom = float(np.linalg.norm(a) * np.linalg.norm(s))
    return float(np.dot(a, s) / denom) if denom > 1e-15 else float("-inf")


def trend_metric(state: np.ndarray, action: np.ndarray, max_lag: int, min_active: int) -> dict[str, Any]:
    causal_scores = []
    for lag in range(max_lag + 1):
        aa, ss = aligned(action, state, lag)
        causal_scores.append((correlation(aa, ss), lag))
    best_corr, best_lag = max(causal_scores)
    all_scores = []
    for lag in range(-max_lag, max_lag + 1):
        aa, ss = aligned(action, state, lag)
        all_scores.append((correlation(aa, ss), lag))
    unconstrained_corr, unconstrained_lag = max(all_scores)
    aa, ss = aligned(action, state, best_lag)
    da = np.diff(aa)
    ds = np.diff(ss)
    scale_a = max(float(np.nanquantile(np.abs(da), 0.9)) if len(da) else 0.0, 1e-10)
    scale_s = max(float(np.nanquantile(np.abs(ds), 0.9)) if len(ds) else 0.0, 1e-10)
    active = np.isfinite(da) & np.isfinite(ds) & (np.abs(da) > scale_a * 1e-3) & (np.abs(ds) > scale_s * 1e-3)
    count = int(active.sum())
    agreement = float(np.mean(da[active] * ds[active] > 0)) if count >= min_active else None
    return {
        "lag": int(best_lag), "correlation": None if not np.isfinite(best_corr) else float(best_corr),
        "unconstrained_lag": int(unconstrained_lag),
        "unconstrained_correlation": None if not np.isfinite(unconstrained_corr) else float(unconstrained_corr),
        "active_steps": count, "directional_agreement": agreement,
    }


def contiguous_valid_slices(step_valid: np.ndarray, minimum_length: int = 1) -> list[slice]:
    """Return runs of valid source frames without joining across invalid gaps."""
    valid = np.asarray(step_valid, dtype=bool)
    padded = np.r_[False, valid, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops)
            if stop - start >= minimum_length]


def stage2_signal_segments(
    state: np.ndarray,
    action: np.ndarray,
    step_valid: np.ndarray,
    cfg: dict[str, Any],
    state_key: str,
    action_key: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Prepare Stage 2 signals while preserving physical time semantics.

    Delta actions are integrated before applying the validity mask. Each valid
    run is then normalized and smoothed independently, so neither smoothing nor
    finite differences bridge an invalid interval.
    """
    raw_state = np.asarray(state, dtype=np.float64)
    raw_action = np.asarray(action, dtype=np.float64)
    if cfg["action_mode"] == "delta":
        raw_action = np.cumsum(fill_nonfinite(raw_action), axis=0)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for valid_slice in contiguous_valid_slices(step_valid, minimum_length=3):
        state_segment = smooth(normalized_signal(raw_state[valid_slice], cfg, state_key), cfg)
        action_segment = smooth(normalized_signal(raw_action[valid_slice], cfg, action_key), cfg)
        segments.append((state_segment, action_segment))
    return segments


def trend_metric_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    state_index: int,
    action_index: int,
    max_lag: int,
    min_active: int,
) -> dict[str, Any]:
    """Score valid runs independently and aggregate DA by active-step count."""
    metrics: list[tuple[dict[str, Any], int]] = []
    for state, action in segments:
        segment_max_lag = min(max_lag, len(state) - 2)
        if segment_max_lag < 0:
            continue
        metric = trend_metric(
            state[:, state_index], action[:, action_index], segment_max_lag, min_active=1
        )
        weight = int(metric["active_steps"])
        metrics.append((metric, weight))
    active_steps = sum(weight for _, weight in metrics)
    scored = [(metric, weight) for metric, weight in metrics
              if weight > 0 and metric["directional_agreement"] is not None]
    if scored:
        agreement = sum(float(metric["directional_agreement"]) * weight
                        for metric, weight in scored) / sum(weight for _, weight in scored)
        lag = int(round(sum(int(metric["lag"]) * weight for metric, weight in scored)
                        / sum(weight for _, weight in scored)))
        correlation_values = [(float(metric["correlation"]), weight) for metric, weight in scored
                              if metric["correlation"] is not None]
        correlation_value = (sum(value * weight for value, weight in correlation_values)
                             / sum(weight for _, weight in correlation_values)) if correlation_values else None
        unconstrained_lag = int(round(
            sum(int(metric["unconstrained_lag"]) * weight for metric, weight in scored)
            / sum(weight for _, weight in scored)
        ))
    else:
        agreement = correlation_value = None
        lag = unconstrained_lag = 0
    if active_steps < min_active:
        agreement = None
    return {
        "lag": lag,
        "correlation": correlation_value,
        "unconstrained_lag": unconstrained_lag,
        "active_steps": active_steps,
        "directional_agreement": agreement,
    }


STAGE2_DIM_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("state_dimension", pa.int32()), ("action_dimension", pa.int32()),
    ("lag_frames", pa.int32()), ("lag_seconds", pa.float64()), ("correlation", pa.float64()),
    ("unconstrained_lag_frames", pa.int32()), ("active_steps", pa.int64()),
    ("directional_agreement", pa.float64()), ("failed", pa.bool_()),
])
STAGE2_EPISODE_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("scored_dimensions", pa.int32()),
    ("failed_dimensions", pa.list_(pa.int32())), ("minimum_da", pa.float64()),
    ("reject_episode", pa.bool_()),
])


def run_stage2(
    ds: Dataset, local_dir: pathlib.Path, max_episodes: int | None
) -> dict[str, Any]:
    started = utc_now()
    labels_dir = local_dir / "labels"
    state_dim = feature_dim(ds.info, ds.state_key)
    action_dim = feature_dim(ds.info, ds.action_key)
    pairs = state_action_map(ds, state_dim, action_dim)
    if not pairs:
        raise ValueError(
            f"{ds.dataset_id}: no trustworthy comparable state/action dimensions; set state_action_map "
            "or allow_positional_mapping in config"
        )
    max_lag = max(0, int(round(float(ds.cfg["stage2_max_lag_seconds"]) * ds.fps)))
    threshold = float(ds.cfg["stage2_da_threshold"])
    min_active = int(ds.cfg["stage2_min_active_steps"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    processed = rejected = unscored = processed_frames = valid_input_frames = output_frames = 0
    with ParquetRows(labels_dir / "dimension_metrics.parquet", STAGE2_DIM_SCHEMA) as dw, ParquetRows(
        labels_dir / "episode_flags.parquet", STAGE2_EPISODE_SCHEMA
    ) as ew, ParquetRows(labels_dir / "episode_filter.parquet", EPISODE_FILTER_SCHEMA) as filter_writer:
        for episode in iter_episodes(ds, [ds.state_key, ds.action_key], max_episodes):
            eid = int(episode["episode_index"][0])
            num_frames = len(episode["frame_index"])
            step_valid = np.asarray(episode["_step_valid"], dtype=bool)
            num_valid_frames = int(step_valid.sum())
            processed += 1
            processed_frames += num_frames
            valid_input_frames += num_valid_frames
            segments = stage2_signal_segments(
                episode[ds.state_key], episode[ds.action_key], step_valid,
                ds.cfg, ds.state_key, ds.action_key,
            )
            failed: list[int] = []
            scores: list[float] = []
            for pair_index, (state_index, action_index) in enumerate(pairs):
                metric = trend_metric_segments(
                    segments, state_index, action_index, max_lag, min_active,
                )
                da = metric["directional_agreement"]
                is_failed = bool(da is not None and da < threshold)
                if da is None:
                    unscored += 1
                else:
                    scores.append(float(da))
                if is_failed:
                    failed.append(pair_index)
                dw.append({
                    "episode_index": eid, "state_dimension": state_index, "action_dimension": action_index,
                    "lag_frames": metric["lag"], "lag_seconds": metric["lag"] / ds.fps,
                    "correlation": metric["correlation"],
                    "unconstrained_lag_frames": metric["unconstrained_lag"],
                    "active_steps": metric["active_steps"], "directional_agreement": da, "failed": is_failed,
                })
            reject_episode = bool(failed)
            rejected += int(reject_episode)
            ew.append({
                "episode_index": eid, "scored_dimensions": len(scores), "failed_dimensions": failed,
                "minimum_da": min(scores) if scores else None, "reject_episode": reject_episode,
            })
            filter_writer.append({
                "episode_index": eid, "num_frames": num_frames, "accepted": not reject_episode,
                "reason_code": "state_action_trend_mismatch" if reject_episode else None,
            })
            if not reject_episode:
                output_frames += num_valid_frames
    result = {
        "paper": PAPER, "stage": 2, "dataset_id": ds.dataset_id, "source": str(ds.path),
        "started_at": started, "finished_at": utc_now(), "max_episodes": max_episodes,
        "mapping": [{"state": s, "action": a} for s, a in pairs],
        "processed_episodes": processed, "rejected_episodes": rejected,
        "processed_frames": processed_frames, "output_episodes": processed - rejected,
        "valid_input_frames": valid_input_frames, "output_frames": output_frames,
        "unscored_dimension_episodes": unscored,
        "config": ds.cfg,
    }
    return result


def safe_id(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


STAGE3_FRAME_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("frame_index", pa.int64()), ("index", pa.int64()),
    ("failed_state_dimensions", pa.list_(pa.int32())), ("failed_action_dimensions", pa.list_(pa.int32())),
    ("flagged_frame", pa.bool_()),
])
STAGE3_EPISODE_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("input_frames", pa.int64()),
    ("flagged_frames", pa.int64()), ("output_frames", pa.int64()),
])
STAGE3_THRESHOLD_SCHEMA = pa.schema([
    ("embodiment", pa.string()), ("feature", pa.string()), ("dimension", pa.int32()),
    ("name", pa.string()), ("q01", pa.float64()), ("q99", pa.float64()),
    ("lower", pa.float64()), ("upper", pa.float64()), ("alpha", pa.float64()),
    ("exempt_gripper", pa.bool_()),
])


def validate_group(group: list[Dataset]) -> tuple[int, int]:
    signatures = {
        (
            feature_dim(ds.info, ds.state_key),
            feature_dim(ds.info, ds.action_key),
            ds.state_key,
            ds.action_key,
            tuple(flattened_names(ds.info["features"].get(ds.state_key), feature_dim(ds.info, ds.state_key))),
            tuple(flattened_names(ds.info["features"].get(ds.action_key), feature_dim(ds.info, ds.action_key))),
        )
        for ds in group
    }
    if len(signatures) != 1:
        raise ValueError(f"embodiment group has incompatible signal schemas: {signatures}")
    state_dim, action_dim, _, _, _, _ = next(iter(signatures))
    return state_dim, action_dim


def calibrate_stage3(
    group: list[Dataset], max_episodes: int | None
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    state_dim, action_dim = validate_group(group)
    total = sum(int(ds.info.get("total_frames") or 0) for ds in group)
    capacity = max(int(ds.cfg["calibration_samples"]) for ds in group)
    collectors = {
        group[0].state_key: StrideSamples(total, capacity),
        group[0].action_key: StrideSamples(total, capacity),
    }
    for ds in group:
        for episode in iter_episodes(ds, [ds.state_key, ds.action_key], max_episodes):
            valid = np.asarray(episode["_step_valid"], dtype=bool)
            if valid.any():
                collectors[group[0].state_key].add(episode[ds.state_key][valid])
                collectors[group[0].action_key].add(episode[ds.action_key][valid])
    alpha = float(group[0].cfg["stage3_alpha"])
    result = {}
    for key in (group[0].state_key, group[0].action_key):
        values = collectors[key].array()
        q01 = np.nanquantile(values, 0.01, axis=0)
        q99 = np.nanquantile(values, 0.99, axis=0)
        width = q99 - q01
        result[key] = (q01, q99, q01 - alpha * width, q99 + alpha * width)
    return result


def run_stage3_dataset(
    ds: Dataset,
    thresholds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    local_dir: pathlib.Path,
    max_episodes: int | None,
) -> dict[str, Any]:
    started = utc_now()
    labels_dir = local_dir / "labels"
    state_dim = feature_dim(ds.info, ds.state_key)
    action_dim = feature_dim(ds.info, ds.action_key)
    labels_dir.mkdir(parents=True, exist_ok=True)
    processed = flagged = processed_frames = output_frames = 0
    with ParquetRows(labels_dir / "frame_flags.parquet", STAGE3_FRAME_SCHEMA) as fw, ParquetRows(
        labels_dir / "episode_summary.parquet", STAGE3_EPISODE_SCHEMA
    ) as ew, ParquetRows(labels_dir / "episode_filter.parquet", EPISODE_FILTER_SCHEMA) as filter_writer, ParquetRows(
        labels_dir / "step_validity.parquet", STEP_VALIDITY_SCHEMA
    ) as validity_writer:
        for episode in iter_episodes(ds, [ds.state_key, ds.action_key], max_episodes):
            eid = int(episode["episode_index"][0])
            n = len(episode["frame_index"])
            stage1_valid = np.asarray(episode["_step_valid"], dtype=bool)
            processed += 1
            processed_frames += n
            feature_flags: dict[str, np.ndarray] = {}
            for key, dim in ((ds.state_key, state_dim), (ds.action_key, action_dim)):
                _, _, lower, upper = thresholds[key]
                values = np.asarray(episode[key], dtype=np.float64)
                flags = (~np.isfinite(values)) | (values < lower) | (values > upper)
                for index in gripper_indices(ds, key, dim):
                    if 0 <= index < dim:
                        flags[:, index] = False
                flags |= ~np.isfinite(values)
                flags &= stage1_valid[:, None]
                feature_flags[key] = flags
            union = feature_flags[ds.state_key].any(axis=1) | feature_flags[ds.action_key].any(axis=1)
            rows = np.flatnonzero(union)
            flagged += len(rows)
            for row in rows:
                failed_state = np.flatnonzero(feature_flags[ds.state_key][row]).astype(int).tolist()
                failed_action = np.flatnonzero(feature_flags[ds.action_key][row]).astype(int).tolist()
                fw.append({
                    "episode_index": eid, "frame_index": int(episode["frame_index"][row]),
                    "index": int(episode["index"][row]),
                    "failed_state_dimensions": failed_state,
                    "failed_action_dimensions": failed_action,
                    "flagged_frame": True,
                })
            stage3_valid = ~union
            final_valid = stage1_valid & stage3_valid
            output_frames += int(final_valid.sum())
            for row in range(n):
                reasons: list[str] = []
                if not stage1_valid[row]:
                    reasons.append("stage1_invalid")
                if union[row]:
                    reasons.append("stage3_extreme_value")
                validity_writer.append({
                    "episode_index": eid,
                    "frame_index": int(episode["frame_index"][row]),
                    "index": int(episode["index"][row]),
                    "stage1_valid": bool(stage1_valid[row]),
                    "stage2_valid": True,
                    "stage3_valid": bool(stage3_valid[row]),
                    "valid": bool(final_valid[row]),
                    "reason_codes": reasons,
                })
            ew.append({
                "episode_index": eid, "input_frames": n,
                "flagged_frames": len(rows), "output_frames": int(final_valid.sum()),
            })
            filter_writer.append({
                "episode_index": eid, "num_frames": n, "accepted": True, "reason_code": None,
            })
    with ParquetRows(labels_dir / "thresholds.parquet", STAGE3_THRESHOLD_SCHEMA) as tw:
        for key, dim in ((ds.state_key, state_dim), (ds.action_key, action_dim)):
            q01, q99, lower, upper = thresholds[key]
            names = flattened_names(ds.info["features"].get(key), dim)
            exempt = gripper_indices(ds, key, dim)
            for index in range(dim):
                tw.append({
                    "embodiment": str(ds.cfg["embodiment"]), "feature": key, "dimension": index,
                    "name": names[index], "q01": float(q01[index]), "q99": float(q99[index]),
                    "lower": float(lower[index]), "upper": float(upper[index]),
                    "alpha": float(ds.cfg["stage3_alpha"]), "exempt_gripper": index in exempt,
                })
    result = {
        "paper": PAPER, "stage": 3, "dataset_id": ds.dataset_id,
        "embodiment": ds.cfg["embodiment"], "source": str(ds.path), "started_at": started,
        "finished_at": utc_now(), "max_episodes": max_episodes, "processed_episodes": processed,
        "processed_frames": processed_frames, "flagged_frames": flagged,
        "output_episodes": processed, "output_frames": output_frames,
        "invalid_frames": processed_frames - output_frames, "rejected_episodes": 0,
        "config": ds.cfg,
    }
    return result


def select_datasets(items: list[Dataset], patterns: list[str]) -> list[Dataset]:
    if not patterns:
        return items
    selected = [ds for ds in items if any(fnmatch.fnmatch(ds.dataset_id, pattern) for pattern in patterns)]
    missing = [pattern for pattern in patterns if not any(fnmatch.fnmatch(ds.dataset_id, pattern) for ds in items)]
    if missing:
        raise ValueError(f"dataset patterns matched nothing: {missing}")
    return selected


def work_stage(work_root: pathlib.Path, stage: int, ds: Dataset) -> pathlib.Path:
    return pathlib.Path(tempfile.mkdtemp(prefix=f"{safe_id(ds.dataset_id)}-", dir=work_root / STAGE_NAMES[stage]))


def write_filter_manifest(
    local_dir: pathlib.Path,
    ds: Dataset,
    stage: int,
    result: dict[str, Any],
) -> None:
    labels = sorted(path.relative_to(local_dir).as_posix() for path in (local_dir / "labels").glob("*.parquet"))
    manifest = {
        "format": "vla_curation_filter",
        "schema_version": 2,
        "stage_id": stage,
        "stage": f"Stage {stage}",
        "detector_version": "qwen_robotmanip_curation-filter-v2",
        "coordinate_system": "episode_frame",
        "dataset_id": ds.dataset_id,
        "source_dataset": str(ds.path.resolve()),
        "data_source": str((ds.path / "data").resolve()),
        "video_source": str((ds.path / "videos").resolve()),
        "video_policy": "reference_original",
        "parent_manifest": str(ds.manifest_path) if ds.manifest_path else None,
        "episode_filter": "labels/episode_filter.parquet",
        "validity_files": ["labels/step_validity.parquet"] if stage in (1, 3) else [],
        "repair_files": [],
        "label_files": labels,
        "input_format": "lerobot_v3.0" if ds.manifest_path is None else "lerobot_v3.0-filter-manifest",
        "output_format": "lerobot_v3.0-filter-manifest",
        "config": ds.cfg,
        "result": result,
        "created_at": utc_now(),
    }
    write_json(local_dir / "manifest.json", manifest)


def previous_filter(
    ds: Dataset, output_root: pathlib.Path, stages: Iterable[int], config: dict[str, Any]
) -> Dataset:
    if ds.manifest_path is not None:
        return ds
    for stage in stages:
        manifest = output_root / STAGE_NAMES[stage] / safe_id(ds.dataset_id) / "manifest.json"
        if manifest.is_file():
            return dataset_from_manifest(manifest, config, ds.dataset_id)
    return ds


def execute(
    datasets: list[Dataset], stages: list[int], output_root: pathlib.Path, work_root: pathlib.Path,
    max_episodes: int | None, overwrite: bool,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        (work_root / STAGE_NAMES[stage]).mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    current = {ds.dataset_id: ds for ds in datasets}
    if 1 in stages:
        for ds in datasets:
            local = work_stage(work_root, 1, ds)
            result = run_stage1(ds, local, max_episodes)
            result.update({"input_format": "lerobot_v3.0", "output_format": "lerobot_v3.0-filter-manifest"})
            write_filter_manifest(local, ds, 1, result)
            destination = output_root / STAGE_NAMES[1] / safe_id(ds.dataset_id)
            publish(local, destination, overwrite)
            shutil.rmtree(local)
            current[ds.dataset_id] = dataset_from_manifest(destination / "manifest.json", {"defaults": ds.cfg, "datasets": {}}, ds.dataset_id)
            print(json.dumps(result, ensure_ascii=False))
            results.append(result)
    if 2 in stages:
        for original in datasets:
            ds = current[original.dataset_id]
            ds = previous_filter(ds, output_root, (1,), {"defaults": original.cfg, "datasets": {}})
            local = work_stage(work_root, 2, ds)
            result = run_stage2(ds, local, max_episodes)
            result.update({
                "input_format": "lerobot_v3.0" if ds.manifest_path is None else "lerobot_v3.0-filter-manifest",
                "output_format": "lerobot_v3.0-filter-manifest",
            })
            write_filter_manifest(local, ds, 2, result)
            destination = output_root / STAGE_NAMES[2] / safe_id(ds.dataset_id)
            publish(local, destination, overwrite)
            shutil.rmtree(local)
            current[ds.dataset_id] = dataset_from_manifest(destination / "manifest.json", {"defaults": ds.cfg, "datasets": {}}, ds.dataset_id)
            print(json.dumps(result, ensure_ascii=False))
            results.append(result)
    if 3 in stages:
        groups: dict[str, list[Dataset]] = {}
        for original in datasets:
            ds = current[original.dataset_id]
            ds = previous_filter(ds, output_root, (2, 1), {"defaults": original.cfg, "datasets": {}})
            groups.setdefault(str(ds.cfg["embodiment"]), []).append(ds)
        for embodiment, group in groups.items():
            alphas = {float(ds.cfg["stage3_alpha"]) for ds in group}
            if len(alphas) != 1:
                raise ValueError(f"embodiment {embodiment}: stage3_alpha differs across datasets")
            thresholds = calibrate_stage3(group, max_episodes)
            for ds in group:
                local = work_stage(work_root, 3, ds)
                result = run_stage3_dataset(ds, thresholds, local, max_episodes)
                result.update({
                    "input_format": "lerobot_v3.0" if ds.manifest_path is None else "lerobot_v3.0-filter-manifest",
                    "output_format": "lerobot_v3.0-filter-manifest",
                })
                write_filter_manifest(local, ds, 3, result)
                destination = output_root / STAGE_NAMES[3] / safe_id(ds.dataset_id)
                publish(local, destination, overwrite)
                shutil.rmtree(local)
                current[ds.dataset_id] = dataset_from_manifest(destination / "manifest.json", {"defaults": ds.cfg, "datasets": {}}, ds.dataset_id)
                print(json.dumps(result, ensure_ascii=False))
                results.append(result)
    write_json(output_root / "latest_run.json", {
        "paper": PAPER, "finished_at": utc_now(), "datasets": [ds.dataset_id for ds in datasets],
        "stages": stages, "max_episodes": max_episodes, "results": results,
    })
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="list completed datasets and signal compatibility")
    inv.add_argument("--input-root", type=pathlib.Path, required=True)
    inv.add_argument("--output", type=pathlib.Path)
    inv.add_argument("--discovery-depth", type=int, default=2)
    run = sub.add_parser("run", help="run one or more stages")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-root", type=pathlib.Path)
    source.add_argument("--dataset-path", type=pathlib.Path)
    run.add_argument("--dataset-id")
    run.add_argument("--select", action="append", default=[], help="dataset id glob; repeatable")
    run.add_argument("--discovery-depth", type=int, default=2)
    run.add_argument("--stages", default="1,2,3")
    run.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"stage output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    run.add_argument("--work-root", type=pathlib.Path, required=True)
    run.add_argument("--max-episodes", type=int)
    run.add_argument("--overwrite", action="store_true")
    return parser


def inventory_rows(datasets: list[Dataset]) -> list[dict[str, Any]]:
    rows = []
    for ds in datasets:
        try:
            state_dim = feature_dim(ds.info, ds.state_key)
            action_dim = feature_dim(ds.info, ds.action_key)
            pairs = state_action_map(ds, state_dim, action_dim)
            status = "ready" if pairs else "stage2_mapping_required"
        except (KeyError, ValueError) as exc:
            state_dim = action_dim = None
            pairs = []
            status = f"unsupported: {exc}"
        rows.append({
            "dataset_id": ds.dataset_id, "path": str(ds.path), "status": status,
            "episodes": ds.info.get("total_episodes"), "frames": ds.info.get("total_frames"),
            "fps": ds.info.get("fps"), "state_dim": state_dim, "action_dim": action_dim,
            "mapped_joint_dimensions": len(pairs), "embodiment": ds.cfg["embodiment"],
        })
    return rows


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "inventory":
        rows = inventory_rows(discover(args.input_root, config, args.discovery_depth))
        payload = {
            "created_at": utc_now(), "input_root": str(args.input_root),
            "format": "lerobot_v3.0", "datasets": rows,
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")
        print(encoded)
        return
    if args.dataset_path:
        datasets = [load_dataset(args.dataset_path, config, args.dataset_id)]
    else:
        datasets = select_datasets(discover(args.input_root, config, args.discovery_depth), args.select)
    stages = [int(value) for value in args.stages.split(",")]
    if any(stage not in STAGE_NAMES for stage in stages):
        raise ValueError("--stages must contain only 1,2,3")
    execute(datasets, stages, args.output_root, args.work_root, args.max_episodes, args.overwrite)


if __name__ == "__main__":
    main()
