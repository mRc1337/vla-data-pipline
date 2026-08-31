#!/usr/bin/env python3
"""Qwen-RobotManip stages 1-3 for LeRobot v3 parquet signals.

Outputs are sparse rejection manifests. Source parquet/video files are never changed.
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

try:
    from .canonical_layout import ACTION_JOINT_INDICES, LAYOUT, STATE_JOINT_INDICES
except ImportError:  # direct ``python path/to/curate.py`` execution
    from canonical_layout import ACTION_JOINT_INDICES, LAYOUT, STATE_JOINT_INDICES


PAPER = "https://arxiv.org/abs/2606.17846"
STAGE_NAMES = {
    1: "stage1_sudden_change",
    2: "stage2_trend_alignment",
    3: "stage3_extreme_value",
}
DEFAULTS: dict[str, Any] = {
    "state_key": "observation.state",
    "action_key": "action",
    "embodiment": None,
    "action_mode": "absolute",
    "stage1_policy": "frame",
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
    "canonical_indices": {"observation.state": None, "action": None},
    "state_action_map": None,
}


@dataclass(frozen=True)
class Dataset:
    path: pathlib.Path
    dataset_id: str
    info: dict[str, Any]
    cfg: dict[str, Any]

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
    if cfg["stage1_policy"] not in {"frame", "episode"}:
        raise ValueError(f"{dataset_id}: stage1_policy must be frame or episode")
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
    return Dataset(path, dataset_id, json.loads(info_path.read_text()), dataset_config(config, dataset_id))


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
        return {key: np.concatenate(value, axis=0) for key, value in parts.items()}

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
                    yield emit(pending)
                    yielded += 1
                    if max_episodes is not None and yielded >= max_episodes:
                        return
                    pending_episode = episode
                    pending = {key: [value] for key, value in piece.items()}
    if pending is not None and (max_episodes is None or yielded < max_episodes):
        yield emit(pending)


def canonical_signal(x: np.ndarray, cfg: dict[str, Any], key: str) -> np.ndarray:
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
    raw = canonical_signal(x, cfg, key)
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


def canonical_indices(ds: Dataset, key: str, dim: int) -> list[int | None]:
    """Return raw-column -> canonical-column mapping, validating the DOCX contract."""
    configured = ds.cfg.get("canonical_indices", {}).get(key)
    if configured is None:
        return [None] * dim
    if len(configured) != dim:
        raise ValueError(
            f"{ds.dataset_id}: canonical_indices[{key!r}] has {len(configured)} entries, expected {dim}"
        )
    limit = int(LAYOUT["state" if key == ds.state_key else "action"]["dimension"])
    mapped = [None if value is None else int(value) for value in configured]
    active = [value for value in mapped if value is not None]
    if any(value < 0 or value >= limit for value in active):
        raise ValueError(f"{ds.dataset_id}: canonical index outside [0,{limit}) for {key}")
    if len(active) != len(set(active)):
        raise ValueError(f"{ds.dataset_id}: duplicate canonical index for {key}")
    if key == ds.action_key and 68 in active:
        raise ValueError(f"{ds.dataset_id}: action canonical index 68 is reserved ARM2 padding")
    return mapped


def map_failed_dimensions(raw_dimensions: list[int], mapping: list[int | None]) -> list[int]:
    return [mapping[index] for index in raw_dimensions if mapping[index] is not None]


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
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in local_dir.iterdir():
        target = output_dir / source.name
        partial = output_dir / f".{source.name}.partial"
        if partial.exists():
            partial.unlink()
        shutil.copy2(source, partial)
        os.replace(partial, target)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


STAGE1_FRAME_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("frame_index", pa.int64()), ("index", pa.int64()),
    ("failed_state_dimensions", pa.list_(pa.int32())), ("failed_action_dimensions", pa.list_(pa.int32())),
    ("failed_state_canonical_dimensions", pa.list_(pa.int32())),
    ("failed_action_canonical_dimensions", pa.list_(pa.int32())),
    ("max_residual_ratio", pa.float64()), ("max_acceleration_ratio", pa.float64()),
    ("max_jerk_ratio", pa.float64()), ("reject_frame", pa.bool_()),
])
STAGE1_EPISODE_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("num_frames", pa.int64()), ("flagged_frames", pa.int64()),
    ("flag_fraction", pa.float64()), ("reject_episode", pa.bool_()),
])
THRESHOLD_SCHEMA = pa.schema([
    ("feature", pa.string()), ("dimension", pa.int32()), ("canonical_dimension", pa.int32()), ("name", pa.string()),
    ("metric", pa.string()), ("threshold", pa.float64()), ("exempt_gripper", pa.bool_()),
])


def run_stage1(ds: Dataset, local_dir: pathlib.Path, max_episodes: int | None) -> dict[str, Any]:
    started = utc_now()
    state_dim = feature_dim(ds.info, ds.state_key)
    action_dim = feature_dim(ds.info, ds.action_key)
    state_canonical = canonical_indices(ds, ds.state_key, state_dim)
    action_canonical = canonical_indices(ds, ds.action_key, action_dim)
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

    local_dir.mkdir(parents=True, exist_ok=True)
    flagged_frames = 0
    rejected_episodes = 0
    frame_count = 0
    with ParquetRows(local_dir / "frame_flags.parquet", STAGE1_FRAME_SCHEMA) as fw, ParquetRows(
        local_dir / "episode_summary.parquet", STAGE1_EPISODE_SCHEMA
    ) as ew:
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
                for index in gripper_indices(ds, key, dim):
                    if 0 <= index < dim:
                        flags[:, index] = False
                feature_flags[key] = flags
            union = feature_flags[ds.state_key].any(axis=1) | feature_flags[ds.action_key].any(axis=1)
            indices = np.flatnonzero(union)
            flagged_frames += len(indices)
            reject_episode = bool(len(indices) and ds.cfg["stage1_policy"] == "episode")
            rejected_episodes += int(reject_episode)
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
                    "failed_state_canonical_dimensions": map_failed_dimensions(failed_state, state_canonical),
                    "failed_action_canonical_dimensions": map_failed_dimensions(failed_action, action_canonical),
                    "max_residual_ratio": ratios["residual"],
                    "max_acceleration_ratio": ratios["acceleration"],
                    "max_jerk_ratio": ratios["jerk"],
                    "reject_frame": True,
                })
            ew.append({
                "episode_index": eid, "num_frames": n, "flagged_frames": len(indices),
                "flag_fraction": float(len(indices) / max(n, 1)), "reject_episode": reject_episode,
            })
    with ParquetRows(local_dir / "thresholds.parquet", THRESHOLD_SCHEMA) as tw:
        for key, dim in ((ds.state_key, state_dim), (ds.action_key, action_dim)):
            names = flattened_names(ds.info["features"].get(key), dim)
            exempt = gripper_indices(ds, key, dim)
            for metric in ("residual", "acceleration", "jerk"):
                for index, value in enumerate(thresholds[(key, metric)]):
                    tw.append({
                        "feature": key, "dimension": index,
                        "canonical_dimension": (state_canonical if key == ds.state_key else action_canonical)[index],
                        "name": names[index], "metric": metric,
                        "threshold": float(value), "exempt_gripper": index in exempt,
                    })
    result = {
        "paper": PAPER, "stage": 1, "dataset_id": ds.dataset_id, "source": str(ds.path),
        "started_at": started, "finished_at": utc_now(), "max_episodes": max_episodes,
        "calibration_episodes": calibration_episodes, "processed_frames": frame_count,
        "flagged_frames": flagged_frames, "rejected_episodes": rejected_episodes,
        "policy": ds.cfg["stage1_policy"], "config": ds.cfg,
    }
    write_json(local_dir / "run.json", result)
    return result


def normalize_joint_name(name: str | None) -> str | None:
    if not name:
        return None
    value = name.lower()
    for token in ("observation", "state", "action", "target", "command", "position", "pos"):
        value = value.replace(token, "")
    value = "".join(char for char in value if char.isalnum())
    return value or None


def state_action_map(ds: Dataset, state_dim: int, action_dim: int) -> list[tuple[int, int]]:
    explicit = ds.cfg.get("state_action_map")
    if explicit is not None:
        return [(int(pair[0]), int(pair[1])) for pair in explicit]
    state_names = flattened_names(ds.info["features"].get(ds.state_key), state_dim)
    action_names = flattened_names(ds.info["features"].get(ds.action_key), action_dim)
    action_lookup: dict[str, int] = {}
    for i, name in enumerate(action_names):
        normalized = normalize_joint_name(name)
        if normalized:
            action_lookup[normalized] = i
    pairs = []
    for i, name in enumerate(state_names):
        normalized = normalize_joint_name(name)
        if normalized in action_lookup:
            pairs.append((i, action_lookup[normalized]))
    if not pairs and ds.cfg.get("allow_positional_mapping"):
        pairs = [(i, i) for i in range(min(state_dim, action_dim))]
    state_canonical = canonical_indices(ds, ds.state_key, state_dim)
    action_canonical = canonical_indices(ds, ds.action_key, action_dim)
    if any(value is not None for value in state_canonical + action_canonical):
        # The paper's Stage 2 is a shared-joint check, not a generic
        # same-shaped-vector check. Canonical slots make that restriction
        # explicit even when raw names happen to match.
        pairs = [
            (s, a) for s, a in pairs
            if state_canonical[s] in STATE_JOINT_INDICES
            and action_canonical[a] in ACTION_JOINT_INDICES
        ]
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


STAGE2_DIM_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("state_dimension", pa.int32()), ("action_dimension", pa.int32()),
    ("state_canonical_dimension", pa.int32()), ("action_canonical_dimension", pa.int32()),
    ("lag_frames", pa.int32()), ("lag_seconds", pa.float64()), ("correlation", pa.float64()),
    ("unconstrained_lag_frames", pa.int32()), ("active_steps", pa.int64()),
    ("directional_agreement", pa.float64()), ("failed", pa.bool_()),
])
STAGE2_EPISODE_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("scored_dimensions", pa.int32()),
    ("failed_dimensions", pa.list_(pa.int32())), ("minimum_da", pa.float64()),
    ("reject_episode", pa.bool_()),
])


def load_stage1_rejected(output_root: pathlib.Path, ds: Dataset) -> set[int]:
    path = output_root / STAGE_NAMES[1] / safe_id(ds.dataset_id) / "episode_summary.parquet"
    if not path.is_file() or ds.cfg["stage1_policy"] != "episode":
        return set()
    table = pq.read_table(path, columns=["episode_index", "reject_episode"])
    eid = arrow_numpy(table["episode_index"].combine_chunks())
    rejected = arrow_numpy(table["reject_episode"].combine_chunks()).astype(bool)
    return {int(x) for x in eid[rejected]}


def run_stage2(
    ds: Dataset, local_dir: pathlib.Path, output_root: pathlib.Path, max_episodes: int | None
) -> dict[str, Any]:
    started = utc_now()
    state_dim = feature_dim(ds.info, ds.state_key)
    action_dim = feature_dim(ds.info, ds.action_key)
    pairs = state_action_map(ds, state_dim, action_dim)
    state_canonical = canonical_indices(ds, ds.state_key, state_dim)
    action_canonical = canonical_indices(ds, ds.action_key, action_dim)
    if not pairs:
        raise ValueError(
            f"{ds.dataset_id}: no trustworthy shared state/action joints; set state_action_map "
            "or allow_positional_mapping in config"
        )
    rejected_upstream = load_stage1_rejected(output_root, ds)
    max_lag = max(0, int(round(float(ds.cfg["stage2_max_lag_seconds"]) * ds.fps)))
    threshold = float(ds.cfg["stage2_da_threshold"])
    min_active = int(ds.cfg["stage2_min_active_steps"])
    local_dir.mkdir(parents=True, exist_ok=True)
    processed = rejected = skipped_upstream = unscored = 0
    with ParquetRows(local_dir / "dimension_metrics.parquet", STAGE2_DIM_SCHEMA) as dw, ParquetRows(
        local_dir / "episode_flags.parquet", STAGE2_EPISODE_SCHEMA
    ) as ew:
        for episode in iter_episodes(ds, [ds.state_key, ds.action_key], max_episodes):
            eid = int(episode["episode_index"][0])
            if eid in rejected_upstream:
                skipped_upstream += 1
                continue
            processed += 1
            state = smooth(canonical_signal(episode[ds.state_key], ds.cfg, ds.state_key), ds.cfg)
            action = canonical_signal(episode[ds.action_key], ds.cfg, ds.action_key)
            if ds.cfg["action_mode"] == "delta":
                action = np.cumsum(fill_nonfinite(action), axis=0)
            action = smooth(action, ds.cfg)
            failed: list[int] = []
            scores: list[float] = []
            for pair_index, (state_index, action_index) in enumerate(pairs):
                metric = trend_metric(state[:, state_index], action[:, action_index], max_lag, min_active)
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
                    "state_canonical_dimension": state_canonical[state_index],
                    "action_canonical_dimension": action_canonical[action_index],
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
    result = {
        "paper": PAPER, "stage": 2, "dataset_id": ds.dataset_id, "source": str(ds.path),
        "started_at": started, "finished_at": utc_now(), "max_episodes": max_episodes,
        "mapping": [{"state": s, "action": a} for s, a in pairs],
        "processed_episodes": processed, "rejected_episodes": rejected,
        "skipped_stage1_episodes": skipped_upstream, "unscored_dimension_episodes": unscored,
        "config": ds.cfg,
    }
    write_json(local_dir / "run.json", result)
    return result


def safe_id(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def load_stage1_frames(output_root: pathlib.Path, ds: Dataset) -> set[tuple[int, int]]:
    path = output_root / STAGE_NAMES[1] / safe_id(ds.dataset_id) / "frame_flags.parquet"
    if not path.is_file():
        return set()
    table = pq.read_table(path, columns=["episode_index", "frame_index"])
    episodes = arrow_numpy(table["episode_index"].combine_chunks())
    frames = arrow_numpy(table["frame_index"].combine_chunks())
    return set(zip(map(int, episodes), map(int, frames)))


def load_stage2_rejected(output_root: pathlib.Path, ds: Dataset) -> set[int]:
    path = output_root / STAGE_NAMES[2] / safe_id(ds.dataset_id) / "episode_flags.parquet"
    if not path.is_file():
        return set()
    table = pq.read_table(path, columns=["episode_index", "reject_episode"])
    episodes = arrow_numpy(table["episode_index"].combine_chunks())
    rejected = arrow_numpy(table["reject_episode"].combine_chunks()).astype(bool)
    return {int(x) for x in episodes[rejected]}


STAGE3_FRAME_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("frame_index", pa.int64()), ("index", pa.int64()),
    ("failed_state_dimensions", pa.list_(pa.int32())), ("failed_action_dimensions", pa.list_(pa.int32())),
    ("failed_state_canonical_dimensions", pa.list_(pa.int32())),
    ("failed_action_canonical_dimensions", pa.list_(pa.int32())),
    ("reject_frame", pa.bool_()),
])
STAGE3_EPISODE_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("input_frames", pa.int64()), ("upstream_kept_frames", pa.int64()),
    ("flagged_frames", pa.int64()), ("output_kept_frames", pa.int64()),
])
STAGE3_THRESHOLD_SCHEMA = pa.schema([
    ("embodiment", pa.string()), ("feature", pa.string()), ("dimension", pa.int32()),
    ("canonical_dimension", pa.int32()), ("name", pa.string()), ("q01", pa.float64()), ("q99", pa.float64()),
    ("lower", pa.float64()), ("upper", pa.float64()), ("alpha", pa.float64()),
    ("exempt_gripper", pa.bool_()),
])


def validate_group(group: list[Dataset]) -> tuple[int, int]:
    signatures = {
        (feature_dim(ds.info, ds.state_key), feature_dim(ds.info, ds.action_key), ds.state_key, ds.action_key)
        for ds in group
    }
    if len(signatures) != 1:
        raise ValueError(f"embodiment group has incompatible signal schemas: {signatures}")
    state_dim, action_dim, _, _ = next(iter(signatures))
    mappings = {
        (
            tuple(canonical_indices(ds, ds.state_key, state_dim)),
            tuple(canonical_indices(ds, ds.action_key, action_dim)),
        )
        for ds in group
    }
    if len(mappings) != 1:
        raise ValueError(
            "embodiment group has different raw-to-canonical column orders; "
            "split the embodiment ids before pooling Stage 3 thresholds"
        )
    return state_dim, action_dim


def calibrate_stage3(
    group: list[Dataset], output_root: pathlib.Path, max_episodes: int | None
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    state_dim, action_dim = validate_group(group)
    total = sum(int(ds.info.get("total_frames") or 0) for ds in group)
    capacity = max(int(ds.cfg["calibration_samples"]) for ds in group)
    collectors = {
        group[0].state_key: StrideSamples(total, capacity),
        group[0].action_key: StrideSamples(total, capacity),
    }
    for ds in group:
        stage1_frames = load_stage1_frames(output_root, ds)
        stage1_episodes = load_stage1_rejected(output_root, ds)
        stage2_episodes = load_stage2_rejected(output_root, ds)
        for episode in iter_episodes(ds, [ds.state_key, ds.action_key], max_episodes):
            eid = int(episode["episode_index"][0])
            if eid in stage1_episodes or eid in stage2_episodes:
                continue
            keep = np.asarray([
                (eid, int(frame)) not in stage1_frames for frame in episode["frame_index"]
            ], dtype=bool)
            collectors[group[0].state_key].add(episode[ds.state_key][keep])
            collectors[group[0].action_key].add(episode[ds.action_key][keep])
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
    output_root: pathlib.Path,
    max_episodes: int | None,
) -> dict[str, Any]:
    started = utc_now()
    state_dim = feature_dim(ds.info, ds.state_key)
    action_dim = feature_dim(ds.info, ds.action_key)
    state_canonical = canonical_indices(ds, ds.state_key, state_dim)
    action_canonical = canonical_indices(ds, ds.action_key, action_dim)
    stage1_frames = load_stage1_frames(output_root, ds)
    stage1_episodes = load_stage1_rejected(output_root, ds)
    stage2_episodes = load_stage2_rejected(output_root, ds)
    local_dir.mkdir(parents=True, exist_ok=True)
    processed = upstream_rejected_episodes = upstream_rejected_frames = flagged = 0
    with ParquetRows(local_dir / "frame_flags.parquet", STAGE3_FRAME_SCHEMA) as fw, ParquetRows(
        local_dir / "episode_summary.parquet", STAGE3_EPISODE_SCHEMA
    ) as ew:
        for episode in iter_episodes(ds, [ds.state_key, ds.action_key], max_episodes):
            eid = int(episode["episode_index"][0])
            n = len(episode["frame_index"])
            if eid in stage1_episodes or eid in stage2_episodes:
                upstream_rejected_episodes += 1
                continue
            processed += 1
            upstream_keep = np.asarray([
                (eid, int(frame)) not in stage1_frames for frame in episode["frame_index"]
            ], dtype=bool)
            upstream_rejected_frames += int((~upstream_keep).sum())
            feature_flags: dict[str, np.ndarray] = {}
            for key, dim in ((ds.state_key, state_dim), (ds.action_key, action_dim)):
                _, _, lower, upper = thresholds[key]
                values = np.asarray(episode[key], dtype=np.float64)
                flags = (~np.isfinite(values)) | (values < lower) | (values > upper)
                for index in gripper_indices(ds, key, dim):
                    if 0 <= index < dim:
                        flags[:, index] = False
                flags[~upstream_keep] = False
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
                    "failed_state_canonical_dimensions": map_failed_dimensions(failed_state, state_canonical),
                    "failed_action_canonical_dimensions": map_failed_dimensions(failed_action, action_canonical),
                    "reject_frame": True,
                })
            kept_before = int(upstream_keep.sum())
            ew.append({
                "episode_index": eid, "input_frames": n, "upstream_kept_frames": kept_before,
                "flagged_frames": len(rows), "output_kept_frames": kept_before - len(rows),
            })
    with ParquetRows(local_dir / "thresholds.parquet", STAGE3_THRESHOLD_SCHEMA) as tw:
        for key, dim in ((ds.state_key, state_dim), (ds.action_key, action_dim)):
            q01, q99, lower, upper = thresholds[key]
            names = flattened_names(ds.info["features"].get(key), dim)
            exempt = gripper_indices(ds, key, dim)
            for index in range(dim):
                tw.append({
                    "embodiment": str(ds.cfg["embodiment"]), "feature": key, "dimension": index,
                    "canonical_dimension": (state_canonical if key == ds.state_key else action_canonical)[index],
                    "name": names[index], "q01": float(q01[index]), "q99": float(q99[index]),
                    "lower": float(lower[index]), "upper": float(upper[index]),
                    "alpha": float(ds.cfg["stage3_alpha"]), "exempt_gripper": index in exempt,
                })
    result = {
        "paper": PAPER, "stage": 3, "dataset_id": ds.dataset_id,
        "embodiment": ds.cfg["embodiment"], "source": str(ds.path), "started_at": started,
        "finished_at": utc_now(), "max_episodes": max_episodes, "processed_episodes": processed,
        "upstream_rejected_episodes": upstream_rejected_episodes,
        "upstream_rejected_frames": upstream_rejected_frames, "flagged_frames": flagged,
        "config": ds.cfg,
    }
    write_json(local_dir / "run.json", result)
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


def execute(
    datasets: list[Dataset], stages: list[int], output_root: pathlib.Path, work_root: pathlib.Path,
    max_episodes: int | None, overwrite: bool,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "canonical_layout.json", LAYOUT)
    for stage in stages:
        (work_root / STAGE_NAMES[stage]).mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if 1 in stages:
        for ds in datasets:
            local = work_stage(work_root, 1, ds)
            result = run_stage1(ds, local, max_episodes)
            publish(local, output_root / STAGE_NAMES[1] / safe_id(ds.dataset_id), overwrite)
            shutil.rmtree(local)
            print(json.dumps(result, ensure_ascii=False))
            results.append(result)
    if 2 in stages:
        for ds in datasets:
            local = work_stage(work_root, 2, ds)
            result = run_stage2(ds, local, output_root, max_episodes)
            publish(local, output_root / STAGE_NAMES[2] / safe_id(ds.dataset_id), overwrite)
            shutil.rmtree(local)
            print(json.dumps(result, ensure_ascii=False))
            results.append(result)
    if 3 in stages:
        groups: dict[str, list[Dataset]] = {}
        for ds in datasets:
            groups.setdefault(str(ds.cfg["embodiment"]), []).append(ds)
        for embodiment, group in groups.items():
            alphas = {float(ds.cfg["stage3_alpha"]) for ds in group}
            if len(alphas) != 1:
                raise ValueError(f"embodiment {embodiment}: stage3_alpha differs across datasets")
            thresholds = calibrate_stage3(group, output_root, max_episodes)
            for ds in group:
                local = work_stage(work_root, 3, ds)
                result = run_stage3_dataset(ds, thresholds, local, output_root, max_episodes)
                publish(local, output_root / STAGE_NAMES[3] / safe_id(ds.dataset_id), overwrite)
                shutil.rmtree(local)
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
    run.add_argument("--output-root", type=pathlib.Path, required=True)
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
            state_canonical = canonical_indices(ds, ds.state_key, state_dim)
            action_canonical = canonical_indices(ds, ds.action_key, action_dim)
            canonical_status = "ready" if any(x is not None for x in state_canonical + action_canonical) else "mapping_required"
        except (KeyError, ValueError) as exc:
            state_dim = action_dim = None
            pairs = []
            status = f"unsupported: {exc}"
            canonical_status = "unsupported"
        rows.append({
            "dataset_id": ds.dataset_id, "path": str(ds.path), "status": status,
            "episodes": ds.info.get("total_episodes"), "frames": ds.info.get("total_frames"),
            "fps": ds.info.get("fps"), "state_dim": state_dim, "action_dim": action_dim,
            "mapped_joint_dimensions": len(pairs), "embodiment": ds.cfg["embodiment"],
            "canonical_status": canonical_status,
        })
    return rows


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "inventory":
        rows = inventory_rows(discover(args.input_root, config, args.discovery_depth))
        payload = {
            "created_at": utc_now(), "input_root": str(args.input_root),
            "canonical_layout": LAYOUT, "datasets": rows,
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
