"""Generic LeRobot v3.0 writer: create -> add_frame -> save_episode -> finalize,
plus write-then-validate-then-atomically-publish, extracted from
``convert_mobile_aloha_to_lerobot.py``'s ``_write_dataset``/``_validate_written_dataset``/
``_publish_temporary_output``/``convert_dataset``. Every reader (hdf5/rlds/
raw_image_json) funnels through this one code path, so a fix or a new safety
check here benefits all of them at once instead of being copy-pasted per
format.

This module only knows about :mod:`convert_core.episode_spec`'s dataclasses;
it never imports h5py/tensorflow_datasets/PIL or looks at a source path.
"""
from __future__ import annotations

import contextlib
import errno
import json
import math
import os
from pathlib import Path
import queue
import shutil
import sys
import time
from typing import Any, Callable, Iterator
import types
import uuid

import numpy as np

from convert_core.checkpoint import (
    CheckpointManager,
    atomic_write_json,
    build_resume_payload,
    decoder_records_from_plan,
    exclusive_resume_lock,
    read_json_object,
)
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError
from convert_core.progress import EtaProgress

IterFrames = Callable[[EpisodePlan], Iterator[dict[str, Any]]]

LEROBOT_CODEBASE_VERSION = "v3.0"
LEROBOT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
LEROBOT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)

_GENERATED_INDEX_FEATURES = frozenset(
    {"frame_index", "episode_index", "index", "task_index"}
)


def normalize_generated_index_stats(
    episode_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Make generated-index episode-stat schemas stable across episode sizes.

    LeRobot's small-episode statistics path preserves the integer dtype of
    ``min``/``max``, while the running-statistics path promotes those values to
    float64. A single metadata Parquet writer cannot append both schemas.
    Normalize every non-count statistic for LeRobot's generated index fields
    to float64. ``count`` intentionally remains int64: it is a sample-count
    metadata field in LeRobot's stats contract.
    """

    normalized = dict(episode_stats)
    for feature_key in _GENERATED_INDEX_FEATURES:
        feature_stats = episode_stats.get(feature_key)
        if feature_stats is None:
            continue
        normalized[feature_key] = {
            stat_key: np.asarray(
                value, dtype=np.int64 if stat_key == "count" else np.float64
            )
            for stat_key, value in feature_stats.items()
        }
    return normalized


def _install_generated_index_stats_normalizer(dataset: Any) -> None:
    """Normalize index stats immediately before LeRobot writes metadata."""

    original_save_episode = dataset.meta.save_episode

    def save_episode(
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict[str, Any]],
        episode_metadata: dict[str, Any],
    ) -> None:
        original_save_episode(
            episode_index,
            episode_length,
            episode_tasks,
            normalize_generated_index_stats(episode_stats),
            episode_metadata,
        )

    dataset.meta.save_episode = save_episode


def _blocking_streaming_feed_frame(self: Any, video_key: str, image: Any) -> None:
    """Bound a streaming encoder queue without LeRobot's frame-drop policy.

    LeRobot 0.6 waits only 100 ms before silently dropping a frame from a full
    queue.  Offline conversion must instead apply backpressure: every planned
    frame is required, and a small fixed queue is what keeps worker memory
    bounded.  The encoder-thread health check mirrors the upstream method.
    """

    if not self._episode_active:
        raise RuntimeError("No active episode. Call start_episode() first.")
    copied = image.copy()
    while True:
        thread = self._threads[video_key]
        if not thread.is_alive():
            try:
                status, message = self._result_queues[video_key].get_nowait()
                if status == "error":
                    raise RuntimeError(
                        f"Encoder thread for {video_key} crashed: {message}"
                    )
            except queue.Empty:
                pass
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")
        try:
            self._frame_queues[video_key].put(copied, timeout=0.1)
            return
        except queue.Full:
            continue


def _enable_blocking_streaming_encoding(
    dataset: Any, *, encoder_temp_root: Path | None = None
) -> None:
    encoder = getattr(getattr(dataset, "writer", None), "_streaming_encoder", None)
    if encoder is not None:
        encoder.feed_frame = types.MethodType(_blocking_streaming_feed_frame, encoder)
        if encoder_temp_root is not None:
            configured_root = Path(encoder_temp_root)
            configured_root.mkdir(parents=True, exist_ok=True)
            original_start_episode = encoder.start_episode

            def start_episode_in_work_root(
                _self: Any,
                video_keys: list[str],
                temp_dir: Path,
                depth_video_keys: list[str] | None = None,
            ) -> None:
                del temp_dir
                original_start_episode(
                    video_keys=video_keys,
                    temp_dir=configured_root,
                    depth_video_keys=depth_video_keys,
                )

            encoder.start_episode = types.MethodType(
                start_episode_in_work_root, encoder
            )


class _SequentialMp4Sink:
    """Write-only file object that never advertises FUSE seek support."""

    def __init__(self, path: Path):
        self._stream = path.open("wb")
        self._position = 0

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def write(self, data: bytes) -> int:
        written = self._stream.write(data)
        self._position += written
        return written

    def tell(self) -> int:
        return self._position

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class _ContainerWithSink:
    """Keep a Python AVIO sink alive and close it with its PyAV container."""

    def __init__(self, container: Any, sink: _SequentialMp4Sink):
        self._container = container
        self._sink = sink
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._container, name)

    def __enter__(self) -> "_ContainerWithSink":
        self._container.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        try:
            return self._container.__exit__(exc_type, exc, traceback)
        finally:
            self._sink.close()
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._container.close()
        finally:
            self._sink.close()
            self._closed = True


@contextlib.contextmanager
def _fragmented_mp4_writes(enabled: bool) -> Iterator[None]:
    """Make MP4 muxing sequential-write compatible for OSSFS/FUSE targets."""

    if not enabled:
        yield
        return
    import lerobot.datasets.dataset_metadata as metadata_module
    import lerobot.datasets.video_utils as video_utils

    original_open = video_utils.av.open
    original_get_video_info = video_utils.get_video_info
    original_metadata_get_video_info = metadata_module.get_video_info

    def open_with_fragmented_mp4(file: Any, mode: str | None = None, *args: Any, **kwargs: Any):
        sequential_mp4 = (
            mode is not None
            and "w" in mode
            and str(file).lower().endswith(".mp4")
        )
        if sequential_mp4:
            options = dict(kwargs.get("options") or {})
            options["movflags"] = (
                "frag_keyframe+empty_moov+default_base_moof+negative_cts_offsets"
            )
            kwargs["options"] = options
        is_mp4_path = isinstance(file, (str, os.PathLike)) and str(file).lower().endswith(".mp4")
        path_output = sequential_mp4 and isinstance(file, (str, os.PathLike))
        # A path-backed AVIO context advertises seekability because OSSFS/FUSE
        # implements seek(2), but movenc fragment bookkeeping can still fail
        # with EINVAL under concurrent decode/encode traffic. A write-only
        # Python object makes the non-seekable contract explicit.
        delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6) if is_mp4_path else ()
        for attempt in range(len(delays) + 1):
            sink = None
            try:
                if path_output:
                    sink = _SequentialMp4Sink(Path(file))
                    kwargs.setdefault("format", "mp4")
                    container = original_open(sink, mode, *args, **kwargs)
                    return _ContainerWithSink(container, sink)
                return original_open(file, mode, *args, **kwargs)
            except Exception as exc:
                if sink is not None:
                    sink.close()
                # PyAV maps FFmpeg's AVERROR(EINVAL) to av.error.ValueError,
                # which inherits from built-in ValueError rather than OSError.
                # Inspect errno instead of the Python exception hierarchy so
                # both kernel/FUSE and FFmpeg wrappers receive the same narrow
                # retry policy. Exceptions without one of the explicitly
                # transient errnos are still raised immediately.
                if (
                    not is_mp4_path
                    or getattr(exc, "errno", None)
                    not in {errno.ENOENT, errno.EINVAL, errno.ESTALE}
                    or attempt == len(delays)
                ):
                    raise
                time.sleep(delays[attempt])

    def get_fragmented_video_info(
        video_path: Any, video_encoder: Any = None
    ) -> dict[str, Any]:
        info = original_get_video_info(video_path, video_encoder=video_encoder)
        with video_utils.av.open(str(video_path), "r") as video_file:
            stream = video_file.streams.video[0]
            if stream.average_rate is not None:
                info["video.fps"] = int(stream.average_rate)
        return info

    video_utils.av.open = open_with_fragmented_mp4
    video_utils.get_video_info = get_fragmented_video_info
    metadata_module.get_video_info = get_fragmented_video_info
    try:
        yield
    finally:
        metadata_module.get_video_info = original_metadata_get_video_info
        video_utils.get_video_info = original_get_video_info
        video_utils.av.open = original_open


@contextlib.contextmanager
def _deferred_video_concatenation(enabled: bool) -> Iterator[None]:
    """Concatenate episode videos once per chunk instead of once per episode.

    LeRobot's stock writer appends an episode by reading and rewriting the
    current chunk MP4.  That is quadratic in the number of episodes and is
    especially expensive on OSSFS/FUSE.  Keep encoded episode files in the
    local temporary directory, then concatenate the complete chunk exactly
    once when it rolls over or the writer is finalized.
    """

    if not enabled:
        yield
        return

    import lerobot.datasets.dataset_metadata as metadata_module
    import lerobot.datasets.dataset_writer as writer_module

    DatasetWriter = writer_module.DatasetWriter
    original_save = DatasetWriter._save_episode_video
    original_flush = DatasetWriter.flush_pending_videos

    def flush_video_state(dataset: Any, video_key: str, state: dict[str, Any]) -> None:
        from lerobot.datasets.dataset_writer import concatenate_video_files

        paths = list(state["paths"])
        if not paths:
            return
        final_path = dataset._root / dataset._meta.video_path.format(
            video_key=video_key,
            chunk_index=state["chunk_idx"],
            file_index=state["file_idx"],
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if len(paths) == 1:
            shutil.move(str(paths[0]), str(final_path))
        else:
            concatenate_video_files(paths, final_path)
        for path in paths:
            shutil.rmtree(path.parent, ignore_errors=True)

        encoder = (
            dataset._depth_encoder
            if video_key in dataset._meta.depth_keys
            else dataset._rgb_encoder
        )
        dataset._meta.update_video_info(video_key, video_encoder=encoder)
        metadata_module.write_info(dataset._meta.info, dataset._meta.root)
        state["paths"] = []
        state["size_in_mb"] = 0.0
        state["duration"] = 0.0

    def flush_dataset_videos(dataset: Any) -> None:
        states = getattr(dataset, "_vla_deferred_video_states", {})
        for video_key, state in list(states.items()):
            flush_video_state(dataset, video_key, state)

    def save_episode_video(
        dataset: Any,
        video_key: str,
        episode_index: int,
        temp_path: Path | None = None,
    ) -> dict:
        if not hasattr(dataset, "_vla_deferred_resume_existing"):
            existing_episodes = getattr(dataset._meta, "episodes", None)
            dataset._vla_deferred_resume_existing = bool(existing_episodes)
        # A resumed writer already has an open final chunk.  Reusing the
        # deferred state would restart at chunk zero and could overwrite it;
        # let LeRobot append to that existing dataset using its safe upstream
        # path. Fresh partition writers (the MimicGen staged path) start with
        # an empty episode list and use the optimized path below.
        if dataset._vla_deferred_resume_existing:
            return original_save(dataset, video_key, episode_index, temp_path=temp_path)
        # Non-streaming/batched callers may not provide a temporary path. Keep
        # their upstream behavior unchanged; MimicGen's streaming path always
        # supplies one.
        if temp_path is None:
            return original_save(dataset, video_key, episode_index, temp_path=None)

        from lerobot.datasets.dataset_writer import (
            get_file_size_in_mb,
            get_video_duration_in_s,
            update_chunk_file_indices,
        )

        ep_path = Path(temp_path)
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)
        states = getattr(dataset, "_vla_deferred_video_states", None)
        if states is None:
            states = {}
            dataset._vla_deferred_video_states = states
        state = states.get(video_key)
        if state is None:
            state = {
                "chunk_idx": 0,
                "file_idx": 0,
                "size_in_mb": 0.0,
                "duration": 0.0,
                "paths": [],
            }
            states[video_key] = state

        # Roll over before adding the next episode. A single oversized episode
        # remains a valid one-file chunk instead of causing an empty chunk.
        threshold = dataset._meta.video_files_size_in_mb
        if state["paths"] and state["size_in_mb"] + ep_size_in_mb >= threshold:
            old_chunk, old_file = state["chunk_idx"], state["file_idx"]
            flush_video_state(dataset, video_key, state)
            state["chunk_idx"], state["file_idx"] = update_chunk_file_indices(
                old_chunk, old_file, dataset._meta.chunks_size
            )

        from_timestamp = state["duration"]
        state["paths"].append(ep_path)
        state["size_in_mb"] += ep_size_in_mb
        state["duration"] += ep_duration_in_s
        return {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": state["chunk_idx"],
            f"videos/{video_key}/file_index": state["file_idx"],
            f"videos/{video_key}/from_timestamp": from_timestamp,
            f"videos/{video_key}/to_timestamp": state["duration"],
        }

    def flush_pending_videos(dataset: Any) -> None:
        # Close the streaming encoders first; all completed episode paths have
        # already been registered by save_episode_video.
        original_flush(dataset)
        flush_dataset_videos(dataset)

    DatasetWriter._save_episode_video = save_episode_video
    DatasetWriter.flush_pending_videos = flush_pending_videos
    try:
        yield
    finally:
        DatasetWriter._save_episode_video = original_save
        DatasetWriter.flush_pending_videos = original_flush


@contextlib.contextmanager
def _deferred_info_stats_writes(dataset: Any, enabled: bool) -> Iterator[None]:
    """Batch LeRobot's per-episode info/stats rewrites at one unit boundary."""

    if not enabled:
        yield
        return
    import lerobot.datasets.dataset_metadata as metadata_module
    import lerobot.datasets.dataset_writer as writer_module

    original_info = metadata_module.write_info
    original_stats = metadata_module.write_stats
    original_writer_info = writer_module.write_info
    succeeded = False
    metadata_module.write_info = lambda _value, _root: None
    metadata_module.write_stats = lambda _value, _root: None
    writer_module.write_info = lambda _value, _root: None
    try:
        yield
        succeeded = True
    finally:
        metadata_module.write_info = original_info
        metadata_module.write_stats = original_stats
        writer_module.write_info = original_writer_info
        if succeeded:
            original_info(dataset.meta.info, dataset.meta.root)
            if dataset.meta.stats is not None:
                original_stats(dataset.meta.stats, dataset.meta.root)


@contextlib.contextmanager
def _local_datasets_cache(output_root: Path) -> Iterator[None]:
    """Keep Hugging Face parquet cache writes beside staging, never in $HOME."""

    configured_cache = os.environ.get("VLA_DATASETS_CACHE_ROOT")
    if configured_cache:
        cache = Path(configured_cache)
    else:
        cache_parent = output_root.parent
        for candidate in (output_root, *output_root.parents):
            if candidate.name == "lerobot_v3_0":
                cache_parent = candidate
                break
        cache = cache_parent / ".lerobot-datasets-cache"
    cache.mkdir(parents=True, exist_ok=True)
    previous_env = os.environ.get("HF_DATASETS_CACHE")
    os.environ["HF_DATASETS_CACHE"] = str(cache)
    try:
        import datasets

        previous_config = datasets.config.HF_DATASETS_CACHE
        datasets.config.HF_DATASETS_CACHE = cache
        try:
            yield
        finally:
            datasets.config.HF_DATASETS_CACHE = previous_config
    finally:
        if previous_env is None:
            os.environ.pop("HF_DATASETS_CACHE", None)
        else:
            os.environ["HF_DATASETS_CACHE"] = previous_env


def plan_summary(plan: DatasetConversionPlan) -> dict[str, Any]:
    return {
        "dataset_uid": plan.dataset_uid,
        "output": str(plan.output_path),
        "robot_type": plan.robot_type,
        "fps": plan.fps,
        "measured_fps": plan.measured_fps,
        "episodes": len(plan.episodes),
        "frames": plan.num_frames,
        "tasks": sorted({episode.instruction for episode in plan.episodes}),
        "features": plan.feature_schema(),
    }


def build_manifest(plan: DatasetConversionPlan, *, reader_format: str) -> dict[str, Any]:
    task_indices: dict[str, int] = {}
    for episode in plan.episodes:
        task_indices.setdefault(episode.instruction, len(task_indices))
    episodes: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(plan.episodes):
        row = {
            "lerobot_episode_index": episode_index,
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
            "checkpoint_unit": episode.extra.get("checkpoint_unit"),
        }
        # Format-specific readers may provide compact, JSON-safe provenance
        # that is important to preserve but should not be copied wholesale
        # from the internal EpisodePlan extras into every manifest.
        provenance = episode.extra.get("manifest_provenance")
        if provenance is not None:
            row["source_provenance"] = provenance
        episodes.append(row)

    manifest = {
        "format": "lerobot_v3_0",
        "converter": plan.extra.get("converter", "convert_dataset.py"),
        "source_format": reader_format,
        "dataset_uid": plan.dataset_uid,
        "robot_type": plan.robot_type,
        "fps": plan.fps,
        "measured_fps": plan.measured_fps,
        "num_episodes": len(plan.episodes),
        "num_frames": plan.num_frames,
        "num_video_features": len(plan.camera_features),
        "features": plan.feature_schema(),
        "episodes": episodes,
    }
    for key in (
        "source_dataset",
        "source_revision",
        "source_files",
        "source_relative_path",
        "source_env_name",
        "source_env_args",
        "source_splits",
        "dangling_split_references",
        "field_mapping",
        "video_encoding",
        "task_provenance",
        "timestamp_provenance",
        "split_summaries",
        "unsupported_source_components",
        "decoder",
        "partition_rules",
        "action_semantics",
        "quaternion_convention",
        "pointcloud_semantics",
        "source_builder",
        "source_data_attributes",
        "payload_scan_coverage",
        "model_sidecar",
    ):
        if key in plan.extra:
            manifest[key] = plan.extra[key]
    manifest["task_index_mapping"] = {
        str(index): task for task, index in task_indices.items()
    }
    decoder_records = decoder_records_from_plan(plan)
    if decoder_records:
        manifest["decoder_records"] = decoder_records
    return manifest


def write_dataset(
    plan: DatasetConversionPlan,
    iter_frames: IterFrames,
    temporary_path: Path,
    *,
    rgb_encoder: Any = None,
    streaming_encoding: bool = False,
    blocking_streaming_encoding: bool = False,
    encoder_queue_maxsize: int = 30,
    encoder_threads: int | None = None,
    batch_metadata_writes: bool = False,
    encoder_temp_root: Path | None = None,
    fragmented_mp4_writes: bool = False,
    deferred_video_concatenation: bool = False,
    frame_completed_hook: Callable[[EpisodePlan, int], None] | None = None,
    episode_completed_hook: Callable[[EpisodePlan, int], None] | None = None,
) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("lerobot==0.6.0 is required; install the project's requirements.txt") from exc

    with _local_datasets_cache(temporary_path), _fragmented_mp4_writes(
        fragmented_mp4_writes
    ), _deferred_video_concatenation(deferred_video_concatenation):
        dataset = LeRobotDataset.create(
            repo_id=plan.dataset_uid,
            fps=plan.fps,
            root=temporary_path,
            features=plan.feature_schema(),
            robot_type=plan.robot_type,
            use_videos=True,
            rgb_encoder=rgb_encoder,
            streaming_encoding=streaming_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )
        _install_generated_index_stats_normalizer(dataset)
        if blocking_streaming_encoding:
            _enable_blocking_streaming_encoding(
                dataset, encoder_temp_root=encoder_temp_root
            )
        try:
            with _deferred_info_stats_writes(dataset, batch_metadata_writes):
                for episode_index, episode in enumerate(plan.episodes):
                    print(
                        f"[{plan.dataset_uid}] episode {episode_index + 1}/{len(plan.episodes)}: "
                        f"{episode.source_relative_path} ({episode.num_frames} frames)",
                        file=sys.stderr,
                    )
                    for frame in iter_frames(episode):
                        dataset.add_frame(frame)
                        if frame_completed_hook is not None:
                            frame_completed_hook(episode, episode_index)
                    dataset.save_episode()
                    if episode_completed_hook is not None:
                        episode_completed_hook(episode, episode_index)
                dataset.finalize()
        except BaseException:
            # Preserve the original error while explicitly closing encoder and
            # parquet resources. The caller will discard this unmarked unit.
            with contextlib.suppress(Exception):
                dataset.clear_episode_buffer(delete_images=True)
            with contextlib.suppress(Exception):
                dataset.finalize()
            del dataset
            raise


def validate_written_dataset(plan: DatasetConversionPlan, temporary_path: Path) -> None:
    validate_written_prefix(plan, temporary_path, len(plan.episodes))


def validate_info_json(
    plan: DatasetConversionPlan,
    root: Path,
    completed_episodes: int,
) -> dict[str, Any]:
    """Validate every required LeRobot v3 ``meta/info.json`` contract.

    LeRobot produces this file from the data it actually finalized.  Reading
    it back (instead of trusting the requested plan) catches stale counts,
    wrong path templates, feature-name loss, and incomplete video metadata.
    """

    info = read_json_object(root / "meta" / "info.json", "LeRobot meta/info.json")
    expected_frames = sum(
        episode.num_frames for episode in plan.episodes[:completed_episodes]
    )
    expected_tasks = list(
        dict.fromkeys(
            episode.instruction for episode in plan.episodes[:completed_episodes]
        )
    )
    scalar_expectations = {
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "robot_type": plan.robot_type,
        "total_episodes": completed_episodes,
        "total_frames": expected_frames,
        "total_tasks": len(expected_tasks),
        "data_path": LEROBOT_DATA_PATH,
        "video_path": LEROBOT_VIDEO_PATH,
    }
    for key, expected in scalar_expectations.items():
        if info.get(key) != expected:
            raise ConversionError(
                f"meta/info.json {key} is {info.get(key)!r}, expected {expected!r}"
            )
    try:
        written_fps = float(info["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError("meta/info.json fps is missing or invalid") from exc
    if not math.isclose(written_fps, plan.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ConversionError(
            f"meta/info.json fps is {written_fps}, expected {plan.fps}"
        )
    for key in ("chunks_size", "data_files_size_in_mb", "video_files_size_in_mb"):
        value = info.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConversionError(f"meta/info.json {key} must be a positive integer")
    if info.get("splits") != {"train": f"0:{completed_episodes}"}:
        raise ConversionError(
            f"meta/info.json splits is {info.get('splits')!r}, expected "
            f"{{'train': '0:{completed_episodes}'}}"
        )

    written_features = info.get("features")
    if not isinstance(written_features, dict):
        raise ConversionError("meta/info.json features must be an object")
    expected_features = plan.feature_schema()
    missing = set(expected_features) - set(written_features)
    if missing:
        raise ConversionError(
            f"meta/info.json is missing planned features: {sorted(missing)}"
        )
    for key, expected in expected_features.items():
        actual = written_features[key]
        if not isinstance(actual, dict):
            raise ConversionError(f"meta/info.json feature {key!r} must be an object")
        for attribute in ("dtype", "names"):
            if actual.get(attribute) != expected[attribute]:
                raise ConversionError(
                    f"meta/info.json feature {key!r} {attribute} is "
                    f"{actual.get(attribute)!r}, expected {expected[attribute]!r}"
                )
        if tuple(actual.get("shape", ())) != tuple(expected["shape"]):
            raise ConversionError(
                f"meta/info.json feature {key!r} shape is {actual.get('shape')!r}, "
                f"expected {expected['shape']!r}"
            )

    written_video_keys = {
        key
        for key, value in written_features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    }
    expected_video_keys = {camera.feature_key for camera in plan.camera_features}
    if written_video_keys != expected_video_keys:
        raise ConversionError(
            f"meta/info.json video features are {sorted(written_video_keys)}, "
            f"expected {sorted(expected_video_keys)}"
        )
    camera_by_key = {camera.feature_key: camera for camera in plan.camera_features}
    for key in expected_video_keys:
        camera = camera_by_key[key]
        video_info = written_features[key].get("info")
        if not isinstance(video_info, dict):
            raise ConversionError(f"meta/info.json video feature {key!r} has no info object")
        required = {
            "video.height": camera.height,
            "video.width": camera.width,
            "video.fps": plan.fps,
            "video.channels": 3,
            "has_audio": False,
            "is_depth_map": False,
        }
        for attribute, expected in required.items():
            actual = video_info.get(attribute)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                try:
                    matches = math.isclose(
                        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                raise ConversionError(
                    f"meta/info.json video feature {key!r} {attribute} is "
                    f"{actual!r}, expected {expected!r}"
                )
        for attribute in ("video.codec", "video.pix_fmt"):
            if not isinstance(video_info.get(attribute), str) or not video_info[attribute]:
                raise ConversionError(
                    f"meta/info.json video feature {key!r} has invalid {attribute}"
                )
    return info


def validate_parquet_feature_schema(plan: DatasetConversionPlan, root: Path) -> None:
    """Verify that physical Arrow columns match planned dtype and shape.

    LeRobot intentionally represents metadata shape ``[1]`` as a scalar Arrow
    ``Value``; wider vectors and arrays use fixed-size lists. Checking every
    shard against that canonical encoding catches real dtype/width drift while
    accepting LeRobot's documented singleton convention.
    """

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_root = root / "data"
    paths = sorted(data_root.rglob("*.parquet")) if data_root.is_dir() else []
    if not paths:
        raise ConversionError(f"written dataset has no Parquet data files: {data_root}")
    expected_types: dict[str, Any] = {}
    for key, feature in plan.feature_schema().items():
        if feature["dtype"] == "video":
            continue
        if feature["dtype"] == "string":
            expected_types[key] = pa.string()
            continue
        if feature["dtype"] == "image":
            # LeRobot v3 stores image payloads as Arrow ``Image`` extension
            # storage (a ``struct<bytes, path>`` in Parquet).  The physical
            # representation intentionally differs from numeric tensors, so
            # validate its structure rather than forcing it through NumPy.
            expected_types[key] = ("image",)
            continue
        arrow_type = pa.from_numpy_dtype(np.dtype(feature["dtype"]))
        shape = tuple(feature["shape"])
        # Hugging Face Datasets uses its ArrayND extension types for features
        # with two or more axes.  Validate their declared shape and scalar
        # value type directly; their storage lists are variable-sized by
        # design even though the extension shape is fixed.
        if len(shape) >= 2:
            expected_types[key] = ("array_nd", shape, arrow_type)
            continue
        if shape != (1,):
            for dimension in reversed(shape):
                arrow_type = pa.list_(arrow_type, int(dimension))
        expected_types[key] = arrow_type
    for path in paths:
        schema = pq.read_schema(path)
        for key, expected_type in expected_types.items():
            if key not in schema.names:
                raise ConversionError(f"{path}: missing planned Parquet column {key!r}")
            actual_type = schema.field(key).type
            if isinstance(expected_type, tuple) and expected_type[0] == "image":
                import pyarrow as pa

                if not isinstance(actual_type, pa.StructType):
                    raise ConversionError(
                        f"{path}: image column {key!r} has physical type {actual_type}, expected struct<bytes, path>"
                    )
                fields = {field.name: field.type for field in actual_type}
                if fields.get("bytes") != pa.binary() or fields.get("path") != pa.string():
                    raise ConversionError(
                        f"{path}: image column {key!r} has physical fields {fields}, expected bytes/path"
                    )
                continue
            if isinstance(expected_type, tuple) and expected_type[0] == "array_nd":
                _, expected_shape, expected_value_type = expected_type
                actual_shape = tuple(getattr(actual_type, "shape", ()))
                actual_value_type = getattr(actual_type, "value_type", None)
                if (
                    not isinstance(actual_type, pa.ExtensionType)
                    or actual_shape != expected_shape
                    or actual_value_type is None
                    or pa.from_numpy_dtype(np.dtype(actual_value_type)) != expected_value_type
                ):
                    raise ConversionError(
                        f"{path}: Parquet column {key!r} has physical type {actual_type}, "
                        f"expected a fixed ArrayND extension with shape {expected_shape} "
                        f"and value type {expected_value_type}"
                    )
                continue
            if actual_type != expected_type:
                raise ConversionError(
                    f"{path}: Parquet column {key!r} has physical type {actual_type}, "
                    f"expected {expected_type}"
                )


def validate_written_prefix(
    plan: DatasetConversionPlan,
    temporary_path: Path,
    completed_episodes: int,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    expected_frames = sum(
        episode.num_frames for episode in plan.episodes[:completed_episodes]
    )
    validate_info_json(plan, temporary_path, completed_episodes)
    validate_parquet_feature_schema(plan, temporary_path)
    with _local_datasets_cache(temporary_path):
        dataset = LeRobotDataset(repo_id=plan.dataset_uid, root=temporary_path)
        if dataset.num_episodes != completed_episodes:
            raise ConversionError(
                f"written dataset has {dataset.num_episodes} episodes, expected {completed_episodes}"
            )
        if len(dataset) != expected_frames:
            raise ConversionError(
                f"written dataset has {len(dataset)} frames, expected {expected_frames}"
            )
        missing = set(plan.feature_schema()) - set(dataset.meta.features)
        if missing:
            raise ConversionError(f"written dataset is missing features: {sorted(missing)}")
        for key, expected in plan.feature_schema().items():
            actual = dataset.meta.features[key]
            if actual.get("dtype") != expected["dtype"]:
                raise ConversionError(
                    f"written feature {key!r} dtype is {actual.get('dtype')!r}, expected {expected['dtype']!r}"
                )
            if tuple(actual.get("shape", ())) != tuple(expected["shape"]):
                raise ConversionError(
                    f"written feature {key!r} shape is {actual.get('shape')!r}, expected {expected['shape']!r}"
                )
            if actual.get("names") != expected["names"]:
                raise ConversionError(
                    f"written feature {key!r} names are {actual.get('names')!r}, "
                    f"expected {expected['names']!r}"
                )
        written_fps = float(dataset.meta.fps)
        if not math.isclose(written_fps, plan.fps, rel_tol=0.0, abs_tol=1e-9):
            raise ConversionError(f"written dataset FPS is {written_fps}, expected {plan.fps}")
        if dataset.meta.robot_type != plan.robot_type:
            raise ConversionError(
                f"written robot_type is {dataset.meta.robot_type!r}, expected {plan.robot_type!r}"
            )
        written_tasks = set(dataset.meta.tasks.index.tolist())
        expected_tasks = {
            episode.instruction for episode in plan.episodes[:completed_episodes]
        }
        if written_tasks != expected_tasks:
            raise ConversionError(
                f"written dataset tasks are {sorted(written_tasks)}, expected {sorted(expected_tasks)}"
            )
        expected_task_indices = {
            task: index
            for index, task in enumerate(
                dict.fromkeys(
                    episode.instruction
                    for episode in plan.episodes[:completed_episodes]
                )
            )
        }
        written_task_indices = {
            str(task): int(index)
            for task, index in dataset.meta.tasks["task_index"].to_dict().items()
        }
        if written_task_indices != expected_task_indices:
            raise ConversionError(
                f"written task indices are {written_task_indices}, expected {expected_task_indices}"
            )
        episode_rows = dataset.meta.episodes
        for episode_index, expected in enumerate(plan.episodes[:completed_episodes]):
            row = episode_rows[episode_index]
            if int(row["episode_index"]) != episode_index:
                raise ConversionError(
                    f"written episode row {episode_index} has episode_index "
                    f"{row['episode_index']!r}"
                )
            if int(row["length"]) != expected.num_frames:
                raise ConversionError(
                    f"written episode {episode_index} has {int(row['length'])} frames, expected {expected.num_frames}"
                )
            if set(row["tasks"]) != {expected.instruction}:
                raise ConversionError(
                    f"written episode {episode_index} tasks are {row['tasks']}, expected {[expected.instruction]}"
                )
        del dataset


def _video_frame_count(path: Path) -> tuple[int, int, int, float | None, str, str | None]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency of lerobot datasets
        raise RuntimeError("PyAV is required to validate checkpoint videos") from exc
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ConversionError(f"video checkpoint has no video stream: {path}")
        stream = container.streams.video[0]
        count = int(stream.frames or 0)
        if count <= 0:
            count = sum(1 for _ in container.decode(stream))
        rate = stream.average_rate or stream.base_rate
        return (
            count,
            int(stream.height),
            int(stream.width),
            float(rate) if rate is not None else None,
            stream.codec.canonical_name,
            stream.codec_context.format.name if stream.codec_context.format else None,
        )


def validate_video_files(
    plan: DatasetConversionPlan,
    root: Path,
    *,
    expected_frames: int,
    relative_paths: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Open actual MP4 streams and verify frames, rate, and dimensions."""

    evidence: dict[str, list[dict[str, Any]]] = {}
    video_encoding = plan.extra.get("video_encoding", {})
    requested_codec = (
        video_encoding.get("target_codec") if isinstance(video_encoding, dict) else None
    )
    expected_codec = {
        "libsvtav1": "av1",
        "h264": "h264",
        "h264_nvenc": "h264",
        "hevc": "hevc",
        "hevc_nvenc": "hevc",
    }.get(requested_codec, requested_codec)
    expected_pix_fmt = (
        video_encoding.get("target_pix_fmt")
        if isinstance(video_encoding, dict)
        else None
    )
    for camera in plan.camera_features:
        camera_root = root / "videos" / camera.feature_key
        paths = sorted(camera_root.rglob("*.mp4")) if camera_root.is_dir() else []
        if relative_paths is not None:
            paths = [
                path
                for path in paths
                if path.relative_to(root).as_posix() in relative_paths
            ]
        if not paths:
            raise ConversionError(f"no written video files found for {camera.feature_key}")
        rows: list[dict[str, Any]] = []
        total = 0
        for path in paths:
            frames, height, width, rate, codec, pix_fmt = _video_frame_count(path)
            if (height, width) != (camera.height, camera.width):
                raise ConversionError(
                    f"{path}: video is {width}x{height}, expected {camera.width}x{camera.height}"
                )
            if rate is None or not math.isclose(rate, plan.fps, rel_tol=0.0, abs_tol=1e-9):
                raise ConversionError(f"{path}: video FPS is {rate}, expected {plan.fps}")
            if expected_codec is not None and codec != expected_codec:
                raise ConversionError(
                    f"{path}: video codec is {codec!r}, expected {expected_codec!r} "
                    f"for requested encoder {requested_codec!r}"
                )
            if expected_pix_fmt is not None and pix_fmt != expected_pix_fmt:
                raise ConversionError(
                    f"{path}: video pixel format is {pix_fmt!r}, expected {expected_pix_fmt!r}"
                )
            total += frames
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "frames": frames,
                    "height": height,
                    "width": width,
                    "fps": rate,
                    "codec": codec,
                    "pix_fmt": pix_fmt,
                }
            )
        if total != expected_frames:
            raise ConversionError(
                f"written videos for {camera.feature_key} contain {total} frames, "
                f"expected {expected_frames}"
            )
        evidence[camera.feature_key] = rows
    return evidence


def _checkpoint_unit(episode: EpisodePlan) -> str:
    return str(episode.extra.get("checkpoint_unit", episode.episode_uid))


def _validate_checkpoint_unit_order(plan: DatasetConversionPlan) -> None:
    closed: set[str] = set()
    previous: str | None = None
    for episode in plan.episodes:
        current = _checkpoint_unit(episode)
        if current != previous:
            if current in closed:
                raise ConversionError(
                    f"checkpoint unit {current!r} is not contiguous in the conversion plan"
                )
            if previous is not None:
                closed.add(previous)
            previous = current


def _unit_end(plan: DatasetConversionPlan, start: int) -> int:
    unit = _checkpoint_unit(plan.episodes[start])
    end = start + 1
    while end < len(plan.episodes) and _checkpoint_unit(plan.episodes[end]) == unit:
        end += 1
    return end


def _open_resumable_writer(
    plan: DatasetConversionPlan,
    root: Path,
    *,
    resume_existing: bool,
    rgb_encoder: Any,
    streaming_encoding: bool,
    blocking_streaming_encoding: bool,
    encoder_queue_maxsize: int,
    encoder_threads: int | None,
    metadata_buffer_size: int,
    encoder_temp_root: Path | None,
) -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lerobot==0.6.0 is required") from exc
    if resume_existing:
        dataset = LeRobotDataset.resume(
            repo_id=plan.dataset_uid,
            root=root,
            rgb_encoder=rgb_encoder,
            streaming_encoding=streaming_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )
        _install_generated_index_stats_normalizer(dataset)
        dataset.meta._metadata_buffer_size = metadata_buffer_size
        if blocking_streaming_encoding:
            _enable_blocking_streaming_encoding(
                dataset, encoder_temp_root=encoder_temp_root
            )
        return dataset
    dataset = LeRobotDataset.create(
        repo_id=plan.dataset_uid,
        fps=plan.fps,
        root=root,
        features=plan.feature_schema(),
        robot_type=plan.robot_type,
        use_videos=True,
        rgb_encoder=rgb_encoder,
        metadata_buffer_size=metadata_buffer_size,
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=encoder_queue_maxsize,
        encoder_threads=encoder_threads,
    )
    _install_generated_index_stats_normalizer(dataset)
    if blocking_streaming_encoding:
        _enable_blocking_streaming_encoding(
            dataset, encoder_temp_root=encoder_temp_root
        )
    dataset.meta._metadata_buffer_size = metadata_buffer_size
    return dataset


def _write_resumable_unit(
    plan: DatasetConversionPlan,
    iter_frames: IterFrames,
    root: Path,
    *,
    start: int,
    end: int,
    progress: EtaProgress,
    rgb_encoder: Any,
    streaming_encoding: bool,
    blocking_streaming_encoding: bool,
    encoder_queue_maxsize: int,
    encoder_threads: int | None,
    metadata_buffer_size: int,
    batch_metadata_writes: bool,
    encoder_temp_root: Path | None,
    fragmented_mp4_writes: bool,
    deferred_video_concatenation: bool,
    frame_completed_hook: Callable[[EpisodePlan, int], None] | None,
    episode_completed_hook: Callable[[EpisodePlan, int], None] | None,
) -> None:
    cache_context = _local_datasets_cache(root)
    cache_context.__enter__()
    dataset = None
    current_episode_index = start
    fragmented_context = _fragmented_mp4_writes(fragmented_mp4_writes)
    fragmented_context.__enter__()
    deferred_context = _deferred_video_concatenation(deferred_video_concatenation)
    deferred_context.__enter__()
    try:
        dataset = _open_resumable_writer(
            plan,
            root,
            resume_existing=start > 0,
            rgb_encoder=rgb_encoder,
            streaming_encoding=streaming_encoding,
            blocking_streaming_encoding=blocking_streaming_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
            metadata_buffer_size=metadata_buffer_size,
            encoder_temp_root=encoder_temp_root,
        )
        with _deferred_info_stats_writes(dataset, batch_metadata_writes):
            for episode_index in range(start, end):
                current_episode_index = episode_index
                episode = plan.episodes[episode_index]
                print(
                    f"[{plan.dataset_uid}] episode {episode_index + 1}/{len(plan.episodes)}: "
                    f"{episode.source_relative_path} ({episode.num_frames} frames)",
                    file=sys.stderr,
                    flush=True,
                )
                written = 0
                for frame in iter_frames(episode):
                    if written >= episode.num_frames:
                        raise ConversionError(
                            f"{episode.episode_uid}: reader yielded more than {episode.num_frames} frames"
                        )
                    dataset.add_frame(frame)
                    written += 1
                    progress.update(
                        progress.completed + 1,
                        context=f"{_checkpoint_unit(episode)} episode {episode_index + 1}",
                    )
                    if frame_completed_hook is not None:
                        frame_completed_hook(episode, episode_index)
                if written != episode.num_frames:
                    raise ConversionError(
                        f"{episode.episode_uid}: reader yielded {written} frames, "
                        f"expected {episode.num_frames}"
                    )
                dataset.save_episode()
                if episode_completed_hook is not None:
                    episode_completed_hook(episode, episode_index)
            dataset.finalize()
    except BaseException:
        if dataset is not None:
            with contextlib.suppress(Exception):
                dataset.writer.cancel_pending_videos()
            with contextlib.suppress(Exception):
                dataset.writer.cleanup_interrupted_episode(current_episode_index)
            with contextlib.suppress(Exception):
                dataset.clear_episode_buffer(delete_images=True)
            with contextlib.suppress(Exception):
                dataset.finalize()
        raise
    finally:
        if dataset is not None:
            del dataset
        fragmented_context.__exit__(*sys.exc_info())
        deferred_context.__exit__(*sys.exc_info())
        cache_context.__exit__(*sys.exc_info())


def _convert_dataset_resumable(
    plan: DatasetConversionPlan,
    iter_frames: IterFrames,
    *,
    reader_format: str,
    eta_interval_seconds: float,
    conversion_options: dict[str, Any],
    rgb_encoder: Any,
    streaming_encoding: bool,
    blocking_streaming_encoding: bool,
    encoder_queue_maxsize: int,
    encoder_threads: int | None,
    metadata_buffer_size: int,
    resume_data_root: Path | None,
    resume_state_root: Path | None,
    resume_lock_path: Path | None,
    publish_on_complete: bool,
    cleanup_resume_state: bool,
    rebuild_corrupt_checkpoint: bool,
    batch_metadata_writes: bool,
    encoder_temp_root: Path | None,
    fragmented_mp4_writes: bool,
    deferred_video_concatenation: bool,
    frame_completed_hook: Callable[[EpisodePlan, int], None] | None,
    episode_completed_hook: Callable[[EpisodePlan, int], None] | None,
) -> Path:
    if publish_on_complete and plan.output_path.exists():
        raise FileExistsError(f"output already exists: {plan.output_path}")
    if not plan.episodes:
        raise ConversionError("cannot convert an empty episode plan")
    _validate_checkpoint_unit_order(plan)
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_resume_payload(
        plan,
        reader_format=reader_format,
        conversion_options=conversion_options,
    )
    manager = CheckpointManager(
        plan.output_path,
        payload,
        data_root=resume_data_root,
        state_root=resume_state_root,
        lock_path=resume_lock_path,
        allow_corrupt_rebuild=rebuild_corrupt_checkpoint,
    )
    if not publish_on_complete and manager.data_root != plan.output_path:
        raise ConversionError(
            "in-place resumable conversion requires resume_data_root to equal output_path"
        )
    succeeded = False
    with exclusive_resume_lock(manager.lock_path):
        position = manager.prepare()
        completed = position.completed_episodes
        expected_completed_frames = sum(
            episode.num_frames for episode in plan.episodes[:completed]
        )
        if position.completed_frames != expected_completed_frames:
            raise ConversionError(
                f"resume state records {position.completed_frames} frames, "
                f"expected {expected_completed_frames} for {completed} episodes"
            )
        while completed:
            try:
                if completed > len(plan.episodes):
                    raise ConversionError(
                        f"checkpoint has {completed} episodes, plan has {len(plan.episodes)}"
                    )
                if completed < len(plan.episodes) and _checkpoint_unit(
                    plan.episodes[completed - 1]
                ) == _checkpoint_unit(plan.episodes[completed]):
                    raise ConversionError("checkpoint ends in the middle of a conversion unit")
                validate_written_prefix(plan, manager.data_root, completed)
                validate_video_files(
                    plan,
                    manager.data_root,
                    expected_frames=expected_completed_frames,
                )
            except (ConversionError, OSError, RuntimeError, ValueError):
                if not rebuild_corrupt_checkpoint:
                    raise
                position = manager.rollback_latest()
                completed = position.completed_episodes
                expected_completed_frames = sum(
                    episode.num_frames for episode in plan.episodes[:completed]
                )
                continue
            print(
                f"[{plan.dataset_uid}] reused {position.reused_units} verified checkpoint "
                f"units, {completed} episodes / {expected_completed_frames} frames",
                file=sys.stderr,
                flush=True,
            )
            break
        progress = EtaProgress(
            f"{plan.dataset_uid} convert",
            plan.num_frames,
            "frames",
            interval_seconds=eta_interval_seconds,
            initial_completed=expected_completed_frames,
        )
        started_at = time.monotonic()
        while completed < len(plan.episodes):
            end = _unit_end(plan, completed)
            unit = _checkpoint_unit(plan.episodes[completed])
            previous_paths = {
                str(row["relative_path"]) for row in manager.prior_inventory
            }
            _write_resumable_unit(
                plan,
                iter_frames,
                manager.data_root,
                start=completed,
                end=end,
                progress=progress,
                rgb_encoder=rgb_encoder,
                streaming_encoding=streaming_encoding,
                blocking_streaming_encoding=blocking_streaming_encoding,
                encoder_queue_maxsize=encoder_queue_maxsize,
                encoder_threads=encoder_threads,
                metadata_buffer_size=metadata_buffer_size,
                batch_metadata_writes=batch_metadata_writes,
                encoder_temp_root=encoder_temp_root,
                fragmented_mp4_writes=fragmented_mp4_writes,
                deferred_video_concatenation=deferred_video_concatenation,
                frame_completed_hook=frame_completed_hook,
                episode_completed_hook=episode_completed_hook,
            )
            completed_frames = sum(
                episode.num_frames for episode in plan.episodes[:end]
            )
            validate_written_prefix(plan, manager.data_root, end)
            new_video_paths = {
                path.relative_to(manager.data_root).as_posix()
                for path in (manager.data_root / "videos").rglob("*.mp4")
            } - previous_paths
            unit_frames = sum(
                episode.num_frames for episode in plan.episodes[completed:end]
            )
            validate_video_files(
                plan,
                manager.data_root,
                expected_frames=unit_frames,
                relative_paths=new_video_paths,
            )
            manager.commit(
                checkpoint_unit=unit,
                completed_episodes=end,
                completed_frames=completed_frames,
            )
            completed = end
            progress.update(
                completed_frames,
                context=f"{unit} verified checkpoint committed",
                force=True,
            )

        validate_written_dataset(plan, manager.data_root)
        video_evidence = validate_video_files(
            plan,
            manager.data_root,
            expected_frames=plan.num_frames,
        )
        manifest = build_manifest(plan, reader_format=reader_format)
        manifest.update(
            {
                "num_video_files": sum(len(rows) for rows in video_evidence.values()),
                "video_validation": video_evidence,
                "resume": {
                    "schema_version": payload["resume_schema_version"],
                    "granularity": "reader-defined unit; episode by default",
                    "fingerprint": manager.fingerprint,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "kill_9_limit": (
                        "the active uncommitted unit is discarded on restart; the last "
                        "finalized and re-opened unit remains reusable"
                    ),
                },
            }
        )
        atomic_write_json(manager.data_root / "conversion_manifest.json", manifest)
        if publish_on_complete:
            publish_temporary_output(manager.data_root, plan.output_path, overwrite=False)
        if cleanup_resume_state:
            manager.cleanup_state()
        progress.finish(
            context=(
                "conversion validated and atomically published"
                if publish_on_complete
                else "conversion validated in final-compatible output"
            )
        )
        succeeded = True
    if succeeded:
        manager.lock_path.unlink(missing_ok=True)
    return plan.output_path


def publish_temporary_output(temporary_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if not output_path.exists():
        temporary_path.rename(output_path)
        return
    if not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    backup_path = output_path.with_name(f".{output_path.name}.backup-{uuid.uuid4().hex}")
    output_path.rename(backup_path)
    try:
        temporary_path.rename(output_path)
    except BaseException:
        backup_path.rename(output_path)
        raise
    else:
        shutil.rmtree(backup_path, ignore_errors=True)


def convert_dataset(
    plan: DatasetConversionPlan,
    iter_frames: IterFrames,
    *,
    reader_format: str,
    overwrite: bool = False,
    resume: bool = False,
    eta_interval_seconds: float = 10.0,
    conversion_options: dict[str, Any] | None = None,
    rgb_encoder: Any = None,
    streaming_encoding: bool = False,
    blocking_streaming_encoding: bool = False,
    encoder_queue_maxsize: int = 30,
    encoder_threads: int | None = None,
    frame_completed_hook: Callable[[EpisodePlan, int], None] | None = None,
    episode_completed_hook: Callable[[EpisodePlan, int], None] | None = None,
    metadata_buffer_size: int = 1,
    resume_data_root: Path | None = None,
    resume_state_root: Path | None = None,
    resume_lock_path: Path | None = None,
    publish_on_complete: bool = True,
    cleanup_resume_state: bool = True,
    rebuild_corrupt_checkpoint: bool = False,
    batch_metadata_writes: bool = False,
    encoder_temp_root: Path | None = None,
    fragmented_mp4_writes: bool = False,
    deferred_video_concatenation: bool = False,
) -> Path:
    if resume and overwrite:
        raise ConversionError("resume and overwrite are mutually exclusive")
    if resume:
        if metadata_buffer_size <= 0:
            raise ConversionError("metadata_buffer_size must be positive")
        options = {
            "streaming_encoding": streaming_encoding,
            "blocking_streaming_encoding": blocking_streaming_encoding,
            "encoder_queue_maxsize": encoder_queue_maxsize,
            "encoder_threads": encoder_threads,
            "rgb_encoder": repr(rgb_encoder),
            "metadata_buffer_size": metadata_buffer_size,
            "publish_on_complete": publish_on_complete,
            "cleanup_resume_state": cleanup_resume_state,
            "rebuild_corrupt_checkpoint": rebuild_corrupt_checkpoint,
            "batch_metadata_writes": batch_metadata_writes,
            "encoder_temp_root": str(encoder_temp_root) if encoder_temp_root else None,
            "fragmented_mp4_writes": fragmented_mp4_writes,
            "deferred_video_concatenation": deferred_video_concatenation,
            **(conversion_options or {}),
        }
        return _convert_dataset_resumable(
            plan,
            iter_frames,
            reader_format=reader_format,
            eta_interval_seconds=eta_interval_seconds,
            conversion_options=options,
            rgb_encoder=rgb_encoder,
            streaming_encoding=streaming_encoding,
            blocking_streaming_encoding=blocking_streaming_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
            metadata_buffer_size=metadata_buffer_size,
            resume_data_root=resume_data_root,
            resume_state_root=resume_state_root,
            resume_lock_path=resume_lock_path,
            publish_on_complete=publish_on_complete,
            cleanup_resume_state=cleanup_resume_state,
            rebuild_corrupt_checkpoint=rebuild_corrupt_checkpoint,
            batch_metadata_writes=batch_metadata_writes,
            encoder_temp_root=encoder_temp_root,
            fragmented_mp4_writes=fragmented_mp4_writes,
            deferred_video_concatenation=deferred_video_concatenation,
            frame_completed_hook=frame_completed_hook,
            episode_completed_hook=episode_completed_hook,
        )
    output_path = plan.output_path
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.incomplete-{uuid.uuid4().hex}")
    try:
        write_dataset(
            plan,
            iter_frames,
            temporary_path,
            rgb_encoder=rgb_encoder,
            streaming_encoding=streaming_encoding,
            blocking_streaming_encoding=blocking_streaming_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
            batch_metadata_writes=batch_metadata_writes,
            encoder_temp_root=encoder_temp_root,
            fragmented_mp4_writes=fragmented_mp4_writes,
            deferred_video_concatenation=deferred_video_concatenation,
            frame_completed_hook=frame_completed_hook,
            episode_completed_hook=episode_completed_hook,
        )
        validate_written_dataset(plan, temporary_path)
        manifest = build_manifest(plan, reader_format=reader_format)
        manifest["num_video_files"] = len(list((temporary_path / "videos").rglob("*.mp4")))
        (temporary_path / "conversion_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_temporary_output(temporary_path, output_path, overwrite=overwrite)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
    return output_path


def resolve_dataset_uids(raw_root: Path, requested_uid: str | None, convert_all: bool) -> list[str]:
    if convert_all:
        if not raw_root.is_dir():
            raise ConversionError(f"raw root does not exist: {raw_root}")
        uids = sorted((path.name for path in raw_root.iterdir() if path.is_dir()), key=str.casefold)
        if not uids:
            raise ConversionError(f"raw root contains no dataset directories: {raw_root}")
        return uids
    if requested_uid is None:
        raise ConversionError("provide exactly one of --dataset-uid or --all")
    return [requested_uid]
