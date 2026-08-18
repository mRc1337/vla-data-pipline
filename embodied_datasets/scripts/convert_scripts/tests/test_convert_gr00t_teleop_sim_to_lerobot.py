import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import convert_gr00t_teleop_sim_to_lerobot as gr00t
from convert_core.progress import EtaProgress, format_duration


TASK = "SyntheticPickAndPlace"
VIDEO_KEY = "observation.images.ego_view"


def test_progress_is_rate_limited_and_reports_eta():
    now = [100.0]
    stream = io.StringIO()
    progress = EtaProgress(
        "preflight", 100, "episode", interval_seconds=5, clock=lambda: now[0], stream=stream
    )
    now[0] = 104.0
    progress.update(10)
    assert stream.getvalue() == ""
    now[0] = 110.0
    progress.update(20, context="part 1")
    assert "20/100 (20.0%)" in stream.getvalue()
    assert "2.00 episode/s" in stream.getvalue()
    assert "ETA 00:00:40 | part 1" in stream.getvalue()
    assert format_duration(2 * 86_400 + 3_661) == "2d 01:01:01"


def test_video_episode_stats_have_one_arrow_compatible_float_dtype(monkeypatch):
    import lerobot.datasets.compute_stats as compute_stats

    monkeypatch.setattr(
        gr00t,
        "_decode_sampled_rgb",
        lambda _path, _length: np.zeros((2, 3, 256, 256), dtype=np.uint8),
    )
    monkeypatch.setattr(
        compute_stats,
        "get_feature_stats",
        lambda _array, axis, keepdims: {
            "min": np.zeros((1, 3, 1, 1), dtype=np.float32),
            "q50": np.zeros((1, 3, 1, 1), dtype=np.float64),
            "count": np.array([2], dtype=np.int64),
        },
    )

    part = SimpleNamespace(features={VIDEO_KEY: {"dtype": "video"}})
    episode = SimpleNamespace(source_video=Path("unused.mp4"), length=2)
    stats = gr00t._episode_stats(part, episode, table=None)[VIDEO_KEY]

    assert stats["min"].dtype == np.float64
    assert stats["q50"].dtype == np.float64
    assert stats["count"].dtype == np.int64


def test_boolean_edge_quantiles_are_canonicalized_before_arrow_concatenation():
    from datasets import Dataset
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.utils.utils import flatten_dict

    short_done = np.zeros(57, dtype=np.bool_)
    short_done[-1] = True
    regular_done = np.zeros(117, dtype=np.bool_)
    regular_done[-1] = True

    raw_short = get_feature_stats(short_done, axis=0, keepdims=True)
    raw_regular = get_feature_stats(regular_done, axis=0, keepdims=True)
    assert raw_short["q99"].dtype == np.float64
    assert raw_regular["q99"].dtype == np.float32

    part = SimpleNamespace(features={"next.done": {"dtype": "bool"}})
    episode = SimpleNamespace()
    short_stats = gr00t._episode_stats(
        part, episode, pa.table({"next.done": short_done})
    )
    regular_stats = gr00t._episode_stats(
        part, episode, pa.table({"next.done": regular_done})
    )

    assert short_stats["next.done"]["q99"].dtype == np.float64
    assert regular_stats["next.done"]["q99"].dtype == np.float64
    rows = [flatten_dict({"stats": stats}) for stats in (short_stats, regular_stats)]
    dataset = Dataset.from_list(rows)
    assert dataset.num_rows == 2
    aggregated = aggregate_stats([short_stats, regular_stats])
    assert aggregated["next.done"]["q99"].dtype == np.float64


def test_episode_stats_schema_is_checked_before_video_remux():
    first = {
        "next.done": {
            "q99": np.array([0.0], dtype=np.float64),
            "count": np.array([57], dtype=np.int64),
        }
    }
    second = {
        "next.done": {
            "q99": np.array([0.0], dtype=np.float32),
            "count": np.array([117], dtype=np.int64),
        }
    }
    part = SimpleNamespace(
        source_task=TASK,
        episodes=[SimpleNamespace(episode_index=0), SimpleNamespace(episode_index=1)],
    )

    with pytest.raises(
        gr00t.ConversionError,
        match=r"source episode 1 for next\.done/q99: expected .*float64.* got .*float32",
    ):
        gr00t._validate_episode_stats_schema(part, [first, second])


def _config(dataset_uid: str = "gr00t_fixture") -> gr00t.Config:
    return gr00t.Config(
        dataset_uid=dataset_uid,
        source_lerobot_subdir="LeRobot",
        source_hdf5_subdir="HDF5",
        source_part_glob="gr1_unified.*",
        source_version="v2.0",
        target_version="v3.0",
        robot_type="GR1ArmsAndWaistFourierHands",
        fps=20,
        task_source="episode_remarks",
        preserve_meta_files=("embodiment.json", "modality.json"),
        data_file_size_in_mb=100,
        video_file_size_in_mb=500,
        crosscheck_hdf5=True,
    )


def _source_features() -> dict:
    scalar_features = {
        "timestamp": "float64",
        "next.reward": "float64",
        "next.done": "bool",
        "task_index": "int64",
        "annotation.human.fine_action": "int64",
        "annotation.human.coarse_action": "int64",
        "episode_index": "int64",
        "index": "int64",
    }
    features = {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": 20.0,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.state": {"dtype": "object", "shape": [44]},
        "action": {"dtype": "object", "shape": [44]},
    }
    features.update({key: {"dtype": dtype, "shape": [1]} for key, dtype in scalar_features.items()})
    return features


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width = 256
        stream.height = 256
        stream.pix_fmt = "yuv420p"
        for color in colors:
            rgb = np.empty((256, 256, 3), dtype=np.uint8)
            rgb[:] = color
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_fixture(
    root: Path,
    *,
    instructions: tuple[str, ...] = ("pick up the red block", "place the red block"),
    bad_action_width: bool = False,
) -> gr00t.Config:
    config = _config()
    dataset_root = root / config.dataset_uid
    part_root = dataset_root / "LeRobot" / f"gr1_unified.{TASK}"
    meta = part_root / "meta"
    meta.mkdir(parents=True)
    episode_lengths = [3] * len(instructions)
    total_frames = sum(episode_lengths)
    info = {
        "codebase_version": "v2.0",
        "robot_type": config.robot_type,
        "total_episodes": len(instructions),
        "total_frames": total_frames,
        "total_tasks": 2,
        "total_videos": len(instructions),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20.0,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _source_features(),
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": ""})
        + "\n"
        + json.dumps({"task_index": 1, "task": TASK})
        + "\n",
        encoding="utf-8",
    )
    for name in ("modality.json", "embodiment.json"):
        (meta / name).write_text("{}\n", encoding="utf-8")

    episode_rows = []
    global_index = 0
    for episode_index, (length, instruction) in enumerate(zip(episode_lengths, instructions, strict=True)):
        trajectory_id = f"{TASK}-{episode_index + 10:05d}"
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [TASK],
                "length": length,
                "trajectory_id": trajectory_id,
                "remarks": instruction,
            }
        )
        width = 43 if bad_action_width and episode_index == 1 else 44
        state = np.arange(length * 44, dtype=np.float64).reshape(length, 44) + episode_index * 1000
        action = np.arange(length * width, dtype=np.float64).reshape(length, width) + episode_index * 2000
        table = pa.table(
            {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                # Match the simulator tick in the public release; the
                # converter must validate but never normalize these values.
                "timestamp": np.arange(length, dtype=np.float64) * 0.04999995231628418,
                "next.reward": np.arange(length, dtype=np.float64),
                "next.done": np.arange(length) == length - 1,
                "task_index": np.ones(length, dtype=np.int64),
                "annotation.human.fine_action": np.full(length, -1, dtype=np.int64),
                "annotation.human.coarse_action": np.full(length, -1, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(global_index, global_index + length, dtype=np.int64),
            }
        )
        parquet_path = part_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        video_path = part_root / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{episode_index:06d}.mp4"
        _write_video(video_path, [(20 + episode_index, 40 + frame, 60) for frame in range(length)])
        global_index += length
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows), encoding="utf-8"
    )

    hdf5_path = dataset_root / "HDF5" / f"{TASK}.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    with h5py.File(hdf5_path, "w") as h5_file:
        data = h5_file.create_group("data")
        for episode_index, (length, instruction) in enumerate(zip(episode_lengths, instructions, strict=True)):
            demo = data.create_group(f"demo_{episode_index + 10}")
            demo.attrs["num_samples"] = length
            demo.attrs["ep_meta"] = json.dumps({"lang": instruction})
            # The official raw HDF5 representation is heterogeneous.  The
            # converter records these schemas but does not force them into the
            # independent 44-D LeRobot representation.
            demo.create_dataset("states", data=np.zeros((length, 145 + 13 * episode_index)))
            demo.create_dataset("actions", data=np.zeros((length, 24)))
            action_dict = demo.create_group("action_dict")
            action_dict.create_dataset("gripper", data=np.zeros((length, 1), dtype=np.float32))
            action_dict.create_dataset("rel_pos", data=np.zeros((length, 3), dtype=np.float32))
            action_dict.create_dataset("rel_rot_6d", data=np.zeros((length, 6), dtype=np.float32))
            action_dict.create_dataset(
                "rel_rot_axis_angle", data=np.zeros((length, 3), dtype=np.float32)
            )
    return config


def _inspect(raw_root: Path, staging_root: Path) -> gr00t.Collection:
    return gr00t.inspect_collection(
        _config(), raw_root, staging_root, eta_interval_seconds=0.001, full_video_scan=True
    )


def _two_part_collection(tmp_path: Path) -> gr00t.Collection:
    raw_root = tmp_path / "raw"
    _write_fixture(raw_root)
    collection = _inspect(raw_root, tmp_path / "staging")
    first = collection.parts[0]
    collection.parts = [first, replace(first, output_name="part-001-synthetic-copy")]
    return collection


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", "20", "fps must be an integer"),
        ("crosscheck_hdf5", 1, "crosscheck_hdf5 must be boolean"),
        ("preserve_meta_files", "modality.json", "preserve_meta_files must contain"),
        ("data_file_size_in_mb", 0, "file-size limits must be positive"),
        ("robot_type", "unknown", "robot_type must match"),
    ],
)
def test_load_config_rejects_invalid_values(tmp_path: Path, field: str, value, message: str):
    config = _config()
    config_dict = {**config.__dict__, "preserve_meta_files": list(config.preserve_meta_files)}
    config_dict[field] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict))

    with pytest.raises(gr00t.ConversionError, match=message):
        gr00t.load_config(config_path)


def test_preflight_remaps_official_language_and_accepts_heterogeneous_hdf5(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_fixture(raw_root)

    collection = _inspect(raw_root, tmp_path / "staging")

    part = collection.parts[0]
    assert collection.episodes == 2
    assert collection.frames == 6
    assert part.tasks == {0: "", 1: "pick up the red block", 2: "place the red block"}
    assert [episode.mapped_task_index for episode in part.episodes] == [1, 2]
    assert len(part.hdf5_schemas) == 2
    first_schema = next(iter(part.hdf5_schemas))
    assert ("action_dict/gripper", (1,), "float32") in first_schema
    assert ("action_dict/rel_pos", (3,), "float32") in first_schema
    assert ("action_dict/rel_rot_6d", (6,), "float32") in first_schema
    assert ("action_dict/rel_rot_axis_angle", (3,), "float32") in first_schema
    assert part.features["observation.state"]["dtype"] == "float64"
    assert part.features["observation.state"]["names"] == list(gr00t.GR1_NAMES)
    assert part.features["action"]["fps"] == 20


def test_preflight_rejects_a_non_44d_lerobot_action(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_fixture(raw_root, bad_action_width=True)

    with pytest.raises(gr00t.ConversionError, match="action contains non-44-D rows"):
        _inspect(raw_root, tmp_path / "staging")


def test_dry_run_writes_no_staging_output(tmp_path: Path, capsys):
    raw_root = tmp_path / "raw"
    config = _write_fixture(raw_root)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({**config.__dict__, "preserve_meta_files": list(config.preserve_meta_files)}))

    result = gr00t.main(
        [
            "--config", str(config_path),
            "--raw-root", str(raw_root),
            "--staging-root", str(tmp_path / "staging"),
            "--dry-run",
            "--eta-interval-seconds", "0.001",
        ]
    )

    assert result == 0
    assert not (tmp_path / "staging").exists()
    assert "preflight complete; no output written" in capsys.readouterr().out


def test_conversion_reopens_and_preserves_values_tasks_and_video(tmp_path: Path, monkeypatch):
    raw_root = tmp_path / "raw"
    _write_fixture(raw_root)
    collection = _inspect(raw_root, tmp_path / "staging")

    output = gr00t.convert_collection(collection, overwrite=False, eta_interval_seconds=0.001)

    part = collection.parts[0]
    part_output = output / part.output_name
    assert (part_output / "meta" / "info.json").is_file()
    assert (part_output / "meta" / "stats.json").is_file()
    manifest = json.loads((part_output / "conversion_manifest.json").read_text())
    assert manifest["video"]["reencoded"] is False
    assert manifest["task_mapping"]["target_tasks"]["2"] == "place the red block"
    assert manifest["official_references"]["dataset_card_and_files"]["commit"] == gr00t.SOURCE_COMMIT
    assert manifest["source_splits"] == manifest["target_splits"] == {"train": "0:100"}
    assert manifest["copied_meta_files"] == ["embodiment.json", "modality.json"]
    nested_paths = {
        field["path"]
        for variant in manifest["hdf5_crosscheck"]["schema_variants"]
        for field in variant
    }
    assert "action_dict/rel_rot_6d" in nested_paths
    mapping = {row["target"]: row for row in manifest["field_mapping"]}
    for target in (
        "data:observation.state",
        "data:action",
        "videos:observation.images.ego_view",
        "data:timestamp",
        "data:next.reward",
        "data:next.done",
        "tasks.parquet task + data:task_index + episodes.parquet:tasks",
        "data:annotation.human.fine_action",
        "data:annotation.human.coarse_action",
        "data:episode_index",
        "data:index",
    ):
        assert target in mapping
        assert mapping[target]["basis"]
    for name in ("embodiment.json", "modality.json"):
        assert (part_output / "meta" / name).read_bytes() == (
            part.source_root / "meta" / name
        ).read_bytes()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")

    dataset = LeRobotDataset(repo_id=part.output_name, root=part_output, video_backend="pyav")
    assert dataset.num_episodes == 2
    assert len(dataset) == 6
    assert dataset[0]["task"] == "pick up the red block"
    assert dataset[3]["task"] == "place the red block"
    expected = pq.read_table(part.episodes[1].source_parquet)["action"][1].as_py()
    assert np.array_equal(dataset[4]["action"].numpy(), np.asarray(expected))
    del dataset

    # The validator scans every packed episode boundary, not just the sampled
    # first/middle/last numeric values. Corrupt the second episode's first row.
    packed_path = part_output / "data" / "chunk-000" / "file-000.parquet"
    packed = pq.read_table(packed_path)
    episode_indices = packed["episode_index"].to_pylist()
    episode_indices[3] = 0
    packed = packed.set_column(
        packed.schema.get_field_index("episode_index"), "episode_index", pa.array(episode_indices)
    )
    pq.write_table(packed, packed_path)
    with pytest.raises(gr00t.ConversionError, match="packed episode boundary/index mismatch at 1"):
        gr00t.validate_part(part, part_output, collection.config)


def test_failed_conversion_removes_temporary_collection(tmp_path: Path, monkeypatch):
    output = tmp_path / "staging" / "lerobot_v3_0" / "failed"
    collection = gr00t.Collection(
        config=_config(),
        raw_dataset_root=tmp_path / "raw",
        output_path=output,
        parts=[SimpleNamespace(source_task=TASK, output_name="part-000-test")],
    )

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(gr00t, "convert_part", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        gr00t.convert_collection(collection, overwrite=False, eta_interval_seconds=1)

    assert not output.exists()
    assert not list(output.parent.glob(".failed.incomplete-*"))


def test_resume_reuses_verified_part_and_rebuilds_partial_part(tmp_path: Path, monkeypatch):
    collection = _two_part_collection(tmp_path)
    first, second = collection.parts
    original_convert_part = gr00t.convert_part
    calls: list[str] = []

    def interrupt_second(part, output_root, config, *, eta_interval_seconds):
        calls.append(part.output_name)
        if part.output_name == second.output_name:
            output_root.mkdir(parents=True)
            (output_root / "partial-junk").write_text("incomplete", encoding="utf-8")
            raise RuntimeError("synthetic interruption")
        original_convert_part(
            part, output_root, config, eta_interval_seconds=eta_interval_seconds
        )

    monkeypatch.setattr(gr00t, "convert_part", interrupt_second)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        gr00t.convert_collection(
            collection,
            overwrite=False,
            eta_interval_seconds=0.001,
            resume=True,
        )

    resume_data, resume_state, _ = gr00t._resume_paths(collection.output_path)
    assert not collection.output_path.exists()
    assert (resume_data / first.output_name).is_dir()
    assert (resume_data / second.output_name / "partial-junk").is_file()
    assert gr00t._resume_marker_path(resume_state, first).is_file()
    assert not gr00t._resume_marker_path(resume_state, second).exists()

    calls.clear()

    def finish_pending(part, output_root, config, *, eta_interval_seconds):
        calls.append(part.output_name)
        assert part.output_name == second.output_name
        assert not output_root.exists()
        original_convert_part(
            part, output_root, config, eta_interval_seconds=eta_interval_seconds
        )

    monkeypatch.setattr(gr00t, "convert_part", finish_pending)
    output = gr00t.convert_collection(
        collection,
        overwrite=False,
        eta_interval_seconds=0.001,
        resume=True,
    )

    assert calls == [second.output_name]
    assert output.is_dir()
    assert not resume_data.exists()
    assert not resume_state.exists()
    assert not list(output.rglob("state.json"))
    assert not list(output.rglob("partial-junk"))


def test_resume_rejects_changed_conversion_fingerprint(tmp_path: Path, monkeypatch):
    collection = _two_part_collection(tmp_path)

    def interrupt(*args, **kwargs):
        raise RuntimeError("stop after creating state")

    monkeypatch.setattr(gr00t, "convert_part", interrupt)
    with pytest.raises(RuntimeError, match="stop after creating state"):
        gr00t.convert_collection(
            collection,
            overwrite=False,
            eta_interval_seconds=0.001,
            resume=True,
        )

    _, resume_state, _ = gr00t._resume_paths(collection.output_path)
    state_before = (resume_state / gr00t.RESUME_STATE_FILE).read_bytes()
    collection.config = replace(collection.config, data_file_size_in_mb=101)
    with pytest.raises(gr00t.ConversionError, match="does not match this conversion"):
        gr00t.convert_collection(
            collection,
            overwrite=False,
            eta_interval_seconds=0.001,
            resume=True,
        )
    assert (resume_state / gr00t.RESUME_STATE_FILE).read_bytes() == state_before


def test_resume_fingerprint_does_not_resolve_episode_paths(tmp_path: Path, monkeypatch):
    collection = _two_part_collection(tmp_path)
    expected_parquet = str(collection.parts[0].episodes[0].source_parquet)
    expected_video = str(collection.parts[0].episodes[0].source_video)

    def reject_filesystem_resolution(self, *args, **kwargs):
        raise AssertionError(f"fingerprinting tried to resolve {self}")

    monkeypatch.setattr(Path, "resolve", reject_filesystem_resolution)

    payload = gr00t._resume_fingerprint_payload(collection)

    first_episode = payload["parts"][0]["episodes"][0]
    assert first_episode["source_parquet"] == expected_parquet
    assert first_episode["source_video"] == expected_video
    assert gr00t._resume_fingerprint(collection)


def test_lexical_absolute_path_matches_resolve_for_production_style_paths(tmp_path: Path):
    absolute_path = tmp_path / "raw" / "episode.parquet"

    assert gr00t._lexical_absolute_path(absolute_path) == str(absolute_path.resolve())


def test_resume_rebuilds_corrupt_completed_part(tmp_path: Path, monkeypatch):
    collection = _two_part_collection(tmp_path)
    first = collection.parts[0]
    resume_data, resume_state, _ = gr00t._resume_paths(collection.output_path)
    fingerprint = gr00t._resume_fingerprint(collection)
    gr00t._prepare_resume_workspace(collection, resume_data, resume_state, fingerprint)
    gr00t.convert_part(
        first,
        resume_data / first.output_name,
        collection.config,
        eta_interval_seconds=0.001,
    )
    gr00t._write_json_atomic(
        gr00t._resume_marker_path(resume_state, first),
        gr00t._resume_marker_payload(fingerprint, first),
    )
    (resume_data / first.output_name / "meta" / "info.json").unlink()

    original_convert_part = gr00t.convert_part
    rebuilt: list[str] = []

    def track_rebuild(part, output_root, config, *, eta_interval_seconds):
        rebuilt.append(part.output_name)
        if part.output_name == first.output_name:
            assert not output_root.exists()
        original_convert_part(
            part, output_root, config, eta_interval_seconds=eta_interval_seconds
        )

    monkeypatch.setattr(gr00t, "convert_part", track_rebuild)
    gr00t.convert_collection(
        collection,
        overwrite=False,
        eta_interval_seconds=0.001,
        resume=True,
    )
    assert rebuilt == [part.output_name for part in collection.parts]


def test_resume_lock_rejects_concurrent_conversion(tmp_path: Path):
    collection = _two_part_collection(tmp_path)
    _, _, lock_path = gr00t._resume_paths(collection.output_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = gr00t._acquire_resume_lock(lock_path)
    try:
        with pytest.raises(gr00t.ConversionError, match="another resume process"):
            gr00t.convert_collection(
                collection,
                overwrite=False,
                eta_interval_seconds=0.001,
                resume=True,
            )
    finally:
        gr00t._release_resume_lock(descriptor)

    assert not collection.output_path.exists()


def test_parallel_parts_publish_one_equivalent_collection(tmp_path: Path):
    collection = _two_part_collection(tmp_path)
    first, second = collection.parts

    output = gr00t.convert_collection(
        collection,
        overwrite=False,
        eta_interval_seconds=0.001,
        workers=2,
        resume=True,
    )

    manifest = json.loads((output / "collection_manifest.json").read_text())
    assert manifest["total_episodes"] == 4
    assert manifest["total_frames"] == 12
    assert [row["path"] for row in manifest["parts"]] == [
        first.output_name,
        second.output_name,
    ]
    first_data = pq.read_table(output / first.output_name / "data/chunk-000/file-000.parquet")
    second_data = pq.read_table(output / second.output_name / "data/chunk-000/file-000.parquet")
    assert first_data.equals(second_data)
    assert not list(output.parent.glob(f".{output.name}.incomplete-*"))
    resume_data, resume_state, resume_lock = gr00t._resume_paths(output)
    assert not resume_data.exists()
    assert not resume_state.exists()
    assert resume_lock.is_file()


def test_parallel_worker_failure_removes_temporary_collection(tmp_path: Path):
    output = tmp_path / "staging" / "lerobot_v3_0" / "parallel-failed"
    collection = gr00t.Collection(
        config=_config(),
        raw_dataset_root=tmp_path / "raw",
        output_path=output,
        parts=[
            SimpleNamespace(source_task="broken-a", output_name="part-000-broken-a"),
            SimpleNamespace(source_task="broken-b", output_name="part-001-broken-b"),
        ],
    )

    with pytest.raises(AttributeError):
        gr00t.convert_collection(
            collection, overwrite=False, eta_interval_seconds=1, workers=2
        )

    assert not output.exists()
    assert not list(output.parent.glob(".parallel-failed.incomplete-*"))


def test_skip_existing_returns_before_reading_source(tmp_path: Path, capsys):
    staging_root = tmp_path / "staging"
    output = staging_root / "lerobot_v3_0" / "already-there"
    output.mkdir(parents=True)
    config = _config(dataset_uid="already-there")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({**config.__dict__, "preserve_meta_files": list(config.preserve_meta_files)})
    )

    result = gr00t.main(
        [
            "--config",
            str(config_path),
            "--raw-root",
            str(tmp_path / "missing-raw"),
            "--staging-root",
            str(staging_root),
            "--skip-existing",
        ]
    )

    assert result == 0
    assert f"skipped existing output: {output}" in capsys.readouterr().out
