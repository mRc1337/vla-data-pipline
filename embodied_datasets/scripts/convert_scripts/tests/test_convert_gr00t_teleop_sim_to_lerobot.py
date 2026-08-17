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


def test_parallel_parts_publish_one_equivalent_collection(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_fixture(raw_root)
    collection = _inspect(raw_root, tmp_path / "staging")
    first = collection.parts[0]
    second = replace(first, output_name="part-001-synthetic-copy")
    collection.parts = [first, second]

    output = gr00t.convert_collection(
        collection, overwrite=False, eta_interval_seconds=0.001, workers=2
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
