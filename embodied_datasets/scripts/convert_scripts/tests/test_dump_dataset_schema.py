import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import dump_dataset_schema as dds


def _write_hdf5(path: Path, *, num_frames: int = 4, width: int = 14, fps: float = 20.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        observations = h5_file.create_group("observations")
        qpos = observations.create_dataset("qpos", data=np.zeros((num_frames, width), dtype="float32"))
        qpos.attrs["fps"] = fps


def _write_rlds_version_dir(version_dir: Path, *, dataset_info: dict, features: dict) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "dataset_info.json").write_text(json.dumps(dataset_info), encoding="utf-8")
    (version_dir / "features.json").write_text(json.dumps(features), encoding="utf-8")
    (version_dir / "my_dataset-train.tfrecord-00000-of-00001").write_bytes(b"not-a-real-tfrecord")


def test_hdf5_detected_and_header_reports_shape_dtype_attrs_without_reading_values(tmp_path: Path):
    root = tmp_path / "ds"
    _write_hdf5(root / "episode_0.hdf5", num_frames=100, width=14, fps=30.0)

    report = dds.dump_schema(root)

    assert report["detected_format"] == "hdf5"
    assert "episode_0.hdf5" in report["format_detection_evidence"]
    header = report["schema"]["per_file"]["episode_0.hdf5"]
    dataset = header["datasets"]["/observations/qpos"]
    assert dataset["shape"] == [100, 14]
    assert dataset["dtype"] == "float32"
    assert dataset["attrs"]["fps"] == 30.0


def test_hdf5_schema_diff_reported_when_sample_files_disagree(tmp_path: Path):
    root = tmp_path / "ds"
    _write_hdf5(root / "episode_0.hdf5", width=14)
    _write_hdf5(root / "episode_1.hdf5", width=16)

    report = dds.dump_schema(root, hdf5_sample_files=2)

    schema = report["schema"]
    assert schema["consistent_across_sample"] is False
    assert schema["schema_diff"]
    assert "changed=" in schema["schema_diff"][0]


def test_hdf5_schema_consistent_ignores_varying_frame_count(tmp_path: Path):
    root = tmp_path / "ds"
    _write_hdf5(root / "episode_0.hdf5", num_frames=10, width=14)
    _write_hdf5(root / "episode_1.hdf5", num_frames=55, width=14)

    report = dds.dump_schema(root, hdf5_sample_files=2)

    assert report["schema"]["consistent_across_sample"] is True


def test_rlds_detected_via_dataset_info_json(tmp_path: Path):
    version_dir = tmp_path / "ds" / "my_dataset" / "1.0.0"
    _write_rlds_version_dir(
        version_dir,
        dataset_info={"name": "my_dataset"},
        features={"pythonClassName": "tfds.features.FeaturesDict"},
    )

    report = dds.dump_schema(tmp_path / "ds")

    assert report["detected_format"] == "rlds"
    assert "dataset_info.json" in report["format_detection_evidence"]


def test_rlds_falls_back_to_raw_sidecar_json_without_tensorflow_datasets_and_never_reads_tfrecord(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    version_dir = tmp_path / "ds" / "my_dataset" / "1.0.0"
    dataset_info = {"name": "my_dataset", "splits": [{"name": "train", "numExamples": "42"}]}
    features = {"pythonClassName": "tfds.features.FeaturesDict"}
    _write_rlds_version_dir(version_dir, dataset_info=dataset_info, features=features)

    original_read_bytes = Path.read_bytes

    def _guard_read_bytes(self: Path, *args, **kwargs):
        if self.name.startswith("my_dataset-train.tfrecord"):
            raise AssertionError("must not read .tfrecord payload")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _guard_read_bytes)
    monkeypatch.setattr(dds, "_require_tfds", lambda: None)

    report = dds.dump_schema(tmp_path / "ds")

    assert report["detected_format"] == "rlds"
    dataset_report = report["schema"]["datasets"][0]
    assert dataset_report["raw_sidecar_json"]["dataset_info.json"] == dataset_info
    assert dataset_report["raw_sidecar_json"]["features.json"] == features
    assert "install tensorflow" in dataset_report["fidelity"]


def test_unknown_format_falls_back_to_generic_without_erroring(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    (root / "frame_000.npy").write_bytes(b"\x00" * 10)
    (root / "labels.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    report = dds.dump_schema(root)

    assert report["detected_format"] == "unknown"
    assert report["file_inventory"]["total_files"] == 2
    assert report["schema"]["note"]


def test_directory_tree_truncates_wide_directories_with_explicit_marker(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    for index in range(120):
        (root / f"episode_{index:04d}.hdf5").write_bytes(b"")

    report = dds.dump_schema(root, max_tree_entries=10)

    children = report["directory_tree"]["children"]
    file_nodes = [child for child in children if child.get("type") == "file"]
    notes = [child["note"] for child in children if "note" in child]
    assert len(file_nodes) == 10
    assert any("more files omitted" in note for note in notes)
    # Truncated display must not affect the exhaustive file inventory count.
    assert report["file_inventory"]["total_files"] == 120


def test_directory_tree_respects_depth_limit_with_explicit_marker(tmp_path: Path):
    root = tmp_path / "ds"
    nested = root
    for level in range(6):
        nested = nested / f"level{level}"
    nested.mkdir(parents=True)
    (nested / "deep.txt").write_text("x", encoding="utf-8")

    report = dds.dump_schema(root, max_tree_depth=2)

    def _collect_notes(node: dict) -> list[str]:
        notes = [child["note"] for child in node.get("children", []) if "note" in child]
        for child in node.get("children", []):
            if child.get("type") == "dir":
                notes.extend(_collect_notes(child))
        return notes

    assert any("depth limit" in note for note in _collect_notes(report["directory_tree"]))
    # Depth-limited display must not affect the exhaustive file inventory count.
    assert report["file_inventory"]["total_files"] == 1


def test_cli_all_mode_writes_one_report_per_dataset_and_continues_past_bad_ones(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_hdf5(raw_root / "good_uid" / "episode_0.hdf5")
    broken_dir = raw_root / "broken_uid"
    broken_dir.mkdir(parents=True)
    (broken_dir / "episode_0.hdf5").write_bytes(b"not a real hdf5 file")

    output_dir = tmp_path / "reports"
    exit_code = dds.main(["--raw-root", str(raw_root), "--all", "--output-dir", str(output_dir)])

    assert exit_code == 0
    good_report = json.loads((output_dir / "good_uid.json").read_text(encoding="utf-8"))
    assert good_report["detected_format"] == "hdf5"
    assert "error" not in good_report["schema"]["per_file"]["episode_0.hdf5"]

    broken_report = json.loads((output_dir / "broken_uid.json").read_text(encoding="utf-8"))
    assert broken_report["detected_format"] == "hdf5"
    assert broken_report["schema"]["per_file"]["episode_0.hdf5"]["error"]


def test_cli_single_dataset_prints_json_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    root = tmp_path / "ds"
    _write_hdf5(root / "episode_0.hdf5")

    exit_code = dds.main(["--dataset-root", str(root)])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["detected_format"] == "hdf5"


def test_cli_single_dataset_writes_output_file(tmp_path: Path):
    root = tmp_path / "ds"
    _write_hdf5(root / "episode_0.hdf5")
    output_path = tmp_path / "report.json"

    exit_code = dds.main(["--dataset-root", str(root), "--output", str(output_path)])

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["detected_format"] == "hdf5"


def test_cli_raw_root_requires_all_flag(tmp_path: Path):
    with pytest.raises(SystemExit):
        dds.main(["--raw-root", str(tmp_path)])


def _write_raw_image_json_episode(episode_dir: Path, *, num_frames: int = 3) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    for index in range(num_frames):
        (episode_dir / f"frame_{index:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    metadata = {
        "language_instruction": "pick up the cup",
        "fps": 10,
        "frames": [{"state": [0.0, 1.0], "action": [0.0], "image": f"frame_{i:04d}.jpg"} for i in range(num_frames)],
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_raw_image_json_detected_via_image_plus_json_sidecar(tmp_path: Path):
    root = tmp_path / "ds"
    _write_raw_image_json_episode(root / "episode_0", num_frames=3)

    report = dds.dump_schema(root)

    assert report["detected_format"] == "raw_image_json"
    assert "metadata.json" in report["format_detection_evidence"]


def test_raw_image_json_schema_summarizes_structure_without_reading_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "ds"
    _write_raw_image_json_episode(root / "episode_0", num_frames=4)
    _write_raw_image_json_episode(root / "episode_1", num_frames=2)

    original_read_bytes = Path.read_bytes

    def _guard_read_bytes(self: Path, *args, **kwargs):
        if self.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
            raise AssertionError("must not read image bytes")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _guard_read_bytes)
    report = dds.dump_schema(root)

    assert report["detected_format"] == "raw_image_json"
    schema = report["schema"]
    assert schema["total_candidate_episode_dirs"] == 2
    episode_0 = next(e for e in schema["sampled_episodes"] if e["episode_dir"] == "episode_0")
    assert episode_0["num_image_files"] == 4
    sidecar = episode_0["sidecars"]["metadata.json"]
    assert sidecar["type"] == "object"
    assert sidecar["frames_count"] == 4
    assert sorted(sidecar["frame_0_keys"]) == ["action", "image", "state"]


def test_raw_image_json_does_not_shadow_rlds_or_hdf5_detection(tmp_path: Path):
    # An HDF5 dataset root that happens to also contain unrelated image+json
    # files elsewhere must still be detected as hdf5, not raw_image_json.
    root = tmp_path / "ds"
    _write_hdf5(root / "episode_0.hdf5")
    _write_raw_image_json_episode(root / "unrelated_images_dir")

    report = dds.dump_schema(root)

    assert report["detected_format"] == "hdf5"


def test_tfds_sidecar_filenames_are_not_misdetected_as_raw_image_json(tmp_path: Path):
    # dataset_info.json/features.json living alongside stray image files
    # (e.g. thumbnails) must not be treated as a raw_image_json JSON sidecar.
    version_dir = tmp_path / "ds" / "my_dataset" / "1.0.0"
    _write_rlds_version_dir(
        version_dir,
        dataset_info={"name": "my_dataset"},
        features={"pythonClassName": "tfds.features.FeaturesDict"},
    )
    (version_dir / "thumbnail.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    report = dds.dump_schema(tmp_path / "ds")

    assert report["detected_format"] == "rlds"

