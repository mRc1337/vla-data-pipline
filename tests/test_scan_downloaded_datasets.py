from common.scan_downloaded_datasets import FoundDataset, scan_raw_directory


def test_scan_raw_directory_matches_exact_id(tmp_path):
    data_root = tmp_path / "data"
    raw = data_root / "public_datasets_raw" / "droid"
    raw.mkdir(parents=True)
    (raw / "episode_0.hdf5").write_bytes(b"x" * 100)

    found, unmatched = scan_raw_directory(data_root, [("droid", "DROID"), ("libero", "LIBERO")])
    assert found == [FoundDataset("droid", "droid", 100)]
    assert unmatched == []


def test_scan_raw_directory_matches_folder_named_after_registry_name(tmp_path):
    data_root = tmp_path / "data"
    raw = data_root / "public_datasets_raw" / "AgiBot-World"
    raw.mkdir(parents=True)
    (raw / "a.bin").write_bytes(b"x" * 50)

    found, unmatched = scan_raw_directory(
        data_root, [("agibot_world", "AgiBot-World")]
    )
    assert found == [FoundDataset("agibot_world", "AgiBot-World", 50)]
    assert unmatched == []


def test_scan_raw_directory_uses_alias_table_for_semantic_mismatches(tmp_path):
    data_root = tmp_path / "data"
    raw = data_root / "public_datasets_raw" / "RobotSet"
    raw.mkdir(parents=True)
    (raw / "a.bin").write_bytes(b"x" * 10)

    found, unmatched = scan_raw_directory(data_root, [("roboset", "RoboSet")])
    assert found == [FoundDataset("roboset", "RobotSet", 10)]
    assert unmatched == []


def test_scan_raw_directory_matches_lerobot_folder_alias(tmp_path):
    data_root = tmp_path / "data"
    raw = data_root / "public_datasets_raw" / "lerobot"
    raw.mkdir(parents=True)
    (raw / "a.bin").write_bytes(b"x" * 10)

    found, unmatched = scan_raw_directory(
        data_root, [("lerobot_full_folding", "lerobot/full_folding")]
    )
    assert found == [FoundDataset("lerobot_full_folding", "lerobot", 10)]
    assert unmatched == []


def test_scan_raw_directory_reports_unmatched_nonempty_folder(tmp_path):
    data_root = tmp_path / "data"
    raw = data_root / "public_datasets_raw" / "mystery_dataset"
    raw.mkdir(parents=True)
    (raw / "a.bin").write_bytes(b"x" * 10)

    found, unmatched = scan_raw_directory(data_root, [("droid", "DROID")])
    assert found == []
    assert unmatched == ["mystery_dataset"]


def test_scan_raw_directory_skips_missing_root(tmp_path):
    data_root = tmp_path / "data"
    found, unmatched = scan_raw_directory(data_root, [("droid", "DROID")])
    assert found == []
    assert unmatched == []


def test_scan_raw_directory_skips_empty_folder(tmp_path):
    data_root = tmp_path / "data"
    raw = data_root / "public_datasets_raw" / "droid"
    raw.mkdir(parents=True)
    found, unmatched = scan_raw_directory(data_root, [("droid", "DROID")])
    assert found == []
    assert unmatched == []


def test_scan_raw_directory_sums_nested_files(tmp_path):
    data_root = tmp_path / "data"
    raw = data_root / "public_datasets_raw" / "droid"
    nested = raw / "sub"
    nested.mkdir(parents=True)
    (raw / "a.bin").write_bytes(b"x" * 50)
    (nested / "b.bin").write_bytes(b"x" * 30)

    found, unmatched = scan_raw_directory(data_root, [("droid", "DROID")])
    assert found == [FoundDataset("droid", "droid", 80)]
    assert unmatched == []
