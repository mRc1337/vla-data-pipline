from common.scan_downloaded_datasets import scan_raw_directory


def test_scan_raw_directory_reads_public_datasets_raw_subdir(tmp_path):
    """Regression test: scan_raw_directory must look under
    public_datasets_raw/ (common/paths.py's raw_dir() convention), not the
    pre-rename raw/ -- a prior version hardcoded raw/ and silently found
    nothing against a real data_root.
    """
    raw_root = tmp_path / "public_datasets_raw" / "some_dataset"
    raw_root.mkdir(parents=True)
    (raw_root / "data.bin").write_bytes(b"x" * 100)

    found, unmatched = scan_raw_directory(tmp_path, [("some_dataset", "Some Dataset")])

    assert not unmatched
    assert len(found) == 1
    assert found[0].dataset_id == "some_dataset"
    assert found[0].folder_name == "some_dataset"
    assert found[0].size_bytes == 100


def test_scan_raw_directory_ignores_legacy_raw_subdir(tmp_path):
    legacy_raw_root = tmp_path / "raw" / "some_dataset"
    legacy_raw_root.mkdir(parents=True)
    (legacy_raw_root / "data.bin").write_bytes(b"x" * 100)

    found, unmatched = scan_raw_directory(tmp_path, [("some_dataset", "Some Dataset")])

    assert found == []
    assert unmatched == []
