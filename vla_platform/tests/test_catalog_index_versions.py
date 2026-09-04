import json
import sqlite3
from pathlib import Path

from vla_platform.catalog import (
    CATALOG_INDEX_VERSION,
    EPISODE_SEARCH_INDEX_VERSION,
    Catalog,
)
from vla_platform.video_proxy import VideoProxyManager


def _make_video_dataset(root: Path) -> list[str]:
    cameras = [
        *(f"observation.rgb.camera_{index}" for index in range(4)),
        *(f"observation.depth_linear.camera_{index}" for index in range(2)),
    ]
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "total_episodes": 1,
        "total_frames": 10,
        "fps": 10,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {camera: {"dtype": "video"} for camera in cameras},
    }))
    for camera in cameras:
        video = root / "videos" / camera / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
    return sorted(cameras)


def _patch_episode_metadata(catalog: Catalog) -> None:
    catalog._iter_episode_metadata = lambda _root: iter([{
        "episode_index": 0,
        "frames": 10,
        "duration": 1.0,
        "instruction": "test episode",
        "task_index": 0,
        # A populated metadata payload is part of the search-index cache
        # invariant; empty metadata represents an older incomplete row.
        "metadata_json": '{"dataset_from_index": 0}',
    }])


def test_quick_scan_migrates_stale_catalog_without_touching_video_files(tmp_path):
    cameras = _make_video_dataset(tmp_path / "behavior_1k")
    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    catalog.scan(mode="quick")

    with catalog._connect() as db:
        db.execute("UPDATE datasets SET cameras='[]'")
        db.execute("UPDATE scan_fingerprints SET catalog_index_version=0")

    def fail_if_video_scan(*_args, **_kwargs):
        raise AssertionError("quick scan must not inspect video files")

    catalog._scan_video_files = fail_if_video_scan
    catalog._scan_parquet_files = fail_if_video_scan
    rows = catalog.scan(mode="quick")

    assert rows[0]["cameras"] == cameras
    assert catalog.get_dataset("behavior_1k")["cameras"] == cameras
    with catalog._connect() as db:
        version = db.execute(
            "SELECT catalog_index_version FROM scan_fingerprints WHERE dataset_uid=?",
            ("behavior_1k",),
        ).fetchone()[0]
    assert version == CATALOG_INDEX_VERSION


def test_legacy_scan_fingerprint_schema_is_migrated(tmp_path):
    dataset = tmp_path / "demo"
    cameras = _make_video_dataset(dataset)[:1]
    info = json.loads((dataset / "meta" / "info.json").read_text())
    info["features"] = {cameras[0]: {"dtype": "video"}}
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    info_stat = (dataset / "meta" / "info.json").stat()
    db_path = tmp_path / "legacy.sqlite3"

    db = sqlite3.connect(db_path)
    db.execute("""CREATE TABLE datasets (
        uid TEXT PRIMARY KEY, root TEXT NOT NULL, codebase_version TEXT,
        episodes INTEGER NOT NULL DEFAULT 0, frames INTEGER NOT NULL DEFAULT 0,
        duration REAL NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,
        cameras TEXT NOT NULL DEFAULT '[]', schema_json TEXT NOT NULL DEFAULT '{}',
        scanned_at REAL NOT NULL
    )""")
    db.execute("""CREATE TABLE scan_fingerprints (
        root TEXT PRIMARY KEY, dataset_uid TEXT NOT NULL,
        info_mtime_ns INTEGER NOT NULL, info_size INTEGER NOT NULL,
        episodes_mtime_ns INTEGER NOT NULL DEFAULT 0, scanned_at REAL NOT NULL
    )""")
    db.execute(
        "INSERT INTO datasets(uid,root,cameras,schema_json,scanned_at) VALUES(?,?,?,?,0)",
        ("demo", str(dataset), "[]", json.dumps(info["features"])),
    )
    db.execute(
        "INSERT INTO scan_fingerprints VALUES(?,?,?,?,?,0)",
        (str(dataset), "demo", info_stat.st_mtime_ns, info_stat.st_size, 0),
    )
    db.commit()
    db.close()

    catalog = Catalog(db_path, tmp_path)
    with catalog._connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(scan_fingerprints)")}
    assert {"episodes_size", "catalog_index_version"} <= columns
    assert catalog.scan(mode="quick")[0]["cameras"] == cameras


def test_search_index_version_rebuilds_stale_camera_projection_and_then_hits_cache(tmp_path):
    cameras = _make_video_dataset(tmp_path / "behavior_1k")
    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    catalog.scan(mode="quick")
    _patch_episode_metadata(catalog)
    catalog.sync_search_index(["behavior_1k"])

    with catalog._connect() as db:
        db.execute(
            "UPDATE datasets SET cameras='[]' WHERE uid='behavior_1k'"
        )
        db.execute(
            "UPDATE episode_search SET camera_count=0,primary_camera=NULL,video_relative_path=NULL "
            "WHERE dataset_uid='behavior_1k'"
        )
        db.execute(
            "UPDATE stage_index_sources SET fingerprint='legacy-source-only-fingerprint' "
            "WHERE dataset_uid='behavior_1k' AND stage_id=0"
        )
        db.execute(
            "UPDATE scan_fingerprints SET catalog_index_version=0 "
            "WHERE dataset_uid='behavior_1k'"
        )

    catalog.scan(mode="quick")
    rebuilt = catalog.sync_search_index(["behavior_1k"])
    entry = catalog.get_episode_search_entry("behavior_1k", 0)
    assert rebuilt["episodes"] == 1
    assert entry["camera_count"] == len(cameras)
    assert entry["primary_camera"] == "observation.rgb.camera_0"
    assert entry["video_relative_path"].endswith(
        "observation.rgb.camera_0/chunk-000/file-000.mp4"
    )
    assert f'"version":{EPISODE_SEARCH_INDEX_VERSION}' in catalog._episode_search_fingerprint(
        tmp_path / "behavior_1k", cameras, entry["primary_camera"]
    )

    catalog._iter_episode_metadata = lambda _root: (_ for _ in ()).throw(
        AssertionError("second search sync must hit the cache")
    )
    catalog.sync_search_index(["behavior_1k"])

    preview = catalog.episode_preview_summary("behavior_1k", 0)
    assert len(preview["videos"]) == 6
    _job, specs, warnings = VideoProxyManager(tmp_path / "proxy")._specs(
        catalog, "behavior_1k", 0
    )
    assert len(specs) == 6
    assert all(spec.source.is_file() for spec in specs)
    assert warnings == []
