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


def test_nested_lerobot_wrappers_get_distinct_stable_semantic_uids(tmp_path):
    data_root = tmp_path / "staging"
    composite = data_root / "lerobot_v3_0" / "robocasa" / "target" / "composite"
    first_root = composite / "WeighIngredients" / "20250812" / "lerobot"
    second_root = composite / "PnPCounterToCab" / "20250813" / "lerobot"
    _make_video_dataset(first_root)
    _make_video_dataset(second_root)
    catalog = Catalog(tmp_path / "catalog.sqlite3", data_root)

    first_scan = catalog.scan(mode="quick", scan_root=composite)
    first_uids = {row["root"]: row["uid"] for row in first_scan}
    second_scan = catalog.scan(mode="quick", scan_root=composite)
    second_uids = {row["root"]: row["uid"] for row in second_scan}

    assert len(first_uids) == 2
    assert len(set(first_uids.values())) == 2
    assert first_uids == second_uids
    assert first_uids[str(first_root)].startswith(
        "robocasa--target--composite--WeighIngredients--20250812--"
    )
    assert first_uids[str(second_root)].startswith(
        "robocasa--target--composite--PnPCounterToCab--20250813--"
    )
    datasets = {row["uid"]: row for row in catalog.list_datasets()}
    assert datasets[first_uids[str(first_root)]]["display_name"] == (
        "WeighIngredients / 20250812"
    )


def test_duplicate_plain_basenames_do_not_overwrite_each_other(tmp_path):
    data_root = tmp_path / "staging"
    first_root = data_root / "collection-a" / "demo"
    second_root = data_root / "collection-b" / "demo"
    _make_video_dataset(first_root)
    _make_video_dataset(second_root)
    catalog = Catalog(tmp_path / "catalog.sqlite3", data_root)

    first_uid = catalog.scan(scan_root=first_root)[0]["uid"]
    second_uid = catalog.scan(scan_root=second_root)[0]["uid"]

    assert first_uid == "demo"
    assert second_uid.startswith("demo--")
    assert second_uid != first_uid
    assert catalog.scan(scan_root=second_root)[0]["uid"] == second_uid
    assert {row["root"] for row in catalog.list_datasets()} == {
        str(first_root), str(second_root),
    }

    # Removing the conflicting row must not silently change an already-issued
    # UID on a later non-cached scan.
    with catalog._connect() as db:
        db.execute("DELETE FROM datasets WHERE uid=?", (first_uid,))
        db.execute(
            "UPDATE scan_fingerprints SET catalog_index_version=0 WHERE root=?",
            (str(second_root),),
        )
    assert catalog.scan(scan_root=second_root)[0]["uid"] == second_uid


def test_scan_rekeys_legacy_wrapper_uid_and_preserves_derived_rows(tmp_path):
    data_root = tmp_path / "staging"
    root = (
        data_root / "lerobot_v3_0" / "robocasa" / "target" / "composite"
        / "WeighIngredients" / "20250812" / "lerobot"
    )
    _make_video_dataset(root)
    catalog = Catalog(tmp_path / "catalog.sqlite3", data_root)
    with catalog._connect() as db:
        db.execute(
            """INSERT INTO datasets(
                uid,root,episodes,frames,cameras,schema_json,scanned_at
            ) VALUES(?,?,?,?,?,?,?)""",
            ("lerobot", str(root), 1, 10, "[]", "{}", 1.0),
        )
        db.execute(
            """INSERT INTO episodes(
                dataset_uid,episode_index,frames,duration,instruction,task_index
            ) VALUES(?,?,?,?,?,?)""",
            ("lerobot", 0, 10, 1.0, "weigh ingredients", 0),
        )
        db.execute(
            """INSERT INTO episode_search(
                dataset_uid,episode_index,collection_name,frame_count,duration,
                camera_count,normalized_text,indexed_at
            ) VALUES(?,?,?,?,?,?,?,?)""",
            ("lerobot", 0, "robocasa", 10, 1.0, 1, "weigh ingredients", 1.0),
        )
        db.execute(
            """INSERT INTO annotations(
                annotation_id,dataset_uid,episode_index,label_type,status,source,created_at
            ) VALUES(?,?,?,?,?,?,?)""",
            ("annotation-1", "lerobot", 0, "quality", "open", "manual", 1.0),
        )
        db.execute(
            "INSERT INTO video_files(dataset_uid,relative_path) VALUES(?,?)",
            ("lerobot", "videos/front.mp4"),
        )
        catalog._ensure_episode_fts(db, "lerobot", replace=True)

    row = catalog.scan(mode="quick", scan_root=root)[0]
    new_uid = row["uid"]

    assert new_uid != "lerobot"
    assert new_uid.startswith(
        "robocasa--target--composite--WeighIngredients--20250812--"
    )
    with catalog._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM datasets WHERE uid='lerobot'").fetchone()[0] == 0
        for table in ("episodes", "episode_search", "annotations", "video_files"):
            assert db.execute(
                f"SELECT DISTINCT dataset_uid FROM {table}"
            ).fetchall()[0][0] == new_uid
        assert db.execute(
            "SELECT COUNT(*) FROM episode_search_fts WHERE dataset_uid=?", (new_uid,)
        ).fetchone()[0] == 1


def test_legacy_duplicate_root_aliases_are_merged_before_unique_index(tmp_path):
    db_path = tmp_path / "legacy-aliases.sqlite3"
    root = tmp_path / "dataset"
    db = sqlite3.connect(db_path)
    db.execute("""CREATE TABLE datasets (
        uid TEXT PRIMARY KEY, root TEXT NOT NULL, codebase_version TEXT,
        episodes INTEGER NOT NULL DEFAULT 0, frames INTEGER NOT NULL DEFAULT 0,
        duration REAL NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,
        cameras TEXT NOT NULL DEFAULT '[]', schema_json TEXT NOT NULL DEFAULT '{}',
        scanned_at REAL NOT NULL
    )""")
    db.executemany(
        "INSERT INTO datasets(uid,root,scanned_at) VALUES(?,?,?)",
        [("older-alias", str(root), 1.0), ("newer-alias", str(root), 2.0)],
    )
    db.commit()
    db.close()

    catalog = Catalog(db_path, tmp_path)

    assert [row["uid"] for row in catalog.list_datasets()] == ["newer-alias"]
    with catalog._connect() as connection:
        indexes = {
            row[1]: bool(row[2])
            for row in connection.execute("PRAGMA index_list(datasets)").fetchall()
        }
    assert indexes["idx_datasets_root"] is True


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
