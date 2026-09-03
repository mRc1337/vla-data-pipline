import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vla_platform.api import app
from vla_platform.catalog import Catalog
from vla_platform.thumbnail import ThumbnailManager


def make_search_dataset(root: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.wrist" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "total_episodes": 3, "total_frames": 60,
        "fps": 20, "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.wrist": {"dtype": "video"},
            "observation.images.front": {"dtype": "video"},
        },
    }))
    pq.write_table(pa.table({
        "task_index": [0, 1], "task": ["put the white mug down", "fold the towel"],
    }), root / "meta" / "tasks.parquet")
    pq.write_table(pa.table({
        "episode_index": [0, 1, 2], "length": [20, 30, 10],
        "tasks": [["put the white mug down"], ["fold the towel"], ["fold the towel"]],
        "videos/observation.images.front/chunk_index": [0, 0, 0],
        "videos/observation.images.front/file_index": [0, 0, 0],
        "videos/observation.images.front/from_timestamp": [5.0, 6.0, 7.5],
        "videos/observation.images.front/to_timestamp": [6.0, 7.5, 8.0],
        "videos/observation.images.wrist/chunk_index": [0, 0, 0],
        "videos/observation.images.wrist/file_index": [0, 0, 0],
        "videos/observation.images.wrist/from_timestamp": [5.0, 6.0, 7.5],
        "videos/observation.images.wrist/to_timestamp": [6.0, 7.5, 8.0],
    }), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for camera in ("observation.images.front", "observation.images.wrist"):
        (root / "videos" / camera / "chunk-000" / "file-000.mp4").write_bytes(b"video")


def make_stage_artifacts(data_root: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    stage1 = data_root / "data_curation" / "stage1" / "demo"
    (stage1 / "labels").mkdir(parents=True)
    (stage1 / "manifest.json").write_text(json.dumps({"stage_id": 1, "format": "vla_curation_filter"}))
    pq.write_table(pa.table({
        "episode_index": [0, 1, 2], "flagged_frames": [2, 0, 0],
        "flag_fraction": [0.1, 0.0, 0.0], "reject_episode": [False, False, True],
    }), stage1 / "labels" / "episode_summary.parquet")
    pq.write_table(pa.table({
        "episode_index": [0, 0], "frame_index": [3, 4], "flagged_frame": [True, True],
    }), stage1 / "labels" / "frame_flags.parquet")
    pq.write_table(pa.table({
        "episode_index": [0, 1, 2], "accepted": [True, True, False],
    }), stage1 / "labels" / "episode_filter.parquet")

    stage2 = data_root / "data_curation" / "stage2" / "demo"
    (stage2 / "labels").mkdir(parents=True)
    (stage2 / "manifest.json").write_text(json.dumps({
        "stage_id": 2, "format": "vla_curation_filter",
        "parent_manifest": str(stage1 / "manifest.json"),
    }))
    pq.write_table(pa.table({
        "episode_index": [0, 1], "scored_dimensions": [3, 3],
        "failed_dimensions": [[], [1]], "minimum_da": [0.95, 0.2],
        "reject_episode": [False, True],
    }), stage2 / "labels" / "episode_flags.parquet")
    pq.write_table(pa.table({
        "episode_index": [0, 1], "accepted": [True, False],
    }), stage2 / "labels" / "episode_filter.parquet")

    stage6 = data_root / "data_curation" / "stage6" / "demo"
    stage6.mkdir(parents=True)
    (stage6 / "manifest.json").write_text(json.dumps({"stage_id": 6, "run_complete": False}))
    (stage6 / "episode_000000.json").write_text(json.dumps({
        "episode_index": 0, "status": "complete", "quality": {"uncertainties": []},
    }))


def indexed_catalog(tmp_path: Path) -> Catalog:
    make_search_dataset(tmp_path / "demo")
    make_stage_artifacts(tmp_path)
    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    catalog.scan(mode="standard")
    result = catalog.sync_search_index()
    assert result["episodes"] == 3
    return catalog


def test_episode_search_text_stage_filters_pagination_and_missing_states(tmp_path):
    catalog = indexed_catalog(tmp_path)
    text_result = catalog.search_episodes(query="white mug")
    assert text_result["total"] == 1
    assert text_result["items"][0]["title"] == "【demo】put the white mug down"
    assert text_result["items"][0]["primary_camera"] == "observation.images.front"
    assert text_result["items"][0]["video_from_timestamp"] == 5.0

    combined = catalog.search_episodes(stage_filters=[
        {"stage_id": 1, "verdicts": ["pass", "anomaly"]},
        {"stage_id": 2, "verdicts": ["pass"]},
    ])
    assert [item["episode_index"] for item in combined["items"]] == [0]
    stage1 = next(item for item in combined["items"][0]["stage_badges"] if item["stage_id"] == 1)
    assert stage1["anomaly_count"] == 2 and stage1["range_count"] == 1

    within_stage_or = catalog.search_episodes(stage_filters=[{
        "stage_id": 6, "verdicts": ["complete"],
        "artifact_statuses": ["episode_pending"],
    }])
    assert within_stage_or["total"] == 3

    missing = catalog.search_episodes(stage_filters=[
        {"stage_id": 8, "artifact_statuses": ["not_generated"]},
    ], page=1, page_size=2)
    assert missing["total"] == 3 and len(missing["items"]) == 2
    assert missing["dataset_count"] == 1 and missing["task_count"] == 2
    assert all(
        next(b for b in item["stage_badges"] if b["stage_id"] == 8)["verdict"] == "not_generated"
        for item in missing["items"]
    )


def test_json_stage_index_updates_incrementally(tmp_path):
    catalog = indexed_catalog(tmp_path)
    source = tmp_path / "data_curation" / "stage6" / "demo" / "episode_000000.json"
    first = catalog.search_episodes(stage_filters=[{"stage_id": 6, "verdicts": ["complete"]}])
    assert first["total"] == 1
    source.write_text(json.dumps({
        "episode_index": 0, "status": "needs_review",
        "quality": {"uncertainties": ["ambiguous instruction"]},
    }))
    os.utime(source, None)
    catalog.sync_search_index(["demo"])
    updated = catalog.search_episodes(stage_filters=[{"stage_id": 6, "verdicts": ["needs_review"]}])
    assert updated["total"] == 1
    badge = next(value for value in updated["items"][0]["stage_badges"] if value["stage_id"] == 6)
    assert badge["severity"] == "warning"


def test_thumbnail_uses_primary_camera_episode_start_and_invalidates_cache(tmp_path, monkeypatch):
    catalog = indexed_catalog(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("vla_platform.thumbnail.shutil.which", lambda value: f"/usr/bin/{value}")
    monkeypatch.setattr("vla_platform.thumbnail.subprocess.run", fake_run)
    manager = ThumbnailManager(tmp_path / "thumbnails", workers=1)
    try:
        queued = manager.request(catalog, "demo", 0)
        deadline = time.time() + 2
        status = queued
        while status["status"] != "ready" and time.time() < deadline:
            time.sleep(0.01)
            status = manager.inspect(catalog, "demo", 0)
        assert status["status"] == "ready"
        assert status["camera"] == "observation.images.front"
        assert "5.000000000" in commands[0]
        assert "observation.images.front" in " ".join(commands[0])
        first_url = status["url"]

        source = tmp_path / "demo" / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
        source.write_bytes(b"changed-video")
        invalidated = manager.inspect(catalog, "demo", 0)
        assert invalidated["status"] == "not_generated"
        assert invalidated["url"] != first_url
    finally:
        manager._executor.shutdown(wait=True)


def test_search_and_thumbnail_api(monkeypatch, tmp_path):
    catalog = indexed_catalog(tmp_path)
    manager = ThumbnailManager(tmp_path / "thumbnails", workers=1)
    monkeypatch.setattr("vla_platform.api.catalog", catalog)
    monkeypatch.setattr("vla_platform.api.thumbnail_manager", manager)
    try:
        with TestClient(app) as client:
            response = client.post("/api/search/episodes", json={"query": "mug", "page_size": 24})
            assert response.status_code == 200
            item = response.json()["items"][0]
            assert item["thumbnail_status"] == "not_generated"
            assert item["thumbnail_url"].startswith("/api/thumbnails/demo/0?v=")
            prewarm = client.post("/api/thumbnails/prewarm", json={
                "episodes": [{"dataset_uid": "demo", "episode_index": 0}],
            })
            assert prewarm.status_code == 202
            facets = client.get("/api/search/facets")
            assert facets.status_code == 200
            assert {item["value"] for item in facets.json()["stages"]["1"]["verdicts"]} == {"anomaly", "filtered", "pass"}

            index_task = client.post("/api/search/index", json={"datasets": ["demo"]})
            assert index_task.status_code == 202
            task_id = index_task.json()["task_id"]
            deadline = time.time() + 3
            while time.time() < deadline:
                task = client.get(f"/api/search/index/{task_id}").json()
                if task["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.02)
            assert task["status"] == "succeeded"
    finally:
        manager._executor.shutdown(wait=True)
