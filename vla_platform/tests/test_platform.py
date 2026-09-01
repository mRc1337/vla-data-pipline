import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from vla_platform.api import app
from vla_platform.catalog import Catalog
from vla_platform.pipeline import PipelineRunner


def make_dataset(root: Path) -> None:
    info = {"codebase_version": "v3.0", "total_episodes": 1, "total_frames": 3,
            "fps": 10, "features": {"action": {"dtype": "float32"},
            "observation.images.front": {"dtype": "video"}}}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "videos").mkdir()
    (root / "videos" / "front.mp4").write_bytes(b"0123456789")


def test_catalog_scan_and_safe_range(tmp_path, monkeypatch):
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    rows = catalog.scan()
    assert rows[0]["codebase_version"] == "v3.0"
    assert catalog.resolve_path("demo", "videos/front.mp4").read_bytes() == b"0123456789"


def test_pipeline_writes_manifest(tmp_path):
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    catalog.scan()
    runner = PipelineRunner(catalog, tmp_path / "curation")
    task = asyncio.run(runner.submit("demo", 1))
    for _ in range(20):
        if task.status in {"succeeded", "failed"}: break
        asyncio.run(asyncio.sleep(0.01))
    # The task coroutine is scheduled on the submit loop; exercise the worker
    # directly as a deterministic unit test.
    asyncio.run(runner._run(task, {}))
    assert task.status == "succeeded"
    assert (tmp_path / "curation" / "stage1" / "demo" / task.task_id / "manifest.json").exists()


def test_health_endpoint():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_video_range_endpoint(monkeypatch, tmp_path):
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    test_catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    monkeypatch.setattr("vla_platform.api.catalog", test_catalog)
    test_catalog.scan()
    response = TestClient(app).get("/api/videos/demo/videos/front.mp4", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"


def test_annotation_review_version(monkeypatch, tmp_path):
    dataset = tmp_path / "demo"; make_dataset(dataset)
    test_catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path); test_catalog.scan()
    monkeypatch.setattr("vla_platform.api.catalog", test_catalog)
    client = TestClient(app)
    created = client.post("/api/annotations", json={"dataset_uid":"demo", "episode_index":0, "label_type":"quality"}).json()
    reviewed = client.patch(f"/api/annotations/{created['annotation_id']}/review", json={"reviewer":"qa"})
    assert reviewed.status_code == 200 and reviewed.json()["review_status"] == "reviewed"


def test_quick_scan_avoids_deep_size_walk_and_reuses_fingerprint(tmp_path):
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    first = catalog.scan(mode="quick")
    assert first[0]["bytes"] == 0
    second = catalog.scan(mode="quick")
    assert second[0]["uid"] == "demo"
    assert catalog.list_datasets()[0]["bytes"] == 0


def test_async_scan_api_reports_terminal_status(monkeypatch, tmp_path):
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    test_catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    monkeypatch.setattr("vla_platform.api.catalog", test_catalog)
    monkeypatch.setattr("vla_platform.api.DATA_ROOT", tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/catalog/scan", json={"mode": "quick"})
        assert response.status_code == 202
        scan_id = response.json()["scan_id"]
        deadline = time.time() + 3
        while time.time() < deadline:
            status = client.get(f"/api/catalog/scans/{scan_id}").json()
            if status["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert status["status"] == "succeeded"
        assert status["datasets"] == 1
