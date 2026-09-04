import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from vla_platform.api import app
from vla_platform.catalog import Catalog
from vla_platform.pipeline import PipelineRunner
from vla_platform.video_proxy import VideoProxyManager


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
    search_manifest = tmp_path / "curation" / "stage1" / "demo" / "search_index_manifest.json"
    assert json.loads(search_manifest.read_text())["file_count"] == 1


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


def test_standard_scan_indexes_video_metadata_without_decoding_payload(tmp_path):
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)

    catalog.scan(mode="standard")
    videos = catalog.list_videos("demo")
    assert len(videos) == 1
    assert videos[0]["relative_path"] == "videos/front.mp4"
    assert videos[0]["size"] == 10
    assert videos[0]["integrity_status"] in {
        "header_ok", "fail", "metadata_unavailable"
    }

    # A second standard scan reuses the size/mtime cache and keeps the row.
    catalog.scan(mode="standard")
    assert len(catalog.list_videos("demo")) == 1


def test_deep_scan_labels_bad_video_instead_of_failing_scan(tmp_path):
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)

    rows = catalog.scan(mode="deep")
    videos = catalog.list_videos("demo")
    assert len(rows) == 1
    assert videos[0]["integrity_status"] in {
        "pass", "fail", "metadata_unavailable"
    }


def test_video_metadata_endpoint(monkeypatch, tmp_path):
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    test_catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    test_catalog.scan(mode="standard")
    monkeypatch.setattr("vla_platform.api.catalog", test_catalog)
    response = TestClient(app).get("/api/datasets/demo/videos")
    assert response.status_code == 200
    assert response.json()[0]["relative_path"] == "videos/front.mp4"


def test_task_map_accepts_explicit_task_index_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    dataset = tmp_path / "task-index"
    (dataset / "meta").mkdir(parents=True)
    pq.write_table(pa.table({
        "task_index": pa.array([0]),
        "task": pa.array(["open the bottle"]),
    }), dataset / "meta" / "tasks.parquet")
    assert Catalog._load_task_map(dataset) == {0: "open the bottle"}


def test_episode_preview_reads_metadata_series_and_video_refs(monkeypatch, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    dataset = tmp_path / "demo"
    (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    (dataset / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True)
    (dataset / "videos" / "observation.images.wrist" / "chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3.0", "total_episodes": 1, "total_frames": 2,
            "fps": 10, "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {"observation.images.front": {
                             "dtype": "video", "shape": [240, 320, 3],
                             "info": {"video.height": 240, "video.width": 320,
                                      "video.codec": "av1", "video.pix_fmt": "yuv420p",
                                      "video.fps": 10, "video.channels": 3, "has_audio": False}},
                         "observation.images.wrist": {"dtype": "video"},
                         "observation.state": {"dtype": "float32", "shape": [2]},
                         "action": {"dtype": "float32", "shape": [2]}}}
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({"task_index": pa.array([0]), "__index_level_0__": pa.array(["move the block"])}),
                   dataset / "meta" / "tasks.parquet")
    pq.write_table(pa.table({
        "episode_index": pa.array([0]), "length": pa.array([2]),
        "tasks": pa.array([["move the block"]]), "data/chunk_index": pa.array([0]),
        "data/file_index": pa.array([0]),
        "videos/observation.images.front/chunk_index": pa.array([0]),
        "videos/observation.images.front/file_index": pa.array([1]),
        "videos/observation.images.front/from_timestamp": pa.array([0.0]),
        "videos/observation.images.front/to_timestamp": pa.array([0.2]),
        "videos/observation.images.wrist/chunk_index": pa.array([0]),
        "videos/observation.images.wrist/file_index": pa.array([0]),
        "videos/observation.images.wrist/from_timestamp": pa.array([10.0]),
        "videos/observation.images.wrist/to_timestamp": pa.array([10.2]),
    }), dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({
        "observation.state": pa.array([[1.0, 2.0], [2.0, 3.0]]),
        "action": pa.array([[0.1, 0.2], [0.2, 0.3]]),
        "timestamp": pa.array([0.0, 0.1]), "frame_index": pa.array([0, 1]),
        "episode_index": pa.array([0, 0]),
    }), dataset / "data" / "chunk-000" / "file-000.parquet")
    front_video = dataset / "videos" / "observation.images.front" / "chunk-000" / "file-001.mp4"
    wrist_video = dataset / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4"
    front_video.write_bytes(b"not-a-real-video")
    wrist_video.write_bytes(b"not-a-real-video")
    stage_audit = tmp_path / "data_curation" / "stage1" / "demo_smoke" / "audit"
    stage_audit.mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": pa.array([0]), "frame_index": pa.array([1]),
        "flagged_frame": pa.array([True]),
    }), stage_audit / "frame_flags.parquet")
    stage_overlay = tmp_path / "data_curation" / "stage2" / "demo"
    (stage_overlay / "labels").mkdir(parents=True)
    (stage_overlay / "manifest.json").write_text(json.dumps({
        "format": "vla_curation_overlay", "stage_id": 2, "stage": "Stage 2",
        "dataset_id": "demo", "detector_version": "test", "result": {"output_episodes": 1},
        "repair_files": ["repairs/state_action.parquet"],
    }))
    pq.write_table(pa.table({
        "episode_index": pa.array([0]), "accepted": pa.array([True]),
    }), stage_overlay / "labels" / "episode_filter.parquet")
    (stage_overlay / "repairs").mkdir()
    pq.write_table(pa.table({
        "episode_index": pa.array([0]), "frame_index": pa.array([1]),
        "observation.state": pa.array([[20.0, 30.0]]),
        "action": pa.array([[2.0, 3.0]]),
    }), stage_overlay / "repairs" / "state_action.parquet")

    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    catalog.scan(mode="standard")
    assert catalog.list_tasks("demo") == [{"task_index": 0, "name": "move the block", "episodes": 1}]
    assert catalog.list_episodes("demo", 0)[0]["task_index"] == 0
    preview = catalog.episode_preview("demo", 0)
    assert preview["episode"]["instruction"] == "move the block"
    assert preview["timeline"] == {
        "coordinate_system": "episode_relative", "frame_count": 2, "fps": 10.0,
        "duration": 0.2, "dataset_from_index": 0,
    }
    videos = {item["camera"]: item for item in preview["videos"]}
    assert videos["observation.images.front"]["file_index"] == 1
    assert videos["observation.images.front"]["source_start"] == 0.0
    assert videos["observation.images.front"]["width"] == 320
    assert videos["observation.images.front"]["height"] == 240
    assert videos["observation.images.front"]["fps"] == 10
    assert videos["observation.images.front"]["codec"] == "av1"
    assert videos["observation.images.front"]["source_bytes"] == len(b"not-a-real-video")
    assert videos["observation.images.wrist"]["file_index"] == 0
    assert videos["observation.images.wrist"]["source_start"] == 10.0
    assert videos["observation.images.wrist"]["source_end"] == 10.2
    assert all(item["integrity_status"] == "pass" for item in videos.values())
    assert preview["stage_results"][0]["run_id"] == "demo_smoke"
    assert preview["stage_results"][0]["coordinate_system"] == "episode_frame"
    assert preview["stage_results"][0]["records"][0]["frame_index"] == 1
    overlay_result = next(item for item in preview["stage_results"] if item["stage_id"] == 2)
    assert overlay_result["summary"] == {"output_episodes": 1}
    assert overlay_result["records"][0]["accepted"] is True
    rows = catalog.episode_series("demo", 0, ["timestamp", "observation.state", "action"], limit=2)
    assert len(rows) == 2 and rows[0]["observation.state"] == [1.0, 2.0]
    assert [row["episode_time"] for row in rows] == [0.0, 0.1]
    repaired = catalog.episode_series(
        "demo", 0, ["timestamp", "frame_index", "observation.state", "action"], limit=2,
        view="repaired",
    )
    assert repaired[0]["observation.state"] == [1.0, 2.0]
    assert repaired[1]["observation.state"] == [20.0, 30.0]
    delta = catalog.episode_series(
        "demo", 0, ["timestamp", "frame_index", "observation.state", "action"], limit=2,
        view="diff",
    )
    assert delta[0]["observation.state"] == [0.0, 0.0]
    assert delta[1]["observation.state"] == [18.0, 27.0]
    monkeypatch.setattr("vla_platform.api.catalog", catalog)
    response = TestClient(app).get("/api/datasets/demo/episodes/0/preview")
    assert response.status_code == 200
    assert response.json()["episode"]["instruction"] == "move the block"
    assert response.json()["timeline"]["duration"] == 0.2
    assert TestClient(app).get("/api/datasets/demo/tasks").json()[0]["episodes"] == 1
    response = TestClient(app).get(
        "/api/datasets/demo/episodes/0/series",
        params={"fields": "frame_index,observation.state,action", "view": "repaired"},
    )
    assert response.status_code == 200
    assert response.json()["view"] == "repaired"
    assert response.json()["rows"][1]["observation.state"] == [20.0, 30.0]


def test_filter_manifest_exposes_validity_without_fake_repairs(monkeypatch, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    dataset = tmp_path / "demo"
    (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0", "total_episodes": 1, "total_frames": 3, "fps": 10,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({
        "episode_index": [0], "length": [3], "data/chunk_index": [0], "data/file_index": [0],
    }), dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({
        "episode_index": [0, 0, 0], "frame_index": [0, 1, 2],
        "observation.state": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        "action": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
    }), dataset / "data" / "chunk-000" / "file-000.parquet")
    stage = tmp_path / "data_curation" / "stage1" / "demo"
    (stage / "labels").mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0, 0, 0], "frame_index": [0, 1, 2], "index": [0, 1, 2],
        "stage1_valid": [True, False, True], "stage2_valid": [True, True, True],
        "stage3_valid": [True, True, True], "valid": [True, False, True],
        "reason_codes": [[], ["stage1_sudden_change"], []],
    }), stage / "labels" / "step_validity.parquet")
    pq.write_table(pa.table({
        "episode_index": [0], "num_frames": [3], "accepted": [True], "reason_code": [None],
    }), stage / "labels" / "episode_filter.parquet")
    (stage / "manifest.json").write_text(json.dumps({
        "format": "vla_curation_filter", "schema_version": 2, "stage_id": 1,
        "stage": "Stage 1", "dataset_id": "demo", "source_dataset": str(dataset),
        "parent_manifest": None, "episode_filter": "labels/episode_filter.parquet",
        "validity_files": ["labels/step_validity.parquet"], "repair_files": [],
    }))
    stage6 = tmp_path / "data_curation" / "stage6" / "demo"
    stage6.mkdir(parents=True)
    (stage6 / "manifest.json").write_text(json.dumps({
        "schema_version": "stage6_semantic_subtasks_manifest_v1", "dataset": "demo",
        "total_episodes": 1, "counts": {"complete": 1}, "run_complete": True,
    }))
    (stage6 / "episode_000000.json").write_text(json.dumps({
        "schema_version": "stage6_semantic_subtasks_v1", "dataset": "demo",
        "episode_index": 0, "status": "complete", "model": "test-model",
        "task": {"original_instructions": ["move object"], "expected_task_plan": ["grasp", "place"]},
        "scene": {"summary": "object moved", "objects": [{"name": "object", "confidence": "high"}]},
        "temporal_segmentation": {"final_boundaries": [0, 1, 3], "segments": [
            {"start_frame": 0, "end_frame_exclusive": 1, "subtask_label": "grasp", "confidence": "high"},
            {"start_frame": 1, "end_frame_exclusive": 3, "subtask_label": "place", "confidence": "high"},
        ]},
        "quality": {"uncertainties": []},
    }))

    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    catalog.scan(mode="standard")
    preview = catalog.episode_preview("demo", 0)
    assert preview["curation"] == {
        "format": "vla_curation_filter", "has_repairs": False, "has_validity": True,
    }
    stage_result = next(item for item in preview["stage_results"] if item["stage_id"] == 1)
    assert stage_result["artifact_status"] == "available"
    assert stage_result["visualization_spec"]["name"] == "Sudden Change Detection"
    invalid_records = [row for row in stage_result["records"] if row.get("valid") is False]
    assert len(invalid_records) == 1
    assert invalid_records[0]["frame_index"] == 1
    assert invalid_records[0]["reason_codes"] == ["stage1_sudden_change"]
    advanced = {item["stage_id"]: item for item in preview["stage_results"] if item["stage_id"] >= 4}
    assert set(advanced) == {4, 5, 6, 7, 8}
    assert advanced[4]["artifact_status"] == "not_generated"
    assert advanced[6]["artifact_status"] == "available"
    assert advanced[6]["detail"]["temporal_segmentation"]["segments"][1]["subtask_label"] == "place"
    assert advanced[8]["visualization_spec"]["expected_fields"][-1] == "dropped_frame_indices"
    missing = {item["stage_id"]: item for item in preview["stage_results"] if item["stage_id"] in {2, 3}}
    assert missing[2]["artifact_status"] == "not_generated"
    assert missing[3]["visualization_spec"]["name"] == "Extreme Value Detection"
    valid_rows = catalog.episode_series(
        "demo", 0, ["frame_index", "observation.state", "action"], limit=3, view="valid"
    )
    assert [row["valid"] for row in valid_rows] == [True, False, True]
    assert valid_rows[1]["observation.state"] == [None, None]
    monkeypatch.setattr("vla_platform.api.catalog", catalog)
    response = TestClient(app).get(
        "/api/datasets/demo/episodes/0/series", params={"view": "valid"}
    )
    assert response.status_code == 200 and response.json()["view"] == "valid"


def test_episode_filter_status_reads_actual_manifest_label(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    stage = tmp_path / "stage1" / "demo"
    (stage / "labels").mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [3, 4], "accepted": [False, True],
    }), stage / "labels" / "episode_filter.parquet")

    assert Catalog._episode_filter_status(stage, 3) is False
    assert Catalog._episode_filter_status(stage, 4) is True
    assert Catalog._episode_filter_status(stage, 5) is None


def test_preview_reports_episode_filtered_by_upstream_stage(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    dataset = tmp_path / "demo"
    make_dataset(dataset)
    stage1 = tmp_path / "data_curation" / "stage1" / "demo"
    (stage1 / "labels").mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0], "num_frames": [2], "accepted": [False],
        "reason_code": ["stage1_episode_rejected"],
    }), stage1 / "labels" / "episode_filter.parquet")
    stage1_manifest = stage1 / "manifest.json"
    stage1_manifest.write_text(json.dumps({
        "format": "vla_curation_filter", "stage_id": 1, "stage": "Stage 1",
        "source_dataset": str(dataset), "episode_filter": "labels/episode_filter.parquet",
    }))
    stage2 = tmp_path / "data_curation" / "stage2" / "demo"
    stage2.mkdir(parents=True)
    (stage2 / "manifest.json").write_text(json.dumps({
        "format": "vla_curation_filter", "stage_id": 2, "stage": "Stage 2",
        "source_dataset": str(dataset), "parent_manifest": str(stage1_manifest),
        "episode_filter": "labels/episode_filter.parquet",
    }))

    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    catalog.scan(mode="standard")
    preview = catalog.episode_preview("demo", 0)
    stage2_result = next(item for item in preview["stage_results"] if item["stage_id"] == 2)
    assert stage2_result["artifact_status"] == "upstream_filtered"
    assert "前序 Stage 被过滤" in stage2_result["placeholder"]


def test_preview_resolves_collection_qualified_stage_directories(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    uid = "part-000-static-velocity-no-effort-4cams-50fps"
    dataset = tmp_path / "lerobot_v3_0" / "mobile_aloha" / uid
    make_dataset(dataset)

    # Current Stage 1-3 writer escapes the collection separator as ``__``.
    flattened = tmp_path / "data_curation" / "stage1" / f"mobile_aloha__{uid}"
    (flattened / "labels").mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0], "num_frames": [3], "accepted": [True],
    }), flattened / "labels" / "episode_filter.parquet")
    (flattened / "manifest.json").write_text(json.dumps({
        "format": "vla_curation_filter", "stage_id": 1, "stage": "Stage 1",
        "dataset_id": f"mobile_aloha/{uid}", "source_dataset": str(dataset),
        "episode_filter": "labels/episode_filter.parquet",
    }))

    # Also support the nested layout used by other Stage producers.
    nested = tmp_path / "data_curation" / "stage4" / "mobile_aloha" / uid
    (nested / "labels").mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0], "accepted": [True],
    }), nested / "labels" / "episode_filter.parquet")
    (nested / "manifest.json").write_text(json.dumps({
        "format": "vla_curation_filter", "stage_id": 4, "stage": "Stage 4",
        "dataset_id": f"mobile_aloha/{uid}", "source_dataset": str(dataset),
        "episode_filter": "labels/episode_filter.parquet",
    }))

    catalog = Catalog(tmp_path / "catalog.sqlite3", tmp_path)
    catalog.scan(mode="standard", scan_root=dataset)
    preview = catalog.episode_preview(uid, 0)
    results = {item["stage_id"]: item for item in preview["stage_results"]}
    assert results[1]["artifact_status"] == "available"
    assert results[1]["records"][0]["file"] == "labels/episode_filter.parquet"
    assert results[4]["artifact_status"] == "available"


def test_stage_episode_json_exposes_stage7_and_stage8_evidence(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "episode_000007.json").write_text(json.dumps({
        "schema_version": "stage7_sam2_video_state_audit_v1",
        "episode_index": 7,
        "decision": "fail",
        "reason": "low overlap",
        "frames": [{"frame_index": 4, "decision": "fail", "iou": 0.1}],
        "summary": {"sampled_frames": 1, "fail_frames": 1},
        "source": {"private_provider_path": "/not/exposed"},
    }))
    stage7 = Catalog._stage_episode_json(stage, 7)
    assert stage7 and stage7["decision"] == "fail"
    assert stage7["frames"][0]["iou"] == 0.1
    assert "source" not in stage7

    (stage / "episode_000008.json").write_text(json.dumps({
        "schema_version": "stage8_video_quality_audit_v1",
        "episode_index": 8,
        "status": "available",
        "data_disposition": "exclude_affected_sample_windows",
        "invalid_ranges": [{"start_frame": 10, "end_frame": 20, "reasons": ["blurred"]}],
        "per_camera_results": [{"camera": "front", "blurred_frames": 10}],
    }))
    stage8 = Catalog._stage_episode_json(stage, 8)
    assert stage8 and stage8["data_disposition"] == "exclude_affected_sample_windows"
    assert stage8["invalid_ranges"][0]["start_frame"] == 10
    assert stage8["per_camera_results"][0]["camera"] == "front"


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
        persisted = test_catalog.get_scan_job(scan_id)
        assert persisted and persisted["status"] == "succeeded"


def test_video_proxy_manager_generates_versioned_cached_clips(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-video")

    class FakeCatalog:
        def episode_preview(self, uid, episode_index):
            assert (uid, episode_index) == ("demo", 7)
            return {
                "timeline": {"fps": 20, "frame_count": 165, "duration": 8.25},
                "videos": [{
                    "camera": "observation.images.front", "relative_path": "source.mp4",
                    "source_start": 6957.4, "source_end": 6965.65,
                }],
            }

        def resolve_path(self, uid, relative):
            assert (uid, relative) == ("demo", "source.mp4")
            return source

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"proxy-video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("vla_platform.video_proxy.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("vla_platform.video_proxy.subprocess.run", fake_run)
    manager = VideoProxyManager(tmp_path / "proxies")
    try:
        task = manager.request(FakeCatalog(), "demo", 7)
        deadline = time.time() + 2
        while task["status"] != "ready" and time.time() < deadline:
            time.sleep(0.01)
            task = manager.status(task["job_id"])
        assert task and task["status"] == "ready"
        assert task["videos"][0]["url"].startswith("/api/video-proxies/files/")
        assert "-ss" in commands[0] and "6957.400000000" in commands[0]
        assert "-frames:v" in commands[0] and "165" in commands[0]
        assert "libx264" in commands[0]
        assert len(commands) == 1
        assert manager.request(FakeCatalog(), "demo", 7)["status"] == "ready"
        assert len(commands) == 1
    finally:
        manager._executor.shutdown(wait=True)


def test_video_proxy_api_and_range(monkeypatch, tmp_path):
    proxy_file = tmp_path / "demo" / "episode-000000" / "front-version.mp4"
    proxy_file.parent.mkdir(parents=True)
    proxy_file.write_bytes(b"0123456789")

    class FakeProxyManager:
        def request(self, catalog, uid, episode_index):
            return {"job_id": "proxy-version", "dataset_uid": uid, "episode_index": episode_index,
                    "status": "ready", "error": None, "videos": []}

        def status(self, job_id):
            return {"job_id": job_id, "status": "ready", "videos": []}

        def resolve_file(self, relative):
            assert relative == "demo/episode-000000/front-version.mp4"
            return proxy_file

    monkeypatch.setattr("vla_platform.api.proxy_manager", FakeProxyManager())
    client = TestClient(app)
    task = client.post("/api/datasets/demo/episodes/0/video-proxies?prewarm=0")
    assert task.status_code == 202 and task.json()["status"] == "ready"
    status = client.get("/api/video-proxy-jobs/proxy-version")
    assert status.status_code == 200 and status.json()["status"] == "ready"
    response = client.get(
        "/api/video-proxies/files/demo/episode-000000/front-version.mp4",
        headers={"Range": "bytes=-4"},
    )
    assert response.status_code == 206 and response.content == b"6789"
    assert response.headers["content-range"] == "bytes 6-9/10"
    assert "immutable" in response.headers["cache-control"]
