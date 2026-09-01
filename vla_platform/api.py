from __future__ import annotations

import os
import time
import uuid
import shutil
import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .catalog import Catalog
from .pipeline import PipelineRunner


DATA_ROOT = Path(os.environ.get("VLA_DATA_ROOT", "/mnt/data/embodied_datasets/public_datasets_staging"))
CURATION_ROOT = Path(os.environ.get("VLA_CURATION_ROOT", str(DATA_ROOT / "data_curation")))
DB_PATH = Path(os.environ.get("VLA_CATALOG_DB", str(CURATION_ROOT / "_catalog" / "catalog.sqlite3")))
try:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    probe = DB_PATH.parent / ".write-probe"
    probe.touch(); probe.unlink()
except OSError:
    # Read-only OSS/FUSE staging mounts must never prevent the API from
    # starting; deployment can point VLA_CATALOG_DB at PostgreSQL/SQLite disk.
    DB_PATH = Path("/tmp/vla-data-pipeline/catalog.sqlite3")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
try:
    catalog = Catalog(DB_PATH, DATA_ROOT)
except (OSError, Exception) as exc:
    # Some FUSE mounts allow touching files but reject SQLite journal/DDL
    # operations. Keep the data root read-only and place the index on local disk.
    if DB_PATH != Path("/tmp/vla-data-pipeline/catalog.sqlite3"):
        DB_PATH = Path("/tmp/vla-data-pipeline/catalog.sqlite3")
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        catalog = Catalog(DB_PATH, DATA_ROOT)
    else:
        raise exc
runner = PipelineRunner(catalog, CURATION_ROOT)
catalog.recover_interrupted_scans()
app = FastAPI(title="VLA Data Governance Platform", version="0.1.0")
scan_tasks: dict[str, dict[str, Any]] = {}
scan_cancellations: dict[str, threading.Event] = {}
scan_lock = threading.Lock()
# SQLite is configured for a single writer.  Keep scan execution serialized in
# the single-process deployment while allowing queued jobs to remain visible
# and cancellable through the API.  A threading lock is used instead of an
# asyncio primitive because TestClient and production workers can use
# different event loops over the lifetime of this module.
scan_execution_lock = threading.Lock()


def _persist_scan(scan_id: str, task: dict[str, Any]) -> None:
    try:
        catalog.update_scan_job(scan_id, task)
    except (AttributeError, OSError):
        # Keep the in-memory fallback usable for lightweight test doubles.
        pass


class ScanRequest(BaseModel):
    root: str | None = None
    mode: str = "quick"


class PipelineRequest(BaseModel):
    dataset_uid: str
    stage_id: int = Field(ge=1, le=8)
    config: dict[str, Any] = Field(default_factory=dict)


class AnnotationRequest(BaseModel):
    dataset_uid: str
    episode_index: int
    stage_id: int | None = None
    label_type: str
    status: str = "warning"
    severity: str | None = None
    score: float | None = None
    threshold: float | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    entity_type: str = "episode"
    entity_name: str | None = None
    reason_code: str | None = None
    source: str = "human"
    review_status: str = "pending"
    reviewer: str | None = None
    comment: str | None = None


class ExportRequest(BaseModel):
    dataset_uid: str
    episode_indices: list[int] = Field(default_factory=list)
    expression: str | None = None
    materialize: bool = False


class ReviewRequest(BaseModel):
    review_status: str = "reviewed"
    reviewer: str | None = None
    comment: str | None = None


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "data_root": str(DATA_ROOT), "catalog": str(DB_PATH)}


def _safe_scan_root(root: str | None) -> str | None:
    if not root:
        return None
    candidate = Path(root)
    if not candidate.is_absolute():
        candidate = DATA_ROOT / candidate
    candidate = candidate.resolve()
    base = DATA_ROOT.resolve()
    if candidate != base and base not in candidate.parents:
        raise HTTPException(400, "scan root must be inside VLA_DATA_ROOT")
    if not candidate.is_dir():
        raise HTTPException(404, "scan root does not exist")
    return str(candidate)


async def _run_catalog_scan(scan_id: str, mode: str, scan_root: str | None) -> None:
    task = scan_tasks[scan_id]
    try:
        await asyncio.to_thread(scan_execution_lock.acquire)
        try:
            cancellation = scan_cancellations[scan_id]
            if cancellation.is_set():
                task.update(status="cancelled", finished_at=time.time())
                _persist_scan(scan_id, task)
                return
            task["status"] = "running"
            task["started_at"] = time.time()
            _persist_scan(scan_id, task)

            def progress(update: dict[str, Any]) -> None:
                with scan_lock:
                    task.update(update)
                    task["completed"] = int(update.get("current", 0))
                    task["skipped"] = int(task.get("skipped", 0)) + int(update.get("skipped", False))
                    elapsed = max(time.time() - task["started_at"], 0.001)
                    task["rate"] = task["completed"] / elapsed
                    total = int(update.get("total", 0))
                    task["eta_seconds"] = max((total - task["completed"]) / task["rate"], 0) if task["rate"] else None
                    _persist_scan(scan_id, task)

            rows = await asyncio.to_thread(catalog.scan, mode, scan_root, progress, cancellation)
            with scan_lock:
                task.update(status="cancelled" if cancellation.is_set() else "succeeded",
                            datasets=len(rows), finished_at=time.time())
                _persist_scan(scan_id, task)
        finally:
            scan_execution_lock.release()
    except Exception as exc:
        with scan_lock:
            task.update(status="failed", error=str(exc), finished_at=time.time())
            _persist_scan(scan_id, task)
    finally:
        scan_cancellations.pop(scan_id, None)


@app.post("/api/catalog/scan", status_code=202)
async def scan(body: ScanRequest | None = None) -> dict[str, Any]:
    body = body or ScanRequest()
    if body.mode not in {"quick", "standard", "deep"}:
        raise HTTPException(400, "scan mode must be quick, standard, or deep")
    scan_id = f"scan-{uuid.uuid4().hex[:12]}"
    task = {"scan_id": scan_id, "status": "queued", "mode": body.mode,
            "root": _safe_scan_root(body.root), "current": 0, "completed": 0,
            "skipped": 0, "datasets": 0, "eta_seconds": None, "created_at": time.time()}
    scan_tasks[scan_id] = task
    scan_cancellations[scan_id] = threading.Event()
    catalog.create_scan_job(task)
    asyncio.create_task(_run_catalog_scan(scan_id, body.mode, task["root"]))
    return task


@app.get("/api/catalog/scans/{scan_id}")
async def scan_status(scan_id: str) -> dict[str, Any]:
    task = scan_tasks.get(scan_id)
    if not task:
        task = catalog.get_scan_job(scan_id)
        if not task:
            raise HTTPException(404, "scan not found")
    with scan_lock:
        return dict(task)


@app.post("/api/catalog/scans/{scan_id}/cancel")
async def cancel_scan(scan_id: str) -> dict[str, Any]:
    task = scan_tasks.get(scan_id)
    if not task:
        raise HTTPException(404, "scan not found")
    event = scan_cancellations.get(scan_id)
    if event:
        event.set()
    task["status"] = "cancelling"
    _persist_scan(scan_id, task)
    return task


@app.get("/api/catalog/scans/{scan_id}/events")
async def scan_events(scan_id: str):
    if scan_id not in scan_tasks:
        raise HTTPException(404, "scan not found")

    async def stream():
        last = None
        while True:
            task = scan_tasks.get(scan_id)
            if task is None:
                break
            payload = json.dumps(task, default=str)
            if payload != last:
                last = payload
                yield f"data: {payload}\n\n"
            if task["status"] in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/datasets")
async def datasets(uid: str | None = None, codebase_version: str | None = None) -> list[dict[str, Any]]:
    rows = catalog.list_datasets()
    if uid: rows = [row for row in rows if uid.lower() in row["uid"].lower()]
    if codebase_version: rows = [row for row in rows if row["codebase_version"] == codebase_version]
    return rows


@app.get("/api/datasets/{uid}")
async def dataset(uid: str) -> dict[str, Any]:
    value = catalog.get_dataset(uid)
    if not value: raise HTTPException(404, "dataset not indexed")
    return value


@app.get("/api/datasets/{uid}/videos")
async def videos(uid: str, integrity_status: str | None = Query(None)) -> list[dict[str, Any]]:
    if not catalog.get_dataset(uid):
        raise HTTPException(404, "dataset not indexed")
    return catalog.list_videos(uid, integrity_status)


@app.get("/api/datasets/{uid}/episodes")
async def episodes(uid: str) -> list[dict[str, Any]]:
    if not catalog.get_dataset(uid): raise HTTPException(404, "dataset not indexed")
    return catalog.list_episodes(uid)


@app.get("/api/datasets/{uid}/episodes/{episode_index}/preview")
async def episode_preview(uid: str, episode_index: int) -> dict[str, Any]:
    try:
        return catalog.episode_preview(uid, episode_index)
    except (FileNotFoundError, IndexError):
        raise HTTPException(404, "episode not found") from None


@app.get("/api/datasets/{uid}/episodes/{episode_index}/series")
async def series(uid: str, episode_index: int, fields: str = "observation.state,action", limit: int = Query(2000, ge=1, le=10000)) -> dict[str, Any]:
    try:
        values = catalog.episode_series(uid, episode_index, [f.strip() for f in fields.split(",")], limit)
    except FileNotFoundError: raise HTTPException(404, "dataset not indexed")
    return {"dataset_uid": uid, "episode_index": episode_index, "fields": fields.split(","), "rows": values}


@app.get("/api/datasets/{uid}/annotations")
async def annotations(uid: str, episode_index: int | None = Query(None)) -> list[dict[str, Any]]:
    return catalog.list_annotations(uid, episode_index)


@app.post("/api/annotations")
async def add_annotation(body: AnnotationRequest) -> dict[str, Any]:
    if not catalog.get_dataset(body.dataset_uid): raise HTTPException(404, "dataset not indexed")
    value = body.model_dump(); value.update(annotation_id=str(uuid.uuid4()), created_at=time.time())
    return catalog.add_annotation(value)


@app.patch("/api/annotations/{annotation_id}/review")
async def review_annotation(annotation_id: str, body: ReviewRequest) -> dict[str, Any]:
    value = catalog.update_annotation_review(annotation_id, body.review_status, body.reviewer, body.comment)
    if not value: raise HTTPException(404, "annotation not found")
    return value


@app.post("/api/exports", status_code=202)
async def export_subset(body: ExportRequest) -> dict[str, Any]:
    dataset = catalog.get_dataset(body.dataset_uid)
    if not dataset: raise HTTPException(404, "dataset not indexed")
    export_id = str(uuid.uuid4())
    target = CURATION_ROOT / "_catalog" / "exports" / export_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"export_id": export_id, "dataset_uid": body.dataset_uid, "source": dataset["root"],
                "episode_indices": body.episode_indices, "filter_expression": body.expression,
                "materialized": body.materialize, "created_at": time.time()}
    if body.materialize:
        # Materialization is explicit because copying videos is expensive. The
        # source remains immutable and the manifest remains the audit anchor.
        shutil.copytree(dataset["root"], target / "dataset", dirs_exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {**manifest, "path": str(target)}


@app.post("/api/pipelines/run", status_code=202)
async def run_pipeline(body: PipelineRequest) -> dict[str, Any]:
    try: task = await runner.submit(body.dataset_uid, body.stage_id, body.config)
    except (ValueError, FileNotFoundError) as exc: raise HTTPException(400, str(exc)) from exc
    return task.__dict__


@app.get("/api/tasks/{task_id}")
async def task(task_id: str) -> dict[str, Any]:
    value = runner.get(task_id)
    if not value: raise HTTPException(404, "task not found")
    return value.__dict__


@app.post("/api/tasks/{task_id}/retry", status_code=202)
async def retry_task(task_id: str) -> dict[str, Any]:
    old = runner.get(task_id)
    if not old: raise HTTPException(404, "task not found")
    if old.status not in {"failed", "cancelled"}: raise HTTPException(409, "task is not retryable")
    return (await runner.submit(old.dataset_uid, old.stage_id, {})).__dict__


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, Any]:
    value = runner.cancel(task_id)
    if not value: raise HTTPException(404, "task not found")
    return value.__dict__


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str):
    if not runner.get(task_id): raise HTTPException(404, "task not found")
    async def stream():
        last = None
        for _ in range(600):
            value = runner.get(task_id)
            payload = json.dumps(value.__dict__, default=str)
            if payload != last:
                last = payload
                yield f"data: {payload}\n\n"
            if value.status in {"succeeded", "failed", "cancelled"}: break
            await asyncio.sleep(0.5)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/videos/{uid}/{relative:path}")
async def video(uid: str, relative: str, request: Request):
    try: path = catalog.resolve_path(uid, relative)
    except (FileNotFoundError, PermissionError): raise HTTPException(404, "video not found")
    size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header: return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    try:
        start, end = range_header.replace("bytes=", "").split("-")
        start, end = int(start), int(end or size - 1)
        if start < 0 or end >= size or start > end: raise ValueError
    except ValueError: return JSONResponse({"detail": "invalid range"}, status_code=416)
    length = end - start + 1
    def iterator():
        with path.open("rb") as fh:
            fh.seek(start); remaining = length
            while remaining:
                chunk = fh.read(min(1024 * 1024, remaining))
                if not chunk: break
                remaining -= len(chunk); yield chunk
    return StreamingResponse(iterator(), status_code=206, media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length)})
