from __future__ import annotations

import os
import time
import uuid
import shutil
import asyncio
import json
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
app = FastAPI(title="VLA Data Governance Platform", version="0.1.0")


class ScanRequest(BaseModel):
    root: str | None = None


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


@app.post("/api/catalog/scan")
async def scan(_: ScanRequest | None = None) -> dict[str, Any]:
    rows = catalog.scan()
    return {"datasets": rows, "count": len(rows), "scanned_at": time.time()}


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


@app.get("/api/datasets/{uid}/episodes")
async def episodes(uid: str) -> list[dict[str, Any]]:
    if not catalog.get_dataset(uid): raise HTTPException(404, "dataset not indexed")
    return catalog.list_episodes(uid)


@app.get("/api/datasets/{uid}/episodes/{episode_index}/series")
async def series(uid: str, episode_index: int, fields: str = "state,action", limit: int = Query(2000, ge=1, le=10000)) -> dict[str, Any]:
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
