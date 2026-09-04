from __future__ import annotations

import os
import time
import uuid
import shutil
import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .catalog import Catalog
from .pipeline import PipelineRunner
from .thumbnail import ThumbnailManager
from .video_proxy import VideoProxyManager


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
VIDEO_PROXY_ROOT = Path(os.environ.get("VLA_VIDEO_PROXY_ROOT", str(DB_PATH.parent / "video_proxy")))
THUMBNAIL_ROOT = Path(os.environ.get("VLA_THUMBNAIL_ROOT", str(CURATION_ROOT / "_catalog" / "thumbnails")))
try:
    THUMBNAIL_ROOT.mkdir(parents=True, exist_ok=True)
except OSError:
    THUMBNAIL_ROOT = DB_PATH.parent / "thumbnails"
    THUMBNAIL_ROOT.mkdir(parents=True, exist_ok=True)
runner = PipelineRunner(catalog, CURATION_ROOT)
proxy_manager = VideoProxyManager(VIDEO_PROXY_ROOT)
thumbnail_manager = ThumbnailManager(
    THUMBNAIL_ROOT, workers=int(os.environ.get("VLA_THUMBNAIL_WORKERS", "2"))
)
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
search_index_tasks: dict[str, dict[str, Any]] = {}
search_index_lock = threading.Lock()
search_index_execution_lock = threading.Lock()


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


class SearchStageFilter(BaseModel):
    stage_id: int = Field(ge=1, le=8)
    verdicts: list[str] = Field(default_factory=list)
    artifact_statuses: list[str] = Field(default_factory=list)
    exclude_verdicts: list[str] = Field(default_factory=list)
    min_score: float | None = None
    max_score: float | None = None


class EpisodeSearchRequest(BaseModel):
    query: str = Field(default="", max_length=500)
    datasets: list[str] = Field(default_factory=list)
    stage_filters: list[SearchStageFilter] = Field(default_factory=list)
    sort: Literal["relevance", "episode", "duration_asc", "duration_desc"] = "relevance"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=24, ge=1, le=30)


class SearchIndexRequest(BaseModel):
    datasets: list[str] = Field(default_factory=list)


class ThumbnailItem(BaseModel):
    dataset_uid: str
    episode_index: int = Field(ge=0)


class ThumbnailPrewarmRequest(BaseModel):
    episodes: list[ThumbnailItem] = Field(default_factory=list, max_length=30)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True, "data_root": str(DATA_ROOT), "catalog": str(DB_PATH),
        "video_proxy_root": str(VIDEO_PROXY_ROOT), "thumbnail_root": str(THUMBNAIL_ROOT),
    }


async def _run_search_index(task_id: str, dataset_uids: list[str]) -> None:
    task = search_index_tasks[task_id]
    try:
        # Do not occupy a worker thread while another index build owns the
        # single-writer lock. A few queued jobs could otherwise exhaust the
        # shared executor and prevent the active job from making progress.
        while not search_index_execution_lock.acquire(blocking=False):
            await asyncio.sleep(0.1)
        try:
            with search_index_lock:
                task.update(status="running", started_at=time.time())

            def progress(update: dict[str, Any]) -> None:
                with search_index_lock:
                    task.update(update)

            result = await asyncio.to_thread(
                catalog.sync_search_index, dataset_uids or None, progress
            )
            with search_index_lock:
                task.update(status="succeeded", result=result, finished_at=time.time())
        finally:
            search_index_execution_lock.release()
    except Exception as exc:
        with search_index_lock:
            task.update(status="failed", error=str(exc), finished_at=time.time())


@app.post("/api/search/index", status_code=202)
async def build_search_index(body: SearchIndexRequest | None = None) -> dict[str, Any]:
    body = body or SearchIndexRequest()
    dataset_uids, missing = await asyncio.to_thread(
        catalog.resolve_search_index_datasets, body.datasets
    )
    if missing:
        raise HTTPException(404, f"datasets not indexed: {', '.join(missing)}")
    with search_index_lock:
        for existing in search_index_tasks.values():
            if (existing["status"] in {"queued", "running"}
                    and existing["datasets"] == dataset_uids):
                return dict(existing)
        task_id = f"search-index-{uuid.uuid4().hex[:12]}"
        task = {
            "task_id": task_id, "status": "queued", "datasets": dataset_uids,
            "current": 0, "total": 0, "created_at": time.time(), "error": None,
        }
        search_index_tasks[task_id] = task
    asyncio.create_task(_run_search_index(task_id, dataset_uids))
    return task


@app.get("/api/search/index")
async def search_index_summary() -> dict[str, Any]:
    return catalog.search_index_stats()


@app.get("/api/search/facets")
async def search_facets() -> dict[str, Any]:
    return catalog.search_stage_facets()


@app.get("/api/search/index/{task_id}")
async def search_index_status(task_id: str) -> dict[str, Any]:
    task = search_index_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "search index task not found")
    with search_index_lock:
        return dict(task)


@app.post("/api/search/episodes")
async def search_episodes(body: EpisodeSearchRequest) -> dict[str, Any]:
    result = await asyncio.to_thread(
        catalog.search_episodes,
        body.query,
        body.datasets or None,
        [value.model_dump() for value in body.stage_filters],
        body.sort,
        body.page,
        body.page_size,
    )
    def attach_thumbnails() -> None:
        for item in result["items"]:
            try:
                thumbnail = thumbnail_manager.inspect(
                    catalog, item["dataset_uid"], int(item["episode_index"])
                )
                item["thumbnail_status"] = thumbnail["status"]
                item["thumbnail_url"] = thumbnail["url"]
                item["primary_camera"] = thumbnail["camera"]
            except (FileNotFoundError, IndexError, OSError, ValueError) as exc:
                item["thumbnail_status"] = "unavailable"
                item["thumbnail_error"] = str(exc)

    await asyncio.to_thread(attach_thumbnails)
    return result


@app.post("/api/thumbnails/prewarm", status_code=202)
async def prewarm_thumbnails(body: ThumbnailPrewarmRequest) -> dict[str, Any]:
    def request_all() -> list[dict[str, Any]]:
        results = []
        for item in body.episodes:
            try:
                results.append(thumbnail_manager.request(
                    catalog, item.dataset_uid, item.episode_index
                ))
            except (FileNotFoundError, IndexError, OSError, ValueError) as exc:
                results.append({
                    "dataset_uid": item.dataset_uid,
                    "episode_index": item.episode_index,
                    "status": "unavailable",
                    "error": str(exc),
                })
        return results

    results = await asyncio.to_thread(request_all)
    return {"items": results}


@app.get("/api/thumbnails/{dataset_uid}/{episode_index}")
async def thumbnail(dataset_uid: str, episode_index: int):
    try:
        status, path = await asyncio.to_thread(
            lambda: (
                thumbnail_manager.request(catalog, dataset_uid, episode_index),
                thumbnail_manager.resolve(catalog, dataset_uid, episode_index),
            )
        )
    except (FileNotFoundError, IndexError, OSError, ValueError):
        raise HTTPException(404, "thumbnail source not found") from None
    if path is None:
        return JSONResponse(status, status_code=202, headers={"Cache-Control": "no-store"})
    return FileResponse(
        path, media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


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


@app.get("/api/datasets/{uid}/tasks")
async def tasks(uid: str) -> list[dict[str, Any]]:
    if not catalog.get_dataset(uid): raise HTTPException(404, "dataset not indexed")
    return await asyncio.to_thread(catalog.list_tasks, uid)


@app.get("/api/datasets/{uid}/episodes")
async def episodes(uid: str, task_index: int | None = Query(None, ge=0)) -> list[dict[str, Any]]:
    if not catalog.get_dataset(uid): raise HTTPException(404, "dataset not indexed")
    return catalog.list_episodes(uid, task_index)


@app.get("/api/datasets/{uid}/episodes/{episode_index}/preview")
async def episode_preview(uid: str, episode_index: int) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(catalog.episode_preview_summary, uid, episode_index)
    except (FileNotFoundError, IndexError):
        raise HTTPException(404, "episode not found") from None


@app.get("/api/datasets/{uid}/episodes/{episode_index}/stages/{stage_id}")
async def episode_stage_detail(uid: str, episode_index: int, stage_id: int) -> dict[str, Any]:
    if not 1 <= stage_id <= 8:
        raise HTTPException(404, "stage not found")
    try:
        return await asyncio.to_thread(
            catalog.episode_stage_detail, uid, episode_index, stage_id
        )
    except (FileNotFoundError, IndexError):
        raise HTTPException(404, "stage artifact not found") from None


@app.get("/api/datasets/{uid}/episodes/{episode_index}/series")
async def series(
    uid: str,
    episode_index: int,
    fields: str = "observation.state,action",
    limit: int = Query(2000, ge=1, le=10000),
    view: Literal["raw", "valid", "repaired", "diff"] = "raw",
) -> dict[str, Any]:
    try:
        values = await asyncio.to_thread(
            catalog.episode_series,
            uid, episode_index, [f.strip() for f in fields.split(",")], limit, view,
        )
    except FileNotFoundError: raise HTTPException(404, "dataset not indexed")
    return {
        "dataset_uid": uid, "episode_index": episode_index,
        "fields": fields.split(","), "view": view, "rows": values,
    }


def _prewarm_video_proxies(uid: str, episode_index: int, distance_limit: int) -> None:
    for distance in range(1, distance_limit + 1):
        for nearby in (episode_index + distance, episode_index - distance):
            if nearby < 0:
                continue
            try:
                proxy_manager.request(catalog, uid, nearby)
            except (FileNotFoundError, IndexError, OSError, RuntimeError, ValueError):
                continue


@app.post("/api/datasets/{uid}/episodes/{episode_index}/video-proxies", status_code=202)
async def create_video_proxies(
    uid: str, episode_index: int, prewarm: int = Query(2, ge=0, le=2)
) -> dict[str, Any]:
    try:
        task = await asyncio.to_thread(proxy_manager.request, catalog, uid, episode_index)
    except (FileNotFoundError, IndexError):
        raise HTTPException(404, "episode not found") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(500, str(exc)) from exc
    # The dedicated executor is intentionally single-worker by default. Queue
    # nearby episodes after responding so metadata reads and future encodes do
    # not delay the selected episode's UI request.
    if prewarm:
        asyncio.create_task(asyncio.to_thread(
            _prewarm_video_proxies, uid, episode_index, prewarm,
        ))
    return task


@app.get("/api/video-proxy-jobs/{job_id}")
async def video_proxy_status(job_id: str) -> dict[str, Any]:
    task = proxy_manager.status(job_id)
    if not task:
        raise HTTPException(404, "video proxy job not found")
    return task


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


def _range_file_response(path: Path, request: Request, *, immutable: bool = False):
    size = path.stat().st_size
    range_header = request.headers.get("range")
    headers = {"Accept-Ranges": "bytes"}
    if immutable:
        headers["Cache-Control"] = "public, max-age=31536000, immutable"
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers=headers)
    try:
        unit, value = range_header.split("=", 1)
        if unit.strip().lower() != "bytes" or "," in value:
            raise ValueError
        start_text, end_text = value.split("-", 1)
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start, end = max(0, size - suffix), size - 1
        else:
            start, end = int(start_text), int(end_text or size - 1)
        if start < 0 or end >= size or start > end: raise ValueError
    except (ValueError, TypeError):
        return JSONResponse(
            {"detail": "invalid range"}, status_code=416,
            headers={**headers, "Content-Range": f"bytes */{size}"},
        )
    length = end - start + 1

    def iterator():
        with path.open("rb") as fh:
            fh.seek(start); remaining = length
            while remaining:
                chunk = fh.read(min(1024 * 1024, remaining))
                if not chunk: break
                remaining -= len(chunk); yield chunk
    headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length)})
    return StreamingResponse(iterator(), status_code=206, media_type="video/mp4",
                             headers=headers)


@app.get("/api/video-proxies/files/{relative:path}")
async def video_proxy_file(relative: str, request: Request):
    try:
        path = proxy_manager.resolve_file(relative)
    except (FileNotFoundError, PermissionError):
        raise HTTPException(404, "video proxy not found") from None
    # mtime is the LRU signal. Proxy filenames contain the immutable source
    # signature, so touching a cache file cannot invalidate its identity.
    try:
        path.touch()
    except OSError:
        pass
    return _range_file_response(path, request, immutable=True)


@app.get("/api/videos/{uid}/{relative:path}")
async def video(uid: str, relative: str, request: Request):
    try:
        path = catalog.resolve_path(uid, relative)
    except (FileNotFoundError, PermissionError):
        raise HTTPException(404, "video not found") from None
    return _range_file_response(path, request)
