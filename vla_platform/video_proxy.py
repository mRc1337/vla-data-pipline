from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


PROXY_VERSION = "h264-v1"


def _safe_name(value: str) -> str:
    readable = "".join(char if char.isalnum() or char in ".-_" else "_" for char in value).strip("._")
    return (readable or "item")[:96]


@dataclass(frozen=True)
class ProxySpec:
    camera: str
    source: Path
    target: Path
    relative_target: str
    source_start: float
    duration: float
    frame_count: int
    fps: float
    version: str


class VideoProxyManager:
    """Generate short, browser-friendly episode clips without changing source data."""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)
        self.workers = max(1, int(os.environ.get("VLA_PROXY_WORKERS", "1")))
        self.encoder_threads = max(1, int(os.environ.get("VLA_PROXY_ENCODER_THREADS", "2")))
        self.cache_limit = max(0, int(os.environ.get("VLA_PROXY_CACHE_BYTES", str(20 * 1024**3))))
        self.timeout = max(30, int(os.environ.get("VLA_PROXY_TIMEOUT_SECONDS", "600")))
        self.work_root = Path(os.environ.get("VLA_PROXY_WORK_ROOT", "/tmp/vla-video-proxy"))
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="video-proxy")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _ready(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    def _specs(
        self, catalog: Any, uid: str, episode_index: int
    ) -> tuple[str, list[ProxySpec], list[dict[str, str]]]:
        preview_loader = getattr(catalog, "episode_preview_summary", catalog.episode_preview)
        preview = preview_loader(uid, episode_index)
        timeline = preview["timeline"]
        fps = float(timeline["fps"])
        frame_count = int(timeline["frame_count"])
        duration = float(timeline["duration"])
        if frame_count <= 0 or duration <= 0:
            raise ValueError("episode contains no video frames")
        source_records: list[tuple[dict[str, Any], Path, os.stat_result]] = []
        warnings: list[dict[str, str]] = []
        digest = hashlib.sha256()
        digest.update(f"{PROXY_VERSION}|{uid}|{episode_index}|{fps}|{frame_count}".encode())
        for video in preview["videos"]:
            try:
                source = catalog.resolve_path(uid, video["relative_path"])
                stat = source.stat()
            except FileNotFoundError:
                warnings.append({
                    "camera": str(video["camera"]),
                    "relative_path": str(video["relative_path"]),
                    "reason": "source video is missing",
                })
                digest.update(
                    f"|missing|{video['camera']}|{video['relative_path']}".encode()
                )
                continue
            source_records.append((video, source, stat))
            digest.update(
                f"|{video['camera']}|{video['relative_path']}|{video['source_start']}|"
                f"{video['source_end']}|{stat.st_size}|{stat.st_mtime_ns}".encode()
            )
        if not source_records:
            missing = ", ".join(item["camera"] for item in warnings)
            detail = f"; missing cameras: {missing}" if missing else ""
            raise ValueError(f"episode contains no available video streams{detail}")
        version = digest.hexdigest()[:16]
        uid_dir = f"{_safe_name(uid)}-{hashlib.sha256(uid.encode()).hexdigest()[:8]}"
        episode_dir = self.cache_root / uid_dir / f"episode-{episode_index:06d}"
        specs: list[ProxySpec] = []
        for video, source, _ in source_records:
            camera_name = f"{_safe_name(video['camera'])}-{hashlib.sha256(video['camera'].encode()).hexdigest()[:8]}"
            target = episode_dir / f"{camera_name}-{version}.mp4"
            specs.append(ProxySpec(
                camera=video["camera"], source=source, target=target,
                relative_target=target.relative_to(self.cache_root).as_posix(),
                source_start=float(video["source_start"]), duration=duration,
                frame_count=frame_count, fps=fps, version=version,
            ))
        return f"proxy-{version}", specs, warnings

    def _video_payload(self, spec: ProxySpec) -> dict[str, Any]:
        return {
            "camera": spec.camera,
            "status": "ready" if self._ready(spec.target) else "pending",
            "url": f"/api/video-proxies/files/{quote(spec.relative_target, safe='/')}?v={spec.version}",
            "duration": spec.duration,
            "frame_count": spec.frame_count,
            "fps": spec.fps,
        }

    def _snapshot(self, job: dict[str, Any]) -> dict[str, Any]:
        specs: list[ProxySpec] = job["specs"]
        future: Future[None] | None = job.get("future")
        if all(self._ready(spec.target) for spec in specs):
            status, error = "ready", None
        elif future is None:
            status, error = "failed", job.get("error", "proxy task was not started")
        elif not future.done():
            status, error = ("generating" if future.running() else "queued"), None
        else:
            exception = future.exception()
            status, error = (("failed", str(exception)) if exception else
                             ("failed", "proxy cache entry is incomplete or was evicted"))
        return {
            "job_id": job["job_id"], "dataset_uid": job["dataset_uid"],
            "episode_index": job["episode_index"], "status": status,
            "error": error, "created_at": job["created_at"],
            "warnings": job.get("warnings", []),
            "videos": [self._video_payload(spec) for spec in specs],
        }

    def request(self, catalog: Any, uid: str, episode_index: int) -> dict[str, Any]:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to generate episode video proxies")
        job_id, specs, warnings = self._specs(catalog, uid, episode_index)
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing:
                snapshot = self._snapshot(existing)
                if snapshot["status"] != "failed":
                    return snapshot
            job: dict[str, Any] = {
                "job_id": job_id, "dataset_uid": uid, "episode_index": episode_index,
                "specs": specs, "warnings": warnings, "created_at": time.time(),
            }
            self._jobs[job_id] = job
            if not all(self._ready(spec.target) for spec in specs):
                job["future"] = self._executor.submit(self._generate, specs)
            return self._snapshot(job)

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._snapshot(job) if job else None

    def resolve_file(self, relative: str) -> Path:
        root = self.cache_root.resolve()
        path = (root / relative).resolve()
        if path == root or root not in path.parents or not self._ready(path):
            raise FileNotFoundError(relative)
        return path

    def _generate(self, specs: list[ProxySpec]) -> None:
        for spec in specs:
            if self._ready(spec.target):
                continue
            spec.target.parent.mkdir(parents=True, exist_ok=True)
            self.work_root.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f"{_safe_name(spec.camera)}-", suffix=".mp4", dir=self.work_root,
            )
            os.close(file_descriptor)
            temporary = Path(temporary_name)
            upload = spec.target.with_name(f".{spec.target.stem}.{uuid.uuid4().hex}.upload.mp4")
            gop = max(1, round(spec.fps))
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-ss", f"{spec.source_start:.9f}", "-i", str(spec.source),
                "-map", "0:v:0", "-an", "-frames:v", str(spec.frame_count),
                "-vf", "setpts=PTS-STARTPTS,scale=min(960\\,iw):-2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop),
                "-sc_threshold", "0", "-movflags", "+faststart",
                "-threads", str(self.encoder_threads), str(temporary),
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout, check=False)
                if result.returncode != 0 or not self._ready(temporary):
                    detail = (result.stderr or result.stdout or "ffmpeg produced no output").strip()
                    raise RuntimeError(f"{spec.camera}: {detail[-2000:]}")
                shutil.copyfile(temporary, upload)
                os.replace(upload, spec.target)
            finally:
                temporary.unlink(missing_ok=True)
                upload.unlink(missing_ok=True)
        self._write_manifest(specs)
        self._prune_cache()

    def _write_manifest(self, specs: list[ProxySpec]) -> None:
        if not specs:
            return
        path = specs[0].target.parent / "manifest.json"
        payload = {
            "proxy_version": PROXY_VERSION, "created_at": time.time(),
            "videos": [{
                "camera": spec.camera, "source": str(spec.source),
                "source_start": spec.source_start, "duration": spec.duration,
                "frame_count": spec.frame_count, "fps": spec.fps,
                "proxy": spec.target.name,
            } for spec in specs],
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        os.replace(temporary, path)

    def _prune_cache(self) -> None:
        if not self.cache_limit or not self.cache_root.is_dir():
            return
        files = [path for path in self.cache_root.rglob("*.mp4") if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        if total <= self.cache_limit:
            return
        for path in sorted(files, key=lambda item: item.stat().st_mtime_ns):
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
            if total <= self.cache_limit:
                break
