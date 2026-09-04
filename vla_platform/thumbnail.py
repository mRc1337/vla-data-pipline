from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


class ThumbnailManager:
    """Generate bounded, versioned Episode cover images from local MP4 shards."""

    VERSION = "v1"

    def __init__(self, root: str | Path, workers: int = 2, width: int = 480):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.width = max(160, int(width))
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="thumbnail")
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _source(catalog: Any, uid: str, episode_index: int) -> dict[str, Any]:
        dataset = catalog.get_dataset(uid)
        if not dataset:
            raise FileNotFoundError(uid)
        root = Path(dataset["root"])
        indexed = catalog.get_episode_search_entry(uid, episode_index)
        if indexed and indexed.get("primary_camera") and indexed.get("video_relative_path"):
            camera = str(indexed["primary_camera"])
            relative = str(indexed["video_relative_path"])
            source_start = float(indexed.get("video_from_timestamp") or 0)
        else:
            metadata = catalog._load_episode_metadata(root, episode_index)
            if metadata is None:
                raise IndexError(episode_index)
            info = catalog._json(root / "meta" / "info.json")
            camera = catalog._primary_camera(list(dataset.get("cameras") or []))
            if not camera:
                raise FileNotFoundError(f"{uid} has no video camera")
            prefix = f"videos/{camera}"
            chunks_size = int(info.get("chunks_size", 1000) or 1000)
            chunk = int(metadata.get(f"{prefix}/chunk_index", episode_index // chunks_size))
            file_index = int(metadata.get(f"{prefix}/file_index", episode_index % chunks_size))
            source_start = float(metadata.get(f"{prefix}/from_timestamp", 0) or 0)
            template = info.get(
                "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            )
            try:
                relative = str(template.format(video_key=camera, chunk_index=chunk, file_index=file_index))
            except (KeyError, ValueError):
                relative = f"videos/{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        source = (root / relative).resolve()
        if not source.is_file() or root.resolve() not in source.parents:
            raise FileNotFoundError(source)
        stat = source.stat()
        fingerprint = hashlib.sha256(
            f"{source}:{stat.st_size}:{stat.st_mtime_ns}:{source_start:.9f}:{camera}:{ThumbnailManager.VERSION}".encode()
        ).hexdigest()[:16]
        return {
            "camera": camera, "source": source, "source_start": source_start,
            "fingerprint": fingerprint,
        }

    def _target(self, uid: str, episode_index: int, fingerprint: str) -> Path:
        dataset_hash = hashlib.sha256(uid.encode()).hexdigest()[:16]
        return self.root / dataset_hash / f"episode-{episode_index:06d}-{fingerprint}.jpg"

    def request(self, catalog: Any, uid: str, episode_index: int) -> dict[str, Any]:
        source = self._source(catalog, uid, episode_index)
        target = self._target(uid, episode_index, source["fingerprint"])
        key = f"{uid}:{episode_index}:{source['fingerprint']}"
        with self._lock:
            if target.is_file() and target.stat().st_size:
                self._tasks[key] = {
                    "status": "ready", "error": None, "target": str(target),
                    "camera": source["camera"],
                }
            else:
                existing = self._tasks.get(key)
                if existing is None or existing["status"] == "failed":
                    self._tasks[key] = {
                        "status": "queued", "error": None, "target": str(target),
                        "camera": source["camera"],
                    }
                    self._executor.submit(self._generate, key, source, target)
            task = dict(self._tasks[key])
        return {
            "dataset_uid": uid, "episode_index": episode_index,
            "status": task["status"], "error": task.get("error"),
            "camera": source["camera"],
            "url": f"/api/thumbnails/{uid}/{episode_index}?v={source['fingerprint']}",
        }

    def inspect(self, catalog: Any, uid: str, episode_index: int) -> dict[str, Any]:
        """Report cache state without scheduling FFmpeg work."""
        source = self._source(catalog, uid, episode_index)
        target = self._target(uid, episode_index, source["fingerprint"])
        key = f"{uid}:{episode_index}:{source['fingerprint']}"
        with self._lock:
            task = self._tasks.get(key)
            if target.is_file() and target.stat().st_size:
                status, error = "ready", None
            elif task:
                status, error = task["status"], task.get("error")
            else:
                status, error = "not_generated", None
        return {
            "dataset_uid": uid,
            "episode_index": episode_index,
            "status": status,
            "error": error,
            "camera": source["camera"],
            "url": f"/api/thumbnails/{uid}/{episode_index}?v={source['fingerprint']}",
        }

    def _generate(self, key: str, source: dict[str, Any], target: Path) -> None:
        temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.jpg")
        try:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("ffmpeg is not installed")
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{source['source_start']:.9f}", "-i", str(source["source"]),
                "-frames:v", "1", "-vf", f"scale='min({self.width},iw)':-2",
                "-q:v", "4", str(temporary),
            ]
            if shutil.which("nice"):
                command = ["nice", "-n", "10", *command]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            if result.returncode != 0 or not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError(result.stderr.strip() or "ffmpeg did not create a thumbnail")
            os.replace(temporary, target)
            with self._lock:
                self._tasks[key].update(status="ready", error=None)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            with self._lock:
                self._tasks[key].update(status="failed", error=str(exc))

    def resolve(self, catalog: Any, uid: str, episode_index: int) -> Path | None:
        source = self._source(catalog, uid, episode_index)
        target = self._target(uid, episode_index, source["fingerprint"])
        return target if target.is_file() and target.stat().st_size else None
