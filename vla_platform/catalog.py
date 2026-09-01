from __future__ import annotations

import json
import os
import sqlite3
import time
import threading
from pathlib import Path
from typing import Any, Callable, Iterable


class Catalog:
    """SQLite-backed index for immutable local LeRobot datasets."""

    def __init__(self, db_path: str | Path, data_root: str | Path):
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA busy_timeout=5000")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
              uid TEXT PRIMARY KEY, root TEXT NOT NULL, codebase_version TEXT,
              episodes INTEGER NOT NULL DEFAULT 0, frames INTEGER NOT NULL DEFAULT 0,
              duration REAL NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,
              cameras TEXT NOT NULL DEFAULT '[]', schema_json TEXT NOT NULL DEFAULT '{}',
              scanned_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes (
              dataset_uid TEXT NOT NULL, episode_index INTEGER NOT NULL,
              frames INTEGER NOT NULL DEFAULT 0, duration REAL NOT NULL DEFAULT 0,
              instruction TEXT, metadata_json TEXT NOT NULL DEFAULT '{}',
              PRIMARY KEY(dataset_uid, episode_index)
            );
            CREATE TABLE IF NOT EXISTS annotations (
              annotation_id TEXT PRIMARY KEY, dataset_uid TEXT NOT NULL,
              episode_index INTEGER NOT NULL, stage_id INTEGER, label_type TEXT NOT NULL,
              status TEXT NOT NULL, severity TEXT, score REAL, threshold REAL,
              frame_start INTEGER, frame_end INTEGER, entity_type TEXT,
              entity_name TEXT, reason_code TEXT, source TEXT NOT NULL,
              review_status TEXT NOT NULL DEFAULT 'pending', reviewer TEXT,
              comment TEXT, created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_annotations_dataset ON annotations(dataset_uid, episode_index);
            CREATE TABLE IF NOT EXISTS scan_fingerprints (
              root TEXT PRIMARY KEY, dataset_uid TEXT NOT NULL,
              info_mtime_ns INTEGER NOT NULL, info_size INTEGER NOT NULL,
              episodes_mtime_ns INTEGER NOT NULL DEFAULT 0,
              scanned_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scan_jobs (
              scan_id TEXT PRIMARY KEY, status TEXT NOT NULL, mode TEXT NOT NULL,
              root TEXT, phase TEXT, current INTEGER NOT NULL DEFAULT 0,
              total INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 0,
              skipped INTEGER NOT NULL DEFAULT 0, datasets INTEGER NOT NULL DEFAULT 0,
              rate REAL, eta_seconds REAL, error TEXT,
              created_at REAL NOT NULL, started_at REAL, finished_at REAL
            );
            """)

    def create_scan_job(self, job: dict[str, Any]) -> None:
        columns = ("scan_id", "status", "mode", "root", "phase", "current", "total",
                   "completed", "skipped", "datasets", "rate", "eta_seconds", "error",
                   "created_at", "started_at", "finished_at")
        values = {column: job.get(column) for column in columns}
        with self._connect() as db:
            db.execute(f"INSERT OR REPLACE INTO scan_jobs({','.join(columns)}) VALUES({','.join(':'+c for c in columns)})", values)

    def update_scan_job(self, scan_id: str, values: dict[str, Any]) -> None:
        allowed = {"status", "mode", "root", "phase", "current", "total", "completed", "skipped",
                   "datasets", "rate", "eta_seconds", "error", "created_at", "started_at", "finished_at"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        assignments = ",".join(f"{key}=:{key}" for key in values)
        values["scan_id"] = scan_id
        with self._connect() as db:
            db.execute(f"UPDATE scan_jobs SET {assignments} WHERE scan_id=:scan_id", values)

    def get_scan_job(self, scan_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM scan_jobs WHERE scan_id=?", (scan_id,)).fetchone()
        return dict(row) if row else None

    def recover_interrupted_scans(self) -> int:
        """Mark jobs from a previous API process as interrupted and retryable."""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scan_jobs SET status='failed', error='scanner process restarted', finished_at=? "
                "WHERE status IN ('queued','running','cancelling')", (time.time(),)
            )
        return cursor.rowcount

    @staticmethod
    def _json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _dataset_roots(self, scan_root: Path | None = None) -> list[tuple[Path, Path]]:
        """Find dataset markers while pruning caches, videos and temp trees."""
        base = (scan_root or self.data_root).resolve()
        if not base.exists():
            return []
        excluded = {".runtime_cache", ".conversion_work", ".conversion_logs", ".conversion_resume",
                    ".lerobot-datasets-cache", ".git", "videos", "video", "artifacts"}
        candidates: list[tuple[Path, Path]] = []

        def add_if_dataset(root: Path) -> None:
            info = root / "meta" / "info.json"
            if info.is_file():
                candidates.append((root, info))

        add_if_dataset(base)
        # Normal staging layout: <root>/<dataset>/meta/info.json. Stage runs
        # are intentionally discovered only under data_curation/stageN.
        containers: list[Path] = []
        default_containers = {"lerobot_v3_0", "lerobot_v2_1"}
        for child in os.scandir(base):
            if not child.is_dir(follow_symlinks=False) or child.name in excluded:
                continue
            child_path = Path(child.path)
            # At the staging root, do not recursively inspect arbitrary cache,
            # raw, or curation trees. They must be requested explicitly via
            # scan(root=...). This keeps the default quick scan bounded.
            if base == self.data_root.resolve() and child.name not in default_containers:
                add_if_dataset(child_path)
                continue
            containers.append(child_path)
        for container in containers:
            add_if_dataset(container)
            if container.name == "data_curation":
                containers.extend(sorted((p for p in container.glob("stage[1-8]") if p.is_dir())))
        visited: set[Path] = set()

        def discover(container: Path, depth: int = 0, max_depth: int = 4) -> None:
            container = container.resolve()
            if container in visited or depth > max_depth or not container.is_dir():
                return
            visited.add(container)
            add_if_dataset(container)
            if (container / "meta" / "info.json").is_file() or depth == max_depth:
                return
            try:
                children = list(os.scandir(container))
            except OSError:
                return
            for child in children:
                if child.is_dir(follow_symlinks=False) and child.name not in excluded and not child.name.startswith("."):
                    discover(Path(child.path), depth + 1, max_depth)

        for container in containers:
            # The staging root is intentionally shallow: only datasets with a
            # direct meta/info marker are considered. Nested layouts can be
            # scanned explicitly with ``root=...`` without walking every
            # payload directory in the global tree.
            depth_limit = 1 if base == self.data_root.resolve() and container.name in default_containers else 4
            discover(container, max_depth=depth_limit)
        unique: dict[Path, Path] = {}
        for root, info in candidates:
            unique.setdefault(root.resolve(), info.resolve())
        return sorted(unique.items(), key=lambda item: str(item[0]))

    @staticmethod
    def _episode_marker(root: Path) -> tuple[int, int]:
        marker = 0
        size = 0
        for path in sorted(root.glob("meta/episodes*.parquet")) + sorted(root.glob("meta/episodes/**/*.parquet")):
            try:
                stat = path.stat()
            except OSError:
                continue
            marker = max(marker, stat.st_mtime_ns)
            size += stat.st_size
        return marker, size

    def _fingerprint(self, root: Path, info_path: Path) -> tuple[int, int, int]:
        info_stat = info_path.stat()
        episode_mtime, _ = self._episode_marker(root)
        return info_stat.st_mtime_ns, info_stat.st_size, episode_mtime

    def scan(
        self,
        mode: str = "quick",
        scan_root: str | Path | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        """Scan metadata without loading videos; ``deep`` alone walks file sizes."""
        if mode not in {"quick", "standard", "deep"}:
            raise ValueError("scan mode must be quick, standard, or deep")
        found: list[dict[str, Any]] = []
        if progress:
            progress({"phase": "discovering", "current": 0, "total": 0, "skipped": False})
        roots = self._dataset_roots(Path(scan_root) if scan_root else None)
        total = len(roots)
        for position, (root, info_path) in enumerate(roots, start=1):
            if cancel and cancel.is_set():
                break
            info_mtime, info_size, episodes_mtime = self._fingerprint(root, info_path)
            with self._connect() as db:
                previous = db.execute("SELECT info_mtime_ns,info_size,episodes_mtime_ns FROM scan_fingerprints WHERE root=?", (str(root),)).fetchone()
                if mode == "quick" and previous and tuple(previous) == (info_mtime, info_size, episodes_mtime):
                    existing = db.execute("SELECT * FROM datasets WHERE root=?", (str(root),)).fetchone()
                    if existing:
                        row = self._dataset_row(existing)
                        found.append(row)
                        if progress:
                            progress({"phase": "indexing", "current": position, "total": total, "uid": row["uid"], "skipped": True})
                        continue
            info = self._json(info_path)
            # Stage runs use stage<N>/<dataset>/<run_id>/dataset; preserve the
            # stable dataset uid while still indexing the immutable artifact.
            uid = root.name
            if uid == "dataset" and root.parent.parent != self.data_root:
                uid = root.parent.parent.name
            features = info.get("features", {})
            cameras = sorted(k for k in features if k.startswith("observation.images"))
            episodes = int(info.get("total_episodes", 0) or 0)
            frames = int(info.get("total_frames", 0) or 0)
            total_bytes = 0
            if mode == "deep":
                for current, dirs, files in os.walk(root):
                    dirs[:] = [d for d in dirs if d not in {".runtime_cache", ".conversion_work", ".conversion_logs", ".conversion_resume"}]
                    for name in files:
                        try:
                            total_bytes += (Path(current) / name).stat().st_size
                        except OSError:
                            pass
            row = {
                "uid": uid, "root": str(root),
                "codebase_version": info.get("codebase_version", "unknown"),
                "episodes": episodes, "frames": frames,
                "duration": frames / float(info.get("fps", 1) or 1),
                "bytes": total_bytes, "cameras": cameras, "schema": features,
                "scanned_at": time.time(),
            }
            with self._connect() as db:
                db.execute("""INSERT INTO datasets(uid,root,codebase_version,episodes,frames,duration,bytes,cameras,schema_json,scanned_at)
                    VALUES(:uid,:root,:codebase_version,:episodes,:frames,:duration,:bytes,:cameras,:schema,:scanned_at)
                    ON CONFLICT(uid) DO UPDATE SET root=excluded.root, codebase_version=excluded.codebase_version,
                    episodes=excluded.episodes, frames=excluded.frames, duration=excluded.duration, bytes=excluded.bytes,
                    cameras=excluded.cameras, schema_json=excluded.schema_json, scanned_at=excluded.scanned_at""",
                    {**row, "cameras": json.dumps(cameras), "schema": json.dumps(features)})
                # Episodes are small metadata parquet files; index them without
                # touching image/video payloads.
                if mode in {"standard", "deep"}:
                    try:
                        import pyarrow.parquet as pq
                        episode_files = sorted(root.glob("meta/episodes*.parquet")) + sorted(root.glob("meta/episodes/**/*.parquet"))
                        episode_rows: list[tuple[Any, ...]] = []
                        for episode_file in episode_files:
                            parquet = pq.ParquetFile(episode_file)
                            names = set(parquet.schema.names)
                            columns = [name for name in ("episode_index", "frame_count", "frames", "instruction") if name in names]
                            if not columns:
                                continue
                            for batch in parquet.iter_batches(columns=columns, batch_size=1000):
                                for item in batch.to_pylist():
                                    index = int(item.get("episode_index", item.get("index", 0)))
                                    count = int(item.get("frame_count", item.get("frames", 0)) or 0)
                                    episode_rows.append((uid, index, count, count / float(info.get("fps", 1) or 1), item.get("instruction"), json.dumps(item, default=str)))
                                    if len(episode_rows) >= 1000:
                                        db.executemany("INSERT OR REPLACE INTO episodes(dataset_uid,episode_index,frames,duration,instruction,metadata_json) VALUES(?,?,?,?,?,?)", episode_rows)
                                        episode_rows.clear()
                        if episode_rows:
                            db.executemany("INSERT OR REPLACE INTO episodes(dataset_uid,episode_index,frames,duration,instruction,metadata_json) VALUES(?,?,?,?,?,?)", episode_rows)
                    except (ImportError, OSError, ValueError):
                        pass
                db.execute("INSERT OR REPLACE INTO scan_fingerprints(root,dataset_uid,info_mtime_ns,info_size,episodes_mtime_ns,scanned_at) VALUES(?,?,?,?,?,?)",
                            (str(root), uid, info_mtime, info_size, episodes_mtime, time.time()))
            found.append(row)
            if progress:
                progress({"phase": "indexing", "current": position, "total": total, "uid": uid, "skipped": False})
        return found

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM datasets ORDER BY uid").fetchall()
        return [self._dataset_row(r) for r in rows]

    def get_dataset(self, uid: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM datasets WHERE uid=?", (uid,)).fetchone()
        return self._dataset_row(row) if row else None

    def _dataset_row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["cameras"] = json.loads(result.pop("cameras") or "[]")
        result["schema"] = json.loads(result.pop("schema_json") or "{}")
        return result

    def list_episodes(self, uid: str) -> list[dict[str, Any]]:
        dataset = self.get_dataset(uid)
        if not dataset:
            return []
        with self._connect() as db:
            rows = db.execute("SELECT * FROM episodes WHERE dataset_uid=? ORDER BY episode_index", (uid,)).fetchall()
        if rows:
            return [dict(r) for r in rows]
        # LeRobot stores episode summaries in meta/episodes/*.parquet. Keep a useful
        # fallback when pyarrow is unavailable and expose the dataset-level count.
        return [{"dataset_uid": uid, "episode_index": i, "frames": 0, "duration": 0, "instruction": None}
                for i in range(dataset["episodes"])]

    def add_annotation(self, annotation: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as db:
            columns = ",".join(annotation)
            placeholders = ",".join(f":{k}" for k in annotation)
            db.execute(f"INSERT INTO annotations({columns}) VALUES({placeholders})", annotation)
        return annotation

    def list_annotations(self, uid: str, episode_index: int | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM annotations WHERE dataset_uid=?", [uid]
        if episode_index is not None:
            query += " AND episode_index=?"; args.append(episode_index)
        with self._connect() as db:
            return [dict(r) for r in db.execute(query + " ORDER BY created_at", args).fetchall()]

    def update_annotation_review(self, annotation_id: str, review_status: str, reviewer: str | None, comment: str | None) -> dict[str, Any] | None:
        with self._connect() as db:
            db.execute("UPDATE annotations SET review_status=?, reviewer=?, comment=? WHERE annotation_id=?",
                       (review_status, reviewer, comment, annotation_id))
            row = db.execute("SELECT * FROM annotations WHERE annotation_id=?", (annotation_id,)).fetchone()
        return dict(row) if row else None

    def resolve_path(self, uid: str, relative: str) -> Path:
        dataset = self.get_dataset(uid)
        if not dataset:
            raise FileNotFoundError(uid)
        base, path = Path(dataset["root"]).resolve(), (Path(dataset["root"]) / relative).resolve()
        if path != base and base not in path.parents:
            raise PermissionError("path escapes dataset root")
        if not path.is_file():
            raise FileNotFoundError(relative)
        return path

    def episode_series(self, uid: str, episode_index: int, fields: list[str], limit: int = 2000) -> list[dict[str, Any]]:
        """Read a bounded state/action sample from LeRobot parquet files."""
        dataset = self.get_dataset(uid)
        if not dataset:
            raise FileNotFoundError(uid)
        files = sorted(Path(dataset["root"]).glob("data/**/*.parquet"))
        if not files:
            return []
        try:
            import pyarrow.dataset as ds
            table = ds.dataset([str(p) for p in files], format="parquet").to_table()
            names = [f for f in fields if f in table.column_names]
            if "episode_index" in table.column_names:
                mask = table["episode_index"] == episode_index
                table = table.filter(mask)
            if names:
                table = table.select(names)
            return table.slice(0, limit).to_pylist()
        except (ImportError, OSError, ValueError):
            return []
