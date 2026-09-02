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
              instruction TEXT, task_index INTEGER, metadata_json TEXT NOT NULL DEFAULT '{}',
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
            CREATE TABLE IF NOT EXISTS video_files (
              dataset_uid TEXT NOT NULL, relative_path TEXT NOT NULL,
              size INTEGER NOT NULL DEFAULT 0, mtime_ns INTEGER NOT NULL DEFAULT 0,
              codec TEXT, width INTEGER, height INTEGER, fps REAL, frames INTEGER,
              integrity_status TEXT NOT NULL DEFAULT 'not_checked', error TEXT,
              PRIMARY KEY(dataset_uid, relative_path)
            );
            CREATE TABLE IF NOT EXISTS parquet_files (
              dataset_uid TEXT NOT NULL, relative_path TEXT NOT NULL,
              size INTEGER NOT NULL DEFAULT 0, mtime_ns INTEGER NOT NULL DEFAULT 0,
              rows INTEGER, schema_json TEXT NOT NULL DEFAULT '{}',
              integrity_status TEXT NOT NULL DEFAULT 'not_checked', error TEXT,
              PRIMARY KEY(dataset_uid, relative_path)
            );
            """)
            # Keep existing local catalogs compatible with the task-aware
            # episode browser introduced after the initial schema.
            episode_columns = {row[1] for row in db.execute("PRAGMA table_info(episodes)").fetchall()}
            if "task_index" not in episode_columns:
                db.execute("ALTER TABLE episodes ADD COLUMN task_index INTEGER")

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

    @staticmethod
    def _media_files(root: Path) -> Iterable[Path]:
        # LeRobot keeps encoded payloads below ``videos/`` (older exports may
        # use ``video/``).  Restrict traversal to those roots so a standard
        # scan never walks the much larger ``data/`` tree just to discover
        # media filenames.  For small/custom datasets, retain a top-level
        # fallback without recursively probing arbitrary directories.
        suffixes = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
        excluded = {".runtime_cache", ".conversion_work", ".conversion_logs", ".conversion_resume", ".git"}
        media_roots = [candidate for name in ("videos", "video")
                       if (candidate := root / name).is_dir()]
        if media_roots:
            for media_root in media_roots:
                for current, dirs, files in os.walk(media_root):
                    dirs[:] = [d for d in dirs if d not in excluded and not d.startswith(".")]
                    for name in files:
                        path = Path(current) / name
                        if path.suffix.lower() in suffixes:
                            yield path
            return
        try:
            for entry in root.iterdir():
                if entry.is_file() and entry.suffix.lower() in suffixes:
                    yield entry
        except OSError:
            return

    def _scan_video_files(self, db: sqlite3.Connection, uid: str, root: Path, deep: bool) -> None:
        rows: list[tuple[Any, ...]] = []
        existing = {
            row["relative_path"]: dict(row)
            for row in db.execute(
                "SELECT * FROM video_files WHERE dataset_uid=?", (uid,)
            ).fetchall()
        }
        seen: set[str] = set()
        for path in self._media_files(root):
            relative = str(path.relative_to(root))
            seen.add(relative)
            try:
                stat = path.stat()
                size, mtime_ns = stat.st_size, stat.st_mtime_ns
            except OSError as exc:
                rows.append((uid, relative, 0, 0, None, None, None, None, None, "error", str(exc)))
                continue
            cached = existing.get(relative)
            # Header metadata is reusable for standard scans.  A deep scan
            # must still decode a file unless a previous deep scan already
            # established pass/fail for this exact size and mtime.
            reusable_statuses = {"header_ok", "pass", "fail", "metadata_unavailable"}
            if (cached and cached["size"] == size and cached["mtime_ns"] == mtime_ns
                    and cached["integrity_status"] in reusable_statuses
                    and (not deep or cached["integrity_status"] in {"pass", "fail"})):
                rows.append(tuple(cached.get(column) for column in (
                    "dataset_uid", "relative_path", "size", "mtime_ns", "codec",
                    "width", "height", "fps", "frames", "integrity_status", "error")))
                if len(rows) >= 500:
                    db.executemany("""INSERT OR REPLACE INTO video_files
                        (dataset_uid,relative_path,size,mtime_ns,codec,width,height,fps,frames,integrity_status,error)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)""", rows)
                    rows.clear()
                continue
            codec = width = height = fps = frames = None
            integrity = "not_checked"
            error = None
            try:
                import av
                with av.open(str(path), mode="r") as container:
                    stream = next((s for s in container.streams if s.type == "video"), None)
                    if stream is None:
                        raise ValueError("no video stream")
                    codec = getattr(stream.codec_context, "name", None) or getattr(stream, "name", None)
                    width, height = stream.width, stream.height
                    fps_value = stream.average_rate
                    fps = float(fps_value) if fps_value else None
                    frames = int(stream.frames or 0) or None
                    if deep:
                        next(container.decode(stream), None)
                        integrity = "pass"
                    else:
                        integrity = "header_ok"
            except ImportError:
                integrity = "metadata_unavailable"
            except Exception as exc:  # codec/container errors become labels, not scan failures
                integrity, error = "fail", str(exc)
            rows.append((uid, relative, size, mtime_ns, codec, width, height, fps, frames, integrity, error))
            if len(rows) >= 500:
                db.executemany("""INSERT OR REPLACE INTO video_files
                    (dataset_uid,relative_path,size,mtime_ns,codec,width,height,fps,frames,integrity_status,error)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""", rows)
                rows.clear()
        if rows:
            db.executemany("""INSERT OR REPLACE INTO video_files
                (dataset_uid,relative_path,size,mtime_ns,codec,width,height,fps,frames,integrity_status,error)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", rows)
        stale = set(existing) - seen
        if stale:
            db.executemany(
                "DELETE FROM video_files WHERE dataset_uid=? AND relative_path=?",
                [(uid, relative) for relative in stale],
            )

    def _scan_parquet_files(self, db: sqlite3.Connection, uid: str, root: Path) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return
        rows: list[tuple[Any, ...]] = []
        existing = {
            row["relative_path"]: dict(row)
            for row in db.execute(
                "SELECT * FROM parquet_files WHERE dataset_uid=?", (uid,)
            ).fetchall()
        }
        seen: set[str] = set()
        for path in root.glob("data/**/*.parquet"):
            relative = str(path.relative_to(root))
            seen.add(relative)
            try:
                stat = path.stat()
                cached = existing.get(relative)
                if (cached and cached["size"] == stat.st_size
                        and cached["mtime_ns"] == stat.st_mtime_ns
                        and cached["integrity_status"] in {"pass", "fail"}):
                    rows.append(tuple(cached.get(column) for column in (
                        "dataset_uid", "relative_path", "size", "mtime_ns", "rows",
                        "schema_json", "integrity_status", "error")))
                    if len(rows) >= 500:
                        db.executemany("""INSERT OR REPLACE INTO parquet_files
                            (dataset_uid,relative_path,size,mtime_ns,rows,schema_json,integrity_status,error)
                            VALUES(?,?,?,?,?,?,?,?)""", rows)
                        rows.clear()
                    continue
                parquet = pq.ParquetFile(path)
                schema = {name: str(parquet.schema_arrow.field(name).type) for name in parquet.schema.names}
                rows.append((uid, relative, stat.st_size, stat.st_mtime_ns, parquet.metadata.num_rows,
                             json.dumps(schema), "pass", None))
            except Exception as exc:
                rows.append((uid, relative, 0, 0, None, "{}", "fail", str(exc)))
            if len(rows) >= 500:
                db.executemany("""INSERT OR REPLACE INTO parquet_files
                    (dataset_uid,relative_path,size,mtime_ns,rows,schema_json,integrity_status,error)
                    VALUES(?,?,?,?,?,?,?,?)""", rows)
                rows.clear()
        if rows:
            db.executemany("""INSERT OR REPLACE INTO parquet_files
                (dataset_uid,relative_path,size,mtime_ns,rows,schema_json,integrity_status,error)
                VALUES(?,?,?,?,?,?,?,?)""", rows)
        stale = set(existing) - seen
        if stale:
            db.executemany(
                "DELETE FROM parquet_files WHERE dataset_uid=? AND relative_path=?",
                [(uid, relative) for relative in stale],
            )

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
                        episode_rows: list[tuple[Any, ...]] = []
                        for item in self._iter_episode_metadata(root):
                            episode_rows.append((
                                uid, item["episode_index"], item["frames"], item["duration"],
                                item.get("instruction"), item.get("task_index"), item.get("metadata_json"),
                            ))
                            if len(episode_rows) >= 1000:
                                db.executemany("INSERT OR REPLACE INTO episodes(dataset_uid,episode_index,frames,duration,instruction,task_index,metadata_json) VALUES(?,?,?,?,?,?,?)", episode_rows)
                                episode_rows.clear()
                        if episode_rows:
                            db.executemany("INSERT OR REPLACE INTO episodes(dataset_uid,episode_index,frames,duration,instruction,task_index,metadata_json) VALUES(?,?,?,?,?,?,?)", episode_rows)
                    except (ImportError, OSError, ValueError):
                        pass
                    self._scan_video_files(db, uid, root, deep=mode == "deep")
                    # ParquetFile reads the footer/schema and row count without
                    # materializing the full table, so it is safe for both
                    # standard browser indexing and deep integrity scans.
                    self._scan_parquet_files(db, uid, root)
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

    def list_videos(self, uid: str, integrity_status: str | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM video_files WHERE dataset_uid=?", [uid]
        if integrity_status:
            query += " AND integrity_status=?"; args.append(integrity_status)
        with self._connect() as db:
            return [dict(row) for row in db.execute(query + " ORDER BY relative_path", args).fetchall()]

    def get_dataset(self, uid: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM datasets WHERE uid=?", (uid,)).fetchone()
        return self._dataset_row(row) if row else None

    def _dataset_row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["cameras"] = json.loads(result.pop("cameras") or "[]")
        result["schema"] = json.loads(result.pop("schema_json") or "{}")
        return result

    def list_episodes(self, uid: str, task_index: int | None = None) -> list[dict[str, Any]]:
        dataset = self.get_dataset(uid)
        if not dataset:
            return []
        with self._connect() as db:
            if task_index is None:
                rows = db.execute("SELECT * FROM episodes WHERE dataset_uid=? ORDER BY episode_index", (uid,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM episodes WHERE dataset_uid=? AND task_index=? ORDER BY episode_index", (uid, task_index)).fetchall()
        if rows:
            return [dict(r) for r in rows]
        # LeRobot stores episode summaries in meta/episodes/*.parquet. Read the
        # compact metadata on demand when only a quick dataset scan exists.
        # The unfiltered quick path intentionally stays O(1) and returns
        # lightweight placeholders; episode_preview loads one row on demand.
        # A task filter needs the compact metadata table to materialize the
        # matching episode indices.
        if task_index is None:
            return [{"dataset_uid": uid, "episode_index": i, "frames": 0, "duration": 0, "instruction": None}
                    for i in range(dataset["episodes"])]
        metadata_rows = list(self._iter_episode_metadata(Path(dataset["root"])))
        if metadata_rows:
            result = [{"dataset_uid": uid, **item} for item in metadata_rows
                      if task_index is None or item.get("task_index") == task_index]
            if result:
                return result
        # Keep a useful fallback when pyarrow is unavailable and expose the
        # dataset-level count.
        return []

    @classmethod
    def _load_task_map(cls, root: Path) -> dict[int, str]:
        try:
            import pyarrow.parquet as pq
            path = root / "meta" / "tasks.parquet"
            if not path.is_file():
                return {}
            # LeRobot exports created by different versions use either the
            # pandas default index name ``__index_level_0__`` or an explicit
            # ``task`` index. Read the compact table once and accept both.
            table = pq.read_table(path)
            names = set(table.column_names)
            if "task_index" not in names:
                return {}
            name_column = next((name for name in ("__index_level_0__", "task", "instruction") if name in names), None)
            if name_column is None:
                return {}
            result: dict[int, str] = {}
            for item in table.to_pylist():
                try:
                    name = item.get(name_column)
                    if name is not None and str(name):
                        result[int(item["task_index"])] = str(name)
                except (TypeError, ValueError):
                    continue
            return result
        except (ImportError, OSError, ValueError, TypeError):
            return {}

    @classmethod
    def list_tasks_for_root(cls, root: Path) -> list[dict[str, Any]]:
        """Return task names and episode counts from a LeRobot dataset root."""
        task_map = cls._load_task_map(root)
        if not task_map:
            return []
        counts = {index: 0 for index in task_map}
        for item in cls._iter_episode_metadata(root):
            index = item.get("task_index")
            if index in counts:
                counts[index] += 1
        return [{"task_index": index, "name": name, "episodes": counts.get(index, 0)}
                for index, name in sorted(task_map.items())]

    def list_tasks(self, uid: str) -> list[dict[str, Any]]:
        dataset = self.get_dataset(uid)
        if not dataset:
            return []
        return self.list_tasks_for_root(Path(dataset["root"]))

    @classmethod
    def _iter_episode_metadata(cls, root: Path) -> Iterable[dict[str, Any]]:
        """Yield normalized compact episode rows without reading frame data."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return
        info = cls._json(root / "meta" / "info.json")
        fps = float(info.get("fps", 1) or 1)
        task_map = cls._load_task_map(root)
        files = sorted(root.glob("meta/episodes*.parquet")) + sorted(root.glob("meta/episodes/**/*.parquet"))
        cameras = sorted(
            name for name, feature in info.get("features", {}).items()
            if name.startswith("observation.images") or (
                isinstance(feature, dict) and feature.get("dtype") == "video"
            )
        )
        logical_columns = [
            "episode_index", "frame_count", "frames", "length", "instruction", "tasks",
            "data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index",
        ]
        for camera in cameras:
            logical_columns.extend(
                f"videos/{camera}/{field}"
                for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            )
        for episode_file in files:
            try:
                parquet = pq.ParquetFile(episode_file)
                names = set(parquet.schema.names)
                columns = [name for name in logical_columns if name in names]
                if "tasks" not in columns:
                    columns.append("tasks")
                try:
                    batches = parquet.iter_batches(columns=columns, batch_size=2048)
                except Exception:
                    columns = [name for name in columns if name != "tasks"]
                    batches = parquet.iter_batches(columns=columns, batch_size=2048)
                for batch in batches:
                    for item in batch.to_pylist():
                        index = int(item.get("episode_index", item.get("index", 0)))
                        count = int(item.get("frame_count", item.get("frames", item.get("length", 0))) or 0)
                        instruction = item.get("instruction")
                        if instruction is None:
                            tasks = item.get("tasks")
                            instruction = tasks[0] if isinstance(tasks, list) and tasks else tasks
                        task_index = next((key for key, value in task_map.items() if value == instruction), None)
                        yield {"episode_index": index, "frames": count,
                               "duration": count / fps, "instruction": instruction,
                               "task_index": task_index,
                               "metadata_json": json.dumps(item, default=str)}
            except (OSError, ValueError):
                continue

    @classmethod
    def _load_episode_metadata(cls, root: Path, episode_index: int) -> dict[str, Any] | None:
        """Load one episode row from the compact metadata parquet."""
        for item in cls._iter_episode_metadata(root):
            if item["episode_index"] == episode_index:
                return {"dataset_uid": "", **item}
        return None

    @staticmethod
    def _stage_episode_records(stage_root: Path, episode_index: int) -> list[dict[str, Any]]:
        """Read small per-episode audit files without touching source videos."""
        records: list[dict[str, Any]] = []
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return records
        for path in stage_root.glob("**/audit/*.parquet"):
            try:
                parquet = pq.ParquetFile(path)
                names = set(parquet.schema.names)
                if "episode_index" not in names:
                    continue
                columns = [name for name in names if name != "episode_index"]
                columns.insert(0, "episode_index")
                for batch in parquet.iter_batches(columns=columns, batch_size=2048):
                    for item in batch.to_pylist():
                        if int(item.get("episode_index", -1)) == episode_index:
                            records.append({"file": str(path.relative_to(stage_root)), **item})
            except (OSError, ValueError):
                continue
        return records

    def episode_preview(self, uid: str, episode_index: int) -> dict[str, Any]:
        """Return bounded episode metadata, video references and stage labels."""
        dataset = self.get_dataset(uid)
        if not dataset:
            raise FileNotFoundError(uid)
        episodes = self.list_episodes(uid)
        episode = next((item for item in episodes if item["episode_index"] == episode_index), None)
        if episode is None:
            raise IndexError(episode_index)
        root = Path(dataset["root"])
        # Always merge the source metadata. Existing catalogs created by older
        # versions only cached basic episode columns and therefore lack the
        # per-camera file/time windows needed for concatenated MP4 shards.
        loaded = self._load_episode_metadata(root, episode_index)
        if loaded:
            episode = {**episode, **loaded, "dataset_uid": uid}
        info = self._json(root / "meta" / "info.json")
        metadata: dict[str, Any] = {}
        try:
            metadata = json.loads(episode.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        fps = float(info.get("fps", 1) or 1)
        frame_count = int(episode.get("frames", 0) or 0)
        duration = frame_count / fps
        dataset_from_index = int(metadata.get("dataset_from_index", 0) or 0)
        video_template = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        videos: list[dict[str, Any]] = []
        chunks_size = int(info.get("chunks_size", 1000) or 1000)
        for camera in dataset.get("cameras", []):
            prefix = f"videos/{camera}"
            chunk = metadata.get(f"{prefix}/chunk_index", episode_index // chunks_size)
            file_index = metadata.get(f"{prefix}/file_index", episode_index % chunks_size)
            try:
                relative = str(video_template.format(video_key=camera, chunk_index=int(chunk), file_index=int(file_index)))
            except (KeyError, ValueError):
                relative = f"videos/{camera}/chunk-{int(chunk):03d}/file-{int(file_index):03d}.mp4"
            path = (root / relative).resolve()
            if path.is_file() and root.resolve() in path.parents:
                source_start = float(metadata.get(f"{prefix}/from_timestamp", 0) or 0)
                source_end = float(metadata.get(f"{prefix}/to_timestamp", source_start + duration) or 0)
                source_duration = max(0.0, source_end - source_start)
                tolerance = max(1.0 / fps, 1e-6)
                videos.append({
                    "camera": camera,
                    "relative_path": relative,
                    "url": f"/api/videos/{uid}/{relative}",
                    "chunk_index": int(chunk),
                    "file_index": int(file_index),
                    "source_start": source_start,
                    "source_end": source_end,
                    "timestamp_start": source_start,
                    "timestamp_end": source_end,
                    "duration": source_duration,
                    "integrity_status": "pass" if abs(source_duration - duration) <= tolerance else "duration_mismatch",
                })

        stage_results: list[dict[str, Any]] = []
        curation_root = self.data_root / "data_curation"
        for stage_id in range(1, 9):
            stage_base = curation_root / f"stage{stage_id}"
            stage_roots = [path for path in [stage_base / uid, *sorted(stage_base.glob(f"{uid}_*"))] if path.is_dir()]
            if not stage_roots:
                continue
            for stage_root in stage_roots:
                manifests = sorted(stage_root.glob("**/manifest.json"))
                for manifest_path in manifests:
                    try:
                        manifest = self._json(manifest_path)
                        summary = self._json(manifest_path.parent / "reports" / "summary.json")
                    except OSError:
                        continue
                    stage_results.append({
                        "stage_id": stage_id,
                        "run_id": manifest_path.parent.name,
                        "stage": manifest.get("stage", f"Stage {stage_id}"),
                        "detector_version": manifest.get("detector_version"),
                        "coordinate_system": manifest.get("coordinate_system", "episode_frame"),
                        "summary": summary,
                        "records": self._stage_episode_records(manifest_path.parent, episode_index),
                    })
                # Some detector implementations write audit files without a
                # manifest (for example the smoke runner); expose them as a run.
                if not manifests:
                    records = self._stage_episode_records(stage_root, episode_index)
                    if records:
                        stage_results.append({
                            "stage_id": stage_id, "run_id": stage_root.name, "stage": f"Stage {stage_id}",
                            "coordinate_system": "episode_frame", "records": records,
                        })

        # Human/automatic annotations are stored separately from immutable
        # stage artifacts.  Surface them as another comparable run instead
        # of overwriting detector output.
        annotations = self.list_annotations(uid, episode_index)
        for annotation in annotations:
            stage_id = annotation.get("stage_id")
            if stage_id is None:
                continue
            result = next((item for item in stage_results
                           if item["stage_id"] == stage_id and item["run_id"] == "annotations"), None)
            if result is None:
                result = {
                    "stage_id": stage_id, "run_id": "annotations", "stage": f"Stage {stage_id}",
                    "coordinate_system": "episode_frame", "records": [],
                }
                stage_results.append(result)
            result.setdefault("records", []).append(annotation)

        return {
            "dataset": dataset,
            "episode": episode,
            "timeline": {
                "coordinate_system": "episode_relative",
                "frame_count": frame_count,
                "fps": fps,
                "duration": duration,
                "dataset_from_index": dataset_from_index,
            },
            "videos": videos,
            "stage_results": stage_results,
        }

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
        root = Path(dataset["root"])
        files = sorted(root.glob("data/**/*.parquet"))
        if not files:
            return []
        # Episode metadata points to the exact data partition.  Restricting
        # the scan to that file is important for datasets with millions of
        # frames; the fallback keeps custom datasets usable.
        episode_meta = next((item for item in self.list_episodes(uid) if item["episode_index"] == episode_index), None)
        loaded = self._load_episode_metadata(root, episode_index)
        metadata_json = loaded.get("metadata_json") if loaded else (episode_meta.get("metadata_json") if episode_meta else None)
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                chunk = metadata.get("data/chunk_index")
                file_index = metadata.get("data/file_index")
                if chunk is not None and file_index is not None:
                    candidate = root / "data" / f"chunk-{int(chunk):03d}" / f"file-{int(file_index):03d}.parquet"
                    if candidate.is_file():
                        files = [candidate]
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        try:
            import pyarrow.dataset as ds
            dataset_obj = ds.dataset([str(p) for p in files], format="parquet")
            available = set(dataset_obj.schema.names)
            aliases = {"state": "observation.state"}
            requested = [aliases.get(field, field) for field in fields]
            names = [field for field in requested if field in available]
            if "timestamp" in available and "timestamp" not in names:
                names.insert(0, "timestamp")
            if not names:
                return []
            filter_expr = ds.field("episode_index") == episode_index if "episode_index" in available else None
            scanner = dataset_obj.scanner(columns=names, filter=filter_expr, batch_size=min(limit, 2048))
            result: list[dict[str, Any]] = []
            for batch in scanner.to_batches():
                result.extend(batch.to_pylist())
                if len(result) >= limit:
                    break
            result = result[:limit]
            fps = float(self._json(root / "meta" / "info.json").get("fps", 1) or 1)
            for position, row in enumerate(result):
                frame_index = row.get("frame_index")
                try:
                    row["episode_time"] = int(frame_index) / fps if frame_index is not None else position / fps
                except (TypeError, ValueError):
                    row["episode_time"] = position / fps
            return result
        except (ImportError, OSError, ValueError):
            return []
