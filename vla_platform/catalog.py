from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


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
            """)

    @staticmethod
    def _json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def scan(self) -> list[dict[str, Any]]:
        """Scan dataset directories without loading video payloads into memory."""
        found: list[dict[str, Any]] = []
        if not self.data_root.exists():
            return found
        info_paths = sorted(self.data_root.glob("*/meta/info.json"))
        info_paths += sorted(self.data_root.glob("*/**/dataset/meta/info.json"))
        seen_roots: set[Path] = set()
        for info_path in info_paths:
            root = info_path.parent.parent
            if root in seen_roots:
                continue
            seen_roots.add(root)
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
            total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
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
                try:
                    import pyarrow.dataset as ds
                    episode_files = sorted(root.glob("meta/episodes*.parquet"))
                    if episode_files:
                        table = ds.dataset([str(p) for p in episode_files], format="parquet").to_table()
                        for item in table.to_pylist():
                            index = int(item.get("episode_index", item.get("index", 0)))
                            count = int(item.get("frame_count", item.get("frames", 0)) or 0)
                            db.execute("INSERT OR REPLACE INTO episodes(dataset_uid,episode_index,frames,duration,instruction,metadata_json) VALUES(?,?,?,?,?,?)",
                                (uid, index, count, count / float(info.get("fps", 1) or 1), item.get("instruction"), json.dumps(item, default=str)))
                except (ImportError, OSError, ValueError):
                    pass
            found.append(row)
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
