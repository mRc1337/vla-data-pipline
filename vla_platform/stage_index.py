from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "vla_stage_search_index_v1"
MANIFEST_NAME = "search_index_manifest.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def publish_stage_index_snapshot(
    stage_root: str | Path,
    stage_id: int,
    payloads: Iterable[dict[str, Any]],
    *,
    shard_size: int = 1024,
) -> dict[str, Any]:
    """Atomically publish a stable, sharded search-index snapshot.

    Shards are bucketed by episode index, so changing one Episode rewrites at
    most one shard. The manifest is committed last and is the sole visibility
    boundary consumed by the catalog indexer.
    """
    if not 1 <= stage_id <= 8:
        raise ValueError("stage_id must be between 1 and 8")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to publish a stage search index") from exc

    root = Path(stage_root)
    shard_root = root / "search_index"
    shard_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    previous = _read_manifest(manifest_path)
    source_manifest_path = root / "manifest.json"
    source_manifest_fingerprint = _file_sha256(source_manifest_path)
    previous_shards = {
        str(item.get("path")): item
        for item in previous.get("shards", [])
        if isinstance(item, dict) and item.get("path")
    }
    previous_episodes = {
        int(value) for value in previous.get("episode_indices", [])
        if isinstance(value, int) or str(value).isdigit()
    }

    by_bucket: dict[int, list[tuple[int, str]]] = defaultdict(list)
    seen: set[int] = set()
    for payload in payloads:
        if payload.get("episode_index") is None:
            raise ValueError("stage index payload is missing episode_index")
        episode_index = int(payload["episode_index"])
        if episode_index in seen:
            raise ValueError(f"duplicate episode_index: {episode_index}")
        seen.add(episode_index)
        by_bucket[episode_index // shard_size].append((episode_index, _canonical_json(payload)))

    shards: list[dict[str, Any]] = []
    changed_shards: list[str] = []
    for bucket, rows in sorted(by_bucket.items()):
        rows.sort(key=lambda value: value[0])
        relative = f"search_index/part-{bucket:06d}.parquet"
        fingerprint = hashlib.sha256(
            "\n".join(payload for _, payload in rows).encode("utf-8")
        ).hexdigest()
        descriptor = {
            "path": relative,
            "fingerprint": fingerprint,
            "rows": len(rows),
            "min_episode_index": rows[0][0],
            "max_episode_index": rows[-1][0],
        }
        shards.append(descriptor)
        target = root / relative
        old = previous_shards.get(relative)
        if old and old.get("fingerprint") == fingerprint and target.is_file():
            continue
        temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
        try:
            table = pa.table({
                "episode_index": pa.array([index for index, _ in rows], type=pa.int64()),
                "payload_json": pa.array([payload for _, payload in rows], type=pa.string()),
            })
            pq.write_table(table, temporary, compression="zstd")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        changed_shards.append(relative)

    removed_shards = sorted(set(previous_shards) - {item["path"] for item in shards})
    tombstones = sorted(previous_episodes - seen)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": stage_id,
        "source_manifest": (
            {"path": "manifest.json", "fingerprint": source_manifest_fingerprint}
            if source_manifest_fingerprint else None
        ),
        "episode_indices": sorted(seen),
        "shards": shards,
    }
    generation = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    if previous.get("generation") == generation:
        return previous
    manifest = {
        **identity,
        "generation": generation,
        "version": generation,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(seen),
        "changes": {
            "upsert_shards": changed_shards,
            "removed_shards": removed_shards,
            "tombstones": tombstones,
        },
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def publish_legacy_json_snapshot(
    stage_root: str | Path, stage_id: int, *, shard_size: int = 1024
) -> dict[str, Any]:
    """Build one compact snapshot from an existing per-Episode JSON layout."""
    root = Path(stage_root)
    payloads: dict[int, dict[str, Any]] = {}
    for directory in (root, root / "episodes", root / "labels"):
        for path in directory.glob("episode_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                episode_index = int(payload["episode_index"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads[episode_index] = payload
    return publish_stage_index_snapshot(
        root, stage_id, payloads.values(), shard_size=shard_size
    )


def publish_catalog_snapshot(
    stage_root: str | Path,
    stage_id: int,
    catalog_db: str | Path,
    dataset_uid: str,
    *,
    shard_size: int = 1024,
) -> dict[str, Any]:
    """Backfill a completed Stage snapshot from already indexed JSON payloads."""
    root = Path(stage_root)
    source_manifest = _read_manifest(root / "manifest.json")
    if source_manifest.get("run_complete") is not True:
        raise ValueError("catalog backfill requires a completed Stage manifest")
    with sqlite3.connect(catalog_db) as db:
        rows = db.execute(
            """SELECT episode_index,details_json FROM stage_episode_results
            WHERE dataset_uid=? AND stage_id=? AND artifact_status='available'
            ORDER BY episode_index""",
            (dataset_uid, stage_id),
        ).fetchall()
    payloads = [json.loads(row[1]) for row in rows]
    expected = source_manifest.get("episode_count", source_manifest.get("requested_episode_count"))
    if expected is not None and len(payloads) != int(expected):
        raise ValueError(
            f"catalog contains {len(payloads)} available rows, completed manifest declares {expected}"
        )
    if any(int(payload.get("episode_index", -1)) != int(row[0]) for payload, row in zip(payloads, rows)):
        raise ValueError("catalog stage payloads are incomplete")
    return publish_stage_index_snapshot(root, stage_id, payloads, shard_size=shard_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a sharded Stage search index from legacy episode JSON files."
    )
    parser.add_argument("stage_root", type=Path)
    parser.add_argument("--stage-id", required=True, type=int, choices=range(1, 9))
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--catalog-db", type=Path)
    parser.add_argument("--dataset-uid")
    args = parser.parse_args()
    if bool(args.catalog_db) != bool(args.dataset_uid):
        parser.error("--catalog-db and --dataset-uid must be provided together")
    if args.catalog_db:
        manifest = publish_catalog_snapshot(
            args.stage_root, args.stage_id, args.catalog_db, args.dataset_uid,
            shard_size=args.shard_size,
        )
    else:
        manifest = publish_legacy_json_snapshot(
            args.stage_root, args.stage_id, shard_size=args.shard_size
        )
    print(json.dumps({
        "manifest": str(args.stage_root / MANIFEST_NAME),
        "generation": manifest["generation"],
        "file_count": manifest["file_count"],
        "changed_shards": len(manifest["changes"]["upsert_shards"]),
        "tombstones": len(manifest["changes"]["tombstones"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
