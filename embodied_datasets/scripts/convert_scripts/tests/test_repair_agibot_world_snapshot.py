from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

import pytest

from repair_agibot_world_snapshot import (
    DEFAULT_MANIFEST,
    FileSpec,
    RepairError,
    RepairManifest,
    _canonical_json,
    _partial_path,
    audit_snapshot,
    download_and_install,
    load_manifest,
    main,
    verify_file,
)


REVISION = "1" * 40


def test_committed_manifest_pins_diagnosed_scope():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert manifest.repo_id == "agibot-world/AgiBotWorld2026"
    assert manifest.revision == "4423752b8fa754533c0cd1e6fa3ff9e4cc690566"
    assert len(manifest.files) == 320
    assert len(manifest.repair_files) == 103
    assert sum(item.baseline_size == 0 for item in manifest.repair_files) == 102
    assert sum(item.baseline_size > 0 for item in manifest.repair_files) == 1
    assert all(item.lfs_sha256 for item in manifest.repair_files)


def _manifest_payload(files: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "repo_id": "owner/dataset",
        "revision": REVISION,
        "scope": {
            "file_count": len(files),
            "repair_file_count": sum(bool(item["repair_required"]) for item in files),
            "total_bytes": sum(int(item["size"]) for item in files),
            "repair_bytes": sum(int(item["size"]) for item in files if item["repair_required"]),
        },
        "files": files,
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _write_manifest(path: Path, files: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(_manifest_payload(files)), encoding="utf-8")


def _lfs_file(path: str, data: bytes, *, baseline_size: int) -> dict[str, object]:
    return {
        "path": path,
        "size": len(data),
        "baseline_size": baseline_size,
        "repair_required": baseline_size != len(data),
        "lfs_sha256": hashlib.sha256(data).hexdigest(),
    }


class _PayloadServer:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.ranges: list[str | None] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                path = unquote(urlsplit(self.path).path)
                marker = f"/resolve/{REVISION}/"
                if marker not in path:
                    self.send_error(404)
                    return
                relative = path.split(marker, 1)[1]
                if relative not in owner.payloads:
                    self.send_error(404)
                    return
                payload = owner.payloads[relative]
                range_value = self.headers.get("Range")
                owner.ranges.append(range_value)
                start = int(range_value.removeprefix("bytes=").removesuffix("-")) if range_value else 0
                body = payload[start:]
                self.send_response(206 if range_value else 200)
                self.send_header("Content-Length", str(len(body)))
                if range_value:
                    self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}", self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def test_manifest_fingerprint_and_official_hash_validation(tmp_path: Path):
    data = b"official-lfs-object"
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_lfs_file("task/object.tar.gz", data, baseline_size=0)])
    manifest = load_manifest(manifest_path)
    assert manifest.revision == REVISION
    assert manifest.repair_files[0].expected_hash == hashlib.sha256(data).hexdigest()

    payload = json.loads(manifest_path.read_text())
    payload["files"][0]["size"] += 1
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(RepairError, match="fingerprint mismatch"):
        load_manifest(manifest_path)


def test_verify_file_supports_lfs_sha256_and_git_blob_sha1(tmp_path: Path):
    lfs_data, git_data = b"lfs", b"small git file\n"
    (tmp_path / "lfs.bin").write_bytes(lfs_data)
    (tmp_path / "README.md").write_bytes(git_data)
    lfs = FileSpec("lfs.bin", len(lfs_data), len(lfs_data), False, hashlib.sha256(lfs_data).hexdigest())
    blob = hashlib.sha1(f"blob {len(git_data)}\0".encode() + git_data).hexdigest()
    git = FileSpec("README.md", len(git_data), len(git_data), False, git_blob_sha1=blob)
    assert verify_file(tmp_path, lfs).ok
    assert verify_file(tmp_path, git).ok

    (tmp_path / "lfs.bin").write_bytes(b"bad")
    assert not verify_file(tmp_path, lfs).ok


def test_main_resumes_partial_then_audits_entire_scope(tmp_path: Path):
    bad_path, good_path = "task/bad archive.tar.gz", "task/good.tar.gz"
    official_bad = b"verified replacement bytes"
    official_good = b"already correct"
    root = tmp_path / "snapshot"
    (root / "task").mkdir(parents=True)
    (root / bad_path).write_bytes(b"")
    (root / good_path).write_bytes(official_good)
    manifest_path = tmp_path / "manifest.json"
    files = sorted(
        [
            _lfs_file(bad_path, official_bad, baseline_size=0),
            _lfs_file(good_path, official_good, baseline_size=len(official_good)),
        ],
        key=lambda item: str(item["path"]),
    )
    _write_manifest(manifest_path, files)
    manifest = load_manifest(manifest_path)
    partial = _partial_path(root / bad_path, REVISION)
    partial.write_bytes(official_bad[:8])
    report = tmp_path / "report.json"

    with _PayloadServer({bad_path: official_bad}) as (endpoint, server):
        result = main(
            [
                "--root",
                str(root),
                "--manifest",
                str(manifest_path),
                "--endpoint",
                endpoint,
                "--report",
                str(report),
                "--retries",
                "1",
            ]
        )

    assert result == 0
    assert server.ranges == ["bytes=8-"]
    assert (root / bad_path).read_bytes() == official_bad
    assert not partial.exists()
    result_payload = json.loads(report.read_text())
    assert result_payload["repair"] == [{"path": bad_path, "status": "downloaded"}]
    assert result_payload["audit"] == {"checked": 2, "failed": [], "passed": 2}


def test_bad_download_never_replaces_existing_file(tmp_path: Path):
    root = tmp_path / "snapshot"
    root.mkdir()
    original, expected, corrupt = b"keep this", b"official data", b"corrupt bytes"
    (root / "object.bin").write_bytes(original)
    spec = FileSpec(
        "object.bin",
        len(expected),
        len(original),
        True,
        hashlib.sha256(expected).hexdigest(),
    )
    manifest = RepairManifest("owner/dataset", REVISION, (spec,), "unused")

    with _PayloadServer({"object.bin": corrupt}) as (endpoint, _server):
        with pytest.raises(RepairError, match="download failed"):
            download_and_install(root, manifest, spec, endpoint=endpoint, retries=1)

    assert (root / "object.bin").read_bytes() == original
    assert not _partial_path(root / "object.bin", REVISION).exists()


def test_audit_reports_missing_file_without_creating_parent(tmp_path: Path):
    spec = FileSpec("missing/entry.bin", 1, 1, False, hashlib.sha256(b"x").hexdigest())
    results = audit_snapshot(tmp_path, [spec])
    assert results[0].error == "missing"
    assert not (tmp_path / "missing").exists()
