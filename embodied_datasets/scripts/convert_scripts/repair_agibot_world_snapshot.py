#!/usr/bin/env python3
"""Repair and verify the pinned AgiBot World 2026 source snapshot.

The repair manifest is immutable and pins both the Hugging Face commit and the
expected object hashes.  Downloads stream into a same-directory partial file;
only a completely verified file is atomically installed.  No dataset-sized
local cache is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
from typing import BinaryIO, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_ROOT = Path("/mnt/data/embodied_datasets/public_datasets_raw/agibot_world")
DEFAULT_MANIFEST = Path(__file__).with_name("manifests") / "agibot_world_4423752.json"
DEFAULT_ENDPOINT = "https://huggingface.co"
CHUNK_BYTES = 8 * 1024 * 1024
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
UNSUPPORTED_FSYNC_ERRNOS = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}


class RepairError(RuntimeError):
    """Raised when the snapshot cannot be safely repaired or verified."""


@dataclass(frozen=True)
class FileSpec:
    path: str
    size: int
    baseline_size: int
    repair_required: bool
    lfs_sha256: str | None = None
    git_blob_sha1: str | None = None

    @property
    def expected_hash(self) -> str:
        return self.lfs_sha256 or self.git_blob_sha1 or ""

    @property
    def hash_kind(self) -> str:
        return "sha256" if self.lfs_sha256 is not None else "git_blob_sha1"


@dataclass(frozen=True)
class RepairManifest:
    repo_id: str
    revision: str
    files: tuple[FileSpec, ...]
    manifest_sha256: str

    @property
    def repair_files(self) -> tuple[FileSpec, ...]:
        return tuple(item for item in self.files if item.repair_required)


@dataclass(frozen=True)
class Verification:
    path: str
    ok: bool
    actual_size: int | None
    actual_hash: str | None
    error: str | None = None


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RepairError("manifest file path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise RepairError(f"unsafe manifest file path: {value!r}")
    return value


def load_manifest(path: Path) -> RepairManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepairError(f"cannot read repair manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RepairError(f"unsupported repair manifest: {path}")
    recorded_digest = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    actual_digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if recorded_digest != actual_digest:
        raise RepairError(f"repair manifest fingerprint mismatch: {path}")

    repo_id = payload.get("repo_id")
    revision = payload.get("revision")
    if not isinstance(repo_id, str) or not repo_id or not isinstance(revision, str) or not HEX_40.fullmatch(revision):
        raise RepairError("manifest repo_id/revision is invalid")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise RepairError("manifest files must be a non-empty list")

    files: list[FileSpec] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise RepairError("manifest file entry is not an object")
        relative = _safe_relative_path(raw.get("path"))
        if relative in seen:
            raise RepairError(f"duplicate manifest path: {relative}")
        seen.add(relative)
        size, baseline_size = raw.get("size"), raw.get("baseline_size")
        repair_required = raw.get("repair_required")
        sha256, blob_sha1 = raw.get("lfs_sha256"), raw.get("git_blob_sha1")
        if not isinstance(size, int) or size < 0 or not isinstance(baseline_size, int) or baseline_size < 0:
            raise RepairError(f"invalid size for {relative}")
        if not isinstance(repair_required, bool) or repair_required != (baseline_size != size):
            raise RepairError(f"invalid repair flag for {relative}")
        if (sha256 is None) == (blob_sha1 is None):
            raise RepairError(f"exactly one official hash is required for {relative}")
        if sha256 is not None and (not isinstance(sha256, str) or not HEX_64.fullmatch(sha256)):
            raise RepairError(f"invalid LFS SHA-256 for {relative}")
        if blob_sha1 is not None and (not isinstance(blob_sha1, str) or not HEX_40.fullmatch(blob_sha1)):
            raise RepairError(f"invalid Git blob SHA-1 for {relative}")
        files.append(FileSpec(relative, size, baseline_size, repair_required, sha256, blob_sha1))

    if [item.path for item in files] != sorted(item.path for item in files):
        raise RepairError("manifest files are not sorted")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise RepairError("manifest scope is missing")
    checks = {
        "file_count": len(files),
        "repair_file_count": sum(item.repair_required for item in files),
        "total_bytes": sum(item.size for item in files),
        "repair_bytes": sum(item.size for item in files if item.repair_required),
    }
    for key, expected in checks.items():
        if scope.get(key) != expected:
            raise RepairError(f"manifest scope {key} is inconsistent: expected {expected}")
    return RepairManifest(repo_id, revision, tuple(files), actual_digest)


def _new_hasher(spec: FileSpec):
    if spec.lfs_sha256 is not None:
        return hashlib.sha256()
    digest = hashlib.sha1()
    digest.update(f"blob {spec.size}\0".encode())
    return digest


def _hash_stream(stream: BinaryIO, digest, *, limit: int | None = None) -> int:
    total = 0
    while limit is None or total < limit:
        requested = CHUNK_BYTES if limit is None else min(CHUNK_BYTES, limit - total)
        chunk = stream.read(requested)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    return total


def _copy_hash_stream(stream: BinaryIO, output: BinaryIO, digest, *, limit: int) -> int:
    total = 0
    while total < limit:
        chunk = stream.read(min(CHUNK_BYTES, limit - total))
        if not chunk:
            break
        output.write(chunk)
        digest.update(chunk)
        total += len(chunk)
    return total


def verify_file(root: Path, spec: FileSpec) -> Verification:
    target = _target_path(root, spec, create_parent=False)
    try:
        stat = target.stat()
    except FileNotFoundError:
        return Verification(spec.path, False, None, None, "missing")
    except OSError as exc:
        return Verification(spec.path, False, None, None, f"stat failed: {exc}")
    if not target.is_file() or target.is_symlink():
        return Verification(spec.path, False, stat.st_size, None, "not a regular non-symlink file")
    if stat.st_size != spec.size:
        return Verification(spec.path, False, stat.st_size, None, f"size expected={spec.size} actual={stat.st_size}")
    digest = _new_hasher(spec)
    try:
        with target.open("rb") as stream:
            consumed = _hash_stream(stream, digest)
    except OSError as exc:
        return Verification(spec.path, False, stat.st_size, None, f"read failed: {exc}")
    actual_hash = digest.hexdigest()
    if consumed != spec.size or actual_hash != spec.expected_hash:
        return Verification(
            spec.path,
            False,
            consumed,
            actual_hash,
            f"{spec.hash_kind} expected={spec.expected_hash} actual={actual_hash}",
        )
    return Verification(spec.path, True, consumed, actual_hash)


def _target_path(root: Path, spec: FileSpec, *, create_parent: bool) -> Path:
    target = root.joinpath(*PurePosixPath(spec.path).parts)
    if target.is_symlink():
        raise RepairError(f"refusing symlink target: {target}")
    parent = target.parent
    if create_parent:
        parent.mkdir(parents=True, exist_ok=True)
    resolved_root, resolved_parent = root.resolve(), parent.resolve()
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise RepairError(f"target escapes snapshot root: {target}") from exc
    return target


def _download_url(endpoint: str, manifest: RepairManifest, spec: FileSpec) -> str:
    repo = quote(manifest.repo_id, safe="/")
    revision = quote(manifest.revision, safe="")
    relative = quote(spec.path, safe="/")
    return f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/{revision}/{relative}?download=true"


def _partial_path(target: Path, revision: str) -> Path:
    return target.with_name(f".{target.name}.{revision[:12]}.part")


def _fsync_compatible(descriptor: int, description: Path) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in UNSUPPORTED_FSYNC_ERRNOS:
            raise
        print(f"warning: fsync is unsupported for {description}; relying on close semantics", file=sys.stderr)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _fsync_compatible(descriptor, path)
    finally:
        os.close(descriptor)


def _check_space(parent: Path, required: int, reserve: int) -> None:
    free = shutil.disk_usage(parent).free
    if free < required + reserve:
        raise RepairError(
            f"insufficient target-filesystem space at {parent}: free={free}, "
            f"required_download={required}, reserve={reserve}"
        )


def _prepare_partial(partial: Path, spec: FileSpec):
    if partial.is_symlink():
        raise RepairError(f"refusing symlink partial file: {partial}")
    size = partial.stat().st_size if partial.exists() else 0
    if size > spec.size:
        partial.unlink()
        size = 0
    digest = _new_hasher(spec)
    if size:
        with partial.open("rb") as stream:
            consumed = _hash_stream(stream, digest)
        if consumed != size:
            raise RepairError(f"could not read complete partial file: {partial}")
    return size, digest


def _validate_range(response, offset: int, expected_size: int) -> bool:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    if offset == 0:
        if status not in {200, 206}:
            raise RepairError(f"unexpected HTTP status {status}")
        if status == 206:
            value = response.headers.get("Content-Range", "")
            match = CONTENT_RANGE.fullmatch(value)
            if match is None or int(match.group(1)) != 0 or int(match.group(3)) != expected_size:
                raise RepairError(f"invalid Content-Range: {value!r}")
        return False
    if status == 200:
        return True
    if status != 206:
        raise RepairError(f"server rejected resume at byte {offset}: HTTP {status}")
    value = response.headers.get("Content-Range", "")
    match = CONTENT_RANGE.fullmatch(value)
    if match is None or int(match.group(1)) != offset or int(match.group(3)) != expected_size:
        raise RepairError(f"invalid resumed Content-Range: {value!r}")
    return False


def download_and_install(
    root: Path,
    manifest: RepairManifest,
    spec: FileSpec,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
    timeout: float = 60.0,
    retries: int = 4,
    retry_delay: float = 2.0,
    min_free_bytes: int = 0,
) -> str:
    current = verify_file(root, spec)
    if current.ok:
        return "already_verified"
    target = _target_path(root, spec, create_parent=True)
    partial = _partial_path(target, manifest.revision)
    url = _download_url(endpoint, manifest, spec)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            offset, digest = _prepare_partial(partial, spec)
            if offset == spec.size:
                if digest.hexdigest() != spec.expected_hash:
                    partial.unlink()
                    raise RepairError(f"completed partial has wrong {spec.hash_kind}: {spec.path}")
            else:
                _check_space(target.parent, spec.size - offset, min_free_bytes)
                headers = {"Accept-Encoding": "identity", "User-Agent": "agibot-world-snapshot-repair/1"}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = Request(url, headers=headers)
                with urlopen(request, timeout=timeout) as response:
                    restart = _validate_range(response, offset, spec.size)
                    mode = "wb" if restart else "ab"
                    if restart:
                        offset, digest = 0, _new_hasher(spec)
                    with partial.open(mode) as output:
                        written = _copy_hash_stream(response, output, digest, limit=spec.size - offset)
                        trailing = response.read(1)
                        output.flush()
                        _fsync_compatible(output.fileno(), partial)
                    if trailing:
                        raise RepairError(f"download exceeds official size for {spec.path}")
                    if offset + written != spec.size:
                        raise RepairError(
                            f"short download for {spec.path}: expected={spec.size} actual={offset + written}"
                        )
                if digest.hexdigest() != spec.expected_hash:
                    partial.unlink(missing_ok=True)
                    raise RepairError(f"downloaded {spec.hash_kind} mismatch for {spec.path}")
            os.replace(partial, target)
            _fsync_directory(target.parent)
            installed = verify_file(root, spec)
            if not installed.ok:
                raise RepairError(f"post-install verification failed for {spec.path}: {installed.error}")
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError, RepairError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(retry_delay * 2 ** (attempt - 1), 30.0))
    raise RepairError(f"download failed after {retries} attempts for {spec.path}: {last_error}")


def audit_snapshot(root: Path, files: Sequence[FileSpec]) -> list[Verification]:
    results: list[Verification] = []
    for index, spec in enumerate(files, 1):
        result = verify_file(root, spec)
        results.append(result)
        state = "ok" if result.ok else "FAILED"
        print(f"[audit {index}/{len(files)}] {state} {spec.path}", flush=True)
    return results


@contextmanager
def exclusive_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".agibot-world-repair.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise RepairError(f"cannot create repair lock {lock_path}: {exc}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RepairError(f"another repair process holds {lock_path}") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _size_plan(root: Path, manifest: RepairManifest) -> dict[str, object]:
    mismatches = []
    for spec in manifest.files:
        target = root.joinpath(*PurePosixPath(spec.path).parts)
        try:
            actual = target.stat().st_size
        except FileNotFoundError:
            actual = None
        if actual != spec.size:
            mismatches.append({"path": spec.path, "actual_size": actual, "expected_size": spec.size})
    return {
        "repo_id": manifest.repo_id,
        "revision": manifest.revision,
        "scope_files": len(manifest.files),
        "manifest_repair_files": len(manifest.repair_files),
        "current_size_mismatches": mismatches,
        "maximum_partial_file_bytes": max((item.size for item in manifest.repair_files), default=0),
    }


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        _fsync_compatible(stream.fileno(), temporary)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="writable snapshot root")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Hugging Face endpoint or compatible mirror")
    parser.add_argument("--token-env", default="HF_TOKEN", help="environment variable containing an optional token")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--min-free-bytes", type=int, default=0, help="free-space reserve on the target filesystem")
    parser.add_argument("--report", type=Path, help="write a JSON result report")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="compare sizes without downloading or hashing")
    mode.add_argument("--audit-only", action="store_true", help="hash-audit all scoped files without repair")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.retries <= 0 or args.min_free_bytes < 0:
        parser.error("timeout/retries must be positive and min-free-bytes must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        root = args.root.resolve()
        if not root.is_dir():
            raise RepairError(f"snapshot root is not a directory: {root}")
        if args.plan_only:
            plan = _size_plan(root, manifest)
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        repair_results: list[dict[str, str]] = []
        if args.audit_only:
            audit = audit_snapshot(root, manifest.files)
        else:
            token = os.environ.get(args.token_env)
            with exclusive_lock(root):
                for index, spec in enumerate(manifest.repair_files, 1):
                    print(f"[repair {index}/{len(manifest.repair_files)}] {spec.path}", flush=True)
                    status = download_and_install(
                        root,
                        manifest,
                        spec,
                        endpoint=args.endpoint,
                        token=token,
                        timeout=args.timeout,
                        retries=args.retries,
                        min_free_bytes=args.min_free_bytes,
                    )
                    repair_results.append({"path": spec.path, "status": status})
                audit = audit_snapshot(root, manifest.files)
        failed = [item for item in audit if not item.ok]
        report: dict[str, object] = {
            "repo_id": manifest.repo_id,
            "revision": manifest.revision,
            "manifest_sha256": manifest.manifest_sha256,
            "repair": repair_results,
            "audit": {
                "checked": len(audit),
                "passed": len(audit) - len(failed),
                "failed": [item.__dict__ for item in failed],
            },
        }
        if args.report:
            _write_report(args.report, report)
        print(json.dumps(report["audit"], ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if failed else 0
    except (OSError, RepairError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
