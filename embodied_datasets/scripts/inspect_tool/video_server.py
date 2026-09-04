"""Local HTTP static file server for one dataset's video files, with HTTP
Range support -- needed for <video> seeking, which Python's stdlib
http.server.SimpleHTTPRequestHandler does not provide (verified against the
installed Python 3.12: no Range handling anywhere in its source). Serving
without Range support would force browsers to download an entire (possibly
multi-episode, up to 200MB per lerobot's own video_files_size_in_mb default)
video file before any playback/seeking could start.

One server instance is rooted at one dataset directory (raw input or final
output) and serves any relative path under it via plain GET/HEAD, matching
the relative paths LeRobotDatasetMetadata.get_video_file_path() returns.
"""
from __future__ import annotations

import http.server
import mimetypes
import threading
from pathlib import Path
from typing import Tuple
from urllib.parse import unquote

mimetypes.add_type("video/mp4", ".mp4")

_CHUNK_SIZE = 65536


def _parse_range(range_header: str, file_size: int) -> Tuple[int, int]:
    """Parses a "bytes=start-end" Range header value (end optional -> EOF).
    A malformed value falls back to the full-file range rather than raising
    -- an unparseable Range header should degrade to "serve everything", not
    a 5xx."""
    try:
        _, _, spec = range_header.partition("=")
        start_str, _, end_str = spec.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
    except ValueError:
        return 0, file_size - 1
    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        return 0, file_size - 1
    return start, end


def _make_handler(root: Path):
    resolved_root = root.resolve()

    class RangeRequestHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._serve(send_body=True)

        def do_HEAD(self) -> None:
            self._serve(send_body=False)

        def _serve(self, send_body: bool) -> None:
            relative = unquote(self.path.lstrip("/"))
            file_path = (resolved_root / relative).resolve()
            if file_path != resolved_root and resolved_root not in file_path.parents:
                self.send_error(403, "Forbidden")
                return
            if not file_path.is_file():
                self.send_error(404, "Not Found")
                return

            file_size = file_path.stat().st_size
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            range_header = self.headers.get("Range")
            if range_header is None:
                start, end, status = 0, file_size - 1, 200
            else:
                start, end = _parse_range(range_header, file_size)
                status = 206 if (start, end) != (0, file_size - 1) else 200

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            if not send_body:
                return
            with file_path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(_CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def log_message(self, format_str: str, *args) -> None:
            pass  # keep test/dev stdout quiet; send_error's response body is still client-visible

    return RangeRequestHandler


def start_video_server(root: Path) -> int:
    """Starts a daemon-threaded HTTP server rooted at `root` on a
    system-assigned free port and returns that port. `root` does not need to
    exist yet -- requests made before it's created (or before a given file
    is written into it) just 404, which callers treat as "not ready yet",
    not an error."""
    handler_cls = _make_handler(root)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server.server_address[1]
