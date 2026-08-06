import http.client
import urllib.error
import urllib.request
from pathlib import Path

from video_server import start_video_server


def test_start_video_server_serves_full_file(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    file_path = tmp_path / "videos" / "clip.mp4"
    file_path.write_bytes(b"0123456789")

    port = start_video_server(tmp_path)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/videos/clip.mp4") as resp:
        assert resp.status == 200
        assert resp.read() == b"0123456789"
        assert resp.headers["Accept-Ranges"] == "bytes"
        assert resp.headers["Content-Type"] == "video/mp4"


def test_start_video_server_serves_partial_range(tmp_path: Path):
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"0123456789")
    port = start_video_server(tmp_path)

    request = urllib.request.Request(f"http://127.0.0.1:{port}/clip.mp4", headers={"Range": "bytes=2-5"})
    with urllib.request.urlopen(request) as resp:
        assert resp.status == 206
        assert resp.read() == b"2345"
        assert resp.headers["Content-Range"] == "bytes 2-5/10"


def test_start_video_server_serves_open_ended_range_to_eof(tmp_path: Path):
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"0123456789")
    port = start_video_server(tmp_path)

    request = urllib.request.Request(f"http://127.0.0.1:{port}/clip.mp4", headers={"Range": "bytes=7-"})
    with urllib.request.urlopen(request) as resp:
        assert resp.status == 206
        assert resp.read() == b"789"
        assert resp.headers["Content-Range"] == "bytes 7-9/10"


def test_start_video_server_returns_404_for_missing_file(tmp_path: Path):
    port = start_video_server(tmp_path)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/nope.mp4")
        raise AssertionError("expected HTTPError")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_start_video_server_returns_403_for_path_traversal(tmp_path: Path):
    inside = tmp_path / "inside"
    inside.mkdir()
    (inside / "secret.mp4").write_bytes(b"secret")
    port = start_video_server(inside)

    # Crafted directly over a raw socket (rather than via urlopen) so the
    # literal "/../secret.mp4" path reaches the server untouched -- some
    # HTTP client libraries normalize ".." segments before sending, which
    # would silently defeat this test.
    conn = http.client.HTTPConnection("127.0.0.1", port)
    try:
        conn.request("GET", "/../secret.mp4")
        resp = conn.getresponse()
        assert resp.status == 403
    finally:
        conn.close()
