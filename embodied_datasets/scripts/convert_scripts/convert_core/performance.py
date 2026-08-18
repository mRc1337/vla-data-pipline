"""Low-overhead process-tree and temporary-space sampling for benchmarks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import threading
import time
from typing import Sequence


@dataclass(frozen=True)
class PerformanceMetrics:
    wall_seconds: float
    cpu_seconds: float
    average_cpu_cores: float
    peak_rss_bytes: int
    read_bytes: int
    write_bytes: int
    read_chars: int
    write_chars: int
    peak_temp_bytes: int
    io_counters_available: bool

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _procfs_self_pid() -> int:
    """Return the PID used by the mounted procfs, even in a PID namespace."""
    stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    return int(stat.split("(", 1)[0].strip())


def _children(pid: int) -> list[int]:
    try:
        value = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
    except OSError:
        return []
    return [int(item) for item in value.split()]


def _process_tree(root_pid: int) -> set[int]:
    found: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(_children(pid))
    return found


def _proc_counters(pid: int) -> tuple[int, int, dict[str, int]] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # Everything following the final ')' starts at procfs field 3.
        fields = stat[stat.rfind(")") + 2 :].split()
        cpu_ticks = int(fields[11]) + int(fields[12])
        rss_pages = int(fields[21])
    except (OSError, ValueError, IndexError):
        return None
    io_values: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            io_values[key] = int(value.strip())
    except (OSError, ValueError, IndexError):
        # Some containers expose stat/RSS but deny /proc/<pid>/io. Preserve
        # the usable CPU and memory evidence and report I/O availability.
        io_values = {}
    return cpu_ticks, rss_pages, io_values


def _tree_size(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


class ProcessTreeSampler:
    """Sample a coordinator and all descendants until :meth:`stop` is called."""

    def __init__(
        self,
        temp_root: Path | Sequence[Path],
        *,
        interval_seconds: float = 0.2,
    ):
        self.root_pid = _procfs_self_pid()
        self.temp_roots = (
            (temp_root,) if isinstance(temp_root, Path) else tuple(temp_root)
        )
        if not self.temp_roots:
            raise ValueError("at least one temporary-space root is required")
        self.interval_seconds = interval_seconds
        self._clock_ticks = os.sysconf("SC_CLK_TCK")
        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._baseline: dict[int, tuple[int, dict[str, int]]] = {}
        self._latest: dict[int, tuple[int, dict[str, int]]] = {}
        self._peak_rss = 0
        self._peak_temp = 0
        self._io_counters_available = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        for pid in _process_tree(self.root_pid):
            counters = _proc_counters(pid)
            if counters is not None:
                cpu, _rss, io_values = counters
                self._baseline[pid] = (cpu, io_values)
                self._io_counters_available |= bool(io_values)
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        rss = 0
        for pid in _process_tree(self.root_pid):
            counters = _proc_counters(pid)
            if counters is None:
                continue
            cpu, rss_pages, io_values = counters
            self._latest[pid] = (cpu, io_values)
            self._io_counters_available |= bool(io_values)
            rss += rss_pages * self._page_size
        self._peak_rss = max(self._peak_rss, rss)
        self._peak_temp = max(
            self._peak_temp, sum(_tree_size(root) for root in self.temp_roots)
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def stop(self) -> PerformanceMetrics:
        if self._thread is None:
            raise RuntimeError("sampler was not started")
        self._sample()
        self._stop_event.set()
        self._thread.join()
        wall = max(0.0, time.monotonic() - self._started_at)
        cpu_ticks = 0
        io_totals = {"read_bytes": 0, "write_bytes": 0, "rchar": 0, "wchar": 0}
        for pid, (cpu, io_values) in self._latest.items():
            baseline_cpu, baseline_io = self._baseline.get(pid, (0, {}))
            cpu_ticks += max(0, cpu - baseline_cpu)
            for key in io_totals:
                io_totals[key] += max(0, io_values.get(key, 0) - baseline_io.get(key, 0))
        cpu_seconds = cpu_ticks / self._clock_ticks
        return PerformanceMetrics(
            wall_seconds=wall,
            cpu_seconds=cpu_seconds,
            average_cpu_cores=cpu_seconds / wall if wall else 0.0,
            peak_rss_bytes=self._peak_rss,
            read_bytes=io_totals["read_bytes"],
            write_bytes=io_totals["write_bytes"],
            read_chars=io_totals["rchar"],
            write_chars=io_totals["wchar"],
            peak_temp_bytes=self._peak_temp,
            io_counters_available=self._io_counters_available,
        )
