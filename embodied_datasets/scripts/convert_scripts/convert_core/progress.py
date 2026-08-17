"""Small rate-limited progress reporter shared by long-running converters."""
from __future__ import annotations

import math
import sys
import time
from typing import Any


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


class EtaProgress:
    """Newline progress suitable for terminals as well as ``nohup`` logs."""

    def __init__(
        self,
        label: str,
        total: int,
        unit: str,
        *,
        interval_seconds: float = 10.0,
        clock: Any = time.monotonic,
        stream: Any = None,
    ) -> None:
        if total <= 0:
            raise ValueError(f"progress total must be positive, got {total}")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("ETA interval must be finite and positive")
        self.label = label
        self.total = total
        self.unit = unit
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.stream = stream if stream is not None else sys.stderr
        self.started_at = float(clock())
        self.last_emitted_at = self.started_at
        self.completed = 0

    def update(self, completed: int, *, context: str | None = None, force: bool = False) -> None:
        if completed < self.completed:
            raise ValueError("progress cannot move backwards")
        self.completed = min(completed, self.total)
        now = float(self.clock())
        if not force and now - self.last_emitted_at < self.interval_seconds:
            return
        elapsed = max(0.0, now - self.started_at)
        rate = self.completed / elapsed if elapsed > 0 and self.completed else 0.0
        eta = (self.total - self.completed) / rate if rate else math.inf
        eta_text = format_duration(eta) if math.isfinite(eta) else "--:--:--"
        suffix = f" | {context}" if context else ""
        print(
            f"[{self.label}] {self.completed}/{self.total} ({100*self.completed/self.total:.1f}%)"
            f" | {rate:.2f} {self.unit}/s | elapsed {format_duration(elapsed)} | ETA {eta_text}{suffix}",
            file=self.stream,
            flush=True,
        )
        self.last_emitted_at = now

    def finish(self, *, context: str | None = None) -> None:
        self.update(self.total, context=context, force=True)
