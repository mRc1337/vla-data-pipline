"""Tiny path-sorting utility shared by every reader (episode files/dirs sort
numerically by embedded integers, not lexicographically -- "episode_10"
after "episode_2", not before)."""
from __future__ import annotations

from pathlib import Path
import re


def natural_sort_key(path: Path) -> list[tuple[int, str | int]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.as_posix())
    ]
