"""Shared helper for scripts that regenerate one marked section of
embodied_datasets/README.md in place, leaving everything else untouched.
"""
from __future__ import annotations


def replace_marked_section(
    text: str, start_marker: str, end_marker: str, content: str
) -> str:
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    if end < start:
        raise ValueError(f"{end_marker!r} appears before {start_marker!r}")
    return text[:start] + "\n\n" + content + "\n\n" + text[end:]
