"""Shared error type for every convert_scripts reader/writer."""
from __future__ import annotations


class ConversionError(ValueError):
    """Raised when raw data cannot be converted without guessing."""
