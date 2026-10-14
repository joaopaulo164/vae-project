"""Small Markdown formatting helpers shared by the report writers."""

from __future__ import annotations

import math


def format_value(value: object, digits: int = 6) -> str:
    """Format a possibly missing value for Markdown.

    Args:
        value: Any value; ``None`` and NaN are rendered as ``n/a``.
        digits: Decimal places for floats.

    Returns:
        Text representation, never an invented number.
    """
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def bullets(items: list[str], empty: str = "none") -> list[str]:
    """Render strings as Markdown bullets with code formatting.

    Args:
        items: Values to list (typically relative file paths).
        empty: Text of the single bullet used when ``items`` is empty.

    Returns:
        Markdown lines.
    """
    return [f"- `{item}`" for item in items] if items else [f"- {empty}"]
