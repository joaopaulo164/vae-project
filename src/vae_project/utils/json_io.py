"""JSON helpers that serialize the value types used across the project.

Paths, enums, numpy scalars/arrays and dataclasses are converted to plain
JSON types. Non-finite floats become ``null`` so that "unavailable" is never
written as a number.
"""

from __future__ import annotations

import dataclasses
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert ``value`` into a structure that ``json.dump`` accepts.

    Args:
        value: Any value built from dicts, lists, dataclasses, enums, paths,
            numpy scalars/arrays and plain Python types.

    Returns:
        A JSON-compatible value. NaN and infinity become ``None``.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> Path:
    """Write ``data`` as UTF-8 JSON, creating parent directories.

    Args:
        path: Destination file.
        data: Value to serialize with :func:`to_jsonable`.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2)
    return path


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded JSON value.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
