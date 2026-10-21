"""Unit tests for the JSON helpers and the shared reconstruction indices."""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from vae_project.experiments import factorial_protocol
from vae_project.experiments.protocol_models import ProtocolError
from vae_project.experiments.protocol_shared import choose_reconstruction_indices
from vae_project.utils.json_io import read_json, to_jsonable, write_json

pytestmark = pytest.mark.unit


class Color(Enum):
    RED = "red"


@dataclasses.dataclass
class Point:
    x: float
    label: Color


def test_to_jsonable_converts_project_value_types() -> None:
    payload = {
        "path": Path("a") / "b.txt",
        "array": np.array([[1, 2], [3, 4]]),
        "scalar": np.float32(0.5),
        "tuple": (1, 2),
        "point": Point(x=1.5, label=Color.RED),
    }

    converted = to_jsonable(payload)

    assert converted["path"] == "a/b.txt"
    assert converted["array"] == [[1, 2], [3, 4]]
    assert converted["scalar"] == 0.5
    assert converted["tuple"] == [1, 2]
    assert converted["point"] == {"x": 1.5, "label": "red"}


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_become_null_never_zero(value: float) -> None:
    assert to_jsonable({"loss": value}) == {"loss": None}


def test_write_and_read_json_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "data.json"

    written = write_json(target, {"value": np.int64(3), "missing": math.nan})

    assert written == target
    assert read_json(target) == {"value": 3, "missing": None}


def test_reconstruction_indices_are_fixed_sorted_and_valid_for_every_split() -> None:
    first = choose_reconstruction_indices([100, 40], count=8, seed=42)
    second = choose_reconstruction_indices([100, 40], count=8, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(set(first)) == 8
    assert max(first) < 40


def test_reconstruction_indices_never_exceed_the_smallest_split() -> None:
    chosen = choose_reconstruction_indices([5, 20], count=8, seed=1)

    assert len(chosen) == 5
    assert sorted(chosen) == [0, 1, 2, 3, 4]


def test_reconstruction_indices_depend_on_the_seed() -> None:
    assert choose_reconstruction_indices([1000], 8, seed=1) != choose_reconstruction_indices(
        [1000], 8, seed=2
    )


class TestRenameWithRetries:
    """The rename helper tolerates transient Windows locks."""

    def test_retries_then_succeeds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        real_rename = Path.rename
        calls = {"count": 0}

        def flaky_rename(self: Path, destination: Path) -> Path:
            calls["count"] += 1
            if calls["count"] < 3:
                raise PermissionError("locked")
            return real_rename(self, destination)

        monkeypatch.setattr(Path, "rename", flaky_rename)
        monkeypatch.setattr(factorial_protocol.time, "sleep", lambda _seconds: None)

        factorial_protocol._rename_with_retries(source, target)

        assert target.is_dir()
        assert calls["count"] == 3

    def test_raises_protocol_error_when_always_locked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        def locked_rename(self: Path, destination: Path) -> Path:
            raise PermissionError("locked")

        monkeypatch.setattr(Path, "rename", locked_rename)
        monkeypatch.setattr(factorial_protocol.time, "sleep", lambda _seconds: None)

        with pytest.raises(ProtocolError, match="locked"):
            factorial_protocol._rename_with_retries(source, tmp_path / "target")
