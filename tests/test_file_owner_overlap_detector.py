"""Validate the hosted oracle without claiming a local hosted stress run."""

from __future__ import annotations

import copy
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

detect = cast(
    Callable[[list[dict[str, Any]], int, dict[str, int]], list[dict[str, Any]]],
    runpy.run_path(str(Path(__file__).parents[1] / "tools/file-owner-native/overlap-stress.py"))[
        "intervals"
    ],
)


def valid_events() -> list[dict[str, Any]]:
    return [
        {"role": role, "iteration": 0, "pid": pid, "event": event, "at_ns": at}
        for role, pid, event, at in [
            ("holder", 1, "enter", 1),
            ("holder", 1, "exit", 2),
            ("free", 2, "enter", 3),
            ("free", 2, "exit", 4),
        ]
    ]


def test_detector_accepts_complete_separate_intervals() -> None:
    assert len(detect(valid_events(), 1, {"holder": 1, "free": 2})) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event", "unknown"),
        ("role", "contended"),
        ("pid", 2),
        ("iteration", True),
        ("iteration", 1),
        ("at_ns", -1),
        ("at_ns", 1.5),
        ("extra", 1),
    ],
)
def test_detector_rejects_invalid_events(field: str, value: Any) -> None:
    events = valid_events()
    events[0][field] = value
    with pytest.raises(AssertionError):
        detect(events, 1, {"holder": 1, "free": 2})


def test_detector_rejects_real_overlap() -> None:
    events = valid_events()
    events[2]["at_ns"] = 1
    with pytest.raises(AssertionError, match="OVERLAP"):
        detect(events, 1, {"holder": 1, "free": 2})


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reordered", "reversed-time"])
def test_detector_requires_complete_once_only_role_iteration_pairs(kind: str) -> None:
    events = valid_events()
    if kind == "missing":
        events.pop()
    elif kind == "duplicate":
        events.append(copy.deepcopy(events[0]))
    elif kind == "reordered":
        events[0], events[1] = events[1], events[0]
    else:
        events[1]["at_ns"] = 0
    with pytest.raises(AssertionError):
        detect(events, 1, {"holder": 1, "free": 2})
