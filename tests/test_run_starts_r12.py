"""G-387 round 12: the round-11 review findings, one rule each.

2. ONE ended mark in all three languages: the row a runner start's task row
   closes the calls with carries ``harness_ended_by_start: {approval_ids:
   [...]}``, the PHP reference shape, last on the row. Only that shape is
   read; the round-11 list never shipped and is not read.
4. Holds are released BEFORE ``RunFailed`` is emitted, so a listener never
   observes a hold that is about to go (I5); and an exception that skips the
   normal handlers (a ``BaseException``) raised in a ``RunStarted`` listener
   never leaves a ROOT run ``running``: it is failed (class name only), its
   holds released, ``RunFailed`` emitted, and the exception propagates (I2).
5. The start's own-run view is entered only by the Session object that
   committed that start: a host calling it on another object, with a run id
   read from storage, confirms nothing.
7. A listener's error is reported by its class name only, by a conversion
   that cannot raise; every run event is emitted inside the run's try, so a
   throw from it fails the run, releases its holds and emits the terminal
   event.
"""

from __future__ import annotations

import json
import socket
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_run_starts import direct, tree_records
from test_run_starts_r9 import outage, refused_as_unbound, start
from test_run_starts_r10 import Abort
from test_step_allowance import World
from test_step_allowance_r6 import dead_approval_child

from prism_harness import (
    HarnessEvent,
    HarnessEvents,
    RunBudget,
    RunContext,
    RunFailed,
    RunFinished,
    RunStarted,
    record_approval,
)
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.thread_rows import ENDED_BY_START, ended_approval_ids


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def rows_of(world: World) -> list[dict[str, Any]]:
    return [entry.message for entry in world.fresh().child("worker").thread().messages()]


def on_disk(world: World, ending: str) -> Any:
    """The stored payload whose key ends with ``ending``, read straight off
    the disk (never through a Session, so nothing is cached or replaced)."""
    for path in sorted(Path(world.directory).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if str(record["key"]).endswith(ending):
            return record["payload"]
    return None


def child_holds(world: World) -> dict[str, Any]:
    record = on_disk(world, "::sub::worker:child")
    steps = record.get("child_steps") if isinstance(record, dict) else None
    return dict(steps.get("holds") or {}) if isinstance(steps, dict) else {}


def tree_holds(world: World) -> list[dict[str, Any]]:
    return [json.loads(text)["payload"]["holds"] for text in tree_records(world).values()]


def listen(world: World, listener: Any) -> list[HarnessEvent]:
    events = HarnessEvents()
    seen: list[HarnessEvent] = []
    events.listen(seen.append)
    events.listen(listener)
    world.runtime._events = events
    return seen


def root_run(world: World) -> dict[str, Any] | None:
    return world.fresh().root().run()


# -- 2. one ended-mark shape: the reference's ------------------------------------


def test_a_start_writes_the_reference_ended_mark_last_on_its_row() -> None:
    world = World()
    approval = dead_approval_child(world)

    assert "Child answer." in start(world, "direct", "New task.")

    rows = rows_of(world)
    (marked,) = [row for row in rows if ENDED_BY_START in row]
    assert marked[ENDED_BY_START] == {"approval_ids": [approval]}
    # The step's answers-step mark (G-387 round 15) comes after every other
    # harness mark, so the ended mark is last but one.
    assert list(marked) == [
        "type",
        "tool_results",
        "tool_approval_responses",
        ENDED_BY_START,
        "harness_answers_step",
    ]
    assert approval in ended_approval_ids(rows)
    assert approval not in approval_decisions(rows)


def test_the_reference_shape_ends_its_approval() -> None:
    """A thread another language's harness wrote reads the same here."""
    world = World()
    world.run_child_in_runner("worker")
    (approval,) = world.pending(world.child("worker"))
    record_approval(world.fresh().child("worker"), approval, True)
    rows = rows_of(world)
    closing = {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [],
        ENDED_BY_START: {"approval_ids": [approval]},
    }
    found = [*rows, closing]
    assert approval in ended_approval_ids(found)
    assert approval not in approval_decisions(found)


@pytest.mark.parametrize(
    "kind",
    ["round_11_list", "ids_text", "ids_not_text", "no_ids", "empty_ids", "text", "nested"],
)
def test_no_other_shape_ends_anything(kind: str) -> None:
    """The round-11 list never shipped and is not read; nothing that is not
    the reference shape naming the approval ends it."""
    world = World()
    world.run_child_in_runner("worker")
    (approval,) = world.pending(world.child("worker"))
    record_approval(world.fresh().child("worker"), approval, True)
    mark: object = {
        "round_11_list": [approval],
        "ids_text": {"approval_ids": approval},
        "ids_not_text": {"approval_ids": [1, None, {"id": approval}]},
        "no_ids": {"approvals": [approval]},
        "empty_ids": {"approval_ids": []},
        "text": approval,
        "nested": {"harness_ended_by_start": {"approval_ids": [approval]}},
    }[kind]
    found = [
        *rows_of(world),
        {
            "type": "tool_result",
            "tool_results": [],
            "tool_approval_responses": [],
            ENDED_BY_START: mark,
        },
    ]

    assert approval not in ended_approval_ids(found)
    assert approval_decisions(found).get(approval) is True


# -- 4 (I5). holds are released before RunFailed --------------------------------


def test_a_failed_child_turn_releases_its_hold_before_run_failed() -> None:
    world = World()
    world.run_child_in_runner("worker")
    (approval,) = world.pending(world.child("worker"))
    record_approval(world.fresh().child("worker"), approval, True)
    world.fresh()
    at_failure: list[dict[str, Any]] = []
    listen(
        world,
        lambda event: (
            at_failure.append(child_holds(world)) if isinstance(event, RunFailed) else None
        ),
    )

    def down() -> None:
        raise OSError("provider down")

    world.on_request = down
    with pytest.raises(OSError):
        world.runtime.send(world.child("worker"), "")

    assert at_failure == [{}]
    assert child_holds(world) == {}


def test_a_failed_root_releases_its_tree_hold_before_run_failed() -> None:
    world = World()
    at_failure: list[list[dict[str, Any]]] = []

    def listener(event: HarnessEvent) -> None:
        if isinstance(event, RunFinished) and event.session_key == world.child("worker").key():

            def down() -> None:
                raise OSError("provider down")

            world.on_request = down
        if isinstance(event, RunFailed):
            at_failure.append(tree_holds(world))

    listen(world, listener)
    world.calls(("worker", {"task": "Look."}))
    world.text("Child answer.")
    with pytest.raises(OSError):
        world.runtime.send(world.root(), "Go")

    assert at_failure == [[{}]]


# -- 4 (I2) and 7. a RunStarted listener never leaves a run running --------------


def _raise_on_started(error: BaseException, child_only: bool = False) -> Any:
    def listener(event: HarnessEvent) -> None:
        if isinstance(event, RunStarted) and (not child_only or "::sub::" in event.session_key):
            raise error

    return listener


def test_a_base_exception_in_a_run_started_listener_fails_the_root() -> None:
    world = World()
    seen = listen(world, _raise_on_started(Abort()))
    world.text("Done.")

    with pytest.raises(Abort):
        world.runtime.send(world.root(), "Go")

    run = root_run(world)
    assert isinstance(run, dict) and run["status"] == "failed"
    assert str(run.get("failure", "")).startswith("Abort")
    failed = [event for event in seen if isinstance(event, RunFailed)]
    assert [event.run_id for event in failed] == [run["id"]]
    assert failed[0].failure.startswith("Abort")
    assert world.runtime._trees.members == {}
    # The next turn runs normally.
    world.runtime._events = None
    world.queue.clear()
    world.text("Again.")
    assert world.runtime.send(world.root(), "Again").text == "Again."


def test_a_base_exception_in_a_run_started_listener_undoes_a_start() -> None:
    world = World()
    approval = dead_approval_child(world)
    world.fresh()
    seen = listen(world, _raise_on_started(Abort(), child_only=True))

    with pytest.raises(Abort):
        direct(world, world.root(), RunContext.root("host", RunBudget(8)), "New task.")

    started = [e for e in seen if isinstance(e, RunStarted) and "::sub::" in e.session_key]
    failed = [e for e in seen if isinstance(e, RunFailed)]
    assert [e.run_id for e in failed] == [e.run_id for e in started]
    assert child_holds(world) == {}
    refused_as_unbound(lambda: record_approval(world.fresh().child("worker"), approval, True))
    world.runtime._events = None
    assert "Child answer." in start(world, "direct", "Again.")


class Unprintable(Exception):
    """A host error whose text cannot be read: ``str()`` raises."""

    def __str__(self) -> str:
        raise RuntimeError("LISTENER-SECRET from __str__")


@pytest.mark.parametrize("kind", ["unprintable", "secret"])
def test_a_run_started_listener_error_never_breaks_the_root(kind: str) -> None:
    world = World()
    error = Unprintable() if kind == "unprintable" else ValueError("LISTENER-SECRET")
    listen(world, _raise_on_started(error))
    world.text("Done.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = world.runtime.send(world.root(), "Go")

    assert response.text == "Done."
    run = root_run(world)
    assert isinstance(run, dict) and run["status"] == "completed"
    said = " | ".join(str(warning.message) for warning in caught)
    assert "run.started" in said
    assert "LISTENER-SECRET" not in said
    assert type(error).__name__ in said


@pytest.mark.parametrize("kind", ["unprintable", "secret"])
def test_a_run_started_listener_error_never_breaks_a_start(kind: str) -> None:
    """S8 in Python: the start completes, its run ends, and the next
    delegation is not refused as busy."""
    world = World()
    error = Unprintable() if kind == "unprintable" else ValueError("LISTENER-SECRET")
    world.fresh()
    listen(world, _raise_on_started(error, child_only=True))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        world.text("Child answer.")
        result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Look.")

    assert result.outcome == "completed", result.text
    assert "LISTENER-SECRET" not in " | ".join(str(w.message) for w in caught)
    assert child_holds(world) == {}
    run = world.fresh().child("worker").run()
    assert isinstance(run, dict) and run["status"] == "completed"
    world.runtime._events = None
    assert "Child answer." in start(world, "direct", "Again.")


@pytest.mark.parametrize("where", ["root", "start"])
def test_an_emit_that_raises_fails_the_run_it_started(
    where: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every run event is emitted inside the run's try: a RunStarted that
    raises fails that run (class name only), releases its holds and emits
    the terminal event."""
    world = World()
    seen: list[HarnessEvent] = []
    events = HarnessEvents()
    world.runtime._events = events
    original = HarnessEvents.emit

    def emit(self: HarnessEvents, event: HarnessEvent) -> None:
        seen.append(event)
        if isinstance(event, RunStarted) and (where == "root" or "::sub::" in event.session_key):
            raise OSError("SECRET emit failure")
        original(self, event)

    monkeypatch.setattr(HarnessEvents, "emit", emit)
    if where == "root":
        world.text("Done.")
        with pytest.raises(OSError):
            world.runtime.send(world.root(), "Go")
        run = root_run(world)
    else:
        world.text("Child answer.")
        result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Look.")
        assert result.outcome == "failed"
        assert "SECRET" not in result.text
        run = None
    started = [event for event in seen if isinstance(event, RunStarted)]
    failed = [event for event in seen if isinstance(event, RunFailed)]
    target = started[0] if where == "root" else started[-1]
    assert [event.run_id for event in failed] == [target.run_id]
    assert failed[0].failure.startswith("OSError")
    assert "SECRET" not in failed[0].failure
    if run is not None:
        assert run["status"] == "failed" and "SECRET" not in str(run)
    assert child_holds(world) == {}
    assert world.runtime._trees.members == {}


# -- 5. the own-run view needs the object that committed the start ---------------


def test_a_forged_own_run_view_on_a_crash_residue_confirms_nothing() -> None:
    """A start failed after (b) in a store outage: its undo and its fail
    both failed, so a provisional run is left ``running``. Host code calling
    the view on another Session object, with that run id read from storage,
    reads nothing confirmed."""
    world = World()
    approval = dead_approval_child(world)
    with pytest.MonkeyPatch.context() as patch:
        outage(patch)
        start(world, "direct", answer=False)
    child = world.fresh().child("worker")
    raw = on_disk(world, "::sub::worker:child")
    assert isinstance(raw, dict)
    residue = raw["provisional"]
    assert child.child_provisional()

    leave = child.enter_own_start(residue)
    try:
        child._cached_state = None
        assert child.child_binding() is None
        assert child.child_ceiling() is None
        assert child.child_provisional()
    finally:
        leave()
    refused_as_unbound(lambda: record_approval(world.fresh().child("worker"), approval, True))
