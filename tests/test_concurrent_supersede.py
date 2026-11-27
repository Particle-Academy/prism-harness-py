"""G-404 (G-387 round 14): a SUPERSEDE never races the resolution of a step.

A row that supersedes a parked step (DECISION B: a new message, or a host's
user or assistant row) denies each of its calls that has no result, "Tool
call was not run because a later step replaced it". The resolver runs an
approved call and records its result under the session lock. Unserialised,
a superseding write landing WHILE the approved call ran gave that call two
outcomes, the false "not run" first.

The rule: a superseding write and the resolver are serialised. The resolver
claims the step's calls in the thread itself before it runs any of them; a
superseding row recorded while the claim stands is refused
(``session_locked``) with nothing written, whoever records it and through
whatever object, and a new message waits on the session lock the resolver
holds. The replaced denial and the superseding row are ONE write, and every
pending reader honours the superseded mark. Every gated call ends with
exactly one outcome, and a decision on a replaced step is refused in every
window.

Deterministic: every interleaving is forced by a hook inside the approved
tool or around a store write, never by timing. Each wait has a generous
bound only so a regression fails instead of hanging.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from test_repeated_call_ids import Recording, park, rows

from prism_harness import AgentRuntime, HarnessError, Session, record_approval
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.stores.file import FileSessionStore
from prism_harness.thread_rows import (
    SUPERSEDED,
    SUPERSEDED_RESULT,
    pending_approval_ids,
    superseded_approval_ids,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


#: A bound on every wait, so a regression fails rather than hangs. Never a
#: sleep: each wait ends on an event the code under test sets.
BOUND = 60.0


class Hooked(Recording):
    """The Recording world whose ``echo`` runs ``during`` (once) while it runs."""

    def __init__(self) -> None:
        self.during: Callable[[], None] | None = None
        super().__init__()

    def _runtime(self) -> AgentRuntime:
        runtime = super()._runtime()
        world = self
        recorded = runtime._tools

        class Echo:
            name = "echo"

            def handle(self, args: dict[str, Any]) -> Any:
                world.ran_with.append(("echo", dict(args)))
                hook, world.during = world.during, None
                if hook is not None:
                    hook()
                return f"echo:{args.get('n')}"

        recorded.register(Echo())
        return runtime


def other_worker(world: Recording) -> Recording:
    """A second worker over the same disk: its own harness and runtime."""
    other = Recording()
    other.directory = world.directory
    other.fresh()
    return other


def outcomes(session: Session, n: int) -> list[str]:
    """Every stored result for the call with argument ``n``, in order."""
    return [
        entry["result"]
        for row in rows(session)
        if row.get("type") == "tool_result"
        for entry in row["tool_results"]
        if entry["args"].get("n") == n
    ]


def approved_and_parked(world: Hooked) -> str:
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    return approvals[0]


def in_thread(
    work: Callable[[], Any], progress: threading.Event | None = None
) -> tuple[threading.Thread, dict[str, Any]]:
    found: dict[str, Any] = {}

    def run() -> None:
        try:
            found["value"] = work()
        except BaseException as error:  # noqa: BLE001 - reported to the test
            found["error"] = error
        finally:
            found["done"] = True
            if progress is not None:
                progress.set()

    worker = threading.Thread(target=run, daemon=True)
    return worker, found


# -- PR2: a new message while the approved call runs ------------------------------


def test_pr2_a_new_message_while_the_approved_call_runs_waits_for_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = Hooked()
    approval = approved_and_parked(world)
    other = other_worker(world)
    other.text("B answer.")
    reached = threading.Event()
    sender, found = in_thread(lambda: other.runtime.send(other.root(), "A new message."), reached)
    original_lock = Session.lock

    def lock(self: Session, *args: Any, **kwargs: Any) -> Any:
        # B reaching the session lock A holds: it waits there.
        if threading.current_thread() is sender:
            reached.set()
        return original_lock(self, *args, **kwargs)

    monkeypatch.setattr(Session, "lock", lock)

    def during() -> None:
        sender.start()
        # A keeps the call running until B is waiting on the lock, or (the
        # defect) B has finished its whole turn without waiting.
        assert reached.wait(BOUND)

    world.during = during
    world.text("A answer.")
    world.fresh().runtime.send(world.root(), "")
    sender.join(BOUND)
    assert found.get("done"), "B never finished"
    assert "error" not in found, found.get("error")

    stored = rows(world.fresh().root())
    # Exactly one outcome: the real result.
    assert world.ran_with == [("echo", {"n": 1})]
    assert outcomes(world.root(), 1) == ["echo:1"]
    assert superseded_approval_ids(stored) == set()
    assert approval_decisions(stored) == {approval: True}
    # B's turn was shown the call's real result, never a "not run".
    replay = str(other.requests[-1].messages)
    assert "echo:1" in replay and SUPERSEDED_RESULT not in replay


# -- PR1: a host row while the approved call runs ---------------------------------

HOST_ROWS: dict[str, dict[str, Any]] = {
    "user": {"type": "user", "content": "A later message."},
    "assistant_text": {
        "type": "assistant",
        "content": "An aside.",
        "tool_calls": [],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    "assistant_step": {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": "later", "name": "lookup", "arguments": {"n": 7}}],
        "additional_content": {},
        "tool_approval_requests": [],
    },
}


@pytest.mark.parametrize("where", ["another_worker", "inside_the_tool"])
@pytest.mark.parametrize("kind", sorted(HOST_ROWS))
def test_pr1_a_host_row_while_the_approved_call_runs_is_refused(kind: str, where: str) -> None:
    world = Hooked()
    approval = approved_and_parked(world)
    attempt: dict[str, Any] = {}

    def record() -> None:
        host = (other_worker(world) if where == "another_worker" else world).root()
        host.thread().record([dict(HOST_ROWS[kind])])

    def during() -> None:
        if where == "another_worker":
            worker, found = in_thread(record)
            worker.start()
            worker.join(BOUND)
            attempt.update(found)
        else:
            try:
                record()
                attempt["done"] = True
            except HarnessError as error:
                attempt["error"] = error

    world.during = during
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    error = attempt.get("error")
    assert isinstance(error, HarnessError) and error.code == "session_locked", attempt
    stored = rows(world.fresh().root())
    # Nothing of the refused row was written, and the call has ONE outcome.
    assert not any(row.get("content") in ("A later message.", "An aside.") for row in stored)
    assert world.ran_with == [("echo", {"n": 1})]
    assert outcomes(world.root(), 1) == ["echo:1"]
    assert superseded_approval_ids(stored) == set()
    assert approval_decisions(stored) == {approval: True}
    # Once the resolution is over the same row records, and supersedes nothing.
    world.fresh().root().thread().record([dict(HOST_ROWS[kind])])
    assert [row for row in rows(world.root()) if SUPERSEDED in row] == []
    assert outcomes(world.root(), 1) == ["echo:1"]


def test_a_host_row_before_the_resolution_claims_the_call_still_supersedes_it() -> None:
    """The claim covers only a resolution in progress: a host row recorded
    before a resume still replaces the step (one denial, nothing runs)."""
    world = Hooked()
    approval = approved_and_parked(world)
    world.fresh().root().thread().record([dict(HOST_ROWS["user"])])
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    assert outcomes(world.root(), 1) == [SUPERSEDED_RESULT]
    assert superseded_approval_ids(rows(world.root())) == {approval}


def test_a_resolution_that_dies_inside_the_call_releases_its_claim() -> None:
    """A resolver that dies inside the call (even on a BaseException) never
    leaves the session refusing every later row."""
    world = Hooked()
    approved_and_parked(world)

    class Crash(BaseException):
        pass

    def crash() -> None:
        raise Crash

    world.during = crash
    with pytest.raises(Crash):
        world.fresh().runtime.send(world.root(), "")
    # The claim is released on the way out, even on a BaseException.
    world.fresh().root().thread().record([dict(HOST_ROWS["user"])])
    assert outcomes(world.root(), 1) == [SUPERSEDED_RESULT]


# -- W2 / W1: the replaced denial and the superseding row are one write -----------


def _thread_writes(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, Any]]]:
    """Every thread document written, as its rows, in write order."""
    written: list[list[dict[str, Any]]] = []
    original_put = FileSessionStore.put

    def put(self: FileSessionStore, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key.endswith(":thread") and isinstance(value, dict):
            written.append(
                [
                    entry["message"]
                    for entry in value.get("messages") or []
                    if isinstance(entry, dict) and isinstance(entry.get("message"), dict)
                ]
            )
        return original_put(self, key, value, *args, **kwargs)

    monkeypatch.setattr(FileSessionStore, "put", put)
    return written


def test_w2_no_write_leaves_a_replaced_step_answerable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every thread state a new message writes: none holds the replaced
    denial while the approval still reads pending (the window a decision
    was accepted in when the denial and the user row were two writes)."""
    world = Hooked()
    _, approvals = park(world, ("c1", "echo", {"n": 1}), ("c2", "echo", {"n": 2}))
    written = _thread_writes(monkeypatch)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "Something else.")

    denied = [state for state in written if any(SUPERSEDED in row for row in state)]
    assert denied, "the new message denied nothing"
    for state in denied:
        assert pending_approval_ids(state) == []
        # The denial is never the last row: the user row is in the same write.
        assert state[-1].get(SUPERSEDED) is None
    for approval in approvals:
        with pytest.raises(HarnessError) as caught:
            record_approval(world.fresh().root(), approval, True)
        assert caught.value.code == "approval_not_pending"


def test_w1_a_failed_new_message_write_never_leaves_a_denial_with_an_answerable_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = Hooked()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    original_put = FileSessionStore.put

    def put(self: FileSessionStore, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key.endswith(":thread") and "Something else." in str(value):
            raise OSError("store down")
        return original_put(self, key, value, *args, **kwargs)

    monkeypatch.setattr(FileSessionStore, "put", put)
    with pytest.raises(OSError):
        world.fresh().runtime.send(world.root(), "Something else.")
    monkeypatch.setattr(FileSessionStore, "put", original_put)

    stored = rows(world.fresh().root())
    denied = any(SUPERSEDED in row for row in stored)
    try:
        record_approval(world.fresh().root(), approvals[0], True)
        accepted = True
    except HarnessError as error:
        assert error.code == "approval_not_pending"
        accepted = False
    # Either the step was not replaced (nothing written) and the decision
    # stands, or it was replaced and the decision is refused; never both.
    assert not (denied and accepted)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert len(outcomes(world.root(), 1)) == 1


def test_every_pending_reader_honours_the_superseded_mark() -> None:
    """A thread holding a replaced denial with nothing after it (a torn
    thread): the approval is not pending, a decision is refused, and no
    resume runs the call."""
    world = Hooked()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    root = world.fresh().root()
    stored = rows(root)
    denial = {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "c1",
                "tool_name": "echo",
                "args": {"n": 1},
                "result": SUPERSEDED_RESULT,
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
        SUPERSEDED: {"approval_ids": approvals},
    }
    root.thread().record([denial])
    stored = rows(world.fresh().root())

    assert pending_approval_ids(stored) == []
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().root(), approvals[0], True)
    assert caught.value.code == "approval_not_pending"
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == []
    assert outcomes(world.root(), 1) == [SUPERSEDED_RESULT]


# -- new channels the fix could open, each pinned ----------------------------------


def test_a_new_message_that_cannot_get_the_lock_in_time_writes_nothing() -> None:
    """A new message waits for the resolver only as long as ``lock_wait``:
    then it is refused ``session_locked`` with nothing written, and the call
    the resolver ran keeps its one outcome."""
    world = Hooked()
    approved_and_parked(world)
    other = other_worker(world)
    other.config = {"lock_wait": 0}
    other.fresh()
    other.text("B answer.")
    done = threading.Event()
    sender, found = in_thread(lambda: other.runtime.send(other.root(), "A new message."), done)

    def during() -> None:
        sender.start()
        assert done.wait(BOUND)

    world.during = during
    world.text("A answer.")
    world.fresh().runtime.send(world.root(), "")
    sender.join(BOUND)

    error = found.get("error")
    assert isinstance(error, HarnessError) and error.code == "session_locked", found
    stored = rows(world.fresh().root())
    assert not any(row.get("content") == "A new message." for row in stored)
    assert outcomes(world.root(), 1) == ["echo:1"]


def test_a_claim_lives_only_on_the_thread_document_never_in_a_row() -> None:
    """A row carrying the claim's field name claims nothing: later rows
    record, and a superseding row still supersedes."""
    world = Hooked()
    approval = approved_and_parked(world)
    forged = {"type": "system", "content": "x", "harness_resolving": {"token": "t", "until": 1e12}}
    world.fresh().root().thread().record([forged])

    world.fresh().root().thread().record([dict(HOST_ROWS["user"])])

    assert superseded_approval_ids(rows(world.root())) == {approval}
