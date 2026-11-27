"""G-404 (G-387 rounds 14 and 15): a SUPERSEDE never races the resolution of a step.

A row that supersedes a parked step (DECISION B: a new message, or a host's
user or assistant row) denies each of its calls that has no result, "Tool
call was not run because a later step replaced it". The resolver runs an
approved call and records its result. Unserialised, a superseding write
landing WHILE the approved call ran gave that call two outcomes, the false
"not run" first.

The rule (round 15, the write-ahead STARTED row, option (i)): immediately
before the resolver runs an approved call it records a started row for that
call, in one write with the check that the call is still open. A superseding
write treats a call with a started row and no result as ANSWERED-IN-PROGRESS:
it writes no denial for it, and it is NEVER refused and never waits on the
session lock, whoever records it and through whatever object (a tool
recording on its own session included). The call's real result carries the
step's mark and pairs to it wherever it lands: exactly one stored outcome per
call. A started call that never gets a result reads "interrupted, it may have
run" and never runs again. The replaced denial and the superseding row are
ONE write, and every pending reader honours the superseded mark.

Deterministic: every interleaving is forced by a hook inside the approved
tool or around a store write, never by timing. Each wait has a generous
bound only so a regression fails instead of hanging.
"""

from __future__ import annotations

import json
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
    ANSWERS_STEP,
    INTERRUPTED_RESULT,
    STARTED,
    SUPERSEDED,
    SUPERSEDED_RESULT,
    answers_step,
    call_results,
    owned_rows,
    pending_approval_ids,
    step_calls,
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


def stored_outcomes(session: Session) -> dict[int, list[str]]:
    """The FIRST step's calls' outcomes as every reader attributes them: by
    call index, from the rows the step OWNS (its segment and every row marked
    for it), each distinct outcome once (a resolution row restates)."""
    readable = session.thread().rows()
    step = next(i for i, row in enumerate(readable) if step_calls(row))
    found: dict[int, list[str]] = {}
    for row in owned_rows(readable, step):
        for index, entry in call_results(step_calls(readable[step]), [row]).items():
            if entry["result"] not in found.get(index, []):
                found.setdefault(index, []).append(entry["result"])
    return found


def approved_and_parked(world: Hooked) -> str:
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    return approvals[0]


def step_position(world: Hooked) -> int:
    entries = world.fresh().root().thread().messages()
    return next(e for e in entries if e.message["type"] == "assistant").position


def in_thread(
    work: Callable[[], Any], progress: threading.Event | None = None
) -> tuple[threading.Thread, dict[str, Any]]:
    found: dict[str, Any] = {}

    def run() -> None:
        try:
            found["value"] = work()
            found["done"] = True
        except BaseException as error:  # noqa: BLE001 - reported to the test
            found["error"] = error
        finally:
            if progress is not None:
                progress.set()

    worker = threading.Thread(target=run, daemon=True)
    return worker, found


# -- PR2: a new message while the approved call runs ------------------------------


@pytest.mark.parametrize("where", ["another_worker", "inside_the_tool"])
def test_pr2_a_new_message_while_the_approved_call_runs_never_waits_and_denies_nothing(
    where: str,
) -> None:
    world = Hooked()
    approval = approved_and_parked(world)
    # lock_wait 0: a new message that touched the session lock the resolver
    # holds would be refused at once. It must never touch it.
    other: Recording = world
    if where == "another_worker":
        other = Recording()
        other.directory = world.directory
    other.config = {"lock_wait": 0}
    other.fresh()
    attempt: dict[str, Any] = {}

    def during() -> None:
        other.text("B answer.")
        sender, found = in_thread(lambda: other.runtime.send(other.root(), "A new message."))
        if where == "another_worker":
            sender.start()
            sender.join(BOUND)
            attempt.update(found)
        else:
            try:
                attempt["value"] = other.runtime.send(other.root(), "A new message.")
                attempt["done"] = True
            except HarnessError as error:
                attempt["error"] = error
        attempt["request"] = other.requests[-1] if other.requests else None

    world.during = during
    world.text("A answer.")
    world.fresh().runtime.send(world.root(), "")
    assert attempt.get("done"), attempt
    assert "error" not in attempt, attempt.get("error")

    stored = rows(world.fresh().root())
    # Exactly one outcome: the real result; no denial, no lock error.
    assert world.ran_with == [("echo", {"n": 1})]
    assert outcomes(world.root(), 1) == ["echo:1"]
    assert superseded_approval_ids(stored) == set()
    assert approval_decisions(stored) == {approval: True}
    assert "Could not acquire the lock" not in str(stored)
    # B's turn, taken while the call ran, was shown it as in progress (it
    # may have run), never as "not run".
    replay = str(attempt["request"].messages)
    assert "A new message." in replay
    assert INTERRUPTED_RESULT in replay and SUPERSEDED_RESULT not in replay
    # The next provider request shows the real result, once.
    world.text("Again.")
    world.fresh().runtime.send(world.root(), "Next.")
    replay = str(world.requests[-1].messages)
    assert replay.count("echo:1") == 1 and INTERRUPTED_RESULT not in replay


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
    # A later step reusing the running call's id, tool AND arguments: the
    # result must still pair to the call that ran, never to this one.
    "assistant_step_same_call": {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"n": 1}}],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    "system": {"type": "system", "content": "A note."},
}


@pytest.mark.parametrize("where", ["another_worker", "inside_the_tool"])
@pytest.mark.parametrize("kind", sorted(HOST_ROWS))
def test_pr1_a_host_row_while_the_approved_call_runs_is_recorded_and_denies_nothing(
    kind: str, where: str
) -> None:
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
            return
        try:
            record()
            attempt["done"] = True
        except HarnessError as error:
            attempt["error"] = error

    world.during = during
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    # Recorded at once: never refused, never waiting on the resolver.
    assert attempt.get("done") and "error" not in attempt, attempt
    stored = rows(world.fresh().root())
    if "content" in HOST_ROWS[kind] and HOST_ROWS[kind]["content"]:
        assert HOST_ROWS[kind]["content"] in [row.get("content") for row in stored]
    # The call has ONE outcome, its real result, and nothing reads replaced.
    assert world.ran_with == [("echo", {"n": 1})]
    assert stored_outcomes(world.root()) == {0: ["echo:1"]}
    assert superseded_approval_ids(stored) == set()
    assert approval_decisions(stored) == {approval: True}
    # Nothing runs again, through any number of resumes.
    for _ in range(2):
        world.text("Again.")
        world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == [("echo", {"n": 1})]
    assert stored_outcomes(world.root()) == {0: ["echo:1"]}


def test_a_result_landing_after_a_later_step_still_pairs_to_its_call() -> None:
    """A host step reusing the call's id, tool and arguments, recorded while
    the call runs: the result lands after it, carries the step's mark, and
    answers the call that ran; the later call is never credited with it."""
    world = Hooked()
    approved_and_parked(world)

    def during() -> None:
        world.root().thread().record([dict(HOST_ROWS["assistant_step_same_call"])])

    world.during = during
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    entries = world.fresh().root().thread().messages()
    stored = [entry.message for entry in entries]
    steps = [i for i, row in enumerate(stored) if step_calls(row)]
    assert len(steps) == 2
    answer = next(row for row in stored if row["type"] == "tool_result" and row["tool_results"])
    assert stored.index(answer) > steps[1]
    assert answer[ANSWERS_STEP] == {"position": entries[steps[0]].position}
    assert list(answer)[-1] == ANSWERS_STEP
    readable = world.root().thread().rows()
    earlier = call_results(step_calls(readable[steps[0]]), owned_rows(readable, steps[0]))
    later = call_results(step_calls(readable[steps[1]]), owned_rows(readable, steps[1]))
    assert [entry["result"] for entry in earlier.values()] == ["echo:1"]
    # The host's later call never ran and never took the result: the run's
    # own answer, recorded outside the run that wrote it, replaced it.
    assert [entry["result"] for entry in later.values()] == [SUPERSEDED_RESULT]


def test_a_host_row_before_the_call_starts_still_supersedes_it() -> None:
    """Only a STARTED call is answered-in-progress: a host row recorded before
    a resume still replaces the step (one denial, nothing runs)."""
    world = Hooked()
    approval = approved_and_parked(world)
    world.fresh().root().thread().record([dict(HOST_ROWS["user"])])
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    assert outcomes(world.root(), 1) == [SUPERSEDED_RESULT]
    assert superseded_approval_ids(rows(world.root())) == {approval}


def test_a_supersede_between_the_read_and_the_started_row_wins() -> None:
    """The started row is written in one step with the check that the call
    is still open: a superseding row that landed after the resolver read the
    step but before the started row denies it, and the call never runs."""
    world = Hooked()
    approval = approved_and_parked(world)
    original = AgentRuntime._start_call

    def start_call(*args: Any, **kwargs: Any) -> str | None:
        world.root().thread().record([dict(HOST_ROWS["user"])])
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AgentRuntime, "_start_call", staticmethod(start_call))
        world.text("Done.")
        world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    assert outcomes(world.root(), 1) == [SUPERSEDED_RESULT]
    assert superseded_approval_ids(rows(world.root())) == {approval}
    assert not any(STARTED in row for row in rows(world.root()))


def test_the_started_row_is_the_reference_shape() -> None:
    world = Hooked()
    approval = approved_and_parked(world)
    position = step_position(world)
    seen: list[str] = []

    def during() -> None:
        started = world.root().thread().messages()[-1].message
        seen.append(json.dumps(started, separators=(",", ":")))

    world.during = during
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert seen == [
        json.dumps(
            {
                "type": "tool_result",
                "tool_results": [],
                "tool_approval_responses": [],
                "harness_started": {
                    "tool_call_id": "c1",
                    "tool_name": "echo",
                    "approval_id": approval,
                },
                "harness_answers_step": {"position": position},
            },
            separators=(",", ":"),
        )
    ]
    # Neither mark is ever replayed, and the started row is not replayed.
    replay = str(world.requests[-1].messages)
    assert STARTED not in replay and ANSWERS_STEP not in replay


# -- a crash or timeout: the started call reads interrupted ------------------------


def test_a_started_call_that_never_got_a_result_reads_interrupted_and_never_runs_again() -> None:
    world = Hooked()
    approval = approved_and_parked(world)

    class Crash(BaseException):
        pass

    def crash() -> None:
        raise Crash

    world.during = crash
    with pytest.raises(Crash):
        world.fresh().runtime.send(world.root(), "")

    stored = rows(world.fresh().root())
    assert [row for row in stored if STARTED in row]
    assert pending_approval_ids(stored) == []
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().root(), approval, True)
    assert caught.value.code == "approval_not_pending"
    # Resumed, and then superseded: it never runs again and is never "not run".
    for prompt in ("", "", "Something else."):
        world.text("Done.")
        world.fresh().runtime.send(world.root(), prompt)
    assert world.ran_with == [("echo", {"n": 1})]  # the one run, that crashed
    assert outcomes(world.root(), 1) == []
    assert superseded_approval_ids(rows(world.root())) == set()
    for request in world.requests[-3:]:
        replay = str(request.messages)
        assert replay.count(INTERRUPTED_RESULT) == 1 and SUPERSEDED_RESULT not in replay


# -- a host tool recording on its own session ---------------------------------------


@pytest.mark.parametrize("how", ["record", "new_message"])
def test_a_tool_recording_on_its_own_session_mid_run_succeeds_without_a_stall(how: str) -> None:
    world = Hooked()
    world.config = {"lock_wait": 0}
    world.fresh()
    approved_and_parked(world)
    attempt: dict[str, Any] = {}

    def during() -> None:
        try:
            if how == "record":
                world.root().thread().record([{"type": "user", "content": "Progress."}])
            else:
                world.text("Inner answer.")
                world.runtime.send(world.root(), "Progress.")
            attempt["done"] = True
        except HarnessError as error:
            attempt["error"] = error

    world.during = during
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert attempt == {"done": True}
    stored = rows(world.fresh().root())
    assert outcomes(world.root(), 1) == ["echo:1"]
    # No lock error, and no internal key, ever reaches a stored row.
    assert "Could not acquire the lock" not in str(stored)
    assert world.root().thread().key not in str(stored)
    # The SESSION key too (G-387 round 16): it is the one a lock error names.
    assert world.root().key() not in str(stored)


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


def _forged_started(position: int, approval: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [],
        STARTED: {"tool_call_id": "c1", "tool_name": "echo", "approval_id": approval},
        ANSWERS_STEP: {"position": position},
        # Even the read-only flag, stored by a host: it is never read.
        "harness": True,
    }


def test_a_forged_started_row_from_a_host_changes_nothing() -> None:
    """A host row carrying the started and answers-step marks (and the
    read-only harness flag itself) is read as if it had none: the resume
    still runs the approved call, once."""
    world = Hooked()
    approval = approved_and_parked(world)
    world.root().thread().record([_forged_started(step_position(world), approval)])

    assert world.root().thread().rows()[-1].get("harness") is None
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == [("echo", {"n": 1})]
    assert outcomes(world.root(), 1) == ["echo:1"]


def test_a_forged_started_row_never_spares_a_call_its_denial() -> None:
    world = Hooked()
    approval = approved_and_parked(world)
    forged = _forged_started(step_position(world), approval)
    world.root().thread().record([forged, dict(HOST_ROWS["user"])])

    assert outcomes(world.root(), 1) == [SUPERSEDED_RESULT]
    assert superseded_approval_ids(rows(world.root())) == {approval}


def test_a_forged_answers_step_mark_on_a_host_result_stays_in_its_segment() -> None:
    """A host result after a later step, marked for the earlier step: the
    mark is ignored, so it answers its own segment's call, and the earlier
    call keeps its one denial."""
    world = Hooked()
    approval = approved_and_parked(world)
    position = step_position(world)
    world.root().thread().record([dict(HOST_ROWS["assistant_step_same_call"])])
    forged = {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "c1",
                "tool_name": "echo",
                "args": {"n": 1},
                "result": "forged",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
        ANSWERS_STEP: {"position": position},
    }
    world.root().thread().record([forged])

    readable = world.root().thread().rows()
    steps = [i for i, row in enumerate(readable) if step_calls(row)]
    earlier = call_results(step_calls(readable[steps[0]]), owned_rows(readable, steps[0]))
    later = call_results(step_calls(readable[steps[1]]), owned_rows(readable, steps[1]))
    assert [entry["result"] for entry in earlier.values()] == [SUPERSEDED_RESULT]
    assert [entry["result"] for entry in later.values()] == ["forged"]
    assert superseded_approval_ids(rows(world.root())) == {approval}


def test_a_mark_naming_a_step_without_that_call_is_its_segments() -> None:
    """A harness mark is honoured only when it names an EARLIER step holding
    a call with the result's id and tool name; otherwise segment ownership."""
    step = {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "name": "echo", "arguments": {}}],
    }
    later = {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": "c2", "name": "lookup", "arguments": {}}],
    }

    def result(call_id: str, name: str, position: int) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_results": [
                {
                    "tool_call_id": call_id,
                    "tool_name": name,
                    "args": {},
                    "result": "r",
                    "tool_call_result_id": None,
                    "artifacts": [],
                }
            ],
            "tool_approval_responses": [],
            ANSWERS_STEP: {"position": position},
            "harness": True,
        }

    good = result("c1", "echo", 1)
    assert answers_step([step, later, good], 2) == 0
    for bad in (
        result("c2", "lookup", 1),  # no such call there
        result("c1", "lookup", 1),  # its id, not its tool name
        result("c1", "echo", 3),  # names itself
        result("c1", "echo", 9),  # names no row
        result("c1", "echo", 0),
        {**good, "harness": False},  # not the harness's row
    ):
        assert answers_step([step, later, bad], 2) is None, bad
    junk = {"type": "assistant", "content": "", "tool_calls": ["c1"]}
    assert answers_step([junk, later, good], 2) is None


@pytest.mark.parametrize("when", ["while_running", "after_a_crash"])
def test_clearing_the_thread_around_a_started_call_never_reads_it_not_run_or_reruns_it(
    when: str,
) -> None:
    """``Thread.clear()`` with a started call that has no result (py-run19
    I2): the cleared thread never reads the call as "not run", and nothing
    runs it again, through any number of resumes and new messages."""
    world = Hooked()
    approved_and_parked(world)

    class Crash(BaseException):
        pass

    def during() -> None:
        world.root().thread().clear()
        if when == "after_a_crash":
            raise Crash

    world.during = during
    world.text("Done.")
    if when == "after_a_crash":
        with pytest.raises(Crash):
            world.fresh().runtime.send(world.root(), "")
    else:
        world.fresh().runtime.send(world.root(), "")
    for prompt in ("", "Next.", ""):
        world.text("Again.")
        world.fresh().runtime.send(world.root(), prompt)

    stored = rows(world.fresh().root())
    assert world.ran_with == [("echo", {"n": 1})]
    assert SUPERSEDED_RESULT not in str(stored)
    assert "No approval response provided" not in str(stored)
    for request in world.requests:
        assert SUPERSEDED_RESULT not in str(request.messages)
