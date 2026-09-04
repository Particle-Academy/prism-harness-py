"""The agent task list: the four-state machine, claim-and-lease, and the record.

Written against ``prism-parity/specs/agent-task-lists.md``, which is the
authority for every observable decision asserted here. The spelling is Python's;
the decisions are the same in PHP and TypeScript or one of the three is wrong.

**Every assertion has a control.** A test that only ever sees the good state
cannot tell "the guard works" from "the guard is never reached" -- so a test
that claims a lease expires also checks it had NOT expired a moment earlier, and
the test for the atomic claim is paired with a deliberately non-atomic one that
hands the same task to two workers. If the racy version passed too, the suite
would be measuring nothing.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

import pytest

from prism_harness import (
    DEFAULT_LEASE_SECONDS,
    AgentTask,
    AgentTaskMixin,
    AgentTaskSource,
    FileSessionStore,
    HarnessError,
    MemorySessionStore,
    Participant,
    PrismHarness,
    RunBudget,
    RunLedger,
    Session,
    SessionStore,
    StoredTask,
    StoreTaskSource,
    TaskCompletionTool,
    TaskOutcome,
    TaskState,
    ToolAuthorizer,
    ToolRegistry,
    canonical_task_json,
    task_record,
)

#: 2025-01-01T00:00:00Z. A fixed instant, so the timestamps in the canonical
#: records below are literals a reader can check rather than a moving target.
EPOCH = 1_735_689_600.0

KEY = "session:tasks"


class Clock:
    """A wall clock a test can move without sleeping.

    Injected rather than monkeypatched: a lease is compared against a stored
    Unix timestamp, and the thing worth testing is the comparison, not
    ``time.time``.
    """

    def __init__(self, now: float = EPOCH) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def a_store() -> FileSessionStore:
    return FileSessionStore(tempfile.mkdtemp(prefix="prism-harness-tasks-"))


def a_source(
    store: SessionStore | None = None,
    clock: Clock | None = None,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> StoreTaskSource:
    return StoreTaskSource(
        store or a_store(),
        KEY,
        lease_seconds=lease_seconds,
        clock=clock or Clock(),
    )


def a_harness(directory: str | None = None) -> PrismHarness:
    """A harness whose durable slot is a real durable store."""
    where = directory or tempfile.mkdtemp(prefix="prism-harness-session-tasks-")
    return PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(where)},
        stores={"ephemeral": "memory", "durable": "files"},
    )


def a_session() -> Session:
    return a_harness().for_(Participant("User", 7)).session("support")


def raw(store: SessionStore) -> list[dict[str, Any]]:
    """What is actually on disk, bypassing the source entirely.

    Several decisions here are about what was WRITTEN, not what was returned,
    and a source that reported the right thing while storing something else
    would satisfy every assertion made through its own API.
    """
    payload = store.get(KEY) or {}
    entries = payload.get("tasks")
    return list(entries) if isinstance(entries, list) else []


# -- the contracts -----------------------------------------------------------


def test_the_store_source_satisfies_the_source_contract() -> None:
    # Checked by the type checker as much as at runtime: the assignment is what
    # makes mypy verify StoreTaskSource against the Protocol.
    concrete = a_source()
    source: AgentTaskSource = concrete
    task: AgentTask = StoredTask("t-1", "Do the thing", TaskState.TODO)

    assert source.pending() == 0
    assert task.state is TaskState.TODO

    # FOUR methods, and `find` is driven THROUGH the protocol-typed name on
    # purpose: `release()` takes a task while every external caller -- a tool, an
    # HTTP route, a worker resuming after a restart -- holds only an id. A
    # contract without it cannot be driven from outside the claim loop, which
    # this package's own completion tool demonstrated before anyone else did.
    assert source.find("t-1") is None

    concrete.add("Do the thing", "t-1")
    found = source.find("t-1")

    assert found is not None
    assert found.id == "t-1"
    assert found.state is TaskState.TODO

    # And the whole contract round-trips off nothing but an id.
    claimed = source.claim("worker-1")
    assert claimed is not None
    source.release(source.find("t-1") or claimed, "worker-1", TaskOutcome.DONE)

    settled = source.find("t-1")
    assert settled is not None
    assert settled.state is TaskState.DONE


def test_a_new_task_is_todo_and_unclaimed() -> None:
    source = a_source()
    task = source.add("Summarise the report")

    assert task.state is TaskState.TODO
    assert task.claimed_by is None
    assert task.claimed_until is None

    # The control: the same three assertions must NOT hold once it is claimed,
    # or they are asserting a constant rather than a state.
    claimed = source.claim("worker-1")
    assert claimed is not None
    assert claimed.state is TaskState.CLAIMED
    assert claimed.claimed_by == "worker-1"
    assert claimed.claimed_until is not None


def test_claim_writes_the_claim_to_the_store_before_it_returns() -> None:
    # "Started and died" has to be distinguishable from "never started". Writing
    # `claimed` after the work would make a crash look like a task nobody ever
    # attempted, and the only way to see the difference is to read the store.
    store = a_store()
    source = a_source(store, Clock())
    source.add("Do the thing", "t-1")

    assert raw(store)[0]["state"] == "todo"
    assert raw(store)[0]["claimed_by"] is None

    source.claim("worker-1")

    assert raw(store)[0]["state"] == "claimed"
    assert raw(store)[0]["claimed_by"] == "worker-1"
    assert raw(store)[0]["claimed_until"] == int(EPOCH + DEFAULT_LEASE_SECONDS)


def test_the_expiry_written_to_the_store_is_an_integer_and_not_a_float() -> None:
    """The type, not just the value -- and the two are not the same check.

    ``1735689900.0 == 1735689900`` is True in Python, so an equality assertion
    passes on a float that PHP and JavaScript would render as ``1735689900``
    and Python renders as ``1735689900.0``. Different bytes in the store, the
    same green test. This is that divergence, caught at the type.
    """
    store = a_store()
    source = a_source(store, Clock())
    source.add("Do the thing", "t-1")
    source.claim("worker-1")

    stored = raw(store)[0]["claimed_until"]

    assert isinstance(stored, int)
    assert not isinstance(stored, float)
    # The control: equality alone would have accepted the float.
    assert stored == int(EPOCH) + 300.0


def test_claim_returns_none_when_nothing_is_claimable() -> None:
    source = a_source()

    assert source.claim("worker-1") is None

    source.add("Do the thing")
    assert source.claim("worker-1") is not None
    assert source.claim("worker-2") is None


# -- one task, one worker ----------------------------------------------------


def test_two_workers_never_get_the_same_task() -> None:
    store = a_store()
    source = a_source(store)
    ids = [source.add(f"Task {index}", f"t-{index}").id for index in range(4)]

    workers = 6
    start = threading.Barrier(workers)
    claimed: list[str | None] = [None] * workers

    def take(index: int) -> None:
        # A SEPARATE source over the same store, which is what two workers
        # actually are. Sharing one object would test a lock this design does
        # not rely on.
        worker = StoreTaskSource(store, KEY)
        start.wait()
        task = worker.claim(f"worker-{index}")
        claimed[index] = None if task is None else task.id

    threads = [threading.Thread(target=take, args=(index,)) for index in range(workers)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    handed_out = [task_id for task_id in claimed if task_id is not None]

    assert sorted(handed_out) == sorted(ids)
    assert len(set(handed_out)) == len(handed_out)
    assert claimed.count(None) == workers - len(ids)


def test_a_read_then_mark_claim_hands_the_same_task_to_two_workers() -> None:
    """The control for the test above, and the reason ``claim()`` is one call.

    This is the shape the spec forbids -- read the next task, then mark it mine
    -- driven with a barrier so the interleaving is deterministic rather than
    lucky. It MUST produce a duplicate. If it did not, the atomic version's
    green tick would be evidence of nothing.
    """
    store = a_store()
    source = a_source(store)
    for index in range(4):
        source.add(f"Task {index}", f"t-{index}")

    workers = 4
    read_done = threading.Barrier(workers)
    claimed: list[str] = []
    guard = threading.Lock()

    def take(index: int) -> None:
        # READ, outside any lock. Every worker sees the same first todo.
        records = raw(store)
        target = next(record for record in records if record["state"] == "todo")
        read_done.wait()

        def mark() -> None:
            payload = store.get(KEY) or {}
            for record in payload.get("tasks", []):
                if record["id"] == target["id"]:
                    record["state"] = "claimed"
                    record["claimed_by"] = f"worker-{index}"
            store.put(KEY, payload)

        store.with_lock(KEY, mark)

        with guard:
            claimed.append(str(target["id"]))

    threads = [threading.Thread(target=take, args=(index,)) for index in range(workers)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(claimed) == workers
    assert len(set(claimed)) == 1, "read-then-mark should hand every worker the same task"


# -- ordering ----------------------------------------------------------------


def test_claim_hands_out_tasks_in_insertion_order() -> None:
    source = a_source()
    source.add("Third alphabetically", "z-1")
    source.add("First alphabetically", "a-2")
    source.add("Second alphabetically", "m-3")

    order = [source.claim(f"worker-{index}") for index in range(3)]
    ids = [task.id for task in order if task is not None]

    assert ids == ["z-1", "a-2", "m-3"]
    # The control: the ids were chosen so that ANY implicit sort would produce a
    # different sequence. Without this the assertion above would also pass on a
    # source that sorted its list, and ordering is the divergence class nothing
    # errors on.
    assert ids != sorted(ids)


def test_a_batch_lands_whole_and_in_order() -> None:
    source = a_source()
    added = source.add_many(["one", "two", "three"])

    assert [task.id for task in added] == ["t-1", "t-2", "t-3"]
    assert [task.instruction for task in source.records()] == ["one", "two", "three"]


def test_a_duplicate_id_is_refused() -> None:
    source = a_source()
    source.add("First", "t-1")

    with pytest.raises(HarnessError) as failure:
        source.add("Second", "t-1")

    assert failure.value.code == "duplicate_task_id"
    # The control: the refusal is about the id, not about adding twice.
    assert source.add("Second", "t-2").id == "t-2"


def test_a_blank_identifier_is_refused() -> None:
    source = a_source()

    with pytest.raises(HarnessError) as blank_task:
        source.add("Do the thing", "")

    assert blank_task.value.code == "task_identifier_blank"

    source.add("Do the thing", "t-1")

    with pytest.raises(HarnessError) as blank_worker:
        source.claim("")

    assert blank_worker.value.code == "task_identifier_blank"
    # The control: a non-blank worker gets the task the blank one was refused.
    assert source.claim("worker-1") is not None


# -- release, and terminality ------------------------------------------------


def test_release_records_the_outcome_and_clears_the_lease() -> None:
    source = a_source()
    task = source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    source.release(claimed, "worker-1", TaskOutcome.DONE)
    settled = source.find(task.id)

    assert settled is not None
    assert settled.state is TaskState.DONE
    assert settled.claimed_by is None
    assert settled.claimed_until is None


def test_done_and_failed_are_terminal() -> None:
    source = a_source()
    source.add("Finish", "t-1")
    source.add("Break", "t-2")

    done = source.claim("worker-1")
    failed = source.claim("worker-2")
    assert done is not None and failed is not None

    source.release(done, "worker-1", TaskOutcome.DONE)
    source.release(failed, "worker-2", TaskOutcome.FAILED)

    # The positive control: both releases actually landed, so the refusals below
    # are about terminality rather than about the tasks never having moved.
    assert [task.state for task in source.records()] == [TaskState.DONE, TaskState.FAILED]

    for task, holder in ((done, "worker-1"), (failed, "worker-2")):
        with pytest.raises(HarnessError) as failure:
            source.release(task, holder, TaskOutcome.DONE)

        assert failure.value.code == "task_already_terminal"


def test_a_failed_task_does_not_return_to_todo_on_its_own() -> None:
    # Automatic retry is a policy, policy needs backoff and attempt counts, and
    # that is the scheduler this must not become. The application re-queues.
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Break", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    source.release(claimed, "worker-1", TaskOutcome.FAILED)
    clock.advance(10_000)

    assert source.pending() == 0
    assert source.claim("worker-2") is None

    settled = source.find("t-1")
    assert settled is not None
    assert settled.state is TaskState.FAILED


def test_releasing_a_task_this_source_does_not_hold_is_refused() -> None:
    source = a_source()

    with pytest.raises(HarnessError) as failure:
        source.release(
            StoredTask("nope", "Do the thing", TaskState.CLAIMED), "worker-1", TaskOutcome.DONE
        )

    assert failure.value.code == "task_not_found"


def test_an_unclaimed_task_cannot_be_released() -> None:
    source = a_source()
    task = source.add("Do the thing", "t-1")

    with pytest.raises(HarnessError) as failure:
        source.release(task, "worker-1", TaskOutcome.DONE)

    assert failure.value.code == "task_lease_not_held"

    # The control: the same call succeeds once the task is genuinely claimed, so
    # the refusal is about the claim and not about the task or the outcome.
    claimed = source.claim("worker-1")
    assert claimed is not None
    source.release(claimed, "worker-1", TaskOutcome.DONE)

    settled = source.find("t-1")
    assert settled is not None
    assert settled.state is TaskState.DONE


# -- leases ------------------------------------------------------------------


def test_an_expired_lease_returns_the_task_to_todo_and_never_to_failed() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")
    source.claim("worker-1")

    # The control: still held a second before the lease ends.
    clock.advance(59)
    held = source.find("t-1")
    assert held is not None
    assert held.state is TaskState.CLAIMED

    clock.advance(1)
    lapsed = source.find("t-1")
    assert lapsed is not None
    # `failed` FIRST, and not merely as a second opinion on the line below: a
    # worker dying is not the task failing, and conflating them burns a retry
    # that never ran. Asserted before the narrowing one so it is a real check
    # rather than one the type checker has already decided.
    assert lapsed.state is not TaskState.FAILED
    assert lapsed.state is TaskState.TODO
    assert lapsed.claimed_by is None
    assert lapsed.claimed_until is None


def test_an_expired_task_is_claimable_by_anyone() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")

    assert source.claim("worker-1") is not None
    # The control: while the lease holds, nobody else gets it.
    assert source.claim("worker-2") is None

    clock.advance(61)
    reclaimed = source.claim("worker-2")

    assert reclaimed is not None
    assert reclaimed.id == "t-1"
    assert reclaimed.claimed_by == "worker-2"
    assert reclaimed.claimed_until == int(clock.now + 60)


def test_a_worker_whose_lease_expired_cannot_release_the_task() -> None:
    # It may well have finished the work -- and another worker may already be
    # redoing it. Accepting a report from a lapsed holder is how two workers
    # both mark one task done.
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    clock.advance(61)

    with pytest.raises(HarnessError) as failure:
        source.release(claimed, "worker-1", TaskOutcome.DONE)

    assert failure.value.code == "task_lease_not_held"


def test_a_second_source_over_the_same_store_sees_the_predecessors_lease() -> None:
    # The reboot story: a restarted process resolves the same address, sees the
    # same list, and finds the task its predecessor held either still leased or
    # expired back to todo.
    store = a_store()
    clock = Clock()
    a_source(store, clock, lease_seconds=60).add("Do the thing", "t-1")
    a_source(store, clock, lease_seconds=60).claim("worker-1")

    restarted = a_source(store, clock, lease_seconds=60)
    assert restarted.pending() == 0
    assert restarted.claim("worker-2") is None

    clock.advance(61)
    assert restarted.pending() == 1
    assert restarted.claim("worker-2") is not None


def test_pending_counts_what_is_claimable_and_nothing_else() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add_many(["one", "two", "three", "four"])

    assert source.pending() == 4

    held = source.claim("worker-1")
    assert held is not None
    assert source.pending() == 3, "a live claim is not claimable"

    done = source.claim("worker-2")
    assert done is not None
    source.release(done, "worker-2", TaskOutcome.DONE)
    assert source.pending() == 2, "a terminal task is not claimable"

    clock.advance(61)
    assert source.pending() == 3, "a lapsed claim is claimable again"


# -- lease extension ---------------------------------------------------------


def a_ledger(elapsed: float = 0.0) -> RunLedger:
    """A ledger that has been running for ``elapsed`` seconds.

    ``RunLedger`` measures against ``time.monotonic``, deliberately, so this
    backdates its start rather than pretending the monotonic clock is
    injectable.
    """
    return RunLedger("run-1", started_at=time.monotonic() - elapsed)


def test_a_holder_may_extend_its_own_lease() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None
    assert claimed.claimed_until == int(EPOCH + 60)

    clock.advance(30)
    extended = source.extend_lease(claimed, "worker-1", a_ledger(), RunBudget(max_steps=8))

    assert extended.claimed_until == int(EPOCH + 30 + 60)
    assert extended.state is TaskState.CLAIMED


def test_only_the_holder_may_extend() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    with pytest.raises(HarnessError) as failure:
        source.extend_lease(claimed, "worker-2", a_ledger(), RunBudget(max_steps=8))

    assert failure.value.code == "task_lease_not_held"

    # The control: the refused call changed nothing, and the real holder can
    # still extend -- so the guard is about the worker, not about the task.
    still_held = source.find("t-1")
    assert still_held is not None
    assert still_held.claimed_until == int(EPOCH + 60)
    assert source.extend_lease(claimed, "worker-1", a_ledger(), RunBudget(max_steps=8)) is not None


def test_a_lapsed_lease_cannot_be_extended() -> None:
    # Only while it still holds it. A worker cannot take a task back by
    # extending; it is todo, and someone else may already have it.
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    clock.advance(61)

    with pytest.raises(HarnessError) as failure:
        source.extend_lease(claimed, "worker-1", a_ledger(), RunBudget(max_steps=8))

    assert failure.value.code == "task_lease_not_held"


def test_a_terminal_task_cannot_be_extended() -> None:
    source = a_source()
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None
    source.release(claimed, "worker-1", TaskOutcome.DONE)

    with pytest.raises(HarnessError) as failure:
        source.extend_lease(claimed, "worker-1", a_ledger(), RunBudget(max_steps=8))

    assert failure.value.code == "task_already_terminal"


def test_extension_is_bounded_by_the_runs_remaining_wall_clock_budget() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=300)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    bounded = source.extend_lease(
        claimed, "worker-1", a_ledger(), RunBudget(max_steps=8, max_seconds=60)
    )

    assert bounded.claimed_until is not None
    assert bounded.claimed_until < int(EPOCH + 300), "the lease outran the run's own allowance"
    assert int(EPOCH + 55) <= bounded.claimed_until <= int(EPOCH + 60)

    # The positive control: with room in the budget the FULL lease is granted,
    # so the clamp above is the budget doing the work and not a shorter lease.
    generous = source.extend_lease(
        claimed, "worker-1", a_ledger(), RunBudget(max_steps=8, max_seconds=6000)
    )

    assert generous.claimed_until == int(EPOCH + 300)


def test_extension_is_refused_once_the_wall_clock_budget_is_spent() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=300)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    with pytest.raises(HarnessError) as failure:
        source.extend_lease(
            claimed, "worker-1", a_ledger(elapsed=120), RunBudget(max_steps=8, max_seconds=60)
        )

    assert failure.value.code == "run_not_permitted"

    # The refusal left the lease alone rather than shortening it on the way out.
    untouched = source.find("t-1")
    assert untouched is not None
    assert untouched.claimed_until == int(EPOCH + 300)

    # The control: a budget with time left still grants an extension, so the
    # refusal is about exhaustion and not about the ledger being present.
    allowed = source.extend_lease(
        claimed, "worker-1", a_ledger(elapsed=55), RunBudget(max_steps=8, max_seconds=60)
    )
    assert allowed.claimed_until is not None
    assert allowed.claimed_until <= int(EPOCH + 5)


def test_a_cancelled_run_cannot_extend_its_lease() -> None:
    # The lease is bounded by the RUN, and a cancelled run may not take another
    # step. Checking only the wall clock left a cancelled worker extending
    # indefinitely: the loop it was extending for would refuse to run, and the
    # task stayed locked away from every worker that could still do it.
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    ledger = a_ledger()
    budget = RunBudget(max_steps=8, max_seconds=6000)

    # The control FIRST, on the same ledger: it extends fine until cancelled,
    # so the refusal below is the cancellation and nothing else.
    assert source.extend_lease(claimed, "worker-1", ledger, budget) is not None

    ledger.cancel("the operator stopped it")

    with pytest.raises(HarnessError) as failure:
        source.extend_lease(claimed, "worker-1", ledger, budget)

    assert failure.value.code == "run_not_permitted"
    assert "the operator stopped it" in failure.value.message


def test_a_step_exhausted_run_cannot_extend_its_lease() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    ledger = a_ledger()
    budget = RunBudget(max_steps=8, max_seconds=6000)
    ledger.record_steps(7)

    # The control: one step below the cap still extends.
    assert source.extend_lease(claimed, "worker-1", ledger, budget) is not None

    ledger.record_steps(1)

    with pytest.raises(HarnessError) as failure:
        source.extend_lease(claimed, "worker-1", ledger, budget)

    assert failure.value.code == "run_not_permitted"
    assert "step budget" in failure.value.message


def test_a_cost_exhausted_run_cannot_extend_its_lease() -> None:
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    ledger = a_ledger()
    budget = RunBudget(max_steps=8, max_cost_usd=1.0, max_seconds=6000)
    ledger.record_cost(0.25)

    assert source.extend_lease(claimed, "worker-1", ledger, budget) is not None

    ledger.record_cost(0.80)

    with pytest.raises(HarnessError) as failure:
        source.extend_lease(claimed, "worker-1", ledger, budget)

    assert failure.value.code == "run_not_permitted"
    assert "cost budget" in failure.value.message


def test_a_budget_with_no_wall_clock_cap_does_not_get_one_invented_here() -> None:
    # Two spellings of one limit is how a bound ends up set in the place that is
    # not enforced. If the operator declined to cap wall-clock, this does not
    # cap it on their behalf -- it grants the configured lease and no more.
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=300)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    extended = source.extend_lease(
        claimed, "worker-1", a_ledger(elapsed=10_000), RunBudget(max_steps=8)
    )

    assert extended.claimed_until == int(EPOCH + 300)


@pytest.mark.parametrize(
    "lease",
    [
        0,
        -1,
        -0.001,
        -300,
        float("nan"),
        float("inf"),
        float("-inf"),
        # Fractional, and positive, and finite. The same rule one scale down --
        # see the docstring below and `_require_lease`.
        0.5,
        90.4,
        299.999,
        300.5,
        # Not a number at all, and not a TypeError either: an untyped caller
        # gets a code like every other refusal here. `None` is deliberately NOT
        # in this list -- it is refused at the constructor, where it is not a
        # lease, and it MEANS "the source's default" at claim() and
        # extend_lease(), which is the documented signature.
        "300",
        # An int too large to become a float. `math.isfinite` RAISES on this
        # rather than returning False, so it needs its own answer.
        10**400,
    ],
)
def test_a_lease_that_is_not_a_positive_whole_number_is_refused(lease: float) -> None:
    """REFUSED, not clamped and not truncated, at every door a lease arrives through.

    TypeScript clamped a non-positive lease up to one second. This refuses,
    because a clamped value is a configuration that silently became a different
    configuration -- and the direction is not neutral: a lease of zero or less
    puts ``claimed_until`` in the PAST, so the claim expires the instant it is
    granted and the next caller steals it. Two workers on one task, from a
    number nobody was told was wrong.

    **Fractional is the same rule one scale down**, and it is here because the
    cross-language corpus caught it: ``suites/agent-task-claim`` atc-0017 asks
    for a lease of ``90.4``, the reference cannot even ask the question because
    ``claim()`` declares ``?int``, ``prism-harness-ts`` refused it, and this
    port accepted it and truncated to 90. ``claimed_until`` is an integer
    timestamp in all three languages, so a fractional lease could never be
    honoured as written -- accepting it means granting a different number and
    saying nothing. "It truncates in the safe direction" is the clamping
    argument restated, and that was not enough for zero.

    NaN is in the list because ``nan <= 0`` is FALSE. A bare positivity check
    lets it through and it detonates later inside ``int(now + nan)`` -- a crash
    several frames from the value that caused it, rather than a code.
    """
    store = a_store()

    with pytest.raises(HarnessError) as constructed:
        StoreTaskSource(store, KEY, lease_seconds=lease)

    assert constructed.value.code == "task_lease_invalid"

    source = a_source(store, Clock())
    source.add("Do the thing", "t-1")

    with pytest.raises(HarnessError) as claimed:
        source.claim("worker-1", lease_seconds=lease)

    assert claimed.value.code == "task_lease_invalid"

    # Nothing was claimed on the way out of that refusal.
    untouched = source.find("t-1")
    assert untouched is not None
    assert untouched.state is TaskState.TODO

    held = source.claim("worker-1")
    assert held is not None

    with pytest.raises(HarnessError) as extended:
        source.extend_lease(
            held, "worker-1", a_ledger(), RunBudget(max_steps=8), lease_seconds=lease
        )

    assert extended.value.code == "task_lease_invalid"
    # And the lease it refused to change is exactly as it was.
    still_held = source.find("t-1")
    assert still_held is not None
    assert still_held.claimed_until == int(EPOCH + DEFAULT_LEASE_SECONDS)


def test_a_positive_whole_lease_goes_through_all_three_doors() -> None:
    # The control for the refusals above: if any of the three simply rejected
    # everything, every case there would pass and the source would be unusable.
    # `30.0` is spelled as a float on purpose -- an INTEGRAL float is a whole
    # number of seconds and must go through, or the guard is refusing a type
    # rather than a value.
    clock = Clock()
    source = StoreTaskSource(a_store(), KEY, lease_seconds=30.0, clock=clock)
    source.add("Do the thing", "t-1")

    claimed = source.claim("worker-1", lease_seconds=120)
    assert claimed is not None
    assert claimed.claimed_until == int(EPOCH + 120)

    extended = source.extend_lease(
        claimed, "worker-1", a_ledger(), RunBudget(max_steps=8), lease_seconds=240
    )
    assert extended.claimed_until == int(EPOCH + 240)


def test_a_fractional_lease_is_refused_on_the_configuration_route_too() -> None:
    """The route a real consumer configures a lease through, not a direct call.

    The reference's fractional lease survived at first precisely BECAUSE the
    typed door already refused it: ``claim()`` declares ``?int``, so nobody
    looked further, and the config route truncated silently because the shipped
    config file cast the setting with ``(int)``. The guard was defeated from
    inside the file that declares the setting.

    This port has no config file, and ``Session.tasks()`` is the equivalent
    route -- the place an application names its lease once. It passes the value
    through untouched, which is what makes it safe, and this is what says so.
    """
    session = a_session()

    with pytest.raises(HarnessError) as configured:
        session.tasks(lease_seconds=90.4)

    assert configured.value.code == "task_lease_invalid"

    # The control, on the same route: a whole lease is configured and honoured.
    assert session.tasks(lease_seconds=90).claim("worker-1") is None


def test_the_default_lease_is_five_minutes() -> None:
    # Pinned in the spec so all three languages use one number. The value
    # matters less than its being the same everywhere, which is why it is
    # asserted rather than left to whatever the constructor happened to take.
    assert DEFAULT_LEASE_SECONDS == 300.0

    clock = Clock()
    source = a_source(clock=clock)
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")

    assert claimed is not None
    assert claimed.claimed_until == int(EPOCH + 300)


# -- durability --------------------------------------------------------------


def test_a_volatile_store_is_refused_at_construction() -> None:
    # A half-finished task list that vanishes on a deploy is indistinguishable
    # from a finished one. Refused when the source is BUILT, so the
    # misconfiguration cannot lie dormant until the first claim.
    with pytest.raises(HarnessError) as failure:
        StoreTaskSource(MemorySessionStore(), KEY)

    assert failure.value.code == "unsafe_state_configuration"
    assert "VOLATILE" in failure.value.message

    # The control: the same construction against a durable store works, so the
    # refusal is the durability check and not a broken constructor.
    assert StoreTaskSource(a_store(), KEY).pending() == 0


def test_the_shipped_harness_refuses_a_task_list_it_could_not_keep() -> None:
    # The configuration an installing application actually receives: no drivers,
    # so both slots resolve to the in-memory store and the durable one is
    # refused. Testing only a configured harness would never exercise it.
    with pytest.raises(HarnessError) as failure:
        PrismHarness().for_(Participant("User", 7)).session("support")

    assert failure.value.code == "unsafe_state_configuration"


def test_a_session_addresses_its_task_list_under_its_own_key() -> None:
    directory = tempfile.mkdtemp(prefix="prism-harness-session-tasks-")
    harness = a_harness(directory)
    session = harness.for_(Participant("User", 7)).session("support")
    session.tasks().add("Do the thing", "t-1")

    # A different process, resolving the same address, sees the same list.
    resumed = a_harness(directory).for_(Participant("User", 7)).session("support")

    assert [task.id for task in resumed.tasks().records()] == ["t-1"]
    # The control: a different scope is a different list, not the same one, so
    # the line above is the address resolving rather than one global list.
    assert harness.for_(Participant("User", 7)).session("other").tasks().pending() == 0


# -- the canonical record ----------------------------------------------------


def test_the_canonical_record_is_the_bytes_the_spec_pins() -> None:
    task = StoredTask("t-1", "Summarise the report", TaskState.TODO)

    assert task.to_canonical_json() == (
        '{"claimed_by":null,"claimed_until":null,"id":"t-1",'
        '"instruction":"Summarise the report","state":"todo"}'
    )


def test_the_claim_keys_are_present_and_null_when_unclaimed() -> None:
    # 0002 makes absent versus null an observable decision. A port modelling
    # unset as `undefined` would drop the keys entirely, which is a different
    # record even though every language would call it "empty".
    record = StoredTask("t-1", "Do the thing", TaskState.TODO).to_dict()

    assert "claimed_by" in record
    assert "claimed_until" in record
    assert record["claimed_by"] is None
    assert record["claimed_until"] is None

    # The control: the keys carry values when there is a claim, so their
    # presence above is not the encoder always emitting nulls.
    claimed = StoredTask("t-1", "Do the thing", TaskState.CLAIMED, "worker-1", 1_735_689_900)
    assert claimed.to_dict()["claimed_by"] == "worker-1"
    assert claimed.to_dict()["claimed_until"] == 1_735_689_900


def test_claimed_until_is_an_integer_unix_timestamp() -> None:
    # Not a formatted date: date formatting is exactly where three languages
    # produce three strings from one instant. Not a float either -- Python
    # renders the float 1.0 as "1.0" where PHP and JavaScript render "1".
    task = StoredTask("t-1", "Do the thing", TaskState.CLAIMED, "worker-1", 1_735_689_900)
    encoded = task.to_canonical_json()

    assert '"claimed_until":1735689900' in encoded
    assert "1735689900.0" not in encoded
    assert '"claimed_until":"' not in encoded
    assert isinstance(task.to_dict()["claimed_until"], int)


def test_the_keys_are_sorted() -> None:
    record = StoredTask("t-1", "Do the thing", TaskState.TODO).to_dict()

    assert list(record) == sorted(record)
    assert list(record) == ["claimed_by", "claimed_until", "id", "instruction", "state"]


def test_pythons_json_defaults_would_put_different_bytes_on_the_wire() -> None:
    """The control for the encoder, and the reason it takes explicit arguments.

    ``json.dumps`` pads every separator with a space and escapes non-ASCII to
    ``\\uXXXX``. Neither is visible from inside Python -- both round-trip
    perfectly -- and both are different bytes from what PHP and JavaScript
    emit. So the divergence is asserted here rather than trusted to a comment.
    """
    task = StoredTask("t-1", "Lire le résumé at https://example.com/日本語", TaskState.TODO)
    canonical = task.to_canonical_json()
    defaults = json.dumps(task.to_dict())

    assert canonical != defaults
    assert ", " not in canonical
    assert '", "' in defaults
    assert "\\u" not in canonical
    assert "\\u00e9" in defaults
    assert "résumé" in canonical
    assert "日本語" in canonical
    assert "\\/" not in canonical
    assert "https://example.com/日本語" in canonical


def test_a_record_built_by_hand_encodes_the_same_way_as_a_stored_one() -> None:
    task = StoredTask("t-1", "Do the thing", TaskState.CLAIMED, "worker-1", 1_735_689_900)

    assert canonical_task_json(task, "worker-1", 1_735_689_900) == task.to_canonical_json()
    assert task_record(task, None, None)["claimed_by"] is None


def test_a_state_the_machine_does_not_have_is_refused() -> None:
    store = a_store()
    source = a_source(store)
    source.add("Do the thing", "t-1")

    payload = store.get(KEY) or {}
    payload["tasks"][0]["state"] = "in_progress"
    store.put(KEY, payload)

    with pytest.raises(HarnessError) as failure:
        source.records()

    assert failure.value.code == "unmappable_content"

    # The control: the four the machine does have are all accepted.
    for state in ("todo", "claimed", "done", "failed"):
        payload["tasks"][0]["state"] = state
        store.put(KEY, payload)
        assert source.records()[0].state.value == state


# -- a consumer's own record -------------------------------------------------


@dataclass(frozen=True)
class ConsumerRow:
    """A consumer's own record. Conforms with no base class and no import."""

    id: str
    instruction: str
    state: TaskState


class ConsumerModel(AgentTaskMixin):
    """A record whose columns are named something else entirely."""

    def __init__(self, pk: int, body: str, status: str) -> None:
        self.pk = pk
        self.body = body
        self.status = status

    def agent_task_id(self) -> str:
        return f"row-{self.pk}"

    def agent_task_instruction(self) -> str:
        return self.body

    def agent_task_state(self) -> TaskState:
        return TaskState(self.status)


class HalfBuiltModel(AgentTaskMixin):
    def agent_task_id(self) -> str:
        return "row-1"


def test_a_consumers_own_record_conforms_without_the_mixin() -> None:
    # Python's answer to the PHP trait: structural typing means there is nothing
    # to mix in. The assignment is the assertion -- mypy checks it.
    row: AgentTask = ConsumerRow("row-7", "Do the thing", TaskState.TODO)

    assert canonical_task_json(row, None, None) == (
        '{"claimed_by":null,"claimed_until":null,"id":"row-7",'
        '"instruction":"Do the thing","state":"todo"}'
    )


def test_the_mixin_maps_a_records_own_columns_onto_the_contract() -> None:
    model: AgentTask = ConsumerModel(7, "Do the thing", "claimed")

    assert model.id == "row-7"
    assert model.instruction == "Do the thing"
    assert model.state is TaskState.CLAIMED
    assert canonical_task_json(model, "worker-1", 1_735_689_900) == (
        '{"claimed_by":"worker-1","claimed_until":1735689900,"id":"row-7",'
        '"instruction":"Do the thing","state":"claimed"}'
    )


def test_the_mixin_says_which_hook_is_missing() -> None:
    # The control on the mixin: a record that implements only part of the
    # mapping fails where the gap is, rather than returning something plausible.
    half = HalfBuiltModel()

    assert half.id == "row-1"

    with pytest.raises(NotImplementedError) as failure:
        _ = half.instruction

    assert "agent_task_instruction" in str(failure.value)


# -- completion authority ----------------------------------------------------


def test_the_shipped_configuration_offers_no_completion_tool() -> None:
    # An agent that can set its own task to done turns "run until the goal is
    # met" into "run until it decides it is met". Nothing registers the tool;
    # the consumer does, deliberately, or it does not exist.
    registry = ToolRegistry()

    assert "complete_task" not in registry.names()

    # The control: the tool is real and registrable, so its absence above is the
    # default being off rather than the tool not existing.
    registry.register(TaskCompletionTool(a_source(), "agent-1"))
    assert "complete_task" in registry.names()


def test_an_authorized_completion_tool_closes_the_task() -> None:
    source = a_source()
    source.add("Do the thing", "t-1")
    claimed = source.claim("agent-1")
    assert claimed is not None

    registry = ToolRegistry().register(TaskCompletionTool(source, "agent-1"))
    authorizer = ToolAuthorizer(enabled=True, call=lambda _session, tool, _args: tool.name != "no")
    tools = authorizer.allowed(a_session(), registry.resolve(["complete_task"]))

    assert tools[0].handle({"task_id": "t-1", "outcome": "done"}) == {
        "task_id": "t-1",
        "state": "done",
    }

    settled = source.find("t-1")
    assert settled is not None
    assert settled.state is TaskState.DONE


def test_a_denied_completion_tool_leaves_the_task_claimed() -> None:
    source = a_source()
    source.add("Do the thing", "t-1")
    claimed = source.claim("agent-1")
    assert claimed is not None

    registry = ToolRegistry().register(TaskCompletionTool(source, "agent-1"))
    authorizer = ToolAuthorizer(enabled=True, call=lambda _session, _tool, _args: False)
    tools = authorizer.allowed(a_session(), registry.resolve(["complete_task"]))

    with pytest.raises(HarnessError) as failure:
        tools[0].handle({"task_id": "t-1", "outcome": "done"})

    assert failure.value.code == "call_not_authorized"

    still_claimed = source.find("t-1")
    assert still_claimed is not None
    assert still_claimed.state is TaskState.CLAIMED


def test_the_completion_tool_can_only_close_the_task_its_own_agent_holds() -> None:
    """What can still be invoked, not what we happened to send.

    The source enforces this too, now that ``release()`` takes the worker. The
    tool checks first and separately, because a third party's
    ``AgentTaskSource`` cannot be made to -- see the unguarded-source tests at
    the end of this file, which are where the tool's check is actually
    observable.
    """
    source = a_source()
    source.add_many(["mine", "someone else's"], ["t-1", "t-2"])
    assert source.claim("agent-1") is not None
    assert source.claim("worker-2") is not None

    tool = TaskCompletionTool(source, "agent-1")

    with pytest.raises(HarnessError) as failure:
        tool.handle({"task_id": "t-2", "outcome": "done"})

    assert failure.value.code == "task_lease_not_held"
    # The refusal names no other worker: a tool's error comes back to the model
    # as a readable result, and the holder's identity is not its business.
    assert "worker-2" not in failure.value.message

    others = source.find("t-2")
    assert others is not None
    assert others.state is TaskState.CLAIMED

    # The control: the agent's own task closes through the same call.
    assert tool.handle({"task_id": "t-1", "outcome": "done"}) == {
        "task_id": "t-1",
        "state": "done",
    }


def test_the_completion_tool_will_not_close_an_unclaimed_task() -> None:
    source = a_source()
    source.add("Do the thing", "t-1")
    tool = TaskCompletionTool(source, "agent-1")

    with pytest.raises(HarnessError) as failure:
        tool.handle({"task_id": "t-1", "outcome": "done"})

    assert failure.value.code == "task_lease_not_held"

    # The control: claimed by this agent, the same call goes through.
    assert source.claim("agent-1") is not None
    assert tool.handle({"task_id": "t-1", "outcome": "done"}) == {
        "task_id": "t-1",
        "state": "done",
    }


def test_a_completion_tool_bound_to_no_worker_is_refused() -> None:
    with pytest.raises(HarnessError) as failure:
        TaskCompletionTool(a_source(), "")

    assert failure.value.code == "task_identifier_blank"


def test_the_completion_tool_refuses_a_task_it_cannot_find() -> None:
    source = a_source()
    tool = TaskCompletionTool(source, "agent-1")

    with pytest.raises(HarnessError) as missing:
        tool.handle({"task_id": "nope", "outcome": "done"})

    assert missing.value.code == "task_not_found"

    with pytest.raises(HarnessError) as blank:
        tool.handle({"task_id": "", "outcome": "done"})

    assert blank.value.code == "task_identifier_blank"


# -- the outcome an agent supplies -------------------------------------------
#
# The TypeScript port coerced anything that was not exactly 'failed' into DONE,
# so `{}`, `{outcome:'complete'}` and `{outcome:'DONE'}` all recorded DONE: an
# agent declaring victory by typo. This port had the same escalation reached the
# other way round -- it hardcoded DONE and IGNORED the argument, so a model
# asking in as many words for `failed` got `done` written to the record.
#
# One rule covers both: a value the agent supplies is either exactly one of the
# two outcomes or it is refused. Never coerced, never ignored, and never
# defaulted toward the more privileged answer.


#: Outcomes padded with whitespace, for the no-trimming rule.
#:
#: EVERY CODEPOINT HERE MUST BE ONE PYTHON'S OWN ``str.strip()`` REMOVES, or the
#: case proves nothing: a trimming implementation would refuse it for the same
#: reason a correct one does, and the test would pass against the bug it exists
#: to catch. ``test_the_padding_fixtures_are_adversarial_in_this_language``
#: enforces that, because it is not visible by reading the list.
#:
#: U+00A0 is here on purpose. PHP's suite used it and it was TOOTHLESS there --
#: PHP's ``trim()`` strips ASCII only, so the case passed against a trimming
#: implementation. Python's ``str.strip()`` does remove it, so the same
#: codepoint is a real test in this language and a dud in that one. U+200B is
#: NOT here for the mirror-image reason: Python does not strip it either.
PADDED_OUTCOMES = [
    " done",
    "done ",
    "\u00a0done",
    "done\u00a0",
    "\u3000done",
    "\tdone",
]


def test_the_padding_fixtures_are_adversarial_in_this_language() -> None:
    # A fixture list that quietly stops being adversarial is worse than no list,
    # because it still reads as coverage. This fails the day someone adds a
    # codepoint Python leaves alone.
    for padded in PADDED_OUTCOMES:
        assert padded.strip() in ("done", "failed"), (
            f"{padded!r} is not stripped by Python's own str.strip(), so refusing it proves "
            "nothing about whether this implementation trims"
        )
        assert padded not in ("done", "failed")


def a_held_task(source: StoreTaskSource) -> TaskCompletionTool:
    source.add("Do the thing", "t-1")
    assert source.claim("agent-1") is not None
    return TaskCompletionTool(source, "agent-1")


def test_the_completion_tool_honours_the_outcome_the_agent_asked_for() -> None:
    source = a_source()
    tool = a_held_task(source)

    assert tool.handle({"task_id": "t-1", "outcome": "failed"}) == {
        "task_id": "t-1",
        "state": "failed",
    }

    settled = source.find("t-1")
    assert settled is not None
    # `done` NOT being recorded is the security property and `failed` being
    # recorded is the assertion, and they are said separately on purpose. The
    # privileged one goes FIRST so it is a real check rather than one the type
    # checker has already narrowed away.
    assert settled.state is not TaskState.DONE
    assert settled.state is TaskState.FAILED


def test_an_absent_outcome_is_refused_exactly_like_a_malformed_one() -> None:
    """Silence is not a request to complete the task.

    This used to return ``done``, on the reasoning that an agent invoking a tool
    called ``complete_task`` had declared its intent. That is the SAME inference
    that produced the hardcoded ``done`` this tool shipped with -- reading the
    privileged outcome out of silence, moved one level up from where it was
    caught rather than removed. Same code as a malformed outcome, because it is
    the same mistake.
    """
    source = a_source()
    tool = a_held_task(source)

    with pytest.raises(HarnessError) as failure:
        tool.handle({"task_id": "t-1"})

    assert failure.value.code == "task_outcome_invalid"

    # Nothing recorded, and specifically not the privileged outcome.
    unchanged = source.find("t-1")
    assert unchanged is not None
    assert unchanged.state is not TaskState.DONE
    assert unchanged.state is TaskState.CLAIMED

    # The control: the same call with the outcome stated goes through, so the
    # refusal is about the missing field and not about the tool being broken.
    assert tool.handle({"task_id": "t-1", "outcome": "done"}) == {
        "task_id": "t-1",
        "state": "done",
    }


def test_an_agent_that_asks_for_failed_reads_back_failed() -> None:
    # The end-to-end shape the reference pins: what the agent stated is what the
    # record says, through the tool rather than through `release()`.
    source = a_source()
    tool = a_held_task(source)

    tool.handle({"task_id": "t-1", "outcome": "failed"})
    settled = source.find("t-1")

    assert settled is not None
    assert settled.state is not TaskState.DONE
    assert settled.state is TaskState.FAILED


@pytest.mark.parametrize(
    "outcome",
    [
        "DONE",
        "Done",
        "complete",
        "completed",
        "success",
        "ok",
        *PADDED_OUTCOMES,
        "",
        "todo",
        "claimed",
        None,
        True,
        1,
        0,
        {},
        [],
        ["done"],
    ],
)
def test_an_outcome_the_agent_supplies_is_never_coerced(outcome: object) -> None:
    """No value an agent controls may select an outcome without validation.

    Every one of these is refused rather than resolved. The list is mostly
    near-misses on purpose -- `'DONE'`, `'complete'`, `' done'` -- because a
    coercing implementation does not fail on obvious rubbish, it fails on the
    values a model plausibly produces while meaning something specific.

    ``None`` is in the list deliberately: present-and-null is not absent. The
    agent said something, and what it said is not an outcome.
    """
    source = a_source()
    tool = a_held_task(source)

    with pytest.raises(HarnessError) as failure:
        tool.handle({"task_id": "t-1", "outcome": outcome})

    assert failure.value.code == "task_outcome_invalid"

    # And nothing was written. A refusal that had already recorded DONE would
    # be the escalation with an exception stapled to it.
    unchanged = source.find("t-1")
    assert unchanged is not None
    assert unchanged.state is TaskState.CLAIMED


def test_the_two_real_outcomes_are_accepted_through_the_same_door() -> None:
    # The control for the parametrised refusals: if `parse` rejected
    # everything, every case above would pass and the tool would be useless.
    assert TaskOutcome.parse("done") is TaskOutcome.DONE
    assert TaskOutcome.parse("failed") is TaskOutcome.FAILED
    assert TaskOutcome.parse(TaskOutcome.DONE) is TaskOutcome.DONE
    assert TaskOutcome.parse(TaskOutcome.FAILED) is TaskOutcome.FAILED


def test_release_refuses_an_outcome_that_is_not_one() -> None:
    # The tool is not the only door. A consumer driving `release()` from a
    # decoded JSON body has no type checker in the way, and the failure there
    # was an AttributeError -- a crash rather than a code, which 0004 says a
    # consumer cannot branch on.
    source = a_source()
    source.add("Do the thing", "t-1")
    claimed = source.claim("agent-1")
    assert claimed is not None

    with pytest.raises(HarnessError) as failure:
        source.release(claimed, "agent-1", "complete")  # type: ignore[arg-type]

    assert failure.value.code == "task_outcome_invalid"

    # The control, and it draws the line deliberately: the two WIRE WORDS are
    # accepted from a bare string, because the word is what is pinned across
    # the three languages. It is `'complete'` that is refused, not the fact
    # that a string arrived.
    source.release(claimed, "agent-1", "done")  # type: ignore[arg-type]
    settled = source.find("t-1")
    assert settled is not None
    assert settled.state is TaskState.DONE


# -- only the holder may release ---------------------------------------------


def test_a_lapsed_worker_cannot_overwrite_the_live_holders_claim() -> None:
    """The sequence this argument exists for. NO ADVERSARY REQUIRED.

    A takes longer than its lease. The lease expires, B legitimately reclaims
    the task and starts work. A finishes and releases.

    Without the worker on `release()`, A's report lands on B's live claim: the
    task reads `done` while B is still working, B's work is discarded, and then
    **B's own release fails as "already terminal"** -- the second worker blamed,
    in the log line a person reads, for the first one's mistake. Every step here
    is legitimate behaviour by both workers.
    """
    clock = Clock()
    source = a_source(clock=clock, lease_seconds=60)
    source.add("Do the thing", "t-1")

    a_task = source.claim("worker-a")
    assert a_task is not None

    clock.advance(61)

    b_task = source.claim("worker-b")
    assert b_task is not None
    assert b_task.id == a_task.id
    assert b_task.claimed_by == "worker-b"

    with pytest.raises(HarnessError) as failure:
        source.release(a_task, "worker-a", TaskOutcome.DONE)

    assert failure.value.code == "task_lease_not_held"
    # The exception NAMES the holder: a developer reads this one.
    assert "worker-b" in failure.value.message

    # B's claim survived A's report untouched.
    still_bs = source.find("t-1")
    assert still_bs is not None
    assert still_bs.state is TaskState.CLAIMED
    assert still_bs.claimed_by == "worker-b"

    # And B is not blamed for A's mistake: its own release goes through, which
    # is the half of this that the "already terminal" error used to swallow.
    source.release(b_task, "worker-b", TaskOutcome.DONE)

    settled = source.find("t-1")
    assert settled is not None
    assert settled.state is TaskState.DONE


def test_release_refuses_a_blank_worker() -> None:
    source = a_source()
    source.add("Do the thing", "t-1")
    claimed = source.claim("worker-1")
    assert claimed is not None

    with pytest.raises(HarnessError) as failure:
        source.release(claimed, "", TaskOutcome.DONE)

    assert failure.value.code == "task_identifier_blank"

    # The control: the real holder releases through the same call.
    source.release(claimed, "worker-1", TaskOutcome.DONE)
    settled = source.find("t-1")
    assert settled is not None
    assert settled.state is TaskState.DONE


class UnguardedTaskSource:
    """A third party's ``AgentTaskSource``, implemented the obvious way.

    ``release()`` here is "find it, set the state" -- which is what the
    signature suggests, and which a Protocol cannot prevent anyone writing. The
    worker argument is accepted and ignored.

    This exists so the completion tool's own check is tested against a source
    that makes NO guarantees, rather than only against the one this package
    ships. Against the shipped source the tool's check is unobservable: delete
    it and every test still passes, because the source catches everything. That
    is precisely why it cannot be tested there.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, StoredTask] = {}

    def add(self, task: StoredTask) -> None:
        self.tasks[task.id] = task

    def claim(self, worker: str, lease_seconds: float | None = None) -> AgentTask | None:
        for task_id, task in self.tasks.items():
            if task.state is TaskState.TODO:
                self.tasks[task_id] = replace(
                    task, state=TaskState.CLAIMED, claimed_by=worker, claimed_until=None
                )
                return self.tasks[task_id]

        return None

    def release(self, task: AgentTask, worker: str, outcome: TaskOutcome) -> None:
        # NO CHECKS AT ALL. Not terminality, not the lease, not the worker.
        current = self.tasks[task.id]
        self.tasks[task.id] = replace(
            current, state=outcome.state(), claimed_by=None, claimed_until=None
        )

    def pending(self) -> int:
        return sum(1 for task in self.tasks.values() if task.state is TaskState.TODO)

    def find(self, task_id: str) -> AgentTask | None:
        return self.tasks.get(task_id)


class HolderlessTaskSource(UnguardedTaskSource):
    """A source whose tasks do not expose a holder AT ALL.

    Perfectly legal: :class:`AgentTask` is ``id``, ``instruction`` and
    ``state``, and nothing on the contract promises ``claimed_by``. A consumer
    keeping the lease in columns this package has never heard of produces
    exactly this.
    """

    def find(self, task_id: str) -> AgentTask | None:
        if task_id not in self.tasks:
            return None

        return ConsumerRow(task_id, "Do the thing", TaskState.CLAIMED)


def test_the_unguarded_fixture_really_is_unguarded() -> None:
    # The fixture has to be checked before it can be used as a control. If it
    # quietly enforced something, the two tests below would prove nothing about
    # the tool -- they would be measuring the fixture.
    source = UnguardedTaskSource()
    source.add(StoredTask("t-1", "Do the thing", TaskState.CLAIMED, "someone-else"))

    contract: AgentTaskSource = source
    contract.release(source.tasks["t-1"], "not-the-holder", TaskOutcome.DONE)

    assert source.tasks["t-1"].state is TaskState.DONE


def test_the_tool_refuses_a_task_another_worker_holds_even_on_a_naive_source() -> None:
    # The guarantee the CONTRACT makes, tested against an implementation that
    # does not make it. This is where the tool's own check earns its place.
    source = UnguardedTaskSource()
    source.add(StoredTask("t-1", "Do the thing", TaskState.CLAIMED, "someone-else"))
    tool = TaskCompletionTool(source, "agent-1")

    with pytest.raises(HarnessError) as failure:
        tool.handle({"task_id": "t-1", "outcome": "done"})

    assert failure.value.code == "task_lease_not_held"
    # The tool's refusal NAMES NOBODY -- a model reads this one.
    assert "someone-else" not in failure.value.message
    assert source.tasks["t-1"].state is TaskState.CLAIMED


def test_the_tool_refuses_when_the_holder_cannot_be_established() -> None:
    # Unknowable is not permission. The task says `claimed` and offers no way to
    # learn by whom, so the tool fails closed -- the same shape as refusing an
    # absent outcome rather than reading it as `done`.
    source = HolderlessTaskSource()
    source.add(StoredTask("t-1", "Do the thing", TaskState.CLAIMED, "agent-1"))
    tool = TaskCompletionTool(source, "agent-1")

    with pytest.raises(HarnessError) as failure:
        tool.handle({"task_id": "t-1", "outcome": "done"})

    assert failure.value.code == "task_lease_not_held"
    assert source.tasks["t-1"].state is TaskState.CLAIMED


def test_the_tool_still_closes_a_task_the_worker_really_holds_on_a_naive_source() -> None:
    # The control for both refusals above. Without it they would both pass on a
    # tool that refused everything, which would be a guard that works by being
    # useless.
    source = UnguardedTaskSource()
    source.add(StoredTask("t-1", "Do the thing", TaskState.TODO))
    assert source.claim("agent-1") is not None

    tool = TaskCompletionTool(source, "agent-1")

    assert tool.handle({"task_id": "t-1", "outcome": "failed"}) == {
        "task_id": "t-1",
        "state": "failed",
    }
    assert source.tasks["t-1"].state is TaskState.FAILED
