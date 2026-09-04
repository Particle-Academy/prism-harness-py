"""Agent task lists -- what remains, who holds it, and when the hold lapses.

An agent given a goal works across many requests until the goal is met, and the
list of what remains has to survive the request, the worker, a crash and a
deploy. The easy part -- what a task IS -- belongs to the consumer and is not
decided here: this module ships two contracts and two adapters, no model, no
schema and no migration.

The four-state machine, the claim rules, the ordering and the canonical record
are IDENTITY: they are pinned in `prism-parity/specs/agent-task-lists.md` and
must not vary between PHP, TypeScript and Python. Everything else in this file
-- snake_case, Protocols instead of interfaces, dataclasses, the synchronous
store contract -- is spelling, per prism-parity decision 0002.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from prism_harness.errors import HarnessError
from prism_harness.stores.base import SessionStore, T
from prism_harness.subagents import RunBudget, RunLedger

__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "AgentTask",
    "AgentTaskMixin",
    "AgentTaskSource",
    "StoreTaskSource",
    "StoredTask",
    "TaskCompletionTool",
    "TaskOutcome",
    "TaskState",
    "canonical_task_json",
    "task_record",
]

#: How long a claim holds a task before anyone may take it again.
#:
#: FIVE MINUTES, and configurable. Long enough for a model call plus tool work,
#: short enough that a crashed worker does not wedge the list for an hour. The
#: number matters less than it being the same number in all three languages, so
#: it is pinned in the spec rather than chosen per port.
#:
#: A WHOLE number, spelled as one. A lease has to be a positive integer of
#: seconds -- see :func:`_require_lease` -- and a shipped default written as a
#: float invites a consumer to derive a fractional one from it.
DEFAULT_LEASE_SECONDS = 300

#: Tells ABSENT from present-and-null, which 0002 makes an observable decision.
#: An argument a model did not send and an argument it sent as null are
#: different statements, and only the first has a safe default.
_MISSING = object()


class TaskState(str, Enum):
    """The four states a task may be in. THERE ARE NO OTHERS.

    ::

            claim()                    release(done)
      todo ---------> claimed -------------------------> done
       ^                 |            release(failed)
       |                 +-------------------------------> failed
       |                 |
       +-----------------+
          lease expires

    An expired lease returns a task to :attr:`TODO`, NEVER to :attr:`FAILED`. A
    worker dying is not the task failing, and conflating them burns a retry that
    never ran.
    """

    TODO = "todo"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """``done`` and ``failed`` are terminal. Re-releasing one is an error."""
        return self is TaskState.DONE or self is TaskState.FAILED


class TaskOutcome(str, Enum):
    """What :meth:`AgentTaskSource.release` records.

    Deliberately NOT :class:`TaskState`. Only two of the four states are
    reachable through ``release``, and a parameter that accepts all four invites
    a caller to release something back to ``todo`` -- which is the lease's job,
    not a caller's, and which would let a failed task re-queue itself.
    """

    DONE = "done"
    FAILED = "failed"

    def state(self) -> TaskState:
        return TaskState.DONE if self is TaskOutcome.DONE else TaskState.FAILED

    @classmethod
    def parse(cls, value: object) -> TaskOutcome:
        """Exactly one of the two, or :class:`HarnessError`. NO COERCION.

        The one rule: **a value that cannot be read as an outcome is refused,
        never resolved.** Resolving it always means choosing an outcome nobody
        asked for, and the choice a lenient implementation makes is invariably
        the privileged one -- "anything that is not ``failed`` is ``done``"
        reads as tidy defaulting right up until a model writes ``'complete'``
        and closes a task it never finished. `prism-harness-ts` shipped that;
        this port had the same escalation by ignoring the argument instead.

        So, deliberately: no case folding (``'DONE'`` is refused), no trimming
        (``' done'`` is refused -- and each language's own ``trim`` strips a
        different codepoint set anyway, which is G-36's lesson), no truthiness,
        and no default. ``None`` is refused rather than treated as absent,
        because present-and-null is not absent (0002) -- the caller said
        something, and what it said is not an outcome.
        """
        if isinstance(value, cls):
            return value

        # A plain string comparison against the two wire words. `TaskState` is
        # also a str enum, so `TaskState.DONE` passes here and `TaskState.TODO`
        # does not -- which is the right answer both times: the wire word is
        # what is pinned, and `todo` is not an outcome.
        if isinstance(value, str) and not isinstance(value, bool):
            for outcome in cls:
                if value == outcome.value:
                    return outcome

        raise HarnessError.task_outcome_invalid(value)


class AgentTask(Protocol):
    """One unit of work.

    STRUCTURAL, not an import. A consumer's own record -- an ORM model, a
    dataclass, a row wrapper -- IS an ``AgentTask`` the moment it exposes these
    three members: no inheritance, no registration, nothing to install. That is
    Python's answer to the trait the PHP reference mixes into an Eloquent model,
    and per 0002 the difference is spelling.

    :class:`AgentTaskMixin` exists for the records that need the mapping written
    down, not because conformance requires a base class.
    """

    @property
    def id(self) -> str:
        """Stable, and unique within its source."""
        ...

    @property
    def instruction(self) -> str:
        """What the model is asked to do."""
        ...

    @property
    def state(self) -> TaskState: ...


class AgentTaskSource(Protocol):
    """Where tasks come from.

    Four methods. :meth:`claim` is the reason this package has a task list at
    all, and :meth:`find` is the reason the other three can be reached from
    outside the loop.
    """

    def claim(self, worker: str, lease_seconds: float | None = None) -> AgentTask | None:
        """Atomically take the next available task, or None when there is none.

        ONE CALL, deliberately. "Read the next task" followed by "mark it mine"
        is two calls with a window between them, and two workers in that window
        both get the same task. Every implementation of this method does the
        read and the write inside one lock.
        """
        ...

    def release(self, task: AgentTask, worker: str, outcome: TaskOutcome) -> None:
        """Record what happened. Terminal, and re-releasing is an error.

        THE WORKER IS AN ARGUMENT, and it is the argument that stops a lapsed
        holder overwriting a live one. No adversary is required to reach that:

        1. Worker A claims a task and takes longer than its lease.
        2. The lease expires, the task returns to ``todo``, and worker B
           legitimately claims it and starts work.
        3. A finishes and calls ``release``.

        Without the worker, step 3 succeeds. The task reads ``done`` while B is
        still working, B's work is thrown away, and then **B's own release fails
        as "already terminal"** -- so the second worker is blamed for the first
        one's mistake, in the log line a person will read.

        This lived only in the completion tool at first, which was the wrong
        place: a guard living in one tool leaves every other caller able to do
        what the guard forbids -- a queued job, an HTTP route, a direct call.
        """
        ...

    def pending(self) -> int:
        """How many tasks remain CLAIMABLE.

        A COUNT, not a listing. It exists to terminate the loop and a count is
        enough for that; a listing invites the source to materialise every task
        on every pass, and a consumer that wants one already has its own query.
        """
        ...

    def find(self, task_id: str) -> AgentTask | None:
        """One task by id, or None when this source does not hold it.

        ON THE CONTRACT, because without it the contract cannot be driven from
        outside the claim loop. :meth:`release` takes a TASK, and every external
        caller -- a tool, an HTTP route, a queue worker resuming after a restart
        -- holds only an ID. This package's own completion tool needed it before
        anyone else did, which is the tell: a method a shipped consumer cannot
        work without does not belong on the concrete class only.

        Still one task by id, and still not a listing. That distinction is what
        keeps :meth:`pending` honest.
        """
        ...


@dataclass(frozen=True)
class StoredTask:
    """The record :class:`StoreTaskSource` hands out.

    Frozen because it is a SNAPSHOT. The authority is the store, and a mutable
    copy in a worker's hand is exactly the thing that gets acted on after the
    lease it describes has lapsed.
    """

    id: str
    instruction: str
    state: TaskState
    #: The worker holding the claim, or None. PRESENT-AND-NULL, never absent.
    claimed_by: str | None = None
    #: An INTEGER Unix timestamp, or None. Not a formatted date -- date
    #: formatting is exactly where three languages produce three strings from
    #: one instant.
    claimed_until: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return task_record(self, self.claimed_by, self.claimed_until)

    def to_canonical_json(self) -> str:
        return canonical_task_json(self, self.claimed_by, self.claimed_until)


class AgentTaskMixin:
    """A consumer's own record, adapted to :class:`AgentTask`.

    PHP spells this as a trait on an Eloquent model, under ``src/Concerns/``.
    Python's equivalent is a mixin -- and the shorter answer above it, worth
    saying before anyone reaches for this class: :class:`AgentTask` is a
    Protocol, so a record that already exposes ``id``, ``instruction`` and
    ``state`` conforms with no base class at all. For that consumer the adapter
    is nothing.

    This is for the other consumer: a record whose columns are named something
    else, or whose state arrives as a raw string out of a database. Implement
    the three hooks and the contract members are derived from them.

    **The hooks are not named** ``id`` / ``instruction`` / ``state``, and that
    is not cosmetic: a property that read the same-named attribute off ``self``
    would recurse into itself the moment a host stored the value under the
    obvious name. Two names, no recursion.

    **A plain class, not an ABC.** ``abc.ABCMeta`` in the bases collides with
    the metaclass of every ORM worth mixing this into -- Django's ``ModelBase``,
    SQLAlchemy's declarative base -- and the whole point of this class is that
    it goes onto a record someone else defined.
    """

    def agent_task_id(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} mixes in AgentTaskMixin but does not implement "
            "agent_task_id(). It must return an id that is stable and unique within its source."
        )

    def agent_task_instruction(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} mixes in AgentTaskMixin but does not implement "
            "agent_task_instruction(). It must return what the model is asked to do."
        )

    def agent_task_state(self) -> TaskState:
        raise NotImplementedError(
            f"{type(self).__name__} mixes in AgentTaskMixin but does not implement "
            "agent_task_state(). Map your own column onto one of the four states -- "
            "TaskState(value) does it when the column already stores the same words."
        )

    @property
    def id(self) -> str:
        return self.agent_task_id()

    @property
    def instruction(self) -> str:
        return self.agent_task_instruction()

    @property
    def state(self) -> TaskState:
        return self.agent_task_state()


def task_record(
    task: AgentTask, claimed_by: str | None, claimed_until: int | None
) -> dict[str, Any]:
    """The canonical dict for one task record.

    Five keys, IN SORTED ORDER, and ``claimed_by`` / ``claimed_until`` are
    present-and-null when unclaimed rather than absent -- 0002 makes absent
    versus null an observable decision, and a port modelling unset as
    ``undefined`` would drop the keys.

    The claim fields are PARAMETERS rather than read off ``task``, because
    :class:`AgentTask` does not carry them: a lease belongs to the source that
    granted it, and a consumer's own record may keep it in columns this package
    has never heard of.
    """
    return {
        "claimed_by": claimed_by,
        "claimed_until": claimed_until,
        "id": task.id,
        "instruction": task.instruction,
        # Coerced rather than trusted: a consumer's adapter can hand back a raw
        # column value, and a fifth state reaching the store would break the
        # machine everywhere downstream of it.
        "state": TaskState(task.state).value,
    }


def canonical_task_json(task: AgentTask, claimed_by: str | None, claimed_until: int | None) -> str:
    """One task record as canonical JSON -- the same bytes in all three languages.

    ``json.dumps`` DEFAULTS ARE WRONG HERE, in two separate ways, and neither is
    visible in the output of a single language:

    * ``separators`` defaults to ``(", ", ": ")``, which pads every separator
      with a space that PHP and JavaScript do not emit.
    * ``ensure_ascii`` defaults to True, which escapes every non-ASCII character
      to ``\\uXXXX`` where the other two pass it through.

    So both are set explicitly, exactly as ``prism-py``'s canonical encoder does
    for the same reason. ``sort_keys`` is belt to :func:`task_record`'s braces:
    the literal above is already in sorted order, and this makes a later edit
    that reorders it a no-op rather than a divergence.
    """
    return json.dumps(
        task_record(task, claimed_by, claimed_until),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


class StoreTaskSource:
    """The default source: tasks in the harness's own DURABLE session store.

    No schema, no migration, works on install. A consumer with an existing task
    table adapts it instead -- see :class:`AgentTaskMixin` -- and both satisfy
    :class:`AgentTaskSource`.

    **It refuses to start on a volatile store.** The list is durable state, not
    a cache: a half-finished task list that vanishes on a deploy is
    indistinguishable from a finished one, so losing it is a correctness failure
    rather than a degradation to a default. This package already draws that line
    with :class:`~prism_harness.stores.base.Durability` and already refuses a
    volatile driver for the durable slot; this inherits the rule rather than
    inventing a second one, and fires it at CONSTRUCTION so a misconfiguration
    cannot lie dormant until the first claim.

    **Every mutation happens inside the store's lock**, and the read that
    decides it happens inside the same lock. That is what makes :meth:`claim`
    one atomic call rather than a read followed by a mark.

    **Ordering is insertion order.** Tasks are a list and stay a list; nothing
    here sorts, re-ranks or shuffles. Ordering is the divergence class 0002
    calls hardest to notice -- nothing errors when it changes, the agent simply
    does the work in a different sequence and produces a different result.
    """

    def __init__(
        self,
        store: SessionStore,
        key: str,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        clock: Callable[[], float] = time.time,
        wait_seconds: float = 5,
    ) -> None:
        if not store.durability().is_durable():
            raise HarnessError.volatile_task_source(type(store).__name__)

        self._store = store
        self._key = key
        self._lease_seconds = _require_lease(lease_seconds)
        #: Wall clock, injectable so a test can move time without sleeping.
        #: ``time.time`` and not ``time.monotonic``: ``claimed_until`` is a Unix
        #: timestamp that another process on another machine has to compare
        #: against, and a monotonic clock is meaningless across processes.
        self._clock = clock
        self._wait_seconds = wait_seconds

    # -- filling the list --------------------------------------------------

    def add(self, instruction: str, task_id: str | None = None) -> StoredTask:
        """Append one task, in ``todo``.

        Not part of :class:`AgentTaskSource`: how tasks arrive is the
        consumer's, and a source over someone else's table has its own answer.
        This one needs a way in, so it has one.
        """
        return self.add_many([instruction], None if task_id is None else [task_id])[0]

    def add_many(
        self, instructions: Sequence[str], task_ids: Sequence[str] | None = None
    ) -> list[StoredTask]:
        """Append several tasks, in order, INSIDE ONE LOCK.

        One lock rather than a loop of :meth:`add`, so a batch either lands
        whole or not at all and nothing interleaves into the middle of it.
        """
        if task_ids is not None and len(task_ids) != len(instructions):
            # A caller's mistake, not a domain failure, so it is a plain
            # ValueError rather than a HarnessError with a code nobody would
            # ever branch on.
            raise ValueError(
                f"{len(instructions)} instruction(s) were given {len(task_ids)} id(s)."
            )

        def append() -> list[StoredTask]:
            stored, records = self._read()
            added: list[StoredTask] = []

            for index, instruction in enumerate(instructions):
                task_id = task_ids[index] if task_ids is not None else f"t-{len(records) + 1}"
                _require_identifier(task_id, "task")

                if any(record.get("id") == task_id for record in records):
                    raise HarnessError.duplicate_task_id(task_id)

                record = {
                    "claimed_by": None,
                    "claimed_until": None,
                    "id": task_id,
                    "instruction": instruction,
                    "state": TaskState.TODO.value,
                }
                records.append(record)
                added.append(_to_task(record))

            self._write(stored, records)
            return added

        return self._locked(append)

    # -- the contract ------------------------------------------------------

    def claim(self, worker: str, lease_seconds: float | None = None) -> StoredTask | None:
        """Take the next claimable task, atomically, and lease it to ``worker``.

        THE WRITE HAPPENS BEFORE THIS RETURNS, which is what makes "started and
        died" distinguishable from "never started": a worker that crashes after
        claiming leaves a ``claimed`` record with an expiry, and a worker that
        never got that far leaves a ``todo`` one. Marking the task after the
        work would make a crash look like a task nobody ever attempted.

        Expired claims are returned to ``todo`` on the way past -- never to
        ``failed`` -- so a dead worker's task is picked up by whoever asks next.
        """
        _require_identifier(worker, "worker")
        lease = self._lease_seconds if lease_seconds is None else _require_lease(lease_seconds)

        def take() -> StoredTask | None:
            stored, records = self._read()
            now = self._clock()
            expired = _expire(records, now)

            target = next(
                (record for record in records if record.get("state") == TaskState.TODO.value),
                None,
            )

            if target is None:
                # Nothing to hand out, but the expiries computed above are real
                # and are worth persisting: the next reader should not have to
                # rediscover them.
                if expired:
                    self._write(stored, records)
                return None

            target["state"] = TaskState.CLAIMED.value
            target["claimed_by"] = worker
            target["claimed_until"] = _timestamp(now + lease)
            self._write(stored, records)

            return _to_task(target)

        return self._locked(take)

    def release(self, task: AgentTask, worker: str, outcome: TaskOutcome) -> None:
        """Record what a WORKER found. TERMINAL.

        Called by the APPLICATION, from evidence -- not by the agent. If the
        model can set its own task to ``done`` then "run until the goal is met"
        silently becomes "run until it decides it is met", and a stalled run
        ends by declaring victory. See :class:`TaskCompletionTool` for the
        explicitly authorized way to hand that over.

        Re-releasing a terminal task RAISES rather than quietly doing nothing: a
        silent no-op there is a second worker's evidence being discarded without
        anyone finding out.

        **Only the worker currently holding the lease may release**, and this
        check lives HERE, next to the rest of the state machine, rather than in
        the one tool that first needed it. A guard living in a tool leaves every
        other caller able to do what the guard forbids -- a queued job, an HTTP
        route, a direct call -- and this package ships all three shapes of
        caller.

        The sequence it stops needs no adversary. A's lease lapses mid-task; B
        legitimately reclaims and starts work; A finishes and releases. Without
        the check A overwrites B's live claim, the task reads ``done`` while B
        is still working, B's work is discarded, and B's own release then fails
        as "already terminal" -- the second worker blamed for the first one's
        mistake.

        A task whose lease has expired is ``todo`` again and cannot be released
        by anyone, which is the same rule seen from the other side.
        """
        _require_identifier(worker, "worker")
        # Parsed rather than trusted. A typed caller is already protected by the
        # signature, but a consumer driving this from a decoded JSON body has no
        # type checker in the way -- and the failure there was an AttributeError
        # rather than a code, which 0004 says a consumer cannot branch on.
        outcome = TaskOutcome.parse(outcome)
        task_id = task.id

        def settle() -> None:
            stored, records = self._read()
            _expire(records, self._clock())
            record = self._locate(records, task_id)
            state = _state_of(record)

            if state.is_terminal():
                raise HarnessError.task_already_terminal(task_id, state.value)

            if state is not TaskState.CLAIMED:
                raise HarnessError.task_lease_not_held(
                    task_id,
                    "it is not claimed -- either it was never claimed, or the lease expired and "
                    "the task returned to todo",
                )

            holder = record.get("claimed_by")

            if holder != worker:
                # NAMES THE HOLDER, deliberately, and note the contrast with
                # `TaskCompletionTool`, which refuses the same fact and names
                # nobody. A developer reads this one and needs to know who
                # actually has the lease; a MODEL reads the tool's, and another
                # worker's identity is not the model's business. Same code, two
                # audiences.
                raise HarnessError.task_lease_not_held(
                    task_id, f"it is held by [{holder}], not by [{worker}]"
                )

            record["state"] = outcome.state().value
            # The lease is over either way, so it is cleared rather than left
            # pointing at a moment nothing will act on.
            record["claimed_by"] = None
            record["claimed_until"] = None
            self._write(stored, records)

        self._locked(settle)

    def pending(self) -> int:
        """How many tasks are claimable RIGHT NOW.

        Counts ``todo``, and counts a ``claimed`` task whose lease has lapsed,
        because that task is ``todo`` by the state machine. Does NOT count a
        live claim: it is not claimable, and a loop that treated it as remaining
        work would spin while another worker held it.
        """
        _stored, records = self._read()
        _expire(records, self._clock())

        return sum(1 for record in records if record.get("state") == TaskState.TODO.value)

    def find(self, task_id: str) -> StoredTask | None:
        """One task by id, or None. Same snapshot caveat as :meth:`records`."""
        return next((task for task in self.records() if task.id == task_id), None)

    # -- beyond the contract -----------------------------------------------

    def extend_lease(
        self,
        task: AgentTask,
        worker: str,
        ledger: RunLedger,
        budget: RunBudget,
        lease_seconds: float | None = None,
    ) -> StoredTask:
        """Push this worker's lease out, BOUNDED BY THE RUN'S REMAINING TIME.

        Two halves, and both are load-bearing:

        **Only while it still holds it.** A worker whose lease lapsed cannot
        take it back by extending; the task is ``todo`` and someone else may
        already have it.

        **Bounded by the run's remaining WALL-CLOCK budget**, which is
        :meth:`RunLedger.remaining_seconds` against the existing
        :class:`RunBudget` -- not a second timeout invented here. Unbounded
        self-extension is how a wedged worker holds a task forever, and a fresh
        limit alongside the budget is the duplicated bound this ecosystem keeps
        finding set in the place that is not enforced. Extension stops when the
        run's own allowance does, so there is nothing new to enforce and nothing
        new to forget to enforce.

        A budget with no wall-clock cap bounds nothing here, deliberately. The
        operator declined to set the limit; inventing one on their behalf is the
        second timeout this is avoiding.

        The new expiry is ``now + granted`` even in the corner where that is
        EARLIER than the current one -- reached when the lease was longer than
        the run's whole allowance. A worker may not hold a task past the point
        its run must stop, and shortening the lease is what lets the next worker
        pick the task up promptly rather than waiting out a lease nobody can
        legitimately use.
        """
        _require_identifier(worker, "worker")
        lease = self._lease_seconds if lease_seconds is None else _require_lease(lease_seconds)

        # EXHAUSTION FIRST, and not only the wall clock. A run that has been
        # cancelled, or has spent its steps or its money, may not take another
        # step -- so it may not hold a task open waiting to take one either. The
        # clock was the only thing checked here at first, which left a cancelled
        # worker extending its lease indefinitely: the loop it was extending FOR
        # would refuse to run, and the task stayed locked away from every worker
        # that could still do it.
        exhausted = ledger.exhaustion(budget)

        if exhausted is not None:
            raise HarnessError.run_not_permitted(
                f"The lease on task [{task.id}] cannot be extended: {exhausted}. The lease is "
                "bounded by the run, and a run that may not spend again may not keep holding "
                "the task either."
            )

        remaining = ledger.remaining_seconds(budget)

        # Still checked separately, and not redundant with the above:
        # `remaining_seconds` truncates, so it reaches 0 while `exhaustion` is
        # still None -- 59.5 seconds into a 60 second budget there is half a
        # second left and no whole second to grant.
        if remaining is not None and remaining <= 0:
            raise HarnessError.run_not_permitted(
                f"The lease on task [{task.id}] cannot be extended: the run's wall-clock budget "
                f"of {budget.max_seconds}s has nothing left. The lease is bounded by the run, so "
                "a run that must stop cannot keep holding the task."
            )

        granted = lease if remaining is None else min(lease, float(remaining))
        task_id = task.id

        def hold() -> StoredTask:
            stored, records = self._read()
            now = self._clock()
            _expire(records, now)
            record = self._locate(records, task_id)
            state = _state_of(record)

            if state.is_terminal():
                raise HarnessError.task_already_terminal(task_id, state.value)

            if state is not TaskState.CLAIMED:
                raise HarnessError.task_lease_not_held(
                    task_id, "the lease expired and the task returned to todo"
                )

            if record.get("claimed_by") != worker:
                raise HarnessError.task_lease_not_held(
                    task_id, f"it is held by [{record.get('claimed_by')}], not by [{worker}]"
                )

            record["claimed_until"] = _timestamp(now + granted)
            self._write(stored, records)

            return _to_task(record)

        return self._locked(hold)

    def records(self) -> list[StoredTask]:
        """Every task, in insertion order, with expired claims shown as ``todo``.

        For inspection and serialisation, and deliberately NOT on
        :class:`AgentTaskSource` -- unlike :meth:`find`, which is. The line
        between them is the one the contract cares about: a count and a single
        task by id are questions every source can answer cheaply, and a LISTING
        invites every source to materialise everything on every pass.
        """
        _stored, records = self._read()
        _expire(records, self._clock())

        return [_to_task(record) for record in records]

    # -- internals ---------------------------------------------------------

    def _locked(self, callback: Callable[[], T]) -> T:
        return self._store.with_lock(self._key, callback, wait_seconds=self._wait_seconds)

    def _read(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        stored = self._store.get(self._key) or {}
        entries = stored.get("tasks")

        if not isinstance(entries, list):
            return stored, []

        # Copied out, so mutating a record cannot reach back into a store that
        # handed out a live reference. Only the in-memory driver could do that,
        # and it is refused here anyway -- but a consumer's own durable driver
        # is someone else's code.
        return stored, [dict(entry) for entry in entries if isinstance(entry, dict)]

    def _write(self, stored: dict[str, Any], records: list[dict[str, Any]]) -> None:
        self._store.put(self._key, {**stored, "tasks": records})

    @staticmethod
    def _locate(records: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
        record = next((entry for entry in records if entry.get("id") == task_id), None)

        if record is None:
            raise HarnessError.task_not_found(task_id)

        return record


class TaskCompletionTool:
    """The explicitly authorized way to let an agent close its own task.

    **NOTHING REGISTERS THIS.** It is not in a mode's default toolset, not in a
    registry this package builds, and not reachable by an agent that has not
    been handed it deliberately. That is the whole mechanism: the default is
    off in all three languages, and a port that ships it on has changed an
    observable decision.

    A consumer that wants it registers it on their own
    :class:`~prism_harness.tools.ToolRegistry` and gates it through the existing
    :class:`~prism_harness.tools.ToolAuthorizer`. NO NEW PERMISSION MECHANISM --
    a second one would be a second place to get the answer wrong, and the
    authorizer already asks both questions this needs: may the run be offered
    this tool, and may THIS call proceed.

    Worth being plain about why it is off: if the model can set its own task to
    ``done``, "run until the goal is met" silently becomes "run until it decides
    it is met", and a run that has stalled ends by declaring victory. It is the
    same failure ``prism-human-plus`` addresses by reserving confirmation for
    the human.

    **ITS OWN task, and the worker is required.** The authoritative check is on
    :meth:`AgentTaskSource.release`, where it belongs -- a guard living in one
    tool leaves a queued job, an HTTP route and a direct call able to do what
    the guard forbids.

    **This tool checks anyway, and that is not redundant.** Against the source
    this package ships, deleting the check below changes nothing observable --
    a mutation run confirms it survives. It earns its place against the sources
    this package does NOT ship: a Protocol cannot make an implementation check
    anything, and a third party writing their own ``release()`` will write
    "find it, set the state", because that is what the signature suggests. The
    tool is the one caller here that hands a MODEL's request to someone else's
    code, so it verifies before it delegates rather than assuming the contract
    was honoured.

    Being defensive at that seam has a consequence worth stating: through
    :class:`AgentTaskSource`, :meth:`find` returns an :class:`AgentTask`, which
    carries ``id``, ``instruction`` and ``state`` and NOT the holder. So the
    tool may be unable to establish who holds a task at all -- and when it
    cannot, it refuses. Failing closed on an unanswerable question is the only
    safe direction when the answer decides whether an agent may close work.
    """

    def __init__(self, source: AgentTaskSource, worker: str) -> None:
        _require_identifier(worker, "worker")
        # Typed to the CONTRACT, not to this package's source. A consumer with
        # their own task table gets the same tool, and the guarantees this tool
        # depends on then have to be ones the contract actually makes.
        self._source = source
        self._worker = worker

    @property
    def name(self) -> str:
        return "complete_task"

    def handle(self, args: dict[str, Any]) -> Any:
        task_id = args.get("task_id")

        if not isinstance(task_id, str):
            raise HarnessError.task_identifier_blank("task")

        _require_identifier(task_id, "task")
        task = self._source.find(task_id)

        if task is None:
            raise HarnessError.task_not_found(task_id)

        if task.state is not TaskState.CLAIMED or self._holder_of(task) != self._worker:
            # NAMES NOBODY, and that is the difference from the same refusal on
            # the source. This message goes back to a MODEL as a readable tool
            # result; another worker's identity is not the model's business. The
            # source raises the same code with the holder named, because a
            # DEVELOPER reads that one.
            raise HarnessError.task_lease_not_held(task_id, "this agent is not holding it")

        outcome = self._outcome(args)
        self._source.release(task, self._worker, outcome)

        return {"task_id": task_id, "state": outcome.state().value}

    @staticmethod
    def _holder_of(task: AgentTask) -> object:
        """Who holds ``task``, or a sentinel that matches NO worker.

        :class:`AgentTask` carries ``id``, ``instruction`` and ``state`` -- not
        the holder -- so against a source outside this package the holder may
        genuinely be unknowable. Unknowable resolves to ``_MISSING``, which
        compares equal to no worker id, so the caller refuses.

        FAILING CLOSED on an unanswerable question, rather than reading silence
        as permission. It is the same mistake as inferring ``done`` from an
        absent outcome, asked about a different field.
        """
        return getattr(task, "claimed_by", _MISSING)

    @staticmethod
    def _outcome(args: dict[str, Any]) -> TaskOutcome:
        """Which outcome the agent asked for. IT HAS TO ASK.

        Two cases now, and they answer the same way:

        * **Present.** Parsed strictly, and HONOURED. Silently ignoring it is
          the same escalation as coercing it: a model asking in as many words
          for ``failed`` and getting ``done`` on the record. This port did
          exactly that until it was caught.
        * **Absent, or present and unreadable.** Refused, with one code.

        Absent used to mean ``done`` here, on the reasoning that an agent
        invoking a tool called ``complete_task`` had declared its intent. The
        reference overruled that, and it was right: **that is the same
        inference that produced the hardcoded** ``done`` **this tool shipped
        with** -- reading the privileged outcome out of silence, moved one level
        up from where it was caught rather than removed. An agent that omitted
        the field has not stated an outcome.

        ``_MISSING`` survives the change so the two cases can say different
        things to the caller. It no longer changes WHICH failure happens, only
        how it reads: absent and present-and-null are now both refused, and
        under 0002 that makes them no longer observably different.
        """
        supplied = args.get("outcome", _MISSING)

        if supplied is _MISSING:
            raise HarnessError.task_outcome_not_supplied()

        return TaskOutcome.parse(supplied)


def _expire(records: Iterable[dict[str, Any]], now: float) -> bool:
    """Return every lapsed claim to ``todo``. Reports whether anything moved.

    ``todo`` and never ``failed``: a worker dying is not the task failing, and
    marking it failed would burn a retry that never ran while also telling the
    application something untrue about the work.

    The comparison is ``<=``, so a lease is over AT its expiry rather than one
    tick after. An observable decision, and pinned that way in all three.
    """
    moved = False

    for record in records:
        if record.get("state") != TaskState.CLAIMED.value:
            continue

        until = record.get("claimed_until")

        if isinstance(until, bool) or not isinstance(until, (int, float)):
            continue

        if until <= now:
            record["state"] = TaskState.TODO.value
            record["claimed_by"] = None
            record["claimed_until"] = None
            moved = True

    return moved


def _to_task(record: dict[str, Any]) -> StoredTask:
    claimed_by = record.get("claimed_by")
    claimed_until = record.get("claimed_until")

    return StoredTask(
        id=str(record.get("id", "")),
        instruction=str(record.get("instruction", "")),
        state=_state_of(record),
        claimed_by=claimed_by if isinstance(claimed_by, str) else None,
        claimed_until=(
            int(claimed_until)
            if isinstance(claimed_until, (int, float)) and not isinstance(claimed_until, bool)
            else None
        ),
    )


def _state_of(record: dict[str, Any]) -> TaskState:
    try:
        return TaskState(record.get("state"))
    except ValueError as error:
        # A fifth state in the store is not something to guess at. Four states,
        # no others, and a record that says otherwise is corrupt.
        raise HarnessError.unmappable_content(
            f"the stored task [{record.get('id')}] is in the state "
            f"[{record.get('state')!r}], which is not one of "
            f"{', '.join(state.value for state in TaskState)}"
        ) from error


def _require_lease(seconds: float) -> float:
    """A lease must be a POSITIVE WHOLE number of seconds.

    REFUSED, never clamped, never truncated, at every door a lease arrives
    through. TypeScript's port clamped a non-positive lease up to one second;
    this refuses, because a clamped value is a configuration that silently
    became a different configuration, and this repository has already shipped
    one of those and stayed green for its whole life.

    The direction matters too. A zero or negative lease is not merely a strange
    number: ``claimed_until`` lands in the past, so the claim expires the instant
    it is granted and the very next caller steals it. Two workers on one task,
    from a config value nobody was told was wrong.

    **Fractional is the same rule one scale down, and it was missed here.**
    ``90.4`` cannot be honoured as written -- ``claimed_until`` is an integer
    timestamp in all three languages -- so accepting it means granting 90 and
    saying nothing, which is the identical silent configuration change. It is
    not saved by landing in the "safe" direction either: that is the clamping
    argument restated, and it was not enough for zero. Found by
    ``suites/agent-task-claim`` atc-0017, where the reference could not even ask
    the question (``claim()`` declares ``?int``) and TypeScript already refused.

    Non-finite is checked as well, and separately. ``nan <= 0`` is FALSE, so a
    NaN lease sails through a bare positivity check and then explodes in
    ``int(now + nan)`` -- a crash rather than a code, several frames from the
    value that caused it.
    """
    try:
        # OverflowError as well as TypeError: an int too large for a float --
        # 10 ** 400 out of a decoded JSON body -- raises here rather than
        # returning False, and a crash is not a code a consumer can branch on.
        finite = math.isfinite(seconds)
    except (TypeError, OverflowError):
        # An untyped caller -- a decoded JSON body, a config file -- gets a code
        # rather than a TypeError, for the same reason the outcome is parsed.
        raise HarnessError.task_lease_invalid(seconds) from None

    if not finite or seconds <= 0 or not float(seconds).is_integer():
        raise HarnessError.task_lease_invalid(seconds)

    return float(seconds)


def _require_identifier(value: str, kind: str) -> None:
    """Refuse a blank worker or task id.

    Not pedantry. ``""`` is FALSY in PHP and truthy-adjacent nowhere useful, so
    a blank ``claimed_by`` reads as "unclaimed" to a reference implementation
    testing the field for truth -- a task that is held and looks free, in the
    one place this design exists to make unambiguous.

    Compared against ``""`` exactly, with no trimming. Each language's own
    ``trim`` strips a different set of codepoints -- PHP's is ASCII-only where
    Python's covers Unicode whitespace -- so trimming here would close a hole in
    one language and open a different one in the others. That is G-36's lesson,
    and this is the cheap version of it.
    """
    if value == "":
        raise HarnessError.task_identifier_blank(kind)


def _timestamp(value: float) -> int:
    """A Unix timestamp as an INTEGER.

    Not a formatted date, because date formatting is exactly where three
    languages produce three strings from one instant. Truncated toward zero,
    which for a wall-clock timestamp is the floor, and which is what PHP's
    ``(int)`` cast and JavaScript's ``Math.floor`` produce for the same input.
    """
    return int(value)
