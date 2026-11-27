"""Property tests for approval pairing (G-392 QC).

Generated steps of one to six tool calls whose ids come from a small pool
(``a``, ``b``, ``c`` and the empty id) so that repeats are common, mixed gated
(``echo``) and ungated (``shout``) tools, per-call decisions (approve, deny,
none) recorded in shuffled order, and resumes through ``send`` and
``send_structured``, repeated, with late decisions in between.

Two ways a step reaches a resume, both through the public seam:

- ``live``: a scripted model asks for the calls and the harness issues the
  approval requests itself;
- ``stored``: the step is in the thread already, as another writer (an earlier
  release, another language's harness on a shared store) left it, so its
  approval requests can be shuffled, missing or duplicated, and some calls
  can already have a result.

The oracle below is a small, independent model of the rule, never a call into
the code under test:

- a step whose call ids repeat (the empty id counts), or whose approval
  requests repeat a ``tool_call_id``, runs NONE of its gated calls, whatever
  was decided; each gets ``REPEATED``;
- otherwise a gated call runs only if its OWN request was approved; a denial
  is its reason; no decision is a refusal;
- a call that already has a result is done and never runs again.

Every call carries a unique token in its arguments, so the test can tell which
call ran even when ids repeat.

``PRISM_PROPERTY_EXAMPLES`` sets the example count, ``PRISM_PROPERTY_BASE_SEED``
the first seed of the sequence, and ``PRISM_PROPERTY_SEED`` replays one seed.
Each, when set, must be a whole number >= 1; anything else is refused.
A failure prints the seed and the smallest failing case found by dropping
calls and requests. No third-party generator: the PRNG is the standard
library's ``random.Random``, seeded per example.
"""

from __future__ import annotations

import functools
import json
import os
import random
import socket
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from prism_harness import (
    AgentResponse,
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    Session,
    ToolRegistry,
    record_approval,
)

REPEATED = "Tool call was not run because another call in this step has the same id"
REPLACED_BY_LATER_STEP = "Tool call was not run because a later step replaced it"
NO_DECISION = "No approval response provided"
DEFAULT_DENIAL = "User denied tool execution"

ID_POOL = ("a", "b", "c", "")
GATED, UNGATED = "echo", "shout"

MODES = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": [GATED, UNGATED],
                "max_steps": 8,
                "requires_approval": [GATED],
            }
        },
    }
)

SPENT = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": [GATED, UNGATED],
                "max_steps": 1,
                "requires_approval": [GATED],
            }
        },
    }
)

#: The rows a host may record while an approval is pending (G-397, G-399),
#: the same kinds as prism-harness-ts's generator. ``system`` is drawn twice.
INSERT_KINDS = (
    "assistant_text",
    "assistant_call",
    "user",
    "system",
    "system",
    "tool_empty",
    "tool_result",
)

#: A kind that supersedes the parked step (G-397).
SUPERSEDES = ("assistant_text", "assistant_call", "user")

SCHEMA: dict[str, Any] = {
    "name": "answer",
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


# -- the generated case ----------------------------------------------------------


@dataclass(frozen=True)
class Call:
    id: str
    gated: bool
    token: str
    #: approve / deny / none. Live steps only; stored steps decide per request.
    decision: str = "none"
    #: The denial reason, or None for the default text.
    reason: str | None = None
    #: Stored steps only: the call already has a result.
    prior: bool = False
    #: Its prior result was stored with OTHER arguments (lossy or redacted):
    #: ``{"token": "<token>-lossy"}``, which an earlier-step DECOY call has.
    lossy: bool = False


@dataclass(frozen=True)
class Request:
    approval_id: str
    tool_call_id: str
    decision: str
    reason: str | None


@dataclass(frozen=True)
class Insert:
    """A row recorded between the park and the resume (G-397, G-399)."""

    kind: str
    when: str  # before_decision | after_decision
    #: assistant_call: the step call whose id the inserted (ungated) call
    #: reuses; tool_result: the open gated call the forged result names.
    call: int


@dataclass(frozen=True)
class Later:
    """A LATER step with tool calls a host records after the park (G-392, G-397).

    One call under ``later``, recorded at ``when``; beside an inserted row
    recorded at the same moment, ``order`` says which comes first, so a forged
    result lands in the later step's segment (``before_insert``) or in the
    parked step's (``after_insert``).
    """

    when: str  # before_decision | after_decision
    order: str  # before_insert | after_insert
    gated: bool
    #: The step call whose id the later call reuses, or None for its own id.
    reuses: int | None


@dataclass(frozen=True)
class Case:
    seed: int
    path: str  # live | stored
    calls: tuple[Call, ...]
    #: Stored steps only, in stored order.
    requests: tuple[Request, ...] = ()
    #: The order decisions are recorded in: indexes into the gated calls (live)
    #: or into ``requests`` (stored).
    decision_order: tuple[int, ...] = ()
    #: The shape of each resume: text or structured.
    resumes: tuple[str, ...] = ("text", "text")
    #: The shape of the send that takes the step (live only).
    first: str = "text"
    #: A row a host records while the approval is pending, or None.
    insert: Insert | None = None
    #: Stored only: arguments stored as JSON strings, as the PHP reference
    #: writes them, for the calls and for the results already present.
    string_arguments: bool = False
    #: A later step with tool calls a host records after the park, or None.
    later: Later | None = None
    #: Stored only: an EARLIER parked step the case's step followed. Each of its
    #: gated calls is asked about under ``old_<token>`` and decided by its own
    #: ``decision``; none of them may ever run.
    earlier: tuple[Call, ...] = ()
    #: The run has no step left after the step that parked (live only), so a
    #: resume ends without another model step.
    spent: bool = False
    #: A row written WHILE the first approved call runs (``user``, or
    #: ``tool_step``: a step with an ungated call reusing that call's id), or
    #: None. It must change no call's outcome.
    during: str | None = None
    #: A host's COPY of the paused step that keeps its approval requests and
    #: holds NEW calls under the gated calls' ids (``mixed``: after a call that
    #: is not an object; ``new-only``), recorded at ``copy_when``, or None. An
    #: approval belongs to the step that first asked for it (G-408).
    copy: str | None = None
    copy_when: str = "before_decision"
    #: After the first resume, a later row in the SAME segment holds a result
    #: under this call's id and tool with OTHER arguments (index of the call,
    #: or None). Every call already has its own result by then, so the stray
    #: one ends nothing and replaces nothing.
    stray: int | None = None

    def describe(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "path": self.path,
                "first": self.first,
                "resumes": list(self.resumes),
                "calls": [
                    {
                        "id": c.id,
                        "tool": GATED if c.gated else UNGATED,
                        "token": c.token,
                        **({"decision": c.decision} if self.path == "live" and c.gated else {}),
                        **({"reason": c.reason} if c.reason is not None else {}),
                        **({"prior_result": True} if c.prior else {}),
                        **({"lossy_result_arguments": True} if c.lossy else {}),
                    }
                    for c in self.calls
                ],
                **(
                    {
                        "requests": [
                            {
                                "approval_id": r.approval_id,
                                "tool_call_id": r.tool_call_id,
                                "decision": r.decision,
                                **({"reason": r.reason} if r.reason is not None else {}),
                            }
                            for r in self.requests
                        ]
                    }
                    if self.path == "stored"
                    else {}
                ),
                "decision_order": list(self.decision_order),
                **(
                    {
                        "insert": {
                            "kind": self.insert.kind,
                            "when": self.insert.when,
                            "call": self.insert.call,
                        }
                    }
                    if self.insert is not None
                    else {}
                ),
                **(
                    {
                        "later": {
                            "when": self.later.when,
                            "order": self.later.order,
                            "tool": GATED if self.later.gated else UNGATED,
                            "reuses": self.later.reuses,
                        }
                    }
                    if self.later is not None
                    else {}
                ),
                **(
                    {
                        "earlier": [
                            {
                                "id": c.id,
                                "tool": GATED if c.gated else UNGATED,
                                "token": c.token,
                                **({"decision": c.decision} if c.gated else {}),
                                **({"prior_result": True} if c.prior else {}),
                            }
                            for c in self.earlier
                        ]
                    }
                    if self.earlier
                    else {}
                ),
                **({"spent": True} if self.spent else {}),
                **({"stray": self.stray} if self.stray is not None else {}),
                **({"during": self.during} if self.during is not None else {}),
                **(
                    {"copy": self.copy, "copy_when": self.copy_when}
                    if self.copy is not None
                    else {}
                ),
                **({"string_arguments": True} if self.string_arguments else {}),
            }
        )


def insertion(seed: int, calls: list[Call]) -> Insert | None:
    """The inserted row, from a stream of its own, so the seed's other draws
    are unchanged (as in prism-harness-ts)."""
    rng = random.Random(seed ^ 0x397397)
    if rng.random() >= 0.6:
        return None
    kind = rng.choice(INSERT_KINDS)
    # A forged result names a gated call that has none yet.
    open_calls = [
        i
        for i, c in enumerate(calls)
        if c.gated and not c.prior and sum(d.token == c.token for d in calls) == 1
    ]
    if kind == "tool_result" and not open_calls:
        kind = "system"
    when = rng.choice(("before_decision", "after_decision"))
    call = rng.choice(open_calls) if kind == "tool_result" else rng.randrange(len(calls))
    return Insert(kind, when, call)


def later_step(seed: int, calls: list[Call], insert: Insert | None) -> Later | None:
    """The later step, from a stream of its own (the other draws are unchanged)."""
    rng = random.Random(seed ^ 0x392392)
    if rng.random() >= 0.4:
        return None
    when = insert.when if insert is not None else rng.choice(("before_decision", "after_decision"))
    return Later(
        when=when,
        order=rng.choice(("before_insert", "after_insert")),
        gated=rng.random() < 0.5,
        reuses=rng.randrange(len(calls)) if rng.random() < 0.6 else None,
    )


def later_row(case: Case) -> dict[str, Any]:
    assert case.later is not None
    reuses = case.later.reuses
    return {
        "type": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": case.calls[reuses].id if reuses is not None else "later",
                "name": GATED if case.later.gated else UNGATED,
                "arguments": {"token": "later"},
            }
        ],
        "tool_approval_requests": [],
    }


def a_copy(seed: int, path: str, calls: list[Call], insert: Insert | None) -> str | None:
    """The copy of the paused step, from a stream of its own, or None. Only
    beside no other inserted row, so the oracle stays a few lines."""
    rng = random.Random(seed ^ 0x408408)
    if insert is not None or not any(c.gated for c in calls) or rng.random() >= 0.2:
        return None
    return rng.choice(("mixed", "new-only"))


def copy_row(session: Session, case: Case) -> dict[str, Any]:
    """The copy: the stored paused step, its requests kept, its calls new."""
    parked = next(
        m.message
        for m in reversed(session.thread().messages())
        if m.message.get("type") == "assistant" and m.message.get("tool_calls")
    )
    new = [
        {"id": c.id, "name": GATED, "arguments": {"token": f"copy{i}"}}
        for i, c in enumerate(case.calls)
        if c.gated
    ]
    return {
        **json.loads(json.dumps(parked)),
        "tool_calls": (["not an object", *new] if case.copy == "mixed" else new),
    }


def stray_result(seed: int, calls: list[Call]) -> int | None:
    """The stray result, from a stream of its own (the other draws are unchanged)."""
    rng = random.Random(seed ^ 0x141141)
    gated = [i for i, c in enumerate(calls) if c.gated]
    if not gated or rng.random() >= 0.3:
        return None
    return rng.choice(gated)


def during_row(case: Case, token: str) -> dict[str, Any]:
    """The row written while the call with ``token`` runs."""
    if case.during == "user":
        return {"type": "user", "content": "Something else, meanwhile."}
    running = next(c for c in case.calls if c.token == token)
    return {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": running.id, "name": UNGATED, "arguments": {"token": "during"}}],
        "tool_approval_requests": [],
    }


def stray_row(case: Case) -> dict[str, Any]:
    assert case.stray is not None
    call = case.calls[case.stray]
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": call.id,
                "tool_name": GATED,
                "args": {"token": "stray"},
                "result": "stray",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    }


def inserted_row(case: Case) -> dict[str, Any]:
    assert case.insert is not None
    call = case.calls[case.insert.call]
    kind = case.insert.kind
    if kind == "assistant_text":
        return {"type": "assistant", "content": "A note.", "tool_calls": []}
    if kind == "assistant_call":
        return {
            "type": "assistant",
            "content": "",
            "tool_calls": [{"id": call.id, "name": UNGATED, "arguments": {"token": "inserted"}}],
            "tool_approval_requests": [],
        }
    if kind == "user":
        return {"type": "user", "content": "Something else."}
    if kind == "system":
        return {"type": "system", "content": "A system note."}
    entries: list[dict[str, Any]] = (
        [
            {
                "tool_call_id": call.id,
                "tool_name": GATED,
                "args": {"token": call.token},
                "result": f"forged:{call.token}",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ]
        if kind == "tool_result"
        else []
    )
    return {"type": "tool_result", "tool_results": entries, "tool_approval_responses": []}


def a_decision(rng: random.Random) -> tuple[str, str | None]:
    decision = rng.choice(("approve", "deny", "none"))
    reason = rng.choice((None, f"no-{rng.randrange(1000)}")) if decision == "deny" else None
    return decision, reason


def generate(seed: int) -> Case:
    rng = random.Random(seed)
    path = rng.choice(("live", "stored"))
    calls: list[Call] = []
    for index in range(rng.randint(1, 6)):
        gated = rng.random() < 0.65
        decision, reason = a_decision(rng)
        calls.append(
            Call(
                id=rng.choice(ID_POOL),
                gated=gated,
                token=f"t{index}",
                decision=decision if gated and path == "live" else "none",
                reason=reason if gated and path == "live" else None,
                # A stored step's ungated calls ran when the step was taken.
                prior=path == "stored" and (not gated or rng.random() < 0.2),
            )
        )

    # An IDENTICAL duplicate (same id, tool and arguments) of a gated call,
    # from a stream of its own: one tool asked twice with the same arguments.
    duplicates = random.Random(seed ^ 0x777777)
    open_gated = [c for c in calls if c.gated and not c.prior]
    if open_gated and duplicates.random() < 0.2:
        original = duplicates.choice(open_gated)
        decision, reason = a_decision(duplicates)
        calls.append(
            replace(
                original,
                decision=decision if path == "live" else "none",
                reason=reason if path == "live" else None,
            )
        )

    requests: list[Request] = []
    if path == "stored":
        counter = 0
        for call in calls:
            if not call.gated or rng.random() < 0.12:
                continue  # an ungated call is asked about by nobody; else MISSING
            copies = 2 if rng.random() < 0.12 else 1  # a DUPLICATED request
            for _ in range(copies):
                counter += 1
                decision, reason = a_decision(rng)
                requests.append(Request(f"apr_{counter}", call.id, decision, reason))
        if len(requests) >= 2 and rng.random() < 0.12:
            # A REPEATED APPROVAL id: two requests under one approval id.
            source, target = rng.sample(range(len(requests)), 2)
            requests[target] = replace(requests[target], approval_id=requests[source].approval_id)
        if rng.random() < 0.5:
            rng.shuffle(requests)

    earlier: list[Call] = []
    if path == "stored" and rng.random() < 0.3:
        for index in range(rng.randint(1, 3)):
            gated = rng.random() < 0.65
            decision, reason = a_decision(rng)
            earlier.append(
                Call(
                    id=rng.choice(ID_POOL),
                    gated=gated,
                    token=f"e{index}",
                    decision=decision if gated else "none",
                    reason=reason if gated else None,
                    prior=not gated or rng.random() < 0.25,
                )
            )
    if path == "stored":
        # Only where its call is the one call with that id and tool, so the
        # step's own fallback (id and tool) can only mean that call.
        lossy = [
            i
            for i, c in enumerate(calls)
            if c.gated
            and c.prior
            and sum(1 for d in calls if d.id == c.id and d.gated == c.gated) == 1
        ]
        if lossy and rng.random() < 0.3:
            index = rng.choice(lossy)
            calls[index] = replace(calls[index], lossy=True)
            decision, reason = a_decision(rng)
            # The DECOY: an earlier call with the same id and tool whose
            # arguments are the ones the lossy result was stored with.
            earlier.append(
                Call(
                    id=calls[index].id,
                    gated=True,
                    token=f"{calls[index].token}-lossy",
                    decision=decision,
                    reason=reason,
                    prior=False,
                )
            )
    if earlier:
        current = [c for c in calls if c.gated and sum(d.token == c.token for d in calls) == 1]
        if current and rng.random() < 0.4:
            # A TWIN: the same id and arguments as a call of the current step,
            # so only the segment a result sits in says which call it answers.
            twin = rng.choice(current)
            decision, reason = a_decision(rng)
            earlier.append(
                Call(
                    id=twin.id,
                    gated=True,
                    token=twin.token,
                    decision=decision,
                    reason=reason,
                    prior=rng.random() < 0.6,
                )
            )

    decidable = (
        [i for i, c in enumerate([c for c in calls if c.gated])]
        if path == "live"
        else list(range(len(requests)))
    )
    rng.shuffle(decidable)
    resumes = tuple(rng.choice(("text", "structured")) for _ in range(2))
    return Case(
        seed=seed,
        path=path,
        calls=tuple(calls),
        requests=tuple(requests),
        decision_order=tuple(decidable),
        resumes=resumes,
        first=rng.choice(("text", "structured")),
        insert=(insert := insertion(seed, calls)),
        stray=stray_result(seed, calls),
        copy=a_copy(seed, path, calls, insert),
        copy_when=random.Random(seed ^ 0x408409).choice(("before_decision", "after_decision")),
        during=(
            random.Random(seed ^ 0x404404).choice(("user", "tool_step"))
            if random.Random(seed ^ 0x404405).random() < 0.25
            else None
        ),
        later=later_step(seed, calls, insert),
        earlier=tuple(earlier),
        string_arguments=path == "stored" and rng.random() < 0.3,
        spent=path == "live" and rng.random() < 0.8,
    )


# -- the oracle: an independent model of the rule ----------------------------------


@dataclass
class Expected:
    #: Tokens of the gated calls that run on the first resume.
    runs: list[str] = field(default_factory=list)
    #: Each gated call's one outcome text, by token.
    outcomes: dict[str, str] = field(default_factory=dict)
    #: Texts stored under a token that are ANOTHER call's outcome: a result
    #: recorded with that token's arguments in a later step's segment.
    extras: dict[str, set[str]] = field(default_factory=dict)
    #: The later step's call was ended by a result in its segment.
    later_credited: bool = False
    repeated: bool = False


def copied(case: Case) -> bool:
    """Whether the case records a copy of the paused step (G-408)."""
    return case.copy is not None and case.later is None


def oracle(case: Case) -> Expected:
    ids = [c.id for c in case.calls]
    if case.path == "live":
        # The harness issues one request per gated call, under the call's id.
        asked = [c.id for c in case.calls if c.gated]
    else:
        asked = [r.tool_call_id for r in case.requests]
    approvals = [r.approval_id for r in case.requests]
    repeated = (
        len(set(ids)) != len(ids)
        or len(set(asked)) != len(asked)
        or len(set(approvals)) != len(approvals)
    )
    expected = Expected(repeated=repeated)
    # A user row or an assistant row after the park supersedes the step: each
    # open gated call gets one replaced-step denial, and nothing runs, whatever
    # was decided before or after (G-397). A forged result is its call's only
    # outcome. Any other row changes nothing (G-399).
    superseded = (
        (case.insert is not None and case.insert.kind in SUPERSEDES)
        or case.later is not None
        or copied(case)
    )
    if copied(case):
        # The copy's calls have no request of their own: the requests it
        # repeats belong to the paused step (G-408).
        gated_ids = [c.id for c in case.calls if c.gated]
        copy_repeated = len(set(gated_ids)) != len(gated_ids) or (
            case.path == "stored"
            and (len(set(asked)) != len(asked) or len(set(approvals)) != len(approvals))
        )
        for i, c in enumerate(case.calls):
            if c.gated:
                expected.outcomes[f"copy{i}"] = REPEATED if copy_repeated else NO_DECISION
    # The later step is itself superseded by a superseding row recorded after
    # it; otherwise it is the step a resume resolves, and nobody decided it.
    if case.later is not None and case.later.gated:
        after_it = (
            case.insert is not None
            and case.insert.kind in SUPERSEDES
            and case.later.order == "before_insert"
        )
        expected.outcomes["later"] = REPLACED_BY_LATER_STEP if after_it else NO_DECISION
    forged = (
        case.insert.call if case.insert is not None and case.insert.kind == "tool_result" else -1
    )
    # A result counts ONLY for the step whose segment it sits in. Recorded
    # after a later step, the forged result is the later step's: it ends the
    # later call when that call has its id, and never the parked call.
    if forged >= 0 and case.later is not None and case.later.order == "before_insert":
        token = case.calls[forged].token
        expected.extras[token] = {f"forged:{token}"}
        later_id = case.calls[case.later.reuses].id if case.later.reuses is not None else "later"
        expected.later_credited = later_id == case.calls[forged].id
        forged = -1

    # A later step replaced the earlier one: its gated calls never run, and
    # each one with no result gets one denial.
    current = {c.token for c in case.calls}
    for call in case.earlier:
        if call.gated:
            key = f"{call.token}#earlier" if call.token in current else call.token
            expected.outcomes[key] = f"prior:{call.token}" if call.prior else REPLACED_BY_LATER_STEP

    for index, call in enumerate(case.calls):
        if not call.gated:
            continue
        if call.prior:
            expected.outcomes[call.token] = f"prior:{call.token}"
            continue
        if index == forged:
            expected.outcomes[call.token] = f"forged:{call.token}"
            continue
        if superseded:
            expected.outcomes[call.token] = REPLACED_BY_LATER_STEP
            continue
        if repeated:
            expected.outcomes[call.token] = REPEATED
            continue
        if case.path == "live":
            decision, reason = call.decision, call.reason
        else:
            own = [r for r in case.requests if r.tool_call_id == call.id]
            decision, reason = (own[0].decision, own[0].reason) if own else ("none", None)
        if decision == "approve":
            expected.runs.append(call.token)
            expected.outcomes[call.token] = f"{GATED}:{call.token}"
        elif decision == "deny":
            expected.outcomes[call.token] = reason or DEFAULT_DENIAL
        else:
            expected.outcomes[call.token] = NO_DECISION
    return expected


# -- driving the harness through its public seam --------------------------------------


class TokenTool:
    def __init__(self, name: str, ran: list[tuple[str, str]]) -> None:
        self._name = name
        self._ran = ran
        #: Called with the call's token while it runs, before it returns.
        self.during: Callable[[str], None] | None = None

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        token = str(args.get("token", ""))
        self._ran.append((self._name, token))
        if self.during is not None:
            self.during(token)
        return f"{self._name}:{token}"


def a_session(directory: str) -> Session:
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("support")
    session.using_mode("guarded").using_provider("anthropic").using_model("claude-sonnet-5-5")
    return session


def a_model(
    case: Case, requests: list[LlmRequest] | None = None
) -> Callable[[LlmRequest], LlmResponse]:
    """Asks for the case's calls once (live), then only ever answers."""
    state = {"asked": case.path == "stored"}

    def client(request: LlmRequest) -> LlmResponse:
        if requests is not None:
            requests.append(request)
        if not state["asked"]:
            state["asked"] = True
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    LlmToolCall(c.id, GATED if c.gated else UNGATED, {"token": c.token})
                    for c in case.calls
                ],
            )
        if request.schema is not None:
            return LlmResponse(text='{"ok": true}', finish_reason="stop", structured={"ok": True})
        return LlmResponse(text="Done.", finish_reason="stop")

    return client


def a_send(runtime: AgentRuntime, session: Session, shape: str, prompt: str) -> AgentResponse:
    if shape == "structured":
        return runtime.send_structured(session, prompt, SCHEMA)
    return runtime.send(session, prompt)


def stored_rows(
    calls: tuple[Call, ...], requests: list[tuple[str, str]], strings: bool = False
) -> list[dict[str, Any]]:
    """One step as another writer left it: the assistant row and its results."""

    def arguments(call: Call) -> Any:
        value = {"token": call.token}
        return json.dumps(value) if strings else value

    def lossy_arguments(call: Call) -> Any:
        value = {"token": f"{call.token}-lossy"}
        return json.dumps(value) if strings else value

    rows: list[dict[str, Any]] = [
        {
            "type": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": c.id,
                    "name": GATED if c.gated else UNGATED,
                    "arguments": arguments(c),
                    "result_id": None,
                    "reasoning_id": None,
                    "reasoning_summary": None,
                }
                for c in calls
            ],
            "additional_content": {},
            "tool_approval_requests": [
                {"approval_id": approval, "tool_call_id": call_id, "arguments_format": 1}
                for approval, call_id in requests
            ],
            "harness_tool_names": [GATED, UNGATED],
        },
    ]
    priors = [c for c in calls if c.prior]
    if priors:
        rows.append(
            {
                "type": "tool_result",
                "tool_results": [
                    {
                        "tool_call_id": c.id,
                        "tool_name": GATED if c.gated else UNGATED,
                        "args": lossy_arguments(c) if c.lossy else arguments(c),
                        "result": f"prior:{c.token}",
                        "tool_call_result_id": None,
                        "artifacts": [],
                    }
                    for c in priors
                ],
                "tool_approval_responses": [],
            }
        )
    return rows


def store_step(session: Session, case: Case) -> None:
    """The step (and any earlier parked step) as another writer left them."""
    rows: list[dict[str, Any]] = [{"type": "user", "content": "go"}]
    if case.earlier:
        rows += stored_rows(
            case.earlier,
            [(f"old_{c.token}", c.id) for c in case.earlier if c.gated],
            case.string_arguments,
        )
    rows += stored_rows(
        case.calls,
        [(r.approval_id, r.tool_call_id) for r in case.requests],
        case.string_arguments,
    )
    session.thread().record(rows)


def insert_between(session: Session, case: Case, when: str) -> None:
    rows: list[dict[str, Any]] = []
    if case.insert is not None and case.insert.when == when:
        rows.append(inserted_row(case))
    if case.later is not None and case.later.when == when:
        if case.later.order == "before_insert":
            rows.insert(0, later_row(case))
        else:
            rows.append(later_row(case))
    if case.copy is not None and case.copy_when == when and case.later is None:
        rows.append(copy_row(session, case))
    if rows:
        session.thread().record(rows)


def outcomes_by_token(session: Session) -> dict[str, list[str]]:
    """Every stored result, by the token of the call it answers, as STORED (never
    the provider view, which folds results by call id). A resume row restates
    the results it found, so one outcome may be stored more than once; what may
    not happen is two DIFFERENT outcomes, or none."""
    found: dict[str, list[str]] = {}
    for entry in session.thread().messages():
        row = entry.message
        if row.get("type") != "tool_result":
            continue
        for result in row.get("tool_results", []):
            args = result.get("args")
            if isinstance(args, str):
                args = json.loads(args)
            token = str((args or {}).get("token", ""))
            text = str(result.get("result"))
            # A result the test recorded names its call in its text; its
            # stored arguments may be another call's (lossy).
            for kind in ("prior:", "forged:"):
                if text.startswith(kind):
                    token = text[len(kind) :]
            found.setdefault(token, []).append(str(result.get("result")))
    return found


def _entries(session: Session) -> list[list[tuple[str, str]]]:
    """Each tool result row's results as (token, text), row by row."""
    rows: list[list[tuple[str, str]]] = []
    for entry in session.thread().messages():
        row = entry.message
        if row.get("type") != "tool_result":
            continue
        found: list[tuple[str, str]] = []
        for result in row.get("tool_results", []):
            args = result.get("args")
            if isinstance(args, str):
                args = json.loads(args)
            token = str((args or {}).get("token", ""))
            text = str(result.get("result"))
            for kind in ("prior:", "forged:"):
                if text.startswith(kind):
                    token = text[len(kind) :]
            found.append((token, text))
        rows.append(found)
    return rows


def recorded_for(session: Session, tokens: set[str]) -> int:
    """How many results are stored for calls with these tokens."""
    return sum(1 for row in _entries(session) for token, _ in row if token in tokens)


def most_per_row(session: Session) -> dict[tuple[str, str], int]:
    """The most results any one row holds, per (token, outcome)."""
    most: dict[tuple[str, str], int] = {}
    for row in _entries(session):
        counts: dict[tuple[str, str], int] = {}
        for key in row:
            counts[key] = counts.get(key, 0) + 1
        for key, n in counts.items():
            most[key] = max(most.get(key, 0), n)
    return most


def late(session: Session, approval: str) -> None:
    """Approve once the step is settled. Refusing the decision is as good as
    ignoring it; running the call is not."""
    try:
        record_approval(session, approval, True)
    except HarnessError as error:
        if error.code != "approval_not_pending":
            raise


def check(case: Case) -> list[str]:
    """Run one case and return every invariant it breaks."""
    expected = oracle(case)
    ran: list[tuple[str, str]] = []
    gated_tool = TokenTool(GATED, ran)
    tools = ToolRegistry().register(gated_tool).register(TokenTool(UNGATED, ran))
    requests: list[LlmRequest] = []
    runtime = AgentRuntime(
        client=a_model(case, requests), modes=SPENT if case.spent else MODES, tools=tools
    )
    with tempfile.TemporaryDirectory(
        prefix="prism-harness-pairing-", ignore_cleanup_errors=True
    ) as directory:
        session = a_session(directory)
        fired: list[str] = []

        def during(token: str) -> None:
            # Once, while the first approved call runs (G-392, G-404). The
            # host writes it (no run id); a tool-call step could equally be
            # another run's.
            if case.during is not None and not fired:
                fired.append(token)
                session.thread().record([during_row(case, token)])

        gated_tool.during = during
        return _check(case, session, runtime, ran, expected) + replayed_twice(requests)


def replayed_twice(requests: list[LlmRequest]) -> list[str]:
    """Replay invariant: no result occurrence (call id, tool, arguments and
    text) is sent to the model more often, within a step's segment, than the
    step has calls with that id, tool and arguments, in any request. The text
    is part of the key because the view reads arguments stored as a JSON string
    as none, so two calls' results can share the rest."""
    broken: list[str] = []
    for number, request in enumerate(requests):
        seen: dict[tuple[int, str, str, str, str], int] = {}
        calls: dict[tuple[int, str, str, str], int] = {}
        segment = 0
        for message in request.messages:
            if message.get("type") == "assistant" and message.get("tool_calls"):
                segment += 1
                for call in message["tool_calls"]:
                    same = (
                        segment,
                        str(call.get("id")),
                        str(call.get("name")),
                        json.dumps(call.get("arguments"), sort_keys=True),
                    )
                    calls[same] = calls.get(same, 0) + 1
            if message.get("type") != "tool_result":
                continue
            for entry in message.get("tool_results", []):
                key = (
                    segment,
                    str(entry.get("tool_call_id")),
                    str(entry.get("tool_name")),
                    json.dumps(entry.get("args"), sort_keys=True),
                    str(entry.get("result")),
                )
                seen[key] = seen.get(key, 0) + 1
        twice = sorted(key for key, n in seen.items() if n > max(1, calls.get(key[:4], 0)))
        if twice:
            broken.append(f"replay: request {number} sends {twice} more than once")
    return broken


def _check(
    case: Case,
    session: Session,
    runtime: AgentRuntime,
    ran: list[tuple[str, str]],
    expected: Expected,
) -> list[str]:
    gated = [c for c in case.calls if c.gated]
    ungated = [c for c in case.calls if not c.gated]
    broken: list[str] = []

    # Which approval belongs to which call, in this case's own terms.
    approval_of: dict[str, str | None] = {}
    decided: dict[str, tuple[str, str | None]] = {}

    if case.path == "live":
        first = a_send(runtime, session, case.first, "go")
        if [name for name, _ in ran] != [UNGATED] * len(ungated) or sorted(
            t for _, t in ran
        ) != sorted(c.token for c in ungated):
            broken.append(f"6: the first step ran {ran}, not exactly the ungated calls")
        pending = first.pending_approvals
        if [(p.tool_call_id, p.arguments.get("token")) for p in pending] != [
            (c.id, c.token) for c in gated
        ]:
            broken.append(
                "6: pending approvals "
                f"{[(p.tool_call_id, p.arguments) for p in pending]} do not name the gated calls"
            )
            return broken
        if len({p.id for p in pending}) != len(pending):
            broken.append("2: two pending approvals share an approval id")
        if first.tool_calls != [UNGATED] * len(ungated):
            broken.append(f"6: the first step reported tool_calls {first.tool_calls}")
        for call, p in zip(gated, pending, strict=True):
            approval_of[call.token] = p.id
        insert_between(session, case, "before_decision")
        for index in case.decision_order:
            call = gated[index]
            if call.decision == "none":
                continue
            approval = approval_of[call.token]
            assert approval is not None
            record_approval(session, approval, call.decision == "approve", call.reason)
            decided[approval] = (call.decision, call.reason)
    else:
        store_step(session, case)
        for old in case.earlier:
            if old.gated and old.decision != "none":
                record_approval(session, f"old_{old.token}", old.decision == "approve", old.reason)
        insert_between(session, case, "before_decision")
        for index in case.decision_order:
            request = case.requests[index]
            if request.decision == "none":
                continue
            record_approval(
                session, request.approval_id, request.decision == "approve", request.reason
            )
            decided[request.approval_id] = (request.decision, request.reason)
        for call in gated:
            own = [r.approval_id for r in case.requests if r.tool_call_id == call.id]
            approval_of[call.token] = own[0] if len(own) == 1 else None

    insert_between(session, case, "after_decision")
    before = len(ran)
    resumed = a_send(runtime, session, case.resumes[0], "")
    ran_now = [token for name, token in ran[before:] if name == GATED]

    # Full equality first: what ran is what the rule says runs.
    if sorted(ran_now) != sorted(expected.runs):
        broken.append(f"equality: ran {sorted(ran_now)}, the rule runs {sorted(expected.runs)}")
    if any(name == UNGATED for name, _ in ran[before:]):
        broken.append("3: a resume ran an ungated call the step had already run")
    # 1. A denied call never runs.
    for call in gated:
        approval = approval_of.get(call.token)
        if call.token in ran_now and (
            approval is None or decided.get(approval, ("none",))[0] != "approve"
        ):
            broken.append(f"1: {call.token} ran without its own approve")
    # 2. One decision decides at most one call.
    used = [approval_of.get(t) for t in ran_now]
    if None in used or len(set(used)) != len(used):
        broken.append(f"2: the calls that ran {ran_now} are not matched to distinct approves")
    # 4. A step with repeated ids runs none of its gated calls.
    if expected.repeated and ran_now:
        broken.append(f"4: a step with a repeated id ran {ran_now}")
    # 6. The run reports what ran.
    if resumed.tool_calls != [GATED] * len(ran_now):
        broken.append(f"6: the resume reported tool_calls {resumed.tool_calls}, ran {ran_now}")
    run = session.run()
    if run is None or run.get("tool_calls") != [GATED] * len(ran_now):
        broken.append(f"6: the run record says {run and run.get('tool_calls')}, ran {ran_now}")

    # 3. Nothing runs twice: resume again, decide everything late, resume again.
    settled = recorded_for(session, {c.token for c in gated})
    if case.stray is not None:
        session.thread().record([stray_row(case)])
    after = len(ran)
    a_send(runtime, session, case.resumes[1], "")
    for approval in (
        [a for a in approval_of.values() if a is not None]
        + [r.approval_id for r in case.requests]
        + [f"old_{c.token}" for c in case.earlier if c.gated]
    ):
        late(session, approval)
    a_send(runtime, session, case.resumes[0], "")
    if ran[after:]:
        broken.append(f"3: a later resume or a later decision ran {ran[after:]}")
    counts: dict[str, int] = {}
    for _, token in ran:
        counts[token] = counts.get(token, 0) + 1
    twice = sorted(t for t, n in counts.items() if n > 1)
    if twice:
        broken.append(f"3: {twice} ran more than once")

    # 5 and 6. Every gated call has exactly one outcome, and the thread says what ran.
    # Once the first resume has settled the step, no further result for its
    # gated calls is written, by any later resume or decision.
    final = recorded_for(session, {c.token for c in gated})
    if final != settled:
        broken.append(f"5: results written after the step settled: {final - settled} more")
    # COUNTED per call: the most results one row holds for a token and an
    # outcome is the number of calls with that token (a restating row repeats
    # them, never adds to them).
    per_row = most_per_row(session)
    for token in {c.token for c in gated}:
        calls_with = sum(1 for c in gated if c.token == token)
        twin = any(c.token == token and c.gated for c in case.earlier)
        counted = {
            text: n
            for (t, text), n in per_row.items()
            if t == token and text not in expected.extras.get(token, set())
        }
        if not twin and sum(counted.values()) != calls_with:
            broken.append(
                f"5: {token} ({calls_with} call(s)) is recorded as {counted}, one outcome each"
            )
    stored = outcomes_by_token(session)
    current_tokens = {c.token for c in case.calls}
    for i, c in enumerate(case.calls):
        if copied(case) and c.gated:
            token = f"copy{i}"
            texts = set(stored.get(token, []))
            if texts != {expected.outcomes[token]} or counts.get(token):
                broken.append(
                    f"1/5: the copy's call {token} ran {counts.get(token, 0)}x and is recorded "
                    f"as {sorted(texts)}, the rule says once {expected.outcomes[token]!r}"
                )
    if case.later is not None and case.later.gated:
        later = set(stored.get("later", []))
        wanted_later = set() if expected.later_credited else {expected.outcomes["later"]}
        if later != wanted_later or counts.get("later"):
            broken.append(
                f"1/5: the later step's call ran {counts.get('later', 0)}x and is recorded as "
                f"{sorted(later)}, the rule says once {expected.outcomes['later']!r}"
            )
    twins = {c.token: c for c in case.earlier if c.gated and c.token in current_tokens}
    for call in [c for c in case.earlier if c.gated and c.token not in twins]:
        recorded = stored.get(call.token, [])
        if recorded != [expected.outcomes[call.token]] or counts.get(call.token):
            broken.append(
                f"1/5: earlier-step call {call.token} ran {counts.get(call.token, 0)}x and is "
                f"recorded as {recorded}, the rule says once {expected.outcomes[call.token]!r}"
            )
    for call in gated:
        texts = set(stored.get(call.token, []))
        texts -= expected.extras.get(call.token, set())
        if call.token in twins:
            # Two calls share this token, one per step: each has its own outcome.
            wanted = {expected.outcomes[call.token], expected.outcomes[f"{call.token}#earlier"]}
            if texts != wanted:
                broken.append(
                    f"5: {call.token} and its earlier twin are recorded as {sorted(texts)}, "
                    f"the rule says {sorted(wanted)}"
                )
            texts = {expected.outcomes[call.token]} if texts == wanted else texts
        if len(texts) != 1:
            broken.append(f"5: {call.token} has {len(texts)} recorded outcomes {sorted(texts)}")
            continue
        [text] = texts
        if text != expected.outcomes[call.token]:
            broken.append(
                f"6: {call.token} is recorded as {text!r}, the rule says "
                f"{expected.outcomes[call.token]!r}"
            )
        says_ran = text == f"{GATED}:{call.token}"
        if says_ran != (counts.get(call.token, 0) == 1):
            broken.append(f"6: the thread says {call.token} ran={says_ran}, it ran={not says_ran}")
    return broken


# -- shrinking and the property ---------------------------------------------------------


def without_call(case: Case, index: int) -> Case:
    gone = case.calls[index]
    calls = case.calls[:index] + case.calls[index + 1 :]
    insert = case.insert
    if insert is not None:
        insert = (
            None
            if insert.call == index
            else replace(insert, call=insert.call - (insert.call > index))
        )
    later = case.later
    if later is not None and later.reuses is not None:
        later = replace(
            later,
            reuses=None if later.reuses == index else later.reuses - (later.reuses > index),
        )
    stray = case.stray
    if stray is not None:
        stray = None if stray == index else stray - (stray > index)
    case = replace(case, insert=insert, later=later, stray=stray)
    if case.path == "live":
        gated_before = [i for i, c in enumerate(case.calls) if c.gated]
        order = case.decision_order
        if gone.gated:
            position = gated_before.index(index)
            order = tuple(i - (i > position) for i in order if i != position)
        return replace(case, calls=calls, decision_order=order)
    # Stored: the call's requests stay (they may now name nothing, or still repeat).
    return replace(case, calls=calls)


def without_request(case: Case, index: int) -> Case:
    requests = case.requests[:index] + case.requests[index + 1 :]
    order = tuple(i - (i > index) for i in case.decision_order if i != index)
    return replace(case, requests=requests, decision_order=order)


def safe_check(case: Case) -> list[str]:
    """``check``, with a raise reported as a broken case rather than losing the seed."""
    try:
        return check(case)
    except Exception as error:  # noqa: BLE001 - any raise is a failure to report
        return [f"raised {type(error).__name__}: {error}"]


def without_earlier(case: Case, index: int) -> Case:
    return replace(case, earlier=case.earlier[:index] + case.earlier[index + 1 :])


def shrink(case: Case) -> tuple[Case, list[str]]:
    broken = safe_check(case)
    progress = True
    while progress:
        progress = False
        candidates = (
            [without_call(case, i) for i in range(len(case.calls))]
            + [without_request(case, i) for i in range(len(case.requests))]
            + [without_earlier(case, i) for i in range(len(case.earlier))]
        )
        for candidate in candidates:
            if not candidate.calls:
                continue
            found = safe_check(candidate)
            if found:
                case, broken, progress = candidate, found, True
                break
    return case, broken


#: The fast set the normal test run uses: a fixed base seed and few enough
#: examples to stay well under a minute on every CI runner.
DEFAULT_EXAMPLES = 50
DEFAULT_BASE_SEED = 0x392


def setting(name: str, default: int | None) -> int | None:
    """A whole number >= 1 from the environment, or the default when UNSET.

    Set but not a whole number >= 1 is refused, never read as the default: a
    property run that quietly examined nothing (``0``) or the wrong count
    would pass without testing anything.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        raise ValueError(f"{name} must be a whole number >= 1, got {raw!r}.")
    return int(text)


def seeds() -> list[int]:
    replay = setting("PRISM_PROPERTY_SEED", None)
    base = setting("PRISM_PROPERTY_BASE_SEED", DEFAULT_BASE_SEED)
    count = setting("PRISM_PROPERTY_EXAMPLES", DEFAULT_EXAMPLES)
    if replay is not None:
        return [replay]
    assert base is not None and count is not None
    return seeds_for(base, count)


def seeds_for(base: int, count: int) -> list[int]:
    """The run's seeds: ``count`` from ``base``, then every shape's seed."""
    run = [base + i for i in range(count)]
    return run + sorted(set(shape_seeds().values()) - set(run))


@pytest.mark.parametrize("value", ["0", "-3", "", " ", "1.5", "ten", "0x10", "١"])
@pytest.mark.parametrize(
    "name", ["PRISM_PROPERTY_EXAMPLES", "PRISM_PROPERTY_SEED", "PRISM_PROPERTY_BASE_SEED"]
)
def test_a_property_setting_that_is_set_must_be_a_whole_number_of_at_least_one(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"^{name} must be a whole number >= 1"):
        seeds()


def test_unset_property_settings_use_the_fast_fixed_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PRISM_PROPERTY_EXAMPLES", "PRISM_PROPERTY_SEED", "PRISM_PROPERTY_BASE_SEED"):
        monkeypatch.delenv(name, raising=False)
    assert seeds() == seeds_for(DEFAULT_BASE_SEED, DEFAULT_EXAMPLES)
    assert seeds()[:DEFAULT_EXAMPLES] == [DEFAULT_BASE_SEED + i for i in range(DEFAULT_EXAMPLES)]
    monkeypatch.setenv("PRISM_PROPERTY_SEED", "929")
    assert seeds() == [929]


def test_every_generated_step_obeys_the_approval_pairing_rule() -> None:
    examined = seeds()
    print(f"approval pairing properties: {len(examined)} examples from seed {examined[0]}")
    for seed in examined:
        case = generate(seed)
        broken = safe_check(case)
        if broken:
            smallest, why = shrink(case)
            pytest.fail(
                f"seed {seed} breaks the approval pairing rule "
                f"(replay with PRISM_PROPERTY_SEED={seed}).\n"
                f"generated case: {case.describe()}\n"
                f"smallest failing case: {smallest.describe()}\n"
                "broken: " + "; ".join(why),
                pytrace=False,
            )


Shape = Callable[[Case, Expected], bool]


def _inserted(k: Case, e: Expected, *, kind: str, when: str) -> bool:
    return k.insert is not None and k.insert.kind == kind and k.insert.when == when


def _inserted_with_later(k: Case, e: Expected, *, kind: str, when: str, order: str) -> bool:
    return _inserted(k, e, kind=kind, when=when) and k.later is not None and k.later.order == order


def _forged_with_later(k: Case, e: Expected, *, order: str, gated_later: bool) -> bool:
    return (
        k.insert is not None
        and k.insert.kind == "tool_result"
        and k.later is not None
        and k.later.order == order
        and k.later.gated == gated_later
    )


def _forged_in_later_segment(k: Case, e: Expected, *, credited: bool) -> bool:
    return (
        k.insert is not None
        and k.insert.kind == "tool_result"
        and k.later is not None
        and k.later.order == "before_insert"
        and e.later_credited == credited
    )


def _twin_across_steps(k: Case, e: Expected, *, prior: bool, strings: bool) -> bool:
    return k.string_arguments == strings and any(
        c.prior == prior and c.token in {d.token for d in k.calls} for c in k.earlier
    )


def _shapes() -> dict[str, Shape]:
    """Every shape the property claims to reach. The normal CI run includes,
    beside its fixed seeds, the first seed that reaches each one, so each is
    RUN on every push, not only proven reachable."""
    shapes: dict[str, Shape] = {
        "repeated id, gated": lambda k, e: e.repeated and any(c.gated for c in k.calls),
        "distinct ids, a call runs": lambda k, e: not e.repeated and bool(e.runs),
        "empty id": lambda k, e: any(c.id == "" for c in k.calls),
        "two requests name one call": lambda k, e: (
            len({r.tool_call_id for r in k.requests}) != len(k.requests)
        ),
        "repeated approval id": lambda k, e: (
            len({r.approval_id for r in k.requests}) != len(k.requests)
        ),
        "stored prior result": lambda k, e: (
            k.path == "stored" and any(c.gated and c.prior for c in k.calls)
        ),
        "missing request": lambda k, e: (
            k.path == "stored"
            and any(
                c.gated and not any(r.tool_call_id == c.id for r in k.requests) for c in k.calls
            )
        ),
        "structured resume": lambda k, e: "structured" in k.resumes,
        "structured first step": lambda k, e: k.path == "live" and k.first == "structured",
        "identical duplicate calls": lambda k, e: any(
            c.gated and sum(d.token == c.token for d in k.calls) > 1 for c in k.calls
        ),
        "identical duplicates, steps spent": lambda k, e: (
            k.spent
            and any(c.gated and sum(d.token == c.token for d in k.calls) > 1 for c in k.calls)
        ),
        "user row while a call runs": lambda k, e: k.during == "user" and bool(e.runs),
        "outcome: denied with the default reason": lambda k, e: (
            DEFAULT_DENIAL in e.outcomes.values()
        ),
        "outcome: no decision": lambda k, e: NO_DECISION in e.outcomes.values(),
        "outcome: repeated id": lambda k, e: REPEATED in e.outcomes.values(),
        "outcome: replaced": lambda k, e: REPLACED_BY_LATER_STEP in e.outcomes.values(),
        "copy of the paused step, mixed, before the decision": lambda k, e: (
            copied(k) and k.copy == "mixed" and k.copy_when == "before_decision"
        ),
        "copy of the paused step, mixed, after the decision": lambda k, e: (
            copied(k) and k.copy == "mixed" and k.copy_when == "after_decision"
        ),
        "copy of the paused step, new calls only, before the decision": lambda k, e: (
            copied(k) and k.copy == "new-only" and k.copy_when == "before_decision"
        ),
        "copy of the paused step, an approved call": lambda k, e: (
            copied(k)
            and (
                any(c.gated and c.decision == "approve" for c in k.calls)
                if k.path == "live"
                else any(r.decision == "approve" for r in k.requests)
            )
        ),
        "outcome: forged result": lambda k, e: any(
            text.startswith("forged:") for text in e.outcomes.values()
        ),
        # The step written while the call runs reuses the running call's id.
        "tool step while a call runs": lambda k, e: k.during == "tool_step" and bool(e.runs),
        "later step, no insert": lambda k, e: k.later is not None and k.insert is None,
        "lossy result arguments": lambda k, e: any(c.lossy for c in k.calls),
        "lossy, string arguments": lambda k, e: (
            any(c.lossy for c in k.calls) and k.string_arguments
        ),
        "stray result, repeated": lambda k, e: k.stray is not None and e.repeated,
        "stray result, runs": lambda k, e: k.stray is not None and bool(e.runs),
        "string arguments, earlier step": lambda k, e: k.string_arguments and bool(k.earlier),
        "earlier id reused, other args": lambda k, e: any(
            c.id in {d.id for d in k.calls} and c.token not in {d.token for d in k.calls}
            for c in k.earlier
        ),
        "earlier open gated call": lambda k, e: any(c.gated and not c.prior for c in k.earlier),
        "G-399: note after decision, no step left": lambda k, e: (
            k.spent
            and k.insert is not None
            and k.insert.kind in ("system", "tool_empty")
            and k.insert.when == "after_decision"
            and bool(e.runs)
        ),
    }
    for kind in sorted(set(INSERT_KINDS)):
        for when in ("before_decision", "after_decision"):
            shapes[f"insert {kind} {when}"] = functools.partial(_inserted, kind=kind, when=when)
            for order in ("before_insert", "after_insert"):
                shapes[f"later step + insert {kind} {when} {order}"] = functools.partial(
                    _inserted_with_later, kind=kind, when=when, order=order
                )
    for order in ("before_insert", "after_insert"):
        for gated_later in (True, False):
            shapes[f"forged result, later {order}, gated={gated_later}"] = functools.partial(
                _forged_with_later, order=order, gated_later=gated_later
            )
    for credited in (True, False):
        shapes[f"forged in later segment, credited={credited}"] = functools.partial(
            _forged_in_later_segment, credited=credited
        )
    for prior in (True, False):
        for strings in (True, False):
            shapes[f"twin across steps, prior={prior}, strings={strings}"] = functools.partial(
                _twin_across_steps, prior=prior, strings=strings
            )
    return shapes


@functools.lru_cache(maxsize=1)
def shape_seeds() -> dict[str, int]:
    """The first seed from the default base that reaches each shape."""
    found: dict[str, int] = {}
    shapes = _shapes()
    for seed in range(DEFAULT_BASE_SEED, DEFAULT_BASE_SEED + 20000):
        case = generate(seed)
        expected = oracle(case)
        for name, reaches in shapes.items():
            if name not in found and reaches(case, expected):
                found[name] = seed
        if len(found) == len(shapes):
            break
    return found


def test_the_ci_set_runs_every_shape_the_property_claims() -> None:
    # A property that never generates the hard case passes vacuously, and
    # one only the long run reaches is not checked on every push.
    missing = sorted(set(_shapes()) - set(shape_seeds()))
    assert not missing, missing
    ci = set(seeds_for(DEFAULT_BASE_SEED, DEFAULT_EXAMPLES))
    assert set(shape_seeds().values()) <= ci
    outcomes = {text for seed in sorted(ci) for text in oracle(generate(seed)).outcomes.values()}
    assert {REPEATED, NO_DECISION, DEFAULT_DENIAL, REPLACED_BY_LATER_STEP} <= outcomes
    assert any(text.startswith("forged:") for text in outcomes)
