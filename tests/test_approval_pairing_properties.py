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
- a call that already has a result is done and never runs again;
- ABSOLUTE SEGMENT OWNERSHIP (G-387 round 13b): a stored result counts only
  for a call of the step whose segment it sits in (after that step's
  assistant row, before the next one with tool calls), matched there by its
  arguments, then its id and tool name, then its id. A result whose
  arguments name an EARLIER step's call exactly answers a same-id call of
  its own step, never the earlier call.

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
#: The text a replaced (superseded) step's call is given: the release lines'
#: (DECISION B, G-387 round 13), the same in every language and line.
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
    #: Stored only: an EARLIER parked step the case's step followed. Each of its
    #: gated calls is asked about under ``old_<token>`` and decided by its own
    #: ``decision``; none of them may ever run.
    earlier: tuple[Call, ...] = ()
    #: The run has no step left after the step that parked (live only), so a
    #: resume ends without another model step.
    spent: bool = False
    #: Stored with an earlier step only: the earlier call (index into
    #: ``earlier``) a forged result names, recorded in the EARLIER segment,
    #: between the earlier step and the case's step (G-387 round 13).
    earlier_forged: int | None = None
    #: With an ``assistant_call`` insert only: the case call (index) a forged
    #: result names, recorded in the LATER segment, after the later step.
    later_forged: int | None = None
    #: Stored with an earlier step only (G-387 round 13b): the earlier call
    #: (index into ``earlier``) whose id, tool and arguments EXACTLY a LOSSY
    #: result carries, recorded in the case step's segment. It may answer
    #: only a same-id call of the case step.
    cross_forged: int | None = None
    #: After the first resume, a later row in the SAME segment holds a result
    #: under this call's id and the gated tool with OTHER arguments (index of
    #: the call, or None; prism-harness-py#75). Every call already has its own
    #: result by then, so the stray one ends nothing and replaces nothing.
    stray: int | None = None
    #: A superseding row (a kind in SUPERSEDES) a host records WHILE the
    #: first approved call is running (G-404, G-387 round 14), or None. It is
    #: refused (``session_locked``) and changes nothing: the call keeps its
    #: one outcome, its real result.
    concurrent: str | None = None

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
                **(
                    {"earlier_forged": self.earlier_forged}
                    if self.earlier_forged is not None
                    else {}
                ),
                **({"later_forged": self.later_forged} if self.later_forged is not None else {}),
                **({"cross_forged": self.cross_forged} if self.cross_forged is not None else {}),
                **({"stray": self.stray} if self.stray is not None else {}),
                **({"concurrent": self.concurrent} if self.concurrent is not None else {}),
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
    open_calls = [i for i, c in enumerate(calls) if c.gated and not c.prior]
    if kind == "tool_result" and not open_calls:
        kind = "system"
    when = rng.choice(("before_decision", "after_decision"))
    call = rng.choice(open_calls) if kind == "tool_result" else rng.randrange(len(calls))
    return Insert(kind, when, call)


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


def stray_result(seed: int, calls: list[Call]) -> int | None:
    """The stray result, from a stream of its own (the other draws are unchanged)."""
    rng = random.Random(seed ^ 0x141141)
    gated = [i for i, c in enumerate(calls) if c.gated]
    if not gated or rng.random() >= 0.3:
        return None
    return rng.choice(gated)


#: The token (and result) of a stray result: no call carries it.
STRAY = "stray"


def stray_row(case: Case) -> dict[str, Any]:
    assert case.stray is not None
    call = case.calls[case.stray]
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": call.id,
                "tool_name": GATED,
                "args": {"token": STRAY},
                "result": STRAY,
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    }


def concurrent_supersede(seed: int) -> str | None:
    """The superseding row recorded while the approved call runs, from a
    stream of its own (G-387 round 14)."""
    rng = random.Random(seed ^ 0x404404)
    return rng.choice(SUPERSEDES) if rng.random() < 0.6 else None


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

    decidable = (
        [i for i, c in enumerate([c for c in calls if c.gated])]
        if path == "live"
        else list(range(len(requests)))
    )
    rng.shuffle(decidable)
    resumes = tuple(rng.choice(("text", "structured")) for _ in range(2))
    case = Case(
        seed=seed,
        path=path,
        calls=tuple(calls),
        requests=tuple(requests),
        decision_order=tuple(decidable),
        resumes=resumes,
        first=rng.choice(("text", "structured")),
        insert=insertion(seed, calls),
        earlier=tuple(earlier),
        spent=path == "live" and rng.random() < 0.8,
    )
    return replace(
        forgeries(case), stray=stray_result(seed, calls), concurrent=concurrent_supersede(seed)
    )


def forgeries(case: Case) -> Case:
    """The NO-BLIND-SPOT combinations (G-387 round 13): a later step together
    with inserted rows, with forged results in the earlier segment and in the
    later one. From a stream of their own, so every other draw of a seed is
    the release line's."""
    rng = random.Random(case.seed ^ 0x13013)
    earlier_open = [i for i, c in enumerate(case.earlier) if c.gated and not c.prior]
    case_open = [i for i, c in enumerate(case.calls) if c.gated and not c.prior]
    earlier_forged = rng.choice(earlier_open) if earlier_open and rng.random() < 0.6 else None
    later_forged = (
        rng.choice(case_open)
        if case.insert is not None
        and case.insert.kind == "assistant_call"
        and case_open
        and rng.random() < 0.7
        else None
    )
    # A LOSSY result in the case step's segment naming an earlier call
    # exactly (G-387 round 13b), from a stream of its own again. Never with a
    # forged ``tool_result`` insert, whose target it could share.
    lossy = random.Random(case.seed ^ 0x13B13B)
    earlier_gated = [i for i, c in enumerate(case.earlier) if c.gated]
    cross_forged = (
        lossy.choice(earlier_gated)
        if earlier_gated
        and (case.insert is None or case.insert.kind != "tool_result")
        and lossy.random() < 0.6
        else None
    )
    return replace(
        case,
        earlier_forged=earlier_forged,
        later_forged=later_forged,
        cross_forged=cross_forged,
    )


def lossy_text(call: Call) -> str:
    return f"lossy:{call.token}"


def cross_target(case: Case) -> int | None:
    """The case call a :attr:`Case.cross_forged` result answers: the first
    call of the case step, in call order, with its id and the gated tool's
    name (the result has no exact match in its own segment), else the first
    with its id. None when no call of the step carries the id."""
    if case.cross_forged is None:
        return None
    wanted = case.earlier[case.cross_forged].id
    named = [i for i, c in enumerate(case.calls) if c.id == wanted and c.gated]
    free = [i for i, c in enumerate(case.calls) if c.id == wanted]
    found = named or free
    return found[0] if found else None


# -- the oracle: an independent model of the rule ----------------------------------


@dataclass
class Expected:
    #: Tokens of the gated calls that run on the first resume.
    runs: list[str] = field(default_factory=list)
    #: Each gated call's one outcome text, by token.
    outcomes: dict[str, str] = field(default_factory=dict)
    repeated: bool = False


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
    # was decided before or after (G-397). A forged result IN THE CALL'S OWN
    # SEGMENT is its only outcome; a result in another segment never is
    # (G-387 round 13b). Any other row changes nothing (G-399).
    superseded = case.insert is not None and case.insert.kind in SUPERSEDES
    forged = (
        case.insert.call if case.insert is not None and case.insert.kind == "tool_result" else -1
    )
    # The lossy result answers the call of ITS segment it falls to; a call
    # that already has a result keeps it (the first result stands), and the
    # earlier call it names exactly keeps its own outcome.
    lossy = cross_target(case)

    # A later step replaced the earlier one: its gated calls never run, and
    # each one with no result gets one denial.
    for index, call in enumerate(case.earlier):
        if call.gated:
            expected.outcomes[call.token] = (
                f"prior:{call.token}"
                if call.prior
                else f"forged:{call.token}"
                if index == case.earlier_forged
                else REPLACED_BY_LATER_STEP
            )

    for index, call in enumerate(case.calls):
        if not call.gated:
            continue
        if call.prior:
            expected.outcomes[call.token] = f"prior:{call.token}"
            continue
        if index == lossy:
            assert case.cross_forged is not None
            expected.outcomes[call.token] = lossy_text(case.earlier[case.cross_forged])
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
    def __init__(
        self,
        name: str,
        ran: list[tuple[str, str]],
        during: list[Callable[[], None]] | None = None,
    ) -> None:
        self._name = name
        self._ran = ran
        #: Host code run once, inside the next call of this tool.
        self._during = during if during is not None else []

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        token = str(args.get("token", ""))
        self._ran.append((self._name, token))
        while self._during:
            self._during.pop(0)()
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
    case: Case, seen: list[LlmRequest] | None = None
) -> Callable[[LlmRequest], LlmResponse]:
    """Asks for the case's calls once (live), then only ever answers."""
    state = {"asked": case.path == "stored"}

    def client(request: LlmRequest) -> LlmResponse:
        if seen is not None:
            seen.append(request)
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


def stored_rows(calls: tuple[Call, ...], requests: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """One step as another writer left it: the assistant row and its results."""
    rows: list[dict[str, Any]] = [
        {
            "type": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": c.id,
                    "name": GATED if c.gated else UNGATED,
                    "arguments": {"token": c.token},
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
                        "args": {"token": c.token},
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
            case.earlier, [(f"old_{c.token}", c.id) for c in case.earlier if c.gated]
        )
        if case.earlier_forged is not None:
            rows.append(forged_row(case.earlier[case.earlier_forged]))
    rows += stored_rows(case.calls, [(r.approval_id, r.tool_call_id) for r in case.requests])
    session.thread().record(rows)
    if case.cross_forged is not None:
        # In the CASE step's segment, after any result the step already has.
        session.thread().record([lossy_row(case.earlier[case.cross_forged])])


def lossy_row(call: Call) -> dict[str, Any]:
    """A result another writer stored with LOSSY arguments: exactly those of
    the earlier ``call``, with its id and the gated tool, in a later segment."""
    row = forged_row(call)
    row["tool_results"][0]["result"] = lossy_text(call)
    return row


def forged_row(call: Call) -> dict[str, Any]:
    """A host-written result for ``call``, as another writer could store it."""
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": call.id,
                "tool_name": GATED,
                "args": {"token": call.token},
                "result": f"forged:{call.token}",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    }


def insert_between(session: Session, case: Case, when: str) -> None:
    if case.insert is not None and case.insert.when == when:
        session.thread().record([inserted_row(case)])
        if case.later_forged is not None:
            # In the LATER segment: after the later step. It names a call of
            # the replaced step, so it belongs to no call of the step it is
            # recorded in, and the replaced call keeps its one denial.
            session.thread().record([forged_row(case.calls[case.later_forged])])


def outcomes_by_token(session: Session) -> dict[str, list[str]]:
    """Every stored result, by the token of the call it answers, as STORED (never
    the provider view, which folds results by call id), from EVERY row after
    the first step through the end of the thread, each attributed by ABSOLUTE
    SEGMENT OWNERSHIP (G-387 round 13b) to the step it was recorded in (the
    last step with tool calls before it): a result counts for a call of THAT
    step only, the one its arguments name, else, within its row, the first
    not yet matched with its id and tool name, else with its id. A resume row
    restates the results it found, so one outcome may be stored more than
    once; what may not happen is two DIFFERENT outcomes, or none."""
    found: dict[str, list[str]] = {}
    step: list[tuple[str, str, str]] = []  # (id, tool, token) per call
    for entry in session.thread().messages():
        row = entry.message
        if row.get("type") == "assistant" and row.get("tool_calls"):
            step = [
                (
                    str(call.get("id", "")),
                    str(call.get("name", "")),
                    str((call.get("arguments") or {}).get("token", "")),
                )
                for call in row["tool_calls"]
                if isinstance(call, dict)
            ]
            continue
        if row.get("type") != "tool_result":
            continue
        taken: set[int] = set()
        for result in row.get("tool_results", []):
            token = str((result.get("args") or {}).get("token", ""))
            if token == STRAY:
                continue  # no call carries it: it answers nothing
            same_id = [
                i
                for i, (call_id, _, _) in enumerate(step)
                if i not in taken and call_id == str(result.get("tool_call_id", ""))
            ]
            exact = [i for i in same_id if step[i][2] == token]
            named = [i for i in same_id if step[i][1] == result.get("tool_name")]
            match = next(iter(exact or named or same_id), None)
            if match is None:
                continue
            taken.add(match)
            found.setdefault(step[match][2], []).append(str(result.get("result")))
    return found


def late_decision(session: Session, approval: str, approved: bool, reason: str | None) -> None:
    try:
        record_approval(session, approval, approved, reason)
    except HarnessError as error:
        if error.code != "approval_not_pending":
            raise


def late(session: Session, approval: str) -> None:
    """Approve once the step is settled. Refusing the decision is as good as
    ignoring it; running the call is not."""
    try:
        record_approval(session, approval, True)
    except HarnessError as error:
        if error.code != "approval_not_pending":
            raise


def decide(
    session: Session,
    case: Case,
    approval: str,
    approved: bool,
    reason: str | None,
    broken: list[str],
) -> bool:
    """Record one decision. After a row that superseded the step it must be
    refused, ``approval_not_pending``; a repeated approval id is answered
    once (the second answer is refused the same way)."""
    superseded = (
        case.insert is not None
        and case.insert.kind in SUPERSEDES
        and case.insert.when == "before_decision"
    )
    try:
        record_approval(session, approval, approved, reason)
    except HarnessError as error:
        if error.code == "approval_not_pending" and (
            superseded or [r.approval_id for r in case.requests].count(approval) > 1
        ):
            return False
        raise
    if superseded and case.insert is not None:
        broken.append(f"G-397: a decision was recorded after a {case.insert.kind} row")
    return True


def check(case: Case) -> list[str]:
    """Run one case and return every invariant it breaks."""
    expected = oracle(case)
    ran: list[tuple[str, str]] = []
    during: list[Callable[[], None]] = []
    requests: list[LlmRequest] = []
    tools = (
        ToolRegistry()
        .register(TokenTool(GATED, ran, during))
        .register(TokenTool(UNGATED, ran))
    )
    runtime = AgentRuntime(
        client=a_model(case, requests), modes=SPENT if case.spent else MODES, tools=tools
    )
    with tempfile.TemporaryDirectory(
        prefix="prism-harness-pairing-", ignore_cleanup_errors=True
    ) as directory:
        broken = _check(case, a_session(directory), runtime, ran, expected, during)
    return broken + replayed_twice(requests)


def replay_key(entry: dict[str, Any]) -> str:
    """A replayed result's identity: id + tool name + canonical arguments,
    the key the replay fold uses (G-403)."""
    return json.dumps(
        [entry.get("tool_call_id"), entry.get("tool_name"), entry.get("args")],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def replayed_twice(requests: list[LlmRequest]) -> list[str]:
    """Item 5 (G-387 round 14): every provider request replays each result
    ONCE. Within a step's segment no result key is emitted twice, and across
    the whole replay no result (key and text) is emitted twice."""
    broken: list[str] = []
    for number, request in enumerate(requests):
        whole: dict[tuple[str, str], int] = {}
        segment: dict[str, int] = {}
        for message in request.messages:
            if message.get("type") == "assistant" and message.get("tool_calls"):
                segment = {}
                continue
            if message.get("type") != "tool_result":
                continue
            for entry in message.get("tool_results", []):
                key = replay_key(entry)
                segment[key] = segment.get(key, 0) + 1
                pair = (key, str(entry.get("result")))
                whole[pair] = whole.get(pair, 0) + 1
                if segment[key] == 2 or whole[pair] == 2:
                    broken.append(f"item 5: request {number} replays {key} more than once")
    return broken


def _check(
    case: Case,
    session: Session,
    runtime: AgentRuntime,
    ran: list[tuple[str, str]],
    expected: Expected,
    during: list[Callable[[], None]] | None = None,
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
            if decide(session, case, approval, call.decision == "approve", call.reason, broken):
                decided[approval] = (call.decision, call.reason)
    else:
        store_step(session, case)
        for old in case.earlier:
            if old.gated and old.decision != "none":
                # The earlier step is not the step whose approvals are
                # requested (G-397): refused is as good as ignored.
                late_decision(session, f"old_{old.token}", old.decision == "approve", old.reason)
        insert_between(session, case, "before_decision")
        for index in case.decision_order:
            request = case.requests[index]
            if request.decision == "none":
                continue
            if decide(
                session,
                case,
                request.approval_id,
                request.decision == "approve",
                request.reason,
                broken,
            ):
                decided[request.approval_id] = (request.decision, request.reason)
        for call in gated:
            own = [r.approval_id for r in case.requests if r.tool_call_id == call.id]
            approval_of[call.token] = own[0] if len(own) == 1 else None

    insert_between(session, case, "after_decision")
    refused: list[str] = []
    if case.concurrent is not None and during is not None:
        row = inserted_row(replace(case, insert=Insert(case.concurrent, "during", 0)))

        def supersede() -> None:
            try:
                session.thread().record([row])
                refused.append("recorded")
            except HarnessError as error:
                refused.append(str(error.code))

        during.append(supersede)
    before = len(ran)
    resumed = a_send(runtime, session, case.resumes[0], "")
    if during:
        during.clear()  # no approved call ran: nothing was in flight
    elif case.concurrent is not None and refused != ["session_locked"]:
        broken.append(f"G-404: a {case.concurrent} row during the approved call was {refused}")
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
    stored = outcomes_by_token(session)
    for call in [c for c in case.earlier if c.gated]:
        recorded = stored.get(call.token, [])
        if recorded != [expected.outcomes[call.token]] or counts.get(call.token):
            broken.append(
                f"1/5: earlier-step call {call.token} ran {counts.get(call.token, 0)}x and is "
                f"recorded as {recorded}, the rule says once {expected.outcomes[call.token]!r}"
            )
    for call in gated:
        texts = set(stored.get(call.token, []))
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
    case = replace(case, insert=insert)
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
    return [base + i for i in range(count)]


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
    assert seeds() == [DEFAULT_BASE_SEED + i for i in range(DEFAULT_EXAMPLES)]
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


def test_the_generator_reaches_every_branch_of_the_rule() -> None:
    # A property that never generates the hard case passes vacuously.
    examined = [generate(seed) for seed in range(DEFAULT_BASE_SEED, DEFAULT_BASE_SEED + 300)]
    expectations = [oracle(case) for case in examined]
    assert any(e.repeated and any(c.gated for c in k.calls) for k, e in zip(examined, expectations))
    assert any(not e.repeated and e.runs for e in expectations)
    assert any(c.id == "" for k in examined for c in k.calls)
    assert any(len({r.tool_call_id for r in k.requests}) != len(k.requests) for k in examined)
    assert any(k.path == "stored" and any(c.gated and c.prior for c in k.calls) for k in examined)
    assert any(
        k.path == "stored"
        and any(c.gated and not any(r.tool_call_id == c.id for r in k.requests) for c in k.calls)
        for k in examined
    )
    assert any("structured" in k.resumes for k in examined)
    assert any(k.path == "live" and k.first == "structured" for k in examined)
    # Every inserted kind, before and after the decision (as prism-harness-ts).
    for kind in set(INSERT_KINDS):
        for when in ("before_decision", "after_decision"):
            assert any(
                k.insert is not None and k.insert.kind == kind and k.insert.when == when
                for k in examined
            ), (kind, when)
    assert any(any(c.gated and not c.prior for c in k.earlier) for k in examined)
    assert any(len({r.approval_id for r in k.requests}) != len(k.requests) for k in examined)
    # The shape that ran an approved call again (G-399): a row that does not
    # supersede, recorded after the decision, and no model step after the
    # resume. Reached by the FIXED set the normal test run uses.
    fast = [
        generate(seed) for seed in range(DEFAULT_BASE_SEED, DEFAULT_BASE_SEED + DEFAULT_EXAMPLES)
    ]
    assert any(
        k.spent
        and k.insert is not None
        and k.insert.kind in ("system", "tool_empty")
        and k.insert.when == "after_decision"
        and oracle(k).runs
        for k in fast
    )
    # NO BLIND SPOT (G-387 round 13): a later step together with every kind
    # of inserted row, and forged results in the earlier and in the later
    # segment, each generated at least once, or the run fails.
    later = [k for k in examined if k.earlier]
    for kind in set(INSERT_KINDS):
        assert any(k.insert is not None and k.insert.kind == kind for k in later), kind
    assert any(k.earlier_forged is not None and k.insert is not None for k in later)
    assert any(
        k.earlier_forged is not None and k.insert is not None and k.insert.kind in SUPERSEDES
        for k in later
    )
    assert any(k.later_forged is not None for k in examined)
    assert any(k.later_forged is not None and k.earlier for k in examined) or any(
        k.later_forged is not None and k.path == "live" for k in examined
    )
    outcomes = {text for e in expectations for text in e.outcomes.values()}
    assert {REPEATED, NO_DECISION, DEFAULT_DENIAL, REPLACED_BY_LATER_STEP} <= outcomes
    assert any(text.startswith("forged:") for text in outcomes)
    # SEGMENT OWNERSHIP (G-387 round 13b): a lossy result naming an earlier
    # call exactly, in the later segment, that answers an open same-id gated
    # call of its own step, that falls to a call already resulted, and that
    # finds no call; naming an earlier call that is open and one already
    # resulted; together with superseding rows and earlier-segment forgeries.
    cross = [k for k in examined if k.cross_forged is not None]
    hits = [(k, t) for k in cross for t in [cross_target(k)] if t is not None]
    assert any(k.calls[t].gated and not k.calls[t].prior for k, t in hits)
    assert any(k.calls[t].prior for k, t in hits)
    assert any(cross_target(k) is None for k in cross)
    assert any(not k.earlier[k.cross_forged].prior for k, _ in hits if k.cross_forged is not None)
    assert any(k.earlier[k.cross_forged].prior for k, _ in hits if k.cross_forged is not None)
    assert any(k.insert is not None and k.insert.kind in SUPERSEDES for k, _ in hits)
    assert any(k.earlier_forged is not None for k, _ in hits)
    assert any(text.startswith("lossy:") for text in outcomes)
    # A stray later result in a step with repeated ids (G-403, replay), and
    # in one whose approved calls run.
    assert any(k.stray is not None and oracle(k).repeated for k in examined)
    assert any(k.stray is not None and oracle(k).runs for k in examined)
    # A superseding row of every kind while an approved call runs (G-404);
    # CONCURRENT_SEEDS pins one of each in every normal run.
    for kind in SUPERSEDES:
        assert any(k.concurrent == kind and oracle(k).runs for k in examined), kind
    assert sorted(
        generate(seed).concurrent or "" for seed in CONCURRENT_SEEDS if oracle(generate(seed)).runs
    ) == sorted(SUPERSEDES)


# -- fixed regression cases (G-387 round 13) ----------------------------------------

#: The seeds that broke invariant 5 on main 6a3f2c2: a gated call sharing its
#: id with an ungated call that already has a result got NO outcome, because
#: a result was matched to its call by id.
REGRESSION_SEEDS = (918, 930)

#: One seed per superseding kind recorded while an approved call runs
#: (G-404, G-387 round 14): user, assistant_call, assistant_text.
CONCURRENT_SEEDS = (950, 951, 978)


@pytest.mark.parametrize("seed", CONCURRENT_SEEDS)
def test_a_supersede_while_the_approved_call_runs_obeys_the_rule(seed: int) -> None:
    case = generate(seed)
    assert case.concurrent is not None and oracle(case).runs
    assert safe_check(case) == [], case.describe()


@pytest.mark.parametrize("seed", REGRESSION_SEEDS)
def test_the_seeds_that_broke_main_obey_the_rule(seed: int) -> None:
    case = generate(seed)
    assert check(case) == [], case.describe()


def _stored(calls: tuple[Call, ...], requests: tuple[Request, ...]) -> Case:
    return Case(
        seed=0,
        path="stored",
        calls=calls,
        requests=requests,
        decision_order=tuple(range(len(requests))),
    )


#: The three shapes the TypeScript property test found on its main, as stored
#: steps: (a) a gated call with no request sharing its id with an ungated call
#: that has a result; (b) two requests naming one call while a call with
#: another id already has a result; (c) two requests naming a gated call that
#: shares its id with an ungated call that already ran.
TYPESCRIPT_SHAPES = {
    "gated-without-request-beside-its-resulted-twin": _stored(
        (Call("a", True, "t0"), Call("a", False, "t1", prior=True)), ()
    ),
    "repeated-requests-beside-a-resulted-call": _stored(
        (Call("b", True, "t0"), Call("", True, "t1", prior=True)),
        (
            Request("apr_1", "b", "approve", None),
            Request("apr_2", "b", "approve", None),
            Request("apr_3", "", "deny", "no-48"),
        ),
    ),
    "two-requests-and-an-ungated-twin": _stored(
        (Call("c", True, "t0"), Call("c", False, "t1", prior=True)),
        (Request("apr_1", "c", "deny", "no-504"), Request("apr_2", "c", "deny", "no-611")),
    ),
}


@pytest.mark.parametrize("shape", sorted(TYPESCRIPT_SHAPES))
def test_the_shapes_found_on_typescript_main_obey_the_rule(shape: str) -> None:
    case = TYPESCRIPT_SHAPES[shape]
    assert check(case) == [], case.describe()


#: Seeds whose lossy later-segment result (G-387 round 13b) falls to an OPEN
#: gated call of its own step, from the coverage window: the fast set the
#: normal run examines reaches none, so they run every time.
SEGMENT_SEEDS = (997, 1002, 1008, 1013, 1035, 1094, 1129, 1192, 1193, 1198, 1202, 1212)


@pytest.mark.parametrize("seed", SEGMENT_SEEDS)
def test_the_lossy_later_segment_seeds_obey_the_rule(seed: int) -> None:
    case = generate(seed)
    target = cross_target(case)
    assert target is not None and case.calls[target].gated and not case.calls[target].prior
    assert check(case) == [], case.describe()


def _lossy(
    calls: tuple[Call, ...], requests: tuple[Request, ...], earlier: tuple[Call, ...]
) -> Case:
    return replace(_stored(calls, requests), seed=0x13B, earlier=earlier, cross_forged=0)


#: ABSOLUTE SEGMENT OWNERSHIP (G-387 round 13b), as stored steps: an earlier
#: step's call ``x`` and the case step's APPROVED call ``x``, with a lossy
#: result in the case step's segment carrying the earlier call's arguments
#: exactly. A rule that moved it to the earlier call would run the approved
#: call (and give the earlier call two outcomes); here it answers the case
#: call, which never runs, and the earlier call keeps its one denial.
SEGMENT_SHAPES = {
    "approved-newest-earlier-open": _lossy(
        (Call("x", True, "t0"),),
        (Request("apr_1", "x", "approve", None),),
        (Call("x", True, "e0", decision="approve"),),
    ),
    "approved-newest-earlier-resulted": _lossy(
        (Call("x", True, "t0"),),
        (Request("apr_1", "x", "approve", None),),
        (Call("x", True, "e0", decision="approve", prior=True),),
    ),
    "approved-newest-beside-an-approved-sibling": _lossy(
        (Call("x", True, "t0"), Call("b", True, "t1")),
        (Request("apr_1", "x", "approve", None), Request("apr_2", "b", "approve", None)),
        (Call("x", True, "e0", decision="deny"),),
    ),
}


@pytest.mark.parametrize("shape", sorted(SEGMENT_SHAPES))
def test_a_lossy_result_in_a_later_segment_answers_only_its_own_step(shape: str) -> None:
    case = SEGMENT_SHAPES[shape]
    assert oracle(case).outcomes["t0"] == "lossy:e0"
    assert check(case) == [], case.describe()


def test_a_structured_first_step_that_parks_obeys_the_rule() -> None:
    """A structured send whose FIRST step stops for a person returns the
    structured shape with the pending approvals and no document (it never
    asserts), and the step then obeys the rule like any other."""
    live = [case for case in map(generate, range(0x392, 0x392 + 60)) if case.path == "live"]
    assert live
    for case in live:
        structured = replace(case, first="structured")
        assert check(structured) == [], structured.describe()
