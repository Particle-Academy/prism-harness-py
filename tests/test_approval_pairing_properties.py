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
from prism_harness.thread_rows import pending_approval_ids

REPEATED = "Tool call was not run because another call in this step has the same id"
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
    #: The shape of the first send (live only).
    first: str = "text"
    #: A row recorded between the park and the decisions (G-397): none, or an
    #: assistant, user, system or tool row (see ``INSERTED``).
    inserted: str = "none"

    def describe(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "path": self.path,
                "first": self.first,
                **({"inserted": self.inserted} if self.inserted != "none" else {}),
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
            }
        )


#: The rows G-397 records between the park and the decisions (``Thread.record``
#: is public). An assistant row or a user row ends the step a decision may
#: apply to: every decision after it is refused, and the resume closes the
#: step's calls as undecided. A system row, or a tool result for a call that is
#: not in the step, changes nothing.
INSERTED: dict[str, dict[str, Any]] = {
    "assistant": {
        "type": "assistant",
        "content": "An aside.",
        "tool_calls": [],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    "user": {"type": "user", "content": "Something else."},
    "system": {"type": "system", "content": "A note."},
    "tool": {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "elsewhere",
                "tool_name": GATED,
                "args": {"token": "elsewhere"},
                "result": "elsewhere",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    },
}
#: The insertions that end the step.
ENDS_THE_STEP = ("assistant", "user")


def an_insertion(seed: int) -> str:
    """Drawn from a generator of its own, so every other draw of a seed, and
    so every seed's case, is what it was before the insertion existed."""
    return random.Random(f"g397:{seed}").choice(
        ("none", "none", "assistant", "user", "system", "tool")
    )


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
        if rng.random() < 0.5:
            rng.shuffle(requests)

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
        # The step that parks is taken through send(): a structured turn that
        # stops for a person is not a shape this release returns, so only
        # the resumes vary their shape.
        first="text",
        inserted=an_insertion(seed),
    )


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
    repeated = len(set(ids)) != len(ids) or len(set(asked)) != len(asked)
    expected = Expected(repeated=repeated)

    for call in case.calls:
        if not call.gated:
            continue
        if call.prior:
            expected.outcomes[call.token] = f"prior:{call.token}"
            continue
        if repeated:
            expected.outcomes[call.token] = REPEATED
            continue
        if case.inserted in ENDS_THE_STEP:
            # No decision applies once the step has ended (G-397).
            expected.outcomes[call.token] = NO_DECISION
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

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        token = str(args.get("token", ""))
        self._ran.append((self._name, token))
        return f"{self._name}:{token}"


def a_session(directory: str) -> Session:
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("support")
    session.using_mode("guarded").using_provider("anthropic").using_model("claude-sonnet-5-5")
    return session


def a_model(case: Case) -> Callable[[LlmRequest], LlmResponse]:
    """Asks for the case's calls once (live), then only ever answers."""
    state = {"asked": case.path == "stored"}

    def client(request: LlmRequest) -> LlmResponse:
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


def store_step(session: Session, case: Case) -> None:
    """The step as another writer left it in the thread."""
    rows: list[dict[str, Any]] = [
        {"type": "user", "content": "go"},
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
                for c in case.calls
            ],
            "additional_content": {},
            "tool_approval_requests": [
                {
                    "approval_id": r.approval_id,
                    "tool_call_id": r.tool_call_id,
                    "arguments_format": 1,
                }
                for r in case.requests
            ],
            "harness_tool_names": [GATED, UNGATED],
        },
    ]
    priors = [c for c in case.calls if c.prior]
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
    session.thread().record(rows)


def insert_row(session: Session, case: Case, broken: list[str]) -> None:
    """G-397: the case's row, recorded between the park and the decisions."""
    if case.inserted == "none":
        return
    session.thread().record([json.loads(json.dumps(INSERTED[case.inserted]))])
    if case.inserted in ENDS_THE_STEP and session.thread() is not None:
        rows = [entry.message for entry in session.thread().messages()]
        if pending_approval_ids(rows):
            broken.append(f"G-397: approvals still pending after a {case.inserted} row")


def a_decision_is_recorded(
    session: Session,
    case: Case,
    approval: str,
    approved: bool,
    reason: str | None,
    broken: list[str],
) -> bool:
    """Record one decision; after a row that ended the step it must be
    refused (``approval_not_pending``), and nowhere else."""
    try:
        record_approval(session, approval, approved, reason)
    except HarnessError as error:
        if case.inserted in ENDS_THE_STEP and error.code == "approval_not_pending":
            return False
        raise
    if case.inserted in ENDS_THE_STEP:
        broken.append(f"G-397: a decision was recorded after a {case.inserted} row ended the step")
    return True


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
            token = str((result.get("args") or {}).get("token", ""))
            found.setdefault(token, []).append(str(result.get("result")))
    return found


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
    tools = ToolRegistry().register(TokenTool(GATED, ran)).register(TokenTool(UNGATED, ran))
    runtime = AgentRuntime(client=a_model(case), modes=MODES, tools=tools)
    with tempfile.TemporaryDirectory(
        prefix="prism-harness-pairing-", ignore_cleanup_errors=True
    ) as directory:
        return _check(case, a_session(directory), runtime, ran, expected)


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
        insert_row(session, case, broken)
        for index in case.decision_order:
            call = gated[index]
            if call.decision == "none":
                continue
            approval = approval_of[call.token]
            assert approval is not None
            if a_decision_is_recorded(
                session, case, approval, call.decision == "approve", call.reason, broken
            ):
                decided[approval] = (call.decision, call.reason)
    else:
        store_step(session, case)
        insert_row(session, case, broken)
        for index in case.decision_order:
            request = case.requests[index]
            if request.decision == "none":
                continue
            if a_decision_is_recorded(
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
    # G-399: a row that ends no step (system, tool) is recorded again between
    # the resumes, so a result is never hidden from the next one.
    if case.inserted not in ("none", *ENDS_THE_STEP):
        insert_row(session, case, broken)
    after = len(ran)
    a_send(runtime, session, case.resumes[1], "")
    for approval in [a for a in approval_of.values() if a is not None] + [
        r.approval_id for r in case.requests
    ]:
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


def shrink(case: Case) -> tuple[Case, list[str]]:
    broken = check(case)
    progress = True
    while progress:
        progress = False
        candidates = [without_call(case, i) for i in range(len(case.calls))] + [
            without_request(case, i) for i in range(len(case.requests))
        ]
        for candidate in candidates:
            if not candidate.calls:
                continue
            found = check(candidate)
            if found:
                case, broken, progress = candidate, found, True
                break
    return case, broken


def seeds() -> list[int]:
    replay = os.environ.get("PRISM_PROPERTY_SEED")
    if replay:
        return [int(replay, 0)]
    base = int(os.environ.get("PRISM_PROPERTY_BASE_SEED", "0x392"), 0)
    count = int(os.environ.get("PRISM_PROPERTY_EXAMPLES", "300"))
    return [base + i for i in range(count)]


def test_every_generated_step_obeys_the_approval_pairing_rule() -> None:
    examined = seeds()
    print(f"approval pairing properties: {len(examined)} examples from seed {examined[0]}")
    for seed in examined:
        case = generate(seed)
        broken = check(case)
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
    examined = [generate(seed) for seed in range(0x392, 0x392 + 300)]
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
    # Every G-397 insertion, on both paths, with repeated ids and without.
    for kind in INSERTED:
        for path in ("live", "stored"):
            assert any(k.inserted == kind and k.path == path for k in examined), (kind, path)
        assert any(k.inserted == kind and e.repeated for k, e in zip(examined, expectations))
        assert any(k.inserted == kind and not e.repeated for k, e in zip(examined, expectations))
    outcomes = {text for e in expectations for text in e.outcomes.values()}
    assert {REPEATED, NO_DECISION, DEFAULT_DENIAL} <= outcomes


# -- fixed regression cases (G-387 round 13) ----------------------------------------

#: The seeds that broke invariant 5 on main 6a3f2c2: a gated call sharing its
#: id with an ungated call that already has a result got NO outcome, because
#: a result was matched to its call by id.
REGRESSION_SEEDS = (918, 930)


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


def test_a_structured_first_step_that_parks_obeys_the_rule() -> None:
    """A structured send whose FIRST step stops for a person returns the
    structured shape with the pending approvals and no document (it never
    asserts), and the step then obeys the rule like any other."""
    live = [case for case in map(generate, range(0x392, 0x392 + 60)) if case.path == "live"]
    assert live
    for case in live:
        structured = replace(case, first="structured")
        assert check(structured) == [], structured.describe()
