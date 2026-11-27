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

#: Rows a host may record while an approval is pending (G-399).
BETWEEN: dict[str, dict[str, Any]] = {
    "assistant": {"type": "assistant", "content": "Waiting for your approval."},
    "user": {"type": "user", "content": "Hurry."},
    "system": {"type": "system", "content": "A note."},
    "other": {"type": "developer", "content": "A note."},
    "tool": {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "elsewhere",
                "tool_name": UNGATED,
                "args": {"token": "elsewhere"},
                "result": f"{UNGATED}:elsewhere",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    },
}

#: The rows after which a pending step can no longer be resumed into a run.
ENDS_THE_PAUSE = ("assistant", "user")

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
    #: The shape of the send that takes the step (live only).
    first: str = "text"
    #: A row a host records while the approval is pending (G-399), or None.
    between: str | None = None
    #: When: before-decision or after-decision.
    between_when: str = "before-decision"
    #: The run has no step left after the step that parked (live only), so a
    #: resume ends without another model step.
    spent: bool = False

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
                    {"between": self.between, "between_when": self.between_when}
                    if self.between is not None
                    else {}
                ),
                **({"spent": True} if self.spent else {}),
            }
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
        first=rng.choice(("text", "structured")),
        between=rng.choice((None, *sorted(BETWEEN))),
        between_when=rng.choice(("before-decision", "after-decision")),
        spent=path == "live" and rng.random() < 0.5,
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
        if case.path == "live":
            decision, reason = call.decision, call.reason
        else:
            own = [r for r in case.requests if r.tool_call_id == call.id]
            decision, reason = (own[0].decision, own[0].reason) if own else ("none", None)
        if decision == "approve" and case.between in ENDS_THE_PAUSE:
            # A user row or an assistant answer after the step ends the pause:
            # the approval stands, but the call is refused, never run.
            expected.outcomes[call.token] = f"Not run: {GATED} is not available to this run."
        elif decision == "approve":
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


def insert_between(session: Session, case: Case, when: str) -> None:
    if case.between is not None and case.between_when == when:
        session.thread().record([BETWEEN[case.between]])


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
    runtime = AgentRuntime(client=a_model(case), modes=SPENT if case.spent else MODES, tools=tools)
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
        insert_between(session, case, "before-decision")
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
        insert_between(session, case, "before-decision")
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

    insert_between(session, case, "after-decision")
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


def safe_check(case: Case) -> list[str]:
    """``check``, with a raise reported as a broken case rather than losing the seed."""
    try:
        return check(case)
    except Exception as error:  # noqa: BLE001 - any raise is a failure to report
        return [f"raised {type(error).__name__}: {error}"]


def shrink(case: Case) -> tuple[Case, list[str]]:
    broken = safe_check(case)
    progress = True
    while progress:
        progress = False
        candidates = [without_call(case, i) for i in range(len(case.calls))] + [
            without_request(case, i) for i in range(len(case.requests))
        ]
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
    for kind in BETWEEN:
        for when in ("before-decision", "after-decision"):
            assert any(k.between == kind and k.between_when == when for k in examined)
    # The shape that ran an approved call again (G-399): a note recorded
    # after the decision, and no model step after the resume. Reached by the
    # FIXED set the normal test run uses, not only by the long run.
    fast = [
        generate(seed) for seed in range(DEFAULT_BASE_SEED, DEFAULT_BASE_SEED + DEFAULT_EXAMPLES)
    ]
    assert any(
        k.spent
        and k.between in ("system", "other")
        and k.between_when == "after-decision"
        and oracle(k).runs
        for k in fast
    )
    outcomes = {text for e in expectations for text in e.outcomes.values()}
    assert {REPEATED, NO_DECISION, DEFAULT_DENIAL} <= outcomes
