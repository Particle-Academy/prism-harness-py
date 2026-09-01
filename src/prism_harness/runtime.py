"""The loop: prompt in, turns out, everything recorded."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from prism_harness.errors import HarnessError
from prism_harness.events import HarnessEvents, RunFailed, RunFinished, RunStarted
from prism_harness.modes import AgentMode, ModeRegistry
from prism_harness.session import Session
from prism_harness.subagents import RunBudget, RunContext
from prism_harness.tools import HarnessTool, ToolAuthorizer, ToolRegistry

__all__ = [
    "AgentResponse",
    "AgentRuntime",
    "LlmClient",
    "LlmRequest",
    "LlmResponse",
    "LlmToolCall",
    "PendingApproval",
    "record_approval",
]


@dataclass(frozen=True)
class LlmToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmRequest:
    """What the runtime needs from a model, and NOTHING MORE.

    An INTERFACE rather than a dependency on ``prism-ai``. The loop below --
    steps, budgets, approvals, thread recording, events -- is the part worth
    porting, and none of it needs to know how a request reaches a provider.
    Keeping the seam here also means this package stays at zero dependencies and
    a consumer can drive it with ``prism-ai``, their own client, or a fake.

    The reference couples these because Prism is already a dependency there.
    """

    system_prompt: str
    #: The conversation so far, serialised -- oldest first.
    messages: list[dict[str, Any]]
    tools: list[HarnessTool]
    provider: str
    model: str


@dataclass(frozen=True)
class LlmResponse:
    text: str
    #: The provider's own reason, passed through: ``stop``, ``tool_calls``, ...
    finish_reason: str
    tool_calls: list[LlmToolCall] = field(default_factory=list)
    #: None when the provider does not report one. NOT zero -- see
    #: :meth:`RunLedger.record_cost`.
    cost_usd: float | None = None


LlmClient = Callable[[LlmRequest], LlmResponse]


@dataclass(frozen=True)
class PendingApproval:
    id: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentResponse:
    run_id: str
    text: str
    steps: int
    #: NAMES only, in call order.
    tool_calls: list[str]
    finish_reason: str
    #: Set when the run stopped because a tool needs a human.
    pending_approvals: list[PendingApproval] = field(default_factory=list)
    #: Set when the tree ran out of budget, or was cancelled.
    stopped_because: str | None = None


class AgentRuntime:
    """Three properties matter more than the mechanics.

    **Every step is checked against the budget BEFORE it is taken**, not after.
    Checking afterwards means the step that broke the limit has already been
    paid for, which makes a budget a report rather than a control.

    **An approval stops the run and is written to the THREAD**, not held in
    memory. That is what makes it survive: the approval a person grants this
    morning is a durable row, so the worker that resumes tonight -- a different
    process, possibly after a deploy -- reads the same answer.

    **A tool that needs approval and has none DOES NOT RUN.** Failing closed is
    the only safe direction: an unanswered approval that executed anyway is
    exactly the outcome the whole mechanism exists to prevent.
    """

    def __init__(
        self,
        client: LlmClient,
        modes: ModeRegistry,
        tools: ToolRegistry,
        authorizer: ToolAuthorizer | None = None,
        events: HarnessEvents | None = None,
    ) -> None:
        self._client = client
        self._modes = modes
        self._tools = tools
        self._authorizer = authorizer
        self._events = events

    def send(
        self,
        session: Session,
        prompt: str,
        tool_names: Sequence[str] | None = None,
        context: RunContext | None = None,
    ) -> AgentResponse:
        """Run a turn.

        An EMPTY prompt is meaningful and not an error: it is how a run resumes
        after an approval, because the conversation already contains the
        request, the decision, and everything before them. A new prompt there
        would be a second instruction competing with the one the tool call came
        from.
        """
        mode = self._modes.resolve(session.mode())
        provider = session.provider() or "unknown"
        model = session.model() or "unknown"
        run_id = str(uuid.uuid4())
        run = context or RunContext.root(run_id, RunBudget(mode.max_steps))
        thread = session.thread()

        if run.too_deep():
            raise HarnessError.run_not_permitted(
                f"This run is nested {run.depth} deep, at or past the ceiling. Nobody debugs a "
                "tree that deep, and a configuration that produced one is a mistake worth "
                "reporting rather than executing."
            )

        session.begin_run(run_id, mode.name, provider, model)
        self._emit(
            RunStarted(
                run_id=run_id,
                session_key=session.key(),
                mode=mode.name,
                provider=provider,
                model=model,
                root_run_id=run.root_run_id,
                depth=run.depth,
                at=_now(),
            )
        )

        if prompt != "":
            thread.record([{"type": "user", "content": prompt}], run_id)

        try:
            return self._loop(session, mode, run, run_id, provider, model, tool_names)
        except Exception as error:
            failure = str(error)
            session.fail_run(run_id, failure)
            self._emit(
                RunFailed(
                    run_id=run_id,
                    session_key=session.key(),
                    failure=failure,
                    steps=run.ledger.steps,
                    at=_now(),
                )
            )
            raise

    def _loop(
        self,
        session: Session,
        mode: AgentMode,
        run: RunContext,
        run_id: str,
        provider: str,
        model: str,
        tool_names: Sequence[str] | None,
    ) -> AgentResponse:
        thread = session.thread()
        resolved = self._tools.resolve(list(tool_names) if tool_names else mode.tools, session)
        offered = (
            self._authorizer.allowed(session, resolved)
            if self._authorizer is not None
            else list(resolved.values())
        )

        called: list[str] = []
        text = ""
        finish_reason = "stop"

        while True:
            # BEFORE the step, never after. Checking afterwards means the step
            # that broke the limit has already been paid for.
            exhausted = run.ledger.exhaustion(run.budget)

            if exhausted is not None:
                return self._finish(
                    session, run_id, called, "budget_exhausted", text, run, exhausted
                )

            response = self._client(
                LlmRequest(
                    system_prompt=mode.system_prompt,
                    messages=[entry.message for entry in thread.messages()],
                    tools=offered,
                    provider=provider,
                    model=model,
                )
            )

            run.ledger.record_steps(1)
            run.ledger.record_cost(response.cost_usd)
            text = response.text
            finish_reason = response.finish_reason

            thread.record(
                [
                    {
                        "type": "assistant",
                        "content": response.text,
                        "tool_calls": [
                            {"id": call.id, "name": call.name} for call in response.tool_calls
                        ],
                    }
                ],
                run_id,
            )

            if not response.tool_calls:
                return self._finish(session, run_id, called, finish_reason, text, run, None)

            pending = self._pending_approvals(session, mode, response.tool_calls)

            if pending:
                # FAILS CLOSED. The run stops here and the request is already in
                # the thread, so a different process can pick it up after a human
                # answers.
                thread.record(
                    [
                        {
                            "type": "tool_approval_request",
                            "approvals": [
                                {"id": p.id, "tool": p.tool, "arguments": p.arguments}
                                for p in pending
                            ],
                        }
                    ],
                    run_id,
                )

                return AgentResponse(
                    run_id=run_id,
                    text=text,
                    steps=run.ledger.steps,
                    tool_calls=called,
                    finish_reason="awaiting_approval",
                    pending_approvals=pending,
                )

            for call in response.tool_calls:
                called.append(call.name)
                thread.record([self._invoke(offered, call)], run_id)

    def _pending_approvals(
        self, session: Session, mode: AgentMode, tool_calls: Sequence[LlmToolCall]
    ) -> list[PendingApproval]:
        """Which of these calls needs a human, and has not had one.

        An approval already answered in the thread is NOT asked again -- that is
        the whole point of recording it durably. An answered-and-denied approval
        is also not asked again; it is simply not executed.
        """
        gated = [call for call in tool_calls if mode.needs_approval(call.name)]

        if not gated:
            return []

        answered = self._answered_approvals(session)

        return [
            PendingApproval(id=call.id, tool=call.name, arguments=call.arguments)
            for call in gated
            if call.id not in answered
        ]

    @staticmethod
    def _answered_approvals(session: Session) -> dict[str, bool]:
        answered: dict[str, bool] = {}

        for entry in session.thread().messages():
            if entry.message.get("type") != "tool_approval_response":
                continue

            approval_id = entry.message.get("approval_id")
            if isinstance(approval_id, str):
                answered[approval_id] = entry.message.get("approved") is True

        return answered

    @staticmethod
    def _invoke(offered: Sequence[HarnessTool], call: LlmToolCall) -> dict[str, Any]:
        tool = next((candidate for candidate in offered if candidate.name == call.name), None)

        if tool is None:
            raise HarnessError.tool_not_available(
                call.name, [candidate.name for candidate in offered]
            )

        try:
            result = tool.handle(call.arguments)
        except HarnessError as error:
            # A refused call propagates. A refusal fed back to the model reads
            # as a retryable failure, which is the opposite of a guard.
            if error.code == "call_not_authorized":
                raise

            return _failed_result(call, str(error))
        except Exception as error:  # noqa: BLE001 - a tool is someone else's code
            # A failed tool is a RESULT, not a crashed run: the model can often
            # recover, and losing the whole turn to one bad call is worse.
            return _failed_result(call, str(error))

        return {
            "type": "tool_result",
            "tool_call_id": call.id,
            "name": call.name,
            "result": result if isinstance(result, str) else json.dumps(result),
        }

    def _finish(
        self,
        session: Session,
        run_id: str,
        called: list[str],
        finish_reason: str,
        text: str,
        run: RunContext,
        stopped_because: str | None,
    ) -> AgentResponse:
        session.complete_run(run_id, finish_reason, called)
        self._emit(
            RunFinished(
                run_id=run_id,
                session_key=session.key(),
                finish_reason=finish_reason,
                tool_calls=tuple(called),
                steps=run.ledger.steps,
                cost_usd=None if run.ledger.unmetered_runs > 0 else run.ledger.cost_usd,
                at=_now(),
            )
        )

        return AgentResponse(
            run_id=run_id,
            text=text,
            steps=run.ledger.steps,
            tool_calls=called,
            finish_reason=finish_reason,
            stopped_because=stopped_because,
        )

    def _emit(self, event: RunStarted | RunFinished | RunFailed) -> None:
        if self._events is not None:
            self._events.emit(event)


def record_approval(
    session: Session, approval_id: str, approved: bool, reason: str | None = None
) -> None:
    """Answer a pending approval, durably.

    The decision is RECORDED IN THE THREAD, not held anywhere else. Who may
    approve is the APPLICATION's decision, not this package's: the session is
    already scoped to a participant, so nobody can answer another participant's
    approval through it, but "this user may approve THIS action" is a question
    only the host can answer. Authorize before calling.
    """
    run = session.run()

    session.thread().record(
        [
            {
                "type": "tool_approval_response",
                "approval_id": approval_id,
                "approved": approved,
                "reason": reason,
            }
        ],
        run["id"] if run else None,
    )


def _failed_result(call: LlmToolCall, message: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_call_id": call.id,
        "name": call.name,
        "result": f"The tool failed: {message}",
        "failed": True,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
