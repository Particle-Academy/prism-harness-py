"""The loop: prompt in, turns out, everything recorded."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from prism_harness.attachments import admit_attachments
from prism_harness.errors import HarnessError
from prism_harness.events import HarnessEvents, RunFailed, RunFinished, RunStarted
from prism_harness.modes import AgentMode, ModeRegistry
from prism_harness.session import Session
from prism_harness.subagents import RunBudget, RunContext
from prism_harness.thread_rows import (
    ToolCallInput,
    assistant_row,
    thread_view,
    tool_result_entry,
    tool_result_row,
)
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
    #: The ids a provider keys the call's result and reasoning by, when they
    #: differ from ``id``. OpenAI's Responses API answers a ``function_call`` by
    #: its ``call_id``, and replays the reasoning item it came from by id.
    #: Recorded on the call, as prism's ToolCall stores them.
    result_id: str | None = None
    reasoning_id: str | None = None
    reasoning_summary: list[Any] | None = None


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
    #: The mode's ``provider_options``, unchanged. A client passes them to its
    #: provider call; the harness does not interpret them.
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmResponse:
    text: str
    #: The provider's own reason, passed through: ``stop``, ``tool_calls``, ...
    finish_reason: str
    tool_calls: list[LlmToolCall] = field(default_factory=list)
    #: None when the provider does not report one. NOT zero -- see
    #: :meth:`RunLedger.record_cost`.
    cost_usd: float | None = None
    #: What the provider needs sent back with this turn on the next request,
    #: such as Anthropic's ``thinking`` and ``thinking_signature``. Recorded with
    #: the assistant turn as ``additional_content``, the key prism's
    #: AssistantMessage uses, so a client built on prism-py can pass
    #: ``response.additional_content`` straight through.
    additional_content: dict[str, Any] = field(default_factory=dict)


LlmClient = Callable[[LlmRequest], LlmResponse]


@dataclass(frozen=True)
class PendingApproval:
    #: The APPROVAL id: what :func:`record_approval` answers. Not the tool call id.
    id: str
    tool: str
    arguments: dict[str, Any]
    tool_call_id: str = ""


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
        additional_content: Sequence[object] = (),
    ) -> AgentResponse:
        """Run a turn.

        An EMPTY prompt is meaningful and not an error: it is how a run resumes
        after an approval, because the conversation already contains the
        request, the decision, and everything before them. A new prompt there
        would be a second instruction competing with the one the tool call came
        from.

        ``additional_content`` is media sent with the prompt. See
        :func:`prism_harness.attachments.admit_attachments` for what is refused.
        """
        # Refused before a run exists: a bad attachment is a mistake in the call,
        # and it should not cost a run, events or budget.
        attachments = admit_attachments(prompt, additional_content)
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

        try:
            resolved = self._tools.resolve(list(tool_names) if tool_names else mode.tools, session)
            offered = (
                self._authorizer.allowed(session, resolved)
                if self._authorizer is not None
                else list(resolved.values())
            )
            called: list[str] = []

            # Decisions recorded since the run stopped are acted on FIRST, before
            # a new prompt is recorded, so the results land after the calls they
            # answer rather than after the new turn.
            self._resolve_approvals(session, mode, offered, run_id, called)

            if prompt != "":
                # With attachments, the shape prism-ai's UserMessage.to_dict()
                # writes: the media parts, then the turn's own text as a trailing
                # text part, which from_dict() strips back off. Without them,
                # unchanged.
                turn: dict[str, Any] = (
                    {"type": "user", "content": prompt}
                    if not attachments
                    else {
                        "type": "user",
                        "content": prompt,
                        "additional_content": [*attachments, {"text": prompt}],
                        "additional_attributes": {},
                    }
                )
                thread.record([turn], run_id)

            return self._loop(session, mode, run, run_id, provider, model, offered, called)
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
        offered: list[HarnessTool],
        called: list[str],
    ) -> AgentResponse:
        thread = session.thread()
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
                    messages=thread_view([entry.message for entry in thread.messages()]),
                    tools=offered,
                    provider=provider,
                    model=model,
                    provider_options=mode.provider_options,
                )
            )

            run.ledger.record_steps(1)
            run.ledger.record_cost(response.cost_usd)
            text = response.text
            finish_reason = response.finish_reason

            calls = [_call_input(call) for call in response.tool_calls]
            gated = [call for call in calls if mode.needs_approval(call.name)]
            requests = [
                {"approval_id": f"apr_{uuid.uuid4().hex}", "tool_call_id": call.id}
                for call in gated
            ]

            # The next step's request is built from this row, so it keeps what a
            # provider needs sent back: each call's arguments and provider ids,
            # the turn's provider state, and the approvals it is waiting on (G-58).
            thread.record(
                [assistant_row(response.text, calls, response.additional_content, requests)],
                run_id,
            )

            if not calls:
                return self._finish(session, run_id, called, finish_reason, text, run, None)

            # The calls that need nobody run now, as in the reference, and their
            # results are recorded even when the step then stops for a person.
            results: list[dict[str, Any]] = []

            for call in calls:
                if call in gated:
                    continue

                called.append(call.name)
                results.append(self._invoke(offered, call))

            if results:
                thread.record([tool_result_row(results)], run_id)

            if requests:
                # FAILS CLOSED. The gated calls have not run, and the requests are
                # in the thread, so a different process can resume after a person
                # answers.
                return AgentResponse(
                    run_id=run_id,
                    text=text,
                    steps=run.ledger.steps,
                    tool_calls=called,
                    finish_reason="awaiting_approval",
                    pending_approvals=[
                        PendingApproval(
                            id=request["approval_id"],
                            tool=call.name,
                            arguments=dict(call.arguments),
                            tool_call_id=call.id,
                        )
                        for call, request in zip(gated, requests, strict=True)
                    ],
                )

    def _resolve_approvals(
        self,
        session: Session,
        mode: AgentMode,
        offered: list[HarnessTool],
        run_id: str,
        called: list[str],
    ) -> None:
        """Act on the decisions recorded for the last turn that stopped for a person.

        The model is NOT asked again. Asked again, a provider issues the call
        afresh under a new id, and a decision recorded against the old one never
        matches. The calls that stopped the run are answered where they are:

        - approved: run, once;
        - denied: the reason, as the result the model sees;
        - no decision: refused, "No approval response provided". Record every
          decision before resuming.

        A call that already has a result is done and never runs again, whatever
        its decision says. The results are recorded as one tool result row
        holding every result and decision for the turn, as the reference writes
        it.
        """
        view = thread_view([entry.message for entry in session.thread().messages()])
        index = next(
            (
                i
                for i in range(len(view) - 1, -1, -1)
                if view[i]["type"] == "assistant" and view[i]["tool_calls"]
            ),
            None,
        )

        if index is None:
            return

        assistant = view[index]
        answered = next((row for row in view[index + 1 :] if row["type"] == "tool_result"), None)
        results = {e["tool_call_id"]: e for e in (answered or {}).get("tool_results", [])}
        decisions = {
            d["approval_id"]: d for d in (answered or {}).get("tool_approval_responses", [])
        }
        approval_ids = {
            r["tool_call_id"]: r["approval_id"] for r in assistant["tool_approval_requests"]
        }
        resolved: list[dict[str, Any]] = []

        for row in assistant["tool_calls"]:
            if row["id"] in results:
                continue

            if row["id"] not in approval_ids and not mode.needs_approval(row["name"]):
                continue

            call = ToolCallInput(
                id=row["id"],
                name=row["name"],
                arguments=row["arguments"],
                result_id=row["result_id"],
            )
            approval_id = approval_ids.get(row["id"])
            decision = decisions.get(approval_id) if approval_id is not None else None

            if decision is not None and decision["approved"]:
                called.append(call.name)
                resolved.append(self._invoke(offered, call))
            elif decision is None:
                resolved.append(tool_result_entry(call, "No approval response provided"))
            else:
                resolved.append(
                    tool_result_entry(call, decision["reason"] or "User denied tool execution")
                )

        if not resolved:
            return

        session.thread().record(
            [tool_result_row([*results.values(), *resolved], list(decisions.values()))], run_id
        )

    @staticmethod
    def _invoke(offered: Sequence[HarnessTool], call: ToolCallInput) -> dict[str, Any]:
        tool = next((candidate for candidate in offered if candidate.name == call.name), None)

        if tool is None:
            raise HarnessError.tool_not_available(
                call.name, [candidate.name for candidate in offered]
            )

        try:
            result = tool.handle(dict(call.arguments))
        except HarnessError as error:
            # A refused call propagates. A refusal fed back to the model reads
            # as a retryable failure, which is the opposite of a guard.
            if error.code == "call_not_authorized":
                raise

            return tool_result_entry(call, f"The tool failed: {error}")
        except Exception as error:  # noqa: BLE001 - a tool is someone else's code
            # A failed tool is a RESULT, not a crashed run: the model can often
            # recover, and losing the whole turn to one bad call is worse.
            return tool_result_entry(call, f"The tool failed: {error}")

        return tool_result_entry(call, result if isinstance(result, str) else json.dumps(result))

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

    Nothing runs until the next ``send()``, which acts on every decision recorded
    by then and refuses any pending call still without one. With several
    pending, record them all first.

    ``approval_id`` is :attr:`PendingApproval.id`, not the tool call id.
    """
    run = session.run()

    session.thread().record(
        [
            tool_result_row(
                [], [{"approval_id": approval_id, "approved": approved, "reason": reason}]
            )
        ],
        run["id"] if run else None,
    )


def _call_input(call: LlmToolCall) -> ToolCallInput:
    return ToolCallInput(
        id=call.id,
        name=call.name,
        arguments=call.arguments,
        result_id=call.result_id,
        reasoning_id=call.reasoning_id,
        reasoning_summary=call.reasoning_summary,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
