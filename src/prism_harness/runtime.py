"""The loop: prompt in, turns out, everything recorded."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypedDict

from prism_harness.address import validate_agent
from prism_harness.attachments import admit_attachments
from prism_harness.errors import HarnessError
from prism_harness.events import HarnessEvents, RunFailed, RunFinished, RunStarted
from prism_harness.modes import AgentMode, ModeRegistry
from prism_harness.session import Session
from prism_harness.structured import schema_name, schema_problems
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
    "StructuredAgentResponse",
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

    An INTERFACE rather than a dependency on ``prism-ai-core``. The loop below --
    steps, budgets, approvals, thread recording, events -- is the part worth
    porting, and none of it needs to know how a request reaches a provider.
    Keeping the seam here also means this package stays at zero dependencies and
    a consumer can drive it with ``prism-ai-core``, their own client, or a fake.

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
    #: The JSON Schema a structured turn asks the answer to satisfy.
    #:
    #: None on an ordinary turn. A client passes it to its provider's structured
    #: mode (for prism-ai-core, ``Prism.structured().with_schema()``) and returns
    #: what it parsed as ``structured``; the harness checks that against this
    #: same schema before the caller sees it.
    schema: dict[str, Any] | None = None

    @property
    def max_output_tokens(self) -> int | None:
        """Optional provider ceiling; absent from default dataclass serialization."""
        value: object = getattr(self, "_max_output_tokens", None)
        return value if isinstance(value, int) else None


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
    #: The document a structured turn parsed out of ``text``.
    #:
    #: None when the text held none -- an apology in prose, a truncated answer,
    #: a fence that never closed. The harness tells that case apart from a
    #: document with the wrong shape, because the caller's next move differs.
    structured: Any = None


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


@dataclass(frozen=True)
class StructuredAgentResponse(AgentResponse):
    """One structured run's result: the document, and the text it was read from.

    BOTH, not one. ``structured`` is what the caller asked for, and ``text`` is
    what the model actually sent -- which is also exactly what the thread stored,
    so a transcript and a parse can be compared rather than trusted.
    """

    #: Checked against the schema before it got here; a run that could not
    #: produce one raised.
    structured: Any = None


#: How long resolving approvals may hold the session lock: long enough for a tool to run.
_RESOLUTION_LOCK_SECONDS = 300

#: How long a second worker waits for that lock before giving up without running anything.
_RESOLUTION_WAIT_SECONDS = 30


class RuntimeConfig(TypedDict, total=False):
    provider: str | None
    model: str | None
    max_steps_ceiling: int | None
    max_output_tokens_per_call: int | None
    lock_ttl: int
    lock_wait: int


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
        config: RuntimeConfig | None = None,
        agents: dict[str, RuntimeConfig] | None = None,
        narrow_requested_tools: bool = False,
    ) -> None:
        if not isinstance(narrow_requested_tools, bool):
            raise TypeError("narrow_requested_tools must be a boolean")
        self._narrow_requested_tools = narrow_requested_tools
        self._client = client
        self._modes = modes
        self._tools = tools
        self._authorizer = authorizer
        self._events = events
        self._config = config.copy() if config is not None else RuntimeConfig()
        self._agents = {agent: options.copy() for agent, options in (agents or {}).items()}
        _validate_runtime_config(self._config)
        for agent, options in self._agents.items():
            validate_agent(agent)
            _validate_runtime_config(options)

    def _settings(self, session: Session) -> RuntimeConfig:
        return {
            **self._config,
            **(self._agents.get(session.agent, {}) if session.agent is not None else {}),
        }

    def _bounded_run(self, session: Session, run: RunContext) -> RunContext:
        options, budget = self._settings(session), run.budget
        steps, output = options.get("max_steps_ceiling"), options.get("max_output_tokens_per_call")
        if steps is None and output is None:
            return run
        return RunContext(
            run.ledger,
            RunBudget(
                budget.max_steps
                if steps is None
                else min(budget.max_steps, run.ledger.steps + steps),
                budget.max_cost_usd,
                budget.max_seconds,
                budget.max_output_tokens_per_call
                if output is None
                else min(budget.max_output_tokens_per_call or output, output),
            ),
            run.parent_run_id,
            run.depth,
        )

    def effective_tools(
        self, session: Session, mode_name: str | None = None
    ) -> list[dict[str, str]]:
        mode = self._modes.for_agent(session.agent).resolve(
            mode_name if mode_name is not None else session.mode()
        )
        return self._offer(session, mode)[1]

    def _offer(
        self, session: Session, mode: AgentMode, requested: Sequence[str] | None = None
    ) -> tuple[list[HarnessTool], list[dict[str, str]]]:
        def resolve(view: Session) -> tuple[list[HarnessTool], list[dict[str, str]]]:
            names = (
                _narrowed_tools(mode.tools, requested)
                if session.agent is not None or self._narrow_requested_tools
                else (requested or mode.tools)
            )
            detailed = self._tools.resolve_detailed(names, view)
            tools = {name: entry.tool for name, entry in detailed.items()}
            if view.agent is not None and "complete_task" in tools:
                authorizer = (
                    self._authorizer.for_agent(view.agent) if self._authorizer is not None else None
                )
                if authorizer is None or not authorizer.enabled or not authorizer.has_call_policy():
                    raise HarnessError(
                        "unsafe_authorization_configuration",
                        "Named task completion requires enabled authorization and an explicit call policy.",
                    )
            offered = (
                self._authorizer.allowed(view, tools)
                if self._authorizer is not None
                else list(tools.values())
            )
            ownership = {entry.tool.name: entry for entry in detailed.values()}
            return offered, [
                {
                    "name": tool.name,
                    "owner": ownership[tool.name].owner,
                    "source": ownership[tool.name].source,
                }
                for tool in offered
            ]

        return session.resolve_read_only(mode.name, resolve)

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
        return self._turn(session, prompt, tool_names, context, additional_content)

    def send_structured(
        self,
        session: Session,
        prompt: str,
        schema: dict[str, Any],
        tool_names: Sequence[str] | None = None,
        context: RunContext | None = None,
        additional_content: Sequence[object] = (),
    ) -> StructuredAgentResponse:
        """A turn whose answer is a document.

        The same run as :meth:`send` -- same mode, tools, budget, events,
        approvals -- with a schema the answer has to satisfy. The schema travels
        on the request for the client to hand its provider, and what comes back
        is checked against it here before the caller sees it.

        WHAT THE THREAD KEEPS IS THE TEXT, with the parsed document beside it as
        ``structured`` in the assistant row's ``additional_content``. Never
        instead of it: a later turn replays this conversation as messages, and a
        transcript that reads differently because of the SHAPE of the request
        that produced it is a difference nothing reports.

        A FAILED DOCUMENT IS STILL RECORDED, and then raised. The exchange
        happened, and a thread that omits the answer it did not like cannot
        explain the retry sitting next to it. The run is marked failed and
        ``RunFailed`` is emitted, as for any other failure.

        :raises HarnessError: ``structured_unreadable`` or
            ``structured_schema_violation``.
        """
        response = self._turn(session, prompt, tool_names, context, additional_content, schema)

        # Narrowed by _turn: with a schema it always returns the structured shape.
        assert isinstance(response, StructuredAgentResponse)

        return response

    def _turn(
        self,
        session: Session,
        prompt: str,
        tool_names: Sequence[str] | None = None,
        context: RunContext | None = None,
        additional_content: Sequence[object] = (),
        schema: dict[str, Any] | None = None,
    ) -> AgentResponse:
        # Refused before a run exists: a bad attachment is a mistake in the call,
        # and it should not cost a run, events or budget.
        attachments = admit_attachments(prompt, additional_content)
        mode = self._modes.for_agent(session.agent).resolve(session.mode())
        options = self._settings(session)
        provider = session.provider() or options.get("provider") or "unknown"
        model = session.model() or options.get("model") or "unknown"
        run_id = str(uuid.uuid4())
        run = self._bounded_run(
            session, context or RunContext.root(run_id, RunBudget(mode.max_steps))
        )
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
            if session.agent is None:
                resolved = self._tools.resolve(
                    list(tool_names) if tool_names else mode.tools, session
                )
                offered = (
                    self._authorizer.allowed(session, resolved)
                    if self._authorizer is not None
                    else list(resolved.values())
                )
            else:
                offered = self._offer(session, mode, tool_names)[0]
            called: list[str] = []

            # Decisions recorded since the run stopped are acted on FIRST, before
            # a new prompt is recorded, so the results land after the calls they
            # answer rather than after the new turn.
            self._resolve_approvals(session, mode, offered, run_id, called)

            if prompt != "":
                # With attachments, the shape prism-ai-core's UserMessage.to_dict()
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

            return self._loop(session, mode, run, run_id, provider, model, offered, called, schema)
        except Exception as error:
            # THE MODEL'S OWN WORDS DO NOT BELONG IN A RUN ROW OR AN EVENT. A
            # schema violation names the values that missed, so its message
            # carries pieces of the document -- and an event carrying those would
            # put model output in every listener's telemetry, which is the same
            # reason tool arguments are names-only here. The document is already
            # in the thread, in full, where it is read deliberately rather than
            # shipped by default. An error that holds one records its CODE.
            failure = (
                error.code
                if isinstance(error, HarnessError) and error.document is not None
                else str(error)
            )
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
        schema: dict[str, Any] | None = None,
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

            request = LlmRequest(
                system_prompt=mode.system_prompt,
                messages=thread_view([entry.message for entry in thread.messages()]),
                tools=offered,
                provider=provider,
                model=model,
                provider_options=mode.provider_options,
                schema=schema,
            )
            if run.budget.max_output_tokens_per_call is not None:
                object.__setattr__(
                    request, "_max_output_tokens", run.budget.max_output_tokens_per_call
                )
            response = self._client(request)

            run.ledger.record_steps(1)
            run.ledger.record_cost(response.cost_usd)
            text = response.text
            finish_reason = response.finish_reason

            calls = [_call_input(call) for call in response.tool_calls]
            failed_finish = finish_reason in ("error", "unknown")
            gated = (
                [] if failed_finish else [call for call in calls if mode.needs_approval(call.name)]
            )
            requests = [
                {"approval_id": f"apr_{uuid.uuid4().hex}", "tool_call_id": call.id}
                for call in gated
            ]

            # The next step's request is built from this row, so it keeps what a
            # provider needs sent back: each call's arguments and provider ids,
            # the turn's provider state, and the approvals it is waiting on (G-58).
            answering = schema is not None and not calls
            metadata = (
                {**response.additional_content, "structured": response.structured}
                if answering
                else response.additional_content
            )

            thread.record([assistant_row(response.text, calls, metadata, requests)], run_id)

            if failed_finish:
                # Refuse, rather than queue, every call so a later resume
                # cannot dispatch a failed response's requests.
                if calls:
                    thread.record(
                        [
                            tool_result_row(
                                [
                                    tool_result_entry(
                                        call, f"Not run: model step failed ({finish_reason})."
                                    )
                                    for call in calls
                                ]
                            )
                        ],
                        run_id,
                    )
                raise HarnessError.model_step_failed(
                    "error" if finish_reason == "error" else "unknown"
                )

            if not calls:
                if schema is not None:
                    # Recorded first, then checked: the exchange happened either
                    # way, and a failure here fails the run through the caller's
                    # except.
                    self._assert_document(schema, response)

                finished = self._finish(session, run_id, called, finish_reason, text, run, None)

                if schema is None:
                    return finished

                return StructuredAgentResponse(
                    run_id=finished.run_id,
                    text=finished.text,
                    steps=finished.steps,
                    tool_calls=finished.tool_calls,
                    finish_reason=finished.finish_reason,
                    pending_approvals=finished.pending_approvals,
                    stopped_because=finished.stopped_because,
                    structured=response.structured,
                )

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
        # A read without the lock first: nearly every send() has nothing to
        # resolve and should not wait on the session lock to find that out.
        if self._unresolved(session, mode) is None:
            return

        # Then under the session lock, reading again inside it. Two workers
        # resuming the same session at once would otherwise both find the
        # approved call with no result, and both run it. The second now waits,
        # finds the result the first recorded, and runs nothing. If it cannot get
        # the lock it raises, and nothing runs.
        def resolve(live: Session) -> None:
            work = self._unresolved(live, mode)

            if work is None:
                return

            calls, results, decisions, approval_ids = work
            resolved: list[dict[str, Any]] = []

            for row in calls:
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
                    resolved.append(self._run_approved(offered, call))
                elif decision is None:
                    resolved.append(tool_result_entry(call, "No approval response provided"))
                else:
                    resolved.append(
                        tool_result_entry(call, decision["reason"] or "User denied tool execution")
                    )

            live.thread().record(
                [tool_result_row([*results.values(), *resolved], list(decisions.values()))],
                run_id,
            )

        options = self._settings(session)
        session.lock(
            resolve,
            options.get("lock_ttl", _RESOLUTION_LOCK_SECONDS),
            options.get("lock_wait", _RESOLUTION_WAIT_SECONDS),
        )

    @staticmethod
    def _unresolved(
        session: Session, mode: AgentMode
    ) -> (
        tuple[
            list[dict[str, Any]],
            dict[str, dict[str, Any]],
            dict[str, dict[str, Any]],
            dict[str, str],
        ]
        | None
    ):
        """The gated calls of the last tool-calling turn that have no result yet."""
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
            return None

        assistant = view[index]
        answered = next((row for row in view[index + 1 :] if row["type"] == "tool_result"), None)
        results = {e["tool_call_id"]: e for e in (answered or {}).get("tool_results", [])}
        decisions = {
            d["approval_id"]: d for d in (answered or {}).get("tool_approval_responses", [])
        }
        approval_ids = {
            r["tool_call_id"]: r["approval_id"] for r in assistant["tool_approval_requests"]
        }

        # A call that has a result is DONE, whatever its decision says. A call that
        # needs nobody is not this method's to run.
        calls = [
            row
            for row in assistant["tool_calls"]
            if row["id"] not in results
            and (row["id"] in approval_ids or mode.needs_approval(row["name"]))
        ]

        return (calls, results, decisions, approval_ids) if calls else None

    @classmethod
    def _run_approved(cls, offered: Sequence[HarnessTool], call: ToolCallInput) -> dict[str, Any]:
        """Run a call a person approved, or record why it could not run.

        Recorded rather than raised. Resolution runs at the start of every
        send(), so a call that raises here, because its tool is no longer
        offered to this run or the authorizer now refuses it, would raise again
        on every later send(), and the session could never move on. The approval
        stands, but the call does not run.
        """
        if not any(tool.name == call.name for tool in offered):
            return tool_result_entry(call, f"Not run: {call.name} is not available to this run.")

        try:
            return cls._invoke(offered, call)
        except HarnessError as error:
            if error.code == "call_not_authorized":
                return tool_result_entry(
                    call, f"Not run: this call to {call.name} is not authorized."
                )
            raise

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

    @staticmethod
    def _assert_document(schema: dict[str, Any], response: LlmResponse) -> None:
        """Refuse a document the caller cannot use, naming which of the two it is.

        Text holding no document at all and a document with the wrong shape are
        separate codes because the next move differs: the first is a prompting or
        budget problem, the second a schema one.
        """
        if response.structured is None:
            raise HarnessError.structured_unreadable(response.text)

        problems = schema_problems(schema, response.structured, schema_name(schema))

        if problems:
            raise HarnessError.structured_schema_violation(response.text, problems)

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


def _validate_runtime_config(config: Mapping[str, object]) -> None:
    for key in ("provider", "model"):
        value = config.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("Agent provider/model defaults must be nonempty strings or null.")
    for key in ("max_steps_ceiling", "max_output_tokens_per_call"):
        value = config.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError("Agent step/output ceilings must be positive integers or null.")
    for key in ("lock_ttl", "lock_wait"):
        if key in config:
            value = config[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if key == "lock_ttl" else 0)
            ):
                raise ValueError(
                    "Agent lock settings must be whole seconds with positive TTL and nonnegative wait."
                )


def _narrowed_tools(ceiling: Sequence[str], requested: Sequence[str] | None) -> Sequence[str]:
    if requested is None or "*" in requested:
        return ceiling
    if "*" in ceiling:
        return requested
    for name in requested:
        if name not in ceiling:
            raise HarnessError.tool_not_available(name, ceiling)
    return requested


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
