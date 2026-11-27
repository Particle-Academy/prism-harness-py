"""The loop: prompt in, turns out, everything recorded."""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypedDict

from prism_harness.address import validate_agent
from prism_harness.attachments import admit_attachments
from prism_harness.errors import HarnessError
from prism_harness.events import (
    HarnessEvent,
    HarnessEvents,
    RunFailed,
    RunFinished,
    RunStarted,
    ToolApprovalAsked,
    ToolApprovalExpired,
    ToolApproved,
    ToolAutoAllowed,
    ToolDenied,
)
from prism_harness.failures import class_label, failure_of, harness_code
from prism_harness.modes import AgentMode, ModeRegistry
from prism_harness.plans import budget as plan_budget
from prism_harness.plans.refusal import PlanRefused
from prism_harness.plans.service import Plans
from prism_harness.run_budgets import allowance, fail_closed, steps_used
from prism_harness.run_starts import StartRefused, gate
from prism_harness.run_trees import RunTrees, kept_refs, saved_refs
from prism_harness.session import CHILD_TURN_LIVE, Session, outside_any_start
from prism_harness.structured import schema_name, schema_problems
from prism_harness.subagents import (
    MAX_DEPTH,
    ChildBinding,
    RunBudget,
    RunContext,
    binding_route,
)
from prism_harness.thread import Thread, ThreadMessage, _record_in_run
from prism_harness.thread_rows import (
    ARGUMENTS_AS_SENT,
    NOT_THE_APPROVED_CALL,
    REPEATED_CALL_ID,
    SUPERSEDED,
    SUPERSEDED_RESULT,
    ToolCallInput,
    ambiguous_calls,
    answering_step,
    approval_run_steps,
    approval_tool_scope,
    assistant_row,
    binds,
    call_results,
    carried_agent_spend,
    decided_unresolved_ids,
    ended_by_start_row,
    first_decisions,
    last_tool_step,
    new_results,
    owned_rows,
    paired_requests,
    paused_approval_row,
    pending_approval_ids,
    placed_view,
    repeats_call_ids,
    replaced_earlier_steps,
    replaced_step_row,
    requested_step,
    standing_decisions,
    started_calls,
    started_row,
    step_calls,
    step_requests,
    step_rows,
    step_superseded,
    thread_view,
    tool_result_entry,
    tool_result_row,
    unusable_stored_calls,
)
from prism_harness.tool_arguments import usable_arguments
from prism_harness.tool_policy import ToolPolicy, args_digest, tool_declaration
from prism_harness.tools import HarnessTool, ToolAuthorizer, ToolRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_harness.plans.handle import PlanHandle
    from prism_harness.plans.tools import PlanTool

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
    #: Set when the tree ran out of budget, or was cancelled. For the host:
    #: a cancel carries the host's own reason here.
    stopped_because: str | None = None
    #: Why the run stopped, as a fixed code (``cancelled``, ``steps``,
    #: ``cost``, ``cost_unenforceable``, ``time``) when it ran out or was
    #: cancelled. What a model is told about a stop is chosen from this code,
    #: never from ``stopped_because`` (G-387).
    stopped_code: str | None = None


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

#: The session key, run context and run id of the turn whose tools are running
#: in this context. A subagent child started by one of those tools spends from
#: it, and joins the run tree that run is recorded in (G-381).
_TURN_RUN: ContextVar[tuple[str, RunContext, str] | None] = ContextVar(
    "prism_harness_turn_run", default=None
)


def current_turn_run_id(session_key: str) -> str | None:
    """The run id of the turn on that session whose tools are running now, or
    None outside one. Internal: the subagent runner marks the context it starts
    a child under with it, so the child joins that run's tree and no other."""
    turn = _TURN_RUN.get()
    if turn is None or turn[0] != session_key:
        return None
    return turn[2]


def current_turn_run(session_key: str | None = None) -> RunContext | None:
    """The run context of the turn whose tools are running now, or None
    outside a turn. Its budget is the turn's own, already narrowed by every
    limit that applies to it, a shared plan budget included.

    With ``session_key``, only a turn on that session: a tool built for one
    session that runs inside another session's turn does not borrow it."""
    turn = _TURN_RUN.get()
    if turn is None or (session_key is not None and turn[0] != session_key):
        return None
    return turn[1]


#: How long a second worker waits for that lock before giving up without running anything.
_RESOLUTION_WAIT_SECONDS = 30

#: The sessions whose approvals are being resolved in this context, each
#: with the session lock's CONFIGURED ttl and wait (G-387 round 16): a
#: decision a tool records on its own session while the resolver holds that
#: lock waits only as long as the agent is configured to, never a fixed 30 s.
_RESOLVING: ContextVar[dict[str, tuple[float, float]] | None] = ContextVar(
    "prism_harness_resolving", default=None
)

#: What a decision recorded from inside a running approved call on its own
#: session is refused with: a fixed text, so no session key, thread key,
#: participant or session name reaches the model through the tool's result.
_OWN_SESSION_LOCKED = (
    "A decision cannot be recorded on this session while one of its approved calls is "
    "running; it was not recorded. Record it after the turn."
)


class RuntimeConfig(TypedDict, total=False):
    provider: str | None
    model: str | None
    max_steps_ceiling: int | None
    max_output_tokens_per_call: int | None
    lock_ttl: int
    lock_wait: int
    #: Positive finite USD this NAMED agent may spend per root run tree, against
    #: the tree's own ledger. Read only from ``agents=``; no per-day window.
    spend_ceiling: float | None


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
        narrow_requested_tools: bool | None = None,
        allow_legacy_tool_list_widening: bool = False,
        tool_policy: ToolPolicy | None = None,
        plans: Plans | Mapping[str, Any] | None = None,
    ) -> None:
        if narrow_requested_tools is not None and not isinstance(narrow_requested_tools, bool):
            raise TypeError("narrow_requested_tools must be a boolean")
        if not isinstance(allow_legacy_tool_list_widening, bool):
            raise TypeError("allow_legacy_tool_list_widening must be a boolean")
        if narrow_requested_tools is False:
            raise ValueError(
                "narrow_requested_tools=False is no longer supported. Remove it; use "
                "allow_legacy_tool_list_widening=True only to explicitly permit legacy tool-list widening."
            )
        if narrow_requested_tools is True and allow_legacy_tool_list_widening:
            raise ValueError(
                "narrow_requested_tools=True conflicts with allow_legacy_tool_list_widening=True."
            )
        self._narrow_requested_tools = not allow_legacy_tool_list_widening
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
        self._tool_policy = self._admit_policy(tool_policy or ToolPolicy.none())
        # Agent plans (G-360): off unless the host turns them on here AND the
        # session's mode sets ``plans``.
        self._plans = plans if isinstance(plans, Plans) else Plans(plans)
        self._plans._attach(modes, events)
        # The durable half of run tree budgets, for children resumed outside
        # the runner.
        self._trees = RunTrees()

    @property
    def plans(self) -> Plans:
        """The agent plan settings and store (G-360)."""
        return self._plans

    def plan(self, session: Session) -> PlanHandle:
        """This session's agent plan (G-360).

        Off unless the host turned plans on (``plans={"enabled": True}``) AND
        the session's mode sets ``plans``; otherwise raises
        :class:`~prism_harness.plans.PlansDisabled`.
        """
        return self._plans.handle(session, self)

    def decide(
        self, session: Session, approval_id: str, approved: bool, reason: str | None = None
    ) -> None:
        """Record a person's decision, as :func:`record_approval` does, and let a
        plan held on it read the decision before the run resumes (G-360).

        Nothing runs until the next resume (``send(session, "")``). Who may decide is the
        application's decision: authorize before calling.
        """
        options = self._settings(session)
        _record_decision(
            session,
            approval_id,
            approved,
            reason,
            options.get("lock_ttl", _RESOLUTION_LOCK_SECONDS),
            options.get("lock_wait", _RESOLUTION_WAIT_SECONDS),
        )
        self._plans.decided(session)

    def lock_ttl(self, session: Session) -> int:
        """How long a held session lock lasts for this session's agent."""
        ttl = self._settings(session).get("lock_ttl", _RESOLUTION_LOCK_SECONDS)
        return (
            ttl if isinstance(ttl, int) and not isinstance(ttl, bool) else _RESOLUTION_LOCK_SECONDS
        )

    @property
    def tool_policy(self) -> ToolPolicy:
        """The host tool policy in force (G-326)."""
        return self._tool_policy

    def set_tool_policy(self, policy: ToolPolicy) -> None:
        """Replace the host tool policy (G-326).

        Applies from the next tool call. A call already waiting on a person
        keeps waiting: a policy that would now allow it does not run it. A call
        the old policy allowed that has not started yet is refused, not run,
        when the new policy no longer allows it, so a revoke takes effect at
        once. Validated against the registered tools' declarations: a policy
        naming a tool that declares ``destructive`` raises ``HarnessError``
        (``tool_policy_refused``) and the old policy stays.
        """
        self._tool_policy = self._admit_policy(policy)

    def _admit_policy(self, policy: ToolPolicy) -> ToolPolicy:
        if not isinstance(policy, ToolPolicy):
            raise TypeError("tool_policy must be a ToolPolicy")
        return ToolPolicy.from_json(policy.to_json(), self._tools.declarations())

    def _tool_decision(
        self,
        kind: type[
            ToolAutoAllowed | ToolApprovalAsked | ToolApproved | ToolDenied | ToolApprovalExpired
        ],
        session: Session,
        run_id: str,
        call: ToolCallInput,
        tool: HarnessTool | None,
        decided_by: str,
    ) -> None:
        if self._events is None:
            return
        self._events.emit(
            kind(
                run_id=run_id,
                session_key=session.key(),
                policy_revision=self._tool_policy.revision,
                server=tool_declaration(tool).server if tool is not None else None,
                tool=call.name,
                args_digest=args_digest(dict(call.arguments)),
                decided_by=decided_by,
                harness_agent=session.agent,
            )
        )

    def _settings(self, session: Session) -> RuntimeConfig:
        return {
            **self._config,
            **(self._agents.get(session.agent, {}) if session.agent is not None else {}),
        }

    def _starting_child(
        self,
        context: RunContext,
        parent_run_id: str | None = None,
        parent_session_key: str | None = None,
        start: Mapping[str, Any] | None = None,
    ) -> RunContext:
        """Internal: the subagent runner marks the context it starts a child
        under as its own, so only that context saves the child's binding, with
        the run and session of the ACTIVE turn starting it (None outside one):
        the child joins that run's tree once the run is verified live.
        ``start`` is what the start is for (``task``, ``mode``, ``hosts``),
        read by the start gate (G-387)."""
        return self._trees.starting_child(context, parent_run_id, parent_session_key, start)

    def mode_for(self, session: Session) -> AgentMode:
        """The mode a session's next run resolves, refusing a malformed one."""
        return self._modes.for_agent(session.agent).resolve(session.mode())

    def _bounded_run(self, session: Session, run: RunContext) -> RunContext:
        """The run under this agent's label and output ceiling. Its STEPS are
        not touched here: ``max_steps_ceiling`` is one input of the run's
        allowance (G-387), absolute on the run's whole count."""
        run = self._labelled(session, run)
        options, budget = self._settings(session), run.budget
        output = options.get("max_output_tokens_per_call")
        if output is None:
            return run
        return RunContext(
            run.ledger,
            RunBudget(
                budget.max_steps,
                budget.max_cost_usd,
                budget.max_seconds,
                budget.max_output_tokens_per_call
                if output is None
                else min(budget.max_output_tokens_per_call or output, output),
            ),
            run.parent_run_id,
            run.depth,
            run.agent,
            run.spend_ceiling,
            run.binding,
        )

    def _labelled(self, session: Session, run: RunContext) -> RunContext:
        """The context spending as this session's agent, under that agent's ceiling.

        Same ledger, always: a label is not a second account.
        """
        ceiling = (
            self._agents.get(session.agent, {}).get("spend_ceiling")
            if session.agent is not None
            else None
        )
        return RunContext(
            run.ledger,
            run.budget,
            run.parent_run_id,
            run.depth,
            session.agent,
            ceiling,
            run.binding,
        )

    def _binding(self, session: Session, mode: AgentMode, run: RunContext) -> tuple[str, str]:
        """Provider and model, each resolved on its own (G-321).

        Root: session override > mode > agent default > global. A subagent
        child: the parent's session override only when the subagent asked,
        then the declaration, its own mode, the parent's mode within the same
        agent, then the agent and global defaults. See :class:`ChildBinding`.
        """
        options = self._settings(session)
        child = run.binding
        if child is None:
            providers = [session.provider(), mode.provider]
            models = [session.model(), mode.model]
        else:
            providers = [
                child.session_provider,
                child.declared_provider,
                mode.provider,
                child.parent_mode_provider,
            ]
            models = [
                child.session_model,
                child.declared_model,
                mode.model,
                child.parent_mode_model,
            ]
        return (
            _first([*providers, options.get("provider")]),
            _first([*models, options.get("model")]),
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
        # A turn is never a runner start's OWN run, even one entered from
        # inside one (a hook or a host callback sending on a session): every
        # reader in it keeps the provisional rule until its own start, if it
        # is one, commits (G-387 round 11).
        with outside_any_start():
            return self._run_turn(session, prompt, tool_names, context, additional_content, schema)

    def _run_turn(
        self,
        session: Session,
        prompt: str,
        tool_names: Sequence[str] | None,
        context: RunContext | None,
        additional_content: Sequence[object],
        schema: dict[str, Any] | None,
    ) -> AgentResponse:
        # Refused before a run exists: a bad attachment is a mistake in the call,
        # and it should not cost a run, events or budget.
        attachments = admit_attachments(prompt, additional_content)
        kind = self._trees.context_kind(context)
        # A runner start names the child's mode on its context: the session's
        # mode is written only once the start is committed (G-387 start gate).
        start = self._trees.start_of(context) if kind == "runner" else None
        start_mode = start.get("mode") if start is not None else None
        mode = self._modes.for_agent(session.agent).resolve(
            start_mode if isinstance(start_mode, str) else session.mode()
        )
        run_id = str(uuid.uuid4())
        base = context or RunContext.root(run_id, RunBudget(mode.max_steps))
        # A resume with no context continues the run that paused (G-321 E):
        # a subagent child keeps the provider and model binding the runner gave
        # it, and a named agent keeps what it had spent against its ceiling.
        # Both are read from the paused row, which the harness wrote.
        history = session.thread().rows()
        paused = paused_approval_row(history) if prompt == "" and context is None else None
        # The binding: from the runner's own context when the runner starts
        # the child, even under a run tree the host built
        # (RunTrees.binding_context_kind); otherwise the one SAVED with the
        # child session, which a context the host supplies never replaces. A
        # child paused before the binding was saved there carries it on its
        # paused row. A session the
        # runner started as a child (it has saved run tree references) with no
        # binding anywhere is refused rather than run on the root rules, which
        # would send its conversation to a provider nobody chose for it.
        # A host context on a session that was never a child keeps the
        # binding it states for that turn; it is never saved. The rule is
        # binding_route(), the one the harness-plans corpus pins.
        saved_binding = ChildBinding.from_row(session.child_binding())
        if saved_binding is None and not session.child_provisional():
            # A runner start's records are absent until its task row (G-387
            # round 9), and so is anything older they replaced: never read
            # back from a paused row instead.
            tail = paused_approval_row(history)
            saved_binding = ChildBinding.from_row(tail.get("harness_binding")) if tail else None
        route = binding_route(
            self._trees.binding_context_kind(context),
            context is not None and context.binding is not None,
            saved_binding is not None,
            session.child_refs() is not None,
        )
        if route["binding"] == "refused":
            raise HarnessError.child_binding_missing()
        runner_binding = (
            context.binding if context is not None and route["binding"] == "runner" else None
        )
        if route["binding"] == "saved" and saved_binding is not None:
            base = base.with_binding(saved_binding)
        run = self._bounded_run(session, base)
        # Which run tree this run belongs to, and how deep it is, decided ONCE
        # (G-381): every consumer below reads this record and nothing else.
        member = self._trees.membership_of(session, run_id, context, prompt)
        run = replace(run, depth=member["depth"])
        if paused is not None and run.agent is not None and "harness_agent_spend" in paused:
            run.ledger.carry_agent_spend(
                run.agent, *carried_agent_spend(paused["harness_agent_spend"])
            )
        provider, model = self._binding(session, mode, run)
        thread = session.thread()

        if kind != "runner" and run.too_deep():
            raise HarnessError.run_not_permitted(
                f"This run is nested {run.depth} deep, at or past the ceiling. Nobody debugs a "
                "tree that deep, and a configuration that produced one is a mistake worth "
                "reporting rather than executing."
            )

        # A runner start READS every fact first, asks the start gate ONCE, and
        # commits only what the gate lets through: the tree hold, then the
        # child's records in one locked write. A refusal writes nothing.
        started = (
            self._start_child(
                session, mode, run, run_id, member, context, prompt, start or {}, runner_binding
            )
            if kind == "runner"
            else None
        )
        # Committed: from here to the end of its run, the start's own run
        # reads what it committed on its own Session (G-387 round 11): its
        # mode, binding, refs, depth, ceiling and run. Every other reader
        # keeps the provisional rule until the task row. Left before any
        # failure is cleaned up, so the cleanup reads as any outside reader.
        leave_own = session.enter_own_start(run_id) if started is not None else _nothing
        try:
            # The run itself: for a runner start, only now that it is committed
            # (its mode was written with the child's records, in (b)).
            # A runner start's run is absent from the run state and the run
            # history until its task row is recorded (G-387 round 9).
            session.begin_run(
                run_id,
                mode.name,
                provider,
                model,
                tree=member["tree"],
                depth=member["depth"],
                provisional=started is not None,
            )
        except BaseException:
            # Every exception, a BaseException included (G-387 round 10):
            # releasing the start's holds does not depend on what was raised,
            # and the exception still propagates. No run began, so there is
            # no run to fail and no event to end.
            leave_own()
            if started is not None:
                # The run could not begin: the child is put back as it was,
                # and the start's tree hold released. Housekeeping: the
                # start's writes are provisional, absent until a task row.
                self._undo_start(session, member, run_id, started)
                _quietly(lambda: self._trees.finished(session, run_id))
            raise

        turn_token = None
        # Set at the task-row write itself: the in-memory fact that the
        # start's task row is in the thread (G-387 round 13). Read before any
        # undo, never a store read.
        task_recorded = [False]
        # From here the run has begun: everything it does, its RunStarted
        # included, is inside this try (G-387 round 12), so whatever raises
        # (a store write, a listener's BaseException, an emit that throws)
        # fails THIS run by class name, releases its holds and ends it with
        # RunFailed, a root run as much as a runner start.
        try:
            # Recorded before any tool can run, so a child started from this
            # turn finds it; dropped when the run finishes.
            self._trees.record(session, run_id, member, run)
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
                    harness_agent=session.agent,
                )
            )
            # The plan context (G-360), computed once per run and never stored:
            # a system prompt contribution, not a message. A turn with a new
            # user message reads the plan as if nothing were pending.
            plan_context = self._plans.context(session, prompt, mode.name)
            system_prompt = (
                mode.system_prompt
                if plan_context == ""
                else "\n\n".join(part for part in (mode.system_prompt, plan_context) if part)
            )
            if session.agent is None and not self._narrow_requested_tools:
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
            stored = thread.rows()
            scope = approval_tool_scope(stored)
            # The run's step allowance (G-387), from facts the harness wrote
            # itself: what the parked run this turn resumes had used (its own
            # row's count, children included), a child's saved ceiling, the
            # context and max_steps_ceiling. Worked out first without the
            # tree and plan grants, which it then asks for.
            if started is not None:
                facts, forged = dict(started["facts"]), False
            else:
                facts, forged = self._allowance_facts(
                    session, mode, kind, context, prompt, stored, saved_binding
                )
            claimed = started is not None
            if kind != "runner" and facts["saved_ceiling"] is not None and not forged:
                # A child's saved ceiling is its DELEGATION's budget: the
                # steps it used since the runner last started it are counted
                # durably, live holds included, and this run claims what it
                # may take from them in the same locked write, so two turns on
                # the child at once, or many in a row, never total more.
                def decide(counted: int | None) -> int:
                    if counted is None:
                        return 0
                    facts["used"] = max(int(facts["used"]), counted)
                    return int(_allowance(facts, False)["own_left"])

                counted = session.claim_child_steps(
                    run_id, self._plans.now(), self.lock_ttl(session), False, decide
                )
                forged = counted is None
                claimed = True
            first = _allowance(facts, forged) if started is None else started["first"]
            if kind in ("none", "metadata"):
                # A resume is the SAME run: the fresh ledger continues its
                # count, so what the run reports includes the steps before the
                # pause. Never on a context's ledger, which is someone else's.
                run.ledger.record_steps(first["used"])
            # A runner start counts from the ledger as its gate READ it: steps
            # spent on the context since then count against the child.
            begin_steps = run.ledger.steps if started is None else int(started["base"])
            self._trees.members[run_id]["begin_steps"] = begin_steps
            self._trees.members[run_id]["child_claim"] = claimed
            # A root run opens a tree; any turn on a child session the runner
            # did not start (an answer, a new message, or a host context)
            # continues the run tree it was saved with, holding what its own
            # allowance leaves, in one locked write. The runner's own context
            # is left alone. Then the plan's own turns hold from the plan's
            # shared budget what the run and its tree leave (G-366).
            run, tree_left = self._trees.root_run(
                session, run_id, run, member, self.lock_ttl(session), first["own_left"]
            )
            # A runner start spends from its parent's run, never from a plan
            # ledger of its own (its gate refused one); its tree grant is the
            # one its commit held.
            if started is None:
                run, plan_left = self._plans.plan_turn(
                    session,
                    run_id,
                    run,
                    context,
                    self.lock_ttl(session),
                    prompt,
                    kind,
                    first["own_left"] if tree_left is None else min(first["own_left"], tree_left),
                )
            else:
                tree_left, plan_left = started["tree_left"], None
            facts = {**facts, "tree_left": tree_left, "plan_left": plan_left}
            allowed = _allowance(facts, forged)
            if started is None and prompt != "" and allowed["allowance"] == 0:
                # A NEW prompt resumes no recorded decision, so with nothing
                # left it is refused before anything is recorded. Only a resume
                # acts on its decisions with no model step (G-387).
                asked = (
                    first["own_left"] if tree_left is None else min(first["own_left"], tree_left)
                )
                if plan_left == 0 and asked > 0:
                    raise HarnessError(
                        plan_budget.EXHAUSTED,
                        "This plan step may not proceed (plan_budget_exhausted): the plan's "
                        "shared budget has nothing left.",
                    )
                raise HarnessError.run_not_permitted(
                    "This run may not proceed: its step allowance is spent."
                )
            # Narrowed ONCE, and recorded beside the membership: the loop's
            # step gate, the paused row, a runner start's gate and a child's
            # tree hold all read this and nothing else.
            run = replace(
                run, budget=replace(run.budget, max_steps=begin_steps + allowed["allowance"])
            )
            self._trees.allot(
                run_id, {**allowed, "facts": facts, "forged": forged}, begin_steps, run.budget
            )
            # From here every tool the turn runs, a subagent included, runs
            # under this run context (G-366).
            turn_token = _TURN_RUN.set((session.key(), run, run_id))
            if scope is not None:
                names, legacy = scope
                if legacy:
                    logging.getLogger(__name__).warning(
                        "Pending approval has no valid stored tool scope; offering approved requests only."
                    )
                offered = [
                    tool
                    for tool in offered
                    if tool.name in names and ("*" in mode.tools or tool.name in mode.tools)
                ]
            offered, plan_gates = self._with_plan_tools(session, mode, offered, scope)
            called: list[str] = []

            if started is None and (run.ledger.cancelled or _time_spent(run)):
                # A cancelled tree, or a context whose time is spent, acts on
                # no decision (an approval or a denial) and records no new
                # prompt: an approved call never runs after its tree was
                # cancelled or its time ran out. The decisions stay recorded
                # for a later resume that can act; the loop ends the run on the
                # cancellation or the time before any model step. A runner
                # start is committed once past its gate, so it records its task
                # as any runner start does, and the loop's gate stops it.
                return self._loop(
                    session,
                    mode,
                    run,
                    run_id,
                    provider,
                    model,
                    offered,
                    called,
                    schema,
                    system_prompt,
                    plan_gates,
                )

            # A resume (an empty prompt) acts on the decisions recorded since
            # the run stopped, FIRST, so the results land after the calls they
            # answer. A new prompt supersedes the step instead (DECISION B).
            # Never by a runner start (G-387 round 9): it acts on no recorded
            # decision, and its task row ends every call still pending on the
            # child, as not run.
            if started is None and prompt != "":
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
                # A NEW message supersedes the paused step (DECISION B, G-387
                # round 13): each of its calls with no result is denied,
                # whatever was decided on it, so nothing of it runs, and the
                # denials stand even if this turn then fails. ONE write, with
                # the user row (G-404), so no decision ever finds the step
                # replaced but still answerable. It never waits on a resume
                # (G-387 round 15): a call that resume is running has its
                # started row and is answered-in-progress, never denied.
                self._record_new_message(session, run_id, turn)
            if started is None:
                offered = self._resolve_approvals(
                    session,
                    mode,
                    offered,
                    run_id,
                    called,
                    scope is not None and scope[1],
                    stored if prompt == "" else None,
                )

            ended: list[ToolCallInput] = []
            if prompt != "" and started is not None:
                ended = self._record_task(session, mode, run_id, prompt, attachments, task_recorded)
            if started is not None:
                # The task is recorded: the start is committed, and its tree
                # hold is the spending run's from now on. The task row is the
                # authority that confirms the start's provisional writes;
                # dropping the mark is housekeeping, and its failure changes
                # nothing.
                # Marked recorded FIRST, right after the write and before any
                # event (G-387 round 11): a listener that raises or re-enters
                # never makes a start whose task row exists be undone.
                started = None
                self._trees.committed(member, run_id)
                _quietly(lambda: session.confirm_child_start(run_id))
            # Only now, with the row that ended them durable and the start
            # marked recorded, the decision events for the calls it ended.
            for call in ended:
                try:
                    self._tool_decision(ToolDenied, session, run_id, call, None, "policy")
                except Exception as problem:  # noqa: BLE001 - the task row is recorded either way
                    logging.getLogger(__name__).warning(
                        "A tool decision event failed: %s.", class_label(problem)
                    )

            return self._loop(
                session,
                mode,
                run,
                run_id,
                provider,
                model,
                offered,
                called,
                schema,
                system_prompt,
                plan_gates,
                read_rows=len(stored),
            )
        except Exception as error:
            # THE MODEL'S OWN WORDS DO NOT BELONG IN A RUN ROW OR AN EVENT. A
            # schema violation names the values that missed, so its message
            # carries pieces of the document -- and an event carrying those would
            # put model output in every listener's telemetry, which is the same
            # reason tool arguments are names-only here. The document is already
            # in the thread, in full, where it is read deliberately rather than
            # shipped by default. An error that holds one records its CODE.
            # NOR DOES AN EXCEPTION'S MESSAGE (G-385): a provider, store, host
            # or tool wrote it, and it can carry anything. The harness has no
            # content capture of its own, so it records the class, the harness
            # code and a fixed description, never the message.
            # CLEANUP FIRST, and every step on its own: an exception whose
            # attributes cannot be read (a host subclass that never ran the
            # base constructor, a getter that raises) must not skip any of it.
            leave_own()
            if started is not None and task_recorded[0]:
                # Its task row is written: the start stands, whatever failed
                # after the write (G-387 round 13). Never undone.
                started = None
                _quietly(lambda: self._trees.committed(member, run_id))
                _quietly(lambda: session.confirm_child_start(run_id))
            if started is not None:
                # Failed before its task was recorded: the child is put back
                # as it was before the start, and the tree hold released.
                # Housekeeping: the start's writes are provisional.
                self._undo_start(session, member, run_id, started)
            failure = _failure(error)
            _quietly(lambda: session.fail_run(run_id, failure))
            _quietly(lambda: self._plans.after_turn(session))
            # The holds released BEFORE RunFailed (G-387 round 12): a
            # listener never observes, or races, a hold about to go, and its
            # own BaseException cannot skip the release.
            _quietly(lambda: self._settle_child(session, run_id))
            _quietly(lambda: self._trees.finished(session, run_id))
            _quietly(
                lambda: self._emit(
                    RunFailed(
                        run_id=run_id,
                        session_key=session.key(),
                        failure=failure,
                        steps=run.ledger.steps,
                        at=_now(),
                        harness_agent=session.agent,
                    )
                )
            )
            raise
        except BaseException as error:
            # Not an Exception (G-387 round 10): host code stopping the
            # process. What does not depend on the exception still runs (the
            # start's holds and the run's own holds released, the run marked
            # failed by class, and RunFailed emitted naming the class only, so
            # a listener never sees a run start with no end: round 11), and it
            # propagates unchanged.
            leave_own()
            if started is not None and task_recorded[0]:
                started = None
                _quietly(lambda: self._trees.committed(member, run_id))
                _quietly(lambda: session.confirm_child_start(run_id))
            if started is not None:
                self._undo_start(session, member, run_id, started)
            failure = _failure(error)
            _quietly(lambda: session.fail_run(run_id, failure))
            # The plan reads the thread the run left, as on every other exit
            # (G-387 round 13).
            _quietly(lambda: self._plans.after_turn(session))
            _quietly(lambda: self._settle_child(session, run_id))
            _quietly(lambda: self._trees.finished(session, run_id))
            # Last: a listener's own BaseException cannot skip the holds.
            _quietly(
                lambda: self._emit(
                    RunFailed(
                        run_id=run_id,
                        session_key=session.key(),
                        failure=failure,
                        steps=run.ledger.steps,
                        at=_now(),
                        harness_agent=session.agent,
                    )
                )
            )
            raise
        finally:
            if turn_token is not None:
                _TURN_RUN.reset(turn_token)
            leave_own()

    def _start_child(
        self,
        session: Session,
        mode: AgentMode,
        run: RunContext,
        run_id: str,
        member: Mapping[str, Any],
        context: RunContext | None,
        prompt: str,
        start: Mapping[str, Any],
        binding: ChildBinding | None,
    ) -> dict[str, Any]:
        """A runner start (G-387 start gate): every fact it depends on READ
        first, :func:`~prism_harness.run_starts.gate` asked ONCE, then the
        commit, in order: (a) the tree hold, a conditional locked write;
        (b) the child's records (its count, binding, depth, ceiling and refs),
        one locked write that re-checks the child is not busy. A refusal at
        the gate or at either write raises
        :class:`~prism_harness.run_starts.StartRefused` having written nothing
        (a refused (b) releases (a)). A read that raises fails closed, before
        any write. The runtime then begins the run, (c)."""
        assert context is not None
        hosts = tuple(host for host in start.get("hosts", ()) if isinstance(host, RunContext))
        contexts = (run, *hosts)
        stored = session.thread().rows()
        # The ledger as the gate reads it: the run is narrowed from THIS, so a
        # step spent on the context after the read counts against the child.
        base = run.ledger.steps
        facts, _ = self._allowance_facts(session, mode, "runner", context, prompt, stored, None)
        # What the start's own allowance gives: the runner context's remainder
        # at that read, under every host context the start obeys, capped by
        # max_steps_ceiling.
        left = context.budget.max_steps - base
        for host in hosts:
            left = min(left, host.budget.max_steps - host.ledger.steps)
        facts["context_left"] = left
        first = _allowance(facts, False)
        now, ttl = self._plans.now(), self.lock_ttl(session)
        pending = "none"
        if _awaiting(stored):
            pending = "answerable" if _decidable(session, stored) else "unanswerable"
        if self._plans._ledger_for(session, prompt, context, "runner", mode.name) is not None:
            # Only the plan's own step run under a runner context routes to a
            # plan ledger, and that is refused (plan.budget_context).
            raise PlanRefused("plan.budget_context")
        tree_left = self._trees.start_grant(session, member, ttl)
        saved_tree, saved_plan_ledger = saved_refs(session)
        spends = member["spends"]
        refs = kept_refs(
            member,
            self._plans.ledger_of(spends) if isinstance(spends, str) else None,
            saved_tree,
            saved_plan_ledger,
            False,
        )
        decided = gate(
            {
                "task": start.get("task", prompt),
                "too_deep": context.too_deep() or run.too_deep(),
                "cancelled": any(bool(each.ledger.cancelled) for each in contexts),
                "time_spent": any(_time_spent_closed(each) for each in contexts),
                "steps_left": first["own_left"],
                "cost": _cost_fact(contexts, run),
                "child_busy": session.child_live_hold(now, ttl),
                "child_pending": pending,
                "tree_left": tree_left,
                "plan_left": None,
            }
        )
        found = decided.get("gate")
        if not isinstance(found, dict):
            raise HarnessError.run_not_permitted("This runner start could not be decided.")
        if "refused" in found:
            raise StartRefused(str(found["refused"]), str(found["outcome"]))
        steps = int(found["start"]["steps"])

        def recheck() -> None:
            # Under (b)'s lock: the child's pending approvals re-read with the
            # gate's answerability rule, so an approval a person can answer,
            # parked since the gate read, refuses the start.
            rows = session.thread().rows()
            if _awaiting(rows) and _decidable(session, rows):
                raise StartRefused("child_waiting", "denied")

        # (a) the tree hold: refused (nothing stored) when a race since the
        # read shrank the grant.
        self._trees.child_started(session, member, ttl, tree_left, run_id)
        # (b) the child's records and its mode, in one locked write
        # re-checking busy and pending. What it replaced is kept, so a failure
        # before the task is recorded puts the child back (_undo_start).
        written: dict[str, Any] | None = None
        try:
            if binding is not None:
                written = session.start_child(
                    run_id,
                    now,
                    ttl,
                    steps,
                    binding.to_row(),
                    member["depth"],
                    first["ceiling"],
                    refs,
                    mode.name,
                    recheck,
                )
        except HarnessError as error:
            self._trees.release_start(session, member, run_id)
            if harness_code(error) == CHILD_TURN_LIVE:
                raise StartRefused("child_busy", "denied") from None
            raise
        except BaseException:
            self._trees.release_start(session, member, run_id)
            raise
        return {
            "facts": facts,
            "first": first,
            "steps": steps,
            "tree_left": tree_left,
            "base": base,
            "written": written,
        }

    def _undo_start(
        self,
        session: Session,
        member: Mapping[str, Any],
        run_id: str,
        started: Mapping[str, Any],
    ) -> None:
        """A committed runner start failed before its task was recorded: the
        child's records and mode go back to what they were before (b) (only
        where they are still what (b) wrote), its hold on the child's count
        goes with them, and the tree hold (a) is released. HOUSEKEEPING ONLY
        (G-387 round 9): what (b) wrote is provisional, absent to every reader
        until a task row confirms it, so a failed undo changes nothing a
        reader sees. Never raises: a failed write is reported by class, and a
        hold left behind is charged in full once it expires."""
        live = self._trees.members.get(run_id)
        if live is not None:
            live["child_claim"] = False
        written = started.get("written")
        if isinstance(written, dict):
            try:
                session.undo_child_start(run_id, written)
            except Exception as failure:  # noqa: BLE001 - charged in full once it expires
                logging.getLogger(__name__).warning(
                    "A child start could not be undone: %s.", class_label(failure)
                )
        _quietly(lambda: self._trees.release_start(session, member, run_id))

    def _record_task(
        self,
        session: Session,
        mode: AgentMode,
        run_id: str,
        prompt: str,
        attachments: Sequence[object],
        recorded: list[bool],
    ) -> list[ToolCallInput]:
        """A runner start's task row (G-387 round 9). It ENDS every call still
        pending on the child: each is recorded as not run, in the same write
        as the task, so no decision recorded on it (before the start, or by
        anyone while it ran) is ever acted on, by this start or any later
        turn. Under the session lock, reading the pending calls inside it.

        The row closing them carries :data:`ENDED_BY_START`,
        ``{approval_ids: [...]}``, the approvals it ended (rounds 11 and 12,
        the reference's shape): the only thing that marks them ended. Returns the
        calls it ended; their events are the caller's, once the start is
        marked recorded."""
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
        ended: list[ToolCallInput] = []

        def record(live: Session) -> None:
            rows: list[dict[str, Any]] = []
            thread = live.thread()
            stored = thread.rows()
            work = self._unresolved(live, mode, stored)
            if work is not None:
                calls, results, decisions, approval_ids, _, _, _, _ = work
                closing = [
                    ToolCallInput(
                        id=row["id"],
                        name=row["name"],
                        arguments=row["arguments"]
                        if isinstance(row.get("arguments"), dict)
                        else {},
                        result_id=row["result_id"],
                    )
                    for row in calls
                ]
                ended.extend(closing)
                # Each call ends with its OWN approval id (G-392), the ids in
                # the reference's mark (G-387 round 12).
                ended_row = ended_by_start_row(
                    closing,
                    approval_ids,
                    results,
                    list(decisions.values()),
                )
                # The step it ends, by position (G-387 round 15).
                rows.append(answering_step(ended_row, _step_position(thread, stored)))
            _record_in_run(thread, [*rows, turn], run_id)
            # Recorded as written AT the write (G-387 round 13): a failure
            # after it (the lock's release, an event, a read in cleanup) never
            # makes the start read as unrecorded, so it is never undone.
            recorded[0] = True

        options = self._settings(session)
        session.lock(
            record,
            options.get("lock_ttl", _RESOLUTION_LOCK_SECONDS),
            options.get("lock_wait", _RESOLUTION_WAIT_SECONDS),
        )
        return ended

    def _allowance_facts(
        self,
        session: Session,
        mode: AgentMode,
        kind: str,
        context: RunContext | None,
        prompt: str,
        stored: Sequence[Mapping[str, Any]],
        saved_binding: ChildBinding | None,
    ) -> tuple[dict[str, Any], bool]:
        """The facts a run's allowance is worked out from (G-387), and whether
        the parked row's steps-used count is not one the harness could have
        written (the run then fails closed: no model step).

        ``used`` is read for any turn that resumes a parked run (no new
        message, a row waiting on approvals) except a runner start, which is
        a new run: the larger of the row's ``harness_steps_used`` and the
        transcript's count of the run's own model steps. A row written before
        that field existed uses the transcript's count; a child with a saved
        ceiling, which always parks with its count, fails closed without one."""
        used = 0
        forged = False
        if kind != "runner":
            ceiling_state, _ = session.child_ceiling_state()
            tail = paused_approval_row(stored)
            # Linked: the runner saved something with the session, or its
            # parked row names the tree a child belongs to, or a binding on a
            # row written before rows carried their steps used (a legacy
            # child). A top-level session a host bound writes its binding on
            # its rows too, with its steps used, and is not a child.
            linked = (
                # A subagent child's address: whatever else was or was not
                # saved (a child parked by v0.5.3 saved nothing but its
                # thread), it is a child, and without a ceiling it fails closed.
                "::sub::" in session.scope
                or session.child_refs() is not None
                or session.child_binding() is not None
                or (tail is not None and "harness_tree" in tail)
                or (
                    tail is not None
                    and "harness_binding" in tail
                    and "harness_steps_used" not in tail
                )
            )
            if ceiling_state == "garbage" or (ceiling_state == "absent" and linked):
                # A linked child with no usable saved ceiling fails closed:
                # it acts on its decisions and takes no model step. Only a
                # session that was never a child runs on its mode.
                forged = True
            row = paused_approval_row(stored) if prompt == "" else None
            found = steps_used(row)
            if row is None:
                used = 0
            elif found == "forged":
                forged = True
            elif found == "absent" and session.child_ceiling() is not None:
                # A child the runner started on this version always parks with
                # its count; a row without one was not written by it, and is
                # never read as a laxer, legacy measure.
                forged = True
            elif found == "absent":
                used = approval_run_steps(stored) or 0
            else:
                # The larger of the row's count and the transcript's, so a row
                # forged lower than the steps it shows gains nothing.
                assert isinstance(found, int)
                used = max(found, approval_run_steps(stored) or 0)
        steps_ceiling = self._settings(session).get("max_steps_ceiling")
        return {
            "context": kind,
            "mode_steps": mode.max_steps,
            "saved_ceiling": None if kind == "runner" else session.child_ceiling(),
            "context_left": None
            if context is None or kind in ("none", "metadata")
            else context.budget.max_steps - context.ledger.steps,
            "steps_ceiling": steps_ceiling,
            "used": used,
            "tree_left": None,
            "plan_left": None,
            "took": 0,
        }, forged

    def _with_plan_tools(
        self,
        session: Session,
        mode: AgentMode,
        offered: list[HarnessTool],
        scope: tuple[list[str], bool] | None,
    ) -> tuple[list[HarnessTool], dict[str, PlanTool]]:
        """The run's tools with the plan tools added, and the plan tools by name.

        ADDED AFTER the mode's list and its approval gate (G-360). A plan tool
        gates itself -- a draft, a change and an approval pause are approvals
        bound to their own arguments -- so neither the mode's
        ``requires_approval`` nor the host tool policy ever decides one. They
        still pass the authorizer, and a resumed run still offers only what its
        stored approval scope names.
        """
        tools = self._plans.tools(session, mode.name)
        if scope is not None:
            tools = [tool for tool in tools if tool.name in scope[0]]
        if not tools:
            return offered, {}
        names = {tool.name for tool in tools}
        by_name: dict[str, HarnessTool] = {tool.name: tool for tool in tools}
        added = (
            self._authorizer.allowed(session, by_name)
            if self._authorizer is not None
            else list(by_name.values())
        )
        # Gates only for the plan tools the authorizer let through: one its
        # offer policy withheld is a tool this run does not have, and a call to
        # it is answered like a call to any tool that was never offered.
        allowed = {tool.name for tool in added}
        return [tool for tool in offered if tool.name not in names] + added, {
            tool.name: tool for tool in tools if tool.name in allowed
        }

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
        system_prompt: str | None = None,
        plan_gates: Mapping[str, PlanTool] | None = None,
        read_rows: int | None = None,
    ) -> AgentResponse:
        """``read_rows``: how many rows the thread held when the run read its
        history; a step recorded at or after that index by another writer
        that waits on approval stops the run before its next model step."""
        thread = session.thread()
        gates = plan_gates or {}
        text = ""
        finish_reason = "stop"

        while True:
            # BEFORE the step, never after. Checking afterwards means the step
            # that broke the limit has already been paid for.
            stop_code = run.ledger.exhaustion_code(run.budget)

            if stop_code is not None:
                exhausted = run.ledger.exhaustion(run.budget) or stop_code
                spent = self._finish(
                    session, run_id, called, "budget_exhausted", text, run, exhausted, stop_code
                )
                # The structured path ends a spent run exactly as send() does,
                # in the structured shape, with no document.
                return spent if schema is None else _structured(spent, None)

            # This agent's own ceiling, in the same ledger. Raised rather than
            # finished: a subagent child refused here comes back to its parent
            # as a framed result, and only this agent stops.
            refused = run.agent_stop()
            if refused is not None:
                raise refused

            rows = thread.rows()
            # A NESTED PARK (G-387 round 16, the PHP reference's rule): a
            # newer step on this session, written after this run read the
            # thread by anyone else (a tool's own nested send, a host), waits
            # on approval. Another model step here would answer past it and
            # replace the step its caller was just told is waiting, so the
            # run stops instead, before the provider is asked anything.
            if read_rows is not None and pending_approval_ids(rows):
                newest = requested_step(rows)
                if (
                    newest is not None
                    and newest >= read_rows
                    and not self._own_step(thread, newest, run_id)
                ):
                    raise HarnessError.parked_elsewhere()

            request = LlmRequest(
                system_prompt=mode.system_prompt if system_prompt is None else system_prompt,
                messages=placed_view(rows),
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
            run.ledger.record_cost(response.cost_usd, run.agent)
            text = response.text
            finish_reason = response.finish_reason

            # A client can hand back arguments it could not parse. Such a call is
            # recorded with none, asks nobody and runs nothing: the model is told.
            # So are arguments whose digest cannot be computed: every decision
            # about a call is recorded with that digest, and one that could not
            # be would fail the call's approval on every later send().
            unusable = {
                call.id
                for call in response.tool_calls
                if not _digestible(usable_arguments(call.arguments))
            }
            calls = [_call_input(call) for call in response.tool_calls]
            failed_finish = finish_reason in ("error", "unknown")
            # The host policy is read per call, never cached for the run, so a
            # replacement applies from the next call (G-326).
            auto_allowed: list[ToolCallInput] = []
            gated: list[ToolCallInput] = []
            for call in [] if failed_finish else calls:
                if call.id in unusable:
                    continue
                gate = gates.get(call.name)
                if gate is not None:
                    # A plan tool decides for itself, with these arguments
                    # (G-360). The host's call policy first: a call it forbids
                    # records nothing, asks no one, and is refused when it runs.
                    authorized = self._authorizer is None or self._authorizer.allows_call(
                        session, gate, dict(call.arguments)
                    )
                    if gate.requires_approval(dict(call.arguments), authorized):
                        gated.append(call)
                    continue
                if not mode.needs_approval(call.name):
                    continue
                declared = _offered(offered, call.name)
                if declared is not None and self._tool_policy.allows(declared):
                    auto_allowed.append(call)
                else:
                    gated.append(call)
            requests: list[dict[str, Any]] = [
                {
                    "approval_id": f"apr_{uuid.uuid4().hex}",
                    "tool_call_id": call.id,
                    "arguments_format": ARGUMENTS_AS_SENT,
                }
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

            row = assistant_row(response.text, calls, metadata, requests)
            if requests:
                row["harness_tool_names"] = [tool.name for tool in offered]
                # The steps this run has used, children included, and what
                # its allowance leaves (G-387): the facts its resume reads,
                # from the allowance recorded when it began and what its
                # ledger gained since. harness_steps_used is what decides the
                # resume; harness_steps_left is reported, never read.
                parked_as = self._trees.member(run_id)
                allotted = parked_as.get("allowance") if parked_as is not None else None
                if parked_as is not None and isinstance(allotted, dict):
                    took = run.ledger.steps - int(parked_as["begin_steps"])
                    at_park = _allowance(
                        {**allotted["facts"], "took": took}, bool(allotted["forged"])
                    )
                    row["harness_steps_used"] = at_park["used_at_park"]
                    row["harness_steps_left"] = at_park["left_at_park"]
                # What a resume needs to continue this run as the same run.
                if run.agent is not None:
                    row["harness_agent_spend"] = {
                        "cost_usd": run.ledger.agent_cost_usd(run.agent),
                        "unmetered_runs": run.ledger.agent_unmetered_runs(run.agent),
                    }
                if run.binding is not None:
                    row["harness_binding"] = run.binding.to_row()
                    # The shared plan budget the tree spends from, so the child
                    # resumed outside the runner spends from it too (G-366).
                    # Only for a run that belongs to a tree it did not open
                    # (a runner-started child, or a turn continuing a saved
                    # tree): a run that opened its own would otherwise read it
                    # back on its next turn as a tree it was saved with.
                    member = self._trees.member(run_id)
                    if member is not None and member["route"] != "open":
                        spends = member["spends"]
                        plan_ledger = self._plans.ledger_of(run_id) or (
                            self._plans.ledger_of(spends) if isinstance(spends, str) else None
                        )
                        if plan_ledger is not None:
                            row["harness_plan_ledger"] = plan_ledger
                        # And the run tree whose budget a resumed child
                        # continues: the one its membership names.
                        tree = member["tree"]
                        if isinstance(tree, dict):
                            row["harness_tree"] = {"session": tree["session"], "run": tree["run"]}
            written = _record_in_run(thread, [row], run_id)
            # Where this step sits: its own result row names it (G-387 round
            # 15), so the result answers it wherever another writer's row
            # lands before it.
            step_position = written[-1].position if written else 0

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

                return _structured(finished, response.structured)

            # The calls that need nobody run now, as in the reference, and their
            # results are recorded even when the step then stops for a person.
            results: list[dict[str, Any]] = []

            for call in gated:
                self._tool_decision(
                    ToolApprovalAsked, session, run_id, call, _offered(offered, call.name), "person"
                )

            # WRITE-AHEAD for the ungated batch too (G-387 round 16): one
            # started row per call the step runs at once, in ONE write before
            # the first of them runs (the reference's shape, approval id
            # null), so a call that ran and whose results row then failed to
            # write reads INTERRUPTED ("it may have run"), never "not run",
            # and is never given a replaced denial.
            batch = [call for call in calls if call not in gated and call.id not in unusable]
            if batch:
                _record_in_run(
                    thread,
                    [
                        started_row({"id": call.id, "name": call.name}, None, step_position)
                        for call in batch
                    ],
                    run_id,
                )

            for call in calls:
                if call in gated:
                    continue

                if call.id in unusable:
                    results.append(
                        tool_result_entry(
                            call, f"Not run: the arguments for {call.name} are not a JSON object."
                        )
                    )
                    continue

                if call in auto_allowed:
                    # Checked again at the moment it would run: a revoke since
                    # the step began refuses it rather than letting it stand.
                    declared = _offered(offered, call.name)
                    if declared is None or not self._tool_policy.allows(declared):
                        self._tool_decision(ToolDenied, session, run_id, call, declared, "policy")
                        results.append(
                            tool_result_entry(
                                call,
                                "Not run: this call needs approval, and the tool policy no longer allows it.",
                            )
                        )
                        continue
                    self._tool_decision(ToolAutoAllowed, session, run_id, call, declared, "policy")

                called.append(call.name)
                results.append(self._invoke(offered, call))

            if results:
                _record_in_run(
                    thread, [answering_step(tool_result_row(results), step_position)], run_id
                )

            if requests:
                # FAILS CLOSED. The gated calls have not run, and the requests are
                # in the thread, so a different process can resume after a person
                # answers.
                self._plans.after_turn(session)
                self._settle_child(session, run_id)
                self._trees.finished(session, run_id)
                parked = AgentResponse(
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
                # A structured turn that parks answers in the structured shape,
                # with no document yet, as a spent one does.
                return parked if schema is None else _structured(parked, None)

    @staticmethod
    def _own_step(thread: Thread, index: int, run_id: str) -> bool:
        """Whether the step at ``index`` of the thread's rows is one this run
        recorded itself (by its stored entry's run id, which only says who
        wrote it, never whether a mark counts)."""
        entries = thread.messages()
        return 0 <= index < len(entries) and entries[index].run_id == run_id

    def _record_new_message(self, session: Session, run_id: str, turn: dict[str, Any]) -> None:
        """A new message's user row, in ONE write (G-404): any earlier step a
        later one replaced with calls left without an outcome (a thread the
        0.5.x line wrote) gets its denials first, then the paused step's
        denials and the row itself (see :meth:`Thread.record`), all read and
        written inside the thread's own lock.

        It never waits on the session lock a resume holds (G-387 round 15): a
        call that resume is running has its started row, so it is
        answered-in-progress and gets no denial; nothing else of the step
        runs once this row is in."""
        thread = session.thread()

        def rows_to_write(rows: list[dict[str, Any]], positions: list[int]) -> list[dict[str, Any]]:
            earlier = replaced_earlier_steps(rows)
            return [*(replaced_step_row(positions[at], calls) for at, calls in earlier), turn]

        _record_in_run(thread, [turn], run_id, rows_to_write)

    def _resolve_approvals(
        self,
        session: Session,
        mode: AgentMode,
        offered: list[HarnessTool],
        run_id: str,
        called: list[str],
        legacy_diagnostic_emitted: bool,
        history: list[dict[str, Any]] | None = None,
    ) -> list[HarnessTool]:
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
        # ``history``: the rows the run already read, when nothing was
        # written since (a resume): no second read (G-387 round 16).
        first = session.thread().rows() if history is None else history
        if self._unresolved(session, mode, first) is None and not replaced_earlier_steps(first):
            return list(offered)

        # Then under the session lock, reading again inside it. Two workers
        # resuming the same session at once would otherwise both find the
        # approved call with no result, and both run it. The second now waits,
        # finds the result the first recorded, and runs nothing. If it cannot get
        # the lock it raises, and nothing runs.
        options = self._settings(session)
        seconds = options.get("lock_ttl", _RESOLUTION_LOCK_SECONDS)

        def resolve(live: Session) -> list[HarnessTool]:
            thread = live.thread()
            # An earlier step a later one replaced, left with calls that have
            # no outcome (a thread the 0.5.x line wrote): each gets its one
            # denial first, and nothing of it runs (G-387 round 13).
            # ONE read of the thread for everything the answer needs (G-387
            # round 16): the rows, their positions and run ids together.
            stored, entries = thread.snapshot()
            earlier = replaced_earlier_steps(stored)
            if earlier:
                _record_in_run(
                    thread,
                    [replaced_step_row(entries[at].position, calls) for at, calls in earlier],
                    run_id,
                )
                stored, entries = thread.snapshot()
            work = self._unresolved(live, mode, stored)

            if work is None:
                return list(offered)
            step = last_tool_step(stored)
            position = entries[step].position if step is not None else 0
            return answer(live, thread, work, position, stored, entries)

        def answer(
            live: Session,
            thread: Thread,
            work: _Unresolved,
            position: int,
            stored: list[dict[str, Any]],
            entries: list[ThreadMessage],
        ) -> list[HarnessTool]:

            calls, results, decisions, approval_ids, scope, ambiguous, repeated, indexes = work
            # A step a row in the thread already superseded, never answered
            # for it (a thread the 0.5.x line wrote): each call with no
            # result gets its one replaced-step denial (G-387 round 13).
            # The owning runs come only from rows the harness wrote (G-387
            # round 16): a run id a caller passed never makes a row a run's.
            superseded = step_superseded(
                list(
                    zip(
                        stored,
                        [entry.run_id if entry.written_by_harness else None for entry in entries],
                        strict=True,
                    )
                )
            )
            # Bind calls, decisions and scope from the same locked snapshot.
            # A concurrent narrower approval also bounds the continuation.
            if scope is not None and scope[1] and not legacy_diagnostic_emitted:
                logging.getLogger(__name__).warning(
                    "Pending approval has no valid stored tool scope; offering approved requests only."
                )
            bounded = [tool for tool in offered if scope is not None and tool.name in scope[0]]
            resolved: list[tuple[int, dict[str, Any]]] = []
            unusable = unusable_stored_calls(stored) | {
                row["id"] for row in calls if not _digestible(row["arguments"])
            }

            for row, approval_id, index in zip(calls, approval_ids, indexes, strict=True):
                call = ToolCallInput(
                    id=row["id"],
                    name=row["name"],
                    arguments=row["arguments"],
                    result_id=row["result_id"],
                )
                decision = decisions.get(approval_id) if approval_id is not None else None

                declared = _offered(bounded, call.name)
                if superseded:
                    self._tool_decision(ToolDenied, live, run_id, call, declared, "policy")
                    resolved.append((index, tool_result_entry(call, SUPERSEDED_RESULT)))
                elif repeated:
                    # A repeated id cannot say which decision answers which
                    # call, so none of the step's gated calls runs (G-392).
                    self._tool_decision(ToolDenied, live, run_id, call, declared, "policy")
                    resolved.append((index, tool_result_entry(call, REPEATED_CALL_ID)))
                elif decision is not None and decision["approved"] and call.id in unusable:
                    resolved.append(
                        (
                            index,
                            tool_result_entry(
                                call,
                                f"Not run: the arguments for {call.name} are not a JSON object.",
                            ),
                        )
                    )
                elif decision is not None and decision["approved"] and call.id in ambiguous:
                    # Stored by 0.5.2 or earlier with no marker. Not run, and
                    # recorded without arguments; the model is told to call
                    # again, and a new call is a new approval for the person.
                    cleared = ToolCallInput(
                        id=call.id, name=call.name, arguments={}, result_id=call.result_id
                    )
                    self._tool_decision(
                        ToolApprovalExpired, live, run_id, cleared, declared, "policy"
                    )
                    resolved.append(
                        (index, tool_result_entry(cleared, _approval_expired(call.name)))
                    )
                elif decision is not None and decision["approved"]:
                    # WRITE-AHEAD (G-387 round 15): the started row first, in
                    # one write with the check that the call is still open. A
                    # superseding row recorded before it has denied the call,
                    # so it does not run; one recorded after it leaves the
                    # call answered-in-progress, and its result below pairs to
                    # it wherever it lands.
                    if not self._start_call(thread, run_id, position, row, approval_id, index):
                        # Not the call the person was shown, or no longer
                        # open: never run, and this its one outcome unless
                        # it has one already (left out below).
                        resolved.append((index, tool_result_entry(call, NOT_THE_APPROVED_CALL)))
                        continue
                    self._tool_decision(ToolApproved, live, run_id, call, declared, "person")
                    called.append(call.name)
                    resolved.append((index, self._run_approved(bounded, call)))
                elif decision is None:
                    self._tool_decision(ToolDenied, live, run_id, call, declared, "person")
                    resolved.append(
                        (index, tool_result_entry(call, "No approval response provided"))
                    )
                else:
                    self._tool_decision(ToolDenied, live, run_id, call, declared, "person")
                    resolved.append(
                        (
                            index,
                            tool_result_entry(
                                call, decision["reason"] or "User denied tool execution"
                            ),
                        )
                    )

            def answer_row(
                rows: list[dict[str, Any]], positions: list[int]
            ) -> list[dict[str, Any]] | None:
                # Read in the answer's own write. EXACTLY ONE STORED OUTCOME
                # PER CALL (G-387 round 15): every result the step already
                # holds is left out (its earlier results, and a denial a
                # superseding row gave a call meanwhile).
                at = _at_position(rows, positions, position)
                every = [*results, *(entry for _, entry in resolved)]
                entries = new_results(rows, at, every) if at is not None else every
                row = tool_result_row(entries, list(decisions.values()))
                if superseded:
                    row[SUPERSEDED] = {
                        "approval_ids": list(
                            dict.fromkeys(approval for approval in approval_ids if approval)
                        )
                    }
                if not row["tool_results"] and not row["tool_approval_responses"]:
                    return None
                return [answering_step(row, position) if row["tool_results"] else row]

            _record_in_run(thread, [], run_id, answer_row)
            return bounded

        wait = options.get("lock_wait", _RESOLUTION_WAIT_SECONDS)
        resolving = _RESOLVING.get() or {}
        token = _RESOLVING.set({**resolving, session.key(): (seconds, wait)})
        try:
            return session.lock(resolve, seconds, wait)
        finally:
            _RESOLVING.reset(token)

    @staticmethod
    def _start_call(
        thread: Thread,
        run_id: str,
        position: int,
        call: Mapping[str, Any],
        approval_id: str | None,
        index: int,
    ) -> bool:
        """Record the started row for the call at ``index`` of the step at
        thread ``position`` (G-387 round 15), in one write with the check
        that the call is still open (no outcome among the rows its step owns,
        not started already). False, writing nothing, when it is not: it
        must not run."""
        started: list[bool] = []

        def row_for(
            rows: list[dict[str, Any]], positions: list[int]
        ) -> list[dict[str, Any]] | None:
            at = _at_position(rows, positions, position)
            if at is None or index in started_calls(rows, at):
                return None
            if index in call_results(step_calls(rows[at]), owned_rows(rows, at)):
                return None
            # Its own request must BIND it (item 7): the tool name and the
            # arguments the person was shown, as stored and as held here.
            own: Mapping[str, Any] | None = next(
                (r for r, c in step_requests(rows, at) if c == index), None
            )
            if own is None and not _dicts_of(rows[at].get("tool_approval_requests")):
                own = {}  # 0.1.0: the requests are rows of their own, unbound
            if own is None or not binds(own, step_calls(rows[at])[index], call.get("arguments")):
                return None
            # And its approval must still STAND (G-387 round 16): a denial
            # recorded by any channel since the resolver read its decisions
            # (a host's raw row while an earlier call ran) refuses it here,
            # in the same write as the started row, so it never runs.
            standing = standing_decisions(rows).get(approval_id or "")
            if approval_id is not None and (standing is None or standing["approved"] is not True):
                return None
            started.append(True)
            return [started_row(call, approval_id or "", position)]

        _record_in_run(thread, [], run_id, row_for)
        return bool(started)

    @staticmethod
    def _unresolved(
        session: Session, mode: AgentMode, stored: list[dict[str, Any]] | None = None
    ) -> _Unresolved | None:
        """The gated calls of the last tool-calling turn that have no result
        yet, each with the approval id of its OWN request (or None), the
        results the turn already has (one per call, in call order), and
        whether the turn repeats a call id (G-392).

        Results are matched to calls BY POSITION (G-387 round 13), never
        through a map keyed by call id: a gated call that shares its id with
        an ungated call that already ran is still pending, so a resume denies
        it and a runner start ends it with its own approval."""
        rows = session.thread().rows() if stored is None else stored
        view = thread_view(rows)
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
        # The same step as stored: its results are read from the stored rows,
        # which the view folds by call id.
        at = last_tool_step(rows)
        if at is None:  # never: the view's step is the stored one (G-408)
            return None
        answered = call_results(assistant["tool_calls"], step_rows(rows, at))
        results = [answered[i] for i in sorted(answered)]
        # Which decision counts comes from the stored rows, the FIRST recorded
        # for an approval id standing; the view folds later over earlier. A
        # decision applies ONLY to the step whose approvals were requested,
        # selected by the one rule every decision reader uses (G-397): when
        # the step this resolves is not that step (a row recorded after the
        # park ended it), none of its calls is acted on as decided.
        decisions = first_decisions(rows) if requested_step(rows) == at else {}
        # Each request paired with its own call by position, never through a
        # map keyed by call id, where the last request under a repeated id
        # would answer every call carrying it (G-392); and never with a call
        # the step ran at once (G-387 round 13).
        # Only the requests this step asked FIRST (G-408): a copy repeating
        # an earlier step's approval id asks for nothing.
        stored_pairs = step_requests(rows, at)
        requested: dict[int, str] = {
            call: request["approval_id"]
            for request, call in stored_pairs
            if call is not None and isinstance(request.get("approval_id"), str)
        }
        if (
            not _dicts_of(rows[at].get("tool_approval_requests"))
            and (assistant["tool_approval_requests"])
        ):
            # Requests kept in rows of their own (0.1.0): paired on the view.
            requested = {
                call: request["approval_id"]
                for request, call in paired_requests(assistant)
                if call is not None
            }

        # A call that has a result is DONE, whatever its decision says. So is
        # one the resolver STARTED (G-387 round 15): running, or interrupted,
        # it may have run, and it never runs again. A call that needs nobody
        # is not this method's to run.
        busy = started_calls(rows, at)
        pending = [
            (row, requested.get(index), index)
            for index, row in enumerate(assistant["tool_calls"])
            if index not in answered
            and index not in busy
            and (index in requested or mode.needs_approval(row["name"]))
        ]

        if not pending:
            return None

        return (
            [row for row, _, _ in pending],
            results,
            decisions,
            [approval_id for _, approval_id, _ in pending],
            approval_tool_scope(rows),
            ambiguous_calls(assistant),
            repeats_call_ids(assistant),
            [index for _, _, index in pending],
        )

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
            if harness_code(error) == "call_not_authorized":
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
            # Written down inside the same guard: a result that cannot be
            # recorded is the tool's failure too. Escaping here would leave
            # every call resolved before it unrecorded, to run again on the
            # next resume.
            text = result if isinstance(result, str) else json.dumps(result)
        except HarnessError as error:
            # A refused call propagates. A refusal fed back to the model reads
            # as a retryable failure, which is the opposite of a guard.
            if harness_code(error) == "call_not_authorized":
                raise

            return tool_result_entry(call, f"The tool failed: {_text_of(error)}")
        except Exception as error:  # noqa: BLE001 - a tool is someone else's code
            # A failed tool is a RESULT, not a crashed run: the model can often
            # recover, and losing the whole turn to one bad call is worse.
            return tool_result_entry(call, f"The tool failed: {_text_of(error)}")

        return tool_result_entry(call, text)

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
        stopped_code: str | None = None,
    ) -> AgentResponse:
        session.complete_run(run_id, finish_reason, called)
        # G-360: when the run is over, the plan reads the thread it left.
        self._plans.after_turn(session)
        # A root run that holds part of a run tree's budget settles it, and
        # a child its delegation's count.
        self._settle_child(session, run_id)
        self._trees.finished(session, run_id)
        self._emit(
            RunFinished(
                run_id=run_id,
                session_key=session.key(),
                finish_reason=finish_reason,
                tool_calls=tuple(called),
                steps=run.ledger.steps,
                cost_usd=None if run.ledger.unmetered_runs > 0 else run.ledger.cost_usd,
                at=_now(),
                harness_agent=session.agent,
            )
        )

        return AgentResponse(
            run_id=run_id,
            text=text,
            steps=run.ledger.steps,
            tool_calls=called,
            finish_reason=finish_reason,
            stopped_because=stopped_because,
            stopped_code=stopped_code,
        )

    def _settle_child(self, session: Session, run_id: str) -> None:
        """Replace a child run's claim on its delegation budget with the steps
        it took, children included (G-387). A failed write is reported; the
        hold is then charged in full once it expires."""
        member = self._trees.members.get(run_id)
        if member is None or not member.get("child_claim"):
            return
        member["child_claim"] = False
        took = member["ledger"].steps - int(member.get("begin_steps", member["ledger"].steps))
        try:
            session.settle_child_steps(run_id, took)
        except Exception as failure:  # noqa: BLE001 - the hold is charged in full once it expires
            logging.getLogger(__name__).warning(
                "A child step count update failed: %s.", class_label(failure)
            )

    def _emit(self, event: HarnessEvent) -> None:
        if self._events is not None:
            self._events.emit(event)


#: What :meth:`AgentRuntime._unresolved` finds: the open calls, the results
#: the step has, the decisions, each open call's approval id, the tool scope,
#: the ambiguous call ids, whether the step repeats an id, and each open
#: call's index in its step.
_Unresolved = tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[str | None],
    tuple[list[str], bool] | None,
    set[str],
    bool,
    list[int],
]


def _dicts_of(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _step_position(thread: Thread, rows: Sequence[Mapping[str, Any]]) -> int:
    """The thread POSITION of the last step with tool calls in ``rows`` (as
    :meth:`Thread.rows` read them), or 0."""
    at = last_tool_step(rows)
    if at is None:
        return 0
    entries = thread.messages()
    return entries[at].position if at < len(entries) else 0


def _at_position(
    rows: Sequence[Mapping[str, Any]], positions: Sequence[int], position: int
) -> int | None:
    """The index in ``rows`` of the step at thread ``position`` (``positions``
    the rows' stored positions, index for index), or None."""
    for index, stored in enumerate(positions):
        if stored == position and index < len(rows):
            return index if step_calls(rows[index]) else None
    return None


def _awaiting(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the session waits on a person's approval a resume would act
    on: one still undecided, or one decided whose call has no result yet
    (G-387 round 10). Either refuses a runner start while it is answerable,
    so a delegation never voids a decision a person recorded."""
    return bool(pending_approval_ids(rows)) or bool(decided_unresolved_ids(rows))


def _text_of(error: BaseException) -> str:
    """A failed tool's message, as its result for the model. A message that
    cannot be read names the class only, so the failure stays a result."""
    try:
        return str(error)
    except Exception:  # noqa: BLE001 - an unreadable message names its class
        return class_label(error)


def _quietly(step: Callable[[], object]) -> None:
    """One cleanup step of a failed run, whose own failure never skips the
    next: reported by class only."""
    try:
        step()
    except Exception as failure:  # noqa: BLE001 - cleanup reports, never raises
        logging.getLogger(__name__).warning(
            "A failed run's cleanup step failed: %s.", class_label(failure)
        )


def _nothing() -> None:
    """The leave of a turn that is not a runner start: it entered no view."""


def _failure(error: BaseException) -> str:
    """What a failed run records (G-385, G-387): a structured failure's code,
    or the class, harness code and a fixed description. Every attribute read
    guarded: a missing or raising one names the class only."""
    code = harness_code(error)
    try:
        document = getattr(error, "document", None) if isinstance(error, HarnessError) else None
    except Exception:  # noqa: BLE001 - a getter that raises holds no document
        document = None
    if document is not None and code:
        return code
    try:
        return failure_of(error)
    except Exception:  # noqa: BLE001 - never fails the cleanup it is part of
        return f"{class_label(error)}: the run failed."


def _allowance(facts: Mapping[str, Any], forged: bool) -> dict[str, Any]:
    """The run's allowance from its facts (G-387); a forged steps-used count
    fails closed."""
    found = allowance(facts)
    if "invalid" in found:
        # Every kind the runtime hands over is known and carries what it
        # needs; an invalid answer is a bug, and a run never starts on one.
        raise HarnessError.run_not_permitted("This run's step allowance could not be decided.")
    result: dict[str, Any] = (fail_closed(found) if forged else found)["allowance"]
    return result


def _time_spent_closed(run: RunContext) -> bool:
    """Whether a context's time is spent, for the start gate: a limit that is
    not a finite number (NaN compares false with everything) is spent, never
    read as no limit."""
    limit: Any = run.budget.max_seconds
    if limit is None:
        return False
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit):
        return True
    return run.ledger.elapsed_seconds() >= limit


def _cost_fact(contexts: Sequence[RunContext], run: RunContext) -> str:
    """The start gate's cost fact: ``spent`` when any cost limit that would
    stop the child is spent (a context's cost cap, or the child agent's spend
    ceiling), ``unenforceable`` when one cannot be metered, else ``ok``. A cap
    that is not a finite number is spent."""
    spent = unenforceable = False
    for each in contexts:
        cap: Any = each.budget.max_cost_usd
        if cap is None:
            continue
        unusable = (
            isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap)
        )
        if unusable or each.ledger.cost_usd >= cap:
            spent = True
        elif each.ledger.unmetered_runs > 0:
            unenforceable = True
    stop = run.agent_stop()
    if stop is not None:
        if stop.code == "agent_spend_ceiling_exceeded":
            spent = True
        else:
            unenforceable = True
    return "spent" if spent else "unenforceable" if unenforceable else "ok"


def _decidable(session: Session, rows: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a decision on the session's pending approval would be accepted
    (the answerability rule, :func:`assert_decidable`). Only the rule's own
    answers make it unanswerable: a read that raises is not an answer, and
    propagates, so the start it decides fails closed (G-387)."""
    return _unanswerable(session, rows) is None


def _time_spent(run: RunContext) -> bool:
    """Whether the run's context has a time limit and it is spent. A limit
    that is not a finite number is spent (G-387)."""
    return _time_spent_closed(run)


def _structured(response: AgentResponse, document: Any) -> StructuredAgentResponse:
    return StructuredAgentResponse(
        run_id=response.run_id,
        text=response.text,
        steps=response.steps,
        tool_calls=response.tool_calls,
        finish_reason=response.finish_reason,
        pending_approvals=response.pending_approvals,
        stopped_because=response.stopped_because,
        stopped_code=response.stopped_code,
        structured=document,
    )


def _approval_expired(tool: str) -> str:
    return (
        f"Not run: the approval for {tool} was recorded by an earlier version and has expired. "
        "Call the tool again if it is still needed, and the person will be asked to approve it again."
    )


def _offered(offered: Sequence[HarnessTool], name: str) -> HarnessTool | None:
    return next((tool for tool in offered if tool.name == name), None)


def record_approval(
    session: Session, approval_id: str, approved: bool, reason: str | None = None
) -> None:
    """Answer a pending approval, durably.

    The decision is RECORDED IN THE THREAD, not held anywhere else. Who may
    approve is the APPLICATION's decision, not this package's: the session is
    already scoped to a participant, so nobody can answer another participant's
    approval through it, but "this user may approve THIS action" is a question
    only the host can answer. Authorize before calling.

    Nothing runs until the next resume, ``send(session, "")``, which acts on
    every decision recorded by then and refuses any pending call still without
    one. With several pending, record them all first. A new message instead,
    or any row that supersedes the paused step (a later step with tool calls,
    or an assistant or user row recorded outside the runs that own the
    step), denies
    its open calls, "Tool call was not run because a later step replaced it", whatever was
    decided.

    ``approval_id`` is :attr:`PendingApproval.id`, not the tool call id.

    It must name an approval the session is waiting on (G-354). An id the run
    never issued, or one already answered, raises ``approval_not_pending``
    before anything is written.
    """
    _record_decision(
        session,
        approval_id,
        approved,
        reason,
        _RESOLUTION_LOCK_SECONDS,
        _RESOLUTION_WAIT_SECONDS,
    )


def _record_decision(
    session: Session,
    approval_id: str,
    approved: bool,
    reason: str | None,
    ttl_seconds: float,
    wait_seconds: float,
) -> None:
    """The pending check and the write, together under the session lock.

    Apart, two decisions for one id (a double click, two reviewers, a retry
    racing the original) would both pass the check and both be written. The
    second now waits, finds the id answered, and is refused.
    """

    def write(live: Session) -> None:
        rows = live.thread().rows()
        # ANY DENY IS FINAL (G-401, the recommended rule): a deny for an
        # approval already approved but not yet acted on is recorded and
        # revokes it; anything else for an answered id is refused.
        revoking = (
            not approved
            and approval_id in decided_unresolved_ids(rows)
            and first_decisions(rows).get(approval_id, {}).get("approved") is True
        )
        if approval_id not in pending_approval_ids(rows) and not revoking:
            raise HarnessError.approval_not_pending(approval_id)
        # A decision the resuming turn could not act on at all is refused
        # here, with nothing written (G-387): a linked child with no saved
        # binding, or one too deep to run.
        assert_decidable(live, rows)
        # No budget refuses a decision (G-387): acting on an approved call
        # makes no provider call, so a resume with no steps or cost left acts
        # on it and takes no model step. Only what the call itself cannot
        # pass (its binding, depth, a call-level limit) is refused, when the
        # resume runs it.

        run = live.run()
        # A step whose call ids repeat cannot say which call a decision
        # answers: every decision on it is recorded as a DENIAL with the
        # G-392 text (G-387 round 13), as the PHP reference records it, so no
        # reader (a plan, a later resume) ever reads one as approved.
        tail = paused_approval_row(rows)
        repeated = tail is not None and repeats_call_ids(tail)

        live.thread().record(
            [
                tool_result_row(
                    [],
                    [
                        {
                            "approval_id": approval_id,
                            "approved": False if repeated else approved,
                            "reason": REPEATED_CALL_ID if repeated else reason,
                        }
                    ],
                )
            ],
            run["id"] if run else None,
        )

    # From inside a call the resolver is running on this very session (a
    # tool recording a decision on its own session): the lock is the one the
    # resolver holds, so wait only as long as the agent is CONFIGURED to, and
    # refuse with a fixed text that names no key (G-387 round 16).
    own = (_RESOLVING.get() or {}).get(session.key())
    if own is not None:
        ttl_seconds, wait_seconds = own
    # A decision is never a runner start's own run: it reads the child's
    # records by the provisional rule, even from inside one (G-387 round 11).
    with outside_any_start():
        try:
            session.lock(write, ttl_seconds, wait_seconds)
        except HarnessError as error:
            if own is None or error.code != "session_locked":
                raise
            raise HarnessError(error.code, _OWN_SESSION_LOCKED) from None


def assert_decidable(session: Session, rows: Sequence[Mapping[str, Any]]) -> None:
    """The checks a resume with no context runs before it acts on anything,
    asked before a decision is recorded (G-387).

    :raises HarnessError: ``child_binding_missing`` or ``run_not_permitted``.
    """
    found = _unanswerable(session, rows)
    if found == "child_binding_missing":
        raise HarnessError.child_binding_missing()
    if found == "too_deep":
        raise HarnessError.run_not_permitted(
            "This decision cannot be acted on: its run is nested at or past the depth ceiling."
        )


def _unanswerable(session: Session, rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Why a decision on the session's pending approval would be refused
    (``child_binding_missing`` or ``too_deep``), or None when it would be
    accepted. A read that raises propagates: it is never an answer."""
    saved = ChildBinding.from_row(session.child_binding())
    if saved is None and not session.child_provisional():
        tail = paused_approval_row(rows)
        saved = ChildBinding.from_row(tail.get("harness_binding")) if tail else None
    route = binding_route("none", False, saved is not None, session.child_refs() is not None)
    if route["binding"] == "refused":
        return "child_binding_missing"
    if session.child_depth() >= MAX_DEPTH:
        return "too_deep"
    return None


def _digestible(arguments: Mapping[str, Any] | None) -> bool:
    """Whether a call's arguments can be digested for its decision events."""
    if arguments is None:
        return False
    try:
        args_digest(dict(arguments))
    except (TypeError, ValueError, RecursionError, UnicodeError):
        return False
    return True


def _call_input(call: LlmToolCall) -> ToolCallInput:
    return ToolCallInput(
        id=call.id,
        name=call.name,
        arguments=usable_arguments(call.arguments) or {},
        result_id=call.result_id,
        reasoning_id=call.reasoning_id,
        reasoning_summary=call.reasoning_summary,
    )


def _validate_runtime_config(config: Mapping[str, object]) -> None:
    for key in ("provider", "model"):
        value = config.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("Agent provider/model defaults must be nonempty strings or null.")
    ceiling = config.get("spend_ceiling")
    if ceiling is not None and (
        isinstance(ceiling, bool)
        or not isinstance(ceiling, (int, float))
        or not math.isfinite(ceiling)
        or ceiling <= 0
    ):
        raise ValueError("Agent spend_ceiling must be a positive finite number of USD or null.")
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


def _first(values: Sequence[str | None]) -> str:
    """The first configured value; empty strings are refused at config time."""
    for value in values:
        if value:
            return value
    return "unknown"


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
