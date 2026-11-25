"""The loop: prompt in, turns out, everything recorded."""

from __future__ import annotations

import json
import logging
import math
import time
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
from prism_harness.modes import AgentMode, ModeRegistry
from prism_harness.plans.service import Plans, refuse_unfunded_decision
from prism_harness.run_trees import RunTrees, kept_refs, refuse_spent_tree, saved_refs
from prism_harness.session import Session
from prism_harness.structured import schema_name, schema_problems
from prism_harness.subagents import ChildBinding, RunBudget, RunContext, binding_route
from prism_harness.thread_rows import (
    ARGUMENTS_AS_SENT,
    ToolCallInput,
    ambiguous_calls,
    approval_run_steps,
    approval_tool_scope,
    assistant_row,
    carried_agent_spend,
    first_decisions,
    paused_approval_row,
    pending_approval_ids,
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

        Nothing runs until the next ``send()``. Who may decide is the
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
            self._plans.now,
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
    ) -> RunContext:
        """Internal: the subagent runner marks the context it starts a child
        under as its own, so only that context saves the child's binding, with
        the run and session of the ACTIVE turn starting it (None outside one):
        the child joins that run's tree once the run is verified live."""
        return self._trees.starting_child(context, parent_run_id, parent_session_key)

    def mode_for(self, session: Session) -> AgentMode:
        """The mode a session's next run resolves, refusing a malformed one."""
        return self._modes.for_agent(session.agent).resolve(session.mode())

    def _bounded_run(self, session: Session, run: RunContext) -> RunContext:
        run = self._labelled(session, run)
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
        # Refused before a run exists: a bad attachment is a mistake in the call,
        # and it should not cost a run, events or budget.
        attachments = admit_attachments(prompt, additional_content)
        mode = self._modes.for_agent(session.agent).resolve(session.mode())
        run_id = str(uuid.uuid4())
        base = context or RunContext.root(run_id, RunBudget(mode.max_steps))
        # A resume with no context continues the run that paused (G-321 E):
        # a subagent child keeps the provider and model binding the runner gave
        # it, and a named agent keeps what it had spent against its ceiling.
        # Both are read from the paused row, which the harness wrote.
        paused = (
            paused_approval_row([entry.message for entry in session.thread().messages()])
            if prompt == "" and context is None
            else None
        )
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
        kind = self._trees.context_kind(context)
        saved_binding = ChildBinding.from_row(session.child_binding())
        if saved_binding is None:
            tail = paused_approval_row([entry.message for entry in session.thread().messages()])
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

        if run.too_deep():
            raise HarnessError.run_not_permitted(
                f"This run is nested {run.depth} deep, at or past the ceiling. Nobody debugs a "
                "tree that deep, and a configuration that produced one is a mistake worth "
                "reporting rather than executing."
            )

        session.begin_run(
            run_id, mode.name, provider, model, tree=member["tree"], depth=member["depth"]
        )
        # Recorded before any tool can run, so a child started from this turn
        # finds it; dropped when the run finishes.
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

        turn_token = None
        try:
            # The plan context (G-360), computed once per run and never stored:
            # a system prompt contribution, not a message. A turn with a new
            # user message reads the plan as if nothing were pending.
            plan_context = self._plans.context(session, prompt)
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
            stored = [entry.message for entry in thread.messages()]
            scope = approval_tool_scope(stored)
            # A resume continues the paused run's step count; it never opens a
            # new allowance. Only for a root run: a child's steps are already on
            # the tree ledger its caller hands back in.
            if prompt == "" and context is None:
                run.ledger.record_steps(approval_run_steps(stored) or 0)
            # G-366: one of the plan's own turns, with a shared budget, holds
            # what the plan has left, after the paused steps it continues are
            # counted. Any other turn is unchanged.
            if context is not None and runner_binding is not None:
                # The runner is starting a child: save its binding and the run
                # tree and plan budget it belongs to, so every later turn on
                # the child the runner did not start continues them. Only the
                # runner's own context writes them.
                session.remember_child_binding(runner_binding.to_row(), member["depth"])
                self._trees.child_started(session, member, self.lock_ttl(session))
                # The tree it is in, with the plan ledger of the run it spends
                # from. Under a run tree the host built, or none, there is no
                # durable tree to name: the references the child saved before
                # are kept, never replaced with nothing, so a later turn the
                # runner does not start still continues the tree it last had
                # rather than a fresh root budget.
                saved_tree, saved_plan_ledger = saved_refs(session)
                spends = member["spends"]
                refs = kept_refs(
                    member,
                    self._plans.ledger_of(spends) if isinstance(spends, str) else None,
                    saved_tree,
                    saved_plan_ledger,
                    False,
                )
                session.remember_child_refs(refs["tree"], refs["plan_ledger"])
            # A root run opens a tree; any turn on a child session the runner
            # did not start (an answer, a new message, or a host context,
            # narrowed to what is left) continues the run tree it was saved
            # with. The runner's own context is left alone.
            run = self._trees.root_run(session, run_id, run, member, self.lock_ttl(session))
            run = self._plans.plan_turn(
                session, run_id, run, context, self.lock_ttl(session), prompt, kind
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

            # Decisions recorded since the run stopped are acted on FIRST, before
            # a new prompt is recorded, so the results land after the calls they
            # answer rather than after the new turn.
            offered = self._resolve_approvals(
                session, mode, offered, run_id, called, scope is not None and scope[1]
            )

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
                    harness_agent=session.agent,
                )
            )
            self._plans.after_turn(session)
            self._trees.finished(session, run_id)
            raise
        finally:
            if turn_token is not None:
                _TURN_RUN.reset(turn_token)

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
    ) -> AgentResponse:
        thread = session.thread()
        gates = plan_gates or {}
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

            # This agent's own ceiling, in the same ledger. Raised rather than
            # finished: a subagent child refused here comes back to its parent
            # as a framed result, and only this agent stops.
            refused = run.agent_stop()
            if refused is not None:
                raise refused

            request = LlmRequest(
                system_prompt=mode.system_prompt if system_prompt is None else system_prompt,
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
            thread.record([row], run_id)

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

            for call in gated:
                self._tool_decision(
                    ToolApprovalAsked, session, run_id, call, _offered(offered, call.name), "person"
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
                thread.record([tool_result_row(results)], run_id)

            if requests:
                # FAILS CLOSED. The gated calls have not run, and the requests are
                # in the thread, so a different process can resume after a person
                # answers.
                self._plans.after_turn(session)
                self._trees.finished(session, run_id)
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
        legacy_diagnostic_emitted: bool,
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
        if self._unresolved(session, mode) is None:
            return list(offered)

        # Then under the session lock, reading again inside it. Two workers
        # resuming the same session at once would otherwise both find the
        # approved call with no result, and both run it. The second now waits,
        # finds the result the first recorded, and runs nothing. If it cannot get
        # the lock it raises, and nothing runs.
        def resolve(live: Session) -> list[HarnessTool]:
            work = self._unresolved(live, mode)

            if work is None:
                return list(offered)

            calls, results, decisions, approval_ids, scope, ambiguous = work
            # Bind calls, decisions and scope from the same locked snapshot.
            # A concurrent narrower approval also bounds the continuation.
            if scope is not None and scope[1] and not legacy_diagnostic_emitted:
                logging.getLogger(__name__).warning(
                    "Pending approval has no valid stored tool scope; offering approved requests only."
                )
            bounded = [tool for tool in offered if scope is not None and tool.name in scope[0]]
            resolved: list[dict[str, Any]] = []
            unusable = unusable_stored_calls(
                [entry.message for entry in live.thread().messages()]
            ) | {row["id"] for row in calls if not _digestible(row["arguments"])}

            for row in calls:
                call = ToolCallInput(
                    id=row["id"],
                    name=row["name"],
                    arguments=row["arguments"],
                    result_id=row["result_id"],
                )
                approval_id = approval_ids.get(row["id"])
                decision = decisions.get(approval_id) if approval_id is not None else None

                declared = _offered(bounded, call.name)
                if decision is not None and decision["approved"] and call.id in unusable:
                    resolved.append(
                        tool_result_entry(
                            call, f"Not run: the arguments for {call.name} are not a JSON object."
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
                    resolved.append(tool_result_entry(cleared, _approval_expired(call.name)))
                elif decision is not None and decision["approved"]:
                    self._tool_decision(ToolApproved, live, run_id, call, declared, "person")
                    called.append(call.name)
                    resolved.append(self._run_approved(bounded, call))
                elif decision is None:
                    self._tool_decision(ToolDenied, live, run_id, call, declared, "person")
                    resolved.append(tool_result_entry(call, "No approval response provided"))
                else:
                    self._tool_decision(ToolDenied, live, run_id, call, declared, "person")
                    resolved.append(
                        tool_result_entry(call, decision["reason"] or "User denied tool execution")
                    )

            live.thread().record(
                [tool_result_row([*results.values(), *resolved], list(decisions.values()))],
                run_id,
            )
            return bounded

        options = self._settings(session)
        return session.lock(
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
            tuple[list[str], bool] | None,
            set[str],
        ]
        | None
    ):
        """The gated calls of the last tool-calling turn that have no result yet."""
        rows = [entry.message for entry in session.thread().messages()]
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
        answered = next((row for row in view[index + 1 :] if row["type"] == "tool_result"), None)
        results = {e["tool_call_id"]: e for e in (answered or {}).get("tool_results", [])}
        # Which decision counts comes from the stored rows, the FIRST recorded
        # for an approval id standing; the view folds later over earlier.
        decisions = first_decisions(rows)
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

        if not calls:
            return None

        return (
            calls,
            results,
            decisions,
            approval_ids,
            approval_tool_scope(rows),
            ambiguous_calls(assistant),
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
        # G-360: when the run is over, the plan reads the thread it left.
        self._plans.after_turn(session)
        # A root run that holds part of a run tree's budget settles it.
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
        )

    def _emit(self, event: HarnessEvent) -> None:
        if self._events is not None:
            self._events.emit(event)


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

    Nothing runs until the next ``send()``, which acts on every decision recorded
    by then and refuses any pending call still without one. With several
    pending, record them all first.

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
    clock: Callable[[], int] | None = None,
) -> None:
    """The pending check and the write, together under the session lock.

    Apart, two decisions for one id (a double click, two reviewers, a retry
    racing the original) would both pass the check and both be written. The
    second now waits, finds the id answered, and is refused.
    """

    def write(live: Session) -> None:
        rows = [entry.message for entry in live.thread().messages()]
        if approval_id not in pending_approval_ids(rows):
            raise HarnessError.approval_not_pending(approval_id)
        # A decision a spent shared plan budget could not run is refused here,
        # with nothing recorded, rather than consumed by a resume that is then
        # refused (G-366).
        now = clock() if clock is not None else int(time.time())
        refuse_unfunded_decision(live, approval_id, now, int(ttl_seconds))
        # The same for a subagent child whose run tree has nothing left.
        refuse_spent_tree(live, int(time.time()), int(ttl_seconds))

        run = live.run()

        live.thread().record(
            [
                tool_result_row(
                    [], [{"approval_id": approval_id, "approved": approved, "reason": reason}]
                )
            ],
            run["id"] if run else None,
        )

    session.lock(write, ttl_seconds, wait_seconds)


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
