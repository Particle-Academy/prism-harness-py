"""Turns a declared Subagent into a tool the parent run can call."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from typing import Any

from prism_harness.errors import HarnessError
from prism_harness.harness import PrismHarness
from prism_harness.modes import AgentMode
from prism_harness.runtime import AgentRuntime, current_turn_run
from prism_harness.session import Session
from prism_harness.subagents import MAX_DEPTH, ChildBinding, RunBudget, RunContext, Subagent
from prism_harness.tools import HarnessTool

__all__ = ["SubagentResult", "SubagentRunner", "SubagentTool"]


@dataclass(frozen=True)
class SubagentResult:
    subagent: str
    run_id: str
    parent_run_id: str
    #: completed / exhausted / cancelled / denied / failed.
    outcome: str
    text: str

    def to_tool_result(self) -> str:
        """What to hand back to the model as the tool's result."""
        return self.text


class SubagentTool:
    """A subagent, dressed as a tool the parent can call."""

    def __init__(
        self,
        runner: SubagentRunner,
        subagent: Subagent,
        parent: Session,
        context: RunContext,
        parent_run_id: str,
    ) -> None:
        self._runner = runner
        self._subagent = subagent
        self._parent = parent
        self._context = context
        self._parent_run_id = parent_run_id

    @property
    def name(self) -> str:
        return self._subagent.name

    def handle(self, args: dict[str, Any]) -> str:
        if self._parent.agent is not None and any(key != "task" for key in args):
            raise HarnessError.run_not_permitted(
                "Named subagent input may contain only task. Identity changes require an explicit host-owned delegation binding."
            )
        task = args.get("task")

        return self._runner.run(
            self._subagent,
            self._parent,
            within_turn(self._context, current_turn_run()),
            self._parent_run_id,
            task if isinstance(task, str) else "",
        ).to_tool_result()


class SubagentRunner:
    """The parent is mid-run and holding its own session lock when this executes.
    Everything here is arranged so that fact stays harmless:

    - the child resolves its OWN session address, so it takes a different lock;
    - the child's authority comes from its declared mode, never the parent's;
    - the child's budget is drawn from the tree's remaining allowance;
    - EVERY way the child can end returns a framed result rather than raising,
      because the parent is a legitimate audience for "that did not work" and
      tearing down the parent run would discard work it had already done.
    """

    def __init__(self, harness: PrismHarness, runtime: AgentRuntime) -> None:
        self._harness = harness
        self._runtime = runtime

    def tool(
        self, subagent: Subagent, parent: Session, context: RunContext, parent_run_id: str
    ) -> HarnessTool:
        return SubagentTool(self, subagent, parent, context, parent_run_id)

    def run(
        self,
        subagent: Subagent,
        parent: Session,
        context: RunContext,
        parent_run_id: str,
        task: str,
    ) -> SubagentResult:
        run_id = f"run_{secrets.token_hex(6)}"

        # Checked BEFORE spawning, so an exhausted tree does not pay for a
        # session, a thread write and a provider call to discover it is
        # exhausted.
        stop = context.ledger.exhaustion(context.budget)

        if stop is not None:
            return _refused(
                subagent.name,
                run_id,
                parent_run_id,
                "cancelled" if context.ledger.cancelled else "exhausted",
                stop,
            )

        child_agent = subagent.agent if subagent.agent is not None else parent.agent
        same_agent = child_agent == parent.agent
        parent_mode = self._parent_mode(parent)
        # The child's provider and model come from declarations, never from a
        # write onto the child's session: that slot belongs to the runner's
        # address, and a copied override there would outlive this call.
        child_context = context.for_child(subagent, parent_run_id).with_binding(
            ChildBinding(
                session_provider=parent.provider() if subagent.inherit_session_model else None,
                session_model=parent.model() if subagent.inherit_session_model else None,
                declared_provider=subagent.provider,
                declared_model=subagent.model,
                parent_mode_provider=parent_mode.provider if same_agent and parent_mode else None,
                parent_mode_model=parent_mode.model if same_agent and parent_mode else None,
            )
        )

        # Refused BEFORE the child's address is built. Two modes naming each
        # other as subagents form a cycle budgets would eventually stop -- but
        # only after each level had appended `::sub::<name>` to an address that
        # may truncate rather than error, and two children truncated to the same
        # string are one conversation.
        if child_context.too_deep():
            return _refused(
                subagent.name,
                run_id,
                parent_run_id,
                "denied",
                f"subagent nesting reached the maximum depth of {MAX_DEPTH}",
            )

        # The agent was checked against the parent mode's `agents` when that mode
        # resolved: a child only ever runs as an agent its parent declared.
        child = self._harness._pending(parent.participant, child_agent, parent.actor).session(
            subagent.scope_under(parent.scope)
        )

        # The child's authority comes from ITS mode, not the parent's.
        child.using_mode(subagent.mode)

        try:
            response = self._runtime.send(child, task, context=child_context)
        except HarnessError as error:
            # Its agent's ceiling is spent, or cannot be enforced: a gate on the
            # tree, not the child's failure, and only that agent stops.
            if error.code in ("agent_spend_ceiling_exceeded", "budget_unpriced"):
                return _refused(subagent.name, run_id, parent_run_id, "exhausted", str(error))
            return _refused(subagent.name, run_id, parent_run_id, "failed", str(error))
        except Exception as error:  # noqa: BLE001 - framed, not propagated
            # Tearing down the parent run would discard work it has already
            # done, and "that subagent failed" is something the parent can act
            # on.
            return _refused(subagent.name, run_id, parent_run_id, "failed", str(error))

        if response.stopped_because is None:
            return SubagentResult(
                subagent=subagent.name,
                run_id=response.run_id,
                parent_run_id=parent_run_id,
                outcome="completed",
                text=response.text,
            )

        return SubagentResult(
            subagent=subagent.name,
            run_id=response.run_id,
            parent_run_id=parent_run_id,
            outcome="exhausted",
            text=(
                f"The [{subagent.name}] subagent stopped early: {response.stopped_because}. "
                f"Partial answer: {response.text}"
            ),
        )

    def _parent_mode(self, parent: Session) -> AgentMode | None:
        """The parent's mode, for the binding its child falls back to.

        Inside a run the parent's mode has already resolved. A runner driven
        directly, for a parent whose mode is not configured, has no parent mode
        to fall back to -- and that is all this tolerates: a MALFORMED mode
        still raises.
        """
        try:
            return self._runtime.mode_for(parent)
        except HarnessError as error:
            if error.code == "mode_not_configured":
                return None
            raise


def within_turn(host: RunContext, turn: RunContext | None) -> RunContext:
    """The run context a subagent tool's child is started under (G-366).

    Inside a turn, the TURN's: its ledger, so the child's spend lands where the
    turn is measured, a shared plan budget included, and its budget narrowed by
    the context the host built the tool with. The host's context can only
    narrow: what it has left is counted from the turn's spend so far, and no
    limit is ever raised past the turn's. Outside a turn (the runner driven
    directly), the host's context, unchanged.
    """
    if turn is None or host is turn:
        return host
    own, spent, ledger = turn.budget, host.ledger, turn.ledger
    # What the host's context has left, by its own ledger (the turn's, when the
    # host handed the turn's context in).
    steps_left = host.budget.max_steps - spent.steps
    cost_left = spent.remaining_cost(host.budget)
    seconds_left = spent.remaining_seconds(host.budget)
    budget = RunBudget(
        max_steps=min(own.max_steps, ledger.steps + max(0, steps_left)),
        max_cost_usd=_lower(
            own.max_cost_usd, None if cost_left is None else ledger.cost_usd + cost_left
        ),
        max_seconds=_lower_int(
            own.max_seconds,
            None if seconds_left is None else int(ledger.elapsed_seconds()) + seconds_left,
        ),
        max_output_tokens_per_call=_lower_int(
            own.max_output_tokens_per_call, host.budget.max_output_tokens_per_call
        ),
    )
    return replace(turn, budget=budget, depth=max(turn.depth, host.depth))


def _lower(a: float | None, b: float | None) -> float | None:
    return b if a is None else a if b is None else min(a, b)


def _lower_int(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else min(a, b)


def _refused(
    subagent: str, run_id: str, parent_run_id: str, outcome: str, reason: str
) -> SubagentResult:
    return SubagentResult(
        subagent=subagent,
        run_id=run_id,
        parent_run_id=parent_run_id,
        outcome=outcome,
        text=f"The [{subagent}] subagent did not run: {reason}.",
    )
