"""Turns a declared Subagent into a tool the parent run can call."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from prism_harness.harness import PrismHarness
from prism_harness.runtime import AgentRuntime
from prism_harness.session import Session
from prism_harness.subagents import MAX_DEPTH, RunContext, Subagent
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
        task = args.get("task")

        return self._runner.run(
            self._subagent,
            self._parent,
            self._context,
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

        child_context = context.for_child(subagent, parent_run_id)

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

        child = self._harness.for_(parent.participant).session(subagent.scope_under(parent.scope))

        # The child's authority comes from ITS mode, not the parent's.
        child.using_mode(subagent.mode)
        child.using_provider(parent.provider() or "unknown")
        child.using_model(parent.model() or "unknown")

        try:
            response = self._runtime.send(child, task, context=child_context)
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
