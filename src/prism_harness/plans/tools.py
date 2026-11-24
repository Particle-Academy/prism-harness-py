"""The agent's plan tools, built per run for one session (G-360).

NO TOOL TAKES A STEP ID OR AN APPROVAL ID. Each acts on the plan's head step,
which the driver chose, so a model that cannot name a step cannot act on someone
else's, nor probe which ids exist. plan_propose_change is the one exception, and
only to keep existing steps unchanged by id.

WHAT THE PERSON APPROVED IS WHAT IS APPLIED. plan_draft, plan_propose_change and
plan_pause_for_person (kind approve) gate THEMSELVES: the runtime asks
:meth:`PlanTool.requires_approval` instead of the mode's list or the host tool
policy. The proposal is recorded when that is asked; it is applied only when the
runtime runs the call a person approved, and only when that call's stored
arguments are the ones proposed. A call that reaches its handler without a
recorded approval for exactly these arguments applies nothing.

Every refusal handed back to the model is a constant sentence and a code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prism_harness.plans import machine
from prism_harness.plans.refusal import PlanRefused, refusal_for_model
from prism_harness.plans.thread_facts import approved_calls_awaiting_result
from prism_harness.tool_policy import args_digest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_harness.plans.service import Plans
    from prism_harness.session import Session

__all__ = ["PlanTool", "plan_tools", "step_schema"]

Args = dict[str, Any]


class PlanTool:
    """A plan tool. ``requires_approval`` is asked by the runtime, per call,
    with the call's arguments, in place of the mode's approval list."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handle: Callable[[Args], str],
        requires_approval: Callable[[Args, bool], bool] | None = None,
    ) -> None:
        self._name = name
        self.description = description
        self.parameters = parameters
        self._handle = handle
        self._requires_approval = requires_approval

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> str:
        return self._handle(args)

    def requires_approval(self, args: dict[str, Any], authorized: bool = True) -> bool:
        """``authorized`` is the host call policy's verdict on this call, asked
        by the runtime first, so a call the host forbids records nothing."""
        return (
            self._requires_approval(args, authorized)
            if self._requires_approval is not None
            else False
        )


def step_schema(with_id: bool) -> dict[str, Any]:
    """JSON Schema for one step of plan_draft (``with_id`` False) or
    plan_propose_change (True)."""
    nullable_string = {"type": ["string", "null"]}
    properties: dict[str, Any] = {
        "title": {
            "type": ["string", "null"] if with_id else "string",
            "description": "A short title for the step, at most 160 characters, one line.",
        },
        "instruction": {
            **nullable_string,
            "description": "What to do in this step, at most 600 characters, one line. "
            "Required for your own steps.",
        },
        "owner": {
            "type": ["string", "null"] if with_id else "string",
            "enum": ["agent", "human", None] if with_id else ["agent", "human"],
            "description": "agent for your own step; human for a question to the person.",
        },
        "interaction": {
            "type": ["string", "null"],
            "enum": ["input", None],
            "description": "input for a question to the person; omit for your own steps.",
        },
        "question": {
            **nullable_string,
            "description": "The question for the person, for a human step.",
        },
        "options": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Suggested answers for a human step; free text is always allowed.",
        },
    }
    if with_id:
        properties = {
            "id": {
                **nullable_string,
                "description": "To keep an existing step unchanged, its id and nothing else.",
            },
            **properties,
        }
    return {
        "type": "object",
        "description": "One step.",
        "properties": properties,
        "required": [] if with_id else ["title", "owner"],
    }


def plan_tools(plans: Plans, session: Session) -> list[PlanTool]:
    return _PlanTools(plans, session).all()


class _PlanTools:
    def __init__(self, plans: Plans, session: Session) -> None:
        self._plans = plans
        self._session = session
        #: Refusal codes found while deciding approval, by tool and digest.
        self._refusals: dict[str, str] = {}

    def all(self) -> list[PlanTool]:
        return [self._draft(), self._complete(), self._pause(), self._change()]

    # -- plan_draft --------------------------------------------------------

    def _draft(self) -> PlanTool:
        def requires_approval(arguments: Args, authorized: bool) -> bool:
            if not self._plans.limits().require_approval:
                return False
            digest = args_digest(arguments)
            try:
                plan = self._proposal(arguments, digest, authorized)
                self._plans.write(self._session, plan, self._revision())
            except PlanRefused as refused:
                self._refusals[machine.DRAFT + digest] = refused.code
                return False
            return True

        def handle(arguments: Args) -> str:
            digest = args_digest(arguments)
            try:
                if self._approved(machine.DRAFT, digest):
                    plan = self._plans.mutate(
                        self._session,
                        lambda current: machine.activate(
                            current, digest, self._plans.now(), self._plans.run_id(self._session)
                        ),
                    )
                    if plan is None:
                        return refusal_for_model("plan.not_waiting")
                    return "The person approved the plan. It is active; the host will run its steps one at a time."
                refused = self._refusals.get(machine.DRAFT + digest)
                if refused is not None:
                    return refusal_for_model(refused)
                if self._plans.limits().require_approval:
                    return refusal_for_model("plan.not_waiting")
                # Approval switched off by the host's config: the plan starts as written.
                plan = self._proposal(arguments, digest)
                plan = machine.activate(
                    plan, digest, self._plans.now(), self._plans.run_id(self._session)
                )
                self._plans.write(self._session, plan, self._revision())
                return "The plan is active; the host will run its steps one at a time."
            except PlanRefused as refused_now:
                return refusal_for_model(refused_now.code)

        return PlanTool(
            machine.DRAFT,
            "Propose a plan: a goal and an ordered list of steps. A person approves it before "
            "any step starts. Steps are your own work (owner agent, with an instruction) or a "
            "question for the person (owner human, interaction input, with a question).",
            {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "What the plan is for, at most 200 characters, one line.",
                    },
                    "steps": {
                        "type": "array",
                        "description": "The steps, in order.",
                        "items": step_schema(False),
                    },
                },
                "required": ["goal", "steps"],
            },
            handle,
            requires_approval,
        )

    def _proposal(self, arguments: Args, digest: str, authorized: bool = True) -> dict[str, Any]:
        limits = self._plans.limits()
        draft = machine.check_draft(arguments, limits, authorized)
        if machine.is_live(self._plans.load(self._session)):
            raise PlanRefused("plan.exists")
        return machine.propose(
            draft,
            digest,
            self._plans.ids(),
            self._plans.now(),
            self._plans.run_id(self._session),
            limits,
        )

    def _revision(self) -> int | None:
        current = self._plans.load(self._session)
        revision = current.get("revision") if current is not None else None
        return revision if isinstance(revision, int) else None

    # -- plan_complete_step -----------------------------------------------

    def _complete(self) -> PlanTool:
        def handle(arguments: Args) -> str:
            if list(arguments) != ["summary"]:
                return refusal_for_model("plan.shape")
            try:
                plan = self._plans.mutate(
                    self._session,
                    lambda current: machine.report(
                        current,
                        arguments["summary"],
                        self._plans.now(),
                        self._plans.run_id(self._session),
                    ),
                )
                if plan is None:
                    return refusal_for_model("plan.no_running_step")
            except PlanRefused as refused:
                return refusal_for_model(refused.code)
            return "Noted. The step will be settled after this turn. Stop work on it now."

        return PlanTool(
            machine.COMPLETE,
            "Report that the CURRENT plan step is finished. It does not start the next step; "
            "stop work after calling it.",
            {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "What you did, at most 200 characters, one line.",
                    }
                },
                "required": ["summary"],
            },
            handle,
        )

    # -- plan_pause_for_person --------------------------------------------

    def _pause(self) -> PlanTool:
        def requires_approval(arguments: Args, authorized: bool) -> bool:
            if arguments.get("kind") != "approve":
                return False
            digest = args_digest(arguments)
            code = self._apply_pause(arguments, digest, authorized)
            if code is not None:
                self._refusals[machine.PAUSE + digest] = code
                return False
            return True

        def handle(arguments: Args) -> str:
            digest = args_digest(arguments)
            if arguments.get("kind") == "approve":
                if self._approved(machine.PAUSE, digest):
                    return "The person approved. Continue the current step."
                return refusal_for_model(
                    self._refusals.get(machine.PAUSE + digest, "plan.no_running_step")
                )
            code = self._apply_pause(arguments, None)
            if code is not None:
                return refusal_for_model(code)
            return "The question is with the person. End your turn now; the plan continues after they answer."

        return PlanTool(
            machine.PAUSE,
            "Stop for the person before continuing the CURRENT step. kind approve asks the "
            "person to approve going on; kind input asks a question. End your turn after "
            "calling it.",
            {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["approve", "input"],
                        "description": "approve or input",
                    },
                    "question": {
                        "type": "string",
                        "description": "What to ask the person, at most 160 characters, one line.",
                    },
                    "options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Suggested answers for kind input; free text is always allowed.",
                    },
                },
                "required": ["kind", "question"],
            },
            handle,
            requires_approval,
        )

    def _apply_pause(
        self, arguments: Args, digest: str | None, authorized: bool = True
    ) -> str | None:
        code: list[str | None] = [None]

        def change(current: dict[str, Any]) -> dict[str, Any]:
            following, code[0] = machine.pause(
                current,
                arguments,
                digest,
                self._plans.ids(),
                self._plans.limits(),
                self._plans.now(),
                self._plans.run_id(self._session),
                authorized,
            )
            return following

        try:
            plan = self._plans.mutate(self._session, change)
        except PlanRefused as refused:
            return refused.code
        return "plan.no_running_step" if plan is None else code[0]

    # -- plan_propose_change ----------------------------------------------

    def _change(self) -> PlanTool:
        def requires_approval(arguments: Args, authorized: bool) -> bool:
            digest = args_digest(arguments)
            try:
                plan = self._plans.mutate(
                    self._session,
                    lambda current: machine.propose_change(
                        current,
                        arguments,
                        digest,
                        self._plans.ids(),
                        self._plans.limits(),
                        self._plans.now(),
                        self._plans.run_id(self._session),
                        authorized,
                    ),
                )
                if plan is None:
                    raise PlanRefused("plan.no_plan")
            except PlanRefused as refused:
                self._refusals[machine.CHANGE + digest] = refused.code
                return False
            return True

        def handle(arguments: Args) -> str:
            digest = args_digest(arguments)
            if not self._approved(machine.CHANGE, digest):
                return refusal_for_model(
                    self._refusals.get(machine.CHANGE + digest, "plan.not_waiting")
                )
            try:
                self._plans.mutate(
                    self._session,
                    lambda current: machine.apply_change(
                        current, digest, self._plans.now(), self._plans.run_id(self._session)
                    ),
                )
            except PlanRefused as refused:
                return refusal_for_model(refused.code)
            return "The person approved the change. The plan now follows the new steps."

        return PlanTool(
            machine.CHANGE,
            "Propose a new list for the plan's remaining steps. Keep every finished, running, "
            "held or person step unchanged by giving only its id. A person approves the change "
            "before it applies.",
            {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the plan should change, at most 200 characters, one line.",
                    },
                    "base_revision": {
                        "type": "number",
                        "description": "The plan revision you are changing, from the plan context.",
                    },
                    "steps": {
                        "type": "array",
                        "description": "The new list of steps, in order.",
                        "items": step_schema(True),
                    },
                },
                "required": ["reason", "base_revision", "steps"],
            },
            handle,
            requires_approval,
        )

    def _approved(self, tool: str, digest: str) -> bool:
        """A person approved exactly this call, and it has not run yet. Read from the thread."""
        rows = [entry.message for entry in self._session.thread().messages()]
        return approved_calls_awaiting_result(rows, tool, digest) > 0
