"""Agent plans (G-360).

A plan is an ordered list of steps an agent proposes, a person approves, and a
driver (``runtime.plan(session).advance()``) works through one step per turn.
It grants nothing: every tool call still goes through the mode's tools,
approvals, the tool policy and the run budget.

Off unless the host turns it on (``AgentRuntime(plans={"enabled": True})``) AND
the session's mode sets ``plans``.
"""

from __future__ import annotations

from prism_harness.events import PlanUpdated
from prism_harness.plans import machine, text, view
from prism_harness.plans.handle import AdvanceOutcome, PlanHandle, PlanSnapshot
from prism_harness.plans.ids import PlanIds, is_step_id, ulid
from prism_harness.plans.limits import EVENT_BYTES_CEILING, STEPS_CEILING, PlanLimits
from prism_harness.plans.refusal import (
    PlanRefused,
    PlansDisabled,
    refusal_for_model,
    refusal_message,
)
from prism_harness.plans.service import Plans
from prism_harness.plans.tools import PlanTool

__all__ = [
    "EVENT_BYTES_CEILING",
    "STEPS_CEILING",
    "AdvanceOutcome",
    "PlanHandle",
    "PlanIds",
    "PlanLimits",
    "PlanRefused",
    "PlanSnapshot",
    "PlanTool",
    "PlanUpdated",
    "Plans",
    "PlansDisabled",
    "is_step_id",
    "machine",
    "refusal_for_model",
    "refusal_message",
    "text",
    "ulid",
    "view",
]
