"""Every refusal a plan can give, as a code and a constant sentence (G-360).

The same codes and sentences in PHP, TypeScript and Python, pinned by the
``harness-plans`` parity corpus. No sentence contains a character the model
chose: a refusal that echoed model text would hand the model a channel into
whatever logs or renders the refusal.
"""

from __future__ import annotations

from prism_harness.errors import HarnessError

__all__ = [
    "MESSAGES",
    "PlanRefused",
    "PlansDisabled",
    "refusal_for_model",
    "refusal_message",
]

MESSAGES: dict[str, str] = {
    "plan.shape": "Refused: the plan is not in the expected shape. Nothing was saved.",
    "plan.bounds": "Refused: the plan has too many or too few steps. Nothing was saved.",
    "plan.text": "Refused: a field is empty, too long, or contains characters that are not allowed. Nothing was saved.",
    "plan.size": "Refused: the plan is too large to send as one update. Nothing was saved.",
    "plan.exists": "Refused: this conversation already has an open plan. It is waiting on the person or still in progress; do not draft another.",
    "plan.no_plan": "Refused: there is no active plan.",
    "plan.no_running_step": "Refused: there is no step of yours running right now.",
    "plan.pause_limit": "Refused: this step has already paused for the person. The plan is now blocked until the person resumes it.",
    "plan.change_refused": "Refused: the change does not fit the current plan. Read the plan again and propose against its current revision.",
    "plan.change_limit": "Refused: this plan has reached its limit of changes.",
    "plan.not_waiting": "Refused: there is no proposal waiting for this decision.",
    "plan.not_answerable": "Refused: that step is not waiting for an answer in this plan.",
    "plan.answer_invalid": "Refused: the answer is empty, too long, or not one of the options.",
    "plan.no_actor": "Refused: a person must be named as the actor.",
    "plan.not_resumable": "Refused: this plan cannot be resumed.",
    "plan.not_live": "Refused: this plan is already finished.",
    "plan.not_authorized": "Refused: the host application does not allow this call. Nothing was saved.",
    "plan.conflict": "Refused: the plan changed while this update was being written.",
}


def refusal_message(code: str) -> str:
    return MESSAGES.get(code, "Refused.")


def refusal_for_model(code: str) -> str:
    """The text a plan tool hands back to the model for a refusal."""
    return f"{refusal_message(code)} [{code}]"


class PlanRefused(HarnessError):
    """A plan operation was refused.

    The message is a constant chosen by the code, never text from the model or
    from the plan, so it is safe to log and to show. Branch on ``code``.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code, refusal_message(code))


class PlansDisabled(HarnessError):
    """Plans are off for this session.

    They are off unless the host turns them on (``plans={"enabled": True}``)
    AND the session's mode sets ``plans``.
    """

    def __init__(self) -> None:
        super().__init__(
            "plan.disabled",
            "Plans are not enabled for this session. Turn on plans.enabled and the mode's "
            "plans flag.",
        )
