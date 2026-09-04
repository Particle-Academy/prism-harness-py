"""Every failure this package raises, each with a stable code."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

__all__ = ["ErrorCode", "HarnessError"]


class ErrorCode(str, Enum):
    """The stable identity of a failure.

    The PHP reference identifies a failure by its exception class and an English
    sentence. A class name does not survive a port and a sentence is not a
    contract, so the code is what a consumer branches on here: treat
    ``HarnessError.code`` as stable and the message as free to change in any
    release. Same decision, and the same reasoning, as ``prism-ai``.
    """

    #: A lock on a session key could not be acquired before the wait expired.
    SESSION_LOCKED = "session_locked"
    #: A store that reports itself volatile was configured for durable state.
    UNSAFE_STATE_CONFIGURATION = "unsafe_state_configuration"
    #: A store slot names a driver nothing is registered under.
    UNKNOWN_STORE_DRIVER = "unknown_store_driver"
    #: A tool was asked for that this session cannot reach.
    TOOL_NOT_AVAILABLE = "tool_not_available"
    #: A run was refused -- by budget, by depth, or by a cancelled ledger.
    RUN_NOT_PERMITTED = "run_not_permitted"
    #: A stored payload could not be mapped back into a value object.
    UNMAPPABLE_CONTENT = "unmappable_content"
    #: A session was asked to do something needing a runtime it does not have.
    NO_AGENT_RUNTIME = "no_agent_runtime"
    #: A mode was asked for that the application has not configured.
    MODE_NOT_CONFIGURED = "mode_not_configured"
    #: A configured mode is malformed, or names something that does not exist.
    MODE_MALFORMED = "mode_malformed"
    #: A tool policy is defined but the authorizer that would consult it is off.
    UNSAFE_AUTHORIZATION_CONFIGURATION = "unsafe_authorization_configuration"
    #: A tool call was refused by the call-time policy.
    CALL_NOT_AUTHORIZED = "call_not_authorized"
    #: A skill file was asked for that would resolve outside its own skill.
    SKILL_PATH_REFUSED = "skill_path_refused"
    #: A task was added under an id the source already holds.
    DUPLICATE_TASK_ID = "duplicate_task_id"
    #: A worker id or a task id was the empty string.
    TASK_IDENTIFIER_BLANK = "task_identifier_blank"
    #: A task was asked for that this source does not hold.
    TASK_NOT_FOUND = "task_not_found"
    #: A ``done`` or ``failed`` task was released again.
    TASK_ALREADY_TERMINAL = "task_already_terminal"
    #: A lease was acted on by someone who does not hold it.
    TASK_LEASE_NOT_HELD = "task_lease_not_held"
    #: A task outcome was supplied that is not exactly ``done`` or ``failed``.
    TASK_OUTCOME_INVALID = "task_outcome_invalid"
    #: A lease duration was not a finite number of seconds greater than zero.
    TASK_LEASE_INVALID = "task_lease_invalid"


class HarnessError(Exception):
    """``code`` is a plain string drawn from :class:`ErrorCode`, so a caller can
    compare against either the enum member or the literal.
    """

    def __init__(self, code: ErrorCode | str, message: str) -> None:
        super().__init__(message)
        self.code: str = code.value if isinstance(code, ErrorCode) else code
        self.message = message

    def __repr__(self) -> str:
        return f"HarnessError(code={self.code!r}, message={self.message!r})"

    # -- factories ---------------------------------------------------------

    @classmethod
    def session_locked(cls, key: str, wait_seconds: float) -> HarnessError:
        return cls(
            ErrorCode.SESSION_LOCKED,
            f"Could not acquire the lock on session [{key}] within {wait_seconds}s. "
            "Another worker is holding it; the callback was NOT run.",
        )

    @classmethod
    def volatile_durable_store(cls, slot: str, driver: str) -> HarnessError:
        """The guard this package exists for.

        A cache is disposable by definition, and the durable slot holds pending
        tool approvals -- a half-executed action waiting on a human. Losing one
        is a correctness failure, not a cache miss, so a store that reports
        itself volatile is refused here rather than accepted and discovered
        later.
        """
        return cls(
            ErrorCode.UNSAFE_STATE_CONFIGURATION,
            f"The [{slot}] slot is configured with the [{driver}] driver, which reports itself "
            "VOLATILE. Durable state (threads, pending tool approvals) must survive a deploy. "
            "Either point this slot at a durable driver, or -- if this store really does persist "
            "-- declare it durable when you register it. That declaration is an assertion about "
            "your infrastructure, not a preference.",
        )

    @classmethod
    def unknown_store_driver(cls, slot: str, driver: str) -> HarnessError:
        return cls(
            ErrorCode.UNKNOWN_STORE_DRIVER,
            f"The [{slot}] slot names the driver [{driver}], which is not registered.",
        )

    @classmethod
    def tool_not_available(cls, name: str, available: Sequence[str]) -> HarnessError:
        return cls(
            ErrorCode.TOOL_NOT_AVAILABLE,
            f"The tool [{name}] is not available to this session "
            f"(has: {', '.join(available) or 'none'}).",
        )

    @classmethod
    def run_not_permitted(cls, reason: str) -> HarnessError:
        return cls(ErrorCode.RUN_NOT_PERMITTED, reason)

    @classmethod
    def unmappable_content(cls, description: str) -> HarnessError:
        return cls(
            ErrorCode.UNMAPPABLE_CONTENT, f"Could not rebuild stored content: {description}."
        )

    @classmethod
    def mode_not_configured(cls, name: str) -> HarnessError:
        return cls(ErrorCode.MODE_NOT_CONFIGURED, f"Harness mode [{name}] is not configured.")

    @classmethod
    def mode_malformed(cls, name: str, detail: str) -> HarnessError:
        return cls(ErrorCode.MODE_MALFORMED, f"Harness mode [{name}] is malformed: {detail}.")

    @classmethod
    def policy_defined_but_disabled(cls) -> HarnessError:
        """Both at once is the one configuration not to leave in place.

        A defined policy that is never consulted looks like a control to every
        reader and is not one -- every registered tool is offered to every run
        while the code says otherwise.
        """
        return cls(
            ErrorCode.UNSAFE_AUTHORIZATION_CONFIGURATION,
            "A tool authorization policy was supplied, but the authorizer is disabled, so that "
            "policy is never consulted and every registered tool is offered to every run. Either "
            "enable the authorizer, or remove the policy so nothing suggests tool access is being "
            "restricted.",
        )

    @classmethod
    def call_not_authorized(cls, tool: str) -> HarnessError:
        """Raised rather than returned as a tool result.

        A refusal handed back as a result reads to the model as a failure it
        might retry differently, and a denied action being retried is the
        opposite of what a guard is for.
        """
        return cls(
            ErrorCode.CALL_NOT_AUTHORIZED,
            f"This call to [{tool}] was refused by the tool authorization policy.",
        )

    @classmethod
    def skill_path_refused(cls, detail: str) -> HarnessError:
        """A skill name or path that would leave the skill directory.

        Refused rather than sanitised. Silently rewriting a traversal to
        something safe teaches a caller -- or a model -- that the request was
        fine.
        """
        return cls(ErrorCode.SKILL_PATH_REFUSED, f"Refused to read a skill file: {detail}.")

    # -- task lists ---------------------------------------------------------

    @classmethod
    def volatile_task_source(cls, driver: str) -> HarnessError:
        """The same guard as :meth:`volatile_durable_store`, at the task list.

        Shares its CODE deliberately: it is one misconfiguration -- durable
        state pointed at a store that says it cannot keep it -- and a consumer
        branching on ``unsafe_state_configuration`` should catch both. Only the
        prose differs, and prose is outside the contract (0004).
        """
        return cls(
            ErrorCode.UNSAFE_STATE_CONFIGURATION,
            f"A task source was pointed at the [{driver}] store, which reports itself VOLATILE. "
            "A task list is durable state: a half-finished list that vanishes on a deploy is "
            "indistinguishable from a finished one, so losing it is a correctness failure rather "
            "than a cache miss. Point the source at a durable store -- or, if this one really "
            "does persist, declare it durable when you register it.",
        )

    @classmethod
    def duplicate_task_id(cls, task_id: str) -> HarnessError:
        return cls(
            ErrorCode.DUPLICATE_TASK_ID,
            f"This source already holds a task with the id [{task_id}]. Ids must be unique "
            "within a source, because a claim, a release and a lease all address a task by it.",
        )

    @classmethod
    def task_identifier_blank(cls, kind: str) -> HarnessError:
        """A blank worker or task id, refused rather than stored.

        One code for both because the caller's move is the same either way:
        supply a non-empty identifier. The prose says which one was blank.
        """
        return cls(
            ErrorCode.TASK_IDENTIFIER_BLANK,
            f"A {kind} identifier was the empty string. It is refused rather than stored: an "
            "empty string is FALSY in PHP, so a blank owner reads as 'unclaimed' to any "
            "implementation testing the field for truth -- a task that is held and looks free.",
        )

    @classmethod
    def task_not_found(cls, task_id: str) -> HarnessError:
        return cls(
            ErrorCode.TASK_NOT_FOUND,
            f"This source holds no task with the id [{task_id}].",
        )

    @classmethod
    def task_already_terminal(cls, task_id: str, state: str) -> HarnessError:
        """Re-releasing a terminal task.

        An ERROR, not a silent no-op. A second release quietly discarded is a
        second worker's evidence being thrown away with nothing to show for it.
        """
        return cls(
            ErrorCode.TASK_ALREADY_TERMINAL,
            f"Task [{task_id}] is already [{state}], which is terminal. Releasing it again is "
            "refused rather than ignored: a silent no-op discards whatever the second caller "
            "had to report. An application that wants the task run again re-queues it.",
        )

    @classmethod
    def task_lease_not_held(cls, task_id: str, detail: str) -> HarnessError:
        return cls(
            ErrorCode.TASK_LEASE_NOT_HELD,
            f"Task [{task_id}] cannot be acted on by this worker: {detail}. Another worker may "
            "already be doing it, so a report from a lapsed holder is refused.",
        )

    @classmethod
    def task_outcome_invalid(cls, value: object) -> HarnessError:
        """An outcome that is not exactly one of the two.

        REFUSED, never resolved, and the direction matters more than the
        refusal. An implementation that treats "anything not `failed`" as
        `done` turns every typo, every casing slip and every empty object into
        the MORE privileged answer -- an agent declaring victory by getting the
        word slightly wrong. `prism-harness-ts` shipped exactly that, and this
        port had the same escalation reached the other way, by ignoring the
        argument entirely.
        """
        return cls(
            ErrorCode.TASK_OUTCOME_INVALID,
            f"[{value!r}] is not a task outcome. It must be exactly 'done' or 'failed' -- not "
            "a different casing, not a synonym, and not absent. A value that cannot be read as "
            "one of the two is refused rather than resolved to either, because resolving it "
            "would always mean choosing an outcome the caller did not ask for.",
        )

    @classmethod
    def task_outcome_not_supplied(cls) -> HarnessError:
        """No outcome at all, which is refused exactly like a malformed one.

        Same code, deliberately. "The agent called ``complete_task``, so it
        meant completion" is the SAME inference that produced the hardcoded
        ``done`` this package shipped and then removed -- reading the privileged
        outcome out of silence, one level up from where it was caught. An agent
        that omitted the field has not stated an outcome, and there is nothing
        to infer from that which is safe to infer.
        """
        return cls(
            ErrorCode.TASK_OUTCOME_INVALID,
            "No task outcome was supplied. It must be stated explicitly as 'done' or 'failed'; "
            "an absent outcome is not a request to complete the task. Inferring the more "
            "privileged answer from silence is the same escalation as coercing an unreadable "
            "one into it.",
        )

    @classmethod
    def task_lease_invalid(cls, seconds: object) -> HarnessError:
        """A lease that is zero, negative, or not a finite number.

        REFUSED, not clamped. A clamped lease is a configuration that silently
        became a different configuration -- this repository has already shipped
        one of those and stayed green the whole time. A zero or negative lease
        is also not merely odd: it expires the instant it is granted, so the
        claim it was meant to protect is stealable by the next caller, and the
        one guarantee this design exists to make quietly stops holding.
        """
        return cls(
            ErrorCode.TASK_LEASE_INVALID,
            f"[{seconds!r}] is not a usable lease. It must be a finite number of seconds "
            "greater than zero. It is refused rather than clamped to something workable: a "
            "lease that expires the moment it is granted leaves the claim it should protect "
            "open to the next caller, and silently substituting a different number would hide "
            "that rather than report it.",
        )

    @classmethod
    def no_agent_runtime(cls, action: str) -> HarnessError:
        return cls(
            ErrorCode.NO_AGENT_RUNTIME,
            f"This session cannot {action}: it was built without an agent runtime.",
        )
