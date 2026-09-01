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

    @classmethod
    def no_agent_runtime(cls, action: str) -> HarnessError:
        return cls(
            ErrorCode.NO_AGENT_RUNTIME,
            f"This session cannot {action}: it was built without an agent runtime.",
        )
