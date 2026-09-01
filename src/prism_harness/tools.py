"""Which tools exist, which a run may be offered, and which calls may proceed."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from prism_harness.errors import HarnessError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_harness.session import Session

__all__ = [
    "AuthorizedTool",
    "CallPolicy",
    "HarnessTool",
    "OfferPolicy",
    "ToolAuthorizer",
    "ToolFactory",
    "ToolProvider",
    "ToolRegistry",
]


class HarnessTool(Protocol):
    """The shape this package needs from a tool.

    STRUCTURAL, not an import. ``prism-ai``'s tool satisfies it, and so does
    anything else with a name and a handler -- which keeps this package at zero
    dependencies and lets a consumer bring their own tool type. The reference
    imports Prism's ``Tool`` directly because it is already a dependency there.
    """

    @property
    def name(self) -> str: ...

    def handle(self, args: dict[str, Any]) -> Any: ...


ToolFactory = Callable[["Session"], HarnessTool]
ToolProvider = Callable[["Session"], Iterable[HarnessTool]]
#: Whether a tool may be OFFERED to a run.
OfferPolicy = Callable[["Session", HarnessTool], bool]
#: Whether THIS call, with THESE arguments, may proceed.
CallPolicy = Callable[["Session", HarnessTool, dict[str, Any]], bool]


class ToolRegistry:
    """Three ways in, because the three answer different questions: a tool that
    is always the same, one that needs the session to build, and a set
    discovered at resolve time -- an MCP server's catalogue, say.
    """

    def __init__(self) -> None:
        self._tools: dict[str, HarnessTool] = {}
        self._factories: dict[str, ToolFactory] = {}
        self._providers: list[ToolProvider] = []

    def register(self, tool: HarnessTool) -> ToolRegistry:
        self._tools[tool.name] = tool
        return self

    def register_many(self, tools: Iterable[HarnessTool]) -> ToolRegistry:
        for tool in tools:
            self.register(tool)
        return self

    def register_factory(self, name: str, factory: ToolFactory) -> ToolRegistry:
        self._factories[name] = factory
        return self

    def register_provider(self, provider: ToolProvider) -> ToolRegistry:
        self._providers.append(provider)
        return self

    def names(self) -> list[str]:
        """Every name this registry can produce without a session."""
        return sorted({*self._tools, *self._factories})

    def resolve(
        self, names: Sequence[str], session: Session | None = None
    ) -> dict[str, HarnessTool]:
        """Resolve the named tools for a session.

        ``'*'`` means every tool this registry can produce. A name that resolves
        to nothing is an ERROR rather than a silent omission: a mode that lists
        a tool it cannot get is a misconfiguration, and dropping it quietly
        would leave a run wondering why the model never called it.
        """
        provided: dict[str, HarnessTool] = {}

        if session is not None:
            for provider in self._providers:
                for tool in provider(session):
                    provided[tool.name] = tool

        available = sorted({*self._tools, *self._factories, *provided})
        selected = available if "*" in names else list(names)
        resolved: dict[str, HarnessTool] = {}

        for name in selected:
            existing = self._tools.get(name) or provided.get(name)

            if existing is not None:
                resolved[name] = existing
                continue

            factory = self._factories.get(name)

            if factory is None:
                raise HarnessError.tool_not_available(name, available)

            if session is None:
                raise HarnessError.tool_not_available(
                    f"{name} (needs a session to build)", available
                )

            resolved[name] = factory(session)

        return resolved


class AuthorizedTool:
    """A tool that asks the policy AGAIN, at call time, with the arguments.

    Refuses by RAISING rather than returning an error string: a refusal handed
    back as a tool result reads to the model as a failure it might retry
    differently, and a denied action being retried is the opposite of what a
    guard is for.
    """

    def __init__(self, tool: HarnessTool, session: Session, authorizer: ToolAuthorizer) -> None:
        self._tool = tool
        self._session = session
        self._authorizer = authorizer

    @property
    def name(self) -> str:
        return self._tool.name

    def handle(self, args: dict[str, Any]) -> Any:
        if not self._authorizer.allows_call(self._session, self._tool, args):
            raise HarnessError.call_not_authorized(self._tool.name)

        return self._tool.handle(args)


class ToolAuthorizer:
    """The two authorization questions, kept apart.

    Whether a tool may be OFFERED to a run, and whether THIS invocation of it --
    with these arguments, this many calls in -- may proceed. At the moment the
    toolset is assembled the arguments do not exist yet, so an offer policy can
    say "may use delete_file" and never "only under /tmp".

    OFF BY DEFAULT, matching the reference. But a policy that is defined and
    never consulted is REFUSED at construction rather than tolerated: it looks
    like a control to every reader and is not one. That is the one configuration
    not to leave in place -- both at once.
    """

    def __init__(
        self,
        enabled: bool = False,
        offer: OfferPolicy | None = None,
        call: CallPolicy | None = None,
    ) -> None:
        self.enabled = enabled
        self._offer = offer
        self._call = call

        if not enabled and (offer is not None or call is not None):
            raise HarnessError.policy_defined_but_disabled()

    def allowed(self, session: Session, tools: dict[str, HarnessTool]) -> list[HarnessTool]:
        """The tools a run may be offered, each wrapped so the call policy is asked again."""
        if not self.enabled:
            return list(tools.values())

        allowed: list[HarnessTool] = []

        for tool in tools.values():
            if self._offer is None or self._offer(session, tool):
                # Wrapped so the SAME policy is asked when the tool is actually
                # called, with arguments. Offer-time filtering alone cannot
                # bound how a tool is used, only whether it is present.
                allowed.append(AuthorizedTool(tool, session, self))

        return allowed

    def allows_call(self, session: Session, tool: HarnessTool, args: dict[str, Any]) -> bool:
        """True when the authorizer is disabled, matching :meth:`allowed` -- the
        constructor has already refused the configuration where that silence
        would be mistaken for enforcement.
        """
        if not self.enabled or self._call is None:
            return True

        return self._call(session, tool, args)
