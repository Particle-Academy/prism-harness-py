"""Which tools exist, which a run may be offered, and which calls may proceed."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from prism_harness.address import validate_agent
from prism_harness.errors import HarnessError
from prism_harness.registration import RegistrationCollision, registration_owner

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

    STRUCTURAL, not an import. ``prism-ai-core``'s tool satisfies it, and so does
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


class AgentAuthorizationOptions(TypedDict, total=False):
    enabled: bool
    offer: OfferPolicy | None
    call: CallPolicy | None


@dataclass(frozen=True)
class ResolvedTool:
    tool: HarnessTool
    owner: str
    source: str


class ToolRegistry:
    """Three ways in, because the three answer different questions: a tool that
    is always the same, one that needs the session to build, and a set
    discovered at resolve time -- an MCP server's catalogue, say.
    """

    def __init__(
        self,
        agent: str | None = None,
        _root: ToolRegistry | None = None,
        *,
        refuse_duplicates: bool = False,
    ) -> None:
        if not isinstance(refuse_duplicates, bool):
            raise TypeError("refuse_duplicates must be a boolean")
        self._refuse_duplicates = refuse_duplicates
        validate_agent(agent)
        self._agent = agent
        self._root = _root
        self._agents: dict[str, ToolRegistry] = {}
        self._tools: dict[str, HarnessTool] = {}
        self._factories: dict[str, ToolFactory] = {}
        self._metadata: dict[str, tuple[str, str]] = {}
        self._static_owners: dict[str, str] = {}
        self._factory_owners: dict[str, str] = {}
        self._providers: dict[str, tuple[ToolProvider, str]] = {}

    def for_agent(self, agent: str | None) -> ToolRegistry:
        validate_agent(agent)
        root = self._root if self._root is not None else self
        if agent is None:
            return root
        if agent not in root._agents:
            root._agents[agent] = ToolRegistry(
                agent, root, refuse_duplicates=root._refuse_duplicates
            )
        return root._agents[agent]

    def register(self, tool: HarnessTool, owner: str | None = None) -> ToolRegistry:
        self._claim(tool.name, owner, "static")
        self._tools[tool.name] = tool
        self._static_owners[tool.name] = registration_owner(owner, self._agent)
        return self

    def register_many(self, tools: Iterable[HarnessTool], owner: str | None = None) -> ToolRegistry:
        for tool in tools:
            self.register(tool, owner)
        return self

    def register_factory(
        self, name: str, factory: ToolFactory, owner: str | None = None
    ) -> ToolRegistry:
        self._claim(name, owner, "factory")
        self._factories[name] = factory
        self._factory_owners[name] = registration_owner(owner, self._agent)
        return self

    def register_provider(
        self, provider: ToolProvider, owner: str | None = None, name: str | None = None
    ) -> ToolRegistry:
        label = registration_owner(owner, self._agent)
        identifier = (
            name
            if name is not None
            else (f"provider:{len(self._providers)}" if self._agent is None else None)
        )
        if not identifier:
            raise ValueError("A named provider requires a registration name.")
        previous = self._providers.get(identifier)
        if self._strict() and previous is not None:
            raise RegistrationCollision(self._agent, identifier, previous[1], label)
        self._providers[identifier] = (provider, label)
        return self

    def replace(
        self, tool: HarnessTool, predecessor_owner: str, owner: str | None = None
    ) -> ToolRegistry:
        self._claim(tool.name, owner, "static", predecessor_owner)
        self._tools[tool.name] = tool
        self._static_owners[tool.name] = registration_owner(owner, self._agent)
        return self

    def replace_factory(
        self, name: str, factory: ToolFactory, predecessor_owner: str, owner: str | None = None
    ) -> ToolRegistry:
        self._claim(name, owner, "factory", predecessor_owner)
        self._factories[name] = factory
        self._factory_owners[name] = registration_owner(owner, self._agent)
        return self

    def replace_provider(
        self, name: str, provider: ToolProvider, predecessor_owner: str, owner: str | None = None
    ) -> ToolRegistry:
        label = registration_owner(owner, self._agent)
        previous = self._providers.get(name)
        if previous is None or previous[1] != predecessor_owner:
            raise ValueError("Replacement must name the registered predecessor owner.")
        self._providers[name] = (provider, label)
        return self

    def _claim(
        self, name: str, owner: str | None, source: str, predecessor: str | None = None
    ) -> None:
        label = registration_owner(owner, self._agent)
        previous = self._metadata.get(name)
        if predecessor is not None:
            if previous is None or previous[0] != predecessor:
                raise ValueError("Replacement must name the registered predecessor owner.")
        elif self._strict() and previous is not None:
            raise RegistrationCollision(self._agent, name, previous[0], label)
        if self._strict() or predecessor is not None:
            self._tools.pop(name, None)
            self._factories.pop(name, None)
            self._static_owners.pop(name, None)
            self._factory_owners.pop(name, None)
        self._metadata[name] = (label, source)

    def _strict(self) -> bool:
        return self._agent is not None or self._refuse_duplicates

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
        return {name: entry.tool for name, entry in self.resolve_detailed(names, session).items()}

    def resolve_detailed(
        self, names: Sequence[str], session: Session | None = None
    ) -> dict[str, ResolvedTool]:
        if session is not None and session.agent != self._agent:
            if self._agent is not None:
                raise ValueError("Harness registry/session agent mismatch.")
            return self.for_agent(session.agent).resolve_detailed(names, session)
        provided: dict[str, HarnessTool] = {}
        metadata = self._metadata.copy()

        if session is not None:
            for provider, owner in self._providers.values():
                for tool in provider(session):
                    previous = metadata.get(tool.name)
                    if self._strict() and previous is not None:
                        raise RegistrationCollision(self._agent, tool.name, previous[0], owner)
                    provided[tool.name] = tool
                    metadata[tool.name] = (owner, "provider")

        available = sorted({*self._tools, *self._factories, *provided})
        selected = available if "*" in names else list(names)
        resolved: dict[str, ResolvedTool] = {}

        for name in selected:
            existing = self._tools.get(name) or provided.get(name)

            if existing is not None:
                details = (
                    (self._static_owners[name], "static") if name in self._tools else metadata[name]
                )
                resolved[name] = ResolvedTool(existing, *details)
                continue

            factory = self._factories.get(name)

            if factory is None:
                raise HarnessError.tool_not_available(name, available)

            if session is None:
                raise HarnessError.tool_not_available(
                    f"{name} (needs a session to build)", available
                )

            built = factory(session)
            if self._strict() and built.name != name:
                raise ValueError(f"Session-bound tool factory [{name}] returned [{built.name}].")
            if built.name != name:
                previous = metadata.get(built.name)
                if self._strict() and previous is not None:
                    raise RegistrationCollision(
                        self._agent, built.name, previous[0], metadata[name][0]
                    )
                metadata[built.name] = metadata[name]
            resolved[name] = ResolvedTool(built, self._factory_owners[name], "factory")

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
        agents: dict[str, AgentAuthorizationOptions] | None = None,
        _agent: str | None = None,
        _root: ToolAuthorizer | None = None,
    ) -> None:
        validate_agent(_agent)
        self.enabled = enabled
        self._offer = offer
        self._call = call
        self._agent, self._root = _agent, _root
        self._agents = {agent: options.copy() for agent, options in (agents or {}).items()}
        self._scoped: dict[str, ToolAuthorizer] = {}
        for agent in self._agents:
            validate_agent(agent)

        if not enabled and (offer is not None or call is not None):
            raise HarnessError.policy_defined_but_disabled()

    def for_agent(self, agent: str | None) -> ToolAuthorizer:
        validate_agent(agent)
        root = self._root if self._root is not None else self
        if agent is None:
            return root
        if agent not in root._scoped:
            options = root._agents.get(agent, {})
            enabled = options.get("enabled", root.enabled)
            offer, call = options.get("offer", root._offer), options.get("call", root._call)
            if (
                not isinstance(enabled, bool)
                or (offer is not None and not callable(offer))
                or (call is not None and not callable(call))
            ):
                raise TypeError("Malformed agent tool authorization options.")
            root._scoped[agent] = ToolAuthorizer(enabled, offer, call, _agent=agent, _root=root)
        return root._scoped[agent]

    def has_call_policy(self) -> bool:
        return self._call is not None

    def allowed(self, session: Session, tools: dict[str, HarnessTool]) -> list[HarnessTool]:
        """The tools a run may be offered, each wrapped so the call policy is asked again."""
        if session.agent != self._agent:
            if self._agent is not None:
                raise ValueError("Harness authorizer/session agent mismatch.")
            return self.for_agent(session.agent).allowed(session, tools)
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
        if session.agent != self._agent:
            if self._agent is not None:
                raise ValueError("Harness authorizer/session agent mismatch.")
            return self.for_agent(session.agent).allows_call(session, tool, args)
        if not self.enabled or self._call is None:
            return True

        return self._call(session, tool, args)
