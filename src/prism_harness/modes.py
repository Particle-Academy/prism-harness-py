"""What a mode lets a run do, and the registry that resolves one."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from prism_harness.address import validate_agent
from prism_harness.errors import HarnessError
from prism_harness.registration import RegistrationCollision, registration_owner
from prism_harness.subagents import Subagent, subagent_from_config

__all__ = ["AgentMode", "ModeRegistry"]


@dataclass(frozen=True)
class AgentMode:
    """A mode is the unit of authority: the tools it may reach, the skills it
    loads, how many steps it gets, which nested agents it may call, and which of
    its tools need a human first.
    """

    name: str
    system_prompt: str
    tools: list[str]
    skills: list[str]
    max_steps: int
    #: Nested agents this mode may call, by name.
    subagents: dict[str, Subagent] = field(default_factory=dict)
    #: Tools that must not run until a human says so.
    requires_approval: list[str] = field(default_factory=list)
    #: Options handed to the provider on every run in this mode, unchanged.
    provider_options: dict[str, Any] = field(default_factory=dict)
    #: The provider and model runs in this mode use, below a session override
    #: and above the agent default (G-321).
    provider: str | None = None
    model: str | None = None
    #: Named agents this mode's subagents may run as. Declared, never inherited.
    agents: list[str] = field(default_factory=list)
    #: Whether runs in this mode may keep an agent plan (G-360). Off unless set.
    plans: bool = False

    def needs_approval(self, tool: str) -> bool:
        """Whether a named tool needs a human before it runs IN THIS MODE.

        Declared per mode rather than on the tool, because the same tool is not
        equally consequential everywhere: ``execute_op`` against a scratch
        project is routine and against production is not, and the tool cannot
        tell which it is in. ``'*'`` gates every tool the mode offers.
        """
        return "*" in self.requires_approval or tool in self.requires_approval


class ModeRegistry:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        _agent: str | None = None,
        _root: ModeRegistry | None = None,
        *,
        refuse_duplicates: bool = False,
    ) -> None:
        if not isinstance(refuse_duplicates, bool):
            raise TypeError("refuse_duplicates must be a boolean")
        self._refuse_duplicates = refuse_duplicates
        validate_agent(_agent)
        self._config = config if config is not None else {}
        self._agent, self._root = _agent, _root
        self._agents: dict[str, ModeRegistry] = {}
        self._owners = {
            name: registration_owner(self._config.get("owner"), _agent) for name in self.names()
        }

    def for_agent(self, agent: str | None) -> ModeRegistry:
        validate_agent(agent)
        root = self._root if self._root is not None else self
        if agent is None:
            return root
        if agent not in root._agents:
            agents = root._config.get("agents", {})
            configured = agents.get(agent, {}) if isinstance(agents, dict) else None
            if not isinstance(configured, dict):
                raise ValueError("Agent modes must be configured as a mapping.")
            config = deepcopy(configured)
            config.setdefault("default", root.default())
            root._agents[agent] = ModeRegistry(
                config, agent, root, refuse_duplicates=root._refuse_duplicates
            )
        return root._agents[agent]

    def register(self, name: str, mode: dict[str, Any], owner: str | None = None) -> ModeRegistry:
        label = registration_owner(owner, self._agent)
        if (self._agent is not None or self._refuse_duplicates) and name in self._owners:
            raise RegistrationCollision(self._agent, name, self._owners[name], label)
        self._set(name, mode, label)
        return self

    def replace(
        self, name: str, mode: dict[str, Any], predecessor_owner: str, owner: str | None = None
    ) -> ModeRegistry:
        label = registration_owner(owner, self._agent)
        if self._owners.get(name) != predecessor_owner:
            raise ValueError("Replacement must name the registered predecessor owner.")
        self._set(name, mode, label)
        return self

    def _set(self, name: str, mode: dict[str, Any], owner: str) -> None:
        modes = self._config.setdefault("modes", {})
        if not isinstance(modes, dict):
            raise TypeError("Modes must be configured as a mapping.")
        modes[name] = deepcopy(mode)
        self._owners[name] = owner

    def set_default(self, name: str) -> ModeRegistry:
        self.resolve(name)
        self._config["default"] = name
        return self

    def default(self) -> str:
        configured = self._config.get("default")
        return configured if isinstance(configured, str) and configured else "chat"

    def names(self) -> list[str]:
        """Every mode name this application has configured."""
        modes = self._config.get("modes")
        return [name for name in modes if isinstance(name, str)] if isinstance(modes, dict) else []

    def all(self) -> dict[str, AgentMode]:
        """Every mode, RESOLVED.

        Resolving them all is the point: :meth:`resolve` validates as it goes,
        so a mode nobody has entered yet keeps its misconfiguration until the
        day somebody switches to it. This is what the doctor uses to find that
        on a Tuesday rather than in front of a user.
        """
        return {name: self.resolve(name) for name in self.names()}

    def resolve(self, name: str | None = None) -> AgentMode:
        wanted = name if name is not None else self.default()
        modes = self._config.get("modes")
        mode = modes.get(wanted) if isinstance(modes, dict) else None

        if not isinstance(mode, dict):
            raise HarnessError.mode_not_configured(wanted)

        prompt = mode.get("system_prompt", "")
        max_steps = mode.get("max_steps", 8)

        if (
            not isinstance(prompt, str)
            or not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise HarnessError.mode_malformed(
                wanted, "system_prompt must be a string and max_steps a positive integer"
            )

        provider, model = _binding(mode)
        if provider is False or model is False:
            raise HarnessError.mode_declaration(
                f"Harness mode [{wanted}] declares a malformed provider/model: "
                "expected a nonempty string or null."
            )
        agents = _agent_list(mode.get("agents"))
        if agents is None:
            raise HarnessError.mode_declaration(
                f"Harness mode [{wanted}] declares a malformed agents list: each entry must be "
                "a lowercase agent id of 1-64 ASCII bytes."
            )

        # G-360: a mode opts in to agent plans. Anything but true or false is
        # refused, so a string "false" cannot quietly turn plans on.
        plans = mode.get("plans", False)
        if plans is None:
            plans = False
        if not isinstance(plans, bool):
            raise HarnessError.mode_declaration(
                f"Harness mode [{wanted}] declares a malformed plans flag: expected true or false."
            )

        return AgentMode(
            name=wanted,
            system_prompt=prompt,
            tools=_strings(mode.get("tools")),
            skills=_strings(mode.get("skills")),
            max_steps=max_steps,
            subagents=self._subagents_for(wanted, mode, agents),
            requires_approval=_strings(mode.get("requires_approval")),
            provider_options=_provider_options(wanted, mode.get("provider_options")),
            provider=provider if isinstance(provider, str) else None,
            model=model if isinstance(model, str) else None,
            agents=agents,
            plans=plans,
        )

    def _subagents_for(
        self, name: str, mode: dict[str, Any], agents: list[str]
    ) -> dict[str, Subagent]:
        """The nested agents a mode may call.

        DECLARED PER MODE, which is the point: a subagent is authority, and
        authority a run inherits by being nested is authority nobody granted. A
        mode that names no subagents cannot spawn one.

        A subagent's own mode must EXIST, and it is checked here rather than at
        call time so a typo surfaces when the parent mode is loaded instead of
        halfway through a run that has already spent budget.
        """
        declared = mode.get("subagents") or {}

        if not isinstance(declared, dict):
            raise HarnessError.mode_malformed(name, "its subagent list is not a mapping")

        subagents: dict[str, Subagent] = {}

        for key, config in declared.items():
            if not isinstance(key, str) or not isinstance(config, dict):
                raise HarnessError.mode_malformed(name, f"subagent [{key}] is not an object")

            self._check_subagent_binding(name, key, config, agents)
            subagent = subagent_from_config(key, config)

            # The child runs in the DELEGATED agent when one is named, so that is
            # where its mode has to exist. Names only: resolving it here would
            # recurse through every agent a mode can reach.
            target = (
                self
                if subagent.agent is None or subagent.agent == self._agent
                else self.for_agent(subagent.agent)
            )
            target_modes = target._config.get("modes")
            if not isinstance(target_modes, dict) or not isinstance(
                target_modes.get(subagent.mode), dict
            ):
                raise HarnessError.mode_malformed(
                    name,
                    f"it declares subagent [{key}], whose mode [{subagent.mode}] is not configured",
                )

            subagents[key] = subagent

        return subagents

    def _check_subagent_binding(
        self, name: str, key: str, config: dict[str, Any], agents: list[str]
    ) -> None:
        """A subagent's provider, model and agent, refused before any run.

        The agent id is held to the identifier rule exactly -- never trimmed,
        never case-folded. A near-miss that was normalised would select another
        agent's authority under a name nobody declared.
        """
        provider, model = _binding(config)
        if provider is False or model is False:
            raise HarnessError.mode_declaration(
                f"Harness mode [{name}] declares subagent [{key}] with a malformed "
                "provider/model: expected a nonempty string or null."
            )

        inherit = config.get("inherit_session_model", False)
        if not isinstance(inherit, bool):
            raise HarnessError.mode_declaration(
                f"Harness mode [{name}] declares subagent [{key}] with a malformed "
                "inherit_session_model: expected a boolean."
            )

        agent = config.get("agent")
        if agent is None:
            return
        if not _valid_agent(agent):
            raise HarnessError.mode_declaration(
                f"Harness mode [{name}] declares subagent [{key}] with a malformed agent id: "
                "agent ids are lowercase identifiers of 1-64 ASCII bytes."
            )
        if agent != self._agent and agent not in agents:
            raise HarnessError.mode_declaration(
                f"Harness mode [{name}] declares subagent [{key}] for agent [{agent}], which "
                "the mode does not list in its agents."
            )


def _valid_agent(value: object) -> bool:
    try:
        validate_agent(value)  # type: ignore[arg-type]
    except ValueError:
        return False
    return isinstance(value, str)


def _binding(config: dict[str, Any]) -> tuple[str | None | bool, str | None | bool]:
    """``provider`` and ``model``: a nonempty string, None, or False when malformed."""

    def one(value: object) -> str | None | bool:
        if value is None:
            return None
        return value if isinstance(value, str) and value else False

    return one(config.get("provider")), one(config.get("model"))


def _agent_list(value: object) -> list[str] | None:
    """A mode's delegable agents, or None when any entry is not an agent id."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(_valid_agent(entry) for entry in value):
        return None
    return list(value)


def _strings(value: Any) -> list[str]:
    return [entry for entry in value if isinstance(entry, str)] if isinstance(value, list) else []


def _provider_options(name: str, declared: object) -> dict[str, Any]:
    """A mode's provider options, refused rather than ignored when not a map.

    A list or a scalar would reach the provider as nothing, and the mode would
    run without the option its author believes is on.
    """
    if declared is None:
        return {}

    if not isinstance(declared, dict) or not all(isinstance(key, str) for key in declared):
        raise HarnessError.mode_malformed(
            name, "provider_options must be a map of option names to values"
        )

    return dict(declared)
