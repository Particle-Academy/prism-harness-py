"""What a mode lets a run do, and the registry that resolves one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism_harness.errors import HarnessError
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

    def needs_approval(self, tool: str) -> bool:
        """Whether a named tool needs a human before it runs IN THIS MODE.

        Declared per mode rather than on the tool, because the same tool is not
        equally consequential everywhere: ``execute_op`` against a scratch
        project is routine and against production is not, and the tool cannot
        tell which it is in. ``'*'`` gates every tool the mode offers.
        """
        return "*" in self.requires_approval or tool in self.requires_approval


class ModeRegistry:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

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

        return AgentMode(
            name=wanted,
            system_prompt=prompt,
            tools=_strings(mode.get("tools")),
            skills=_strings(mode.get("skills")),
            max_steps=max_steps,
            subagents=self._subagents_for(wanted, mode),
            requires_approval=_strings(mode.get("requires_approval")),
        )

    def _subagents_for(self, name: str, mode: dict[str, Any]) -> dict[str, Subagent]:
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
        modes = self._config.get("modes")

        for key, config in declared.items():
            if not isinstance(key, str) or not isinstance(config, dict):
                raise HarnessError.mode_malformed(name, f"subagent [{key}] is not an object")

            subagent = subagent_from_config(key, config)

            if not isinstance(modes, dict) or not isinstance(modes.get(subagent.mode), dict):
                raise HarnessError.mode_malformed(
                    name,
                    f"it declares subagent [{key}], whose mode [{subagent.mode}] is not configured",
                )

            subagents[key] = subagent

        return subagents


def _strings(value: Any) -> list[str]:
    return [entry for entry in value if isinstance(entry, str)] if isinstance(value, list) else []
