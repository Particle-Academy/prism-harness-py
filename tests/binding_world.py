"""A harness wired the way an application wires it, for provider-binding tests (G-321 E).

Every mode gets a system prompt naming its agent and mode, so a recorded request
says which run sent it. A mode's declared subagents are offered as tools: an
application builds the subagent tool from the runner, which is what the factory
below does.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessEvents,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    ModeRegistry,
    Participant,
    PrismHarness,
    RunBudget,
    RunContext,
    Session,
    SubagentRunner,
    ToolRegistry,
)
from prism_harness.runtime import RuntimeConfig
from prism_harness.tools import HarnessTool

PARENT_RUN = "parent-run"


def prompt_for(agent: str | None, mode: str) -> str:
    return f"mode:{agent or '-'}:{mode}"


def _with_prompts(modes: dict[str, Any], agent: str | None) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for name, mode in modes.items():
        mode = deepcopy(mode)
        mode.setdefault("system_prompt", prompt_for(agent, name))
        mode.setdefault("max_steps", 8)
        subagents = mode.get("subagents")
        mode.setdefault("tools", list(subagents) if isinstance(subagents, dict) else [])
        prepared[name] = mode
    return prepared


class World:
    def __init__(
        self,
        *,
        global_config: dict[str, Any] | None = None,
        modes: dict[str, Any] | None = None,
        agents: dict[str, dict[str, Any]] | None = None,
        reply: Callable[[World, LlmRequest], LlmResponse] | None = None,
    ) -> None:
        agents = agents or {}
        self.requests: list[LlmRequest] = []
        self.events: list[Any] = []
        self._reply = reply or World.default_reply
        self.delegate: str | None = None
        self.parent_prompt = ""
        self.modes = ModeRegistry(
            {
                "modes": _with_prompts(modes or {}, None),
                "agents": {
                    agent: {
                        "owner": "binding-test",
                        "modes": _with_prompts(options.get("modes", {}), agent),
                    }
                    for agent, options in agents.items()
                },
            }
        )
        settings: dict[str, RuntimeConfig] = {}
        for agent, options in agents.items():
            setting: dict[str, Any] = {
                key: options[key]
                for key in ("provider", "model", "spend_ceiling")
                if key in options
            }
            settings[agent] = setting  # type: ignore[assignment]
        self.tools = ToolRegistry()
        bus = HarnessEvents()
        bus.listen(self.events.append)
        self.runtime = AgentRuntime(
            client=self._client,
            modes=self.modes,
            tools=self.tools,
            events=bus,
            config=dict(global_config or {}),  # type: ignore[arg-type]
            agents=settings,
        )
        directory = tempfile.mkdtemp(prefix="prism-harness-binding-")
        self.harness = PrismHarness(
            drivers={"files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "files", "durable": "files"},
        )
        self.runner = SubagentRunner(self.harness, self.runtime)
        self.context = RunContext.root(PARENT_RUN, RunBudget(8))

    # -- the model -------------------------------------------------------

    def _client(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return self._reply(self, request)

    @staticmethod
    def default_reply(world: World, request: LlmRequest) -> LlmResponse:
        parent_calls = [r for r in world.requests if r.system_prompt == world.parent_prompt]
        if (
            request.system_prompt == world.parent_prompt
            and world.delegate is not None
            and len(parent_calls) == 1
        ):
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall(id="call-1", name=world.delegate, arguments={"task": "t"})],
            )
        return LlmResponse(text="answered", finish_reason="stop")

    # -- the application -------------------------------------------------

    def session(
        self,
        agent: str | None,
        mode: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        scope: str = "binding",
    ) -> Session:
        session = (
            self.harness.for_agent(agent).for_(Participant("User", 1)).session(scope)
            if agent is not None
            else self.harness.for_(Participant("User", 1)).session(scope)
        )
        session.using_mode(mode)
        if provider is not None:
            session.using_provider(provider)
        if model is not None:
            session.using_model(model)
        self.parent_prompt = prompt_for(agent, mode)
        return session

    def offer_subagents(self, agent: str | None, mode: str) -> None:
        """Register each subagent the mode declares as a tool, resolved lazily.

        Lazily, because resolving the mode is what refuses a bad declaration,
        and the runtime must be the one to meet that refusal.
        """
        raw = self.modes.for_agent(agent)._config.get("modes", {}).get(mode, {})
        for name in (raw.get("subagents") or {}) if isinstance(raw, dict) else {}:

            def factory(session: Session, name: str = name) -> HarnessTool:
                subagent = self.modes.for_agent(agent).resolve(mode).subagents[name]
                return self.runner.tool(subagent, session, self.context, PARENT_RUN)

            self.tools.for_agent(agent).register_factory(name, factory, owner="binding-test")

    def add_tool(self, agent: str | None, name: str) -> None:
        """A static tool that answers ``ok``, for a model that keeps calling tools."""

        class Again:
            def __init__(self) -> None:
                self.name = name

            def handle(self, args: dict[str, Any]) -> str:
                return "ok"

        self.tools.for_agent(agent).register(Again(), owner="binding-test")

    def send(self, session: Session, prompt: str = "go") -> Any:
        return self.runtime.send(session, prompt, context=self.context)

    def parent_requests(self) -> list[LlmRequest]:
        return [r for r in self.requests if r.system_prompt == self.parent_prompt]

    def child_requests(self) -> list[LlmRequest]:
        return [r for r in self.requests if r.system_prompt != self.parent_prompt]
