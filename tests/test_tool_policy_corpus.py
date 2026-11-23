"""The cross-language tool-policy corpus (G-326).

Mirrors prism-harness-ts/test/tool-policy-corpus.test.ts. Every row's
``expected`` was written by hand before any language recorded. This runner
drives the real runtime with a fake model that asks for the row's calls, then
answers. Every tool needs approval by mode; a tool with ``provider`` is
supplied at resolve time by a provider registered under that name, as an MCP
bridge supplies one, and ``server`` on it is only its claim.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    Session,
    ToolPolicy,
    ToolRegistry,
)

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "harness-tool-policy.json").read_text(encoding="utf-8")
)


@dataclass
class CorpusTool:
    name: str
    ran: list[str]
    server: str | None = None
    tier: str | None = None

    def handle(self, args: dict[str, Any]) -> Any:
        self.ran.append(self.name)
        return "ok"


def record(row: dict[str, Any]) -> dict[str, Any]:
    ran: list[str] = []
    events: list[HarnessEvent] = []
    registry = ToolRegistry()
    provided: dict[str, list[CorpusTool]] = {}
    for spec in row["tools"]:
        tool = CorpusTool(spec["name"], ran, spec.get("server"), spec.get("tier"))
        if "provider" not in spec:
            registry.register(tool)
        else:
            provided.setdefault(spec["provider"], []).append(tool)
    for name, tools in provided.items():

        def supply(session: Session, tools: list[CorpusTool] = tools) -> list[CorpusTool]:
            return list(tools)

        registry.register_provider(supply, "mcp", name)
    modes = ModeRegistry(
        {
            "default": "guarded",
            "modes": {
                "guarded": {
                    "system_prompt": "Careful.",
                    "tools": [tool["name"] for tool in row["tools"]],
                    "max_steps": 6,
                    "requires_approval": ["*"],
                }
            },
        }
    )
    steps = [0]

    def client(request: LlmRequest) -> LlmResponse:
        steps[0] += 1
        if steps[0] == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    LlmToolCall(f"c{index}", call["name"], call["args"])
                    for index, call in enumerate(row["calls"])
                ],
            )
        return LlmResponse(text="done", finish_reason="stop")

    bus = HarnessEvents()
    bus.listen(events.append)
    try:
        runtime = AgentRuntime(
            client=client,
            modes=modes,
            tools=registry,
            events=bus,
            tool_policy=ToolPolicy.from_json(row["policy"]),
        )
    except HarnessError as error:
        if error.code == "tool_policy_refused":
            return {"refused": error.code, "ran": [], "asked": [], "events": []}
        raise

    directory = tempfile.mkdtemp(prefix="prism-harness-tool-policy-corpus-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("corpus")
    session.using_provider("anthropic").using_model("fake")

    response = runtime.send(session, "go")
    if row["then"] is not None:
        runtime.set_tool_policy(ToolPolicy.from_json(row["then"]["policy"]))
        response = runtime.send(session, "")

    decisions = [HarnessEvents.to_dict(event) for event in events if event.type.startswith("tool_")]
    return {
        "refused": None,
        "ran": ran,
        "asked": [pending.tool for pending in response.pending_approvals],
        "events": [
            {
                "type": event["type"],
                "server": event["server"],
                "tool": event["tool"],
                "args_digest": event["args_digest"],
                "decided_by": event["decided_by"],
                "policy_revision": event["policy_revision"],
            }
            for event in decisions
        ],
    }


def test_has_rows() -> None:
    assert CORPUS["cases"]


@pytest.mark.parametrize("row", CORPUS["cases"], ids=[row["id"] for row in CORPUS["cases"]])
def test_row(row: dict[str, Any]) -> None:
    assert record(row) == row["expected"]
