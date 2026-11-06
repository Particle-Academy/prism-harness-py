"""Execute vendored authority rows through public send and structured turn entries."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    LlmRequest,
    LlmResponse,
    ModeRegistry,
    Participant,
    PrismHarness,
    ToolAuthorizer,
    ToolRegistry,
)

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures/harness-tool-list-default.json").read_text(encoding="utf-8")
)


def test_vendored_tool_list_case_census() -> None:
    assert len(CORPUS["cases"]) == 11


@pytest.mark.parametrize("entry", ["send", "structured"])
@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_native_tool_list_corpus(tmp_path: Path, entry: str, case: dict[str, Any]) -> None:
    store = FileSessionStore(tmp_path)
    harness = PrismHarness(
        drivers={"owned": lambda: store}, stores={"ephemeral": "owned", "durable": "owned"}
    )
    session = (
        (harness.for_agent("support") if case["named"] else harness)
        .for_(Participant("test", "corpus"))
        .session("probe")
    )
    handled: list[str] = []

    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

        def handle(self, args: dict[str, Any]) -> str:
            handled.append(self.name)
            raise AssertionError("Offering probes must not execute tools")

    tools = ToolRegistry()
    for name in ["inside", "second", "outside"]:
        tools.register(Tool(name))
        tools.for_agent("support").register(Tool(name), "support")
    ceiling = ["*"] if case["wildcardMode"] else ["inside", "second"]
    modes = ModeRegistry({"default": "chat", "modes": {"chat": {"tools": ceiling}}})
    modes.for_agent("support").register("chat", {"tools": ceiling}, "support").set_default("chat")
    requests: list[list[str]] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append([tool.name for tool in request.tools])
        return LlmResponse(
            text="done", finish_reason="stop", structured={"answer": "done"}, cost_usd=0
        )

    authorizer = (
        ToolAuthorizer(
            enabled=True, offer=lambda _s, tool: tool.name == "inside", call=lambda _s, _t, _a: True
        )
        if case["authorize"]
        else None
    )
    runtime = AgentRuntime(client, modes, tools, authorizer=authorizer)
    error = None
    try:
        if entry == "structured":
            runtime.send_structured(session, "probe", {"type": "object"}, case["names"])
        else:
            runtime.send(session, "probe", case["names"])
    except HarnessError as failure:
        error = {"code": failure.code}
    actual = {"entry": entry, "requests": requests, "error": error}
    if output := os.environ.get("PRISM_TOOL_LIST_RECORDING"):
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": case["id"], **actual}) + "\n")
    else:
        assert actual == case["result"]["py"][entry]
    if case["expected"] is None:
        assert error == {"code": "tool_not_available"}
        assert requests == []
    else:
        assert error is None
        assert requests == [case["expected"]]
    assert handled == []
