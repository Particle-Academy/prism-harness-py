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
    LlmToolCall,
    ModeRegistry,
    Participant,
    PrismHarness,
    ToolAuthorizer,
    ToolRegistry,
    record_approval,
)

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures/harness-tool-list-default.json").read_text(encoding="utf-8")
)


def test_vendored_tool_list_case_census() -> None:
    assert len(CORPUS["cases"]) == 14


@pytest.mark.parametrize("entry", ["send", "structured"])
@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_native_tool_list_corpus(tmp_path: Path, entry: str, case: dict[str, Any]) -> None:
    if "resumeApproval" in case:
        approval_resume(tmp_path, entry, case)
        return
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


def approval_resume(directory: Path, entry: str, case: dict[str, Any]) -> None:
    store = FileSessionStore(directory)
    harness = PrismHarness(
        drivers={"owned": lambda: store}, stores={"ephemeral": "owned", "durable": "owned"}
    )
    session = harness.for_(Participant("test", "corpus")).session("probe")
    handlers = {"inside": 0, "second": 0}

    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

        def handle(self, _args: dict[str, Any]) -> str:
            handlers[self.name] += 1
            return "done"

    tools = ToolRegistry().register(Tool("inside")).register(Tool("second"))
    modes = ModeRegistry(
        {
            "default": "chat",
            "modes": {
                "chat": {
                    "tools": ["inside", "second"],
                    "requires_approval": ["inside"],
                    "max_steps": 4,
                }
            },
        }
    )
    requests: list[list[str]] = []
    responses = [
        LlmResponse(
            text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("inside-call", "inside")]
        ),
        LlmResponse(
            text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("second-call", "second")]
        ),
        LlmResponse(text='{"answer":"done"}', finish_reason="stop", structured={"answer": "done"}),
    ]

    def client(request: LlmRequest) -> LlmResponse:
        requests.append([tool.name for tool in request.tools])
        assert "harness_tool_names" not in json.dumps(request.messages)
        assert responses, "Unexpected extra native model call"
        return responses.pop(0)

    first = AgentRuntime(client, modes, tools).send(session, "probe", case["names"])
    assert len(first.pending_approvals) == 1
    assert handlers == {"inside": 0, "second": 0}
    fresh = harness.for_(Participant("test", "corpus")).session("probe")
    assert fresh is not session and fresh.key() == session.key()
    record_approval(fresh, first.pending_approvals[0].id, case["resumeApproval"])
    error = None
    try:
        if entry == "structured":
            AgentRuntime(client, modes, tools).send_structured(fresh, "", {"type": "object"})
        else:
            AgentRuntime(client, modes, tools).send(fresh, "")
    except HarnessError as failure:
        error = {"code": failure.code}
    assert handlers == case["expectedHandlers"]
    assert len(requests) >= 2
    assert all(offer == case["expected"] for offer in requests)
    assert error == (None if "second" in case["expected"] else {"code": "tool_not_available"})
    actual = {"entry": entry, "requests": requests, "error": error, "handlers": handlers}
    if output := os.environ.get("PRISM_TOOL_LIST_RECORDING"):
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": case["id"], **actual}) + "\n")
    else:
        assert actual == case["result"]["py"][entry]
