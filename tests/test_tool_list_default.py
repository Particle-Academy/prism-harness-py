"""Exercise the real turn entry points before the model receives any tools."""

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


@pytest.mark.parametrize("shape", ["send", "structured"])
@pytest.mark.parametrize(
    "names,expected,flags,named,authorize,wildcard_mode",
    [
        (["outside"], None, {}, False, False, False),
        (["inside", "outside"], None, {}, False, False, False),
        (["inside"], ["inside"], {}, False, False, False),
        (None, ["inside", "second"], {}, False, False, False),
        ([], [], {}, False, False, False),
        (["*"], ["inside", "second"], {}, False, False, False),
        (["*", "outside"], ["inside", "second"], {}, False, False, False),
        (["outside"], ["outside"], {}, False, False, True),
        (["outside"], ["outside"], {"allow_legacy_tool_list_widening": True}, False, False, False),
        (None, ["inside"], {}, False, True, False),
        (["outside"], [], {"allow_legacy_tool_list_widening": True}, False, True, False),
        (["outside"], None, {"allow_legacy_tool_list_widening": True}, True, False, False),
        (["inside"], ["inside"], {"allow_legacy_tool_list_widening": True}, True, False, False),
    ],
)
def test_actual_turn_tool_ceiling(
    tmp_path: Path,
    shape: str,
    names: list[str] | None,
    expected: list[str] | None,
    flags: dict[str, Any],
    named: bool,
    authorize: bool,
    wildcard_mode: bool,
) -> None:
    store = FileSessionStore(tmp_path)
    harness = PrismHarness(
        drivers={"owned": lambda: store}, stores={"ephemeral": "owned", "durable": "owned"}
    )
    session = (
        (harness.for_agent("support") if named else harness)
        .for_(Participant("test", "default-tools"))
        .session("probe")
    )
    handled: list[str] = []

    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

        def handle(self, args: dict[str, Any]) -> str:
            handled.append(self.name)
            raise AssertionError("No tool should execute in an offering probe")

    tools = ToolRegistry()
    for name in ["inside", "second", "outside"]:
        tools.register(Tool(name))
        tools.for_agent("support").register(Tool(name), "support")
    mode_tools = ["*"] if wildcard_mode else ["inside", "second"]
    modes = ModeRegistry({"default": "chat", "modes": {"chat": {"tools": mode_tools}}})
    modes.for_agent("support").register("chat", {"tools": mode_tools}, "support").set_default(
        "chat"
    )
    requests: list[list[str]] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append([offered.name for offered in request.tools])
        return LlmResponse(
            text="done", finish_reason="stop", structured={"answer": "done"}, cost_usd=0
        )

    authorizer = (
        ToolAuthorizer(
            enabled=True,
            offer=lambda _session, tool: tool.name == "inside",
            call=lambda _s, _t, _a: True,
        )
        if authorize
        else None
    )
    runtime = AgentRuntime(client, modes, tools, authorizer=authorizer, **flags)

    def turn() -> None:
        if shape == "structured":
            runtime.send_structured(session, "probe", {"type": "object"}, names)
        else:
            runtime.send(session, "probe", names)

    if expected is None:
        with pytest.raises(HarnessError) as failure:
            turn()
        assert failure.value.code == "tool_not_available"
        assert requests == []
    else:
        turn()
        assert requests == [expected]
    assert handled == []


@pytest.mark.parametrize(
    "flags",
    [
        {"narrow_requested_tools": False},
        {"narrow_requested_tools": False, "allow_legacy_tool_list_widening": True},
        {"narrow_requested_tools": True, "allow_legacy_tool_list_widening": True},
    ],
)
def test_old_or_conflicting_authority_configuration(flags: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError), match="allow_legacy_tool_list_widening"):
        AgentRuntime(
            lambda _request: LlmResponse(text="done", finish_reason="stop"),
            ModeRegistry({}),
            ToolRegistry(),
            **flags,
        )


@pytest.mark.parametrize("value", [None, 1, "true", [], {}])
def test_widening_option_requires_a_boolean(value: Any) -> None:
    with pytest.raises(TypeError, match="allow_legacy_tool_list_widening"):
        AgentRuntime(
            lambda _request: LlmResponse(text="done", finish_reason="stop"),
            ModeRegistry({}),
            ToolRegistry(),
            allow_legacy_tool_list_widening=value,
        )
