"""Legacy narrowing is explicit; named mode ceilings always apply to every turn."""

from pathlib import Path

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
    ToolRegistry,
)


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "narrow,named,names,expected",
    [
        (True, False, ["outside"], None),
        (True, False, ["*"], ["inside"]),
        (True, False, [], []),
        (False, False, ["outside"], ["outside"]),
        (False, False, [], ["inside"]),
        (False, True, ["outside"], None),
        (False, True, ["*"], ["inside"]),
    ],
)
def test_explicit_legacy_narrowing_and_native_defaults(
    tmp_path: Path,
    structured: bool,
    narrow: bool,
    named: bool,
    names: list[str],
    expected: list[str] | None,
) -> None:
    store = FileSessionStore(tmp_path)
    harness = PrismHarness(
        drivers={"owned": lambda: store}, stores={"ephemeral": "owned", "durable": "owned"}
    )
    pending = harness.for_agent("support") if named else harness
    session = pending.for_(Participant("test", "narrowing")).session("tools")

    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

        def handle(self, args: dict[str, object]) -> str:
            raise AssertionError("No handler may run")

    tools = ToolRegistry().register(Tool("inside")).register(Tool("outside"))
    tools.for_agent("support").register(Tool("inside"), "support").register(
        Tool("outside"), "support"
    )
    modes = ModeRegistry({"default": "chat", "modes": {"chat": {"tools": ["inside"]}}})
    modes.for_agent("support").register("chat", {"tools": ["inside"]}, "support").set_default(
        "chat"
    )
    requests: list[list[str]] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append([tool.name for tool in request.tools])
        return LlmResponse(
            text="done", finish_reason="stop", structured={"answer": "done"}, cost_usd=0
        )

    runtime = AgentRuntime(client, modes, tools, narrow_requested_tools=narrow)

    def turn() -> None:
        if structured:
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
