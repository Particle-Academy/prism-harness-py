"""A running call's outcome, and copies of a paused step (G-392, G-404, G-408).

1. An approved call that runs past the resolution lock's ttl keeps the lock:
   a second resume from inside the call is refused, and a step with tool
   calls the host records meanwhile neither denies the running call nor takes
   its result. Deterministic: the tool advances a fake clock itself.
2. A copy of the paused step whose tool calls are not objects (strings,
   numbers, null, a JSON string, an array of strings) is never read as a step
   by any reader: the call that was approved runs at most once and ends with
   exactly one outcome, and resuming writes nothing more.

Entered at the public seam; the model is scripted.
"""

from __future__ import annotations

import copy
import tempfile
import time
from typing import Any

import pytest
from test_runtime import CountingTool

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    Session,
    ToolRegistry,
    record_approval,
)

REPLACED = "Tool call was not run because a later step replaced it"
MODES = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo", "shout"],
                "max_steps": 8,
                "requires_approval": ["echo"],
            }
        },
    }
)


def a_session(lock_store: str) -> Session:
    durable = tempfile.mkdtemp(prefix="prism-harness-running-")
    ephemeral = tempfile.mkdtemp(prefix="prism-harness-running-lock-")
    harness = PrismHarness(
        drivers={
            "memory": MemorySessionStore,
            "files": lambda: FileSessionStore(durable),
            "lock-files": lambda: FileSessionStore(ephemeral),
        },
        stores={
            "ephemeral": "memory" if lock_store == "memory" else "lock-files",
            "durable": "files",
        },
    )
    session = harness.for_(Participant("User", 1)).session("support")
    session.using_mode("guarded").using_provider("anthropic").using_model("claude-sonnet-5-5")
    return session


def a_model(calls: list[LlmToolCall]) -> Any:
    state = {"asked": False}

    def client(_request: LlmRequest) -> LlmResponse:
        if not state["asked"]:
            state["asked"] = True
            return LlmResponse(text="", finish_reason="tool_calls", tool_calls=calls)
        return LlmResponse(text="Done.", finish_reason="stop")

    return client


def outcomes(session: Session) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for m in session.thread().messages():
        if m.message["type"] == "tool_result":
            for entry in m.message["tool_results"]:
                found.setdefault(entry["args"].get("value"), []).append(entry["result"])
    return found


@pytest.mark.parametrize("lock_store", ["memory", "files"])
def test_a_call_running_past_the_lock_ttl_keeps_the_lock_and_its_one_outcome(
    lock_store: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = a_session(lock_store)
    counts: dict[str, int] = {}
    refused: list[str] = []
    runtime_box: list[AgentRuntime] = []
    real_time, real_monotonic = time.time, time.monotonic

    class LongTool(CountingTool):
        def handle(self, args: dict[str, Any]) -> Any:
            if args.get("value") == "a":
                # The call outlives the lock's ttl (300 s by default).
                monkeypatch.setattr(time, "time", lambda: real_time() + 10_000)
                monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 10_000)
                session.thread().record(
                    [
                        {
                            "type": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "c1", "name": "echo", "arguments": {"value": "b"}}
                            ],
                            "tool_approval_requests": [
                                {
                                    "approval_id": "apr_later",
                                    "tool_call_id": "c1",
                                    "arguments_format": 1,
                                }
                            ],
                            "harness_tool_names": ["echo"],
                        }
                    ]
                )
                try:
                    runtime_box[0].send(session, "")
                except HarnessError as error:
                    refused.append(error.code)
            return super().handle(args)

    runtime = AgentRuntime(
        client=a_model([LlmToolCall("c1", "echo", {"value": "a"})]),
        modes=MODES,
        tools=ToolRegistry()
        .register(LongTool("echo", counts))
        .register(CountingTool("shout", counts)),
        config={"lock_wait": 0},
    )
    runtime_box.append(runtime)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    runtime.send(session, "")
    for _ in range(3):
        runtime.send(session, "")

    assert refused == ["session_locked"]
    found = outcomes(session)
    assert set(found["a"]) == {"echo:a"}
    assert REPLACED not in found["a"]
    # The later call is never credited with the running call's result.
    assert "echo:a" not in found.get("b", [])
    assert set(found["b"]) == {"No approval response provided"}
    assert counts == {"echo": 1}


SHAPES: dict[str, Any] = {
    "strings": lambda calls: [str(c) for c in calls],
    "numbers": lambda calls: [1, 2],
    "null": lambda calls: None,
    "json-string": lambda calls: '[{"id": "c1", "name": "echo"}]',
    "array-of-strings": lambda calls: ["c1"],
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("when", ["before-result", "after-result"])
@pytest.mark.parametrize("requests", [True, False], ids=["requests-kept", "requests-dropped"])
def test_a_copy_of_the_paused_step_with_unreadable_calls_is_never_a_step(
    shape: str, when: str, requests: bool
) -> None:
    session = a_session("memory")
    counts: dict[str, int] = {}
    runtime = AgentRuntime(
        client=a_model([LlmToolCall("c1", "echo", {"value": "a"})]),
        modes=MODES,
        tools=ToolRegistry()
        .register(CountingTool("echo", counts))
        .register(CountingTool("shout", counts)),
    )
    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    parked = next(
        m.message
        for m in session.thread().messages()
        if m.message["type"] == "assistant" and m.message.get("tool_calls")
    )
    duplicate = copy.deepcopy(parked)
    duplicate["tool_calls"] = SHAPES[shape](parked["tool_calls"])
    if not requests:
        duplicate["tool_approval_requests"] = []
    if when == "after-result":
        runtime.send(session, "")
    session.thread().record([duplicate])
    runtime.send(session, "")
    settled = len(session.thread().messages())
    for _ in range(3):
        runtime.send(session, "")

    found = outcomes(session)["a"]
    assert counts.get("echo", 0) <= 1
    assert len(set(found)) == 1
    assert (found[0] == "echo:a") == (counts.get("echo", 0) == 1)
    # Resuming writes no further result row.
    later = [
        m
        for m in session.thread().messages()[settled:]
        if m.message["type"] == "tool_result" and m.message["tool_results"]
    ]
    assert later == []
