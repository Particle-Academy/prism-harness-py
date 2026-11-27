"""The cross-language approval-pairing corpus (harness-approval-pairing, G-392).

Mirrors prism-harness-ts/test/approval-pairing-corpus.test.ts. Every row's
``expected`` was written by hand before any language ran. This runner drives
the real runtime offline with a fake model client: the first reply is the
row's calls, each under the row's id, and every later reply is the text
``done``. ``echo`` is gated by mode and ``free`` is not. Python has no batch
decision call, so each round records the listed decisions one by one with
``record_approval`` and then resumes once with an empty prompt; a decision the
harness refuses (``approval_not_pending``) is swallowed. ``decided`` (0.2.0)
is what the thread recorded for each approval, read from the stored rows.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
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
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    ToolCallInput,
    ToolRegistry,
    assistant_row,
    record_approval,
)

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "harness-approval-pairing.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


class Echo:
    name = "echo"

    def __init__(self, ran: list[int]) -> None:
        self.ran = ran

    def handle(self, args: dict[str, Any]) -> Any:
        self.ran.append(int(args["n"]))
        return "ok"


class Free:
    name = "free"

    def __init__(self) -> None:
        self.count = 0

    def handle(self, args: dict[str, Any]) -> Any:
        self.count += 1
        return "free ok"


def record(row: dict[str, Any], directory: Path) -> dict[str, Any]:
    ran: list[int] = []
    free = Free()
    tools = ToolRegistry().register(Echo(ran)).register(free)
    modes = ModeRegistry(
        {
            "default": "guarded",
            "modes": {
                "guarded": {
                    "system_prompt": "Careful.",
                    "tools": ["echo", "free"],
                    "max_steps": 6,
                    "requires_approval": ["echo"],
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
                    LlmToolCall(call["id"], call["name"], call["args"]) for call in row["calls"]
                ],
            )
        return LlmResponse(text="done", finish_reason="stop")

    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(str(directory))},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("corpus")
    runtime = AgentRuntime(client=client, modes=modes, tools=tools)

    pending = runtime.send(session, "go").pending_approvals
    # DECISION B (0.2.0): a row that supersedes the parked step, recorded by a
    # host (outside the run that owns the step), or a new message.
    supersede = row.get("supersede")
    if supersede == "assistant_tool_step":
        session.thread().record([assistant_row("", [ToolCallInput("later", "free", {})])])
    elif supersede == "assistant_text":
        session.thread().record([assistant_row("A later answer.", [])])
    elif supersede == "user":
        session.thread().record([{"type": "user", "content": "A later message."}])
    elif supersede == "new_message":
        runtime.send(session, "Something else.")
    for round_ in row["rounds"]:
        for index, approved in round_:
            try:
                record_approval(session, pending[index].id, approved)
            except HarnessError as error:
                # A decision the harness refuses is swallowed; the round still resumes.
                if error.code != "approval_not_pending":
                    raise
        runtime.send(session, "")

    # What the thread RECORDED for each approval the first step issued, in
    # issue order: the first answer, or None when none was recorded (0.2.0).
    recorded: dict[str, bool] = {}
    for entry in session.thread().messages():
        if entry.message.get("type") != "tool_result":
            continue
        for response in entry.message.get("tool_approval_responses", []):
            recorded.setdefault(response["approval_id"], response["approved"] is True)
    outcome: dict[str, Any] = {
        "approvals": len(pending),
        "free": free.count,
        "ran": sorted(ran),
        "decided": [recorded.get(approval.id) for approval in pending],
    }
    if "supersede" in row:
        # The result each call of the first step ended with, in call order:
        # its first result, or None.
        first: dict[str, str] = {}
        after = False
        for entry in session.thread().messages():
            message = entry.message
            if message.get("type") == "assistant" and message.get("tool_calls") and not after:
                after = True
                continue
            if after and message.get("type") == "tool_result":
                for result in message.get("tool_results", []):
                    first.setdefault(result["tool_call_id"], result["result"])
        outcome["outcomes"] = [first.get(call["id"]) for call in row["calls"]]
    return outcome


def test_has_rows() -> None:
    assert CORPUS["cases"]


@pytest.mark.parametrize("row", CORPUS["cases"], ids=[row["id"] for row in CORPUS["cases"]])
def test_records_the_corpus_row(row: dict[str, Any], tmp_path: Path) -> None:
    assert record(row, tmp_path) == row["expected"]
