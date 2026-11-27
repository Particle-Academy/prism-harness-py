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
    Session,
    ToolCallInput,
    ToolRegistry,
    assistant_row,
    record_approval,
    tool_result_entry,
    tool_result_row,
)
from prism_harness.thread_rows import ARGUMENTS_AS_SENT

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

    parked = runtime.send(session, "go")
    pending = parked.pending_approvals
    # DECISION B (0.2.0): a row that supersedes the parked step, recorded by a
    # host (outside the run that owns the step), or a new message.
    supersede = row.get("supersede")
    if supersede == "assistant_tool_step":
        session.thread().record([assistant_row("", [ToolCallInput("later", "free", {})])])
    elif supersede == "assistant_text":
        session.thread().record([assistant_row("A later answer.", [])])
    elif supersede == "user":
        session.thread().record([{"type": "user", "content": "A later message."}])
    elif supersede == "user_owning_run_id":
        # A caller passing the parked step's OWN run id: still outside.
        session.thread().record([{"type": "user", "content": "A later message."}], parked.run_id)
    elif supersede == "new_message":
        runtime.send(session, "Something else.")
    # SEGMENT OWNERSHIP (0.2.0, G-387 round 13b): a later step a host records
    # after the park, each echo call asking its own approval
    # (``apr_later_<k>``, k counting the step's echo calls), then ONE result
    # row a host records right after it.
    later = row.get("later")
    later_ids: list[str] = []
    if later is not None:
        requests: list[dict[str, Any]] = []
        for call in later["calls"]:
            if call["name"] == "echo":
                approval_id = f"apr_later_{len(later_ids)}"
                later_ids.append(approval_id)
                requests.append(
                    {
                        "approval_id": approval_id,
                        "tool_call_id": call["id"],
                        "arguments_format": ARGUMENTS_AS_SENT,
                    }
                )
        session.thread().record(
            [
                assistant_row(
                    "",
                    [
                        ToolCallInput(call["id"], call["name"], call["args"])
                        for call in later["calls"]
                    ],
                    approval_requests=requests,
                )
            ]
        )
        result = later["result"]
        session.thread().record(
            [
                tool_result_row(
                    [
                        tool_result_entry(
                            ToolCallInput(result["id"], result["name"], result["args"]),
                            result["result"],
                        )
                    ],
                    [
                        {"approval_id": later_ids[k], "approved": True, "reason": None}
                        for k in result["approved"]
                    ],
                )
            ]
        )
    # The later step's rounds first, while its approvals are pending, then the
    # row's rounds on the first step's approvals.
    rounds: list[tuple[list[Any], list[str]]] = [
        *((round_, later_ids) for round_ in (later or {}).get("rounds", [])),
        *((round_, [approval.id for approval in pending]) for round_ in row["rounds"]),
    ]
    refusals: list[str | None] = []
    for round_, ids in rounds:
        refused: str | None = None
        for index, approved in round_:
            try:
                record_approval(session, ids[index], approved)
            except HarnessError as error:
                # A decision the harness refuses is swallowed; the round still resumes.
                if error.code != "approval_not_pending":
                    raise
                refused = refused or str(error.code)
        refusals.append(refused)
        runtime.send(session, "")
    for _ in range((later or {}).get("resumes", 0)):
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
    if "supersede" in row or later is not None:
        # The result each call of a step ended with, in call order: its first
        # result IN ITS STEP'S SEGMENT (the rows after the step and before the
        # next assistant row with tool calls), or None (G-387 round 13b).
        found = segments(session)
        outcome["outcomes"] = segment_outcomes(row["calls"], found[0] if found else [])
        if later is not None:
            outcome["later_outcomes"] = segment_outcomes(
                later["calls"], found[1] if len(found) > 1 else []
            )
        # The code each round of decisions was refused with, or None, the
        # later step's rounds first. Python records decisions one by one (it
        # has no batch call): a round is refused when any of its decisions is.
        outcome["refusals"] = refusals
    return outcome


def segments(session: Session) -> list[list[dict[str, Any]]]:
    """The tool result entries of each step's segment, oldest first: one list
    per assistant row with tool calls, in thread order."""
    found: list[list[dict[str, Any]]] = []
    for entry in session.thread().messages():
        message = entry.message
        if message.get("type") == "assistant" and message.get("tool_calls"):
            found.append([])
        elif found and message.get("type") == "tool_result":
            found[-1].extend(message.get("tool_results", []))
    return found


def segment_outcomes(
    calls: list[dict[str, Any]], entries: list[dict[str, Any]]
) -> list[str | None]:
    """The first result each call got in its segment, in call order, or None.
    An entry is a call's by exact arguments, then id and tool name, then id:
    the first call so matched with no result yet; an entry whose match
    already has one is a repeat, and an entry that matches no call counts for
    nothing (the PHP reference's reading)."""
    first: list[str | None] = [None] * len(calls)
    for entry in entries:
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        for level in (3, 2, 1):
            matched = [at for at, call in enumerate(calls) if _matches(call, entry, args, level)]
            if not matched:
                continue
            open_ = [at for at in matched if first[at] is None]
            if open_:
                first[open_[0]] = str(entry["result"])
            break
    return first


def _matches(call: dict[str, Any], entry: dict[str, Any], args: Any, level: int) -> bool:
    """Level 3: id, tool name and arguments; 2: id and tool name; 1: id."""
    return (
        call["id"] == entry["tool_call_id"]
        and (level < 2 or call["name"] == entry["tool_name"])
        and (level < 3 or call["args"] == args)
    )


def test_has_rows() -> None:
    assert CORPUS["cases"]


@pytest.mark.parametrize("row", CORPUS["cases"], ids=[row["id"] for row in CORPUS["cases"]])
def test_records_the_corpus_row(row: dict[str, Any], tmp_path: Path) -> None:
    assert record(row, tmp_path) == row["expected"]
