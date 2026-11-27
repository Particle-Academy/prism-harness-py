from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from typing import Any, TypeVar
from unittest.mock import patch

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
    ToolRegistry,
    record_approval,
)
from prism_harness.thread import _record_in_run
from prism_harness.thread_rows import ToolCallInput, approval_tool_scope, assistant_row

R = TypeVar("R")


def exercise(
    *,
    requested: list[str] | None = None,
    decision: bool | None = None,
    structured: bool = False,
    ceiling: list[str] | None = None,
    ceiling_after: list[str] | None = None,
    seed: dict[str, Any] | None = None,
    late_after: bool = False,
    legacy_widening: bool = False,
    refresh_before_lock: bool = False,
) -> dict[str, Any]:
    directory = tempfile.mkdtemp(prefix="prism-approval-scope-")

    def open_session() -> Session:
        harness = PrismHarness(
            drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "memory", "durable": "files"},
        )
        return (
            harness.for_(Participant("User", 1))
            .session("scope")
            .using_mode("chat")
            .using_provider("synthetic")
            .using_model("offline")
        )

    counts = {"scope_safe": 0, "scope_ceiling": 0, "scope_late": 0}
    offers: list[list[str]] = []
    current_ceiling = ["scope_safe", "scope_ceiling"] if ceiling is None else ceiling
    late = False
    responses = (
        []
        if decision is None or seed is not None
        else [
            LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("safe-call", "scope_safe")],
            )
        ]
    ) + [
        LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall("ceiling-call", "scope_ceiling")],
        ),
        LlmResponse(text='{"ok":true}', structured={"ok": True}, finish_reason="stop"),
    ]

    def client(request: LlmRequest) -> LlmResponse:
        offers.append([tool.name for tool in request.tools])
        assert "harness_tool_names" not in json.dumps(request.messages)
        assert responses, "Unexpected extra native model call"
        return responses.pop(0)

    class InertTool:
        def __init__(self, name: str) -> None:
            self.name = name

        def handle(self, _args: dict[str, Any]) -> str:
            counts[self.name] += 1
            return "done"

    def runtime() -> AgentRuntime:
        tools = ToolRegistry()
        for name in ["scope_safe", "scope_ceiling", *(["scope_late"] if late else [])]:
            tools.register(InertTool(name))
        return AgentRuntime(
            client=client,
            tools=tools,
            allow_legacy_tool_list_widening=legacy_widening,
            modes=ModeRegistry(
                {
                    "default": "chat",
                    "modes": {
                        "chat": {
                            "tools": current_ceiling,
                            "max_steps": 4,
                            "requires_approval": ["scope_safe", "scope_ceiling"]
                            if refresh_before_lock
                            else ["scope_safe"],
                        }
                    },
                }
            ),
        )

    session = open_session()
    pending_id: str | None = None
    stored_scope: Any = None
    refusal: str | None = None
    try:
        if seed is not None:
            row = assistant_row(
                "",
                [ToolCallInput("safe-call", "scope_safe", {})],
                {},
                [{"approval_id": "approval-1", "tool_call_id": "safe-call", "arguments_format": 1}],
            )
            if "scope" in seed:
                row["harness_tool_names"] = seed["scope"]
            # As the runtime stores its own paused row (G-387 round 16: a
            # host-recorded row's harness_tool_names reads as absent).
            _record_in_run(session.thread(), [row], "seeded-run")
            pending_id = "approval-1"
        else:
            first = runtime().send(session, "A test turn.", requested)
            pending_id = first.pending_approvals[0].id if first.pending_approvals else None
            if decision is not None:
                assert len(first.pending_approvals) == 1
        if decision is not None:
            assert counts == {"scope_safe": 0, "scope_ceiling": 0, "scope_late": 0}
            rows = session.thread().messages()
            stored_scope = next(
                (
                    entry.message.get("harness_tool_names")
                    for entry in rows
                    if entry.message.get("type") == "assistant"
                ),
                None,
            )
            fresh = open_session()
            assert fresh is not session and fresh.key() == session.key()
            current_ceiling = current_ceiling if ceiling_after is None else ceiling_after
            late = late_after
            assert pending_id is not None
            record_approval(fresh, pending_id, decision)
            if refresh_before_lock:
                native_lock = fresh.lock
                scheduled = False

                def refreshed_lock(
                    callback: Callable[[Session], R],
                    ttl_seconds: float = 10,
                    wait_seconds: float = 5,
                ) -> R:
                    nonlocal scheduled
                    if not scheduled:
                        scheduled = True
                        row = assistant_row(
                            "",
                            [ToolCallInput("new-call", "scope_ceiling", {})],
                            {},
                            [
                                {
                                    "approval_id": "new-approval",
                                    "tool_call_id": "new-call",
                                    "arguments_format": 1,
                                }
                            ],
                        )
                        row["harness_tool_names"] = ["scope_safe"]
                        _record_in_run(
                            fresh.thread(),
                            [
                                {"type": "user", "content": "A concurrent narrowed turn."},
                                row,
                                {
                                    "type": "tool_result",
                                    "tool_results": [],
                                    "tool_approval_responses": [
                                        {
                                            "approval_id": "new-approval",
                                            "approved": True,
                                            "reason": None,
                                        }
                                    ],
                                },
                            ],
                            "concurrent-run",
                        )
                    return native_lock(callback, ttl_seconds, wait_seconds)

                with patch.object(fresh, "lock", new=refreshed_lock):
                    runtime().send(fresh, "")
                assert offers
                return {
                    "counts": counts,
                    "offers": offers,
                    "stored_scope": stored_scope,
                    "refusal": refusal,
                }
            if structured:
                runtime().send_structured(
                    fresh,
                    "",
                    {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    },
                )
            else:
                runtime().send(fresh, "")
    except HarnessError as failure:
        if failure.code != "tool_not_available":
            raise
        refusal = failure.code
    assert offers
    return {"counts": counts, "offers": offers, "stored_scope": stored_scope, "refusal": refusal}


def test_fresh_narrowed_offer_stays_closed() -> None:
    row = exercise(requested=["scope_safe"])
    assert row["counts"]["scope_ceiling"] == 0
    assert row["offers"][0] == ["scope_safe"]
    assert row["refusal"] == "tool_not_available"


def test_full_ceiling_positive_control_stays_reachable() -> None:
    row = exercise()
    assert row["counts"]["scope_ceiling"] == 1
    assert row["refusal"] is None


@pytest.mark.parametrize("decision", [True, False])
@pytest.mark.parametrize("structured", [True, False])
def test_durable_resume_does_not_widen(decision: bool, structured: bool) -> None:
    row = exercise(requested=["scope_safe"], decision=decision, structured=structured)
    assert row["counts"]["scope_safe"] == (1 if decision else 0)
    assert row["counts"]["scope_ceiling"] == 0
    assert row["stored_scope"] == ["scope_safe"]
    assert all("scope_ceiling" not in offer for offer in row["offers"])


def test_a_narrower_current_ceiling_wins() -> None:
    row = exercise(requested=["scope_safe"], decision=True, ceiling_after=[])
    assert row["counts"]["scope_safe"] == 0
    assert row["offers"][-1] == []


@pytest.mark.parametrize("decision", [True, False])
def test_legacy_decisions_fail_closed_with_a_constant_diagnostic(
    decision: bool, caplog: pytest.LogCaptureFixture
) -> None:
    row = exercise(seed={}, decision=decision)
    assert row["counts"]["scope_safe"] == (1 if decision else 0)
    assert row["counts"]["scope_ceiling"] == 0
    # The requested tools, approved or not (G-387 round 17, the reference's
    # rule): the resolution can answer a denied call as it answers a step the
    # runtime wrote; a denied call still never runs.
    assert row["offers"][0] == ["scope_safe"]
    assert [record.message for record in caplog.records] == [
        "Pending approval has no valid stored tool scope; offering the requested tools only."
    ]


def test_an_empty_stored_list_stays_empty() -> None:
    row = exercise(seed={"scope": []}, decision=True)
    assert row["counts"]["scope_safe"] == 0
    assert row["offers"][0] == []


def test_a_malformed_stored_wildcard_does_not_grant_authority() -> None:
    row = exercise(seed={"scope": ["*"]}, decision=False)
    assert row["counts"]["scope_ceiling"] == 0
    # Read as no stored offer: the requested tool only (G-387 round 17),
    # never the wildcard's whole mode; the denied call runs nothing.
    assert row["offers"][0] == ["scope_safe"]
    assert row["counts"]["scope_safe"] == 0


def test_a_wildcard_mode_freezes_actual_names() -> None:
    row = exercise(ceiling=["*"], decision=True, late_after=True)
    assert set(row["stored_scope"]) == {"scope_safe", "scope_ceiling"}
    assert row["stored_scope"] == row["offers"][0]
    assert "scope_late" not in row["offers"][-1]
    assert row["counts"]["scope_ceiling"] == 1


def test_explicit_legacy_widening_still_respects_current_resume_ceiling() -> None:
    row = exercise(requested=["scope_safe"], decision=True, ceiling_after=[], legacy_widening=True)
    assert row["counts"]["scope_safe"] == 0
    assert row["offers"][-1] == []


def test_completed_turn_does_not_reuse_a_scope() -> None:
    assert approval_tool_scope([assistant_row("", []), {"type": "user", "content": "new"}]) is None


def test_refreshed_approval_keeps_its_own_scope_after_waiting_for_the_real_lock() -> None:
    row = exercise(decision=True, refresh_before_lock=True)
    assert row["counts"]["scope_ceiling"] == 0
    assert row["offers"][-1] == ["scope_safe"]
