"""What a plan reads from the stored thread (G-360).

Read from the persisted rows only: a plan binds to these approval ids and to no
other, and the evidence that a person approved a call is the decision row the
host wrote, never anything the model sent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism_harness.thread_rows import (
    ended_approval_ids,
    paired_requests,
    pending_approval_ids,
    repeats_call_ids,
)
from prism_harness.tool_policy import args_digest

__all__ = [
    "approval_decisions",
    "approved_calls_awaiting_result",
    "pending_approval_calls",
]

Rows = Sequence[Mapping[str, Any]]


def _dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _digest(arguments: object) -> str:
    return args_digest(arguments if isinstance(arguments, dict) else {})


def _approval_call(rows: Rows, approval_id: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        if row.get("type") != "assistant":
            continue
        # Each request names its own call, paired by position (G-392).
        calls = _dicts(row.get("tool_calls"))
        for request, index in paired_requests(row):
            if request.get("approval_id") != approval_id or index is None:
                continue
            call = calls[index]
            if isinstance(call.get("name"), str):
                return {"name": call["name"], "arguments": call.get("arguments")}
    return None


def pending_approval_calls(rows: Rows) -> list[dict[str, str]]:
    """Every pending approval at the transcript tail, with the tool it was asked
    for and the digest of its stored arguments."""
    calls: list[dict[str, str]] = []
    for approval_id in pending_approval_ids(rows):
        call = _approval_call(rows, approval_id)
        if call is None:
            continue
        calls.append(
            {"id": approval_id, "name": call["name"], "digest": _digest(call["arguments"])}
        )
    return calls


def approval_decisions(rows: Rows) -> dict[str, bool]:
    """Every recorded decision, by approval id: True when approved. The first
    decision recorded for an id is the one that counts. A call a runner start
    ended as not run has no decision standing (G-387 round 10): it was never
    acted on, and reads as ended, never as approved."""
    ended = ended_approval_ids(rows)
    decided: dict[str, bool] = {}
    for row in rows:
        if row.get("type") != "tool_result":
            continue
        for response in _dicts(row.get("tool_approval_responses")):
            approval_id = response.get("approval_id")
            if isinstance(approval_id, str) and approval_id not in ended:
                decided.setdefault(approval_id, response.get("approved") is True)
    return decided


def approved_calls_awaiting_result(rows: Rows, name: str, digest: str) -> int:
    """How many calls of the turn paused at the transcript tail a person approved,
    for this tool and these arguments, that have no result yet.

    The evidence a self-gated plan tool requires before it applies anything.
    """
    decided: dict[str, bool] = {}
    finished: set[str] = set()
    for row in reversed(rows):
        kind = row.get("type")
        if kind == "tool_result":
            for result in _dicts(row.get("tool_results")):
                if isinstance(result.get("tool_call_id"), str):
                    finished.add(result["tool_call_id"])
            # Newest first, so the last write is the FIRST decision recorded:
            # the one that counts, as in approval_decisions().
            for response in reversed(_dicts(row.get("tool_approval_responses"))):
                if isinstance(response.get("approval_id"), str):
                    decided[response["approval_id"]] = response.get("approved") is True
            continue
        if kind != "assistant":
            return 0
        if repeats_call_ids(row):
            # No call of a step with a repeated id is approved (G-392).
            return 0
        approved = {approval_id for approval_id, yes in decided.items() if yes}
        approved_calls = {
            request["tool_call_id"]
            for request in _dicts(row.get("tool_approval_requests"))
            if isinstance(request.get("approval_id"), str)
            and request["approval_id"] in approved
            and isinstance(request.get("tool_call_id"), str)
        }
        count = 0
        for call in _dicts(row.get("tool_calls")):
            call_id = call.get("id")
            if (
                not isinstance(call_id, str)
                or call_id not in approved_calls
                or call_id in finished
                or call.get("name") != name
            ):
                continue
            if _digest(call.get("arguments")) == digest:
                count += 1
        return count
    return 0
