"""The cross-language thread-rows corpus from prism-parity.

The rows this port stores for a run, and what it replays them as. A PHP app and
a Python agent can share a session; if the rows differed, one would resume the
other's approval against the wrong call, or replay a conversation a provider
refuses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prism_harness import (
    ToolCallInput,
    assistant_row,
    ended_by_start_row,
    thread_view,
    tool_result_row,
)
from prism_harness.thread_rows import ended_approval_ids, paired_requests

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "harness-thread-rows.json").read_text(encoding="utf-8")
)


def rows_for(case: dict[str, Any]) -> list[dict[str, Any]]:
    """The same conversion prism-parity's recorder makes."""
    if case.get("read") == "approval_states":
        # Which approvals read ended by a runner start (G-387 round 13), read
        # from the stored rows by the harness's own reader, one entry per id
        # asked about, in order.
        ended = ended_approval_ids(case["stored"])
        return [
            {"approval_id": approval, "ended_by_start": approval in ended}
            for approval in case["approvals"]
        ]

    if "fold" in case:
        return thread_view(case["fold"])

    spec = case["input"]

    if spec["write"] == "ended_by_start":
        # The row a runner start's task row closes a child's pending calls
        # with (G-387): the harness's own writer, handed the calls and each
        # call's own approval, paired as the runtime pairs the parked step.
        step = {
            "tool_calls": spec["tool_calls"],
            "tool_approval_requests": spec["approval_requests"],
        }
        own: dict[int, str] = {
            index: request["approval_id"]
            for request, index in paired_requests(step)
            if index is not None
        }
        return [
            ended_by_start_row(
                [
                    ToolCallInput(
                        id=call["id"],
                        name=call["name"],
                        arguments=call["arguments"],
                        result_id=call.get("result_id"),
                    )
                    for call in spec["tool_calls"]
                ],
                [own.get(index) for index in range(len(spec["tool_calls"]))],
            )
        ]

    if spec["write"] == "assistant":
        return [
            assistant_row(
                spec["content"],
                [
                    ToolCallInput(
                        id=call["id"],
                        name=call["name"],
                        arguments=call["arguments"],
                        result_id=call.get("result_id"),
                        reasoning_id=call.get("reasoning_id"),
                        reasoning_summary=call.get("reasoning_summary"),
                    )
                    for call in spec["tool_calls"]
                ],
                spec["additional_content"],
                spec["approval_requests"],
            )
        ]

    return [
        tool_result_row(
            [
                {
                    "tool_call_id": result["tool_call_id"],
                    "tool_name": result["tool_name"],
                    "args": result["args"],
                    "result": result["result"],
                    "tool_call_result_id": result.get("tool_call_result_id"),
                    "artifacts": [],
                }
                for result in spec["results"]
            ],
            spec["decisions"],
        )
    ]


def test_is_the_whole_suite_not_a_subset_someone_trimmed_to_green() -> None:
    assert len(CORPUS["cases"]) == 15


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_stores_and_replays_the_rows_the_corpus_records_for_this_port(case: dict[str, Any]) -> None:
    produced = json.dumps(rows_for(case), ensure_ascii=False, separators=(",", ":"))

    assert produced == case["rows"]["py"]


def test_agrees_with_the_reference_and_the_typescript_port_on_every_row() -> None:
    for case in CORPUS["cases"]:
        assert [case["rows"]["ts"], case["rows"]["py"]] == [case["rows"]["php"]] * 2, case["id"]
        assert case["agrees"] is True, case["id"]
