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
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.thread_rows import (
    answering_step,
    bound_step,
    call_results,
    ended_approval_ids,
    new_results,
    owned_rows,
    paired_requests,
    pending_approval_ids,
    placed_view,
    started_row,
    step_calls,
)

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "harness-thread-rows.json").read_text(encoding="utf-8")
)


def rows_for(case: dict[str, Any]) -> list[Any]:
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

    if case.get("read") == "call_outcomes":
        # 0.6.0 (G-387 round 16): whether each call of the step at 1-based
        # ``step`` has an outcome among the rows it owns, paired as every
        # reader pairs results.
        at = case["step"] - 1
        answered = call_results(step_calls(case["stored"][at]), owned_rows(case["stored"], at))
        return [index in answered for index in range(len(step_calls(case["stored"][at])))]

    if case.get("read") == "approval_reads":
        # 0.6.0: what each approval reads: pending, and the decision that
        # stands (true, false, or none).
        stored = case["stored"]
        pending = pending_approval_ids(stored)
        decided = approval_decisions(stored)
        return [
            {
                "approval_id": approval,
                "pending": approval in pending,
                "decision": decided.get(approval),
            }
            for approval in case["approvals"]
        ]

    if "fold" in case:
        return thread_view(case["fold"])

    if "place" in case:
        # 0.5.1 (G-387 round 15): the rows AS STORED, ``harness: true`` on a
        # row the harness wrote, positions 1-based by index; what a provider
        # is sent (the harness's own placement, then the fold).
        return placed_view(case["place"])

    spec = case["input"]

    if spec["write"] == "answer":
        # 0.6.0: the answer row the resolver writes for the step at
        # ``position``: only the results the step does not hold yet, the
        # decisions, and the step's mark when a result is kept.
        stored = spec["stored"]
        at = spec["position"] - 1
        row = tool_result_row(new_results(stored, at, spec["results"]), spec["decisions"])
        if not row["tool_results"] and not row["tool_approval_responses"]:
            return []
        return [answering_step(row, spec["position"]) if row["tool_results"] else row]

    if spec["write"] == "started":
        # The write-ahead row the resolver records before it runs a call.
        return [
            started_row(
                {"id": spec["tool_call_id"], "name": spec["tool_name"]},
                spec["approval_id"],
                spec["position"],
            )
        ]

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
        # ``bind`` (0.5.1): the step as the thread stores it, each request
        # bound to its call (``harness_call``).
        bind = bound_step if spec.get("bind") else (lambda row: row)
        return [
            bind(
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
    assert len(CORPUS["cases"]) == 35


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_stores_and_replays_the_rows_the_corpus_records_for_this_port(case: dict[str, Any]) -> None:
    produced = json.dumps(rows_for(case), ensure_ascii=False, separators=(",", ":"))

    assert produced == case["rows"]["py"]


def test_agrees_with_the_reference_and_the_typescript_port_on_every_row() -> None:
    for case in CORPUS["cases"]:
        assert [case["rows"]["ts"], case["rows"]["py"]] == [case["rows"]["php"]] * 2, case["id"]
        assert case["agrees"] is True, case["id"]
