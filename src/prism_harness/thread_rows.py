"""The rows a run writes to a thread, in the shape the PHP reference stores.

prism-harness stores Prism's own messages: an assistant turn carries its tool
calls (arguments and provider ids included), its provider state and its
approval requests; a tool result turn carries every result of a step and any
approval decisions. These builders write exactly that, with the row's ``type``
beside it, so a thread written here reads the same as one written there.
prism-parity's ``harness-thread-rows`` corpus pins it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prism_harness.tool_arguments import usable_arguments

__all__ = [
    "ToolCallInput",
    "assistant_row",
    "thread_view",
    "tool_result_entry",
    "tool_result_row",
]


@dataclass(frozen=True)
class ToolCallInput:
    id: str
    name: str
    arguments: Mapping[str, Any]
    result_id: str | None = None
    reasoning_id: str | None = None
    reasoning_summary: Sequence[Any] | None = None


#: The marker on an approval request whose call was stored exactly as the model
#: sent it. 0.5.2 and earlier wrote none, and stored a call that a client handed
#: over as no arguments, which earlier cores also did for a bare number or
#: boolean on the wire, as ``{}``. Such an approval expires instead of running.
ARGUMENTS_AS_SENT = 1


def is_marked(request: Mapping[str, Any]) -> bool:
    """Whether an approval request carries the marker, as an integer and nothing else."""
    value = request.get("arguments_format")
    return type(value) is int and value == ARGUMENTS_AS_SENT


def assistant_row(
    content: str,
    tool_calls: Sequence[ToolCallInput],
    additional_content: Mapping[str, Any] | None = None,
    approval_requests: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "arguments": dict(call.arguments),
                "result_id": call.result_id,
                "reasoning_id": call.reasoning_id,
                "reasoning_summary": list(call.reasoning_summary)
                if call.reasoning_summary is not None
                else None,
            }
            for call in tool_calls
        ],
        "additional_content": dict(additional_content or {}),
        "tool_approval_requests": [dict(request) for request in approval_requests],
    }


def tool_result_entry(call: ToolCallInput, result: str) -> dict[str, Any]:
    return {
        "tool_call_id": call.id,
        "tool_name": call.name,
        "args": dict(call.arguments),
        "result": result,
        "tool_call_result_id": call.result_id,
        "artifacts": [],
    }


def tool_result_row(
    results: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_results": [dict(entry) for entry in results],
        "tool_approval_responses": [dict(decision) for decision in decisions],
    }


def thread_view(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The thread as the model is shown it.

    Rows written by 0.1.0 read in today's shape, and consecutive tool result
    rows fold into one. An approval leaves several in a row: the results of the
    calls a stopped step did run, the decisions, and the results written when
    the run resumes. Rows are never rewritten, so all of them stay stored; shown
    to a provider separately, a tool output would go twice. Folded, a result is
    keyed by its tool call id and a decision by its approval id, and a later one
    replaces an earlier one in the earlier one's position.
    """
    view: list[dict[str, Any]] = []

    for row in rows:
        normalised = _normalise(row, view)

        if normalised is None:
            continue

        if normalised["type"] == "tool_result" and view and view[-1]["type"] == "tool_result":
            view[-1] = _fold(view[-1], normalised)
        else:
            view.append(normalised)

    return view


def approval_run_steps(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Model steps the run paused at the transcript tail has already taken.

    None when nothing at the tail is waiting on an approval. A run that pauses
    for a person and is resumed is the SAME run, so its step limit continues
    rather than starting again. Counted from the stored rows (one assistant row
    per model step since the turn's user row), so the count survives a reload,
    a different worker and a deploy, with no extra state.
    """
    if approval_tool_scope(rows) is None:
        return None
    steps = 0
    for row in reversed(rows):
        if row.get("type") == "user":
            break
        if row.get("type") == "assistant":
            steps += 1
    return steps


def pending_approval_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The approval ids the run paused at the transcript tail is still waiting
    on: issued by the last assistant step and not yet answered (G-354).

    Read from the stored rows, newest first. A user turn, or an assistant step
    that asked nothing, ends the scope: nothing before it is pending. An id
    answered once is no longer pending.
    """
    answered: set[str] = set()
    for row in reversed(rows):
        kind = row.get("type")
        if kind == "tool_result":
            for response in _dicts(row.get("tool_approval_responses")):
                if isinstance(response.get("approval_id"), str):
                    answered.add(response["approval_id"])
            continue
        if kind == "user":
            return []
        if kind != "assistant":
            continue
        pending: list[str] = []
        for request in _dicts(row.get("tool_approval_requests")):
            approval_id = request.get("approval_id")
            if isinstance(approval_id, str) and approval_id and approval_id not in answered:
                pending.append(approval_id)
        return list(dict.fromkeys(pending))
    return []


def approval_tool_scope(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], bool] | None:
    """Stored approval authority, separate from the provider replay view."""
    # Which decision counts is read from the stored rows, first one standing;
    # the view's fold is for replay to a provider, not for authority.
    decisions = {
        approval_id: decision["approved"] is True
        for approval_id, decision in first_decisions(rows).items()
    }
    for row in reversed(thread_view(rows)):
        if row.get("type") == "tool_result":
            continue
        if row.get("type") == "user":
            return None
        if row.get("type") != "assistant":
            continue
        requests = row.get("tool_approval_requests", [])
        if not requests:
            return None
        ids = {request["approval_id"] for request in requests}
        stored = next(
            (
                candidate
                for candidate in reversed(rows)
                if candidate.get("type") == "assistant"
                and isinstance(candidate.get("tool_approval_requests"), list)
                and len(candidate["tool_approval_requests"]) == len(requests)
                and all(
                    isinstance(request, dict)
                    and isinstance(request.get("approval_id"), str)
                    and request["approval_id"] in ids
                    for request in candidate["tool_approval_requests"]
                )
            ),
            None,
        )
        scope = stored.get("harness_tool_names") if stored is not None else None
        if isinstance(scope, list) and all(
            isinstance(name, str) and name not in ("", "*") for name in scope
        ):
            return list(dict.fromkeys(scope)), False
        approved_calls = {
            request["tool_call_id"]
            for request in requests
            if decisions.get(request["approval_id"]) is True
        }
        names = [
            call["name"]
            for call in row.get("tool_calls", [])
            if call.get("id") in approved_calls
            and isinstance(call.get("name"), str)
            and call["name"] not in ("", "*")
        ]
        return list(dict.fromkeys(names)), True
    return None


def unusable_stored_calls(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """The ids of stored calls whose arguments cannot be used.

    Read from the rows as written, because the thread view has already replaced
    such arguments with none, and an approved call must not run on an empty set
    nobody sent.

    Read from the one turn being resumed -- the last stored assistant row with
    tool calls -- never the whole thread: providers that number calls per
    response reuse ids, and an older turn's unusable call must not refuse a
    newer call that carries the same id.
    """
    for row in reversed(rows):
        if row.get("type") == "assistant" and _dicts(row.get("tool_calls")):
            return {
                _text(call.get("id"))
                for call in _dicts(row.get("tool_calls"))
                if usable_arguments(call.get("arguments")) is None
            }
    return set()


def first_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The decisions recorded for the turn being resumed, by approval id.

    Read from the stored rows after the last assistant row with tool calls:
    ``tool_approval_responses`` on tool result rows, and 0.3.0's single
    ``tool_approval_response`` rows. The FIRST decision recorded for an id is
    the one that counts, here and in every reader that decides what runs; a
    later one for the same id is an answer to a question already answered.

    The thread VIEW still folds a later decision over an earlier one in the
    earlier one's position: that is what a provider is shown, pinned by the
    cross-language thread rows corpus, and it decides nothing.
    """
    start = next(
        (
            index
            for index in range(len(rows) - 1, -1, -1)
            if rows[index].get("type") == "assistant" and _dicts(rows[index].get("tool_calls"))
        ),
        None,
    )
    decisions: dict[str, dict[str, Any]] = {}
    if start is None:
        return decisions
    for row in rows[start + 1 :]:
        kind = row.get("type")
        if kind == "tool_result":
            responses = _dicts(row.get("tool_approval_responses"))
        elif kind == "tool_approval_response":
            responses = [dict(row)]
        else:
            continue
        for response in responses:
            if isinstance(response.get("approval_id"), str):
                decisions.setdefault(response["approval_id"], _decision(response))
    return decisions


def paused_approval_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The stored assistant row whose approvals the transcript tail is waiting
    on, or None.

    Newest first: results and decisions recorded since it are skipped; a user
    turn, or an assistant step that asked nothing, ends the search.
    """
    for row in reversed(rows):
        kind = row.get("type")
        if kind == "user":
            return None
        if kind != "assistant":
            continue
        return row if _dicts(row.get("tool_approval_requests")) else None
    return None


def carried_agent_spend(value: object) -> tuple[float, int]:
    """What a paused row's ``harness_agent_spend`` says its agent had spent.

    ``(cost_usd, unmetered_runs)``. A value that is not exactly a finite
    non-negative cost and a non-negative whole count is carried as one
    unmetered run, so a spend ceiling refuses (``budget_unpriced``) rather than
    starting again from nothing.
    """
    if isinstance(value, Mapping):
        cost, unmetered = value.get("cost_usd"), value.get("unmetered_runs")
        if (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost >= 0
            and isinstance(unmetered, int)
            and not isinstance(unmetered, bool)
            and unmetered >= 0
        ):
            return float(cost), unmetered
    return 0.0, 1


def ambiguous_calls(assistant: Mapping[str, Any]) -> set[str]:
    """The ids of the calls in one assistant turn whose approval expires rather than runs.

    A request for the call has no marker, and the call is stored as no
    arguments, which is also how 0.5.2 and earlier stored a call an earlier
    core had parsed from a bare number or boolean.

    Read from the one turn being resumed, never the whole thread: providers that
    number calls per response reuse ids, and an older turn's request must not
    decide a newer one. Pass the turn as the thread view gives it; a call whose
    stored arguments cannot be used is refused before this is asked.
    """
    unmarked = {
        _text(request.get("tool_call_id"))
        for request in _dicts(assistant.get("tool_approval_requests"))
        if not is_marked(request)
    }
    return {
        _text(call.get("id"))
        for call in _dicts(assistant.get("tool_calls"))
        if _text(call.get("id")) in unmarked
        and isinstance(call.get("arguments"), Mapping)
        and len(call["arguments"]) == 0
    }


def _normalise(row: Mapping[str, Any], view: list[dict[str, Any]]) -> dict[str, Any] | None:
    kind = row.get("type")

    if kind == "assistant":
        return assistant_row(
            row["content"] if isinstance(row.get("content"), str) else "",
            [
                ToolCallInput(
                    id=_text(call.get("id")),
                    name=_text(call.get("name")),
                    arguments=usable_arguments(call.get("arguments")) or {},
                    result_id=_nullable_text(call.get("result_id")),
                    reasoning_id=_nullable_text(call.get("reasoning_id")),
                    reasoning_summary=call["reasoning_summary"]
                    if isinstance(call.get("reasoning_summary"), list)
                    else None,
                )
                for call in _dicts(row.get("tool_calls"))
            ],
            row["additional_content"] if isinstance(row.get("additional_content"), dict) else {},
            [
                {
                    "approval_id": _text(request.get("approval_id")),
                    "tool_call_id": _text(request.get("tool_call_id")),
                    **({"arguments_format": ARGUMENTS_AS_SENT} if is_marked(request) else {}),
                }
                for request in _dicts(row.get("tool_approval_requests"))
            ],
        )

    if kind == "tool_result":
        if isinstance(row.get("tool_results"), list) or isinstance(
            row.get("tool_approval_responses"), list
        ):
            return tool_result_row(
                [
                    {
                        "tool_call_id": _text(entry.get("tool_call_id")),
                        "tool_name": _text(entry.get("tool_name")),
                        "args": entry["args"] if isinstance(entry.get("args"), dict) else {},
                        "result": _text(entry.get("result")),
                        "tool_call_result_id": _nullable_text(entry.get("tool_call_result_id")),
                        "artifacts": entry["artifacts"]
                        if isinstance(entry.get("artifacts"), list)
                        else [],
                    }
                    for entry in _dicts(row.get("tool_results"))
                ],
                [_decision(decision) for decision in _dicts(row.get("tool_approval_responses"))],
            )

        # 0.1.0: one row per call, with no arguments on it.
        return tool_result_row(
            [
                {
                    "tool_call_id": _text(row.get("tool_call_id")),
                    "tool_name": _text(row.get("name")),
                    "args": {},
                    "result": _text(row.get("result")),
                    "tool_call_result_id": None,
                    "artifacts": [],
                }
            ]
        )

    if kind == "tool_approval_request":
        # 0.1.0 kept the request in its own row, keyed by the CALL id. It
        # belongs to the assistant turn before it.
        assistant = next((c for c in reversed(view) if c["type"] == "assistant"), None)

        if assistant is not None:
            for approval in _dicts(row.get("approvals")):
                assistant["tool_approval_requests"].append(
                    {
                        "approval_id": _text(approval.get("id")),
                        "tool_call_id": _text(approval.get("id")),
                    }
                )

        return None

    if kind == "tool_approval_response":
        return tool_result_row([], [_decision(row)])

    return dict(row)


def _fold(earlier: Mapping[str, Any], later: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}

    for entry in [*earlier["tool_results"], *later["tool_results"]]:
        results[entry["tool_call_id"]] = entry

    for decision in [*earlier["tool_approval_responses"], *later["tool_approval_responses"]]:
        decisions[decision["approval_id"]] = decision

    return tool_result_row(list(results.values()), list(decisions.values()))


def _decision(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "approval_id": _text(value.get("approval_id")),
        "approved": value.get("approved") is True,
        "reason": _nullable_text(value.get("reason")),
    }


def _dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _text(value: object) -> str:
    if isinstance(value, str):
        return value

    return "" if value is None else json.dumps(value)


def _nullable_text(value: object) -> str | None:
    return value if isinstance(value, str) else None
