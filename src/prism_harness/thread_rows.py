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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prism_harness.tool_arguments import usable_arguments

__all__ = [
    "REPEATED_CALL_ID",
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


def approval_tool_scope(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], bool] | None:
    """Stored approval authority, separate from the provider replay view."""
    decisions: dict[str, bool] = {}
    for row in reversed(thread_view(rows)):
        if row.get("type") == "tool_result":
            for response in row.get("tool_approval_responses", []):
                if isinstance(response.get("approval_id"), str):
                    decisions.setdefault(response["approval_id"], response.get("approved") is True)
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
    """
    return {
        _text(call.get("id"))
        for row in rows
        if row.get("type") == "assistant"
        for call in _dicts(row.get("tool_calls"))
        if usable_arguments(call.get("arguments")) is None
    }


#: The result every gated call of a step with a repeated call id is given
#: (G-392), identical in every language.
REPEATED_CALL_ID = "Tool call was not run because another call in this step has the same id"


def repeats_call_ids(assistant: Mapping[str, Any]) -> bool:
    """Whether a tool call id repeats among one step's calls (an empty or
    missing id counts), or a ``tool_call_id`` among its approval requests
    (G-392).

    Such a step cannot say which decision answers which call, so none of its
    gated calls runs.
    """
    ids = [_text(call.get("id")) for call in _dicts(assistant.get("tool_calls"))]
    asked = [
        _text(request.get("tool_call_id"))
        for request in _dicts(assistant.get("tool_approval_requests"))
    ]
    return len(set(ids)) != len(ids) or len(set(asked)) != len(asked)


def paired_requests(assistant: Mapping[str, Any]) -> list[tuple[dict[str, Any], int | None]]:
    """Each approval request of one step with the index of the call it was
    issued for, or None (G-392).

    Paired by position: a request takes the first call, in call order, that
    has its ``tool_call_id`` and no request yet. The harness issues requests
    in call order, so with repeated ids every call still gets its own request,
    never the last one issued under that id.
    """
    calls = _dicts(assistant.get("tool_calls"))
    taken: set[int] = set()
    pairs: list[tuple[dict[str, Any], int | None]] = []
    for request in _dicts(assistant.get("tool_approval_requests")):
        wanted = _text(request.get("tool_call_id"))
        index = next(
            (
                i
                for i, call in enumerate(calls)
                if i not in taken and _text(call.get("id")) == wanted
            ),
            None,
        )
        if index is not None:
            taken.add(index)
        pairs.append((request, index))
    return pairs


def step_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every result stored for the last tool-calling turn, in stored order and
    NOT folded by call id (G-392).

    These are the entries of the tool result rows the thread view folds into the
    one row answering that turn. Folded, a result is keyed by its call id, so
    with a repeated id one call's result reads as another's.
    """
    view: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    state = "none"  # none | waiting | collecting | done

    for row in rows:
        normalised = _normalise(row, view)

        if normalised is None:
            continue

        if normalised["type"] == "assistant" and normalised["tool_calls"]:
            entries, state = [], "waiting"
        elif normalised["type"] == "tool_result":
            if state in ("waiting", "collecting"):
                entries.extend(normalised["tool_results"])
                state = "collecting"
        elif state == "collecting":
            state = "done"

        if normalised["type"] == "tool_result" and view and view[-1]["type"] == "tool_result":
            view[-1] = _fold(view[-1], normalised)
        else:
            view.append(normalised)

    return entries


def paired_results(
    assistant: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Each stored result of one turn, by the index of the call it answers (G-392).

    Paired by position, never through a map keyed by call id: a result takes
    the first call, in call order, with its id and tool name and no result yet,
    preferring one whose arguments match. So a result for an ungated call never
    stands for a gated call that shares its id. An entry identical to one
    already paired is the same result restated by a later row, and pairs with
    nothing.

    A result whose tool name matches no such call (written without one, or by
    another writer) still ends the first unanswered call with its id, so a
    finished call never reads as unanswered and can never run again. With a
    repeated id the step runs none of its gated calls whichever call that is.
    """
    calls = _dicts(assistant.get("tool_calls"))
    paired: dict[int, dict[str, Any]] = {}

    for entry in entries:
        if any(dict(entry) == done for done in paired.values()):
            continue
        candidates = [
            i
            for i, call in enumerate(calls)
            if i not in paired
            and _text(call.get("id")) == _text(entry.get("tool_call_id"))
            and _text(call.get("name")) == _text(entry.get("tool_name"))
        ]
        exact = [i for i in candidates if calls[i].get("arguments") == entry.get("args")]
        same_id = [
            i
            for i, call in enumerate(calls)
            if i not in paired and _text(call.get("id")) == _text(entry.get("tool_call_id"))
        ]
        chosen = exact or candidates or same_id
        if chosen:
            paired[chosen[0]] = dict(entry)

    return paired


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
