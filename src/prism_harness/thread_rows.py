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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prism_harness.tool_arguments import usable_arguments

__all__ = [
    "REPEATED_CALL_ID",
    "REPLACED_BY_LATER_STEP",
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

    Rows written by 0.1.0 read in today's shape, and a step's tool result
    rows fold into one, consecutive or not (each result is replayed once, in
    the step's segment, the first such row's position). An approval leaves several in a row: the results of the
    calls a stopped step did run, the decisions, and the results written when
    the run resumes. Rows are never rewritten, so all of them stay stored; shown
    to a provider separately, a tool output would go twice. Folded, a result is
    keyed by its tool call id, tool name and arguments (never the id alone; see
    :func:`_fold`) and a decision by its approval id, and a later one replaces
    an earlier one in the earlier one's position.
    """
    view: list[dict[str, Any]] = []
    # The view row holding the current step's results: a step's result rows
    # that are not consecutive (a note between them, say) fold into the FIRST,
    # so each result is replayed exactly once, in its own segment.
    segment_results: int | None = None

    for row in rows:
        if is_replaced_row(row):
            # Answers an EARLIER step, written after a later one: folded in
            # here it would stand for the later step's calls that share an id.
            continue

        normalised = _normalise(row, view)

        if normalised is None:
            continue

        if normalised["type"] == "assistant" and normalised["tool_calls"]:
            segment_results = None

        if normalised["type"] == "tool_result" and segment_results is not None:
            view[segment_results] = _fold(view[segment_results], normalised)
        elif normalised["type"] == "tool_result" and view and view[-1]["type"] == "tool_result":
            view[-1] = _fold(view[-1], normalised)
            segment_results = len(view) - 1
        else:
            view.append(normalised)
            if normalised["type"] == "tool_result":
                segment_results = len(view) - 1

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
    missing id counts), or a ``tool_call_id`` or an ``approval_id`` among its
    approval requests (G-392).

    Such a step cannot say which decision answers which call, so none of its
    gated calls runs.
    """
    ids = [_text(call.get("id")) for call in _dicts(assistant.get("tool_calls"))]
    requests = _dicts(assistant.get("tool_approval_requests"))
    asked = [_text(request.get("tool_call_id")) for request in requests]
    approvals = [_text(request.get("approval_id")) for request in requests]
    return (
        len(set(ids)) != len(ids)
        or len(set(asked)) != len(asked)
        or len(set(approvals)) != len(approvals)
    )


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


def step_answers(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Every result and every decision stored for the last tool-calling turn
    (G-392, G-399).

    Read from EVERY tool result row after that turn, whatever sits between
    them: a host can record an assistant note, a user row or any other row
    while an approval is pending, and the results written after it are still
    this turn's. Reading only the first run of tool result rows would read a
    call that already ran as unanswered, and run it again.

    The results are in stored order and NOT folded by call id: folded, a
    result is keyed by its call id, so with a repeated id one call's result
    reads as another's. A decision is keyed by its approval id, and a later
    one replaces an earlier one, as the thread view folds them.
    """
    view: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}

    for row in rows:
        if is_replaced_row(row):
            continue

        normalised = _normalise(row, view)

        if normalised is None:
            continue

        if normalised["type"] == "assistant" and normalised["tool_calls"]:
            entries, decisions = [], {}
        elif normalised["type"] == "tool_result":
            entries.extend(normalised["tool_results"])
            for decision in normalised["tool_approval_responses"]:
                decisions[decision["approval_id"]] = decision

        if normalised["type"] == "tool_result" and view and view[-1]["type"] == "tool_result":
            view[-1] = _fold(view[-1], normalised)
        else:
            view.append(normalised)

    return entries, decisions


#: The result each gated call of an earlier parked step gets once a later
#: step with tool calls follows it (G-392), identical in every language.
REPLACED_BY_LATER_STEP = "Tool call was not run because a later step replaced it"

#: The key on the row holding those results, naming the step they answer by
#: the thread position of its assistant row. A harness-only field: a tool
#: controls its result text, never this key.
REPLACED_STEP = "harness_replaced_step"


#: The rows that supersede a parked step recorded before them (G-397).
SUPERSEDING = ("user", "assistant")


def is_replaced_row(row: Mapping[str, Any]) -> bool:
    return row.get("type") == "tool_result" and isinstance(row.get(REPLACED_STEP), dict)


def _thread_steps(
    messages: Sequence[tuple[int, Mapping[str, Any]]],
) -> tuple[list[tuple[int, dict[str, Any], list[dict[str, Any]]]], int]:
    """Every tool-calling step, by the thread position of its assistant row,
    with the results stored in its SEGMENT (the tool result rows after it, up
    to the next step with tool calls, and the rows written for it under
    ``harness_replaced_step``), and how many steps from the first are
    superseded (G-392, G-397, G-399)."""
    view: list[dict[str, Any]] = []
    steps: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    superseded = 0
    written: dict[int, list[dict[str, Any]]] = {}

    for position, row in messages:
        if is_replaced_row(row):
            target = row[REPLACED_STEP].get("position")
            if type(target) is int:
                normalised = _normalise(row, view)
                if normalised is not None:
                    written.setdefault(target, []).extend(normalised["tool_results"])
            continue

        normalised = _normalise(row, view)

        if normalised is None:
            continue

        if normalised["type"] in SUPERSEDING:
            superseded = len(steps)
        if normalised["type"] == "assistant" and normalised["tool_calls"]:
            steps.append((position, normalised, []))
        elif normalised["type"] == "tool_result" and steps:
            steps[-1][2].extend(_parsed_arguments(row, normalised["tool_results"]))

        if normalised["type"] == "tool_result" and view and view[-1]["type"] == "tool_result":
            view[-1] = _fold(view[-1], normalised)
        else:
            view.append(normalised)

    return [
        (position, assistant, [*entries, *written.get(position, [])])
        for position, assistant, entries in steps
    ], superseded


def _parsed_arguments(
    row: Mapping[str, Any], entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The entries with each result's arguments PARSED as a call's are (G-392).

    Another writer may store a result's arguments as a JSON string; the view
    reads those as none, while a call's stored as a string are parsed. Compared
    in two different forms, a result would never match its own call.
    """
    raw = _dicts(row.get("tool_results"))
    if len(raw) != len(entries):
        return entries
    return [
        {**entry, "args": usable_arguments(stored.get("args")) or {}}
        for entry, stored in zip(entries, raw, strict=True)
    ]


def _paired(
    steps: Sequence[tuple[int, dict[str, Any], list[dict[str, Any]]]],
) -> list[dict[int, dict[str, Any]]]:
    """Each stored result paired with the ONE call it answers, by step and
    call index (G-392, G-397).

    A result counts ONLY for the calls of the step whose segment it sits in,
    never for another step's. Inside the segment it ends the first unanswered
    call, in call order, with its id, tool name and parsed arguments; else the
    first with its id and tool name; else the first with its id alone (written
    without a name, by another writer, or with lossy arguments). So a finished
    call never reads as unanswered and never runs again, and an earlier step's
    unanswered gated calls get the replaced-step denial.

    An entry identical to one already paired in its own step is the same
    result restated by a later row, and pairs with nothing.
    """

    def same(call: Mapping[str, Any], entry: Mapping[str, Any], exact: int) -> bool:
        return (
            _text(call.get("id")) == _text(entry.get("tool_call_id"))
            and (exact < 1 or _text(call.get("name")) == _text(entry.get("tool_name")))
            and (exact < 2 or call.get("arguments") == entry.get("args"))
        )

    calls = [_dicts(assistant.get("tool_calls")) for _, assistant, _ in steps]
    paired: list[dict[int, dict[str, Any]]] = [{} for _ in steps]

    def first(step: int, entry: Mapping[str, Any], exact: int) -> int | None:
        return next(
            (
                i
                for i, call in enumerate(calls[step])
                if i not in paired[step] and same(call, entry, exact)
            ),
            None,
        )

    for step, (_, _, entries) in enumerate(steps):
        left: list[dict[str, Any]] = []
        for entry in entries:
            if any(dict(entry) == done for done in paired[step].values()):
                continue
            index = first(step, entry, 2)
            if index is None:
                left.append(dict(entry))
            else:
                paired[step][index] = dict(entry)
        for entry in left:
            if any(entry == done for done in paired[step].values()):
                continue
            index = first(step, entry, 1)
            if index is None:
                index = first(step, entry, 0)
            if index is not None:
                paired[step][index] = entry

    return paired


def last_step_results(
    messages: Sequence[tuple[int, Mapping[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """The results of the last tool-calling step, by call index (G-392)."""
    steps, _ = _thread_steps(messages)
    return _paired(steps)[-1] if steps else {}


def replaced_steps(
    messages: Sequence[tuple[int, Mapping[str, Any]]],
    needs_approval: Callable[[str], bool],
) -> list[tuple[int, list[dict[str, Any]]]]:
    """The gated calls with no result of their own of every SUPERSEDED
    tool-calling step, by the thread position of the step's assistant row
    (G-392, G-397).

    A step is superseded once a user row or any assistant row (an answer in
    text, or a later step with tool calls) is recorded after it. Only a step
    that is not is resolved, so a decision for a superseded one never runs
    anything, whenever it was made. Each such call gets one result saying a
    later step replaced it, so no gated call is left without an outcome. A
    call with a result in its own step's segment gets none; a result in
    another step's segment never counts for it.
    """
    steps, superseded = _thread_steps(messages)
    paired = _paired(steps)
    replaced: list[tuple[int, list[dict[str, Any]]]] = []
    for step, (position, assistant, _) in enumerate(steps[:superseded]):
        requested = {index for _, index in paired_requests(assistant) if index is not None}
        calls = [
            call
            for index, call in enumerate(assistant["tool_calls"])
            if index not in paired[step] and (index in requested or needs_approval(call["name"]))
        ]
        if calls:
            replaced.append((position, calls))
    return replaced


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
    """Two consecutive tool result rows as one (G-392).

    A later row's result for the SAME call replaces an earlier one, in the
    earlier one's position. The same call is the same call id, tool name and
    arguments, the n-th such result of a row replacing the n-th of the earlier
    one, never the call id alone: a provider may give two calls of one step the
    same id, and keyed by id one call's result would replace the other's.
    """
    results: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}

    for row in (earlier, later):
        seen: dict[tuple[str, str, str], int] = {}
        for entry in row["tool_results"]:
            same = (entry["tool_call_id"], entry["tool_name"], _canonical(entry["args"]))
            seen[same] = seen.get(same, 0) + 1
            results[(same[0], same[1], same[2], seen[same])] = entry

    for decision in [*earlier["tool_approval_responses"], *later["tool_approval_responses"]]:
        decisions[decision["approval_id"]] = decision

    return tool_result_row(list(results.values()), list(decisions.values()))


def _canonical(value: Any) -> str:
    """The arguments as one canonical string: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
