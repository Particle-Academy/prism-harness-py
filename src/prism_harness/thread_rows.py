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
from typing import Any, Protocol

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


class StoredRow(Protocol):
    """A stored row as the readers here need it (a :class:`ThreadMessage`)."""

    @property
    def position(self) -> int: ...

    @property
    def message(self) -> dict[str, Any]: ...

    @property
    def run_id(self) -> str | None: ...

    @property
    def harness(self) -> bool: ...


@dataclass(frozen=True)
class _Row:
    position: int
    message: dict[str, Any]
    run_id: str | None = None
    harness: bool = False


def _stored(
    rows: Sequence[Mapping[str, Any]], stored: Sequence[StoredRow] | None
) -> Sequence[StoredRow]:
    """The rows with what was stored beside them. Rows given alone are read as
    a host's (no run, not the harness's), position by position from 1."""
    if stored is not None:
        return stored
    return [_Row(index + 1, dict(row)) for index, row in enumerate(rows)]


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


def thread_view(
    rows: Sequence[Mapping[str, Any]], stored: Sequence[StoredRow] | None = None
) -> list[dict[str, Any]]:
    """The thread as the model is shown it.

    Given ``stored`` (the :class:`ThreadMessage` rows the rows came from), it is
    the runtime's replay: marks the harness itself recorded are honoured, so a
    replaced step's denial row is hidden and a result row is shown in the place
    of the step it answers. Given the rows alone it is the RAW view: every row
    is read as a host's, so no mark is honoured.

    Rows written by 0.1.0 read in today's shape, and a step's tool result
    rows fold into one, consecutive or not (each result is replayed once, in
    the step's segment, the first such row's position). An approval leaves several in a row: the results of the
    calls a stopped step did run, the decisions, and the results written when
    the run resumes. Rows are never rewritten, so all of them stay stored; shown
    to a provider separately, a tool output would go twice. Folded, a result is
    keyed by its tool call id, tool name and arguments (never the id alone; see
    :func:`_fold`), and a later one replaces an earlier one in the earlier
    one's position. A decision is keyed by its approval id, and the
    FIRST decision for an approval stands (G-401).
    """
    view: list[dict[str, Any]] = []
    # The view row holding the current step's results: a step's result rows
    # that are not consecutive (a note between them, say) fold into the FIRST,
    # so each result is replayed exactly once, in its own segment.
    segment_results: int | None = None
    # A step's assistant row and its results row in the view, by the step's
    # thread position, for result rows marked with the step they answer.
    current: int | None = None
    steps: dict[int, int] = {}
    answers: dict[int, int] = {}
    entries = _stored(rows, stored)
    marks = _valid_marks(entries)

    for index, row in enumerate(rows):
        if is_replaced_row(row) and index in marks:
            # Answers an EARLIER step, written after a later one: folded in
            # here it would stand for the later step's calls that share an id.
            continue

        normalised = _normalise(row, view)

        if normalised is None:
            continue

        if normalised["type"] == "assistant" and normalised["tool_calls"]:
            segment_results = None
            current = entries[index].position
            view.append(normalised)
            steps[current] = len(view) - 1
            continue

        anchor = marks.get(index)
        if (
            normalised["type"] == "tool_result"
            and anchor is not None
            and anchor != current
            and anchor in steps
        ):
            # Resolution results for an EARLIER step that landed after a later
            # one: shown in that step's place, as that step's (G-392).
            if anchor in answers:
                view[answers[anchor]] = _fold(view[answers[anchor]], normalised)
            else:
                at = steps[anchor] + 1
                view.insert(at, normalised)
                steps = {k: v + (v >= at) for k, v in steps.items()}
                answers = {k: v + (v >= at) for k, v in answers.items()}
                if segment_results is not None and segment_results >= at:
                    segment_results += 1
                answers[anchor] = at
            continue

        if normalised["type"] == "tool_result" and segment_results is not None:
            view[segment_results] = _fold(view[segment_results], normalised)
        elif normalised["type"] == "tool_result" and view and view[-1]["type"] == "tool_result":
            view[-1] = _fold(view[-1], normalised)
            segment_results = len(view) - 1
        else:
            view.append(normalised)
            if normalised["type"] == "tool_result":
                segment_results = len(view) - 1
        if normalised["type"] == "tool_result" and current is not None:
            answers[current] = segment_results if segment_results is not None else len(view) - 1

    return view


def approval_run_steps(
    rows: Sequence[Mapping[str, Any]], stored: Sequence[StoredRow] | None = None
) -> int | None:
    """Model steps the run paused at the transcript tail has already taken.

    An empty prompt continues the turn: a run that pauses for a person and is
    resumed is the SAME run, so its step limit continues rather than starting
    again, and a resume that ended the run out of steps leaves it out of
    steps, however many empty sends follow. None when the turn has no step yet.
    Counted from the stored rows (one assistant row per model step since the
    turn's user row), so the count survives a reload, a different worker and a
    deploy, with no extra state.
    """
    steps = 0
    for row in reversed(rows):
        if row.get("type") == "user":
            break
        if row.get("type") == "assistant":
            steps += 1
    return steps or None


def approval_tool_scope(
    rows: Sequence[Mapping[str, Any]], stored: Sequence[StoredRow] | None = None
) -> tuple[list[str], bool] | None:
    """Stored approval authority, separate from the provider replay view.

    The tools the paused step was offered, for the step a resume resolves:
    the last step with tool calls, when it is not superseded and still waits
    on at least one of its approval requests. None otherwise.
    """
    entries = _stored(rows, stored)
    steps, superseded = _thread_steps(entries)
    if not steps or superseded[-1]:
        return None
    position, assistant, _ = steps[-1]
    own = own_approvals(entries)
    requests = [r for r in assistant.get("tool_approval_requests", []) if r["approval_id"] in own]
    if not requests:
        return None
    results = _paired(steps)[-1]
    if all(
        index is not None and index in results
        for request, index in paired_requests(assistant)
        if request["approval_id"] in own
    ):
        return None
    raw = next(entry.message for entry in entries if entry.position == position)
    scope = raw.get("harness_tool_names")
    if isinstance(scope, list) and all(
        isinstance(name, str) and name not in ("", "*") for name in scope
    ):
        return list(dict.fromkeys(scope)), False
    _, decisions = step_answers(rows, stored)
    approved_calls = {
        request["tool_call_id"]
        for request in requests
        if decisions.get(request["approval_id"], {}).get("approved") is True
    }
    names = [
        call["name"]
        for call in assistant.get("tool_calls", [])
        if call.get("id") in approved_calls
        and isinstance(call.get("name"), str)
        and call["name"] not in ("", "*")
    ]
    return list(dict.fromkeys(names)), True


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
    rows: Sequence[Mapping[str, Any]], stored: Sequence[StoredRow] | None = None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Every result and every decision stored for the last tool-calling turn
    (G-392, G-399).

    Read from EVERY tool result row after that turn, whatever rows a host
    recorded between them while an approval was pending.

    The results are in stored order and NOT folded by call id: folded, a
    result is keyed by its call id, so with a repeated id one call's result
    reads as another's. A decision is keyed by its approval id, and the FIRST
    one for an approval stands, as the thread view folds them (G-401).
    """
    view: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    marks = _valid_marks(_stored(rows, stored))

    for index, row in enumerate(rows):
        if is_replaced_row(row) and index in marks:
            continue

        normalised = _normalise(row, view)

        if normalised is None:
            continue

        if normalised["type"] == "assistant" and normalised["tool_calls"]:
            entries, decisions = [], {}
        elif normalised["type"] == "tool_result":
            entries.extend(normalised["tool_results"])
            for decision in normalised["tool_approval_responses"]:
                decisions.setdefault(decision["approval_id"], decision)

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


#: The key on a resolution row naming the step it answers, by the thread
#: position of the step's assistant row (G-392). A harness-only field: a tool
#: controls its result text, never this key. Wherever such a row lands (a step
#: with tool calls may be written while an approved call runs), its results
#: count for that step.
ANSWERS_STEP = "harness_answers_step"


def is_replaced_row(row: Mapping[str, Any]) -> bool:
    return row.get("type") == "tool_result" and isinstance(row.get(REPLACED_STEP), dict)


def _mark(row: Mapping[str, Any]) -> int | None:
    """The step a tool result row's mark names, as written (unchecked)."""
    if row.get("type") != "tool_result":
        return None
    mark = row.get(ANSWERS_STEP)
    if not isinstance(mark, dict):
        mark = row.get(REPLACED_STEP)
    position = mark.get("position") if isinstance(mark, dict) else None
    return position if type(position) is int else None


def _valid_marks(entries: Sequence[StoredRow]) -> dict[int, int]:
    """The rows (by index) whose mark is HONOURED, with the step it names.

    Only a row the harness itself recorded, whose mark names an EARLIER row
    that is a step with tool calls holding a call with each of the row's
    result's id and tool. Any other mark is ignored, and its row counts for
    the step whose segment holds it, failing closed.
    """
    by_position = {entry.position: entry for entry in entries}
    valid: dict[int, int] = {}
    for index, entry in enumerate(entries):
        target = _mark(entry.message)
        if not entry.harness or target is None or target >= entry.position:
            continue
        step = by_position.get(target)
        if step is None or step.message.get("type") != "assistant":
            continue
        calls = {
            (_text(call.get("id")), _text(call.get("name")))
            for call in _dicts(step.message.get("tool_calls"))
        }
        results = _dicts(entry.message.get("tool_results"))
        started = entry.message.get(STARTED_KEY)
        if isinstance(started, dict):
            results = [
                *results,
                {
                    "tool_call_id": started.get("tool_call_id"),
                    "tool_name": started.get("tool_name"),
                },
            ]
        if not calls or not all(
            (_text(r.get("tool_call_id")), _text(r.get("tool_name"))) in calls for r in results
        ):
            continue
        valid[index] = target
    return valid


def superseded_at(entries: Sequence[StoredRow], index: int) -> int:
    """The index of the row that SUPERSEDES the step with tool calls at
    ``index``, or -1 while it is not superseded (G-397):

    - any later assistant row WITH tool calls, always;
    - a later assistant row without tool calls, or a user row, recorded
      OUTSIDE the runs that own the step: the run the harness recorded it in
      and the runs the harness recorded its results in. Rows the runtime
      writes inside those runs (a resume's answer) never supersede; a row
      recorded through the public ``Thread.record`` does, whatever run id it
      was given.

    Rows of any other kind (a system row, a tool result row) never supersede.
    """

    def inside(entry: StoredRow) -> str | None:
        return entry.run_id if entry.harness and isinstance(entry.run_id, str) else None

    owners: set[str] = set()
    own = inside(entries[index])
    if own is not None:
        owners.add(own)
    for at in range(index + 1, len(entries)):
        entry = entries[at]
        message = entry.message
        run = inside(entry)
        if message.get("type") == "assistant" and _dicts(message.get("tool_calls")):
            return at
        if (
            message.get("type") == "tool_result"
            and run is not None
            and _dicts(message.get("tool_results"))
        ):
            owners.add(run)
        if message.get("type") in SUPERSEDING and (run is None or run not in owners):
            return at
    return -1


def _thread_steps(
    messages: Sequence[StoredRow],
) -> tuple[list[tuple[int, dict[str, Any], list[dict[str, Any]]]], list[bool]]:
    """Every tool-calling step, by the thread position of its assistant row,
    with the results stored in its SEGMENT (the tool result rows after it, up
    to the next step with tool calls, and the rows written for it under
    ``harness_replaced_step``), and whether each is superseded
    (G-392, G-397, G-399)."""
    view: list[dict[str, Any]] = []
    steps: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    indexes: list[int] = []
    written: dict[int, list[dict[str, Any]]] = {}
    marks = _valid_marks(messages)

    for at, stored in enumerate(messages):
        position, row = stored.position, stored.message
        if is_replaced_row(row) and at in marks:
            normalised = _normalise(row, view)
            if normalised is not None:
                written.setdefault(marks[at], []).extend(_occurrences(normalised["tool_results"]))
            continue

        normalised = _normalise(row, view)

        if normalised is None:
            continue

        anchor = marks.get(at)
        if anchor is not None and normalised["type"] == "tool_result":
            written.setdefault(anchor, []).extend(
                _occurrences(_parsed_arguments(row, normalised["tool_results"]))
            )
        if normalised["type"] == "assistant" and normalised["tool_calls"]:
            steps.append((position, normalised, []))
            indexes.append(at)
        elif normalised["type"] == "tool_result" and steps and anchor is None:
            steps[-1][2].extend(_occurrences(_parsed_arguments(row, normalised["tool_results"])))

        if normalised["type"] == "tool_result" and view and view[-1]["type"] == "tool_result":
            view[-1] = _fold(view[-1], normalised)
        else:
            view.append(normalised)

    return [
        (position, assistant, [*entries, *written.get(position, [])])
        for position, assistant, entries in steps
    ], [superseded_at(messages, at) != -1 for at in indexes]


#: The key an entry carries, inside this module only, for its occurrence among
#: identical entries of its own row (G-392).
_OCCURRENCE = "_harness_occurrence"


def _occurrences(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each entry with its occurrence among IDENTICAL entries of its row.

    Two identical calls in one step (same id, tool and arguments) get two
    identical results in one row. Counted per occurrence, the second is the
    second call's own result; a later row restating both restates the first
    and the second occurrence, and pairs with nothing.
    """
    seen: list[dict[str, Any]] = []
    tagged: list[dict[str, Any]] = []
    for entry in entries:
        plain = {k: v for k, v in entry.items() if k != _OCCURRENCE}
        tagged.append({**plain, _OCCURRENCE: sum(1 for done in seen if done == plain)})
        seen.append(plain)
    return tagged


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

    An entry identical to one already paired in its own step, AND the same
    occurrence among identical entries of its row, is the same result restated
    by a later row, and pairs with nothing. Two identical calls' results are
    the first and the second occurrence, and each pairs with its own call.
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

    # The occurrence decided restatement; the results handed back are plain.
    for results in paired:
        for index, entry in results.items():
            results[index] = {k: v for k, v in entry.items() if k != _OCCURRENCE}
    return paired


#: What a call the harness recorded as started, with no result, reads: it may
#: have run, and it is never run again (G-410), identical in every language.
INTERRUPTED = "Tool call was interrupted, it may have run"

#: What an approved call reads when, at the moment it would start, it is no
#: longer the call its standing decision approves, or no longer open (G-410).
NOT_OPEN = "Not run: this call is not the one that was approved, or it is no longer open."

#: The key on the row the harness records just BEFORE it runs one approved
#: call: ``{"tool_call_id", "tool_name", "approval_id"}`` (G-410).
STARTED_KEY = "harness_started"


def started_row(
    tool_call_id: str, tool_name: str, approval_id: str, position: int
) -> dict[str, Any]:
    """The write-ahead row for one approved call, keys in the shared order:
    an empty result row, then ``harness_started``, then ``harness_answers_step``
    (the 1-based position of the step) as the LAST key."""
    return {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [],
        STARTED_KEY: {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "approval_id": approval_id,
        },
        ANSWERS_STEP: {"position": position},
    }


def started_calls(messages: Sequence[StoredRow]) -> set[tuple[int, str, str]]:
    """The calls the harness recorded as started, by (step position, call id,
    tool name): only a started row the harness itself recorded, whose mark
    names an earlier step with tool calls holding that call (G-410)."""
    marks = _valid_marks(messages)
    started: set[tuple[int, str, str]] = set()
    for index, entry in enumerate(messages):
        mark = entry.message.get(STARTED_KEY)
        if index in marks and isinstance(mark, dict):
            started.add(
                (marks[index], _text(mark.get("tool_call_id")), _text(mark.get("tool_name")))
            )
    return started


def own_approvals(messages: Sequence[StoredRow]) -> set[str]:
    """The approval ids the last step with tool calls asked for FIRST (G-408).

    An approval belongs to the step that first asked for it. A later row that
    repeats its approval id (a host's copy of the paused step, say) asks for
    nothing, so a decision for that approval never runs a call of the later
    row.
    """
    steps, _ = _thread_steps(messages)
    if not steps:
        return set()
    first: dict[str, int] = {}
    for position, assistant, _ in steps:
        for request in assistant.get("tool_approval_requests", []):
            first.setdefault(request["approval_id"], position)
    position, assistant, _ = steps[-1]
    return {
        request["approval_id"]
        for request in assistant.get("tool_approval_requests", [])
        if first.get(request["approval_id"]) == position
    }


def step_position(messages: Sequence[StoredRow]) -> int:
    """The thread position of the last tool-calling step's assistant row."""
    steps, _ = _thread_steps(messages)
    return steps[-1][0] if steps else 0


def last_step_superseded(messages: Sequence[StoredRow]) -> bool:
    """Whether the last step with tool calls is superseded (G-397)."""
    _, superseded = _thread_steps(messages)
    return bool(superseded) and superseded[-1]


def last_step_results(
    messages: Sequence[StoredRow],
) -> dict[int, dict[str, Any]]:
    """The results of the last tool-calling step, by call index (G-392)."""
    steps, _ = _thread_steps(messages)
    return _paired(steps)[-1] if steps else {}


def replaced_steps(
    messages: Sequence[StoredRow],
    needs_approval: Callable[[str], bool],
) -> list[tuple[int, list[dict[str, Any]]]]:
    """The gated calls with no result of their own of every SUPERSEDED
    tool-calling step, by the thread position of the step's assistant row
    (G-392, G-397).

    A step is superseded as :func:`superseded_at` says. Only a step
    that is not is resolved, so a decision for a superseded one never runs
    anything, whenever it was made. Each such call gets one result saying a
    later step replaced it, so no gated call is left without an outcome. A
    call with a result in its own step's segment gets none; a result in
    another step's segment never counts for it.
    """
    steps, superseded = _thread_steps(messages)
    paired = _paired(steps)
    replaced: list[tuple[int, list[dict[str, Any]]]] = []
    for step, (position, assistant, _) in enumerate(steps):
        if not superseded[step]:
            continue
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
        decisions.setdefault(decision["approval_id"], decision)

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
