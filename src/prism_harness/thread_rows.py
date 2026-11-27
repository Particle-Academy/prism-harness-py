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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prism_harness.tool_arguments import usable_arguments

__all__ = [
    "ENDED_BY_START",
    "REPEATED_CALL_ID",
    "SUPERSEDED",
    "SUPERSEDED_RESULT",
    "ToolCallInput",
    "assistant_row",
    "call_results",
    "ended_by_start_row",
    "requested_step",
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


def ended_by_start_row(
    calls: Sequence[ToolCallInput],
    approval_ids: Sequence[str | None],
    results: Sequence[Mapping[str, Any]] = (),
    decisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The tool result row a runner start's task row closes the calls still
    pending on a child with (G-387): the results the step already had, then
    each ended call with the one fixed :data:`DELEGATION_ENDED` result (its
    arguments and provider ids kept), the decisions, and LAST on the row the
    harness's own mark, ``{approval_ids: [...]}``: the approvals those calls
    were asked under (``approval_ids``, each call's OWN, paired by
    :func:`paired_requests`, None for a call asked nothing), in CALL order. The
    PHP reference's row, so a thread reads the same in every language;
    prism-parity's ``harness-thread-rows`` corpus pins it (``write:
    ended_by_start``)."""
    approvals = [
        approval
        for _, approval in zip(calls, approval_ids, strict=True)
        if isinstance(approval, str) and approval
    ]
    row = tool_result_row(
        [*results, *(tool_result_entry(call, DELEGATION_ENDED) for call in calls)], decisions
    )
    row[ENDED_BY_START] = {"approval_ids": list(dict.fromkeys(approvals))}
    return row


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

    The LEGACY measure (G-387): read only for a root run whose paused row was
    written before ``harness_steps_used`` existed. It counts the session's own
    rows, so it never holds a subagent child's steps; the row's own count does.

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


def requested_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """THE step a decision may apply to (G-397): the index of the stored
    assistant row whose approval requests the transcript tail is waiting on,
    or None.

    Newest first: tool result and decision rows (and any other row that is
    not a turn: a system note) are skipped; a user turn ends the search, and
    so does the first assistant row reached when it asked for nothing. Every
    reader that selects "the paused step" -- the decision checks, the
    repeated-id guard, the resolver's decisions, a runner start's
    answerability and the plan readers -- selects it with this function and
    no other rule, so a row recorded after the park can only make all of them
    see no step, never make one of them see another.

    A 0.1.0 thread kept a step's requests in rows of their own after it;
    such a row counts as the step having asked.
    """
    asked_apart = False
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        kind = row.get("type")
        if kind == "user":
            return None
        if kind == "tool_approval_request":
            asked_apart = asked_apart or bool(_dicts(row.get("approvals")))
            continue
        if kind != "assistant":
            continue
        return index if _dicts(row.get("tool_approval_requests")) or asked_apart else None
    return None


def pending_approval_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The approval ids the run paused at the transcript tail is still waiting
    on: issued by the last assistant step and not yet answered (G-354).

    Read from the stored rows: the step :func:`requested_step` selects (a
    user turn, or an assistant step that asked nothing, ends the scope), less
    every id answered after it. An id answered once is no longer pending.
    """
    index = requested_step(rows)
    if index is None:
        return []
    answered: set[str] = set()
    for row in rows[index + 1 :]:
        if row.get("type") == "tool_result":
            for response in _dicts(row.get("tool_approval_responses")):
                if isinstance(response.get("approval_id"), str):
                    answered.add(response["approval_id"])
    pending: list[str] = []
    for request in _dicts(rows[index].get("tool_approval_requests")):
        approval_id = request.get("approval_id")
        if isinstance(approval_id, str) and approval_id and approval_id not in answered:
            pending.append(approval_id)
    return list(dict.fromkeys(pending))


def decided_unresolved_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The approval ids the run paused at the transcript tail asked for that
    a person has DECIDED and whose call has no result yet: decisions a
    resume would still act on (G-387 round 10).

    The same scope as :func:`pending_approval_ids`, newest first: a user
    turn, or an assistant step that asked nothing, ends it.
    """
    index = requested_step(rows)
    if index is None:
        return []
    decided: set[str] = set()
    for row in rows[index + 1 :]:
        kind = row.get("type")
        if kind == "tool_result":
            for response in _dicts(row.get("tool_approval_responses")):
                if isinstance(response.get("approval_id"), str):
                    decided.add(response["approval_id"])
        elif kind == "tool_approval_response" and isinstance(row.get("approval_id"), str):
            decided.add(row["approval_id"])
    # Which call has a result is matched BY POSITION (G-387 round 13): a
    # result under a repeated id answers ONE call, never every call that
    # carries the id.
    done = call_results(_dicts(rows[index].get("tool_calls")), rows[index + 1 :])
    found: list[str] = []
    for request, call in step_requests(rows, index):
        approval_id = request.get("approval_id")
        if (
            isinstance(approval_id, str)
            and approval_id in decided
            and (call is None or call not in done)
        ):
            found.append(approval_id)
    return list(dict.fromkeys(found))


def ended_approval_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """The approval ids whose call a runner start ended, recorded as not run
    (G-387 rounds 10 and 11): never approved, whatever decision was recorded.

    Recognised ONLY by :data:`ENDED_BY_START`, the field the harness writes on
    the tool result row its task row closes the calls with, naming the ended
    APPROVAL ids. A result's text is never read (a tool can return any text),
    and nothing a tool, a model or a host prompt sends reaches that field: a
    tool's output is a result string, its arguments and a provider's state
    live inside other rows. Scoped to its own step: a mark ends only approval
    ids the last assistant step before it asked for, never anything matched by
    a tool call id, which providers reuse across turns.

    ANY tool result row after the approval's request row and before the next
    assistant row counts (G-387 round 13), not only the first one carrying the
    call's id: a step whose calls share an id has an earlier, unmarked result
    row for the call that already ran, and that row must not hide the mark the
    start wrote. The key is the harness's alone, so this is no looser.
    """
    return _marked(rows, ENDED_BY_START)


def superseded_approval_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """The approval ids whose call was denied because a later step replaced
    the step it was asked on (DECISION B, G-387 round 13): never approved,
    whatever decision was recorded. Recognised only by :data:`SUPERSEDED`,
    scoped exactly as :func:`ended_approval_ids` is."""
    return _marked(rows, SUPERSEDED)


def _marked(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    """The approval ids a harness mark ``key`` names on a tool result row of
    the step that asked for them (after its request row, before the next
    assistant row)."""
    ended: set[str] = set()
    asked: set[str] = set()
    for row in rows:
        kind = row.get("type")
        if kind == "assistant":
            asked = {
                request["approval_id"]
                for request in _dicts(row.get("tool_approval_requests"))
                if isinstance(request.get("approval_id"), str) and request["approval_id"]
            }
        elif kind == "tool_result":
            ended.update(approval for approval in _ended_mark(row.get(key)) if approval in asked)
    return ended


def _ended_mark(value: object) -> list[str]:
    """The approval ids an ended mark names: ONLY the reference shape,
    ``{approval_ids: [...]}`` (G-387 round 12). Any other value, the
    round-11 list included (it never shipped), names nothing."""
    if not isinstance(value, dict):
        return []
    ids = value.get("approval_ids")
    if not isinstance(ids, list):
        return []
    return [approval for approval in ids if isinstance(approval, str) and approval]


#: The field a runner start's task row writes on the tool result row that
#: closes the calls still pending on the child (G-387 rounds 11 and 12):
#: ``{approval_ids: [...]}``, the approvals it ended, the PHP reference's
#: shape. Written by the harness alone, and the only thing that marks a call
#: as ended by a start.
ENDED_BY_START = "harness_ended_by_start"


#: The field on the tool result row that denies a superseded step's open
#: gated calls (DECISION B, G-387 round 13): ``{approval_ids: [...]}``, the
#: PHP reference's shape, last on the row. Written by the harness alone.
SUPERSEDED = "harness_superseded"

#: The result each open gated call of a superseded step is given: the PHP
#: reference's text, the same in every language.
SUPERSEDED_RESULT = "Not run: a later step replaced it."


def superseding_denials(
    stored: Sequence[tuple[Mapping[str, Any], str | None]],
    batch: Sequence[dict[str, Any]],
    run_id: str | None,
) -> list[dict[str, Any]]:
    """``batch`` (recorded under ``run_id``) with, before each row of it that
    SUPERSEDES the paused step, that step's denials (DECISION B, G-387 round
    13). ``stored`` is the thread so far, each row with the run that recorded
    it.

    The paused step is the last assistant row with tool calls. Its OWNING
    runs are the run that recorded it and every run that recorded a result of
    one of its calls. A later assistant row with tool calls always supersedes
    it; an assistant text row or a user row supersedes it only when recorded
    outside those runs (no run id, or another run), so the harness's own rows
    in the runs resolving it never do, and a host's row always does. Each of
    its calls with no result (by position, see :func:`call_results`) gets ONE
    denial, :data:`SUPERSEDED_RESULT`, on one tool result row marked
    :data:`SUPERSEDED` with the approvals those calls were asked under (each
    call's own, :func:`step_requests`), in call order. Applied row by row, so
    a batch holding several steps denies each one it replaces."""
    rows: list[Mapping[str, Any]] = [row for row, _ in stored]
    runs: list[str | None] = [run for _, run in stored]
    out: list[dict[str, Any]] = []
    for message in batch:
        denial = _denial_before(rows, runs, message, run_id)
        for row in ([] if denial is None else [denial]) + [message]:
            out.append(row)
            rows.append(row)
            runs.append(run_id)
    return out


def _denial_before(
    rows: Sequence[Mapping[str, Any]],
    runs: Sequence[str | None],
    message: Mapping[str, Any],
    run_id: str | None,
) -> dict[str, Any] | None:
    kind = message.get("type")
    with_calls = kind == "assistant" and bool(_dicts(message.get("tool_calls")))
    if not with_calls and kind not in ("assistant", "user"):
        return None
    step = next(
        (
            index
            for index in range(len(rows) - 1, -1, -1)
            if rows[index].get("type") == "assistant" and _dicts(rows[index].get("tool_calls"))
        ),
        None,
    )
    if step is None:
        return None
    calls = _dicts(rows[step].get("tool_calls"))
    answered: set[int] = set()
    owners: set[str | None] = {runs[step]}
    for index in range(step + 1, len(rows)):
        found = call_results(calls, [rows[index]])
        if found:
            answered |= set(found)
            owners.add(runs[index])
    if not with_calls and run_id is not None and run_id in owners:
        return None  # the harness's own row, in a run that owns the step
    open_calls = [index for index in range(len(calls)) if index not in answered]
    if not open_calls:
        return None
    own = {call: request.get("approval_id") for request, call in step_requests(rows, step)}
    row = tool_result_row(
        [
            {
                "tool_call_id": _text(calls[index].get("id")),
                "tool_name": _text(calls[index].get("name")),
                "args": usable_arguments(calls[index].get("arguments")) or {},
                "result": SUPERSEDED_RESULT,
                "tool_call_result_id": _nullable_text(calls[index].get("result_id")),
                "artifacts": [],
            }
            for index in open_calls
        ]
    )
    row[SUPERSEDED] = {
        "approval_ids": list(
            dict.fromkeys(
                approval
                for approval in (own.get(index) for index in open_calls)
                if isinstance(approval, str) and approval
            )
        )
    }
    return row


#: What a call still pending on a child is answered with when a runner start
#: gives the child a new task (G-387 round 9): the PHP reference's text (round
#: 12), so the row reads the same in every language. Nothing reads it to
#: decide anything; :data:`ENDED_BY_START` marks the call ended.
DELEGATION_ENDED = "Not run: the delegation this call was proposed under ended."


def approval_tool_scope(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], bool] | None:
    """Stored approval authority, separate from the provider replay view."""
    # Which decision counts is read from the stored rows, first one standing;
    # the view's fold is for replay to a provider, not for authority.
    decisions = {
        approval_id: decision["approved"] is True
        for approval_id, decision in first_decisions(rows).items()
    }
    # The step is the one requested_step() selects (G-397), read on the
    # view, where a 0.1.0 step's separate request rows are folded into it.
    view = thread_view(rows)
    at = requested_step(view)
    for row in [] if at is None else [view[at]]:
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

    The step :func:`requested_step` selects (G-397), when it carries its
    requests itself.
    """
    index = requested_step(rows)
    if index is None or not _dicts(rows[index].get("tool_approval_requests")):
        return None
    return rows[index]


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


#: The result every gated call of a step with a repeated call id is given
#: (G-392), identical in every language.
REPEATED_CALL_ID = "Tool call was not run because another call in this step has the same id"


def repeats_call_ids(assistant: Mapping[str, Any]) -> bool:
    """Whether a tool call id repeats among one step's calls (an empty or
    missing id counts), or a ``tool_call_id`` or an ``approval_id`` among its
    approval requests (G-392, G-387 round 13).

    Such a step cannot say which decision answers which call, so none of its
    gated calls runs.
    """
    ids = [_text(call.get("id")) for call in _dicts(assistant.get("tool_calls"))]
    requests = _dicts(assistant.get("tool_approval_requests"))
    asked = [_text(request.get("tool_call_id")) for request in requests]
    # A repeated APPROVAL id (G-387 round 13) cannot say which call it
    # decides either: the same step, failed closed.
    approvals = [_text(request.get("approval_id")) for request in requests]
    return (
        len(set(ids)) != len(ids)
        or len(set(asked)) != len(asked)
        or len(set(approvals)) != len(approvals)
    )


def paired_requests(
    assistant: Mapping[str, Any], ran: Collection[int] = ()
) -> list[tuple[dict[str, Any], int | None]]:
    """Each approval request of one step with the index of the call it was
    issued for, or None (G-392).

    Paired by position: a request takes the first call, in call order, that
    has its ``tool_call_id`` and no request yet. The harness issues requests
    in call order, so with repeated ids every call still gets its own request,
    never the last one issued under that id. ``ran`` are the calls the step
    ran at once (G-387 round 13): a call that ran was asked nothing, so a
    request never pairs with one of them (it pairs with no call instead); an
    ungated call never receives an approval.
    """
    calls = _dicts(assistant.get("tool_calls"))
    taken: set[int] = set()
    pairs: list[tuple[dict[str, Any], int | None]] = []
    for request in _dicts(assistant.get("tool_approval_requests")):
        wanted = _text(request.get("tool_call_id"))
        free = [
            i for i, call in enumerate(calls) if i not in taken and _text(call.get("id")) == wanted
        ]
        index = next((i for i in free if i not in ran), None)
        if index is not None:
            taken.add(index)
        pairs.append((request, index))
    return pairs


def step_requests(
    rows: Sequence[Mapping[str, Any]], index: int
) -> list[tuple[dict[str, Any], int | None]]:
    """:func:`paired_requests` for the stored assistant row ``rows[index]``,
    knowing which of its calls the step ran at once: the ones the tool result
    row written right after it answers, before any decision (G-387 round 13).
    A request is never paired with an ungated call that shares its id."""
    assistant = rows[index]
    following = rows[index + 1] if index + 1 < len(rows) else None
    ran: set[int] = set()
    if (
        following is not None
        and following.get("type") == "tool_result"
        and not _dicts(following.get("tool_approval_responses"))
        and ENDED_BY_START not in following
    ):
        ran = set(call_results(_dicts(assistant.get("tool_calls")), [following]))
    return paired_requests(assistant, ran)


def call_results(
    calls: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Which of one step's ``calls`` the tool result ``rows`` after it
    answer, by call INDEX, each with its first result (G-387 round 13).

    Matched BY POSITION, never through a map keyed by call id, where a result
    under a repeated id would answer every call carrying it. Within each row a
    result takes one call: the first, in call order, with its id that the row
    has not matched yet, preferring the one with its tool name and arguments,
    then its tool name. So a gated call that shares its id with an ungated
    one that ran stays unanswered. A row that repeats earlier results (a
    resume's, a start's) matches the same calls again."""
    matched: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("type") != "tool_result":
            continue
        normalised = _normalise(row, [])
        if normalised is None:
            continue
        taken: set[int] = set()
        for entry in normalised["tool_results"]:
            index = _answered_call(calls, entry, taken)
            if index is not None:
                taken.add(index)
                matched.setdefault(index, entry)
    return matched


def _answered_call(
    calls: Sequence[Mapping[str, Any]], entry: Mapping[str, Any], taken: Collection[int]
) -> int | None:
    """The call one stored result answers: see :func:`call_results`."""
    wanted = _text(entry.get("tool_call_id"))
    free = [i for i, call in enumerate(calls) if i not in taken and _text(call.get("id")) == wanted]
    named = [i for i in free if _text(calls[i].get("name")) == entry.get("tool_name")]
    exact = [
        i
        for i in named
        if (usable_arguments(calls[i].get("arguments")) or {}) == (entry.get("args") or {})
    ]
    for found in (exact, named, free):
        if found:
            return found[0]
    return None


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
