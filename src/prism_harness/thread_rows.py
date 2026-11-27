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
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from prism_harness.tool_arguments import usable_arguments
from prism_harness.tool_policy import args_digest

__all__ = [
    "ANSWERS_STEP",
    "CALL_BINDING",
    "ENDED_BY_START",
    "INTERRUPTED_RESULT",
    "REPEATED_CALL_ID",
    "STARTED",
    "SUPERSEDED",
    "SUPERSEDED_RESULT",
    "ToolCallInput",
    "assistant_row",
    "call_results",
    "ended_by_start_row",
    "is_tool_step",
    "last_tool_step",
    "placed_view",
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
    keyed by its tool call id, tool name and arguments (never the id alone; see
    :func:`_fold`) and a decision by its approval id, and a later one replaces
    an earlier one in the earlier one's position.

    And each result is shown ONCE, in its own step's segment (G-387 round 14):
    a tool result row that is not next to the earlier ones (a host's row sits
    between them) adds no copy of a result or a decision that segment has
    already shown (see :func:`_once`).
    """
    return _as_decided(_folded(rows), rows)


def placed_view(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The thread as a PROVIDER is sent it (G-387 round 15), the PHP
    reference's ``placedRows()`` then the fold of :func:`thread_view`, over
    the rows as :meth:`~prism_harness.thread.Thread.rows` reads them:

    - a started row is never replayed, and a started call with no result is
      given ONE result, :data:`INTERRUPTED_RESULT`;
    - a harness row whose honoured mark names a step is replayed in that
      step's own result run (after the step and the result and system rows
      that follow it), not where it sits;
    - a step that is not the paused step whose requested call has no outcome
      (a thread an older version wrote) is given the replaced result, and
      every other call of it with no outcome the interrupted one (G-387
      round 16), so every tool call a provider is sent has a result;
    - a request repeating an approval id an earlier step asked for is void,
      and not replayed; an assistant row that is not a step replays with no
      requests;
    - each decision is replayed as the one that STANDS (G-401): a denial when
      any decision recorded for its approval is one."""
    placed: list[Mapping[str, Any]] = []
    # Every approval id a row before this one asked for, folded as the rows
    # go by (G-387 round 16: one pass, never a rescan per row).
    before: set[str] = set()
    for row in _replayed_rows(rows):
        original = row
        if row.get("type") == "assistant" and isinstance(row.get("tool_approval_requests"), list):
            if not is_tool_step(row):
                row = {**row, "tool_approval_requests": []}
            else:
                row = {
                    **row,
                    "tool_approval_requests": [
                        request
                        for request in row["tool_approval_requests"]
                        if not isinstance(request, dict) or request.get("approval_id") not in before
                    ],
                }
        before |= _asks(original)
        placed.append(row)
    return _as_decided(_folded(placed), rows)


def _as_decided(
    view: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """``view`` (folded from ``rows``) with every decision entry for an
    approval a step asked for replaced by the one that STANDS
    (:func:`standing_decisions`), and dropped when none does (every decision
    came after its call's outcome, or none in its step's own rows); an entry
    for an approval no step asked for keeps the standing rule over every row.
    A result row left with neither results nor decisions is dropped (G-387
    round 16, the reference's replay and fold)."""
    standing = standing_decisions(rows)
    asked = _step_approvals(rows)
    out: list[dict[str, Any]] = []
    for row in view:
        if row["type"] == "tool_result" and row["tool_approval_responses"]:
            decisions = [
                standing.get(decision["approval_id"], decision)
                for decision in row["tool_approval_responses"]
                if decision["approval_id"] in standing or decision["approval_id"] not in asked
            ]
            if not decisions and not row["tool_results"]:
                continue
            row = {**row, "tool_approval_responses": decisions}
        out.append(row)
    return out


def _folded(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []

    for row in rows:
        normalised = _normalise(row, view)

        if normalised is None:
            continue

        if normalised["type"] == "tool_result" and view and view[-1]["type"] == "tool_result":
            view[-1] = _fold(view[-1], normalised)
        else:
            view.append(normalised)

    return _once(view)


def standing_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The decision that STANDS for every approval id the thread holds one
    for (G-401, the recommended rule): the first DENIAL when any decision
    recorded for it is one, else the first approval -- counting only the
    decisions recorded while the approval's call had not been ACTED on yet
    (G-387 round 16, :func:`acted_at`): a decision recorded after its call
    started or got its outcome is not the one that stood when the call was
    resolved, so it changes no reader and never what the replay shows.

    Read only from the rows that belong to the approval's step (its segment
    and the rows a harness mark places with it; the step that FIRST asked for
    it): a decision row anywhere else decides nothing. An approval no step
    asked for reads every row."""
    acted = acted_at(rows)
    ix = _index(rows)
    asked = _step_approvals(rows)
    owned: dict[int, set[int]] = {}
    found: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        kind = row.get("type")
        if kind == "tool_result":
            responses = _dicts(row.get("tool_approval_responses"))
        elif kind == "tool_approval_response":
            responses = [dict(row)]
        else:
            continue
        for response in responses:
            approval = response.get("approval_id")
            if not isinstance(approval, str) or not _counts(acted, approval, index):
                continue
            step = asked.get(approval)
            if step is not None:
                if step not in owned:
                    owned[step] = set(ix.owned.get(step, ()))
                if index not in owned[step]:
                    continue
            current = found.get(approval)
            if current is None or (current["approved"] and response.get("approved") is not True):
                found[approval] = _decision(response)
    return found


def _step_approvals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Every approval id a step asked for FIRST, with that step's index."""
    ix = _index(rows)
    if ix.asked_by is None:
        asked: dict[str, int] = {}
        for step in ix.steps:
            for request, _ in step_requests(rows, step):
                approval = request.get("approval_id")
                if isinstance(approval, str) and approval:
                    asked.setdefault(approval, step)
        ix.asked_by = asked
    return ix.asked_by


def acted_at(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Where each approval's call was first ACTED on (G-387 round 16), by
    approval id: the index of the first row its step owns that holds the
    call's started row or its first result (its outcome). The approval's
    call is the one its own request names in the step that FIRST asked for
    it (:func:`step_requests`), paired exactly as the resolver pairs it. An
    approval whose call has neither, or that names no call, is absent.

    THE rule for which decisions count: one recorded at a row AFTER that
    index does not (a decision in the same row as the outcome does: the
    resolution row restates the decisions it acted on)."""
    ix = _index(rows)
    if ix.acted is not None:
        return ix.acted
    acted: dict[str, int] = {}
    for step in ix.steps:
        pairs = step_requests(rows, step)
        if not pairs:
            continue
        calls = step_calls(rows[step])
        first: dict[int, int] = dict(ix.started_at.get(step, {}))
        if step not in ix.started:
            started_calls(rows, step)
            first = dict(ix.started_at.get(step, {}))
        matched: dict[int, dict[str, Any]] = {}
        for index in ix.owned.get(step, []):
            row = rows[index]
            if row.get("type") != "tool_result":
                continue
            normalised = _normalise(row, [])
            if normalised is None:
                continue
            taken: set[int] = set()
            for entry in normalised["tool_results"]:
                call = _answered_call(calls, entry, taken, matched)
                if call is None:
                    continue
                taken.add(call)
                if call not in matched:
                    matched[call] = entry
                    first[call] = min(first.get(call, index), index)
        for request, call in pairs:
            approval = request.get("approval_id")
            if isinstance(approval, str) and call is not None and call in first:
                acted.setdefault(approval, first[call])
    ix.acted = acted
    return acted


def _counts(acted: Mapping[str, int], approval: str, index: int) -> bool:
    """Whether a decision for ``approval`` recorded at row ``index`` counts:
    its call had not been acted on before that row (:func:`acted_at`)."""
    at = acted.get(approval)
    return at is None or index <= at


def _replayed_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The stored rows in the order a provider is shown them (G-387 round
    15): a started row is never replayed; a harness row whose HONOURED mark
    names a step is replayed in THAT step's result run (directly after the
    step and the result and system rows that follow it), not where it sits;
    and a started call with no result is given ONE result there,
    :data:`INTERRUPTED_RESULT`, after those rows. A 0.5.x
    :data:`REPLACED_STEP` row (or one main writes for such a thread) is
    moved to the step it names the same way (G-387 round 16): written after
    a later step, it would otherwise replay in that step's segment, under
    calls it does not answer."""
    ix = _index(rows)
    moved: dict[int, list[Mapping[str, Any]]] = {}
    skip: set[int] = set()
    started: set[int] = set()
    for index in range(len(rows)):
        step = ix.marked[index]
        if step is None or step not in ix.owned or step >= index:
            continue
        skip.add(index)
        if STARTED in rows[index]:
            started.add(step)
        else:
            moved.setdefault(step, []).append(rows[index])
    synthesised: dict[int, dict[int, str]] = {}
    for step in started:
        for call in in_progress_calls(rows, step):
            synthesised.setdefault(step, {})[call] = INTERRUPTED_RESULT
    # UPGRADE (item 9): a step that is not the paused step whose requested
    # call has no outcome (a thread an older version wrote, which never
    # denied it) is replayed with the replaced result for it.
    paused = requested_step(rows)
    for step in ix.steps:
        if step == paused:
            continue
        calls = step_calls(rows[step])
        done = set(call_results(calls, owned_rows(rows, step)))
        busy = set(started_calls(rows, step))
        for _, asked in step_requests(rows, step):
            if asked is not None and asked not in done and asked not in busy:
                synthesised.setdefault(step, {})[asked] = SUPERSEDED_RESULT
        # EVERY other call of a step no longer paused that has no outcome
        # (G-387 round 16): it may have run (an ungated call whose results
        # row never landed, a host's step), so it reads interrupted, and no
        # provider request ever carries a tool call without a result.
        for call in range(len(calls)):
            if call not in done and call not in synthesised.get(step, {}):
                synthesised.setdefault(step, {})[call] = INTERRUPTED_RESULT
    if not skip and not synthesised:
        return list(rows)
    for step, texts in synthesised.items():
        calls = step_calls(rows[step])
        moved.setdefault(step, []).append(
            tool_result_row(
                [
                    {
                        "tool_call_id": _text(calls[call].get("id")),
                        "tool_name": _text(calls[call].get("name")),
                        "args": usable_arguments(calls[call].get("arguments")) or {},
                        "result": texts[call],
                        "tool_call_result_id": _nullable_text(calls[call].get("result_id")),
                        "artifacts": [],
                    }
                    for call in sorted(texts)
                ]
            )
        )
    out: list[Mapping[str, Any]] = []
    waiting: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if index in skip:
            continue
        if waiting and row.get("type") not in ("tool_result", "system"):
            out.extend(waiting)
            waiting = []
        out.append(row)
        if index in moved:
            out.extend(waiting)
            waiting = list(moved[index])
    out.extend(waiting)
    return out


def _once(view: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The folded view with each result shown once per segment (G-387 round
    14). A segment is the rows after an assistant row with tool calls, up to
    the next one. A tool result row there drops each result the segment has
    already shown (the same call id, tool name and arguments, its n-th such
    result of the row against the n shown so far) and each decision for an
    approval id already shown; a row left with neither is dropped. The
    resolution row restates the results its step already had, so with a
    host's row between them a provider would otherwise get one tool output
    twice, which some providers refuse."""
    out: list[dict[str, Any]] = []
    shown: dict[tuple[str, str, str], int] = {}
    decided: set[str] = set()
    for row in view:
        if row["type"] == "assistant" and row["tool_calls"]:
            shown, decided = {}, set()
        if row["type"] != "tool_result":
            out.append(row)
            continue
        seen: dict[tuple[str, str, str], int] = {}
        results: list[dict[str, Any]] = []
        for entry in row["tool_results"]:
            same = (entry["tool_call_id"], entry["tool_name"], _canonical(entry["args"]))
            seen[same] = seen.get(same, 0) + 1
            if seen[same] > shown.get(same, 0):
                results.append(entry)
        decisions = [d for d in row["tool_approval_responses"] if d["approval_id"] not in decided]
        for same, count in seen.items():
            shown[same] = max(shown.get(same, 0), count)
        decided.update(d["approval_id"] for d in row["tool_approval_responses"])
        if len(results) == len(row["tool_results"]) and len(decisions) == len(
            row["tool_approval_responses"]
        ):
            out.append(row)
        elif results or decisions:
            out.append(tool_result_row(results, decisions))
    return out


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
        # G-408: only a STEP (an assistant row holding an object call, the one
        # rule every reader selects steps by) can be the paused step; any
        # other assistant row, a copy whose calls are a string, a number,
        # null or strings included, asked for nothing and ends the search.
        if not is_tool_step(row):
            return None
        # Only the requests it asked FIRST count (G-408, item 8): a step whose
        # every request repeats an earlier step's approval id asked nothing.
        ix = _index(rows)
        asked = [
            request
            for request in _dicts(row.get("tool_approval_requests"))
            if not ix.asked_earlier(request.get("approval_id"), index)
        ]
        return index if asked or asked_apart else None
    return None


def is_tool_step(row: Mapping[str, Any]) -> bool:
    """Whether a stored row is a STEP with tool calls (G-408, the TS 0.11.x
    #92 rule the PHP reference adopted): an assistant row whose
    ``tool_calls`` is a list holding AT LEAST ONE object; its calls are
    exactly those objects, in order (:func:`step_calls`). THE one rule every
    reader selects a step by -- the paused step, a step's calls, their
    answered lookup, the replaced check, a mark's target, the supersede and
    the replay -- so no reader ever takes a row another reader passes over. A
    row whose calls are a JSON string, a number, null, an object, a list of
    strings or of JSON-string calls is not a step: it replays as a text
    row with no calls."""
    return row.get("type") == "assistant" and bool(step_calls(row))


def step_calls(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A step's calls, by the :func:`is_tool_step` rule: the objects of its
    ``tool_calls`` list, in order (none when it is not a list)."""
    return _dicts(row.get("tool_calls"))


def last_tool_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """The index of the last :func:`is_tool_step` row, or None."""
    return next((i for i in range(len(rows) - 1, -1, -1) if is_tool_step(rows[i])), None)


def pending_approval_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The approval ids the run paused at the transcript tail is still waiting
    on: issued by the last assistant step and not yet answered (G-354).

    Read from the stored rows: the step :func:`requested_step` selects (a
    user turn, or an assistant step that asked nothing, ends the scope), less
    every id answered after it. An id answered once is no longer pending, and
    neither is one a harness mark ended or replaced (G-387 round 14): a
    replaced step is never answerable, whatever row follows its denial. Nor
    is one whose call already has an outcome (G-387 round 16).
    """
    index = requested_step(rows)
    if index is None:
        return []
    answered: set[str] = ended_approval_ids(rows) | superseded_approval_ids(rows)
    # A started call is answered-in-progress (G-387 round 15), never pending.
    answered |= {a for a in started_calls(rows, index).values() if isinstance(a, str)}
    for row in rows[index + 1 :]:
        if row.get("type") == "tool_result":
            for response in _dicts(row.get("tool_approval_responses")):
                if isinstance(response.get("approval_id"), str):
                    answered.add(response["approval_id"])
    # A call that already has an OUTCOME (any result its step owns: a
    # denial, "No approval response provided", a host's result) keeps no
    # approval pending (G-387 round 16), paired exactly as the resolver
    # (:func:`step_requests`, :func:`call_results`) and the replay read it.
    done = set(call_results(step_calls(rows[index]), owned_rows(rows, index)))
    done |= set(started_calls(rows, index))  # answered-in-progress, or interrupted
    pending: list[str] = []
    # Only the requests the step asked FIRST (G-408): see step_requests().
    for request, call in step_requests(rows, index):
        approval_id = request.get("approval_id")
        if call is not None and call in done:
            continue
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
    # Ended or replaced (a harness mark): never acted on (G-387 round 14).
    closed = ended_approval_ids(rows) | superseded_approval_ids(rows)
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
    done = set(call_results(step_calls(rows[index]), step_rows(rows, index)))
    done |= set(started_calls(rows, index))
    found: list[str] = []
    for request, call in step_requests(rows, index):
        approval_id = request.get("approval_id")
        if (
            isinstance(approval_id, str)
            and approval_id in decided
            and approval_id not in closed
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
    scoped exactly as :func:`ended_approval_ids` is; and, in a thread the
    0.5.x line wrote, the approvals of the calls its :data:`REPLACED_STEP`
    rows answer (G-387 round 13)."""
    found = _marked(rows, SUPERSEDED)
    for row in rows:
        mark = honoured(row, REPLACED_STEP)
        if row.get("type") != "tool_result" or not isinstance(mark, dict):
            continue
        position = mark.get("position")
        if type(position) is not int or not 1 <= position <= len(rows):
            continue
        step = position - 1
        if rows[step].get("type") != "assistant":
            continue
        answered = call_results(step_calls(rows[step]), [row])
        for request, call in step_requests(rows, step):
            approval = request.get("approval_id")
            if call in answered and isinstance(approval, str) and approval:
                found.add(approval)
    return found


def _marked(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    """The approval ids a harness mark ``key`` names on a tool result row of
    the step that asked for them (after its request row, before the next
    assistant row). Honoured only on a row the harness wrote (G-387 round
    16, ONE TRUST SIGNAL): see :func:`honoured`."""
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
            ended.update(
                approval for approval in _ended_mark(honoured(row, key)) if approval in asked
            )
    return ended


def honoured(row: Mapping[str, Any], key: str) -> object:
    """The value of the harness mark ``key`` on ``row`` when it COUNTS, else
    None (G-387 round 16, ONE TRUST SIGNAL). A mark counts only on a tool
    result row the harness itself wrote: read with :data:`HARNESS` set, which
    :meth:`~prism_harness.thread.Thread.rows` sets from the stored entry's
    own flag, never from the message, and only the runtime's private in-run
    write (and the denials a superseding write makes the harness write) set
    that flag. A run id is never a trust signal: a row any caller records,
    under any run id or none, carries no mark.

    The one exception is the 0.5.x line's :data:`REPLACED_STEP` row, which
    that line stored with no flag: an unflagged one counts only in the shape
    that line wrote, every result on it the replaced denial
    (:data:`SUPERSEDED_RESULT`). So the most such a row can do is deny an
    earlier step's call, which never runs anyway: it fails closed."""
    if row.get("type") != "tool_result" or key not in row:
        return None
    if row.get(HARNESS) is True:
        return row.get(key)
    if key == REPLACED_STEP:
        results = _dicts(row.get("tool_results"))
        raw = row.get("tool_results")
        if (
            isinstance(raw, list)
            and len(results) == len(raw)
            and all(entry.get("result") == SUPERSEDED_RESULT for entry in results)
        ):
            return row.get(key)
    return None


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
SUPERSEDED_RESULT = "Tool call was not run because a later step replaced it"


def superseding_denials(
    stored: Sequence[tuple[Mapping[str, Any], str | None]],
    batch: Sequence[dict[str, Any]],
    run_id: str | None,
    inside: str | None = None,
) -> list[dict[str, Any]]:
    """``batch`` (recorded under ``run_id``) with, before each row of it that
    SUPERSEDES the paused step, that step's denials (DECISION B, G-387 round
    13). ``stored`` is the thread so far, each row with the run that recorded
    it.

    The paused step is the last assistant row with tool calls. Its OWNING
    runs are the run that recorded it and every run that recorded a result of
    one of its calls. A later assistant row with tool calls always supersedes
    it; an assistant text row or a user row supersedes it unless the
    RUNTIME records it inside one of those runs (``inside``, the run the
    runtime is recording for; None for any caller). A run id a caller passes,
    the step's own included, is outside: a host's row always supersedes. Each of
    its calls with no result (by position, see :func:`call_results`) gets ONE
    denial, :data:`SUPERSEDED_RESULT`, on one tool result row marked
    :data:`SUPERSEDED` with the approvals those calls were asked under (each
    call's own, :func:`step_requests`), in call order. Applied row by row, so
    a batch holding several steps denies each one it replaces."""
    rows: list[Mapping[str, Any]] = [row for row, _ in stored]
    runs: list[str | None] = [run for _, run in stored]
    out: list[dict[str, Any]] = []
    for message in batch:
        denial = _denial_before(rows, runs, message, inside)
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
    with_calls = is_tool_step(message)
    if not with_calls and kind not in ("assistant", "user"):
        return None
    step = last_tool_step(rows)
    if step is None:
        return None
    calls = step_calls(rows[step])
    answered: set[int] = set()
    owners: set[str | None] = {runs[step]}
    mine = [index for index in range(step + 1, len(rows)) if belongs_to(rows, index, step)]
    # Read across the step's rows together, never row by row: a runtime row
    # holds only the results its step did not hold yet (G-387 round 15).
    answered |= set(call_results(calls, [rows[index] for index in mine]))
    for index in mine:
        if _dicts(rows[index].get("tool_results")) or answers_step(rows, index) == step:
            owners.add(runs[index])  # a result's run, or the resolver's started row
    # A call the resolver has STARTED is answered-in-progress (G-387 round
    # 15): it may be running right now, so it gets no denial; its result
    # pairs to it wherever it lands.
    answered |= set(started_calls(rows, step))
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


#: The key the 0.5.x line writes on the row answering an EARLIER step's
#: calls that a later step replaced, ``{position: <the thread position of
#: that step's assistant row>}``, with :data:`SUPERSEDED_RESULT` per call.
#: Main reads it (G-387 round 13: a 0.27 main reads the threads its released
#: line wrote) and writes it for the same case: an earlier step of a thread
#: that line wrote whose calls a later step left with no outcome.
REPLACED_STEP = "harness_replaced_step"


def belongs_to(rows: Sequence[Mapping[str, Any]], index: int, step: int) -> bool:
    """Whether the row at ``index``, recorded after the step at index
    ``step``, may answer it: every row does, except one MARKED for another
    step -- an honoured :data:`ANSWERS_STEP` mark (see :func:`answers_step`)
    or a :data:`REPLACED_STEP` row -- which answers only the step it names,
    wherever it sits."""
    marked = _marked_for(rows, index)
    return marked is None or marked == step


def _marked_for(rows: Sequence[Mapping[str, Any]], index: int) -> int | None:
    """The step index a mark on the row at ``index`` names, or None when it
    carries none that counts. A :data:`REPLACED_STEP` row whose position is
    not a whole number names no step (-1)."""
    return _index(rows).marked[index]


def _marked_for_raw(rows: Sequence[Mapping[str, Any]], index: int, named: int | None) -> int | None:
    if named is not None:
        return named
    mark = honoured(rows[index], REPLACED_STEP)
    if not isinstance(mark, dict):
        return None
    position = mark.get("position")
    return position - 1 if type(position) is int else -1


#: The mark on every tool result row the runtime writes for a step's calls
#: (G-387 round 15, the PHP reference's shape): ``{position}``, the thread
#: position of the step it answers, LAST on the row. Such a row answers that
#: step wherever it lands, so a step another writer records while a call
#: runs never takes the call's result. Honoured only on a row the harness
#: wrote (``"harness": true`` as read, never a key a caller can store: see
#: :meth:`~prism_harness.thread.Thread.rows`) and only when it names an
#: EARLIER step holding a call with the id and tool name of every result on
#: the row; otherwise the row is its segment's, which fails closed. Stripped
#: from the replay.
ANSWERS_STEP = "harness_answers_step"

#: The write-ahead row the resolver records immediately before it runs ONE
#: approved call (G-387 round 15), in that call's own segment:
#: ``{tool_results: [], tool_approval_responses: [], harness_started:
#: {tool_call_id, tool_name, approval_id}, harness_answers_step: {position}}``.
#: A call with a started row and no result is ANSWERED-IN-PROGRESS: a
#: superseding row denies it nothing, it is not pending, awaits no decision
#: and never runs again. If no result ever lands it reads
#: :data:`INTERRUPTED_RESULT`. Never replayed.
STARTED = "harness_started"

#: What a started call that never got a result reads as (a crash, a
#: timeout): it may have run, so it is never run again without a fresh
#: approval. The PHP reference's text, the same in every language.
INTERRUPTED_RESULT = "Tool call was interrupted, it may have run"

#: The read-only flag :meth:`~prism_harness.thread.Thread.rows` puts on a
#: tool result row the harness itself wrote (under its private in-run
#: write); the only rows whose marks are honoured. Never taken from a
#: stored message.
HARNESS = "harness"


def started_row(call: Mapping[str, Any], approval_id: str | None, position: int) -> dict[str, Any]:
    """The write-ahead row for ``call`` of the step at thread ``position``:
    ``approval_id`` the approval it runs under, or None for a call the step
    runs at once (an UNGATED call, G-387 round 16: the reference's shape,
    ``"approval_id": null``)."""
    return {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [],
        STARTED: {
            "tool_call_id": _text(call.get("id")),
            "tool_name": _text(call.get("name")),
            "approval_id": approval_id,
        },
        ANSWERS_STEP: {"position": position},
    }


def answering_step(row: Mapping[str, Any], position: int) -> dict[str, Any]:
    """``row`` with :data:`ANSWERS_STEP` naming the step at ``position``, LAST."""
    marked = {key: value for key, value in row.items() if key != ANSWERS_STEP}
    marked[ANSWERS_STEP] = {"position": position}
    return marked


def _started_of(row: Mapping[str, Any]) -> tuple[str, str, str | None] | None:
    value = row.get(STARTED)
    if not isinstance(value, dict):
        return None
    call_id, name, approval = (
        value.get("tool_call_id"),
        value.get("tool_name"),
        value.get("approval_id"),
    )
    if (
        isinstance(call_id, str)
        and isinstance(name, str)
        and (approval is None or isinstance(approval, str))
    ):
        return call_id, name, approval
    return None


def answers_step(rows: Sequence[Mapping[str, Any]], index: int) -> int | None:
    """The index of the step the row at ``index`` answers by an HONOURED
    :data:`ANSWERS_STEP` mark, or None. Honoured only on a tool result row
    the harness wrote (:data:`HARNESS`), naming by position an EARLIER step
    (:func:`is_tool_step`) that has a call with the id and tool name of every
    result on the row (of a started row: of its call). A mark failing any of
    that changes nothing: the row is read as if it had none."""
    return _index(rows).answers[index]


def _answers_step_raw(rows: Sequence[Mapping[str, Any]], index: int) -> int | None:
    row = rows[index]
    if row.get("type") != "tool_result" or row.get(HARNESS) is not True:
        return None
    mark = row.get(ANSWERS_STEP)
    position = mark.get("position") if isinstance(mark, dict) else None
    if type(position) is not int or not 1 <= position <= index:
        return None
    step = position - 1
    calls = step_calls(rows[step]) if rows[step].get("type") == "assistant" else []
    if not calls:
        return None
    named = {(_text(call.get("id")), _text(call.get("name"))) for call in calls}
    if STARTED in row:
        started = _started_of(row)
        if started is None or _dicts(row.get("tool_results")):
            return None
        wanted = [(started[0], started[1])]
    else:
        wanted = [
            (_text(entry.get("tool_call_id")), _text(entry.get("tool_name")))
            for entry in _dicts(row.get("tool_results"))
        ]
    return step if all(pair in named for pair in wanted) else None


def owned_rows(rows: Sequence[Mapping[str, Any]], step: int) -> list[Mapping[str, Any]]:
    """Every row that answers the step at index ``step`` (G-387 rounds 13b
    and 15): the rows of its SEGMENT (after it, before the next step) not
    marked for another step, and every row marked for it anywhere after."""
    owned = _index(rows).owned.get(step)
    if owned is not None:
        return [rows[index] for index in owned]
    found: list[Mapping[str, Any]] = []
    segment = True
    for index in range(step + 1, len(rows)):
        if is_tool_step(rows[index]):
            segment = False
            continue
        marked = _marked_for(rows, index)
        if (marked is None and segment) or marked == step:
            found.append(rows[index])
    return found


def started_calls(rows: Sequence[Mapping[str, Any]], step: int) -> dict[int, str | None]:
    """The calls of the step at index ``step`` an HONOURED started row names
    (G-387 round 15), by call index, each with the approval it runs under.
    Paired with the call whose own request carries that approval and whose
    id and tool name it names, else the first such call not yet named."""
    ix = _index(rows)
    memo = ix.started.get(step)
    if memo is not None:
        return dict(memo)
    calls = step_calls(rows[step])
    own = {call: request.get("approval_id") for request, call in step_requests(rows, step)}
    found: dict[int, str | None] = {}
    candidates = ix.owned.get(step)
    for index in candidates if candidates is not None else range(step + 1, len(rows)):
        if ix.answers[index] != step:
            continue
        started = _started_of(rows[index])
        if started is None:
            continue
        call_id, name, approval = started
        same = [
            i
            for i, call in enumerate(calls)
            if i not in found
            and _text(call.get("id")) == call_id
            and _text(call.get("name")) == name
        ]
        mine = [i for i in same if own.get(i) == approval]
        for pick in (mine, same):
            if pick:
                found[pick[0]] = approval
                ix.started_at.setdefault(step, {})[pick[0]] = index
                break
    ix.started[step] = found
    return dict(found)


def in_progress_calls(rows: Sequence[Mapping[str, Any]], step: int) -> set[int]:
    """The calls of the step at index ``step`` that are ANSWERED-IN-PROGRESS
    (or, once nothing is running, INTERRUPTED): a started row and no result
    among the rows the step owns."""
    done = call_results(step_calls(rows[step]), owned_rows(rows, step))
    return {call for call in started_calls(rows, step) if call not in done}


def step_rows(rows: Sequence[Mapping[str, Any]], step: int) -> list[Mapping[str, Any]]:
    """The rows after the step at index ``step`` that may answer it.

    ABSOLUTE SEGMENT OWNERSHIP (G-387 round 13b): read for the LAST step with
    tool calls, so these are its segment; an earlier step reads only the rows
    before the next step with tool calls (:func:`replaced_earlier_steps`). A
    result is never moved to another step's call because its arguments match
    that call exactly: every call keeps exactly one outcome, and nothing runs
    twice. The one exception is the 0.5.x line's
    :data:`REPLACED_STEP` row, which names its step's position: it answers
    only that step (denials, never a run), and never this one."""
    return [rows[i] for i in range(step + 1, len(rows)) if belongs_to(rows, i, step)]


def replaced_earlier_steps(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Every EARLIER step with tool calls (one a later step with tool calls
    followed) whose calls have no outcome, by index, with those calls
    (G-387 round 13). Main denies such calls when the later step is recorded;
    a thread the 0.5.x line wrote can hold one it never answered. A result
    counts when recorded between the step and the next one, or marked for it
    (:data:`REPLACED_STEP`) anywhere after."""
    steps = _index(rows).steps
    found: list[tuple[int, list[dict[str, Any]]]] = []
    for at, _ in pairwise(steps):
        calls = step_calls(rows[at])
        # Its segment's rows not marked elsewhere and every row marked for
        # it: exactly the rows it owns (one pass, G-387 round 16).
        answered = call_results(calls, owned_rows(rows, at))
        # A started call that has no result is answered-in-progress (G-387
        # round 15): it may still be running, so it is never denied.
        busy = set(started_calls(rows, at))
        open_calls = [
            call for index, call in enumerate(calls) if index not in answered and index not in busy
        ]
        if open_calls:
            found.append((at, open_calls))
    return found


def step_superseded(
    stored: Sequence[tuple[Mapping[str, Any], str | None]],
) -> bool:
    """Whether the last step with tool calls was superseded by a row already
    in the thread (an assistant or user row recorded outside its owning
    runs, :func:`superseding_denials`) without being answered for it: a
    thread the 0.5.x line wrote, which denied a superseded step only when it
    resumed it. Its calls with no result are then answered with
    :data:`SUPERSEDED_RESULT` (G-387 round 13)."""
    rows = [row for row, _ in stored]
    runs = [run for _, run in stored]
    step = last_tool_step(rows)
    if step is None:
        return False
    calls = step_calls(rows[step])
    owners: set[str | None] = {runs[step]}
    for index in range(step + 1, len(rows)):
        if belongs_to(rows, index, step) and (
            call_results(calls, [rows[index]]) or answers_step(rows, index) == step
        ):
            owners.add(runs[index])
    return any(
        rows[index].get("type") in ("assistant", "user")
        and (runs[index] is None or runs[index] not in owners)
        for index in range(step + 1, len(rows))
    )


def replaced_step_row(position: int, calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The row answering ``calls`` of the earlier step at thread POSITION
    ``position`` (its ``entry.position``, never a list index) with one denial
    each, in the 0.5.x line's shape."""
    row = tool_result_row(
        [
            {
                "tool_call_id": _text(call.get("id")),
                "tool_name": _text(call.get("name")),
                "args": usable_arguments(call.get("arguments")) or {},
                "result": SUPERSEDED_RESULT,
                "tool_call_result_id": _nullable_text(call.get("result_id")),
                "artifacts": [],
            }
            for call in calls
        ]
    )
    row[REPLACED_STEP] = {"position": position}
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
        if is_tool_step(row):
            return {
                _text(call.get("id"))
                for call in step_calls(row)
                if usable_arguments(call.get("arguments")) is None
            }
    return set()


def first_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The decisions recorded for the turn being resumed, by approval id.

    Read from the stored rows after the last assistant row with tool calls:
    ``tool_approval_responses`` on tool result rows, and 0.3.0's single
    ``tool_approval_response`` rows. The FIRST decision recorded for an id is
    the one that counts, here and in every reader that decides what runs; a
    later one for the same id is an answer to a question already answered --
    except that ANY DENY IS FINAL (G-401): a later deny revokes an approve.

    The thread VIEW still folds a later decision over an earlier one in the
    earlier one's position: that is what a provider is shown, pinned by the
    cross-language thread rows corpus, and it decides nothing.
    """
    start = last_tool_step(rows)
    decisions: dict[str, dict[str, Any]] = {}
    if start is None:
        return decisions
    # Only a decision recorded before its call was acted on counts (G-387
    # round 16): a raw denial after the call started or ran is not one.
    acted = acted_at(rows)
    for index in range(start + 1, len(rows)):
        row = rows[index]
        if not belongs_to(rows, index, start):
            continue  # another step's answer, wherever it landed
        kind = row.get("type")
        if kind == "tool_result":
            responses = _dicts(row.get("tool_approval_responses"))
        elif kind == "tool_approval_response":
            responses = [dict(row)]
        else:
            continue
        for response in responses:
            if isinstance(response.get("approval_id"), str) and _counts(
                acted, response["approval_id"], index
            ):
                approval = response["approval_id"]
                found = decisions.get(approval)
                # ANY DENY IS FINAL (G-401): the first decision stands, unless
                # a later one denies an approve.
                if found is None or (found["approved"] and response.get("approved") is not True):
                    decisions[approval] = _decision(response)
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
    ids = [_text(call.get("id")) for call in step_calls(assistant)]
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
    calls = step_calls(assistant)
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
    A request is never paired with an ungated call that shares its id. A row
    carrying any harness mark (a start's ended row, a superseded or a 0.5.x
    replaced-step denial) is a denial, never calls the step ran (round 14).

    An approval belongs ONLY to the step that FIRST asked for it (G-408): a
    request repeating an approval id an earlier assistant row already asked
    for (a copy of a step a host recorded) asks for nothing, so one decision
    never decides a call of another step."""
    ix = _index(rows)
    memo = ix.requests.get(index)
    if memo is not None:
        return list(memo)
    assistant = rows[index]
    requests = _dicts(assistant.get("tool_approval_requests"))
    if any(ix.asked_earlier(request.get("approval_id"), index) for request in requests):
        assistant = {
            **assistant,
            "tool_approval_requests": [
                request
                for request in requests
                if not ix.asked_earlier(request.get("approval_id"), index)
            ],
        }
    calls = step_calls(assistant)
    ids = [_text(call.get("id")) for call in calls]
    # The calls the step RAN at once: those its rows answer BEFORE the first
    # decision (the reference's CallPairing), and those an UNGATED started
    # row names (G-387 round 16: approval id null). Only a call whose id
    # REPEATS in the step can be one a request must skip (G-387 round 13): a
    # call alone with its id that a request names was gated, so the request
    # pairs with it whatever answered it (the resolver's "No approval
    # response provided" included).
    before: list[Mapping[str, Any]] = []
    named: set[int] = set()
    for at in ix.owned.get(index, []):
        row = rows[at]
        if row.get("type") == "tool_approval_response" or _dicts(
            row.get("tool_approval_responses")
        ):
            break
        if any(
            honoured(row, mark) is not None for mark in (ENDED_BY_START, SUPERSEDED, REPLACED_STEP)
        ):
            continue
        started = _started_of(row) if ix.answers[at] == index else None
        if started is not None and started[2] is None:
            pick = next(
                (
                    i
                    for i, call in enumerate(calls)
                    if i not in named
                    and _text(call.get("id")) == started[0]
                    and _text(call.get("name")) == started[1]
                ),
                None,
            )
            if pick is not None:
                named.add(pick)
        before.append(row)
    ran = {call for call in [*call_results(calls, before), *named] if ids.count(ids[call]) > 1}
    pairs = paired_requests(assistant, ran)
    ix.requests[index] = pairs
    return list(pairs)


def asked_before(rows: Sequence[Mapping[str, Any]], index: int) -> set[str]:
    """Every approval id an assistant row (or a 0.1.0 request row) before
    ``rows[index]`` asked for."""
    asked: set[str] = set()
    for row in rows[:index]:
        asked |= _asks(row)
    return asked


def _asks(row: Mapping[str, Any]) -> set[str]:
    """The approval ids one row asks for: an assistant row's requests, or a
    0.1.0 request row's approvals."""
    if row.get("type") == "assistant":
        requests = _dicts(row.get("tool_approval_requests"))
        return {r["approval_id"] for r in requests if isinstance(r.get("approval_id"), str)}
    if row.get("type") == "tool_approval_request":
        approvals = _dicts(row.get("approvals"))
        return {a["id"] for a in approvals if isinstance(a.get("id"), str)}
    return set()


class _Index:
    """What every reader needs of one list of rows, in ONE forward pass
    (G-387 round 16): the steps, the first row asking each approval id, each
    row's honoured mark, the rows each step OWNS, and (filled on first use)
    each step's paired requests and started calls. Readers that ask the same
    rows many questions (the replay, the resolver, the pending readers) share
    it, so no reader rescans the thread per row or per step."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        count = len(rows)
        self.steps: list[int] = []
        self.first_ask: dict[str, int] = {}
        self.answers: list[int | None] = []
        self.marked: list[int | None] = []
        self.owned: dict[int, list[int]] = {}
        self.requests: dict[int, list[tuple[dict[str, Any], int | None]]] = {}
        self.started: dict[int, dict[int, str | None]] = {}
        self.started_at: dict[int, dict[int, int]] = {}
        self.acted: dict[str, int] | None = None
        self.asked_by: dict[str, int] | None = None
        current: int | None = None
        for index in range(count):
            row = rows[index]
            for approval in _asks(row):
                self.first_ask.setdefault(approval, index)
            named = _answers_step_raw(rows, index)
            marked = _marked_for_raw(rows, index, named)
            self.answers.append(named)
            self.marked.append(marked)
            if is_tool_step(row):
                self.steps.append(index)
                self.owned[index] = []
                current = index
                continue
            if marked is None:
                if current is not None:
                    self.owned[current].append(index)
            elif marked in self.owned and marked < index:
                self.owned[marked].append(index)

    def asked_earlier(self, approval: object, index: int) -> bool:
        """Whether a row before ``rows[index]`` asked for ``approval``."""
        return isinstance(approval, str) and self.first_ask.get(approval, index) < index


_INDEXES = threading.local()


def _index(rows: Sequence[Mapping[str, Any]]) -> _Index:
    """The :class:`_Index` of ``rows``, built once per list: kept for the
    last list read on this thread while it is the same object of the same
    length (each reader is handed the rows as read, never edits them in
    place)."""
    cached = getattr(_INDEXES, "entry", None)
    if cached is not None and cached[0] is rows and cached[1] == len(rows):
        found: _Index = cached[2]
        return found
    built = _Index(rows)
    _INDEXES.entry = (rows, len(rows), built)
    return built


#: The binding on every stored approval request, LAST (G-387 round 15, the
#: PHP reference's shape): ``{tool_name, args_digest}`` of the call the
#: request was issued for, ``args_digest`` being ``sha256:`` and the hex
#: SHA-256 of the RFC 8785 canonical JSON of its arguments. A call runs only
#: when its request's binding matches it.
CALL_BINDING = "harness_call"

#: The one outcome of an approved call that may not run because its request
#: does not bind it, or because it is no longer open: the reference's text.
NOT_THE_APPROVED_CALL = (
    "Not run: this call is not the one that was approved, or it is no longer open."
)


def binding_digest(arguments: object) -> str:
    """The digest a binding names for ``arguments``: of the decoded value when
    they are stored as a JSON string, and of ``{}`` for anything that is not
    an object or an array."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {}
    if not isinstance(arguments, (dict, list)):
        arguments = {}
    return args_digest(arguments)


def bound_step(row: dict[str, Any]) -> dict[str, Any]:
    """An assistant step with each approval request carrying its binding,
    LAST: the call it was issued for, paired by position (the first call
    with its ``tool_call_id`` an earlier request of the step has not taken).
    A request no call answers gets none. Any other row unchanged."""
    if not is_tool_step(row) or not isinstance(row.get("tool_approval_requests"), list):
        return row
    calls = step_calls(row)
    taken: set[int] = set()
    requests: list[Any] = []
    for request in row["tool_approval_requests"]:
        if not isinstance(request, dict):
            requests.append(request)
            continue
        wanted = request.get("tool_call_id")
        index = next(
            (
                i
                for i, call in enumerate(calls)
                if i not in taken and isinstance(wanted, str) and call.get("id") == wanted
            ),
            None,
        )
        bare = {key: value for key, value in request.items() if key != CALL_BINDING}
        if index is not None:
            taken.add(index)
            bare[CALL_BINDING] = {
                "tool_name": _text(calls[index].get("name")),
                "args_digest": binding_digest(calls[index].get("arguments")),
            }
        requests.append(bare)
    return {**row, "tool_approval_requests": requests}


def binds(request: Mapping[str, Any], call: Mapping[str, Any], held: object = None) -> bool:
    """Whether an approval request's binding binds ``call``: its tool name
    and the digest of its arguments as stored (and ``held``, the arguments
    the resolver holds, when given) match. A request with NO binding (a row
    written before the rule) binds the call it is paired with; one present
    but malformed never binds."""
    if CALL_BINDING not in request:
        return True
    binding = request.get(CALL_BINDING)
    if not isinstance(binding, dict):
        return False
    name, digest = binding.get("tool_name"), binding.get("args_digest")
    if not isinstance(name, str) or not isinstance(digest, str):
        return False
    if name != call.get("name") or digest != binding_digest(call.get("arguments")):
        return False
    return held is None or digest == binding_digest(held)


def held_results(rows: Sequence[Mapping[str, Any]], step: int) -> dict[tuple[str, str, str], int]:
    """How many results of each key (call id, tool name, canonical args) the
    step at index ``step`` already holds: the most any ONE row it owns holds
    (a resolution row restates; G-387 round 15, the fold's rule)."""
    held: dict[tuple[str, str, str], int] = {}
    for row in owned_rows(rows, step):
        if row.get("type") != "tool_result":
            continue
        normalised = _normalise(row, [])
        if normalised is None:
            continue
        here: dict[tuple[str, str, str], int] = {}
        for entry in normalised["tool_results"]:
            key = (entry["tool_call_id"], entry["tool_name"], _canonical(entry["args"]))
            here[key] = here.get(key, 0) + 1
        for key, count in here.items():
            held[key] = max(held.get(key, 0), count)
    return held


def new_results(
    rows: Sequence[Mapping[str, Any]], step: int, entries: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """``entries`` less every result the step at index ``step`` already holds:
    EXACTLY ONE STORED OUTCOME PER CALL. A key (call id, tool name, canonical
    args) keeps only its occurrences BEYOND the most one owned row already
    holds, occurrence by occurrence (G-387 round 16, the reference's rule): a
    new result for one of two IDENTICAL calls answers the first still open
    (:func:`call_results`), so no earlier result is restated."""
    held = held_results(rows, step)

    def key_of(entry: Mapping[str, Any]) -> tuple[str, str, str]:
        args = entry.get("args")
        return (
            _text(entry.get("tool_call_id")),
            _text(entry.get("tool_name")),
            _canonical(args if isinstance(args, dict) else {}),
        )

    seen: dict[tuple[str, str, str], int] = {}
    kept: list[dict[str, Any]] = []
    for entry in entries:
        key = key_of(entry)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > held.get(key, 0):
            kept.append(dict(entry))
    return kept


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
            index = _answered_call(calls, entry, taken, matched)
            if index is not None:
                taken.add(index)
                matched.setdefault(index, entry)
    return matched


def _answered_call(
    calls: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
    taken: Collection[int],
    earlier: Mapping[int, Mapping[str, Any]] | None = None,
) -> int | None:
    """The call one stored result answers: see :func:`call_results`.

    Among the calls it matches best, the first call still without an
    outcome; else the first not yet taken in its own row (G-387 round 16, the
    reference's CallPairing: a runtime row never restates the results its
    step holds, so a new result for one of two calls with the same arguments
    answers the one still open)."""
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
            if earlier is None:
                return found[0]
            still = [i for i in found if i not in earlier]
            return (still or found)[0]
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
        for call in step_calls(assistant)
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
                for call in step_calls(row)
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
    """Two consecutive tool result rows as one (G-403).

    A later row's result for the SAME call replaces an earlier one, in the
    earlier one's position. The same call is the same call id, tool name and
    arguments, the n-th such result of a row replacing the n-th of the earlier
    one, never the call id alone: a provider may give two calls of one step the
    same id, and keyed by id one call's result would replace the other's.
    PHP main's ``ToolResultRuns`` (prism-harness#141) and the 0.5.x line
    (prism-harness-py#75) fold the same way.
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
