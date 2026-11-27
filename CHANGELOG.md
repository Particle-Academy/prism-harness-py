# Changelog

## 0.5.4 (unreleased)

Security fixes on the 0.5 line. Each tool call is now decided only by its own approval, and gets exactly one outcome.

### Fixed

- If a resumed run's pending step holds two tool calls with the same call id (an empty id counts), none of the step's approval-gated calls run, whatever was decided. The same applies when two approval requests name the same call id, or share one approval id. Each of those calls gets its own result, `Tool call was not run because another call in this step has the same id`, and no later decision or resume runs it. When the ids in a step are distinct, each call is decided on its own approval, as before.
- Stored results are paired with the step's tool calls by position, never through a map keyed by the call id, whether their arguments were stored as a map or as a JSON string. A result counts only for the calls of the step it was recorded after, never for another step's. A call that has its own result never gets a second outcome and never runs again, whatever arguments that result was stored with, and another call's result never ends it.
- When two calls of one step share an id, the history sent to the model keeps each call's own result; a later row replaces only the result of the same call (same id, tool name and arguments).
- The history sent to the model holds each of a step's results once, in the step's place, even when other rows were recorded between its result rows.
- Results and decisions are read from every tool result row recorded after the step, not only the first group of them. A call that already has a result never runs again on a later resume.
- A paused step is replaced when a later step with tool calls is recorded after it, or when the application records a user or assistant message on the thread while the approval is pending. Each of the replaced step's approval-gated calls that has no result gets `Tool call was not run because a later step replaced it`, once, and none of them runs. A decision recorded for a replaced step runs nothing.
- An approved call refused because its tool is not available to the run, or because it is not authorized, is no longer listed in the run's `tool_calls`.
- `send_structured` returns a `StructuredAgentResponse` with `structured` `None` when the run stops for an approval or runs out of steps, instead of failing with an `AssertionError`.

### Upgrading

- A thread holding a paused step that was already replaced gets a one-time row on its first resume, answering each unanswered approval-gated call of that step with `Tool call was not run because a later step replaced it`. The row is not sent to the model.
- If your application records user or assistant messages while an approval is pending, record the decisions and resume first.

## 0.5.3

Security-only release on the 0.5 line. Nothing else from main is included.

### Changed

- Approval requests are now written with `arguments_format: 1`, which records
  that the call's arguments were stored as the model sent them.
- An approved call whose request has no `arguments_format` and whose stored
  arguments are empty no longer runs on resume. 0.5.2 and earlier stored a
  call that the client handed over as no arguments as `{}`, and
  prism-ai-core 0.10.0 and earlier handed over some garbled calls that way (a
  Mistral argument that was not a string, an Anthropic input of `false` or
  `0`), so the two cannot be told apart. The call's result, recorded with no
  arguments, says the approval expired and asks the model to call the tool
  again; a new call needs a new approval. A `ToolApprovalExpired` event
  (`run_id`, `session_key`, `tool`, `args_digest` of no arguments,
  `decided_by="policy"`; never the arguments) records it. Only the turn being
  resumed decides this, so a call id reused across turns does not expire a
  marked approval.

### Upgrading

- **One-time expiry.** Approvals that are pending when you upgrade, on a call
  stored with no arguments, expire once instead of running, including ones
  written by 0.5.2, which wrote no marker either. The model is told to call
  the tool again, and the person is asked again. Approvals on calls with
  named arguments are not affected. Resuming a thread written by another
  language's harness is not supported for this check.
- If your code writes approval request rows itself, write
  `arguments_format: 1` only for calls stored exactly as the model sent them.

## 0.5.2

### Fixed

- A run's step limit (`max_steps`, and `max_steps_ceiling` for a named agent)
  now covers the whole run, including every pause for an approval and every
  resume after one. A resume (`send(session, "")`) continues the paused run's
  step count instead of starting a new one, also when another worker resumes
  it from persisted state. The count is read from the stored thread rows, so
  no new state is written. A resume that finds the limit already reached still
  runs the calls a person approved, then finishes with `budget_exhausted`
  without another model call. A new prompt starts a new run with its own
  limit, as before.
- A tool call whose arguments are not a JSON object (unparsed text, a list,
  a number, a map with integer-like keys, or values a follow-up request could
  not carry) no longer fails the turn. It is recorded with no arguments, asks
  nobody to approve it, is not passed to a call policy or handler, and is
  answered to the model as not run. The same check applies when an approved
  call is resumed from stored rows. An empty string, `0`, `null`, `[]` and
  `{}` all mean no arguments, and a tool taking none still runs with `{}`.
  JSON text naming values is read as those values.

No breaking changes.

## 0.5.1

### Fixed

- Resuming a turn that paused for approval offers exactly the tools that turn
  offered, intersected with the mode's current tools.

No breaking changes.

## 0.5.0

### Fixed

- Fixed: on Windows, a lock cleanup refused because another worker still holds the file now waits instead of crashing.

### Breaking changes

- Caller-supplied tool names now narrow a legacy session's mode list by default,
  through both `send()` and `send_structured()`. A name outside the ceiling is
  refused before a model call; an explicit empty list offers no tools.
- To retain legacy replacement behavior, explicitly set
  `allow_legacy_tool_list_widening=True`. This also preserves its previous
  empty-list fallback. The authorizer still applies, and named agents always
  enforce their mode ceiling.
- Remove an explicit `narrow_requested_tools=False`; it now raises a migration
  error naming the new opt-out. Explicit `True` remains valid, but combining it
  with the widening opt-out is a configuration error.

Legacy duplicate-registration and address defaults retain their existing behavior.
