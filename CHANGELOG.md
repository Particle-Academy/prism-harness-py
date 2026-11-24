# Changelog

## Unreleased

### Changed

- Approval requests are now written with `arguments_format: 1`, which records
  that the call's arguments were stored as the model sent them.
- An approved call whose request has no `arguments_format` and whose stored
  arguments are empty no longer runs on resume. 0.5.1 and earlier stored a
  call whose arguments arrived as a bare number or boolean as no arguments, so
  the two cannot be told apart. The call's result, recorded with no arguments,
  says the approval expired and asks the model to call the tool again; a new
  call needs a new approval. A `tool_approval_expired` event records it. Approvals pending
  with no arguments when you upgrade expire once, including ones written by the
  previous release.

### Fixed

- A decision for an approval the session is not waiting on is refused with
  `approval_not_pending` (G-354). Before, `record_approval()` wrote any id to
  the thread: a made-up id recorded while a call was pending did nothing for
  that call, which the resume then closed as "No approval response provided",
  and one recorded while nothing was pending invited a model turn with no new
  user message. An id the run never issued, or one already answered, is now
  refused before anything is written.
- A run's step limit (`max_steps`, and `max_steps_ceiling` for a named agent)
  now covers the whole run, including every pause for an approval and every
  resume after one. A resume (`send(session, "")`) continues the paused run's step count
  instead of starting a new one, also when another worker resumes it from
  persisted state. The count is read from the stored thread rows, so no new
  state is written. A resume that finds the limit already reached still runs
  the calls a person approved, then finishes with `budget_exhausted` without
  another model call. A new prompt starts a new run with its own limit, as
  before.

### Added

- `ToolPolicy`, a host-owned list of approval-gated tools that run without
  asking a person (G-326). Built with `ToolPolicy.from_json({"revision": ...,
  "allow": [{"server"?: ..., "tool": ...}], "honor": ...})`, the same JSON
  shape as the PHP and TypeScript harnesses. Passed per launch as
  `AgentRuntime(tool_policy=...)` and replaced live with
  `runtime.set_tool_policy()`; the default `ToolPolicy.none()` changes nothing.
  Matching is exact: a padded, look-alike or case-variant name never matches,
  an entry without `server` never matches a tool that has one, and there are
  no wildcards. Confirmation and approval tools, malformed entries, and tools
  declaring `tier = "destructive"` are refused with `tool_policy_refused`.
  `honor: false` turns the list off. A replacement applies from the next call;
  a call already waiting keeps waiting, and a call the old policy allowed is
  refused if the new one no longer does. The policy is never sent to the model
  and has no tool surface.
- A tool may carry optional `server` (the MCP server it comes from) and `tier`
  (`read_only`, `write`, `destructive`; advisory).
  `ToolRegistry.declarations()` lists the static tools.
- A tool that arrives through `register_provider` is scoped by the registry:
  its `server` is the provider's registered name, whatever the tool itself
  declares (`ScopedTool`). A provider that forgets to set `server`, or a tool
  that claims another server, can no longer make a provider's tool match an
  unscoped entry or another server's entry. Only a natively registered tool
  matches an entry without `server`. Name a provider
  (`register_provider(provider, owner, name)`) to give a policy a stable scope
  to list; an unnamed one is scoped as `provider:<n>`. `AuthorizedTool` now
  exposes `server` and `tier`, which it used to drop.
- Events `ToolAutoAllowed`, `ToolApprovalAsked`, `ToolApproved` and
  `ToolDenied` (`tool_auto_allowed` ... `tool_denied`) for every call to an
  approval-gated tool, carrying `policy_revision`, `server`, `tool`,
  `args_digest` (sha256 of the arguments' RFC 8785 canonical JSON; never the
  arguments), `decided_by` (`policy` or `person`) and `harness_agent`. They are
  emitted whether or not a policy is set.
- `canonical_json()` and `args_digest()`.
- Modes and subagent declarations take an optional `provider` and `model`. A run
  resolves each field as session override, then mode, then agent default, then
  global default.
- A subagent can run as another named agent with `agent: <id>`, only when the
  parent mode lists that id in its `agents`. An unlisted, padded or mixed-case id
  is refused with `mode_malformed` when the mode resolves, before any model call.
  The child's mode is looked up in the named agent.
- `agents.<id>.spend_ceiling` (positive finite USD) caps one agent's spend in a
  run tree, counted in the tree's own ledger. A call by that agent with no
  reported cost fails with `budget_unpriced`. Spend at the ceiling fails with
  `agent_spend_ceiling_exceeded`. A subagent stopped this way returns a framed
  result to its parent. Ceilings are per run tree; there is no per-day window.
- `RunStarted`, `RunFinished` and `RunFailed` carry `harness_agent` for a named
  agent's run. `HarnessEvents.to_dict()` leaves it out when it is `None`.
- `AgentRuntime.mode_for(session)` and `RunLedger.agent_cost_usd(agent)`.

### Breaking changes

- A subagent's child no longer copies the parent's session provider and model.
  The runner no longer writes `using_provider()` / `using_model()` onto the child
  session, and it no longer writes `"unknown"` there for a legacy parent. The
  child now takes the first of: the parent's session override (only with
  `inherit_session_model: True`), the subagent's own `provider` / `model`, the
  mode it runs in, the parent's mode (same agent only), the agent default, and
  the global default. This matches the PHP reference and prism-harness-ts.
- A child session written by an earlier version may still hold a copied
  override. A child run with a binding ignores its session's override slot, so
  that stored value no longer applies.

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
