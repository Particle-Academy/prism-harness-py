# Changelog

## Unreleased

### Fixed

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
- A tool may carry optional `server` (the MCP server it comes from, set by the
  code that connected it) and `tier` (`read_only`, `write`, `destructive`;
  advisory). `ToolRegistry.declarations()` lists the static tools.
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
