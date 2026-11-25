# Changelog

## Unreleased

### Added

- `binding_route()`: the one rule for which provider/model binding a turn
  runs on and whether it is saved with the session, now asked by the runtime
  and pinned by the harness-plans corpus (hpl-0056..hpl-0058) in all three
  languages. Behaviour is unchanged.
- Plan budgets (G-366): the `Plans` config's `budget.mode` is `per_step` (the
  default, unchanged behaviour), `shared` (one ledger of `max_steps` model
  steps and optionally `max_cost_usd` across the plan's own turns,
  kept in the plan store) or `none`. Host code can choose it per plan with
  `PlanHandle.use_budget()`; `PlanHandle.budget()` reports it. A spent shared
  budget blocks the plan with `plan_budget_exhausted`, emits
  `PlanBudgetExhausted` once, and cannot be resumed. See "Plan budgets" in the
  README.
- A shared plan budget charges only the plan's own turns: the turn
  `advance()` runs for the step it claimed, the resume of an approval that step
  asked for, and a subagent child those turns started, resumed outside the
  runner. A person chatting while a step waits is no longer charged to the plan
  or refused for it. A decision on the step's approval is refused with
  `plan_budget_exhausted`, recording nothing, once the shared budget is spent,
  and the next `advance()` blocks the plan.
- A subagent child started inside a turn spends from the turn's run context,
  as the reference does: its steps and cost count against the turn, a shared
  plan budget included. The `RunContext` given to `SubagentRunner.tool()` can
  only narrow the turn's limits, never widen them.
- A subagent child resumed outside the runner continues the run tree it was
  started in instead of starting a fresh root budget. The tree's allowance is
  kept in the durable store from the first child on; the resumed child holds
  what is left, settles what it spent, and is refused (`run_not_permitted`)
  once it is spent, and a decision on its approval is refused the same way
  before anything is recorded.
- Every turn on a subagent child that the runner did not start continues the
  child's run tree and shared plan budget, not only an empty-prompt resume: a
  new message sent to the child instead of an answer no longer starts a fresh
  root budget, and a `RunContext` the host hands `send()` for the child is
  narrowed to what the tree and the plan budget have left. The references are
  saved with the child session (`Session.child_refs()`) when the runner starts
  it. Which budgets a turn spends from is one pure rule,
  `plans.budget.route()`, pinned by the harness-plans corpus 0.1.5
  (hpl-0053..hpl-0055).
- A subagent tool built for one session no longer borrows another session's
  running turn: `current_turn_run()` takes the tool's session key.

- Agent plans (G-360). An agent proposes a plan, a person approves it, and
  `runtime.plan(session).advance()` works through it one step per turn. Off
  unless both `AgentRuntime(plans=Plans({"enabled": True}))` and the mode's
  `plans` flag are on. See "Agent plans" in the README. This port does not
  stream `plan_updated` steps; it emits a `PlanUpdated` event instead.
- `AgentRuntime.decide()` records an approval decision and lets a plan read it.
- No `ToolPolicy` may name a plan tool, in any spelling.

### Changed

- Approval requests are now written with `arguments_format: 1`, which records
  that the call's arguments were stored as the model sent them.
- An approved call whose request has no `arguments_format` and whose stored
  arguments are empty no longer runs on resume. 0.5.2 and earlier stored a
  call whose arguments arrived as a bare number or boolean as no arguments, so
  the two cannot be told apart. The call's result, recorded with no arguments,
  says the approval expired and asks the model to call the tool again; a new
  call needs a new approval. A `tool_approval_expired` event records it. Approvals pending
  with no arguments when you upgrade expire once, including ones written by the
  previous release.

### Fixed

- A subagent child the runner starts under a run tree the HOST built (the
  parent's turn was handed a `RunContext` the host made) now saves its
  provider/model binding with the child session, as the PHP reference does.
  Before, only a child under a tree this runtime opened was saved, so the
  child's next turn outside the runner ran on the default provider. Only the
  context the runner itself builds and marks counts as the runner's; a
  context the host builds, or binds again, never saves a binding. Pinned by
  harness-plans hpl-0059 and hpl-0060.

- A subagent child's provider and model binding is now kept with the child
  session's durable data, as prism-harness keeps it, not only on the row a
  paused run stopped on. A new message to a child session, a resume, and a
  turn under a context the host built all run on the binding the runner gave
  the child; before, a new message resolved its provider and model like a
  root run. A child session with saved run tree references and no saved
  binding is refused with `child_binding_missing`, the reference's code,
  before any request. A context the host supplies never replaces the binding
  or run tree a child saved: only the runner's own context writes them.

- Two decisions for the same approval can no longer both be recorded.
  `record_approval()` and `AgentRuntime.decide()` now check that the approval
  is pending and write the decision under the session lock, so a second answer
  (a double click, two reviewers, a retry) waits and is refused with
  `approval_not_pending`. A thread that already holds two decisions for one id
  is read the same way everywhere that decides what runs: the first one
  recorded counts, for the resumed call, for the plan, and for the plan tool's
  own check. The thread view shown to a provider is unchanged.
  Recording a decision now takes the session lock, which is not re-entrant: a
  host that calls `record_approval()` or `decide()` from inside its own
  `session.lock(...)` callback waits for the lock and then gets
  `session_locked`, with nothing written. Record the decision outside the
  lock. The module-level `record_approval()` waits 30 seconds with a
  300-second lock TTL; `AgentRuntime.decide()` uses the runtime's `lock_ttl`
  and `lock_wait`.
- Tool-call arguments nested as deep as the harness accepts (just under 500
  levels) no longer make the approval path raise `RecursionError`. The args
  digest is computed without recursion. A call whose arguments cannot be
  digested is answered as not run, like a call whose arguments are not a JSON
  object, so it can no longer stop every later `send()` of the session.
- A plan tool the authorizer's offer policy withholds is now a tool the run
  does not have. Before, a model calling `plan_draft` (or another plan tool)
  that it was never offered still recorded a proposed plan and asked a person.
- A named agent's `spend_ceiling` now holds across an approval pause: the
  resume counts what the agent spent before the pause. A paused approval row
  records it as `harness_agent_spend`; a malformed value is counted as an
  unpriced call, so a ceiling refuses with `budget_unpriced`. Each new
  `send()` and each `plan().advance()` without a shared `RunContext` is still
  its own run tree; see "Spend ceilings per agent" in the README.
- A subagent child resumed after an approval pause outside the subagent
  runner keeps the provider and model its runner bound it to, recorded on the
  paused row as `harness_binding`, instead of resolving them as a root run.
- A resume reads stored calls with unusable arguments from the turn it
  resumes only. Before, an older turn's unusable call refused a later approved
  call that reused its id.
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
