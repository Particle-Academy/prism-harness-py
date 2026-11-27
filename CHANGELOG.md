# Changelog

## Unreleased

### Changed

- One authority for a run's step allowance (G-387). How many model steps a run
  may take is worked out once, when the run begins, by the pure
  `RunBudgets.allowance` (`prism_harness.run_budgets`), and every consumer
  reads that one answer: the loop's step gate, the paused row, a runner start's
  gate, a child's tree hold and the plan's hold.
- A subagent child's step ceiling is the runner's grant: the least of its
  declared `max_steps`, what its parent had left and `max_steps_ceiling`. It
  is saved with the child's binding and is the DELEGATION's lifetime budget.
  The steps the child takes after the runner starts it are counted durably
  beside the ceiling, and every run on the child claims what it may take from
  that count in one locked write. So answers, new messages and host
  `RunContext`s on the child, sequential or at once, never total more than the
  ceiling; a host context can only lower it. A hold whose run never settles
  is charged in full once the lock TTL has passed; when that run does settle,
  the steps it took replace the full charge, so it is counted once. Only a
  runner start writes a fresh count, with the ceiling it saves.
- `::sub::` in a session scope is reserved for subagent children. A subagent
  child session (its scope is a child's, `...::sub::<name>`, or the
  runner saved references, a binding or a tree with it) with no saved ceiling,
  or a saved ceiling or step count that is missing a field or is not a whole
  number, fails closed: it acts on its recorded decisions and takes no model
  step, and a new message to it is refused before anything is recorded. This
  covers a child parked by v0.5.x, which saved only its thread. Only a
  session that is not a subagent child runs on its mode's `max_steps`.
- A run that parks on an approval writes the steps it has used, its children's
  included, on the row it parks on (`harness_steps_used`). Its resume counts
  the larger of that, the child's durable count and the session's own
  transcript count. A value there that is not a whole number from 0 to
  2^53-1 fails closed, and so does a child row without one; a root row
  written before the field existed uses the transcript count.
  `harness_steps_left` is still written, including a shared plan budget's
  narrowing, and nothing reads it to decide a budget.
- `max_steps_ceiling` is absolute on a run's whole count under every context.
  A root under a host `RunContext` runs on the host's budget (what the host
  context has left), not its mode's `max_steps`; `max_steps_ceiling` still
  bounds it.
- `RunBudget` refuses a `max_steps` that is not a whole number of zero or more
  (`TypeError` / `ValueError`), and a subagent declaration's `max_steps` must
  be one (`mode_malformed`). `RunBudgets.allowance` reads any input that is
  not a finite number as 0.
- One decision rule: acting on an approved call makes no provider call, so no
  budget (steps or cost; the run's own, its run tree's or a shared plan
  budget's) refuses a decision or a resume. The resume runs the approved call
  and takes no model step, ending `budget_exhausted`. An approved subagent
  call on a spent run is acted on: its result is the runner's "did not run"
  refusal, and no child model step is taken. A turn with a NEW prompt and
  nothing left is refused before anything is recorded: `plan_budget_exhausted`
  when the plan's grant is what is spent, otherwise `run_not_permitted`.
- A decision the resuming turn could not act on is refused before anything is
  recorded: `child_binding_missing` for a linked child with no saved binding,
  `run_not_permitted` for one nested at or past the depth ceiling.
- A turn whose run context's ledger is cancelled, or whose context's time
  limit is spent, acts on no recorded decision (an approval or a denial) and
  leaves the thread unchanged: it ends on the cancellation or the time before
  any model step, and the decisions stay recorded for a later resume.
- A runner start (`SubagentRunner.run`, and the subagent tool that calls it)
  is decided by ONE start gate, the pure `RunStarts.gate`
  (`prism_harness.run_starts`). Every fact a start depends on is read first,
  with no write of any kind, and the gate is asked once, in a fixed order: a
  task that is not a string (`denied`); a blank task, one made only of
  U+0009-U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000-U+200D, U+2028,
  U+2029, U+202F, U+205F, U+2060, U+3000 and U+FEFF (one explicit set, the
  same in every language, `denied`); nesting at the depth ceiling (`denied`);
  a cancel (`cancelled`); spent time, no steps left, a spent cost cap or
  agent spend ceiling, or a cost cap that cannot be metered (`exhausted`); a
  turn on the child still holding steps from its count, or a pending approval
  on it a person could still answer (`denied`); and a run tree that cannot
  fund the start (`exhausted`). The cancel, time, steps and cost of the
  context the host wired a subagent tool with are facts of that gate too, so
  they stop the tool inside a turn. The reason the parent model reads is
  written by the harness, never the host's cancel text.
- A runner start is narrowed from the ledger as its gate read it: steps spent
  on the run context after that read (by another turn sharing the ledger)
  count against the child, so the context's `max_steps` holds however the
  start is timed.
- A refused start writes nothing: not the child's records (its binding,
  ceiling, references and count), not a run state or run history entry, not
  its mode, not the parent's tree record. A start the gate lets through
  commits in one order: the tree hold, a conditional locked write that
  refuses the start (`exhausted`) when the tree's grant shrank since the
  read; then the child's records, in one locked write that refuses the start
  (`denied`) when a turn claimed steps on the child since the read, or when
  the child now has a pending approval a person could answer (re-read under
  that lock with the gate's rule), releasing the tree hold; then the run
  itself. The child's records and its session mode are written together in
  that one locked write, and everything it writes is PROVISIONAL: marked with
  the start's run id, and treated as absent by every reader (a decision, a
  resume, a new message or host turn on the child, the start gate, the
  delegation count, the paused-row reads) until that run's task row is in the
  child's thread. Confirmation is read from the thread, never written, so a
  start that fails before its task row needs no second write to stay
  harmless: the child reads as one with no binding (`child_binding_missing`),
  no saved ceiling and no count, and a re-delegation is not blocked once the
  start's hold has expired. A record with no mark (written by an earlier
  version) reads as it did; a mark that is not a run id never confirms. The
  undo that puts the child's records and mode back after such a failure is
  housekeeping only, and its failure changes nothing a reader sees. The run a
  start begins is provisional the same way: until its task row, the child's
  run state still reads as the run it is parked on (or as no run, when that
  run's state has expired: the attempt's outcome is written with its mark),
  and the attempted run is absent from its run history. So is the session
  mode it writes: `session.mode()` reads the mode from before the start until
  the task row exists, and a mode the host sets afterwards reads as set.
  The start's OWN run is the one exception: from its commit to the end of that
  run, every reader of the Session object the run uses (a tool provider or
  factory, the authorizer, the plan service, a tool body) reads what the
  start committed, its mode, binding, run tree references, depth, ceiling and
  run, so host code that decides tools from `session.mode()` sees the mode the
  child runs in. Every reader outside that run (another Session object, a
  host turn or a decision, even one entered from inside the run) keeps the
  provisional rule. The view is held in memory only and cleared with the run,
  before a failed start is cleaned up.
  Absent never widens anything: while a start is unconfirmed the child still
  reads as in the run tree and shared plan budget it was last CONFIRMED in,
  so a later start the host drives keeps them, and the child's turns stay
  bounded by that tree and charged to that plan budget. Once the task is
  recorded, a later cancel or time-out only stops its model steps.
- A runner start acts on no decision recorded on the child, whenever it was
  recorded. Its task row ends every call still pending there, recorded as not
  run in the same write, so a re-delegation never runs a call a bare resume
  would refuse, and no later turn acts on it either. A call it ended reads as
  ended, never as approved (a plan reading the thread sees no decision on
  it), and the `ToolDenied` events for those calls are emitted only after the
  task row recording them is durable and the start is marked recorded, so a
  listener that raises cannot undo a start whose task row exists. "Ended by a
  start" is recognised only by `harness_ended_by_start`, a field the harness
  writes last on the row that closes those calls, `{approval_ids: [...]}`
  naming the approval ids it ended (the PHP reference's shape, the same in
  all three languages and pinned by the `harness-thread-rows` corpus; no other
  shape is read), and only for the approvals the turn before it asked for: never by a
  result's text (a tool may return any text) and never by a tool call id
  (providers reuse them across turns), so a person's approval of a call that
  ran, or of a later call under a reused id, always stands.
- A decision a person recorded that a resume could still act on (an approved
  or denied call with no result yet, on a child whose approvals are
  answerable) refuses a runner start exactly like a pending one
  (`child_waiting`, `denied`, nothing written), at the start gate and again
  under the child write's lock: a model cannot void a person's decision by
  delegating. The host resumes the child to act on it.
- A failed run's cleanup (undoing a start, failing the run, `RunFailed`, the
  plan and count bookkeeping) runs first and step by step, and an exception's
  attributes are read only guarded: a host exception that never ran
  `HarnessError`'s constructor, or whose `code` or `document` raises, is
  named by its class and skips none of it. A `BaseException` that is not an
  `Exception` (host code stopping the process) still releases the start's and
  the run's holds, marks the run failed and emits `RunFailed` (its class
  only), and propagates unchanged. Every run event, `RunStarted` included, is
  emitted inside the run's own error handling, so anything raised from it (a
  listener's `BaseException`, an emit that fails) fails that run, root or
  child, by class name, releases its holds and emits `RunFailed`; a run never
  stays `running` because of a listener. The holds are released BEFORE
  `RunFailed` is emitted, so a listener never observes one about to go. A
  raising listener is warned about by its class name only, never its text,
  and a listener error whose text cannot be read does not break the emit. The harness's own log lines name a class
  by the same identifier rule as a failed run. A read that raises refuses the start as `failed`, naming the
  exception's class, with nothing written; the answerability read counts only
  its own answers (no saved binding, too deep) as unanswerable, and any other
  error refuses the start. A child stuck on an approval no decision could act
  on can be started again.
- A tree hold records the start that took it, and only that start releases or
  hands it over: a re-entrant start (from a `RunStarted` listener) that fails
  never releases the hold of the start it re-entered.
- A runner start's own run reads the records it committed on the Session
  object it uses; only the object that committed that start enters this
  view. Any other Session object, whatever run id it is handed, keeps the
  provisional rule.
- Everything the runner does around a child (building its session, reading
  its thread, its count, the budget arithmetic) is framed: any exception comes
  back to the parent as a `failed` result naming the exception's class (and a
  harness error's code), never its message, so no store, host or provider
  text reaches the model. That covers the subagent tool's narrowing of its
  host context and everything inside the child's own run (its store writes,
  its count, its provider client).
- What the runner tells the parent model is chosen by code from fixed
  sentences only: the gate's reasons; for a child that stopped early, one
  sentence per stop (`AgentResponse.stopped_code`, new: `cancelled`, `steps`,
  `cost`, `cost_unenforceable`, `time`), never `stopped_because`, which keeps
  the host's own cancel reason for the host; the child agent's spend ceiling as
  a fixed sentence; and for anything else the exception's class and a code the
  harness itself raises. A refusal's outcome is one of `denied`, `cancelled`
  or `exhausted`, anything else is `failed`, and an exception's `code` is read
  once, guarded. A task lease refused on a cancelled run no longer repeats the
  cancel reason in its message.
- A failed run's state (`session.run()["failure"]`) and its `RunFailed` event
  record the exception's class, the harness error code where there is one, and
  a fixed description (`"HarnessError (model_step_failed): the run failed."`),
  never the exception's message: the harness has no content capture of its
  own, so text a provider, store, host or tool wrote is never kept or emitted.
  A class is named only by an identifier-shaped name (ASCII letters, digits
  and `_`): a class whose `__name__` was set to anything else (a path, a
  dotted module path, text with spaces or line breaks) is named by its nearest
  ancestor that has one, so no file path or line reaches a framed result, the
  parent model, `RunFailed` or a run row.
  The exception itself still reaches the caller. A schema violation still
  records its code.
- `RunBudget` refuses a `max_cost_usd` or `max_seconds` that is NaN, infinite
  or negative (`ValueError`), and so does `RunBudget.from_config`. A limit that
  is present but not a number (text such as `"60"`, `"NAN"`, `"INF"` or
  `"0,50"`, a boolean, a list) is refused the same way, never read as no
  limit; a subagent declaring one makes its mode malformed. A limit
  forged past the constructor is read as spent by the loop's gate and the start
  gate, never as no limit.
- `send_structured()` returns a `StructuredAgentResponse`, with no document,
  when the run parks on an approval or runs out of steps.
- A call a runner start ended reads ended whatever else its step wrote: any
  tool result row after the approval's request and before the next assistant
  step that carries the harness's `harness_ended_by_start` mark naming the
  approval ends it, not only the first row carrying the call's id.
- A step whose tool call ids repeat records every decision on it as a denial,
  with the same text its calls are given ("Tool call was not run because
  another call in this step has the same id"), as the PHP reference does, so
  neither a plan nor a later turn reads one as approved. Results are matched
  to the step's calls by position, never by call id alone: a gated call that
  shares its id with an ungated call that already ran is still pending, so a
  resume denies it and a runner start ends it with its own approval, and its
  approval request is never paired with the ungated call.
- A runner start whose task row is written stands, whatever fails after the
  write (a lock release, an event, a read while its run ends): the harness
  records the task row as written at the write itself, and only a start that
  fails before it is put back. The Session object that committed a start can
  enter that start's own-run view only until its run ends. A run stopped by
  an exception that is not an `Exception` runs the plan's after-turn hook too.

- A decision only ever applies to the step whose approvals were requested:
  every reader that selects the paused step (the decision checks, the
  repeated-id guard, the resolver and the plan readers) uses one rule, and a
  row recorded after the park ends that step for all of them at once. Every
  result row after the step counts, so a row in between never makes an
  answered call read as unanswered.
- A parked step is superseded by a new message, a later assistant step with
  tool calls, or an assistant or user row recorded outside the runs that own
  it (the run that recorded it and every run that recorded one of its
  results); a row a caller records is outside whatever run id it carries. Each of its calls that has no result is answered once, "Tool call
  was not run because a later step replaced it", whatever was decided; nothing of it runs and a
  later decision is refused. Only a resume (an empty prompt) acts on
  decisions.

- Threads the 0.5.x line wrote resume correctly: its `harness_replaced_step`
  row answers only the step at its position, wherever it sits, and its
  approvals read as no decision. An earlier step a later one replaced, or a
  last step a recorded row already superseded, with calls the 0.5.x line
  never answered, gets its one denial when main resumes it, and nothing of it
  runs.
- A stored result counts only for a call of the step whose segment it sits in
  (after that step's assistant row, before the next one with tool calls),
  matched there by its arguments, then its id and tool name, then its id. A
  result whose arguments match an earlier step's call exactly answers a
  same-id call of its own step, never the earlier call, so an approved call
  runs at most once whatever another writer stored. The earlier step's calls
  keep their one "replaced" denial. Pinned by seam tests and the approval
  pairing property test.
- A new message denies the paused step's approved calls that have not run
  yet, even when that turn then fails: the denials and the message's row are
  one write. A resume records a write-ahead `started` row immediately before it
  runs each approved call; a call that has started and has no result is
  answered-in-progress, so a row recorded meanwhile (a host row, another
  worker's new message, or the tool recording on its own session) is
  recorded at once, never waits and is never refused, and denies that call
  nothing. Its result names its step (`harness_answers_step`, honoured only on
  rows the harness wrote) and pairs to it wherever it lands. A started call
  with no result reads "Tool call was interrupted, it may have run" and is
  never run again. Every gated call ends with exactly one stored outcome, and
  a decision on a replaced step is refused at every moment.
- Every stored approval request is bound to its call (`harness_call`: the
  tool name and the digest of the arguments), and a call runs only when that
  still matches; an approval belongs only to the step that first asked for
  it. Any denial is final: a denial recorded after an approval that has not
  been acted on revokes it.
- One rule for what a step is: an assistant row whose `tool_calls` holds at
  least one object call. Any other row is a text row for every reader.
- The history sent to the model keeps each call's own result when two calls
  of one step share an id: a later row replaces only the result of the same
  call (same id, tool name and arguments). Each result is sent once, in its
  own step's place, even when another row sits between the rows holding it.
- A `harness_replaced_step` position written by the 0.5.x line is read as the
  row's thread position, so a thread whose positions do not start at 1 still
  reads its replaced step. New rows are numbered after the highest position.
- The runtime's own in-run recording is private and scoped to the one record
  call it makes; nothing else recorded meanwhile counts as inside the run.

### Parity

- harness-plans 0.1.9: the `step_allowance` op (hpl-0072..hpl-0083) runs
  `RunBudgets.allowance`; all twelve rows agree with the reference.
- harness-plans 0.1.10: the `start_gate` op (hpl-0084..hpl-0089) runs
  `RunStarts.gate`; all six rows agree with the reference.

## 0.6.0

Includes every fix released on the 0.5 line in 0.5.1, 0.5.2 and 0.5.3.

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

- A subagent child the runner starts again under a run tree the host built
  is no longer held against the run tree its session saved before (which
  refused it once that tree was spent, or charged it to that tree), and that
  saved tree is kept rather than replaced with nothing, so a later turn the
  runner does not start still continues it instead of a fresh root budget.
  The runner's own context is the runner's for the budget routes too, as in
  the PHP reference.
- A subagent child's depth is saved with its binding, and every later turn on
  the child runs at least that deep, a host context included, as in the PHP
  reference: a resumed child can no longer reset its nesting depth.

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
