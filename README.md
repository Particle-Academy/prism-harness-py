# Prism Harness for Python

Durable agent sessions — threads, session state and store drivers. The Python
port of [`particle-academy/prism-harness`](https://github.com/Particle-Academy/prism-harness).

Zero runtime dependencies. Python 3.10+.

```
pip install prism-ai-harness
```

```python
from prism_harness import FileSessionStore, MemorySessionStore, Participant, PrismHarness

harness = PrismHarness(
    drivers={
        "memory": MemorySessionStore,
        "files": lambda: FileSessionStore("./storage/harness"),
    },
    stores={"ephemeral": "memory", "durable": "files"},
)

session = harness.for_(Participant("User", 7)).session("support")
session.using_mode("plan").using_model("claude-sonnet-5-5")


def advance(live):
    live.begin_run("run-1", "plan", "anthropic", "claude-sonnet-5-5")


session.lock(advance)  # whatever must not happen twice
```

`for` is a Python keyword, so the method is `for_` — forced rather than chosen,
like `Media.as_` in `prism-ai-core`.

## Named agents

Choose a stable id in trusted application code, using [a-z][a-z0-9_-]{0,63}.
Calls without an agent keep their existing addresses. Named agents have separate
tools, providers, modes, session state, threads and default task lists.

```python
from prism_harness import AgentRuntime, LlmResponse, ModeRegistry, ToolRegistry


class Lookup:
    name = "lookup"

    def handle(self, args: dict[str, object]) -> str:
        return "found"


tools, modes = ToolRegistry(), ModeRegistry()
tools.for_agent("support").register(Lookup(), "app.support")
modes.for_agent("support").register("reply", {"tools": ["lookup"]}, "app.support").set_default(
    "reply"
)
runtime = AgentRuntime(
    client=lambda _: LlmResponse(text="done", finish_reason="stop"),
    tools=tools,
    modes=modes,
)  # Replace the synthetic client when connecting a model.
session = harness.for_agent("support").for_(Participant("User", 7)).session("case:42")
descriptors = runtime.effective_tools(session, "reply")
```

A harness constructed with runtime also
offers for_agent(...).effective_tools(mode, participant, scope). The descriptors
contain name, owner and source from the model offer resolver. Inspection creates
no session, thread or index entry, invokes no handler and makes no model/provider
request or logical state change. Reads may physically purge expired entries,
exactly as an ordinary store read does. Explicit state writes during resolution
refuse.

The mode list is the ceiling for named and legacy sessions. Caller-supplied
names narrow it; omitting names uses the mode's full list, and an explicit empty
list offers no tools. A requested `*` selects the mode's ceiling. Other names
outside it fail before the model is called.

Legacy callers that intentionally need the previous replacement behavior can
set `AgentRuntime(..., allow_legacy_tool_list_widening=True)`. This permits
widening only in the legacy namespace; the configured authorizer still filters
the offer and tool calls. This opt-out also preserves the old empty-list
fallback. Named agents always retain their ceiling. Remove an old explicit
`narrow_requested_tools=False`: it now raises a migration error. Explicit `True`
remains supported, and conflicts with the widening opt-out.
Named agents inherit no legacy authority. Builtins require explicit registration
and mode selection. A named-agent repeated registration fails even for the same owner;
replace(replacement, predecessor_owner, owner) makes intended replacement
explicit. Mode replacement likewise names its predecessor.

Legacy registries retain duplicate overwrite and existing cross-kind precedence.
Use `ToolRegistry(refuse_duplicates=True)` and
`ModeRegistry(..., refuse_duplicates=True)` to refuse legacy duplicates.
Named session addresses are immutable and validated; legacy participant/scope
values retain their existing mutable behavior.

AgentRuntime's agents map supports provider/model, max_steps_ceiling,
max_output_tokens_per_call and lock_ttl/lock_wait. Unset scalars use config's
fallback; explicit None clears nullable defaults. Caps narrow the existing
RunContext ledger and output budget. ToolAuthorizer's agents map supplies
enabled/offer/call overrides with invocation-time rechecks. PrismHarness's agents
map separately controls ttl_seconds, default_scope and default_task_list.
ParticipantRef requires a registered ParticipantAdapters actor resolver; child
construction retains its parent's bound actor. Plain Participant values continue
to work.

## Resolved, never held

A server handles a request and moves on, so a session cannot be an object kept
in memory the way a single-process agent's is. Every call rebuilds one from a
store, which is what makes a fresh worker see the same mode, model and
conversation as the request that set them.

## The two halves

| Slot | Holds | Losing it means |
|---|---|---|
| `ephemeral` | active mode, selected model, run bookkeeping | falls back to a default |
| `durable` | threads, stored capabilities | work is gone |

**A store that reports itself volatile is REFUSED for the durable slot**, at the
moment a session is opened. That is the guard the package exists for: a cache is
disposable by definition, and the durable slot holds approvals a human has not
answered yet.

Construct a `PrismHarness` with no drivers and it works — and then refuses
durable state, loudly, with a message that names the fix. A package that
silently accepted an in-memory durable store would pass every test in one
process and lose a half-executed action the first time it ran on two.

## The same address as PHP

`Session.key()` is byte for byte what the reference builds: `session:` plus the
sha1 of the participant type truncated to 12, the participant id, and the scope.
Matching exactly is what lets a PHP app and a Python agent **share one store and
resolve the same session**.

## Threads

`record()` assigns positions inside the store lock, reading the current length
and writing the new messages as one operation. Two turns landing concurrently
would otherwise both read position 4 and both write position 5, silently losing
a message — the race the reference tracks as prism-harness#2.

## Attachments

A turn can carry media alongside its prompt:

```python
from prism import Image

runtime.send(
    session,
    "What is wrong with this layout?",
    additional_content=[Image.from_base64(screenshot, "image/png")],
)
```

Pass `prism-ai-core` media objects or media already serialized. Attachments are stored
with the turn in the shape `UserMessage.from_dict()` rebuilds. Each one must be an
image, document, audio or video that carries its bytes, a provider file id or
document chunks. The same rules as the PHP reference and the TypeScript port
apply, pinned across all three by prism-parity's `harness-turn-attachments`
corpus.

**Refused, with a `HarnessError` whose `code` names the problem:**

| Code | When |
|---|---|
| `attachment_by_reference` | built from a URL or read from a file |
| `attachment_not_media` | anything other than those four media types |
| `attachment_empty` | no bytes, no file id and no chunks |
| `attachment_without_prompt` | attachments with an empty prompt |

A refusal happens before a run starts: no run, no events, nothing in the thread.

## Provider options per mode

A mode can declare `provider_options`. They reach your model client unchanged, as
`request.provider_options`, on every step of every run in that mode:

```python
from prism_harness import ModeRegistry

ModeRegistry({"modes": {"overseer": {"provider_options": {"thinking": {"type": "adaptive"}}}}})
```

A value that is not a map is refused when the mode is resolved, with
`mode_malformed`.

## Provider and model per mode and subagent

A mode, and a subagent declaration, can name a `provider` and a `model`. Each is
resolved on its own, so a mode can change the model and keep the agent's provider:

```python
ModeRegistry(
    {
        "agents": {
            "planner": {
                "owner": "app",
                "modes": {
                    "plan": {
                        "model": "big-model",
                        "agents": ["researcher"],
                        "subagents": {
                            "summarise": {"mode": "summarise", "model": "small-model"},
                            "research": {"mode": "research", "agent": "researcher"},
                        },
                    },
                    "summarise": {},
                },
            },
            "researcher": {"owner": "app", "modes": {"research": {}}},
        },
    }
)
```

A run takes the first of: the session override (`using_provider()` /
`using_model()`), the mode, the agent default (`agents=` on `AgentRuntime`),
the global default.

A subagent's child run takes the first of:

1. the parent's session override, only when the subagent sets
   `inherit_session_model: True`;
2. the subagent's own `provider` / `model`;
3. the mode the child runs in;
4. the parent's mode, only when the child stays in the parent's agent;
5. the child agent's default, then the global default.

The child's own session override slot is not consulted, and the runner never
writes one.

A subagent can run as another named agent with `agent: <id>`, but only when the
parent mode lists that id in `agents`. Its mode is looked up in that agent. An
agent id is never trimmed or case-folded: `" researcher"` and `"Researcher"` are
refused, like an agent the mode does not list. All of these fail with
`mode_malformed` when the mode resolves, before any model call.

### Spend ceilings per agent

`agents={"researcher": {"spend_ceiling": 0.50}}` caps what that agent may spend
in one run tree, in USD. It is counted in the tree's own ledger, beside every
other agent's spend; it is not a second account. Before each model call:

- a call by that agent that reported no cost fails the run with
  `budget_unpriced`, because a ceiling with no numbers cannot hold;
- spend at or past the ceiling fails it with `agent_spend_ceiling_exceeded`.

A subagent refused this way comes back to its parent as a framed result, and the
parent carries on. Ceilings are per run tree; there is no per-day window.

What one run tree is:

- A run that pauses for an approval and is resumed with `send(session, "")` is
  the same run. The approval row it paused on records what its agent had spent
  (`harness_agent_spend`) and the model steps it had used, its subagent
  children's included (`harness_steps_used`), and the resume counts both
  against its limits. A subagent child resumed this way also
  keeps the provider and model its runner bound it to (`harness_binding`), and
  continues the run tree it was started in (`harness_tree`): the first time a
  root run starts a child, the tree's allowance is kept in the durable store
  with the root run holding all of it, and the root run settles what the whole
  tree spent when it ends. The resumed child holds what the tree has left
  before it calls a model and settles what it spent when it ends. Once the
  allowance is spent (steps or cost) it takes no further model step; a
  decision on its approval is still accepted and acted on, because acting on a
  decision makes no provider call. A new message to it with nothing left is
  refused (`run_not_permitted`) before anything is recorded. The tree and the
  shared
  plan budget a child belongs to are also saved with the child session when
  the runner starts it, so EVERY turn on the child the runner did not start
  continues them: a new message instead of an answer as much as the answer,
  and a `RunContext` the host hands `send()`, which is narrowed to what the
  tree and the plan budget have left and can never widen them. A child started
  under a run context the host built outside a turn has no tree record and
  resumes as before.
- A subagent child's step ceiling is the runner's grant (the least of its
  declared `max_steps`, what its parent had left and `max_steps_ceiling`),
  saved with the child when the runner starts it. It is a lifetime budget for
  that delegation: the child's steps are counted durably from the runner's
  start, and every later turn on it the runner did not start (answers, new
  messages, host `RunContext`s, at once or in a row) claims from what is
  left, in one locked write. A host `RunContext` can lower it, never raise it.
  A new runner start begins a new count; the runner refuses to start a child
  while a turn on it still holds steps. A subagent child (its scope is
  `...::sub::<name>`, or the runner saved something with it) with no saved
  ceiling or count, or one that is not a whole number, takes no model step:
  that includes a child parked by v0.5.x.
- A host `RunContext` belongs to ONE session's turn. Its ledger is shared by
  reference, so a context handed to turns on several sessions charges every
  one of them (and their children) the steps all of them took on it. Build a
  context per turn.
- A root under a host `RunContext` runs on the host's budget (what that context
  has left), not its mode's `max_steps`; `max_steps_ceiling` still bounds it,
  absolute on the run's whole count.
- An approved subagent call on a run with no steps left is acted on: its
  result is the runner's "did not run: ... exhausted" refusal, and the child
  takes no model step. The runner also refuses a task that is not a string
  or is blank (only the codepoints listed in the changelog), and a delegation
  into a child session with a pending approval a person could still answer,
  so a model cannot dismiss a person's pending decision by delegating again.
  Anything that raises around a child comes back as a framed `failed` result
  naming the exception's class, never its message.
- A decision is refused before anything is recorded only when its resume could
  not act on it at all: a linked child with no saved binding
  (`child_binding_missing`) or one too deep (`run_not_permitted`). A turn
  whose context's ledger is cancelled runs no approved call.
- Every other `send()` starts a new run tree, with nothing spent.
- So does every `runtime.plan(session).advance()` that is not handed a
  `RunContext`: each step of a plan gets the full ceiling. To hold a whole plan
  to one ceiling, build one context and pass it to every advance:
  `context = RunContext.root(plan_run_id, RunBudget(max_steps))`, then
  `handle.advance(context)`. The ceiling is checked before each model call, so
  the call that crosses it is paid for.

Every event of a named agent's run carries `harness_agent`. A legacy run's events
leave it out of `HarnessEvents.to_dict()`, so their shape is unchanged.

## What your model client receives

`request.messages` is the thread, oldest first, in the shape the PHP reference
stores Prism's messages:

| `type` | Carries |
|---|---|
| `user` | `content`, and `additional_content` when the turn has attachments |
| `assistant` | `content`, `tool_calls` (`id`, `name`, `arguments`, `result_id`, `reasoning_id`, `reasoning_summary`), `additional_content`, `tool_approval_requests` |
| `tool_result` | `tool_results` (`tool_call_id`, `tool_name`, `args`, `result`, `tool_call_result_id`, `artifacts`), `tool_approval_responses` |

Consecutive tool result rows reach the client as one, so each call's result is
sent to the provider once. Return a call's provider ids on `LlmToolCall`
(`result_id`, `reasoning_id`, `reasoning_summary`), and provider state for the
turn as `LlmResponse.additional_content`: they are recorded and come back in
`messages`.

## Structured turns

When the answer is a document rather than prose, hand the turn a schema:

```python
schema = {
    "name": "plan",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "steps"],
}

response = runtime.send_structured(session, "Plan the release", schema)

response.structured  # {"title": "Ship it", "steps": ["write", "test"]}
response.text  # the document as the model wrote it
```

It is the same run as `send()` — the mode's system prompt, its tools, the step
budget, the events — asking the provider for structured output. Your client
receives the schema as `request.schema` and returns what it parsed as
`LlmResponse.structured`; with `prism-ai-core` that is
`Prism.structured().with_schema()` and the response's `structured`.

**The thread keeps the text, with the parsed object beside it.** The assistant
message is the raw document, and `structured` rides along in the row's
`additional_content`. A later turn replays the conversation as messages and reads
the text, so a thread that contains a structured answer reads like any other.

**A document that misses the schema is refused, not repaired.**

```python
from prism_harness import HarnessError

try:
    plan = runtime.send_structured(session, brief, schema).structured
except HarnessError as error:
    error.code  # structured_schema_violation, or structured_unreadable
    error.problems  # every way it missed, not the first
    error.document  # what the model actually said
```

Nothing is coerced, nothing is trimmed to the fields that fit, and the result is
never an empty document. An empty plan settles a batch as `done`, which reads
exactly like a considered answer of "nothing to propose" — the failure this
refusal exists to prevent. The exchange is still recorded, and the run is marked
failed: a thread that omits the answer it did not like cannot explain the retry
sitting next to it. The run row and the `RunFailed` event name the CODE, not the
message — the message quotes the values that missed, and an event carrying those
would ship model output to every listener, which is the same reason tool
arguments are names-only here.

The check reads the schema's own JSON Schema, so a hand-written schema is held to
the same terms. `schema_problems()` is exported if you want it directly. It
checks declared types, required keys, enum members, array items, and — where a
schema closes itself — keys nobody declared. It does not read `$ref`, `allOf`,
`oneOf` or the numeric and string facets; what it cannot read, it passes, rather
than reporting a constraint it did not actually check. The same rules as the PHP
reference, message for message, so a document refused in one language is refused
in the other for the same stated reason. `True` is not a number here, and a whole
float reads as the integer JSON writes: both are ways Python alone would have
answered differently.

## Failed model steps

A client response with `finish_reason="error"` or `"unknown"` ends the run with
`HarnessError.code == "model_step_failed"`. Its `finish_reason` identifies which
of those two finishes occurred; the error message contains no model output.
The step, cost, and assistant response are recorded before the error is raised.
Tools requested by that response are refused, with no pending approval authority,
so approving or resuming later cannot execute them. A new turn can still be sent.

## Approvals

A mode names the tools a person must approve:

```python
ModeRegistry(
    {
        "modes": {
            "guarded": {
                "system_prompt": "...",
                "tools": ["read", "delete"],
                "requires_approval": ["delete"],
            }
        }
    }
)
```

A step that calls one stops with `finish_reason == "awaiting_approval"`. The
calls that need nobody have already run. Record a decision for every pending
approval, then resume with an empty prompt:

```python
from prism_harness import record_approval

response = runtime.send(session, "Clean up the failed run")

if response.finish_reason == "awaiting_approval":
    for pending in response.pending_approvals:
        record_approval(session, pending.id, True)  # or False, "not on production"

    runtime.send(session, "")
```

On resume the approved calls run once and denied calls return their reason, and
then the model continues. The model is not asked to make the calls again. A
pending call with no decision is refused with "No approval response provided".
A call that has a result never runs again. Who may approve is your application's
decision: authorize before calling `record_approval()`.

A decision must answer an approval the session is waiting on. `record_approval()`
raises a `HarnessError` coded `approval_not_pending` for an approval id the run
never issued, or one already answered, and writes nothing, so a made-up id cannot
cancel a real one.

`runtime.decide(session, approval_id, approved, reason=None)` records the same
decision and also lets an agent plan read it before the run resumes.

## Agent plans

An agent can propose a plan: a goal and an ordered list of steps. A person
approves the plan, and your code works through it one step per turn. Plans are
off unless you turn them on for the runtime and for the mode:

```python
from prism_harness import AgentRuntime, ModeRegistry, Plans

modes = ModeRegistry(
    {"modes": {"chat": {"tools": ["search"], "requires_approval": ["search"], "plans": True}}}
)
runtime = AgentRuntime(
    client=client,
    modes=modes,
    tools=tools,
    plans=Plans({"enabled": True}, index_store=harness.durable_store()),
)
```

The mode's `plans` flag must be `True` or `False`; anything else is refused when
the mode is resolved. With plans on, the model is offered four tools:
`plan_draft`, `plan_complete_step`, `plan_pause_for_person` and
`plan_propose_change`. None of them takes an approval id or names another step.

**A person approves every plan before any step runs.** A draft stops the run
like any approval. Answer it with `runtime.decide()` and resume with
`runtime.send(session, "")`. Only your config can turn this off
(`Plans({"enabled": True, "require_approval": False})`); the model cannot, and
neither can a per-call option. A plan change and a `kind: approve` pause are
always approved by a person. No `ToolPolicy` can name a plan tool, however it is
spelled: such a policy is refused when it is built.

Drive the plan from your code:

```python
handle = runtime.plan(session)  # raises PlansDisabled when plans are off
outcome = handle.advance()  # at most one step: "ran", "waiting", "blocked", "done", "not_active"
handle.current()  # PlanSnapshot or None; .to_dict() for your API
handle.answer_step(step_id, "Paris", None, actor="user:7")
handle.resume(actor="user:7")  # unblock; a failed step gets a retry step
handle.cancel(actor="user:7")
```

Each `advance()` settles the previous step, then runs one turn with the user
message "Continue the plan. Work on step N.". The step's instruction reaches the
model only in the plan context, which is added to the system prompt of every
turn in the session and never stored in the thread. A step is done only when the
agent called `plan_complete_step` and the turn ended cleanly. A step whose turn
ended without that call is tried again, and after `max_attempts` the plan is
blocked. Only one `advance()` runs at a time per session; a second one raises a
`HarnessError` coded `session_locked`. Who may answer, resume or cancel is your
application's decision: authorize first, and pass the person as `actor`.

Bounds (Fancy contract g282.3): at most 40 steps (20 by default), 160-character
step titles, 200-character notes, and a plan whose first update would be over
8 KB. A plan over a bound is refused when drafted, never cut short, and nothing
is saved. Every refusal the model sees is a fixed sentence and a code, such as
`[plan.bounds]`.

`Plans` takes the same settings as the PHP package's `prism-harness.plans`
config: `require_approval`, `max_steps`, `max_attempts`, `max_pauses_per_step`,
`max_changes`, `max_advances`, `context: {max_chars, window}`,
`events: {max_bytes}` and `reaper: {stale_after}`. A value out of range is
refused when `Plans` is built.

Every write emits a `PlanUpdated` event (`plan.updated`) carrying the plan id,
revision, status, the approval holding it and step counts. It never carries a
title, instruction, note, question or answer; fetch the plan with
`current()` when the revision changes.

**This Python port does not stream `plan_updated` steps.** It has no run step
stream, so watch `PlanUpdated` events instead.

If a queue or worker advances your plans, register a callback and call
`reap()` on a schedule. It hands every stalled plan to your callback, and
`diagnose(plans=runtime.plans)` fails until it has run recently. Pass
`plans_queued=False` when every advance runs inline.

```python
runtime.plans.reap_using(lambda session_key, plan_id: queue.push(session_key))
runtime.plans.reap()  # every few minutes
```

### Plan budgets

How a plan's steps are budgeted is the app's choice, set in the `Plans`
config's `budget` or per plan with `runtime.plan(session).use_budget({...})`.
The model and per-call options cannot choose it or widen it.

| mode | what it means |
|---|---|
| `per_step` (default) | Each step's turn gets the per-turn budget, as before. |
| `shared` | The plan's own turns also spend from one ledger: `max_steps` model steps in total (required) and optionally `max_cost_usd`. Per-turn limits still apply to each turn. When the ledger is spent, the plan is blocked with `plan_budget_exhausted` and a `PlanBudgetExhausted` event (`plan.budget_exhausted`) is emitted once. |
| `none` | No plan-level budget. Per-turn limits, the plan's own limits and agent spend ceilings still apply. |

```python
plans = Plans({"enabled": True, "budget": {"mode": "shared", "max_steps": 40, "max_cost_usd": 2.0}})

# or for one plan, in host code:
runtime.plan(session).use_budget({"mode": "shared", "max_steps": 12})
runtime.plan(session).budget()  # mode, limits, and what a shared budget has spent
```

A shared ledger lives in the plan store, so it carries across approval pauses,
restarts and workers. Each turn holds what it may spend before it starts and
settles what it spent when it ends, in one locked write each, so two turns are
never granted the same remainder. A hold whose turn never settled is charged in
full once the session lock TTL has passed. A plan's budget is fixed when it is
first chosen or first used: after that its mode cannot change
(`plan.budget_fixed`) and its limits can only be lowered. A step whose turn is
handed a run tree of its own is refused (`plan.budget_context`) rather than run
outside the ledger. An invalid mode or limit raises `PlanBudgetInvalid` when
`Plans` is built. A turn the ledger refuses raises `HarnessError` with code
`plan_budget_exhausted`.

The plan's own turns are the turn `advance()` runs for the step it claimed, the
resume of an approval that step asked for, and a subagent child those turns
started, on any turn the runner did not start: an answer to its approval, a
new message, or a host `RunContext` (narrowed to what the plan has left, never
refused for having one). Any other turn on the session,
such as the person chatting while a step waits on their approval, spends
nothing from the plan's budget and is never refused for it. A decision
(`record_approval()` or `runtime.decide()`) on an approval the step asked for
is accepted and acted on even when the shared budget is spent: acting on a
decision makes no provider call, so the resume runs the approved call and
takes no model step, and the next `advance()` blocks the plan. A plan turn with
a new prompt and nothing left (another turn holds the remainder) is refused
with `plan_budget_exhausted` before anything is recorded. A subagent child paused inside a
plan's turn carries the plan's ledger on its paused row
(`harness_plan_ledger`), so its resume spends from that budget, under the same
rule.

A subagent child started by a tool inside a turn spends from that turn's run
context: its ledger, so a shared plan budget counts the child's steps, and its
limits. The `RunContext` handed to `SubagentRunner.tool()` can only narrow
them: what it has left is counted from the turn's spend so far, and no limit is
raised past the turn's. Called outside a turn, the runner uses that context
unchanged.

## Task lists

An agent given a goal keeps working across many requests. `session.tasks()` is
the list of what remains, in the durable half:

```python
from prism_harness import TaskOutcome

tasks = session.tasks()
tasks.add_many(["Read the brief", "Draft the reply", "Check the numbers"])

while (task := tasks.claim("worker-1")) is not None:
    # The APPLICATION releases, from evidence -- not the agent.
    worked = do_the_work(task.instruction)
    tasks.release(task, "worker-1", TaskOutcome.DONE if worked else TaskOutcome.FAILED)
```

Four states — `todo`, `claimed`, `done`, `failed` — and no others, behind four
methods: `claim`, `release`, `pending` and `find`. `find` is on the contract
because `release` takes a task while every caller outside the loop — a tool, an
HTTP route, a worker resuming after a restart — holds only an id.

A lease must be a finite number of seconds greater than zero. A zero or negative
one is refused rather than clamped: it would expire the instant it was granted,
so the claim it should protect would be stealable by the next caller.

`claim()` is **one call**, not a read followed by a mark: the read that picks the
task and the write that takes it happen inside one store lock, so two workers get
different tasks or one gets `None`. Never the same task twice.

A claim carries an owner **and an expiry**, five minutes by default. When a lease
lapses the task returns to `todo` — **never to `failed`**, because a worker dying
is not the task failing, and marking it failed burns a retry that never ran. A
worker may push its own lease out while it still holds it, bounded by what the
run's `RunBudget` has left on the wall clock; there is deliberately no second
timeout here to set in the place that is not enforced.

`done` and `failed` are terminal, and re-releasing one raises rather than quietly
doing nothing.

**An agent cannot mark its own task complete.** If the model can set its own task
to `done`, "run until the goal is met" becomes "run until it decides it is met",
and a stalled run ends by declaring victory. `TaskCompletionTool` exists for
consumers who want that, and nothing registers it: you register it on your own
`ToolRegistry` and gate it through the `ToolAuthorizer` you already have. It is
bound to one worker and closes only the task that worker is holding.

That rule is enforced on the source, not just in the tool: `release()` takes the
worker, and only the worker currently holding the lease may release. Without it,
a worker whose lease lapsed mid-task overwrites the claim of whoever legitimately
picked the task up — the task reads `done` while the second worker is still
working, its work is discarded, and its own release then fails as "already
terminal", blaming it for the first worker's mistake. The tool checks too, because
a third party's `AgentTaskSource` cannot be made to.

The agent must state an outcome, and it is either exactly `done` or exactly
`failed` or refused with `task_outcome_invalid`. No case folding, no trimming,
no truthiness — and an ABSENT outcome is refused too, not read as `done`. An
implementation that resolves what it cannot read always resolves it toward the
privileged answer, whether it is coercing `complete` into `done` or inferring
completion from a field the agent never sent.

**The list is durable state, so a task source refuses to start on a volatile
store** — a half-finished list that vanishes on a deploy is indistinguishable
from a finished one.

No task model, no schema, no migration. `AgentTask` is a Protocol, so a record
you already have — an ORM model, a dataclass, a row wrapper — becomes one the
moment it exposes `id`, `instruction` and `state`. If your columns are named
something else, `AgentTaskMixin` maps them; that is what the PHP reference
spells as a trait on an Eloquent model.

## Synchronous, deliberately

The store contract is **sync**, unlike `prism-harness-ts`. That port is async
because Node's filesystem API is, not because the work is slow. Python's is not,
the PHP reference is synchronous too, and a caller who needs this off the event
loop can wrap it in `asyncio.to_thread` — which is what `prism-workspace-py`
does for the same reason. Forcing async here would make every consumer of a
plain WSGI application write `asyncio.run` around a dictionary lookup.

## Drivers

- **`MemorySessionStore`** — volatile, and says so. Its lock is a real
  `threading.Lock` but process-local.
- **`FileSessionStore`** — durable. Payload writes remain atomic. Cross-process
  local locks follow [decision 0022](https://github.com/Particle-Academy/prism-parity/blob/main/docs/decisions/0022-local-file-lock-ownership.md):
  a complete nonce-bound owner directory is published atomically; TTL expiry
  never proves that a live holder died. Recovery requires a matching established
  native boot/PID realm and a proven dead or explicitly released owner. Local
  filesystems on one host only; SMB/NFS are unsupported.

  State readers close their handles before decoding the document. On Windows,
  replacement of a state file can fail while another process holds it open,
  even with delete sharing. Permission failures retry for up to 750 ms. If the
  reader stays open or access is permanently denied, the original native error
  is raised and the temporary file is removed; the previous state stays intact.
  Other errors and POSIX replacement do not retry. Callbacks are never replayed.

  On Windows, the first file-lock operation makes one fixed, read-only metadata
  query using the system Windows PowerShell and CIM module, with a fifteen-second
  timeout. Established identities and detected container conditions are cached
  for the process lifetime. Transient query failures are cached for sixty seconds,
  then permit a serialized new sample. With unknown identity a worker can acquire
  a free lock and release its own owner record, but unknown owners and unknown
  probers never authorize PID probes or recovery. A crashed unknown owner's lock
  stays held for explicit operator inspection. `FileSessionStore.identity_status()`
  is an in-memory snapshot and performs no query or filesystem work. No runtime
  compilation or extra dependency is used.

  **BREAKING CHANGE: Stop all old workers before upgrading the file-lock
  protocol.** Inspect and archive stale legacy timestamp lockfiles explicitly.
  The library refuses them with `session_locked` and
  `legacy_lock_requires_maintenance`, naming the canonical path and action
  `stop_old_workers_then_inspect_and_archive`. On Windows, Node can replace a
  legacy file created between preflight and rename; leaving old workers running
  can break exclusion. No rolling upgrade is supported. The hard stop is a
  correctness requirement in both ports.

  A blocked Windows reader can delay release. The callback may already have
  completed when release raises an error; the library does not replay it.
  Pending cleanup is retried at the owning process's next file-lock operation.
  The `file_lock_release_pending` warning includes only the key digest and nonce.
  Read `store.pending_releases()` for this directory's pending digest/nonce entries;
  the snapshot's length is the count. Querying does not retry or write anything.


Implement `SessionStore` for anything else, and declare your own `durability()`:
only you know whether your Redis is persistent or a disposable cache, and that
declaration is an assertion about your infrastructure, not a preference.
