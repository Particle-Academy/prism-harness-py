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
session.using_mode("plan").using_model("claude-sonnet-4-5")


def advance(live):
    live.begin_run("run-1", "plan", "anthropic", "claude-sonnet-4-5")


session.lock(advance)  # whatever must not happen twice
```

`for` is a Python keyword, so the method is `for_` — forced rather than chosen,
like `Media.as_` in `prism-ai-core`.

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
ModeRegistry({"modes": {"overseer": {"provider_options": {"thinking": {"type": "adaptive"}}}}})
```

A value that is not a map is refused when the mode is resolved, with
`mode_malformed`.

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
response.text        # the document as the model wrote it
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
try:
    plan = runtime.send_structured(session, brief, schema).structured
except HarnessError as error:
    error.code      # structured_schema_violation, or structured_unreadable
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

## Approvals

A mode names the tools a person must approve:

```python
"guarded": {"system_prompt": "...", "tools": ["read", "delete"], "requires_approval": ["delete"]}
```

A step that calls one stops with `finish_reason == "awaiting_approval"`. The
calls that need nobody have already run. Record a decision for every pending
approval, then resume with an empty prompt:

```python
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

## Task lists

An agent given a goal keeps working across many requests. `session.tasks()` is
the list of what remains, in the durable half:

```python
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
- **`FileSessionStore`** — durable. Atomic writes (`os.replace`) and a
  cross-process lock built on `O_CREAT | O_EXCL`, the one primitive atomic on
  every filesystem worth supporting. Two workers on one machine genuinely
  exclude each other; two machines over a network filesystem do not, and no
  file lock can promise that — use a database there.

Implement `SessionStore` for anything else, and declare your own `durability()`:
only you know whether your Redis is persistent or a disposable cache, and that
declaration is an assertion about your infrastructure, not a preference.
