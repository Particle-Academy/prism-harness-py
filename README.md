# Prism Harness for Python

Durable agent sessions — threads, session state and store drivers. The Python
port of [`particle-academy/prism-harness`](https://github.com/Particle-Academy/prism-harness).

Zero runtime dependencies. Python 3.10+.

This package is private while coordinated parity work is in progress.

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
like `Media.as_` in `prism-ai`.

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
