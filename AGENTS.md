# AGENTS.md — prism-harness-py

The Python port of `particle-academy/prism-harness`. Read the shared agent guide
in `prism-parity/docs/AGENTS.md` first.

## Gates — run them on EXIT CODES

```sh
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src tests
python -m pytest
```

Never pipe a gate into `head`/`tail`/`grep` and read `$?` — that is the
FILTER's exit code. Redirect to a file, echo `$?`, then look.

## Five things that are load-bearing

1. **`Session.key()` must stay byte-identical to PHP's.** The null namespace
   retains sha1 of the participant type, truncated to 12. Named SessionAddress
   uses the domain-separated compact UTF-8 JSON digest including agent, owner
   and scope. Shared vectors bind both forms. Never implicitly migrate a legacy
   address or derive an agent id from a persona name.
2. **A volatile store is refused for the durable slot**, at resolve time.
3. **Thread positions are assigned inside the lock.** Read-then-write outside
   one loses a message when two turns land together, and nothing reports it.
4. **Decision 0022 binds the local file lock.** A complete owner directory is
   published before callback entry. TTL is never death proof. Recovery matches
   established boot/PID realm before any read-only PID probe, then unlinks the
   observed nonce and uses nonrecursive rmdir only after that unlink succeeds.
   Preserve the legacy maintenance diagnostic and mandatory old-worker hard stop.
   See prism-parity/docs/decisions/0022-local-file-lock-ownership.md.
5. **`StoreTaskSource.claim()` is ONE call.** The read that picks the task and
   the write that takes it are inside one store lock. Splitting them hands the
   same task to two workers, and the test that proves the atomic version works
   is paired with a deliberately racy one that must keep failing.

## Traps already hit here

- **Releasing the file lock swallowed a failed unlink**, which leaks the lock:
  on Windows a waiter attempting the same path makes the unlink raise, the
  lockfile survives with a live ttl, and every later caller blocks out its whole
  wait. Two threads recording concurrently found it on the first run. Release
  historically retried then marked the file dead. Decision 0022 replaces that
  timestamp fallback with a closed released:true record and observable cleanup
  failures; keep this historical failure as the reason release is tested.
- **The sync/async split from `prism-harness-ts` is deliberate**, not drift.
  See the README.
- **A lease must be a positive WHOLE number of seconds**, and that was not true
  here until the cross-language corpus asked. `_require_lease` checked
  finite-and-positive, so `90.4` was accepted and then truncated to a 90 second
  lease by `_timestamp` -- a configuration silently becoming a different
  configuration, in the unsafe direction, since a lease that lapses early hands
  the task out while its holder still believes it holds it. Nothing errored and
  no per-language test could see it: the reference cannot even ask (`claim()`
  declares `?int`) and each language asserts against the value its own code
  produced. `suites/agent-task-claim` atc-0017 found it. G-40, closed.
  **Check every route into a lease, not only the typed one** -- the reference's
  own guard was defeated from inside its shipped config file, by an `(int)`
  cast. `Session.tasks()` is the equivalent route here and has its own test.
