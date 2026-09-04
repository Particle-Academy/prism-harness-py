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

## Four things that are load-bearing

1. **`Session.key()` must stay byte-identical to PHP's.** sha1 of the
   participant type, truncated to 12. It is what lets all three languages share
   one store. Changing it silently splits a conversation in two.
2. **A volatile store is refused for the durable slot**, at resolve time.
3. **Thread positions are assigned inside the lock.** Read-then-write outside
   one loses a message when two turns land together, and nothing reports it.
4. **`StoreTaskSource.claim()` is ONE call.** The read that picks the task and
   the write that takes it are inside one store lock. Splitting them hands the
   same task to two workers, and the test that proves the atomic version works
   is paired with a deliberately racy one that must keep failing.

## Traps already hit here

- **Releasing the file lock swallowed a failed unlink**, which leaks the lock:
  on Windows a waiter attempting the same path makes the unlink raise, the
  lockfile survives with a live ttl, and every later caller blocks out its whole
  wait. Two threads recording concurrently found it on the first run. Release
  retries, then marks the file dead.
- **The sync/async split from `prism-harness-ts` is deliberate**, not drift.
  See the README.
