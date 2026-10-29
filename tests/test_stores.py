"""Both drivers, the same way. Mirrors prism-harness-ts/test/stores.test.ts."""

from __future__ import annotations

import errno
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from prism_harness import (
    FileSessionStore,
    HarnessError,
    MemorySessionStore,
    SessionStore,
    SessionStoreManager,
)
from prism_harness.stores._file_lock_identity import native_identity


def a_file_store() -> FileSessionStore:
    return FileSessionStore(tempfile.mkdtemp(prefix="prism-harness-store-"))


#: A test written against only the in-memory driver proves nothing about the
#: driver a deployment actually uses, and the two differ in exactly the places
#: that matter -- copying, expiry, and whether a lock crosses a process.
DRIVERS: list[tuple[str, Callable[[], SessionStore]]] = [
    ("memory", MemorySessionStore),
    ("file", a_file_store),
]


@pytest.fixture(params=DRIVERS, ids=[name for name, _ in DRIVERS])
def store(request: pytest.FixtureRequest) -> Iterator[SessionStore]:
    _name, make = request.param
    yield make()


def test_returns_none_for_a_key_it_has_never_seen(store: SessionStore) -> None:
    assert store.get("nothing") is None


def test_round_trips_a_payload(store: SessionStore) -> None:
    store.put("k", {"mode": "plan", "depth": 2})

    assert store.get("k") == {"mode": "plan", "depth": 2}


def test_forgets_a_key(store: SessionStore) -> None:
    store.put("k", {"a": 1})
    store.forget("k")

    assert store.get("k") is None


def test_expires_a_payload_once_its_ttl_has_passed(store: SessionStore) -> None:
    store.put("k", {"a": 1}, -1)

    assert store.get("k") is None


def test_hands_back_a_copy_so_a_caller_cannot_mutate_what_is_stored(store: SessionStore) -> None:
    # Only the in-memory driver could ever get this wrong, which is exactly why
    # it is checked on both: a bug that appears on one driver is the worst kind,
    # because the in-memory one is what tests run against.
    store.put("k", {"list": [1, 2]})

    first = store.get("k")
    assert first is not None
    first["list"].append(3)

    assert store.get("k") == {"list": [1, 2]}


def test_runs_a_locked_callback_and_returns_its_value(store: SessionStore) -> None:
    assert store.with_lock("k", lambda: "done") == "done"


def test_releases_the_lock_when_the_callback_raises(store: SessionStore) -> None:
    def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        store.with_lock("k", boom)

    # Would block or time out if the lock leaked.
    assert store.with_lock("k", lambda: "free") == "free"


def test_serialises_two_callers_on_the_same_key(store: SessionStore) -> None:
    order: list[str] = []

    def slow() -> None:
        order.append("first-in")
        time.sleep(0.05)
        order.append("first-out")

    def quick() -> None:
        order.append("second-in")

    first = threading.Thread(target=lambda: store.with_lock("k", slow))
    first.start()
    time.sleep(0.01)  # let the first caller take the lock
    store.with_lock("k", quick)
    first.join()

    # The second caller must not begin before the first has finished.
    assert order == ["first-in", "first-out", "second-in"]


def test_does_not_serialise_different_keys_against_each_other(store: SessionStore) -> None:
    # Barrier-driven, with no sleeps and no deadline.
    #
    # This used to hold "a" for 100ms and require the main thread to take "b"
    # and observe the holder still running -- which asserts the property by
    # WINNING A RACE against a timer. A loaded runner that spent 90ms inside the
    # file store's lock acquisition failed a store that was behaving perfectly.
    # It failed on Windows CI for Python 3.10 and 3.12 while passing on 3.11,
    # 3.13 and every Ubuntu job: the signature of a clock, not of a defect.
    #
    # Now "a" is held open until the main thread says otherwise. If "b" ever
    # serialised behind "a" the two would wait on each other, and the test fails
    # by TIMING OUT rather than by being unlucky -- and it cannot pass by being
    # lucky, which is the half that matters.
    a_is_held = threading.Event()
    a_may_release = threading.Event()
    b_ran_while_a_was_held = threading.Event()

    def hold() -> None:
        a_is_held.set()
        # Bounded so a broken store fails the suite instead of hanging it.
        a_may_release.wait(timeout=10)

    held = threading.Thread(target=lambda: store.with_lock("a", hold))
    held.start()

    try:
        assert a_is_held.wait(timeout=10), "the holder never entered its critical section"

        # The whole property: this must not wait on "a", which is still open.
        store.with_lock("b", b_ran_while_a_was_held.set)

        assert b_ran_while_a_was_held.is_set()
        assert not a_may_release.is_set(), '"a" was released before "b" was taken'
    finally:
        a_may_release.set()
        held.join(timeout=10)

    assert not held.is_alive()


def test_raises_session_locked_rather_than_running_the_callback_anyway(
    store: SessionStore,
) -> None:
    ran_while_held = threading.Event()
    acquired, release = threading.Event(), threading.Event()
    failures: list[BaseException] = []

    def hold() -> None:
        acquired.set()
        assert release.wait(timeout=10), "the holder was never released"

    def holder() -> None:
        try:
            store.with_lock("k", hold)
        except BaseException as error:
            failures.append(error)
            acquired.set()
            raise

    held = threading.Thread(target=holder)
    held.start()
    try:
        assert acquired.wait(timeout=20), "the holder never acquired its lock"
        assert not failures
        with pytest.raises(HarnessError) as caught:
            store.with_lock("k", ran_while_held.set, wait_seconds=0.05)
        assert caught.value.code == "session_locked"
        assert not ran_while_held.is_set()
        assert not release.is_set()
    finally:
        release.set()
        held.join(timeout=10)
    assert not held.is_alive()
    assert not failures


# -- driver-specific ---------------------------------------------------------


def test_memory_reports_itself_volatile() -> None:
    # Which is what gets it refused for durable state.
    assert MemorySessionStore().durability().value == "volatile"


def test_file_reports_itself_durable() -> None:
    assert a_file_store().durability().value == "durable"


def test_file_survives_a_new_store_instance_over_the_same_directory() -> None:
    # The point of the driver. An in-memory store returns None here, and that
    # difference is the whole reason durability is declared rather than inferred.
    directory = tempfile.mkdtemp(prefix="prism-harness-store-")
    FileSessionStore(directory).put("k", {"approval": "pending"})

    assert FileSessionStore(directory).get("k") == {"approval": "pending"}


def test_file_reclaims_a_proven_dead_same_identity_holder() -> None:
    store = a_file_store()
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    plant_owner(store, int(child.stdout), False)
    assert store.with_lock("k", lambda: "recovered") == "recovered"


# -- the stolen-lock window --------------------------------------------------
#
# `claim()` atomicity in `prism_harness.tasks` rests ENTIRELY on this lock. A
# lock that can be stolen from a live holder is two workers on one task, so the
# question these ask is not "does the lock work" but "what does a waiter do with
# a lockfile whose expiry it cannot fully read".
#
# The window is real: `os.open` creates the file EMPTY and the expiry is written
# after, so a waiter can observe a lockfile with no expiry in it at all. Every
# unreadable state must fail in the SAFE direction -- wait, do not steal.


def a_lock_path(store: FileSessionStore, key: str) -> Path:
    return store._path_for(key).with_suffix(".json.lock")


def plant_owner(store: FileSessionStore, pid: int, released: bool) -> None:
    identity = native_identity()
    assert identity is not None and identity["pid_namespace"] is not None
    nonce = secrets.token_hex(16)
    path = a_lock_path(store, "k")
    path.mkdir()
    (path / ("owner-" + nonce + ".json")).write_text(
        json.dumps(
            {
                "version": 2,
                "pid": pid,
                "identity": identity,
                "nonce": nonce,
                "released": released,
                "expires_at": int(time.time()) + 60,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def steals(store: FileSessionStore, key: str) -> bool:
    """Did a waiter take a lock that is planted on the key?

    True means the lockfile was reclaimed and the callback ran; False means the
    waiter respected it and timed out, which is the safe answer for every
    expiry it cannot read completely.
    """
    try:
        store.with_lock(key, lambda: "stolen", wait_seconds=0.05)
    except HarnessError as error:
        assert error.code == "session_locked"
        return False

    return True


def test_file_never_reads_an_empty_lockfile_as_expired() -> None:
    # THE TYPESCRIPT DEFECT, asked of Python. There it was fatal: the lock is
    # created empty and the expiry written after, and `Number('') === 0` read as
    # "expired in 1970", so a waiter deleted a lock another process was actively
    # holding. `float('')` RAISES in Python, which is why the same window is not
    # the same bug -- a language difference doing the work, which is exactly the
    # kind of thing that must be asserted rather than assumed.
    store = a_file_store()
    a_lock_path(store, "k").write_text("", encoding="utf-8")

    assert steals(store, "k") is False


def test_file_never_reads_a_truncated_expiry_as_expired() -> None:
    # The adjacent variant, and the one Python DOES get wrong: a prefix of a
    # real expiry is still a valid float, and every prefix of a ten-digit
    # timestamp is a smaller number -- which is to say, a time in the past.
    # `float('1735689')` parses happily and reads as expired in 1970.
    store = a_file_store()
    a_lock_path(store, "k").write_text("1735689", encoding="utf-8")

    assert steals(store, "k") is False


def test_file_never_reads_an_expiry_it_cannot_tell_is_complete_as_expired() -> None:
    # A value with no terminator could be the whole expiry or the first half of
    # one, and nothing in the file says which. Unreadable means WAIT.
    store = a_file_store()
    a_lock_path(store, "k").write_text(str(time.time() - 1000), encoding="utf-8")

    assert steals(store, "k") is False


def test_file_refuses_expired_legacy_even_with_a_complete_timestamp() -> None:
    store = a_file_store()
    path = a_lock_path(store, "k")
    path.write_text("1\n", encoding="utf-8")
    os.utime(path, (0, 0))
    assert steals(store, "k") is False
    assert path.read_text(encoding="utf-8") == "1\n"


def test_file_real_release_marks_owner_released_and_reports_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = a_file_store()
    real_unlink = Path.unlink

    def blocked(path: Path, missing_ok: bool = False) -> None:
        if path.name.startswith("owner-") and path.suffix == ".json":
            raise PermissionError(errno.EACCES, "synthetic cleanup denial")
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", blocked)
        with pytest.raises(PermissionError):
            store.with_lock("k", lambda: "finished")
        paths = list(a_lock_path(store, "k").iterdir())
        assert len(paths) == 1
        assert json.loads(paths[0].read_bytes())["released"] is True
    assert steals(store, "k") is True


def test_file_publishes_complete_owner_before_callback() -> None:
    store = a_file_store()
    seen: list[bytes] = []

    def capture() -> None:
        paths = list(a_lock_path(store, "k").iterdir())
        assert len(paths) == 1
        seen.append(paths[0].read_bytes())

    store.with_lock("k", capture)
    assert seen[0].endswith(b"\n")
    value = json.loads(seen[0])
    assert value["pid"] == os.getpid() and value["released"] is False
    assert type(value["expires_at"]) is int and value["expires_at"] >= int(time.time())


def test_real_writer_reader_roundtrip_refuses_live_and_reclaims_released() -> None:
    store = a_file_store()
    path = a_lock_path(store, "k")
    captured: list[tuple[str, bytes]] = []

    def capture() -> None:
        files = list(path.iterdir())
        captured.append((files[0].name, files[0].read_bytes()))
        # Expired timestamp does not authorize reentrant acquisition.
        assert steals(store, "k") is False

    store.with_lock("k", capture, ttl_seconds=-1000)
    path.mkdir()
    filename, data = captured[0]
    (path / filename).write_bytes(data)
    assert steals(store, "k") is False
    released = json.loads(data)
    released["released"] = True
    (path / filename).write_text(json.dumps(released) + "\n", encoding="utf-8")
    assert steals(store, "k") is True


def test_two_threads_never_hold_one_key_at_once() -> None:
    # The property `claim()` actually depends on, asserted directly rather than
    # inferred from a lock that looks right.
    store = a_file_store()
    inside = 0
    overlaps: list[int] = []
    guard = threading.Lock()

    def critical() -> None:
        nonlocal inside
        with guard:
            inside += 1
            if inside > 1:
                overlaps.append(inside)
        time.sleep(0.002)
        with guard:
            inside -= 1

    threads = [threading.Thread(target=lambda: store.with_lock("k", critical)) for _ in range(8)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert overlaps == []


def test_file_refuses_a_stored_payload_that_is_not_valid_json() -> None:
    directory = Path(tempfile.mkdtemp(prefix="prism-harness-store-"))
    store = FileSessionStore(directory)
    store.put("k", {"a": 1})
    store._path_for("k").write_text("not json", encoding="utf-8")

    with pytest.raises(HarnessError) as caught:
        store.get("k")

    assert caught.value.code == "unmappable_content"


# -- the manager -------------------------------------------------------------


def test_the_manager_refuses_a_volatile_store_for_the_durable_slot() -> None:
    # The guard the package exists for. Accepting it and finding out later is
    # exactly the failure it was written to avoid.
    manager = SessionStoreManager(
        drivers={"memory": MemorySessionStore},
        stores={"ephemeral": "memory", "durable": "memory"},
    )

    manager.ephemeral()

    with pytest.raises(HarnessError) as caught:
        manager.durable()

    assert caught.value.code == "unsafe_state_configuration"
    # The message has to name the fix, not just the fault.
    assert "durable driver" in caught.value.message


def test_the_manager_accepts_a_durable_store_for_the_durable_slot() -> None:
    manager = SessionStoreManager(
        drivers={"memory": MemorySessionStore, "files": a_file_store},
        stores={"ephemeral": "memory", "durable": "files"},
    )

    assert manager.durable().durability().is_durable()


def test_the_manager_names_the_slot_and_driver_when_a_driver_is_missing() -> None:
    manager = SessionStoreManager(drivers={}, stores={"durable": "redis"})

    with pytest.raises(HarnessError, match="redis"):
        manager.durable()


def test_the_manager_builds_each_driver_at_most_once() -> None:
    built = 0

    def make() -> SessionStore:
        nonlocal built
        built += 1
        return MemorySessionStore()

    manager = SessionStoreManager(
        drivers={"memory": make}, stores={"ephemeral": "memory", "durable": "memory"}
    )
    manager.ephemeral()
    manager.ephemeral()

    assert built == 1


def test_a_stored_payload_is_json_and_names_its_key() -> None:
    # The filename is a digest, so the key lives inside the file to keep the
    # mapping inspectable.
    directory = Path(tempfile.mkdtemp(prefix="prism-harness-store-"))
    store = FileSessionStore(directory)
    store.put("session:abc:7:support", {"a": 1})

    document: dict[str, Any] = __import__("json").loads(
        store._path_for("session:abc:7:support").read_text(encoding="utf-8")
    )

    assert document["key"] == "session:abc:7:support"
