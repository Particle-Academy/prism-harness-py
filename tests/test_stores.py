"""Both drivers, the same way. Mirrors prism-harness-ts/test/stores.test.ts."""

from __future__ import annotations

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

    def hold() -> None:
        time.sleep(0.3)

    held = threading.Thread(target=lambda: store.with_lock("k", hold))
    held.start()
    time.sleep(0.02)

    with pytest.raises(HarnessError) as caught:
        store.with_lock("k", ran_while_held.set, wait_seconds=0.05)

    assert caught.value.code == "session_locked"
    assert not ran_while_held.is_set()
    held.join()


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


def test_file_reclaims_a_lock_whose_holder_died() -> None:
    # Left alone, a stale lockfile wedges the key forever -- worse than the
    # small race in reclaiming it.
    directory = Path(tempfile.mkdtemp(prefix="prism-harness-store-"))
    store = FileSessionStore(directory)
    store.put("k", {})

    lock_path = store._path_for("k").with_suffix(".json.lock")
    # TERMINATED, because an expiry with nothing marking its end could equally
    # be the first half of a longer one, and a reader that trusted it would
    # delete a live holder's lock. This test wrote the unterminated form until
    # that was fixed, which is the tell that the lockfile FORMAT is shared
    # surface: a PHP or TypeScript worker in the same directory has to write
    # the terminator too, or this store will wait its stale locks out instead
    # of reclaiming them.
    lock_path.write_text(f"{time.time() - 1000}\n", encoding="utf-8")

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


def test_file_still_reclaims_a_lock_whose_expiry_is_complete_and_past() -> None:
    # The control, and it is load-bearing: without it every test above passes on
    # a store that simply never reclaims anything, which would wedge a key
    # forever the first time a worker died holding it.
    store = a_file_store()
    a_lock_path(store, "k").write_text(f"{time.time() - 1000}\n", encoding="utf-8")

    assert steals(store, "k") is True


def test_file_reclaims_a_lock_its_own_release_marked_dead() -> None:
    # `_release` rewrites a lock it could not unlink with an already-past
    # expiry. That path has to keep working in whatever format the reader
    # trusts, or the Windows leak it exists to prevent comes straight back.
    store = a_file_store()
    lock_path = a_lock_path(store, "k")
    FileSessionStore._release(lock_path)
    lock_path.write_text("", encoding="utf-8")
    FileSessionStore._release(lock_path)

    if lock_path.exists():
        assert steals(store, "k") is True


def test_file_writes_the_expiry_before_the_callback_runs() -> None:
    # The window between "the lockfile exists" and "the lockfile says when it
    # expires" is where a waiter can be fooled. It must be shut by the time the
    # holder is doing anything a waiter could contend with.
    store = a_file_store()
    seen: list[str] = []

    store.with_lock("k", lambda: seen.append(a_lock_path(store, "k").read_text(encoding="utf-8")))

    assert seen[0] != ""
    assert float(seen[0].strip()) > time.time()


def test_a_lock_this_store_wrote_is_read_back_by_this_store() -> None:
    # THE WRITER AND THE READER HAVE TO AGREE, and every test above plants its
    # lockfile by hand -- so all of them would stay green if the store started
    # writing a payload its own reader refuses. Nothing would break loudly:
    # locks would simply stop being reclaimable, and a dead holder would wedge
    # the key until every later caller timed out. A mutation run found this gap
    # by dropping the terminator from the payload and going green.
    store = a_file_store()
    written: list[str] = []

    def capture() -> None:
        written.append(a_lock_path(store, "k").read_text(encoding="utf-8"))

    # A ttl already in the past: what the store writes here is a lock that is
    # expired the moment it exists.
    store.with_lock("k", capture, ttl_seconds=-1000)
    a_lock_path(store, "k").write_text(written[0], encoding="utf-8")

    assert steals(store, "k") is True

    # The control, through the same round trip: a live ttl must NOT be
    # reclaimable, or the assertion above would pass on a reader that reclaims
    # everything it is shown.
    store.with_lock("k", capture, ttl_seconds=1000)
    a_lock_path(store, "k").write_text(written[1], encoding="utf-8")

    assert steals(store, "k") is False


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
