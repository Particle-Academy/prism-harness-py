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
    released = threading.Event()

    def hold() -> None:
        time.sleep(0.1)
        released.set()

    held = threading.Thread(target=lambda: store.with_lock("a", hold))
    held.start()
    time.sleep(0.01)

    store.with_lock("b", lambda: None)
    assert not released.is_set()

    held.join()


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
    lock_path.write_text(str(time.time() - 1000), encoding="utf-8")

    assert store.with_lock("k", lambda: "recovered") == "recovered"


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
