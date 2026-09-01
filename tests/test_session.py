"""Sessions and threads. Mirrors prism-harness-ts/test/session.test.ts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading

import pytest

from prism_harness import (
    FileSessionStore,
    MemorySessionStore,
    Participant,
    PrismHarness,
    Session,
)


def a_harness() -> PrismHarness:
    directory = tempfile.mkdtemp(prefix="prism-harness-session-")

    return PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )


def a_session(scope: str = "support") -> Session:
    return a_harness().for_(Participant("App\\Models\\User", 7)).session(scope)


# -- addressing --------------------------------------------------------------


def test_produces_the_same_key_the_php_reference_produces() -> None:
    # sha1 of the participant type, truncated to 12 -- byte for byte what
    # `Prism\Harness\Sessions\Session::key()` builds. Matching exactly is what
    # lets a PHP app and a Python agent share one store and resolve the same
    # session; a different digest would give two conversations that look
    # identical and never meet.
    digest = hashlib.sha1(b"App\\Models\\User").hexdigest()[:12]

    assert a_session().key() == f"session:{digest}:7:support"


def test_keeps_two_scopes_for_one_participant_apart() -> None:
    harness = a_harness()
    participant = Participant("User", 1)

    support = harness.for_(participant).session("support")
    billing = harness.for_(participant).session("billing")

    support.using_mode("plan")

    assert support.mode() == "plan"
    assert billing.mode() is None


def test_keeps_the_same_id_under_two_participant_types_apart() -> None:
    # `7` means a different participant in each table, which is why the type is
    # part of the address rather than just the id.
    harness = a_harness()
    user = harness.for_(Participant("User", 7)).session("s")
    team = harness.for_(Participant("Team", 7)).session("s")

    user.using_model("claude-sonnet-4-5")

    assert team.model() is None
    assert user.key() != team.key()


# -- the ephemeral half ------------------------------------------------------


def test_round_trips_mode_model_and_provider() -> None:
    live = a_session()
    live.using_mode("plan").using_model("claude-sonnet-4-5").using_provider("anthropic")

    assert live.mode() == "plan"
    assert live.model() == "claude-sonnet-4-5"
    assert live.provider() == "anthropic"


def test_is_resolved_not_held() -> None:
    # The whole design. Nothing survives in memory between turns, so a fresh
    # worker resolving the same address must see the same state.
    harness = a_harness()
    participant = Participant("User", 7)

    harness.for_(participant).session("support").using_mode("plan")

    assert harness.for_(participant).session("support").mode() == "plan"


def test_forget_drops_the_ephemeral_half_and_leaves_the_conversation() -> None:
    live = a_session()
    live.using_mode("plan")
    live.thread().record([{"type": "user", "content": "hello"}])

    live.forget()

    assert live.mode() is None
    assert live.thread().count() == 1


# -- the durable half --------------------------------------------------------


def test_stores_and_forgets_a_capability() -> None:
    live = a_session()
    live.using_capability("search", {"index": "docs", "k": 5})

    assert live.capability("search") == {"index": "docs", "k": 5}

    live.forget_capability("search")
    assert live.capability("search") is None


def test_a_capability_survives_the_ephemeral_half_being_dropped() -> None:
    live = a_session()
    live.using_capability("search", {"index": "docs"})
    live.forget()

    assert live.capability("search") == {"index": "docs"}


# -- runs --------------------------------------------------------------------


def test_records_a_run_through_its_lifecycle() -> None:
    live = a_session()
    live.begin_run("run-1", "plan", "anthropic", "claude-sonnet-4-5")

    run = live.run()
    assert run is not None
    assert run["id"] == "run-1"
    assert run["status"] == "running"

    live.complete_run("run-1", "stop", ["search", "write"])

    run = live.run()
    assert run is not None
    assert run["status"] == "completed"
    assert run["finish_reason"] == "stop"
    assert run["tool_calls"] == ["search", "write"]


def test_records_a_failure() -> None:
    live = a_session()
    live.begin_run("run-1", "plan", "anthropic", "m")
    live.fail_run("run-1", "provider timed out")

    run = live.run()
    assert run is not None
    assert run["status"] == "failed"
    assert run["failure"] == "provider timed out"


def test_records_tool_names_only_never_arguments() -> None:
    # A tool name is not PII and is what an operator needs to audit a guardrail.
    # Arguments are, and prism-opentelemetry already carries them behind an
    # opt-in capture gate -- recording them again here, ungated, would quietly
    # undo that decision for anyone who installed both.
    live = a_session()
    live.begin_run("run-1", "plan", "anthropic", "m")
    live.complete_run("run-1", "stop", ["search"])

    run = live.run()
    assert run is not None
    assert "argument" not in json.dumps(run)
    assert run["tool_calls"] == ["search"]


def test_a_superseded_run_does_not_overwrite_the_one_in_flight() -> None:
    # A late worker reporting on a run that has already been replaced would
    # otherwise mark the live one finished.
    live = a_session()
    live.begin_run("run-1", "plan", "anthropic", "m")
    live.begin_run("run-2", "plan", "anthropic", "m")

    live.complete_run("run-1", "stop")

    run = live.run()
    assert run is not None
    assert run["id"] == "run-2"
    assert run["status"] == "running"


# -- lock --------------------------------------------------------------------


def test_lock_runs_the_callback_and_returns_its_value() -> None:
    assert a_session().lock(lambda _live: "done") == "done"


def test_lock_re_reads_state_inside_the_lock() -> None:
    # State written by whoever held the lock before us is otherwise invisible to
    # this instance, and acting on a stale read is what the lock exists to
    # prevent.
    harness = a_harness()
    participant = Participant("User", 7)
    one = harness.for_(participant).session("support")
    two = harness.for_(participant).session("support")

    assert one.mode() is None  # primes one's cache
    two.using_mode("plan")

    assert one.lock(lambda live: live.mode()) == "plan"


# -- threads -----------------------------------------------------------------


def test_assigns_positions_from_one_in_order() -> None:
    thread = a_session().thread()
    recorded = thread.record(
        [{"type": "user", "content": "one"}, {"type": "assistant", "content": "two"}]
    )

    assert [entry.position for entry in recorded] == [1, 2]
    assert [entry.position for entry in thread.messages()] == [1, 2]


def test_continues_numbering_across_separate_calls() -> None:
    thread = a_session().thread()
    thread.record([{"type": "user", "content": "one"}])
    second = thread.record([{"type": "user", "content": "two"}])

    assert second[0].position == 2


def test_does_not_lose_a_message_when_two_turns_land_concurrently() -> None:
    # Read-and-write inside one lock. Both callers would otherwise read length
    # 0, both write position 1, and the conversation would silently lose a
    # message -- the race the reference tracks as prism-harness#2.
    thread = a_session().thread()

    threads = [
        threading.Thread(target=lambda letter=letter: thread.record([{"content": letter}]))
        for letter in ("a", "b")
    ]
    for worker in threads:
        worker.start()
    for worker in threads:
        worker.join()

    assert [entry.position for entry in thread.messages()] == [1, 2]


def test_carries_the_run_id_that_produced_a_message() -> None:
    thread = a_session().thread()
    thread.record([{"type": "user", "content": "x"}], "run-1")

    assert thread.messages()[0].run_id == "run-1"


def test_records_nothing_for_an_empty_list() -> None:
    thread = a_session().thread()

    assert thread.record([]) == []
    assert thread.count() == 0


def test_clear_empties_the_conversation_without_touching_session_state() -> None:
    live = a_session()
    live.using_mode("plan")
    live.thread().record([{"type": "user", "content": "x"}])

    live.thread().clear()

    assert live.thread().count() == 0
    assert live.mode() == "plan"


# -- the default harness -----------------------------------------------------


def test_the_default_harness_opens_and_refuses_durable_state() -> None:
    # Not an oversight to smooth over. A package that silently accepted an
    # in-memory durable store would pass every test in one process and lose a
    # half-executed action the first time it ran on two.
    harness = PrismHarness()

    harness.ephemeral_store()

    with pytest.raises(Exception, match="VOLATILE"):
        harness.for_(Participant("User", 1)).session("s")
