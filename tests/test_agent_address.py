from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from prism_harness import (
    AddressLimitExceeded,
    FileSessionStore,
    HarnessError,
    IndexLockScopeUnsupported,
    IndexRecoveryRequired,
    MemorySessionStore,
    Participant,
    ParticipantAdapters,
    ParticipantRef,
    PrismHarness,
    Session,
    SessionAddress,
    StoreAddressIndex,
)


def test_typed_child_inherits_bound_actor_without_resolving_adapter_again(tmp_path: Path) -> None:
    store, adapters = FileSessionStore(tmp_path), ParticipantAdapters()
    harness = PrismHarness(
        drivers={"owned": lambda: store},
        stores={"ephemeral": "owned", "durable": "owned"},
        participant_adapters=adapters,
    )
    participant, actor, later = ParticipantRef("automation", "child-owner"), object(), object()
    resolutions = 0
    with pytest.raises(ValueError, match="actor adapter"):
        harness._pending(participant, "support", actor).session("missing")

    def resolve(_: ParticipantRef) -> object:
        nonlocal resolutions
        resolutions += 1
        return actor if resolutions == 1 else later

    adapters.register("automation", resolve)
    parent = harness.for_agent("support").for_(participant).session("parent")
    child = harness._pending(parent.participant, parent.agent, parent.actor).session(
        "parent::sub::worker"
    )
    assert child.actor is actor
    assert child.participant is participant
    assert resolutions == 1


def setup(tmp_path: Path) -> tuple[PrismHarness, FileSessionStore, ParticipantAdapters]:
    store, adapters = FileSessionStore(tmp_path), ParticipantAdapters()
    harness = PrismHarness(
        drivers={"owned": lambda: store},
        stores={"ephemeral": "owned", "durable": "owned"},
        participant_adapters=adapters,
    )
    return harness, store, adapters


def test_isolates_named_state_transcripts_capabilities_and_task_lists(tmp_path: Path) -> None:
    harness, _, _ = setup(tmp_path)
    participant = Participant("User", 7)
    legacy = harness.for_(participant).session("same:/雪")
    support = harness.for_agent("support").for_(participant).session("same:/雪")
    charlotte = harness.for_agent("charlotte").for_(participant).session("same:/雪")
    digest = hashlib.sha1(b"User").hexdigest()[:12]
    assert legacy.key() == f"session:{digest}:7:same:/雪"
    support.using_mode("reply").using_capability("private", {"value": True})
    support.tasks().add("support only", "one")
    support.thread().record([{"type": "user", "text": "support only"}])
    assert charlotte.mode() is None
    assert charlotte.capability("private") is None
    assert charlotte.tasks().claim("worker") is None
    assert charlotte.thread().messages() == legacy.thread().messages() == []
    assert (
        harness.for_agent("support").for_(Participant("User", "7")).session("same:/雪").mode()
        == "reply"
    )


def test_noncreating_reads_allocate_no_files_index_records_or_locks(tmp_path: Path) -> None:
    harness, _, _ = setup(tmp_path)
    session = harness.for_agent("support").for_(Participant("User", 7)).session("absent")
    before = list(tmp_path.iterdir())
    assert session.find_thread() is None
    assert harness.for_agent("support").find_shared("absent") is None
    assert list(tmp_path.iterdir()) == before
    session.thread().record([{"type": "user", "text": "one"}])
    found = session.find_thread()
    assert found is not None and found.count() == 1
    assert (
        harness.for_agent("charlotte").for_(session.participant).session("absent").find_thread()
        is None
    )


def test_explicit_references_require_actor_bindings_separate_from_ownership(tmp_path: Path) -> None:
    harness, _, adapters = setup(tmp_path)
    participant, actor = ParticipantRef("automation", "bot/雪"), object()
    with pytest.raises(ValueError, match="actor adapter"):
        harness.for_agent("support").for_(participant).session("s")
    adapters.register("automation", lambda ref: actor)
    session = harness.for_agent("support").for_(participant).session("s")
    assert session.actor is actor and session.participant == participant
    with pytest.raises(ValueError, match="already registered"):
        adapters.register("automation", lambda ref: actor)
    with pytest.raises(TypeError, match="registered actor adapter"):
        harness.for_(participant).as_actor(actor)
    plain = Participant("User", 1)
    assert (
        harness.for_(plain).as_actor(actor).session("s").key()
        == harness.for_(plain).session("s").key()
    )


def test_hostile_namespaces_are_refused_without_normalization(tmp_path: Path) -> None:
    harness, _, _ = setup(tmp_path)
    for agent in (
        "",
        "Support",
        " support",
        "support ",
        "support\n",
        "../support",
        "a/b",
        "sυpport",
        "a" * 65,
    ):
        with pytest.raises(ValueError):
            harness.for_agent(agent)


def test_bounded_index_and_named_application_lists(tmp_path: Path) -> None:
    harness, store, _ = setup(tmp_path)
    participant = Participant("User", 7)
    session, index = (
        harness.for_agent("support").for_(participant).session("s"),
        StoreAddressIndex(store),
    )
    assert index.page(participant)["entries"] == []
    session.using_mode("reply")
    session.thread().record([{"type": "user", "text": "one"}])
    rows = index.page(participant)
    assert (
        len(rows["entries"]) == 1
        and rows["entries"][0]["address_key"] == session.address.index_key()
    )
    assert index.page(participant, 1)["entries"] == []
    with pytest.raises(ValueError):
        index.page(participant, limit=101)
    harness.for_agent("support").tasks("same").add("owned", "one")
    assert harness.for_agent("charlotte").tasks("same").claim("worker") is None
    assert harness.tasks("same").claim("worker") is None


def test_every_shared_vector_against_independently_authored_expectations() -> None:
    cases = json.loads(
        (Path(__file__).parent / "fixtures/harness-agent-addresses.json").read_text(
            encoding="utf-8"
        )
    )["cases"]
    assert len(cases) == 46
    for row in cases:
        data = row["input"]
        participant = None if data["participant"] is None else Participant(**data["participant"])
        try:
            address = SessionAddress(participant, data["scope"], data["agent"])
            result = {
                "session_key": None if participant is None else address.session_key(),
                "thread_key": address.thread_key(),
                "index_key": address.index_key(),
            }
        except (ValueError, TypeError):
            result = {"error": "invalid_address"}
        assert result == row["expected"], row["id"]


def test_index_failure_prevents_resource_data_write(tmp_path: Path) -> None:
    _, store, _ = setup(tmp_path)

    class RefusingIndex(StoreAddressIndex):
        def record(self, address: SessionAddress, slot: str, kind: str, key: str) -> None:
            raise ValueError("Synthetic index failure")

    session = Session(
        Participant("User", 7),
        "s",
        store,
        store,
        agent="support",
        address_index=RefusingIndex(store),
    )
    with pytest.raises(ValueError, match="Synthetic index failure"):
        session.using_mode("reply")
    assert store.get(f"{session.key()}:ephemeral") is None


def test_fixed_page_boundary_and_optional_quota_before_state(tmp_path: Path) -> None:
    harness, store, _ = setup(tmp_path)
    participant = Participant("User", 7)
    for index in range(101):
        harness.for_agent("support").for_(participant).session(f"scope:{index}").using_mode("reply")
    addresses = StoreAddressIndex(store)
    first = addresses.page(participant)
    assert len(first["entries"]) == 100 and first["next_cursor"] == 100
    second = addresses.page(participant, first["next_cursor"])
    assert len(second["entries"]) == 1 and second["next_cursor"] is None
    limited = PrismHarness(
        drivers={"owned": lambda: store},
        stores={"ephemeral": "owned", "durable": "owned"},
        max_addresses_per_participant=101,
    )
    extra = limited.for_agent("support").for_(participant).session("scope:extra")
    with pytest.raises(AddressLimitExceeded) as refusal:
        extra.using_mode("refused")
    assert refusal.value.code == "address_limit_exceeded"
    assert extra.mode() is None
    limited.for_agent("support").for_(participant).session("scope:0").using_mode("still usable")


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_interrupted_publication_recovers_before_resource_data(
    tmp_path: Path, fail_at: int, phase: str
) -> None:
    class InterruptedStore(FileSessionStore):
        armed = True
        writes = 0

        def put(self, key: str, value: dict[str, Any], ttl_seconds: float | None = None) -> None:
            if self.armed and key.startswith("harness-address-index:"):
                self.writes += 1
                if self.writes == fail_at and phase == "before":
                    raise ValueError("Synthetic publication interruption")
            super().put(key, value, ttl_seconds)
            if self.armed and self.writes == fail_at and phase == "after":
                raise ValueError("Synthetic publication interruption")

    store = InterruptedStore(tmp_path)
    participant = Participant("User", 7)
    harness = PrismHarness(
        drivers={"owned": lambda: store}, stores={"ephemeral": "owned", "durable": "owned"}
    )
    session = harness.for_agent("support").for_(participant).session("s")
    with pytest.raises(ValueError, match="Synthetic publication interruption"):
        session.using_mode("reply")
    assert store.get(f"{session.key()}:ephemeral") is None
    index = StoreAddressIndex(store)
    if phase == "before" and fail_at == 1:
        assert index.page(participant)["entries"] == []
    elif phase == "after" and fail_at == 4:
        assert len(index.page(participant)["entries"]) == 1
    else:
        with pytest.raises(IndexRecoveryRequired):
            index.page(participant)
    store.armed = False
    index.recover(participant)
    index.recover(participant)
    session.using_mode("reply")
    session.using_mode("recovery is idempotent")
    assert len(StoreAddressIndex(store).page(participant)["entries"]) == 1
    assert (
        harness.for_agent("support").for_(participant).session("s").mode()
        == "recovery is idempotent"
    )


def test_process_only_locks_refuse_indexed_namespaces(tmp_path: Path) -> None:
    with pytest.raises(HarnessError, match="durable store") as refusal:
        StoreAddressIndex(MemorySessionStore())
    assert refusal.value.code == "unsafe_state_configuration"

    class ProcessOnly(FileSessionStore):
        def lock_scope(self) -> str:
            return "process"

    store = ProcessOnly(tmp_path)
    harness = PrismHarness(
        drivers={"owned": lambda: store}, stores={"ephemeral": "owned", "durable": "owned"}
    )
    with pytest.raises(IndexLockScopeUnsupported):
        harness.for_agent("support").for_(Participant("User", 7)).session("s")
    with pytest.raises(IndexLockScopeUnsupported):
        Session(Participant("User", 7), "s", store, store, agent="support")
    harness.for_(Participant("User", 7)).session("s").using_mode("legacy works")


def test_memory_state_with_durable_file_slot_supports_named_agents(tmp_path: Path) -> None:
    store, memory = FileSessionStore(tmp_path), MemorySessionStore()
    harness = PrismHarness(
        drivers={"memory": lambda: memory, "files": lambda: store},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_agent("support").for_(Participant("User", 7)).session("s")
    session.using_mode("memory works")
    session.thread().record([{"type": "user", "text": "durable"}])
    assert session.mode() == "memory works" and session.thread().count() == 1


def test_undeclared_custom_scope_refused_before_named_state_write(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)

    class Undeclared:
        get = staticmethod(store.get)
        put = staticmethod(store.put)
        forget = staticmethod(store.forget)
        with_lock = staticmethod(store.with_lock)
        durability = staticmethod(store.durability)

    custom = Undeclared()
    harness = PrismHarness(
        drivers={"owned": lambda: custom}, stores={"ephemeral": "owned", "durable": "owned"}
    )
    with pytest.raises(IndexLockScopeUnsupported) as refusal:
        harness.for_agent("support").for_(Participant("User", 7)).session("s")
    assert refusal.value.code == "index_lock_scope_unsupported"
    with pytest.raises(IndexLockScopeUnsupported):
        Session(Participant("User", 7), "s", custom, custom, agent="support")
    assert list(tmp_path.iterdir()) == []
    harness.for_(Participant("User", 7)).session("s").using_mode("legacy works")
