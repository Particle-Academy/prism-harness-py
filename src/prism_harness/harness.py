"""The entry point: ``harness.for_(participant).session('support')``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from prism_harness.runtime import AgentRuntime

from prism_harness.address import (
    ParticipantAdapters,
    ParticipantRef,
    SessionAddress,
    validate_agent,
)
from prism_harness.address_index import IndexedSessionStore, StoreAddressIndex, address_limit
from prism_harness.session import Participant, Session
from prism_harness.stores.base import SessionStore
from prism_harness.stores.manager import SessionStoreManager, StoreFactory
from prism_harness.stores.memory import MemorySessionStore
from prism_harness.tasks import StoreTaskSource
from prism_harness.thread import Thread

__all__ = ["PendingSession", "PrismHarness"]


class AgentHarnessConfig(TypedDict, total=False):
    ttl_seconds: float | None
    default_scope: str
    default_task_list: str


class PrismHarness:
    """Mirrors the reference's ``PrismHarness::for($user)->session('support')``,
    including the two-step shape -- the participant is chosen first and the
    scope second, because one participant holds several unrelated conversations
    and the pair is what addresses one of them.

    ``for`` is a Python keyword, so the method is ``for_``. Forced rather than
    chosen, like ``Media.as_`` in ``prism-ai-core``.

    **The default is deliberately not usable for durable state.** With no
    drivers configured both slots resolve to an in-memory store, and the manager
    then REFUSES it for the durable slot -- because it reports itself volatile
    and the durable slot holds approvals a human has not answered yet.
    Constructing a harness therefore works, and asking it for durable state
    fails loudly with a message that names the fix.

    That is not an oversight to be smoothed over. A package that silently
    accepted an in-memory durable store would pass every test in one process and
    lose a half-executed action the first time it was deployed on two.
    """

    def __init__(
        self,
        drivers: Mapping[str, StoreFactory] | None = None,
        stores: Mapping[str, str] | None = None,
        ttl_seconds: float | None = None,
        participant_adapters: ParticipantAdapters | None = None,
        max_addresses_per_participant: int | None = None,
        runtime: AgentRuntime | None = None,
        default_scope: str = "default",
        default_task_list: str = "default",
        agents: dict[str, AgentHarnessConfig] | None = None,
    ) -> None:
        self._runtime = runtime
        self._default_scope, self._default_task_list = default_scope, default_task_list
        _validate_harness_config(
            {"default_scope": default_scope, "default_task_list": default_task_list}
        )
        self._agents = {agent: options.copy() for agent, options in (agents or {}).items()}
        for agent, options in self._agents.items():
            validate_agent(agent)
            _validate_harness_config(options)
        self._ttl_seconds = ttl_seconds
        self._max_addresses = address_limit(max_addresses_per_participant)
        self._participants = (
            participant_adapters if participant_adapters is not None else ParticipantAdapters()
        )
        self._stores = SessionStoreManager(
            drivers=drivers if drivers is not None else {"default": MemorySessionStore},
            stores=stores,
        )

    def for_(self, participant: Participant | ParticipantRef) -> PendingSession:
        """Bind to a participant. Returns a builder, because the scope comes next."""
        return self._pending(participant, None)

    def for_agent(self, agent: str | None) -> AgentHarness:
        validate_agent(agent)
        return AgentHarness(self, agent)

    def effective_tools(
        self,
        agent: str | None,
        mode: str,
        participant: Participant | ParticipantRef,
        scope: str | None = None,
    ) -> list[dict[str, str]]:
        if self._runtime is None:
            raise ValueError("Tool inspection requires a configured Harness runtime.")
        return self._runtime.effective_tools(self._pending(participant, agent).session(scope), mode)

    def _pending(
        self,
        participant: Participant | ParticipantRef,
        agent: str | None,
        actor: object | None = None,
    ) -> PendingSession:
        validate_agent(agent)
        options = self._agents.get(agent, {}) if agent is not None else AgentHarnessConfig()
        return PendingSession(
            participant,
            self._stores,
            options.get("ttl_seconds", self._ttl_seconds),
            agent,
            actor,
            self._participants,
            self._max_addresses,
            options.get("default_scope", self._default_scope),
        )

    def tasks(self, list: str | None = None, agent: str | None = None) -> StoreTaskSource:
        if list is None:
            options = self._agents.get(agent, {}) if agent is not None else AgentHarnessConfig()
            list = options.get("default_task_list", self._default_task_list)
        address = SessionAddress(None, f"task-list:{list}", agent)
        durable = self._stores.durable()
        store = (
            durable
            if agent is None
            else IndexedSessionStore(
                durable,
                StoreAddressIndex(durable, self._max_addresses),
                address,
                "durable",
                "tasks",
            )
        )
        key = f"{list}:tasks" if agent is None else f"agent-list:{address.index_key()}:tasks"
        return StoreTaskSource(store, key)

    def shared_thread(self, scope: str, agent: str | None = None) -> Thread:
        address, durable = SessionAddress(None, scope, agent), self._stores.durable()
        store = (
            durable
            if agent is None
            else IndexedSessionStore(
                durable,
                StoreAddressIndex(durable, self._max_addresses),
                address,
                "durable",
                "thread",
            )
        )
        return Thread(store, f"thread:{address.thread_key()}")

    def find_shared(self, scope: str, agent: str | None = None) -> Thread | None:
        thread = self.shared_thread(scope, agent)
        return thread if thread.exists() else None

    def ephemeral_store(self) -> SessionStore:
        return self._stores.ephemeral()

    def durable_store(self) -> SessionStore:
        return self._stores.durable()


class PendingSession:
    def __init__(
        self,
        participant: Participant | ParticipantRef,
        stores: SessionStoreManager,
        ttl_seconds: float | None,
        agent: str | None = None,
        actor: object | None = None,
        participants: ParticipantAdapters | None = None,
        max_addresses_per_participant: int | None = None,
        default_scope: str = "default",
    ) -> None:
        self._participant = participant
        self._stores = stores
        self._ttl_seconds = ttl_seconds
        self._agent = agent
        self._max_addresses = address_limit(max_addresses_per_participant)
        self._default_scope = default_scope
        self._actor = actor
        self._participants = participants if participants is not None else ParticipantAdapters()

    def as_actor(self, actor: object) -> PendingSession:
        if isinstance(self._participant, ParticipantRef):
            raise TypeError("Typed participants require a registered actor adapter.")
        return PendingSession(
            self._participant,
            self._stores,
            self._ttl_seconds,
            self._agent,
            actor,
            self._participants,
            self._max_addresses,
            self._default_scope,
        )

    def session(self, scope: str | None = None) -> Session:
        """Resolve the session for a scope.

        The stores are resolved HERE, which is what makes the volatile-durable
        guard fire when a session is opened rather than at some later moment
        when an approval needs saving.
        """
        actor = (
            self._participants.actor(self._participant, self._actor)
            if isinstance(self._participant, ParticipantRef)
            else (self._participant if self._actor is None else self._actor)
        )
        durable = self._stores.durable()
        return Session(
            participant=self._participant,
            scope=self._default_scope if scope is None else scope,
            ephemeral=self._stores.ephemeral(),
            durable=durable,
            ttl_seconds=self._ttl_seconds,
            agent=self._agent,
            actor=actor,
            address_index=(
                StoreAddressIndex(durable, self._max_addresses)
                if self._agent is not None or isinstance(self._participant, ParticipantRef)
                else None
            ),
        )


class AgentHarness:
    """Namespace-bound view; stores are shared without a mutable current-agent flag."""

    def __init__(self, harness: PrismHarness, agent: str | None) -> None:
        validate_agent(agent)
        self._harness, self._agent = harness, agent

    def for_(self, participant: Participant | ParticipantRef) -> PendingSession:
        return self._harness._pending(participant, self._agent)

    def effective_tools(
        self, mode: str, participant: Participant | ParticipantRef, scope: str | None = None
    ) -> list[dict[str, str]]:
        return self._harness.effective_tools(self._agent, mode, participant, scope)

    def tasks(self, list: str | None = None) -> StoreTaskSource:
        return self._harness.tasks(list, self._agent)

    def shared_thread(self, scope: str) -> Thread:
        return self._harness.shared_thread(scope, self._agent)

    def find_shared(self, scope: str) -> Thread | None:
        return self._harness.find_shared(scope, self._agent)


def _validate_harness_config(config: AgentHarnessConfig) -> None:
    ttl = config.get("ttl_seconds")
    if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, int) or ttl < 0):
        raise ValueError("Agent ephemeral TTL must be nonnegative whole seconds or null.")
    for key in ("default_scope", "default_task_list"):
        value: object = config.get(key)
        if value is not None and (
            not isinstance(value, str) or (key == "default_task_list" and not value)
        ):
            raise ValueError("Agent scope/list defaults must be valid strings.")
