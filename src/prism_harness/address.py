"""Immutable, host-selected addresses, independent of display/persona names."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_harness.session import Participant


def validate_agent(agent: str | None) -> None:
    if agent is not None and (
        not isinstance(agent, str) or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", agent) is None
    ):
        raise ValueError(
            "Harness namespace must be a configured lowercase identifier of 1–64 ASCII bytes."
        )


@dataclass(frozen=True, init=False)
class ParticipantRef:
    """An explicit identity whose actor is resolved through a host adapter."""

    type: str
    id: str

    def __init__(self, type: str, id: str | int) -> None:
        if not isinstance(type, str) or isinstance(id, bool) or not isinstance(id, (str, int)):
            raise TypeError(
                "Participant references require string types and string or integer ids."
            )
        for value in (type, str(id)):
            if not value or len(value.encode("utf-8")) > 255:
                raise ValueError(
                    "Participant references require non-empty UTF-8 type and id, at most 255 bytes each."
                )
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "id", str(id))


class ParticipantAdapters:
    def __init__(self) -> None:
        self._actors: dict[str, Callable[[ParticipantRef], object]] = {}

    def register(self, type: str, actor: Callable[[ParticipantRef], object]) -> None:
        ParticipantRef(type, "validation")
        if type in self._actors:
            raise ValueError("A participant adapter is already registered for this type.")
        self._actors[type] = actor

    def actor(self, participant: ParticipantRef, inherited_actor: object | None = None) -> object:
        resolve = self._actors.get(participant.type)
        if resolve is None:
            raise ValueError("No participant actor adapter is registered for this type.")
        actor = resolve(participant) if inherited_actor is None else inherited_actor
        if actor is None:
            raise ValueError("Participant actor adapter must return an actor object.")
        return actor


@dataclass(frozen=True, init=False)
class SessionAddress:
    participant_type: str | None
    participant_id: str | None
    scope: str
    agent: str | None

    def __init__(
        self, participant: Participant | ParticipantRef | None, scope: str, agent: str | None = None
    ) -> None:
        validate_agent(agent)
        if agent is not None or isinstance(participant, ParticipantRef):
            if not isinstance(scope, str):
                raise TypeError("Harness addresses require UTF-8 scopes.")
            scope.encode("utf-8")
            if participant is not None:
                if (
                    not isinstance(participant.type, str)
                    or isinstance(participant.id, bool)
                    or not isinstance(participant.id, (str, int))
                ):
                    raise ValueError(
                        "Harness addresses require UTF-8 types and integer or string ids."
                    )
                participant.type.encode("utf-8")
                str(participant.id).encode("utf-8")
        object.__setattr__(
            self, "participant_type", None if participant is None else participant.type
        )
        object.__setattr__(
            self, "participant_id", None if participant is None else str(participant.id)
        )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "agent", agent)

    def session_key(self) -> str:
        if self.participant_type is None or self.participant_id is None:
            raise ValueError("A session requires a participant.")
        if self.agent is None:
            digest = hashlib.sha1(self.participant_type.encode("utf-8")).hexdigest()[:12]
            return f"session:{digest}:{self.participant_id}:{self.scope}"
        return f"session:{self._digest('prism-harness.session.v1')}"

    def thread_key(self) -> str:
        if self.agent is not None:
            return self._digest("prism-harness.thread.v1")
        legacy = json.dumps(
            [self.participant_type, self.participant_id, self.scope],
            separators=(",", ":"),
            ensure_ascii=True,
        ).replace("/", "\\/")
        return hashlib.sha256(legacy.encode("utf-8")).hexdigest()

    def index_key(self) -> str:
        return self._digest("prism-harness.address.v1")

    def _digest(self, domain: str) -> str:
        value = json.dumps(
            [domain, self.agent, self.participant_type, self.participant_id, self.scope],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
