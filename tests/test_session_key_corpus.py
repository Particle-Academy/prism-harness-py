"""The cross-language session-key corpus from ``prism-parity``.

This is the most consequential string in the ecosystem and the one whose drift
is hardest to notice. A session key is an ADDRESS: it is what lets a PHP
application and this agent resolve the same conversation, and it is what the
store is keyed by.

If it drifted, nothing would error. Each language would read and write its own
key perfectly happily, and the two would simply stop seeing each other's turns
-- a conversation that appears empty, with no exception, no log line and no
failing test anywhere in either codebase. That is precisely the shape a
per-language suite cannot see, because each one asserts against the key its own
code produced.

Mirrors prism-harness-ts/test/session-key-corpus.test.ts case for case.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from prism_harness import Durability, Participant, Session

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "harness-session-key.json").read_text(encoding="utf-8")
)
CASES: list[dict[str, Any]] = CORPUS["cases"]


class UnusedStore:
    """Never read from. ``key()`` derives from the participant and scope alone."""

    def get(self, key: str) -> dict[str, Any] | None:
        return None

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        return None

    def forget(self, key: str) -> None:
        return None

    def with_lock(
        self,
        key: str,
        callback: Callable[[], Any],
        ttl_seconds: float = 10,
        wait_seconds: float = 5,
    ) -> Any:
        return callback()

    def durability(self) -> Durability:
        return Durability.VOLATILE


def _id(case: dict[str, Any]) -> str:
    return str(case["id"])


def _key_of(case: dict[str, Any]) -> str:
    session = Session(
        Participant(type=case["participant"]["type"], id=case["participant"]["id"]),
        case["scope"],
        UnusedStore(),
        UnusedStore(),
    )

    return session.key()


def _case(case_id: str) -> dict[str, Any]:
    return next(case for case in CASES if case["id"] == case_id)


def test_the_corpus_is_whole_not_a_subset_someone_trimmed_to_green() -> None:
    assert len(CASES) == 9


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_resolves_to_the_reference_address(case: dict[str, Any]) -> None:
    assert _key_of(case) == case["key"]["php"]


def test_agrees_with_the_reference_on_every_row() -> None:
    assert [case for case in CASES if not case["agrees"]] == []


def test_two_participant_types_with_the_same_id_get_different_addresses() -> None:
    # User 7 and Team 7 must not share a conversation, and the hashed type
    # segment is the only thing keeping them apart. Asserted directly rather
    # than inferred from two rows happening to differ.
    user = _case("key-0001")
    team = _case("key-0004")

    assert user["participant"]["id"] == team["participant"]["id"]
    assert _key_of(user) != _key_of(team)


def test_hashes_the_type_as_bytes_so_non_ascii_agrees_across_languages() -> None:
    # sha1 is over bytes. A language hashing UTF-16 code units, or one that
    # latin-1'd the string first, produces a different digest from source that
    # looks identical in an editor.
    case = _case("key-0008")

    assert _key_of(case) == case["key"]["php"]


def test_does_not_treat_a_zero_id_as_absent() -> None:
    # Falsy in all three languages. A truthiness check anywhere on the id path
    # silently produces a different address, or an empty segment.
    assert ":0:" in _key_of(_case("key-0006"))
