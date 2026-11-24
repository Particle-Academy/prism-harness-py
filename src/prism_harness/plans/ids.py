"""Host-minted plan and step ids: ``pl_`` / ``ps_`` and a ULID (G-360).

Never chosen by the model. The generator is replaceable so the parity corpus
can mint the same ids in all three languages.
"""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable

__all__ = ["PlanIds", "is_step_id", "ulid"]

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_STEP = re.compile(r"ps_[0-9A-HJKMNP-TV-Z]{26}")


class PlanIds:
    """``mint`` is given a prefix (``pl`` or ``ps``) and returns a full id."""

    def __init__(self, mint: Callable[[str], str] | None = None) -> None:
        self._mint = mint

    def plan(self) -> str:
        return self._mint("pl") if self._mint is not None else "pl_" + ulid()

    def step(self) -> str:
        return self._mint("ps") if self._mint is not None else "ps_" + ulid()


def is_step_id(value: object) -> bool:
    return isinstance(value, str) and _STEP.fullmatch(value) is not None


def ulid() -> str:
    moment = int(time.time() * 1000)
    out = ""
    for _ in range(10):
        out = _CROCKFORD[moment % 32] + out
        moment //= 32
    bits = int.from_bytes(secrets.token_bytes(10), "big")
    tail = ""
    for _ in range(16):
        tail = _CROCKFORD[bits % 32] + tail
        bits //= 32
    return out + tail
