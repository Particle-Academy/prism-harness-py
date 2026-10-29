"""Owned registration diagnostics; labels never grant authority."""

from __future__ import annotations

import re


class RegistrationCollision(ValueError):
    code = "registration_collision"

    def __init__(self, agent: str | None, name: str, previous: str, incoming: str) -> None:
        super().__init__(
            f"Harness registry collision: agent [{agent if agent is not None else 'legacy'}], "
            f"name [{name}], owners [{previous}] and [{incoming}]. "
            f"Use replace naming predecessor owner [{previous}]."
        )


def registration_owner(owner: str | None, agent: str | None) -> str:
    label = owner if owner is not None else ("legacy" if agent is None else None)
    if (
        not isinstance(label, str)
        or not label
        or len(label.encode("utf-8")) > 128
        or re.search(r"[\x00-\x1f\x7f]", label) is not None
    ):
        raise ValueError(
            "A named registration requires a non-empty UTF-8 owner label of at most 128 bytes without control characters."
        )
    return label
