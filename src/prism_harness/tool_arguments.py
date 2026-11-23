"""Whether a tool call's arguments can be used."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

# The deepest nesting accepted. A provider wraps the arguments in its own request
# envelope, and the PHP reference encodes that request with a depth limit of 512,
# so the limit leaves room for the deepest envelope.
_MAX_ARGUMENT_DEPTH = 500

_INTEGER_KEY = re.compile(r"0|-?[1-9][0-9]*", re.ASCII)

_CONTROL_ESCAPES = {
    "\x08": "\\b",
    "\x09": "\\t",
    "\x0a": "\\n",
    "\x0c": "\\f",
    "\x0d": "\\r",
}


def usable_arguments(value: Any) -> dict[str, Any] | None:
    """The arguments as a map of named values, or ``None`` when unusable.

    This package has no dependency on prism core, so this is core's
    classification, spelled the same way in each language.

    An empty string, the text ``0``, ``null``, an empty list and an empty object
    all mean no arguments, and read as ``{}``. Anything else that is not a map
    of named values, or that a follow-up request could not carry, is refused.
    """
    if value is None or (isinstance(value, str) and value in ("", "0")):
        return {}

    decoded = value

    if isinstance(value, str):
        try:
            decoded = _decode(value)
        except (ValueError, RecursionError):
            return None

    if decoded is None or (isinstance(decoded, list) and not decoded):
        return {}

    if not isinstance(decoded, Mapping):
        return None

    mapped = dict(decoded)

    if any(_is_integer_key(key) for key in mapped) or not _can_be_sent_back(mapped):
        return None

    return mapped


def _refuse_constant(name: str) -> Any:
    """``NaN`` and ``Infinity`` are not JSON; Python's decoder accepts them."""
    raise ValueError(f"{name} is not valid JSON")


def _decode(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_refuse_constant)
    except ValueError:
        return json.loads(
            _escape_control_characters_in_strings(text), parse_constant=_refuse_constant
        )


def _has_lone_surrogate(text: str) -> bool:
    return any("\ud800" <= char <= "\udfff" for char in text)


def _is_finite_number(value: float) -> bool:
    """Whether the number fits a double, which is what the other languages hold."""
    if isinstance(value, float):
        return math.isfinite(value)

    try:
        float(value)
    except OverflowError:
        return False

    return True


def _can_be_sent_back(root: dict[str, Any]) -> bool:
    """Walked with a stack, so deep input cannot overflow the call stack here."""
    pending: list[tuple[Any, int]] = [(root, 1)]

    while pending:
        value, depth = pending.pop()

        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if not _is_finite_number(value):
                return False
        elif isinstance(value, str):
            if _has_lone_surrogate(value):
                return False
        elif isinstance(value, (list, dict)):
            if depth > _MAX_ARGUMENT_DEPTH:
                return False
            if isinstance(value, dict):
                for key in value:
                    if (
                        not isinstance(key, str)
                        or key.startswith("\x00")
                        or _has_lone_surrogate(key)
                    ):
                        return False
                pending.extend((item, depth + 1) for item in value.values())
            else:
                pending.extend((item, depth + 1) for item in value)

    try:
        json.dumps(root, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return False

    return True


def _is_integer_key(key: object) -> bool:
    """A key PHP would store as an integer: canonical decimal, within 64 bits."""
    if isinstance(key, int):
        return True

    # Twenty characters is the longest 64-bit integer, with its sign.
    if not isinstance(key, str) or len(key) > 20 or _INTEGER_KEY.fullmatch(key) is None:
        return False

    return -(2**63) <= int(key) <= 2**63 - 1


def _escape_control_characters_in_strings(raw: str) -> str:
    """Escape raw control characters inside JSON string literals.

    Anything else is left as it is, so that the decode refuses it: a control
    character outside a string, or straight after a backslash, has no value it
    can be repaired to.
    """
    result: list[str] = []
    in_string = False
    escaped = False

    for char in raw:
        if ord(char) <= 0x1F:
            if in_string and not escaped:
                result.append(_CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}"))
            else:
                result.append(char)
            escaped = False
            continue

        result.append(char)

        if escaped:
            escaped = False
        elif in_string and char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string

    return "".join(result)
