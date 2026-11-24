"""A host-owned "runs without asking" tool policy (G-326).

Mirrors prism-harness-ts/src/tool-policy.ts. The JSON shape is identical in
every language, so a host builds it once::

    {"revision": "r7", "allow": [{"server": "files", "tool": "read"}, {"tool": "search"}], "honor": true}
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .errors import HarnessError

__all__ = [
    "RESERVED_TOOLS",
    "TOOL_TIERS",
    "ToolDeclaration",
    "ToolPolicy",
    "ToolPolicyEntry",
    "ToolTier",
    "args_digest",
    "canonical_json",
    "is_person_only_tool",
    "is_well_formed_tool_name",
    "normalise_tool_name",
    "tool_declaration",
]

#: A tool's own, advisory, declaration of how much it can change.
ToolTier = Literal["read_only", "write", "destructive"]
TOOL_TIERS: tuple[ToolTier, ...] = ("read_only", "write", "destructive")


class ToolDeclaration(Protocol):
    """What the policy needs to know about a tool: its name and declarations."""

    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class _Declared:
    name: str
    server: str | None = None
    tier: str | None = None


def tool_declaration(tool: object) -> _Declared:
    """A tool's name, ``server`` and ``tier``, read structurally."""
    name = getattr(tool, "name", None)
    server = getattr(tool, "server", None)
    tier = getattr(tool, "tier", None)
    if isinstance(tool, Mapping):
        name, server, tier = tool.get("name"), tool.get("server"), tool.get("tier")
    return _Declared(
        name if isinstance(name, str) else "",
        server if isinstance(server, str) else None,
        tier if isinstance(tier, str) else None,
    )


#: Characters that are invisible at either end of a tool name. The same
#: codepoint set human-plus's TrustPolicy uses, spelled out identically in all
#: three harness languages rather than left to each language's own strip().
_INVISIBLE_CLASS = (
    "\u0000\u0009-\u000d\u0020\u0085\u00a0\u1680\u2000-\u200d\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_INVISIBLE = re.compile(f"^[{_INVISIBLE_CLASS}]+|[{_INVISIBLE_CLASS}]+\\Z")
_WELL_FORMED = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}\Z")
_SEPARATORS = re.compile(r"[_.:-]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_RESERVED_VERBS = frozenset({"confirm", "reject", "accept", "approve", "deny"})

#: The agent plan tools (G-360): a plan is a person's agreement, never auto-allowed.
RESERVED_TOOLS: tuple[str, ...] = (
    "plan_draft",
    "plan_propose_change",
    "plan_pause_for_person",
    "plan_complete_step",
)


def normalise_tool_name(name: str) -> str:
    """A name with the invisible edges removed."""
    return _INVISIBLE.sub("", name)


def is_well_formed_tool_name(name: str) -> bool:
    """ASCII letters, digits, ``_ . : -``; at most 128; ASCII-only defeats homoglyphs."""
    return _WELL_FORMED.match(name) is not None


def is_person_only_tool(name: str) -> bool:
    """A confirmation or approval tool: any token is a reserved verb, or the name
    is a plan tool (G-360).

    The plan tools are named here because the verb tokeniser does not catch
    them: a plan_draft or a plan change that a policy let run without asking
    would be a plan no person approved.
    """
    if normalise_tool_name(name).lower() in RESERVED_TOOLS:
        return True
    return any(
        token.lower() in _RESERVED_VERBS
        for segment in _SEPARATORS.split(normalise_tool_name(name))
        for token in _CAMEL.split(segment)
    )


def _is_exact_name(name: object) -> bool:
    return (
        isinstance(name, str)
        and normalise_tool_name(name) == name
        and is_well_formed_tool_name(name)
    )


def _refuse(reason: str) -> HarnessError:
    return HarnessError("tool_policy_refused", f"This tool policy is refused: {reason}")


@dataclass(frozen=True)
class ToolPolicyEntry:
    tool: str
    server: str | None = None

    def to_json(self) -> dict[str, str]:
        return (
            {"tool": self.tool}
            if self.server is None
            else {"server": self.server, "tool": self.tool}
        )


class ToolPolicy:
    """Which approval-gated tools run without asking a person. Owned by the HOST.

    Matching is exact: a name must already be in normal form and well formed,
    so a padded or look-alike name never matches. An entry without ``server``
    matches only a native tool, never one an MCP server provides. No wildcards.

    REFUSED AT CONSTRUCTION, never silently dropped: a malformed entry, a
    confirmation or approval tool, and an entry naming a tool that declares
    ``tier: destructive`` among the declarations passed in. ``honor: false``
    turns the list off. The policy has no tool surface and is never sent to
    the model.
    """

    __slots__ = ("_allow", "_honor", "_revision")

    def __init__(self, revision: str, allow: tuple[ToolPolicyEntry, ...], honor: bool) -> None:
        self._revision = revision
        self._allow = allow
        self._honor = honor

    @classmethod
    def none(cls) -> ToolPolicy:
        """Nothing runs without asking."""
        return cls("", (), True)

    @classmethod
    def from_json(cls, data: object, tools: Iterable[object] = ()) -> ToolPolicy:
        """Raises ``HarnessError`` with code ``tool_policy_refused``."""
        if not isinstance(data, Mapping):
            raise _refuse("it must be an object.")
        for key in data:
            if key not in ("revision", "allow", "honor"):
                raise _refuse(f'unknown key "{key}".')
        revision, allow, honor = data.get("revision"), data.get("allow"), data.get("honor")
        if not isinstance(revision, str):
            raise _refuse("revision must be a string.")
        if not isinstance(honor, bool):
            raise _refuse("honor must be a boolean.")
        if not isinstance(allow, list):
            raise _refuse("allow must be a list.")

        declared = [tool_declaration(tool) for tool in tools]
        entries: list[ToolPolicyEntry] = []
        for raw in allow:
            if not isinstance(raw, Mapping):
                raise _refuse("every allow entry must be an object.")
            for key in raw:
                if key not in ("server", "tool"):
                    raise _refuse(f'unknown entry key "{key}".')
            tool, server = raw.get("tool"), raw.get("server")
            if not _is_exact_name(tool):
                raise _refuse("an entry names a tool that is not an exact, well-formed name.")
            if "server" in raw and not _is_exact_name(server):
                raise _refuse("an entry names a server that is not an exact, well-formed name.")
            tool = str(tool)
            scoped = server if isinstance(server, str) else None
            if is_person_only_tool(tool):
                raise _refuse(
                    f'"{tool}" is a confirmation or approval tool, which only a person may run.'
                )
            if any(item.tier == "destructive" and _same(item, scoped, tool) for item in declared):
                raise _refuse(f'"{tool}" declares tier destructive, which is never auto-allowed.')
            entries.append(ToolPolicyEntry(tool, scoped))

        return cls(revision, tuple(entries), honor)

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def honor(self) -> bool:
        return self._honor

    @property
    def allow(self) -> tuple[ToolPolicyEntry, ...]:
        return self._allow

    def allows(self, tool: object) -> bool:
        """Does this call run without asking?"""
        if not self._honor:
            return False
        declared = tool_declaration(tool)
        if (
            not _is_exact_name(declared.name)
            or is_person_only_tool(declared.name)
            or declared.tier == "destructive"
        ):
            return False
        if declared.server is not None and not _is_exact_name(declared.server):
            return False
        return any(_same(declared, entry.server, entry.tool) for entry in self._allow)

    def to_json(self) -> dict[str, Any]:
        return {
            "revision": self._revision,
            "allow": [entry.to_json() for entry in self._allow],
            "honor": self._honor,
        }


def _same(tool: _Declared, server: str | None, name: str) -> bool:
    return tool.name == name and tool.server == server


_EXACT_INTEGER = 2**53


def _number(value: float) -> str:
    """ECMAScript Number::toString, as RFC 8785 requires."""
    if not math.isfinite(value):
        raise TypeError("Canonical JSON has no representation for a non-finite number.")
    if value == 0:
        return "0"
    if value == int(value) and abs(value) < _EXACT_INTEGER:
        return str(int(value))
    sign = "-" if value < 0 else ""
    mantissa, _, exponent = repr(abs(value)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = whole + fraction
    # The decimal point sits `point` digits into `digits`.
    point = len(whole) + (int(exponent) if exponent else 0)
    stripped = digits.lstrip("0")
    point -= len(digits) - len(stripped)
    digits = stripped.rstrip("0") or "0"
    k = len(digits)
    if k <= point <= 21:
        return sign + digits + "0" * (point - k)
    if 0 < point <= 21:
        return sign + digits[:point] + "." + digits[point:]
    if -6 < point <= 0:
        return sign + "0." + "0" * (-point) + digits
    shift = point - 1
    tail = "." + digits[1:] if k > 1 else ""
    return f"{sign}{digits[0]}{tail}e{'+' if shift > 0 else '-'}{abs(shift)}"


def _utf16(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


def canonical_json(value: object) -> str:
    """RFC 8785 (JCS): UTF-16 key order, no whitespace, ECMAScript numbers."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        # A JSON number is a double: beyond 2**53 it is formatted as one.
        return str(value) if abs(value) < _EXACT_INTEGER else _number(float(value))
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        keys = sorted((str(key) for key in value), key=_utf16)
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{canonical_json(value[key])}"
                for key in keys
            )
            + "}"
        )
    raise TypeError(f"Canonical JSON has no representation for {type(value).__name__}.")


def args_digest(args: object) -> str:
    """``sha256:`` and the hex sha256 of the arguments' canonical JSON. Never the arguments."""
    canonical = canonical_json({} if args is None else args)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
