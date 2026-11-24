"""The rules for text a model writes into a plan (G-360).

Every field is model output, and the model may have been steered by a document
it read. So a field is REFUSED, never truncated, when it is empty, over its cap,
not valid Unicode text (a lone surrogate, which PHP's UTF-8 check also refuses),
or contains a character from FORBIDDEN: the C0 and C1 controls (line breaks
included), the guillemets that fence model text in the plan context, zero-width
and bidi controls, the line and paragraph separators, word joiners and the byte
order mark. A field naming one of the plan frame's own markers is refused as
well, so a title cannot close the frame and write text that reads as the host's.

The same codepoint set is spelled out in all three languages. Lengths are
counted in Unicode code points in all three.
"""

from __future__ import annotations

import re

__all__ = [
    "ANSWER",
    "FORBIDDEN",
    "GOAL",
    "INSTRUCTION",
    "MARKERS",
    "OPTION",
    "OPTIONS",
    "QUESTION",
    "REASON",
    "RENDER_ANSWER",
    "RENDER_INSTRUCTION",
    "RENDER_NOTE",
    "RENDER_TITLE",
    "SUMMARY",
    "TITLE",
    "acceptable",
    "length",
    "render",
    "valid",
]

GOAL = 200
TITLE = 160
INSTRUCTION = 600
QUESTION = 300
OPTION = 80
OPTIONS = 6
SUMMARY = 200
REASON = 200
ANSWER = 1000

#: Render caps inside the plan context.
RENDER_TITLE = 120
RENDER_INSTRUCTION = 400
RENDER_NOTE = 160
RENDER_ANSWER = 300

FORBIDDEN = (
    "\u0000-\u001f\u007f-\u009f\u00ab\u00bb\u200b-\u200f\u2028-\u202e\u2060-\u2069"
    "\ufeff\ufff9-\ufffb"
)
_FORBIDDEN = re.compile(f"[{FORBIDDEN}]")
_MARKER = re.compile(r"\[(end of plan|plan)", re.IGNORECASE)

#: Lower-case markers of the plan frame; a field containing one is refused.
MARKERS = ("[plan", "[end of plan")


def valid(value: str) -> bool:
    """Whether the text can be written as UTF-8: no lone surrogate."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def acceptable(value: object, maximum: int) -> bool:
    """Whether a model-written field is acceptable as it stands."""
    if not isinstance(value, str) or value == "" or not valid(value):
        return False
    if len(value) > maximum:
        return False
    if _FORBIDDEN.search(value) is not None:
        return False
    lower = value.lower()
    return not any(marker in lower for marker in MARKERS)


def render(value: str, maximum: int) -> str:
    """Text cleaned for the plan context: forbidden characters removed, markers
    defused (keeping their case), cut to ``maximum`` code points."""
    if not valid(value):
        return ""
    clean = _FORBIDDEN.sub("", value)
    clean = _MARKER.sub(r"(\1", clean)
    if len(clean) > maximum:
        clean = clean[: maximum - 1] + "\u2026"
    return clean


def length(value: str) -> int:
    """Code points, the unit every cap here is counted in."""
    return len(value)
