"""Which branch the nightly property run checks out (G-387 round 16).

Run by .github/workflows/property-nightly.yml before it checks anything out
for the tests:

    python tools/property_ref.py <event name> <ref input> <default branch>

A scheduled run has no input and takes the default branch. A dispatched run
takes the ref it was given, which must be EXACTLY the default branch or a
release line, ``release/<name>``: ASCII letters, digits, ``.``, ``_`` and
``-`` in each part, no empty part, no ``..``, no leading ``-``. Anything else
(empty, a SHA, a tag, ``refs/...``, another branch, a fork's ref, a path
trick) FAILS the job; a value that was given is never replaced by the
default. Prints the branch on success.
"""

from __future__ import annotations

import re
import sys

#: One part of a branch name: safe ASCII only, never starting with ``-`` or
#: ``.``, never holding ``..``.
_PART = r"[A-Za-z0-9_][A-Za-z0-9._-]*"
_RELEASE = re.compile(rf"release/{_PART}(?:/{_PART})*")


def ref(event: str, given: str, default: str) -> str:
    """The branch to check out, or ValueError naming why the input is refused."""
    if not re.fullmatch(_PART, default) or ".." in default:
        raise ValueError(f"the default branch {default!r} is not a plain branch name")
    if event == "schedule":
        return default
    if given == default:
        return given
    if _RELEASE.fullmatch(given) and ".." not in given and not given.endswith((".", ".lock")):
        return given
    raise ValueError(
        f"ref must be the default branch ({default}) or a release line (release/<name>), "
        f"got {given!r}"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: property_ref.py <event name> <ref input> <default branch>", file=sys.stderr)
        return 2
    try:
        print(ref(argv[1], argv[2], argv[3]))
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
