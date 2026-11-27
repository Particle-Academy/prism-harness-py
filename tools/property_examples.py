"""How many examples the nightly property run takes (G-392 QC).

Run by .github/workflows/property-nightly.yml before the property tests:

    python tools/property_examples.py <event name> <examples input>

A scheduled run has no input and takes the default. A dispatched run takes
the value it was given, which must be a whole number from 1 to
``MAXIMUM``, written in ASCII digits: anything else (empty, 0, negative,
fractional, non-numeric, too large) FAILS the job. A value that was given is
never replaced by the default. Prints the count on success.
"""

from __future__ import annotations

import sys

DEFAULT = 5000
MAXIMUM = 1_000_000


def examples(event: str, given: str) -> int:
    """The example count, or ValueError naming why the input is refused."""
    if event == "schedule":
        return DEFAULT
    if given == "" or not given.isascii() or not given.isdigit():
        raise ValueError(f"examples must be a whole number from 1 to {MAXIMUM}, got {given!r}")
    count = int(given)
    if count < 1 or count > MAXIMUM:
        raise ValueError(f"examples must be a whole number from 1 to {MAXIMUM}, got {given!r}")
    return count


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: property_examples.py <event name> <examples input>", file=sys.stderr)
        return 2
    try:
        print(examples(argv[1], argv[2]))
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
