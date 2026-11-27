"""The nightly property run's examples input (tools/property_examples.py):
a whole number from 1 up, or the job fails. Driven as the workflow drives
it, as a script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "property_examples.py"


def run(event: str, given: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), event, given],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@pytest.mark.parametrize(
    "given",
    [
        "",
        " ",
        "0",
        "-1",
        "-0",
        "1.5",
        "1e3",
        "+5",
        " 5",
        "5 ",
        "abc",
        "0x10",
        "١٢",
        "1000001",
        "9" * 40,
    ],
)
def test_a_dispatched_value_that_is_not_a_whole_number_from_one_fails(given: str) -> None:
    done = run("workflow_dispatch", given)
    assert done.returncode == 1
    assert done.stdout == ""
    assert "::error::examples must be a whole number" in done.stderr


@pytest.mark.parametrize(("given", "count"), [("1", 1), ("5000", 5000), ("1000000", 1_000_000)])
def test_a_dispatched_whole_number_is_taken_as_given(given: str, count: int) -> None:
    done = run("workflow_dispatch", given)
    assert done.returncode == 0
    assert done.stdout.strip() == str(count)


def test_a_scheduled_run_takes_the_default() -> None:
    done = run("schedule", "")
    assert done.returncode == 0
    assert done.stdout.strip() == "5000"


def test_a_missing_argument_fails() -> None:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "workflow_dispatch"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
