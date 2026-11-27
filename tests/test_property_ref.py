"""The nightly property run's ref input (tools/property_ref.py): the default
branch or a release line, or the job fails. Driven as the workflow drives
it, as a script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "property_ref.py"


def run(event: str, given: str, default: str = "main") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), event, given, default],
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
        "main ",
        " main",
        "Main",
        "develop",
        "feature/x",
        "refs/heads/main",
        "refs/heads/release/0.5.x",
        "v0.5.3",
        "cd38a1f2ef4776dfd0c47c1e7a4caccb24280882",
        "cd38a1f",
        "release",
        "release/",
        "release//0.5.x",
        "release/0.5.x/",
        "release/../main",
        "release/0..5",
        "release/-x",
        "release/.x",
        "release/0.5.x.lock",
        "release/0.5.x.",
        "release/0.5 x",
        "release/0.5.x;rm",
        "release/$(id)",
        "release/0.5.x\n",
        "release/０.5",
        "pull/1/head",
        "someone:release/0.5.x",
    ],
)
def test_a_dispatched_ref_that_is_not_the_default_branch_or_a_release_line_fails(
    given: str,
) -> None:
    done = run("workflow_dispatch", given)
    assert done.returncode == 1
    assert done.stdout == ""
    assert "::error::ref must be the default branch" in done.stderr


@pytest.mark.parametrize("given", ["main", "release/0.5.x", "release/0.6.x", "release/1.0"])
def test_the_default_branch_or_a_release_line_is_taken_as_given(given: str) -> None:
    done = run("workflow_dispatch", given)
    assert done.returncode == 0
    assert done.stdout.strip() == given


def test_a_scheduled_run_takes_the_default_branch_whatever_was_given() -> None:
    done = run("schedule", "")
    assert done.returncode == 0
    assert done.stdout.strip() == "main"


@pytest.mark.parametrize("default", ["", "main/../x", "-main", "ma in"])
def test_a_default_branch_that_is_not_a_plain_name_fails(default: str) -> None:
    done = run("schedule", "", default)
    assert done.returncode == 1
    assert done.stdout == ""


def test_the_wrong_number_of_arguments_is_a_usage_error() -> None:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "workflow_dispatch", "main"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert done.returncode == 2
