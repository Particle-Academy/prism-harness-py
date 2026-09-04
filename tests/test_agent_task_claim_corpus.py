"""The cross-language agent-task-claim corpus from ``prism-parity``.

A task list is DURABLE state that outlives the process which wrote it, and
nothing says the process reading it back is in the same language. A PHP worker
and a Python worker drawing from one list have to agree on when a lease has
lapsed, on who may release, and on the bytes of the stored record -- because the
list itself is the shared surface.

A one-tick disagreement about expiry hands one task to two workers. A
disagreement about how ``claimed_by`` is compared lets one worker close
another's work. NEITHER ERRORS, and neither is visible to a per-language suite,
because each one asserts against the value its own code produced.

Twenty of the twenty-one rows agree with the PHP reference. ``atc-0017`` does
not, and it is recorded as a DIVERGENCE rather than skipped, because a skip
hides the finding: **this port accepts a fractional lease and truncates it**,
where the row exists to pin that a fractional lease is refused.

That row is the one worth reading twice. PHP cannot express it at all --
``claim()`` declares ``?int``, so the type system rejects 90.4 before any guard
in the package runs -- and is skipped for it under decision 0002.
``prism-harness-ts`` has no such constraint, reaches its own guard, and REFUSES
the value as ``task_lease_invalid``. This port has no such constraint either,
reaches ``_require_lease``, which only asks for finite-and-positive, and grants
a 90-second lease for a 90.4-second request. So the two languages that can
express the row disagree, and this is the one that accepts it -- a
configuration silently becoming a different configuration, which is the exact
rule ``_require_lease``'s own docstring says it holds. Recorded here rather
than fixed, because closing it changes shipped behaviour and belongs in a
decision across the three repositories.

Drives the rows the way ``prism-parity:tools/generate-agent-task-claim.php``
drives them, and mirrors
``prism-harness-ts/test/agent-task-claim-corpus.test.ts`` case for case --
including handing the release outcome over UNVALIDATED, so the package's own
guard is what gets measured rather than the runner's.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest

from prism_harness.errors import HarnessError
from prism_harness.stores.base import Durability
from prism_harness.tasks import StoreTaskSource, TaskOutcome

T = TypeVar("T")

FIXTURE = Path(__file__).parent / "fixtures" / "agent-task-claim.json"
CORPUS: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = CORPUS["cases"]

#: The list this suite seeds. The name is arbitrary; that the SOURCE and the
#: seeding agree on it is not -- see :func:`_seeded`.
KEY = "corpus"

AGREEING = [case for case in CASES if case["result"]["py"] == case["result"]["php"]]
DIVERGING = [case for case in CASES if case["result"]["py"] != case["result"]["php"]]


class SeedingFailed(RuntimeError):
    """A row seeded tasks and the source cannot see any of them.

    A broken runner, not a finding -- and the difference is INVISIBLE in the
    recorded output, which is what makes it worth its own exception. See
    :func:`_seeded`.
    """


class ArrayStore:
    """The smallest store that satisfies the contract.

    Deliberately NOT one of the shipped drivers. The file store would measure
    the filesystem too, and the memory store reports itself VOLATILE, which
    :class:`StoreTaskSource` refuses at construction -- correctly, since a task
    list that vanishes on a deploy is indistinguishable from a finished one.

    The lock is a no-op because this process is single-threaded. The locking
    PRIMITIVE has its own failure modes and is explicitly outside this suite's
    scope; measuring it here would conflate two things.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        self.rows[key] = payload

    def forget(self, key: str) -> None:
        self.rows.pop(key, None)

    def with_lock(
        self,
        key: str,
        callback: Callable[[], T],
        ttl_seconds: float = 10,
        wait_seconds: float = 5,
    ) -> T:
        return callback()

    def durability(self) -> Durability:
        return Durability.DURABLE


def _id(case: dict[str, Any]) -> str:
    return str(case["id"])


def _case(case_id: str) -> dict[str, Any]:
    return next(case for case in CASES if case["id"] == case_id)


def _canonical(value: object) -> str:
    """Canonical JSON per decision 0005, and every argument is load-bearing.

    ``separators`` because the default pads every one with a space that PHP and
    JavaScript do not emit; ``ensure_ascii=False`` because the default escapes
    non-ASCII to ``\\uXXXX`` where the other two pass it through; no
    ``sort_keys`` because key order is INSERTION order and is part of the
    contract -- sorting here would hide a mapper that started emitting its keys
    in a different sequence.

    Comparison is over these strings rather than over the dicts, so a float that
    compares equal to an integer still fails: ``1735689900.0 == 1735689900`` is
    True and ``"1735689900.0" == "1735689900"`` is not.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _seeded(case: dict[str, Any], seed_at: str | None = None) -> StoreTaskSource:
    """A source holding this row's ``given``, with time FROZEN at ``given.now``.

    Seeded DIRECTLY rather than through ``add()``, so a row may describe a state
    the public API would refuse to build -- a duplicate id, say, or a claim with
    no holder. A corpus that can only express states the implementation is
    willing to create cannot test what it does when it meets one it did not.

    The clock is injected rather than patched. ``claimed_until`` is a Unix
    timestamp another process on another machine compares against, so every
    expiry boundary in this suite is exact rather than raced.
    """
    now = float(case["given"]["now"])
    store = ArrayStore()
    source = StoreTaskSource(store=store, key=KEY, clock=lambda: now)

    given: list[dict[str, Any]] = case["given"]["tasks"]

    # Seeded at the source's OWN key. The reference generator first wrote to the
    # bare list name instead, so every row read an EMPTY list and it recorded
    # "nothing happened" for all 21 as though that were an answer -- outcome ok,
    # pending 0, no record. It looked entirely plausible on the page.
    store.put(
        seed_at if seed_at is not None else KEY,
        {
            "tasks": [
                {
                    "claimed_by": task.get("claimed_by"),
                    "claimed_until": task.get("claimed_until"),
                    "id": task["id"],
                    "instruction": task["instruction"],
                    "state": task.get(
                        "state", "claimed" if task.get("claimed_by") is not None else "todo"
                    ),
                }
                for task in given
            ]
        },
    )

    # The guard that would have caught the above immediately, and the reason it
    # is here rather than in a comment: a generator that silently measures
    # nothing is the defect this repository exists to catch, one level up.
    if given and source.find(given[0]["id"]) is None:
        raise SeedingFailed(
            f"SEEDING FAILED for {case['id']}: the source cannot see the tasks it was given."
        )

    return source


def run_case(case: dict[str, Any], seed_at: str | None = None) -> dict[str, Any]:
    """Seed this row's ``given``, perform its ``when``, and record what came back.

    OUTCOME and CODE rather than prose, per decision 0004: error messages are
    explicitly outside the contract, so a row that compared them would fail on
    wording and teach nothing.
    """
    source = _seeded(case, seed_at)
    when = case["when"]
    operation = when["op"]

    try:
        record = None

        if operation == "claim":
            record = source.claim(when["worker"], when.get("lease_seconds"))
        elif operation == "find":
            record = source.find(when["task_id"])
        elif operation == "pending":
            record = None
        elif operation == "release":
            task = source.find(when["task_id"])

            if task is not None:
                # HANDED OVER UNVALIDATED, on purpose. The reference converts
                # the string through `TaskOutcome::from` at the call site
                # because PHP's enum makes that the only way in; doing the same
                # here would supply the guard from the RUNNER and record
                # agreement the package has not earned. So the raw value goes
                # through, which is also the path a real untyped caller takes --
                # a decoded JSON body, a config file -- and `release` is left to
                # refuse it or not.
                source.release(task, when["worker"], when["outcome"])
                record = source.find(when["task_id"])
        elif operation == "claim_then_find":
            source.claim(when["worker"], when["lease_seconds"])
            record = source.find(when["task_id"])
        else:
            raise RuntimeError(f"Unknown op {operation}")

        return {
            "outcome": "ok",
            "code": None,
            "record": record.to_dict() if record is not None else None,
            "pending": source.pending(),
        }
    except HarnessError as refused:
        return {
            "outcome": "refused",
            "code": refused.code,
            "record": None,
            "pending": None,
        }


# -- the corpus is whole ----------------------------------------------------


def test_the_corpus_is_whole_not_a_subset_someone_trimmed_to_green() -> None:
    assert len(CASES) == 21


def test_every_row_has_a_recorded_python_half() -> None:
    # A half-recorded corpus passes every comparison below it by comparing
    # nothing, which is the same failure as a trimmed one wearing a full count.
    assert [case["id"] for case in CASES if case["result"]["py"] is None] == []


# -- the rows ---------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_produces_this_languages_recorded_result(case: dict[str, Any]) -> None:
    assert _canonical(run_case(case)) == _canonical(case["result"]["py"])


@pytest.mark.parametrize("case", AGREEING, ids=_id)
def test_agrees_with_the_php_reference(case: dict[str, Any]) -> None:
    assert _canonical(run_case(case)) == _canonical(case["result"]["php"])


def test_diverges_on_exactly_the_one_row_the_manifest_names() -> None:
    assert [case["id"] for case in DIVERGING] == ["atc-0017"]


def test_the_fractional_lease_is_truncated_here_and_unexpressible_in_php() -> None:
    """The finding, asserted as behaviour rather than left as a recorded string.

    PHP is skipped for this row: ``claim()`` declares ``?int``, so 90.4 is
    rejected by the type system before any guard in the package runs. Python has
    no such constraint, so the row IS expressible here -- and what it shows is
    that the lease is accepted and then silently truncated by ``_timestamp``.

    A configuration that silently became a different configuration, which is the
    exact rule ``_require_lease`` states it holds. The direction matters too: 90
    seconds where 90.4 was asked for expires EARLY, so the next caller may take
    the task while the first worker believes it still holds it.
    """
    case = _case("atc-0017")

    assert "skipped" in case["result"]["php"]

    produced = run_case(case)
    record = produced["record"]

    assert produced["outcome"] == "ok"
    assert record is not None

    granted = record["claimed_until"] - case["given"]["now"]

    assert granted == 90
    assert granted != case["when"]["lease_seconds"]


def test_the_fractional_lease_is_a_three_way_split_and_stays_visible_as_one() -> None:
    # Asserted from this side as well as from the TypeScript runner's, so a
    # re-vendor that quietly changed any of the three answers goes red rather
    # than passing. The three-way split IS the finding: one row, three
    # languages, three behaviours, on the value that decides when another
    # worker may take a task.
    result = _case("atc-0017")["result"]

    assert "skipped" in result["php"]
    assert result["ts"] == {
        "outcome": "refused",
        "code": "task_lease_invalid",
        "record": None,
        "pending": None,
    }
    assert result["py"]["outcome"] == "ok"


def test_agrees_with_typescript_everywhere_except_the_three_recorded_rows() -> None:
    # atc-0011 and atc-0012 are G-39, where this port holds the line the
    # reference holds and TypeScript does not. atc-0017 is G-40, the other way
    # round. Naming the set is what makes a fourth disagreement appear as a
    # failure rather than as a number nobody was watching.
    differ = [case["id"] for case in CASES if case["result"]["py"] != case["result"]["ts"]]

    assert differ == ["atc-0011", "atc-0012", "atc-0017"]


# -- the bytes --------------------------------------------------------------


def test_claimed_until_is_an_integer_and_not_a_float_that_compares_equal() -> None:
    # `type(...) is int` and not `isinstance`, deliberately, twice over. A float
    # passes every equality assertion in this file's dicts -- 1735689900.0 ==
    # 1735689900 is True -- and `isinstance(True, int)` is also True, so a bool
    # would sail through an isinstance check as well.
    checked = 0

    for case in CASES:
        record = run_case(case)["record"]

        if record is None or record["claimed_until"] is None:
            continue

        assert type(record["claimed_until"]) is int, case["id"]
        checked += 1

    # Vacuity guard. A loop that examined nothing passes silently, and this one
    # would if `record` ever stopped carrying the field.
    assert checked == 6


def test_pending_is_an_integer_count() -> None:
    checked = 0

    for case in CASES:
        pending = run_case(case)["pending"]

        if pending is None:
            continue

        assert type(pending) is int, case["id"]
        checked += 1

    assert checked == 13


def test_the_record_carries_the_references_keys_in_the_references_order() -> None:
    # Key order is insertion order and is part of the contract (0005). It does
    # not change what the JSON means, but it is the cheapest way to notice a
    # mapper being rewritten, and `claimed_by`/`claimed_until` being PRESENT AND
    # NULL rather than absent is an observable decision (0002).
    checked = 0

    for case in CASES:
        reference = case["result"]["php"]

        if "record" not in reference or reference["record"] is None:
            continue

        produced = run_case(case)["record"]

        assert produced is not None, case["id"]
        assert list(produced) == list(reference["record"]), case["id"]
        checked += 1

    assert checked == 7


# -- the guard on the runner itself -----------------------------------------


def test_the_seeding_guard_fires_when_the_tasks_land_under_the_wrong_key() -> None:
    # The claim above -- "the source can see what it was given" -- is only worth
    # anything if something fails when it is false. This is that something: seed
    # the same row one key to the side and the guard must raise rather than let
    # the row report an empty list as an answer.
    with pytest.raises(SeedingFailed):
        run_case(_case("atc-0001"), seed_at=f"{KEY}-but-not-the-sources-key")


def test_every_seeded_row_is_visible_to_the_source_in_full() -> None:
    # Stronger than the guard inside `_seeded`, which only checks the FIRST
    # task. A seeding that dropped every row but one would pass that and quietly
    # change what `pending()` means.
    for case in CASES:
        source = _seeded(case)

        assert len(source.records()) == len(case["given"]["tasks"]), case["id"]


# -- what the rows are for, said directly -----------------------------------


def test_the_outcome_guard_is_the_packages_and_not_the_runners() -> None:
    """atc-0012's refusal is earned by ``release``, with no help from above.

    ``prism-harness-ts`` records this row as a divergence for the opposite
    reason: there ``TaskOutcome`` is a compile-time union with no runtime
    existence, ``release`` never looks at the outcome, and the padded string
    lands in the durable list as a fifth state. So this runner hands the value
    over unvalidated, and this test says where the guard has to be --
    ``_require_identifier`` first, then ``TaskOutcome.parse``, both inside the
    package.

    ``outcome`` is annotated ``Any`` deliberately: that is exactly the caller
    with no type checker in the way -- a decoded JSON body, a config file --
    and the guard exists for them.
    """
    case = _case("atc-0012")
    source = _seeded(case)
    task = source.find(case["when"]["task_id"])
    outcome: Any = case["when"]["outcome"]

    assert task is not None
    # A real U+00A0, written as an escape so it is VISIBLE here. PHP's trim
    # leaves it where Python's strip takes it, so an implementation that
    # trimmed would read this as `done` in one language and refuse it in
    # another. G-36's lesson, one field over.
    assert outcome == "\u00a0done"
    # The positive control. The unpadded word IS an outcome, so the refusal
    # below is about the padding rather than about `release` refusing
    # everything -- which a suite of refusals cannot otherwise tell apart.
    assert TaskOutcome.parse(outcome.strip()) is TaskOutcome.DONE

    with pytest.raises(HarnessError) as refused:
        source.release(task, case["when"]["worker"], outcome)

    assert refused.value.code == "task_outcome_invalid"


def test_a_duplicate_id_already_in_the_list_resolves_to_the_first_of_them() -> None:
    # The finding atc-0015 records: `duplicate_task_id` guards `add()`, the door
    # this package controls, and a list that ALREADY holds two tasks with one id
    # is read back and counted rather than refused. That matters because the
    # list is shared surface -- a less careful port, or a hand-edited row,
    # produces exactly this list.
    source = _seeded(_case("atc-0015"))
    found = source.find("t-1")

    assert found is not None
    assert found.instruction == "first"
    assert source.pending() == 2


def test_a_lapsed_lease_returns_the_task_to_todo_and_never_to_failed() -> None:
    # atc-0005, stated as the security property rather than as a recorded
    # string. A worker dying is not the task failing: marking it failed burns a
    # retry that never ran and tells the application something untrue.
    source = _seeded(_case("atc-0005"))
    found = source.find("t-1")

    assert found is not None
    assert found.state.value == "todo"
    assert found.claimed_by is None


def test_expiry_is_inclusive_so_a_lease_is_over_at_its_expiry() -> None:
    # atc-0006. A one-tick boundary difference between ports is a real
    # divergence and nothing errors on it -- the task is simply handed to two
    # workers, once, at the edge.
    case = _case("atc-0006")
    source = _seeded(case)
    found = source.find("t-1")

    assert found is not None
    assert case["given"]["tasks"][0]["claimed_until"] == case["given"]["now"]
    assert found.state.value == "todo"


# -- recording -------------------------------------------------------------


def _record(path: Path) -> int:
    """Write this language's half of the corpus into ``path``.

    Read fresh and written back whole, because the other two languages record
    into the same file: loading a stale copy would silently drop a half somebody
    else had just recorded.

    ``ensure_ascii=False`` is not cosmetic here -- ``atc-0012``'s outcome is
    padded with a real U+00A0, and the default would rewrite it as an escape and
    change the bytes of a row whose entire point is that the padding survives.
    """
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    changed = 0

    for case in document["cases"]:
        produced = run_case(case)

        if case["result"].get("py") != produced:
            changed += 1

        case["result"]["py"] = produced

    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return changed


if __name__ == "__main__":
    # python tests/test_agent_task_claim_corpus.py ../prism-parity/suites/agent-task-claim/cases.json
    #
    # Then copy that file over tests/fixtures/agent-task-claim.json, which is
    # how every vendored corpus in this ecosystem travels: the fixture ships
    # inside the package so the tests run from one checkout.
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tests/test_agent_task_claim_corpus.py <path-to-cases.json>")

    count = _record(Path(sys.argv[1]))
    print(f"agent-task-claim: recorded the Python half, {count} row(s) changed.", file=sys.stderr)
