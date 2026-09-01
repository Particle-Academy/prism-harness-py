"""Mirrors prism-harness-ts/test/modes-and-budgets.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from prism_harness import (
    MAX_DEPTH,
    AgentMode,
    HarnessError,
    ModeRegistry,
    RunBudget,
    RunContext,
    RunLedger,
    Subagent,
    subagent_from_config,
)

CONFIG: dict[str, Any] = {
    "default": "chat",
    "modes": {
        "chat": {"system_prompt": "Be brief.", "tools": ["search"], "max_steps": 4},
        "plan": {
            "system_prompt": "Plan first.",
            "tools": ["search", "write"],
            "max_steps": 8,
            "requires_approval": ["write"],
            "subagents": {"researcher": {"mode": "chat", "max_steps": 3}},
        },
    },
}


# -- AgentMode ---------------------------------------------------------------


def test_approval_is_gated_per_mode_not_per_tool() -> None:
    # The same tool is not equally consequential everywhere: `execute_op`
    # against a scratch project is routine and against production is not, and
    # the tool cannot tell which it is in.
    mode = ModeRegistry(CONFIG).resolve("plan")

    assert mode.needs_approval("write") is True
    assert mode.needs_approval("search") is False
    assert ModeRegistry(CONFIG).resolve("chat").needs_approval("write") is False


def test_star_gates_every_tool_the_mode_offers() -> None:
    mode = AgentMode("m", "", ["a", "b"], [], 4, {}, ["*"])

    assert mode.needs_approval("a") is True
    assert mode.needs_approval("anything") is True


# -- ModeRegistry ------------------------------------------------------------


def test_falls_back_to_the_configured_default() -> None:
    assert ModeRegistry(CONFIG).resolve(None).name == "chat"
    assert ModeRegistry({}).default() == "chat"


def test_names_a_mode_that_is_not_configured() -> None:
    with pytest.raises(HarnessError, match="ghost"):
        ModeRegistry(CONFIG).resolve("ghost")


def test_refuses_a_malformed_max_steps() -> None:
    with pytest.raises(HarnessError, match="malformed"):
        ModeRegistry({"modes": {"bad": {"max_steps": 0}}}).resolve("bad")


def test_resolves_every_mode() -> None:
    # A mode nobody has entered yet keeps its misconfiguration until the day
    # somebody switches to it, and the first person to find out is a user.
    assert sorted(ModeRegistry(CONFIG).all()) == ["chat", "plan"]


def test_refuses_a_subagent_whose_mode_is_not_configured_when_the_parent_loads() -> None:
    # A typo surfaces when the parent mode is loaded, rather than halfway
    # through a run that has already spent budget.
    broken = {"modes": {"plan": {"subagents": {"helper": {"mode": "nope"}}}}}

    with pytest.raises(HarnessError, match="nope"):
        ModeRegistry(broken).resolve("plan")


def test_a_mode_gets_only_the_subagents_it_declares() -> None:
    # A subagent is authority, and authority a run inherits by being nested is
    # authority nobody granted.
    registry = ModeRegistry(CONFIG)

    assert list(registry.resolve("plan").subagents) == ["researcher"]
    assert list(registry.resolve("chat").subagents) == []


# -- Subagent ----------------------------------------------------------------


def test_the_child_gets_a_different_scope_which_avoids_the_deadlock() -> None:
    # A session's lock is taken on its address. A nested run asking for the
    # parent's address inside the parent's own lock would wait on a lock it
    # already holds; a separate address removes the contention rather than
    # making the lock reentrant.
    subagent = subagent_from_config("researcher", {"mode": "chat"})

    assert subagent.scope_under("support") == "support::sub::researcher"
    assert subagent.scope_under("support") != "support"


def test_an_explicit_scope_suffix_is_honoured() -> None:
    assert subagent_from_config("r", {"mode": "chat", "scope": "fixed"}).scope_under("s") == (
        "s::sub::fixed"
    )


def test_a_subagent_defaults_its_mode_to_its_own_name() -> None:
    subagent = subagent_from_config("researcher", {})

    assert subagent.mode == "researcher"
    assert "researcher" in subagent.description


# -- RunBudget ---------------------------------------------------------------


def test_budgets_nest_rather_than_reset() -> None:
    # A parent limited to 8 steps that may spawn subagents each entitled to a
    # fresh 8 has no bound at all -- it has a bound per node in a tree whose
    # width it also controls, which is unbounded spend wearing a limit's
    # clothing.
    parent = RunBudget(8, 1.0)
    ledger = RunLedger.start("root")
    ledger.record_steps(6)
    ledger.record_cost(0.75)

    child = RunBudget(8, 1.0).nested_within(parent, ledger)

    assert child.max_steps == 2
    assert child.max_cost_usd == pytest.approx(0.25)


def test_a_child_may_ask_for_less_than_remains() -> None:
    child = RunBudget(2).nested_within(RunBudget(8), RunLedger.start("root"))

    assert child.max_steps == 2


def test_a_child_may_never_ask_for_more_than_remains() -> None:
    ledger = RunLedger.start("root")
    ledger.record_steps(8)

    assert RunBudget(99).nested_within(RunBudget(8), ledger).max_steps == 0


def test_a_parent_cap_reaches_a_child_that_declared_none() -> None:
    child = RunBudget(4).nested_within(RunBudget(8, 2.0), RunLedger.start("r"))

    assert child.max_cost_usd == pytest.approx(2.0)


# -- RunLedger ---------------------------------------------------------------


def test_reports_why_the_tree_may_not_spend_again() -> None:
    # The states are genuinely different, and a caller that cannot tell them
    # apart writes one message for four causes.
    budget = RunBudget(2, 1.0, 60)
    ledger = RunLedger.start("root")

    assert ledger.exhaustion(budget) is None

    ledger.record_steps(2)
    assert "step budget exhausted" in (ledger.exhaustion(budget) or "")


def test_a_cancellation_is_reported_ahead_of_any_budget() -> None:
    ledger = RunLedger.start("root")
    ledger.record_steps(99)
    ledger.cancel("the user closed the tab")

    assert ledger.exhaustion(RunBudget(1)) == "the user closed the tab"


def test_a_cost_cap_that_cannot_be_enforced_fails_closed() -> None:
    # A provider that reports no cost would otherwise fold into `+= 0.0`,
    # leaving a cap that can never trip -- enforced in the documentation, absent
    # at runtime, and indistinguishable from a tree that spent nothing.
    ledger = RunLedger.start("root")
    ledger.record_cost(None)

    assert "cannot be enforced" in (ledger.exhaustion(RunBudget(10, 5.0)) or "")
    assert ledger.unmetered_runs == 1


def test_unmetered_runs_are_not_a_problem_without_a_cost_cap() -> None:
    ledger = RunLedger.start("root")
    ledger.record_cost(None)

    assert ledger.exhaustion(RunBudget(10)) is None


def test_reports_an_exhausted_cost_budget() -> None:
    ledger = RunLedger.start("root")
    ledger.record_cost(1.5)

    assert "cost budget exhausted" in (ledger.exhaustion(RunBudget(10, 1.0)) or "")


# -- RunContext --------------------------------------------------------------


def test_one_ledger_is_shared_down_the_whole_tree() -> None:
    # A child with its own ledger would let every node report itself inside
    # budget while the tree spent without limit.
    root = RunContext.root("run-1", RunBudget(8))
    child = root.for_child(Subagent("r", "", "chat", RunBudget(4)), "run-1")

    child.ledger.record_steps(3)

    assert root.ledger.steps == 3
    assert child.root_run_id == "run-1"
    assert child.is_child() is True
    assert root.is_child() is False


def test_the_tree_stops_at_the_depth_ceiling() -> None:
    context = RunContext.root("run-1", RunBudget(64))
    subagent = Subagent("r", "", "chat", RunBudget(64))

    for _ in range(MAX_DEPTH):
        assert context.too_deep() is False
        context = context.for_child(subagent, "run-1")

    assert context.too_deep() is True
