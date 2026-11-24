"""G-360: a plan is a person's agreement. No host policy may let a plan tool
run without asking, however the name is spelled."""

from __future__ import annotations

import pytest

from prism_harness import HarnessError, ToolPolicy, is_person_only_tool
from prism_harness.tool_policy import RESERVED_TOOLS

SPELLINGS = {
    "plan_draft": "plan_draft",
    "plan_propose_change": "plan_propose_change",
    "plan_pause_for_person": "plan_pause_for_person",
    "plan_complete_step": "plan_complete_step",
    "upper case": "PLAN_DRAFT",
    "mixed case": "Plan_Complete_Step",
    "trailing space": "plan_draft ",
    "trailing NBSP": "plan_draft\u00a0",
    "leading ZWSP": "\u200bplan_draft",
    "Cyrillic a": "pl\u0430n_draft",
}


@pytest.mark.parametrize("name", SPELLINGS.values(), ids=SPELLINGS.keys())
def test_a_policy_naming_a_plan_tool_is_refused_in_any_spelling(name: str) -> None:
    with pytest.raises(HarnessError) as caught:
        ToolPolicy.from_json({"revision": "r1", "allow": [{"tool": name}], "honor": True})
    assert caught.value.code == "tool_policy_refused"


def test_no_policy_lets_a_plan_tool_run() -> None:
    policy = ToolPolicy.from_json({"revision": "r1", "allow": [{"tool": "search"}], "honor": True})
    assert RESERVED_TOOLS == (
        "plan_draft",
        "plan_propose_change",
        "plan_pause_for_person",
        "plan_complete_step",
    )
    for name in RESERVED_TOOLS:
        assert policy.allows({"name": name}) is False
        assert is_person_only_tool(name) is True
        assert is_person_only_tool(name.upper()) is True


def test_a_name_that_only_resembles_a_plan_tool_is_not_reserved() -> None:
    assert is_person_only_tool("plan_drafts") is False
    ToolPolicy.from_json({"revision": "r1", "allow": [{"tool": "plan_drafts"}], "honor": True})
