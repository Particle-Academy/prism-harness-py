"""Mirrors prism-harness-ts/test/skills-and-subagents.test.ts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    LlmRequest,
    LlmResponse,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    RunBudget,
    RunContext,
    Session,
    SkillRegistry,
    SubagentRunner,
    ToolRegistry,
)


def a_skill_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="prism-harness-skills-"))
    (root / "research" / "notes").mkdir(parents=True)
    (root / "research" / "SKILL.md").write_text("Search carefully.", encoding="utf-8")
    (root / "research" / "notes" / "deep.md").write_text("Deep note.", encoding="utf-8")
    (root / "secret.txt").write_text("not a skill", encoding="utf-8")
    return root


# -- SkillRegistry -----------------------------------------------------------


def test_appends_each_named_skill_as_a_tagged_section() -> None:
    prompt = SkillRegistry(a_skill_root()).augment_prompt("Be brief.", ["research"])

    assert "Be brief." in prompt
    assert '<skill name="research">' in prompt
    assert "Search carefully." in prompt


def test_leaves_the_prompt_unchanged_when_no_skills_are_named() -> None:
    # Appending an empty preamble would tell the model skills are available when
    # none are.
    assert SkillRegistry(a_skill_root()).augment_prompt("Be brief.", []) == "Be brief."


def test_reads_a_nested_file_inside_a_skill() -> None:
    assert SkillRegistry(a_skill_root()).read("research", "notes/deep.md") == "Deep note."


def test_refuses_a_traversing_path() -> None:
    skills = SkillRegistry(a_skill_root())

    with pytest.raises(HarnessError, match="stay inside"):
        skills.read("research", "../secret.txt")

    with pytest.raises(HarnessError, match="stay inside"):
        skills.read("research", "notes/../../secret.txt")


def test_refuses_an_absolute_path() -> None:
    with pytest.raises(HarnessError, match="stay inside"):
        SkillRegistry(a_skill_root()).read("research", "/etc/passwd")


def test_refuses_a_traversing_name_before_it_is_joined_to_anything() -> None:
    skills = SkillRegistry(a_skill_root())

    with pytest.raises(HarnessError, match="not a valid name"):
        skills.read("../", "SKILL.md")

    with pytest.raises(HarnessError, match="not a valid name"):
        skills.read("Research", "SKILL.md")


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs privilege on Windows")
def test_refuses_a_symlink_that_points_out_of_the_skill() -> None:
    # The check the lexical ones cannot make: `notes/link.md` is lexically
    # innocent and resolves elsewhere.
    root = a_skill_root()
    (root / "research" / "notes" / "link.md").symlink_to(root / "secret.txt")

    with pytest.raises(HarnessError, match="outside the skill"):
        SkillRegistry(root).read("research", "notes/link.md")


# -- subagents ---------------------------------------------------------------

MODES = ModeRegistry(
    {
        "default": "parent",
        "modes": {
            "parent": {
                "system_prompt": "You delegate.",
                "tools": [],
                "max_steps": 8,
                "subagents": {"researcher": {"mode": "child", "max_steps": 2}},
            },
            "child": {"system_prompt": "You research.", "tools": [], "max_steps": 2},
        },
    }
)


def harnessed() -> tuple[PrismHarness, Session]:
    directory = tempfile.mkdtemp(prefix="prism-harness-sub-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    parent = harness.for_(Participant("User", 1)).session("support")
    parent.using_mode("parent").using_provider("anthropic").using_model("m")
    return harness, parent


def runtime_returning(text: str) -> AgentRuntime:
    return AgentRuntime(
        client=lambda _request: LlmResponse(text=text, finish_reason="stop"),
        modes=MODES,
        tools=ToolRegistry(),
    )


def a_subagent() -> object:
    return MODES.resolve("parent").subagents["researcher"]


def test_runs_the_child_and_hands_its_text_back_as_the_tool_result() -> None:
    harness, parent = harnessed()
    runner = SubagentRunner(harness, runtime_returning("The child answered."))
    context = RunContext.root("root", RunBudget(8))

    tool = runner.tool(a_subagent(), parent, context, "root")  # type: ignore[arg-type]

    assert tool.handle({"task": "Find something"}) == "The child answered."


def test_gives_the_child_its_own_session_under_a_different_scope() -> None:
    # The parent is mid-run holding its own lock. A child asking for the
    # parent's address would wait on a lock its own caller holds.
    harness, parent = harnessed()
    runner = SubagentRunner(harness, runtime_returning("done"))
    subagent = MODES.resolve("parent").subagents["researcher"]

    runner.run(subagent, parent, RunContext.root("root", RunBudget(8)), "root", "go")

    child = harness.for_(parent.participant).session(subagent.scope_under(parent.scope))

    assert child.key() != parent.key()
    assert child.thread().count() > 0
    # The child's authority is its own mode, not the parent's.
    assert child.mode() == "child"


def test_refuses_before_spawning_when_the_tree_is_already_exhausted() -> None:
    # An exhausted tree must not pay for a session, a thread write and a
    # provider call to discover it is exhausted.
    harness, parent = harnessed()
    called = {"n": 0}

    def client(_request: LlmRequest) -> LlmResponse:
        called["n"] += 1
        return LlmResponse(text="x", finish_reason="stop")

    context = RunContext.root("root", RunBudget(2))
    context.ledger.record_steps(2)

    result = SubagentRunner(
        harness, AgentRuntime(client=client, modes=MODES, tools=ToolRegistry())
    ).run(MODES.resolve("parent").subagents["researcher"], parent, context, "root", "go")

    assert called["n"] == 0
    assert result.outcome == "exhausted"
    assert "did not run" in result.to_tool_result()


def test_reports_a_cancelled_tree_as_cancelled() -> None:
    harness, parent = harnessed()
    context = RunContext.root("root", RunBudget(8))
    context.ledger.cancel("stopped by the user")

    result = SubagentRunner(harness, runtime_returning("x")).run(
        MODES.resolve("parent").subagents["researcher"], parent, context, "root", "go"
    )

    assert result.outcome == "cancelled"


def test_frames_a_child_failure_rather_than_tearing_down_the_parent_run() -> None:
    # The parent is a legitimate audience for "that did not work", and raising
    # would discard work the parent has already done.
    harness, parent = harnessed()

    def client(_request: LlmRequest) -> LlmResponse:
        raise RuntimeError("the child provider is down")

    result = SubagentRunner(
        harness, AgentRuntime(client=client, modes=MODES, tools=ToolRegistry())
    ).run(
        MODES.resolve("parent").subagents["researcher"],
        parent,
        RunContext.root("root", RunBudget(8)),
        "root",
        "go",
    )

    assert result.outcome == "failed"
    assert "provider is down" in result.to_tool_result()


def test_draws_the_child_budget_from_what_the_tree_has_left() -> None:
    harness, parent = harnessed()
    steps = {"n": 0}

    def client(_request: LlmRequest) -> LlmResponse:
        steps["n"] += 1
        return LlmResponse(text="again", finish_reason="tool_calls", tool_calls=[])

    context = RunContext.root("root", RunBudget(8))
    context.ledger.record_steps(7)

    SubagentRunner(harness, AgentRuntime(client=client, modes=MODES, tools=ToolRegistry())).run(
        MODES.resolve("parent").subagents["researcher"], parent, context, "root", "go"
    )

    # One step left in the tree, so the child takes one -- not the two its own
    # declaration asks for.
    assert steps["n"] == 1
    assert context.ledger.steps == 8
