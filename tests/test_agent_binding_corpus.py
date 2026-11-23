"""The cross-language agent-binding corpus from ``prism-parity`` (G-321 E).

Which provider and model a run and its subagent child actually request, and which
delegations are refused before any request. Each case drives the real runtime;
the expectation was written before any language recorded it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from binding_world import World

from prism_harness import HarnessError

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures" / "harness-agent-binding.json").read_text(encoding="utf-8")
)


def record(case: dict[str, Any]) -> dict[str, Any]:
    run = case["run"]
    world = World(global_config=case["global"], modes=case["modes"], agents=case["agents"])
    world.delegate = run["delegate"]
    if run["delegate"] is not None:
        world.offer_subagents(run["agent"], run["mode"])
    session = world.session(
        run["agent"],
        run["mode"],
        provider=run["session_provider"],
        model=run["session_model"],
    )
    try:
        world.send(session)
    except (HarnessError, ValueError):
        if world.requests:
            raise
        return {"refused": "config", "parent": None, "child": None}

    def first(requests: list[Any]) -> dict[str, str] | None:
        return (
            None if not requests else {"provider": requests[0].provider, "model": requests[0].model}
        )

    return {
        "refused": None,
        "parent": first(world.parent_requests()),
        "child": first(world.child_requests()),
    }


@pytest.mark.parametrize("case", CORPUS["cases"], ids=[c["id"] for c in CORPUS["cases"]])
def test_agent_binding_corpus(case: dict[str, Any]) -> None:
    assert record(case) == case["expected"], case["title"]
