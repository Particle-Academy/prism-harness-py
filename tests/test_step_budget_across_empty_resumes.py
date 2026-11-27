"""The step budget holds across any number of empty resumes (G-392).

A resume continues the paused run's step count. A run that ends
``budget_exhausted`` after resolving its approvals stays spent: further empty
sends take no model step, so neither the steps nor the model calls ever
exceed ``max_steps``. Entered at the public seam; the model is scripted and
asks for a gated call on every step.
"""

from __future__ import annotations

from test_runtime import a_session, guarded_runtime

from prism_harness import LlmRequest, LlmResponse, LlmToolCall, record_approval


def test_empty_sends_after_budget_exhausted_never_take_another_model_step() -> None:
    session = a_session("guarded")  # max_steps 4
    counts: dict[str, int] = {}
    calls = {"model": 0}

    def client(_request: LlmRequest) -> LlmResponse:
        calls["model"] += 1
        n = calls["model"]
        return LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(f"c{n}", "echo", {"value": str(n)})],
        )

    runtime = guarded_runtime(client, counts)
    response = runtime.send(session, "go")
    steps = [response.steps]
    for _ in range(8):
        for pending in response.pending_approvals:
            record_approval(session, pending.id, True)
        response = runtime.send(session, "")
        steps.append(response.steps)

    assert max(steps) <= 4
    assert calls["model"] <= 4
    # One approved run per model step that asked, and never more.
    assert counts.get("echo", 0) <= calls["model"]
