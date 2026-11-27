"""An approved call whose result could not be recorded never runs again (G-392).

The store fails once, while the resume records the results of the approved
calls it ran. The resume fails loudly. No later resume runs the call again:
before it runs, the harness records that the call was started, and a call
recorded as started that has no result gets the one outcome "Not run again:
the call was started, and its result was not recorded". Entered at the public
seam; the model is scripted.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import LlmToolCall, Session, record_approval
from prism_harness.thread import Thread

STARTED = "Not run again: the call was started, and its result was not recorded"


def results(session: Session, value: str) -> list[str]:
    return [
        e["result"]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result"
        for e in m.message["tool_results"]
        if e["args"].get("value") == value
    ]


@pytest.mark.parametrize("resumes", [1, 3])
def test_a_call_whose_result_write_failed_is_never_run_again(
    resumes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)
    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)

    native = Thread.record
    failed: list[bool] = []

    def record(self: Thread, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if not failed and any(
            m.get("type") == "tool_result"
            and any(e.get("result") == "echo:P" for e in m.get("tool_results", []))
            for m in messages
        ):
            failed.append(True)
            raise OSError("the store refused the write")
        return native(self, messages, *args, **kwargs)

    monkeypatch.setattr(Thread, "record", record)
    with pytest.raises(OSError):
        runtime.send(session, "")
    for _ in range(resumes):
        runtime.send(session, "")

    assert failed == [True]
    assert counts == {"echo": 1}
    assert results(session, "P") == [STARTED]
