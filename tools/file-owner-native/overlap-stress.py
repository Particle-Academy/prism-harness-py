"""Hosted POSIX process exclusion stress; no sleeps stand in for acquisition."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

ITERATIONS = 200


def intervals(
    events: list[dict[str, Any]], iterations: int, role_pids: dict[str, int]
) -> list[dict[str, Any]]:
    assert type(iterations) is int and iterations > 0, "invalid iteration count"
    assert set(role_pids) == {"holder", "free"}, "invalid roles"
    assert len(set(role_pids.values())) == 2, "workers must have distinct PIDs"
    assert all(type(pid) is int and pid > 0 for pid in role_pids.values()), "invalid worker PID"
    expected = {
        (role, iteration, event)
        for role in role_pids
        for iteration in range(iterations)
        for event in ("enter", "exit")
    }
    seen = set()
    opened: dict[tuple[str, int], dict[str, Any]] = {}
    completed = []
    for event in events:
        assert set(event) == {"role", "iteration", "event", "pid", "at_ns"}, "invalid event fields"
        assert event["role"] in role_pids, "invalid event role"
        assert event["event"] in ("enter", "exit"), "invalid event kind"
        assert type(event["iteration"]) is int, "invalid event iteration"
        assert type(event["pid"]) is int and event["pid"] == role_pids[event["role"]], (
            "wrong role PID"
        )
        assert type(event["at_ns"]) is int and event["at_ns"] >= 0, "invalid monotonic timestamp"
        identity = event["role"], event["iteration"], event["event"]
        assert identity in expected and identity not in seen, "duplicate or unexpected event"
        seen.add(identity)
        key = event["role"], event["iteration"]
        if event["event"] == "enter":
            assert key not in opened, "duplicate interval entry"
            opened[key] = event
        else:
            assert key in opened, "exit without entry"
            start = opened.pop(key)
            assert event["pid"] == start["pid"] and event["at_ns"] >= start["at_ns"]
            completed.append({**start, "exit_ns": event["at_ns"]})
    assert not opened and seen == expected, "incomplete role/iteration coverage"
    ordered = sorted(completed, key=lambda row: row["at_ns"])
    for before, after in pairwise(ordered):
        assert after["at_ns"] >= before["exit_ns"], "OVERLAP: simultaneous lock callbacks"
    return ordered


def worker(connection: Any, root: str, nonce: str, holder: bool) -> None:
    from prism_harness import FileSessionStore, HarnessError

    directory = Path(root)
    assert json.loads((directory / "owned.json").read_text()) == {"root": root, "nonce": nonce}
    store = FileSessionStore(directory / "store")
    events = directory / "events.jsonl"

    def mark(role: str, iteration: int, event: str) -> None:
        data = (
            json.dumps(
                {
                    "role": role,
                    "iteration": iteration,
                    "event": event,
                    "pid": os.getpid(),
                    "at_ns": time.monotonic_ns(),
                }
            ).encode()
            + b"\n"
        )
        fd = os.open(events, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            assert os.write(fd, data) == len(data)
        finally:
            os.close(fd)

    try:
        while True:
            assert connection.poll(30), "parent command deadline"
            command = connection.recv()
            if command["action"] == "stop":
                return
            iteration, action = command["iteration"], command["action"]
            callbacks = 0

            def callback(action: str = action, iteration: int = iteration) -> None:
                nonlocal callbacks
                callbacks += 1
                role = "holder" if holder else action
                mark(role, iteration, "enter")
                try:
                    if holder:
                        connection.send(
                            {"state": "held", "iteration": iteration, "pid": os.getpid()}
                        )
                        assert connection.poll(30), "holder release deadline"
                        release = connection.recv()
                        assert release == {"action": "release", "iteration": iteration}
                finally:
                    mark(role, iteration, "exit")

            failure = None
            try:
                store.with_lock("hosted-overlap", callback, ttl_seconds=60, wait_seconds=0)
            except HarnessError as error:
                assert error.code == "session_locked"
                failure = error.code
            connection.send(
                {
                    "state": "done",
                    "iteration": iteration,
                    "callbacks": callbacks,
                    "failure": failure,
                    "pid": os.getpid(),
                }
            )
    except BaseException as error:
        connection.send({"state": "error", "class": type(error).__name__, "message": str(error)})
        raise
    finally:
        connection.close()


def main() -> None:
    assert os.environ.get("GITHUB_ACTIONS") == "true", (
        "Only actual hosted Actions runs may claim this stress proof"
    )
    assert sys.platform in ("darwin", "linux"), (
        "This requested stress row is hosted macOS/Linux only"
    )
    bad = [
        {"role": role, "iteration": 0, "pid": pid, "event": event, "at_ns": at}
        for role, pid, event, at in [
            ("holder", 1, "enter", 1),
            ("free", 2, "enter", 2),
            ("free", 2, "exit", 3),
            ("holder", 1, "exit", 4),
        ]
    ]
    try:
        intervals(bad, 1, {"holder": 1, "free": 2})
    except AssertionError as error:
        assert "OVERLAP" in str(error)
    else:
        raise AssertionError("The overlap detector accepted its hostile control")

    root = Path(tempfile.mkdtemp(prefix="prism-file-overlap-", dir=os.environ["RUNNER_TEMP"]))
    nonce = os.urandom(16).hex()
    (root / "owned.json").write_text(json.dumps({"root": str(root), "nonce": nonce}))
    context = multiprocessing.get_context("spawn")
    holder_pipe, holder_child = context.Pipe()
    contender_pipe, contender_child = context.Pipe()
    holder = context.Process(target=worker, args=(holder_child, str(root), nonce, True))
    contender = context.Process(target=worker, args=(contender_child, str(root), nonce, False))
    children = [holder, contender]
    finished = 0
    report: dict[str, Any] = {
        "status": "FAIL",
        "platform": sys.platform,
        "expected_iterations": ITERATIONS,
        "root": str(root),
    }

    def receive(connection: Any) -> dict[str, Any]:
        assert connection.poll(30), "owned worker response deadline"
        value = connection.recv()
        assert value.get("state") != "error", value
        return value

    try:
        holder.start()
        contender.start()
        holder_child.close()
        contender_child.close()
        for iteration in range(ITERATIONS):
            holder_pipe.send({"action": "hold", "iteration": iteration})
            held = receive(holder_pipe)
            assert held == {"state": "held", "iteration": iteration, "pid": holder.pid}
            contender_pipe.send({"action": "contended", "iteration": iteration})
            result = receive(contender_pipe)
            holder_pipe.send({"action": "release", "iteration": iteration})
            released = receive(holder_pipe)
            assert (
                released["iteration"] == iteration
                and released["callbacks"] == 1
                and released["failure"] is None
            )
            assert (
                result["iteration"] == iteration
                and result["callbacks"] == 0
                and result["failure"] == "session_locked"
            ), "Contender entered a held critical section"
            contender_pipe.send({"action": "free", "iteration": iteration})
            positive = receive(contender_pipe)
            assert (
                positive["iteration"] == iteration
                and positive["callbacks"] == 1
                and positive["failure"] is None
            )
            finished += 1
        for connection in (holder_pipe, contender_pipe):
            connection.send({"action": "stop"})
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0, "Owned worker did not finish"
        raw = (root / "events.jsonl").read_bytes()
        events = [json.loads(line) for line in raw.splitlines()]
        measured = intervals(events, ITERATIONS, {"holder": holder.pid, "free": contender.pid})
        assert len(events) == 800 and len(measured) == 400
        assert {row["pid"] for row in events} == {holder.pid, contender.pid}
        assert all(row["role"] != "contended" for row in events)
        assert {row["iteration"] for row in measured} == set(range(ITERATIONS))
        report.update(
            status="PASS",
            iterations=finished,
            callbacks=len(measured),
            events=len(events),
            overlaps=0,
            contended_refusals=ITERATIONS,
            events_sha256=hashlib.sha256(raw).hexdigest(),
            detector_negative_control="rejected",
        )
    finally:
        for child in children:
            if child.pid is not None and child.is_alive():
                child.terminate()
                child.join(timeout=5)
        report["completed_iterations"] = finished
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    print(
        "Hosted owner exclusion passed: 200 iterations; 400 callbacks; 800 events; 0 overlaps; detector control rejected."
    )


if __name__ == "__main__":
    main()
