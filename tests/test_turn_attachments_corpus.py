"""The cross-language turn-attachments corpus from ``prism-parity``.

Which attachments a turn will carry. If this port admitted what the reference
refuses, an application moving an agent from PHP to Python would lose the guard
with no error anywhere, and a file read from a request-supplied path would
simply go to the model.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from prism_harness import HarnessError, admit_attachments

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "harness-turn-attachments.json").read_text(
        encoding="utf-8"
    )
)


class _FromFile:
    """Media that knows it came from a file, as prism-ai-core's from_local_path() does."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)

    def is_url(self) -> bool:
        return False

    def is_file(self) -> bool:
        return True


def _attachment_for(spec: dict[str, Any]) -> object:
    """The same spec, built the way a Python caller would hold it. Mirrors the recorder."""
    if spec["$"] == "Text":
        return {"text": spec["text"]}
    if spec["$"] == "String":
        return spec["value"]

    kind = "document" if spec["$"] == "Document" else "image"
    base: dict[str, Any] = {
        "kind": kind,
        "url": None,
        "base64": None,
        "mime_type": spec.get("mimeType"),
        "file_id": None,
        "filename": None,
    }

    def titled(payload: dict[str, Any]) -> dict[str, Any]:
        if kind != "document":
            return payload
        return {**payload, "document_title": spec.get("title"), "chunks": payload.get("chunks")}

    source = spec["from"]
    if source == "base64":
        return titled({**base, "base64": spec["base64"]})
    if source == "url":
        return titled({**base, "url": spec["url"]})
    if source == "urlWithBytes":
        return titled({**base, "url": spec["url"], "base64": spec["base64"]})
    if source == "localPath":
        encoded = base64.b64encode(spec["bytes"].encode("utf-8")).decode("ascii")
        return _FromFile(titled({**base, "base64": encoded}))
    if source == "fileId":
        return titled({**base, "file_id": spec["fileId"]})
    if source == "chunks":
        return titled({**base, "chunks": spec["chunks"]})
    if source == "text":
        encoded = base64.b64encode(spec["text"].encode("utf-8")).decode("ascii")
        return titled({**base, "base64": encoded, "mime_type": "text/plain"})
    if source == "nothing":
        return titled(base)
    raise ValueError(f"Unknown media source {source}")


def _verdict(case: dict[str, Any]) -> str:
    try:
        admit_attachments(case["prompt"], [_attachment_for(spec) for spec in case["attachments"]])
    except HarnessError as refused:
        return str(refused.code)
    return "admitted"


def test_is_the_whole_suite_not_a_subset_trimmed_to_green() -> None:
    assert len(CORPUS["cases"]) == 18


@pytest.mark.parametrize("case", CORPUS["cases"], ids=[case["id"] for case in CORPUS["cases"]])
def test_gets_the_reference_verdict(case: dict[str, Any]) -> None:
    assert _verdict(case) == case["verdict"]["php"], case["title"]


def test_agrees_with_the_reference_on_every_row() -> None:
    for case in CORPUS["cases"]:
        verdict = case["verdict"]
        assert [verdict["ts"], verdict["py"]] == [verdict["php"], verdict["php"]], case["id"]
        assert case["agrees"] is True, case["id"]
