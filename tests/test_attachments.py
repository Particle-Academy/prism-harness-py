from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import Any

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
    Session,
    ToolRegistry,
)

MODES = ModeRegistry(
    {
        "default": "chat",
        "modes": {
            "chat": {"system_prompt": "Be brief.", "max_steps": 4},
            "thinking": {
                "system_prompt": "Think.",
                "max_steps": 4,
                "provider_options": {"thinking": {"enabled": True, "budgetTokens": 4000}},
            },
        },
    }
)

IMAGE: dict[str, Any] = {
    "kind": "image",
    "url": None,
    "base64": "UE5HQllURVM=",
    "mime_type": "image/png",
    "file_id": None,
    "filename": None,
}


def a_session(mode: str = "chat") -> Session:
    directory = tempfile.mkdtemp(prefix="prism-harness-attachments-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("support")
    session.using_mode(mode)
    return session


def recording() -> tuple[Callable[[LlmRequest], LlmResponse], list[LlmRequest]]:
    """A model that remembers what it was asked."""
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="Seen.", finish_reason="stop")

    return client, requests


def a_runtime(client: Callable[[LlmRequest], LlmResponse]) -> AgentRuntime:
    return AgentRuntime(client=client, modes=MODES, tools=ToolRegistry())


class _Media:
    """Stands in for a prism-ai-core media object, which is asked where it came from."""

    def __init__(self, *, url: bool = False, file: bool = False) -> None:
        self._url = url
        self._file = file

    def to_dict(self) -> dict[str, Any]:
        return dict(IMAGE)

    def is_url(self) -> bool:
        return self._url

    def is_file(self) -> bool:
        return self._file


def test_sends_attachments_and_stores_them_in_the_shape_prism_ai_rebuilds() -> None:
    session = a_session()
    client, requests = recording()

    a_runtime(client).send(session, "What is this?", additional_content=[IMAGE])

    turn = requests[0].messages[0]
    assert turn == {
        "type": "user",
        "content": "What is this?",
        "additional_content": [IMAGE, {"text": "What is this?"}],
        "additional_attributes": {},
    }
    assert session.thread().messages()[0].message == turn


def test_keeps_a_turn_without_attachments_in_its_existing_shape() -> None:
    session = a_session()
    client, requests = recording()

    a_runtime(client).send(session, "Hi")

    assert requests[0].messages[0] == {"type": "user", "content": "Hi"}


REFUSALS: list[tuple[str, object, str, str]] = [
    (
        "a url",
        {**IMAGE, "base64": None, "url": "http://169.254.169.254/latest/meta-data/"},
        "attachment_by_reference",
        "Look",
    ),
    (
        "a url that was fetched",
        {**IMAGE, "url": "https://example.com/a.png"},
        "attachment_by_reference",
        "Look",
    ),
    (
        "a stored local path",
        {**IMAGE, "local_path": "/etc/passwd"},
        "attachment_by_reference",
        "Look",
    ),
    ("a media object from a file", _Media(file=True), "attachment_by_reference", "Look"),
    ("a media object from a url", _Media(url=True), "attachment_by_reference", "Look"),
    ("a string", "UE5HQllURVM=", "attachment_not_media", "Look"),
    ("a text part", {"text": "hello"}, "attachment_not_media", "Look"),
    ("an unknown kind", {**IMAGE, "kind": "spreadsheet"}, "attachment_not_media", "Look"),
    ("empty base64", {**IMAGE, "base64": ""}, "attachment_empty", "Look"),
    ("nothing at all", {"kind": "image"}, "attachment_empty", "Look"),
    (
        "no chunks",
        {"kind": "document", "chunks": [], "document_title": "Empty"},
        "attachment_empty",
        "Look",
    ),
    ("an empty prompt", IMAGE, "attachment_without_prompt", ""),
]


@pytest.mark.parametrize(
    ("label", "attachment", "code", "prompt"), REFUSALS, ids=[row[0] for row in REFUSALS]
)
def test_refuses_an_attachment_before_a_run_or_a_request_exists(
    label: str, attachment: object, code: str, prompt: str
) -> None:
    session = a_session()
    client, requests = recording()

    with pytest.raises(HarnessError) as refused:
        a_runtime(client).send(session, prompt, additional_content=[attachment])

    assert refused.value.code == code, label
    assert requests == []
    assert session.thread().messages() == []
    assert session.run() is None


def test_admits_a_provider_file_id_and_document_chunks() -> None:
    session = a_session()
    client, requests = recording()

    a_runtime(client).send(
        session,
        "Read these",
        additional_content=[
            {
                "kind": "document",
                "url": None,
                "base64": None,
                "file_id": "file_123",
                "document_title": "Report",
                "chunks": None,
            },
            {
                "kind": "document",
                "url": None,
                "base64": None,
                "file_id": None,
                "document_title": "Chunked",
                "chunks": ["One.", "Two."],
            },
        ],
    )

    assert len(requests) == 1


def test_hands_the_modes_provider_options_to_the_model() -> None:
    session = a_session("thinking")
    client, requests = recording()

    a_runtime(client).send(session, "Think about it")

    assert requests[0].provider_options == {"thinking": {"enabled": True, "budgetTokens": 4000}}


def test_hands_an_empty_map_when_a_mode_declares_none() -> None:
    session = a_session()
    client, requests = recording()

    a_runtime(client).send(session, "Hi")

    assert requests[0].provider_options == {}


def test_refuses_provider_options_that_are_not_a_map() -> None:
    broken = ModeRegistry({"modes": {"chat": {"provider_options": ["thinking"]}}})

    with pytest.raises(HarnessError) as refused:
        broken.resolve("chat")

    assert refused.value.code == "mode_malformed"
