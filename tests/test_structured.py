"""A turn whose answer is a document.

The PHP reference shipped this as ``Session::sendStructured()``
(prism-harness#13); these tests mirror its suite and the TypeScript port's,
because the decisions worth porting are the RECORDED SHAPE and the refusal, not
the plumbing.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    RunFailed,
    Session,
    ToolRegistry,
    schema_problems,
)

MODES = ModeRegistry(
    {
        "default": "chat",
        "modes": {"chat": {"system_prompt": "Be brief.", "tools": ["echo"], "max_steps": 4}},
    }
)

PLAN_SCHEMA: dict[str, Any] = {
    "name": "plan",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["title", "steps"],
}

PLAN_DOCUMENT: dict[str, Any] = {"title": "Ship it", "steps": ["write", "test"], "confidence": 0.8}


class EchoTool:
    @property
    def name(self) -> str:
        return "echo"

    def handle(self, args: dict[str, Any]) -> Any:
        return f"echoed:{args.get('value', '')}"


def scripted(
    responses: list[LlmResponse], seen: list[LlmRequest] | None = None
) -> Callable[[LlmRequest], LlmResponse]:
    state = {"call": 0}

    def client(request: LlmRequest) -> LlmResponse:
        if seen is not None:
            seen.append(request)
        response = responses[min(state["call"], len(responses) - 1)]
        state["call"] += 1
        return response

    return client


def a_session() -> Session:
    directory = tempfile.mkdtemp(prefix="prism-harness-structured-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("support")
    session.using_mode("chat").using_provider("anthropic").using_model("claude-sonnet-4-5")
    return session


def a_runtime(
    client: Callable[[LlmRequest], LlmResponse], events: HarnessEvents | None = None
) -> AgentRuntime:
    return AgentRuntime(
        client=client, modes=MODES, tools=ToolRegistry().register(EchoTool()), events=events
    )


def rows(session: Session) -> list[dict[str, Any]]:
    return [entry.message for entry in session.thread().messages()]


# -- a structured turn -------------------------------------------------------


def test_returns_the_parsed_document_and_the_text_it_was_read_from() -> None:
    session = a_session()
    client = scripted(
        [
            LlmResponse(
                text=json.dumps(PLAN_DOCUMENT), finish_reason="stop", structured=PLAN_DOCUMENT
            )
        ]
    )

    response = a_runtime(client).send_structured(session, "Plan the release", PLAN_SCHEMA)

    assert response.structured == PLAN_DOCUMENT
    assert response.text == json.dumps(PLAN_DOCUMENT)
    assert response.finish_reason == "stop"


def test_hands_the_schema_to_the_client_and_changes_nothing_else() -> None:
    session = a_session()
    seen: list[LlmRequest] = []
    client = scripted(
        [
            LlmResponse(
                text=json.dumps(PLAN_DOCUMENT), finish_reason="stop", structured=PLAN_DOCUMENT
            )
        ],
        seen,
    )

    a_runtime(client).send_structured(session, "Plan the release", PLAN_SCHEMA)

    assert seen[0].schema == PLAN_SCHEMA
    assert seen[0].system_prompt == "Be brief."
    assert [tool.name for tool in seen[0].tools] == ["echo"]


def test_records_the_document_as_text_with_the_parsed_object_beside_it() -> None:
    # The thread stays readable by everything that reads text today. A
    # transcript that differs by the SHAPE of the request that produced it is
    # the same defect as one that differs when streamed.
    session = a_session()
    client = scripted(
        [
            LlmResponse(
                text=json.dumps(PLAN_DOCUMENT), finish_reason="stop", structured=PLAN_DOCUMENT
            )
        ]
    )

    a_runtime(client).send_structured(session, "Plan the release", PLAN_SCHEMA)
    stored = rows(session)

    assert [row["type"] for row in stored] == ["user", "assistant"]
    assert stored[1]["content"] == json.dumps(PLAN_DOCUMENT)
    assert stored[1]["additional_content"]["structured"] == PLAN_DOCUMENT


def test_replays_to_a_later_turn_as_the_text_the_model_wrote() -> None:
    session = a_session()
    seen: list[LlmRequest] = []
    client = scripted(
        [
            LlmResponse(
                text=json.dumps(PLAN_DOCUMENT), finish_reason="stop", structured=PLAN_DOCUMENT
            ),
            LlmResponse(text="Yes, two steps.", finish_reason="stop"),
        ],
        seen,
    )
    agent = a_runtime(client)

    agent.send_structured(session, "Plan the release", PLAN_SCHEMA)
    agent.send(session, "Is that all?")

    replayed = [m for m in seen[1].messages if m["type"] == "assistant"]

    assert [m["content"] for m in replayed] == [json.dumps(PLAN_DOCUMENT)]
    assert seen[1].schema is None


def test_refuses_a_document_that_misses_the_schema_and_says_every_way() -> None:
    session = a_session()
    wrong = {"title": "Ship it", "confidence": "very"}
    client = scripted([LlmResponse(text=json.dumps(wrong), finish_reason="stop", structured=wrong)])

    with pytest.raises(HarnessError) as raised:
        a_runtime(client).send_structured(session, "Plan the release", PLAN_SCHEMA)

    assert raised.value.code == "structured_schema_violation"
    assert raised.value.document == json.dumps(wrong)
    assert raised.value.problems == [
        "plan.steps is required and missing.",
        "plan.confidence is the string \"very\", and the schema asks for 'number'.",
    ]


def test_refuses_text_that_holds_no_document_under_its_own_code() -> None:
    session = a_session()
    client = scripted(
        [LlmResponse(text="I am afraid I cannot help with that.", finish_reason="stop")]
    )

    with pytest.raises(HarnessError) as raised:
        a_runtime(client).send_structured(session, "Plan the release", PLAN_SCHEMA)

    assert raised.value.code == "structured_unreadable"
    assert raised.value.document == "I am afraid I cannot help with that."
    assert raised.value.problems == []


def test_records_the_refused_document_too_and_fails_the_run() -> None:
    # The exchange happened. A thread that omits the answer it did not like
    # cannot explain the retry sitting next to it.
    session = a_session()
    wrong = {"title": "Ship it"}
    client = scripted([LlmResponse(text=json.dumps(wrong), finish_reason="stop", structured=wrong)])
    seen: list[HarnessEvent] = []
    events = HarnessEvents()
    events.listen(seen.append)

    with pytest.raises(HarnessError):
        a_runtime(client, events).send_structured(session, "Plan the release", PLAN_SCHEMA)

    assert [row["type"] for row in rows(session)] == ["user", "assistant"]
    assert any(isinstance(event, RunFailed) for event in seen)
    assert not any(type(event).__name__ == "RunFinished" for event in seen)


def test_keeps_the_models_words_out_of_the_event_and_names_the_code() -> None:
    # The document is in the thread, in full, where it is read deliberately. An
    # event carrying pieces of it would ship model output to every listener by
    # default -- the same reason tool arguments are names-only here.
    session = a_session()
    wrong = {"title": "Ship it", "steps": "do-not-log"}
    client = scripted([LlmResponse(text=json.dumps(wrong), finish_reason="stop", structured=wrong)])
    seen: list[HarnessEvent] = []
    events = HarnessEvents()
    events.listen(seen.append)

    with pytest.raises(HarnessError):
        a_runtime(client, events).send_structured(session, "Plan the release", PLAN_SCHEMA)

    failed = [event for event in seen if isinstance(event, RunFailed)]

    assert failed[0].failure == "structured_schema_violation"
    assert "do-not-log" not in repr(seen)
    run = session.run()
    assert run is not None
    assert run["status"] == "failed"
    assert run["failure"] == "structured_schema_violation"


def test_records_the_tool_rounds_behind_the_answer() -> None:
    # A thread holding the document and forgetting the tool calls behind it
    # shows a later turn an agent that knew something for no reason.
    session = a_session()
    client = scripted(
        [
            LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("c1", "echo", {"value": "notes"})],
            ),
            LlmResponse(
                text=json.dumps(PLAN_DOCUMENT), finish_reason="stop", structured=PLAN_DOCUMENT
            ),
        ]
    )

    response = a_runtime(client).send_structured(session, "Plan the release", PLAN_SCHEMA)

    assert response.tool_calls == ["echo"]
    assert [row["type"] for row in rows(session)] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]


# -- the schema check --------------------------------------------------------
#
# The same rules as the PHP reference's SchemaCheck and the TypeScript port's
# schemaProblems, message for message. It reads a JSON Schema rather than any
# object model, so a hand-written schema is held to the same terms. What it
# cannot read it passes, and THAT is the part worth pinning: a validator
# silently reporting nothing for a constraint it did not check is the failure
# mode worth naming.


def test_passes_a_document_that_satisfies_its_schema() -> None:
    assert schema_problems(PLAN_SCHEMA, PLAN_DOCUMENT, "plan") == []


def test_names_a_required_field_that_is_missing_by_path() -> None:
    assert schema_problems(PLAN_SCHEMA, {"steps": []}, "plan") == [
        "plan.title is required and missing."
    ]


def test_reports_every_problem_not_the_first() -> None:
    assert len(schema_problems(PLAN_SCHEMA, {}, "plan")) == 2


def test_checks_the_type_of_a_value_that_is_present() -> None:
    document = {"title": "Ship", "steps": [], "confidence": "very"}

    assert schema_problems(PLAN_SCHEMA, document, "plan") == [
        "plan.confidence is the string \"very\", and the schema asks for 'number'."
    ]


def test_accepts_an_integer_for_a_number_and_refuses_a_float_for_an_integer() -> None:
    number: dict[str, Any] = {"type": "object", "properties": {"value": {"type": "number"}}}
    integer: dict[str, Any] = {"type": "object", "properties": {"value": {"type": "integer"}}}

    assert schema_problems(number, {"value": 2}) == []
    assert schema_problems(number, {"value": 2.5}) == []
    assert schema_problems(integer, {"value": 2}) == []
    assert len(schema_problems(integer, {"value": 2.5})) == 1


def test_does_not_let_a_bool_pass_for_a_number() -> None:
    # True == 1 in Python and nowhere else. A document the other two languages
    # refuse must be refused here, or the corpora agree on different behaviour.
    schema: dict[str, Any] = {"type": "object", "properties": {"value": {"type": "number"}}}

    assert schema_problems(schema, {"value": True}) == [
        "document.value is true, and the schema asks for 'number'."
    ]


def test_checks_inside_an_array_item_by_item() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"do": {"type": "string"}},
                    "required": ["do"],
                    "additionalProperties": False,
                },
            }
        },
    }
    document = {"steps": [{"do": "write"}, {"note": "oops"}]}

    assert schema_problems(schema, document, "plan") == [
        "plan.steps[1].do is required and missing.",
        "plan.steps[1].note was returned, and the schema declares no such property.",
    ]


def test_refuses_a_value_outside_an_enum_and_names_the_members() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"mode": {"enum": ["fast", "careful"]}},
    }

    assert schema_problems(schema, {"mode": "reckless"}, "plan") == [
        "plan.mode is the string \"reckless\", which is not one of 'fast', 'careful'."
    ]


def test_accepts_null_only_where_the_schema_says_so() -> None:
    strict: dict[str, Any] = {"type": "object", "properties": {"title": {"type": "string"}}}
    nullable: dict[str, Any] = {
        "type": "object",
        "properties": {"title": {"type": ["string", "null"]}},
    }

    assert len(schema_problems(strict, {"title": None})) == 1
    assert schema_problems(nullable, {"title": None}) == []


def test_reports_an_undeclared_key_only_when_the_schema_closed_itself() -> None:
    closed: dict[str, Any] = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "additionalProperties": False,
    }
    open_: dict[str, Any] = {"type": "object", "properties": {"title": {"type": "string"}}}

    assert len(schema_problems(closed, {"title": "Ship", "extra": 1})) == 1
    assert schema_problems(open_, {"title": "Ship", "extra": 1}) == []


def test_tells_an_object_from_a_list() -> None:
    obj: dict[str, Any] = {"type": "object", "properties": {}}
    array: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

    assert len(schema_problems(obj, ["a", "b"])) == 1
    assert len(schema_problems(array, {"title": "Ship"})) == 1
    assert schema_problems(array, []) == []


def test_passes_what_it_cannot_read_rather_than_guessing() -> None:
    assert schema_problems({"$ref": "#/definitions/Thing"}, {"anything": True}) == []
    assert schema_problems({"type": "integer", "minimum": 10}, 1) == []


def test_takes_any_branch_of_an_any_of() -> None:
    schema: dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "number"}]}

    assert schema_problems(schema, "text") == []
    assert schema_problems(schema, 3) == []
    assert len(schema_problems(schema, True)) == 1


def test_a_whole_float_reads_as_the_number_the_other_languages_write() -> None:
    # 2.0 is 2 in every JSON document, and a message reading "the number 2.0"
    # describes a value PHP and TypeScript both call 2.
    schema: dict[str, Any] = {"type": "object", "properties": {"title": {"type": "string"}}}

    assert schema_problems(schema, {"title": 2.0}) == [
        "document.title is the number 2, and the schema asks for 'string'."
    ]
