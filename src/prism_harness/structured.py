"""Does a document the model returned actually satisfy the schema it was given?

A provider that supports strict structured output answers yes by construction.
Most do not, and the ones that do can still be pointed at a model or an endpoint
where the mode is unavailable -- so the question has to be asked here, once, on
the way back.

WHY THIS IS NOT COERCION. The tempting alternative is to keep the fields that fit
and drop the rest, which turns a wrong answer into a plausible one: a planning
agent that returns a document missing its ``nodes`` array becomes a plan with no
steps, and an empty plan settles a batch as done.

WHAT IT CHECKS is what a JSON schema says without needing a resolver: the
declared type, the required keys, the members of an enum, the items of an array,
and, where a schema closes itself to additions, keys nobody declared.

WHAT IT DOES NOT CHECK: ``$ref``, ``allOf``, ``oneOf``, ``not``, and the numeric
and string facets (``minimum``, ``pattern``, ``minLength``). Each would be a
partial implementation of a specification this package has no business owning,
and a partial validator that reports nothing for a constraint it cannot read is
worse than one whose limits are written down. What it cannot read, it passes.

The same rules as ``Prism\\Harness\\Structured\\SchemaCheck`` in the PHP
reference and ``schemaProblems`` in the TypeScript port, message for message, so
a document refused in one language is refused in the others for the same stated
reason.
"""

from __future__ import annotations

from typing import Any

__all__ = ["schema_name", "schema_problems"]


def schema_problems(schema: dict[str, Any], document: Any, name: str = "document") -> list[str]:
    """Every way ``document`` misses ``schema``, rather than only the first.

    A model that drops one required field usually drops several, and fixing them
    one exception at a time costs a provider call each.
    """
    return _check(schema, document, name)


def schema_name(schema: dict[str, Any]) -> str:
    """What to call the document in a problem message.

    A schema that names itself -- Prism's ``ObjectSchema`` writes ``name`` --
    gets its own name in the path, so ``plan.steps[1].do`` reads the way the
    schema's author wrote it. Anything else is just "document".
    """
    named = schema.get("name")

    return named if isinstance(named, str) and named != "" else "document"


def _check(schema: dict[str, Any], value: Any, path: str) -> list[str]:
    any_of = schema.get("anyOf")

    if isinstance(any_of, list):
        return _check_any_of(any_of, value, path)

    members = schema.get("enum")

    if isinstance(members, list):
        if any(_same(member, value) for member in members):
            return []

        return [f"{path} is {_describe(value)}, which is not one of {_members(members)}."]

    types = _declared_types(schema)

    if not types:
        # Nothing declared to check against. A schema that says nothing about a
        # value cannot be violated by it.
        return []

    if not any(_matches(declared, value) for declared in types):
        return [f"{path} is {_describe(value)}, and the schema asks for {_members(types)}."]

    if "object" in types and _is_object(value):
        return _check_object(schema, value, path)

    if "array" in types and _is_array(value):
        return _check_array(schema, value, path)

    return []


def _same(member: Any, value: Any) -> bool:
    """JavaScript's ``===`` over JSON values, which is what the ports compare with.

    ``True == 1`` in Python and nowhere else, so an enum of ``[1, 2]`` would
    accept ``true`` here and refuse it in the other two languages. Numbers still
    compare across int and float, because ``1`` and ``1.0`` are one JSON value.
    """
    if isinstance(member, bool) != isinstance(value, bool):
        return False

    return bool(member == value)


def _check_any_of(branches: list[Any], value: Any, path: str) -> list[str]:
    for branch in branches:
        if isinstance(branch, dict) and not _check(branch, value, path):
            return []

    return [
        (
            f"{path} is {_describe(value)}, which satisfies none of the alternatives "
            "the schema allows."
        )
    ]


def _check_object(schema: dict[str, Any], value: dict[str, Any], path: str) -> list[str]:
    problems: list[str] = []
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required")
    required = required if isinstance(required, list) else []

    for name in required:
        if isinstance(name, str) and name not in value:
            problems.append(f"{path}.{name} is required and missing.")

    for name, prop in properties.items():
        if not isinstance(prop, dict) or name not in value:
            continue

        problems.extend(_check(prop, value[name], f"{path}.{name}"))

    # Only where the schema closed itself. An open object invites the extra key,
    # so reporting it would be this package's opinion rather than the schema's.
    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in properties:
                problems.append(
                    f"{path}.{name} was returned, and the schema declares no such property."
                )

    return problems


def _check_array(schema: dict[str, Any], value: list[Any], path: str) -> list[str]:
    items = schema.get("items")

    if not isinstance(items, dict):
        return []

    problems: list[str] = []

    for index, item in enumerate(value):
        problems.extend(_check(items, item, f"{path}[{index}]"))

    return problems


def _declared_types(schema: dict[str, Any]) -> list[str]:
    """The declared types, as a list -- a nullable schema declares two."""
    declared = schema.get("type")

    if isinstance(declared, str):
        return [declared]

    if isinstance(declared, list):
        return [entry for entry in declared if isinstance(entry, str)]

    return []


def _matches(declared: str, value: Any) -> bool:
    # ``bool`` is a subclass of ``int`` in Python and NOTHING else here is: a
    # bare isinstance(value, int) accepts True where a schema asked for a number,
    # and the document that produced it reads as valid in one language and not in
    # the other two. Checked first, every time.
    if declared == "boolean":
        return isinstance(value, bool)

    if declared == "string":
        return isinstance(value, str)

    # An integer satisfies ``number``, as JSON Schema says it does. The other
    # direction does not.
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if declared == "null":
        return value is None

    if declared == "array":
        return _is_array(value)

    if declared == "object":
        return _is_object(value)

    return True


def _describe(value: Any) -> str:
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if _is_array(value):
        return "an array"

    if _is_object(value):
        return "an object"

    if isinstance(value, str):
        return f'the string "{value[:40] + "…" if len(value) > 40 else value}"'

    return f"the number {_number(value)}"


def _members(values: list[Any]) -> str:
    return ", ".join(
        f"'{value}'" if isinstance(value, str) else _literal(value) for value in values
    )


def _literal(value: Any) -> str:
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    return _number(value) if isinstance(value, (int, float)) else str(value)


def _number(value: Any) -> str:
    """A whole float prints as JSON writes it, not as Python repr()s it.

    ``2.0`` is ``2`` in every JSON document, and a problem message reading
    "the number 2.0" describes a value the other two languages call 2.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def _is_array(value: Any) -> bool:
    return isinstance(value, (list, tuple))
