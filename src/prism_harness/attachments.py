"""Which media a turn will carry, decided before a run starts."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from prism_harness.errors import HarnessError

__all__ = ["admit_attachments"]

_MEDIA_KINDS = frozenset({"image", "audio", "video", "document"})


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _carries_bytes(value: object) -> bool:
    """Bytes, decoded the way PHP's base64_decode() and Node's Buffer read them.

    Lenient, and padded so a short final group does not raise: a string that
    decodes to nothing (whitespace, or no base64 characters at all) carries
    nothing, however long it is.
    """
    if not isinstance(value, str):
        return False
    try:
        return len(base64.b64decode(value + "==", validate=False)) > 0
    except binascii.Error:
        return False


def _asks(attachment: object, question: str) -> bool:
    method = getattr(attachment, question, None)
    return callable(method) and method() is True


def admit_attachments(prompt: str, attachments: Sequence[object]) -> list[dict[str, Any]]:
    """Admit a turn's attachments, or refuse the first one it will not send.

    The same rules and the same four codes as the PHP reference and the
    TypeScript port, pinned across all three by prism-parity's
    ``harness-turn-attachments`` corpus. Accepts a ``prism-ai-core`` media object
    (anything with ``to_dict()``) or media already serialized, and returns them
    serialized in the form ``prism-ai-core``'s ``UserMessage.from_dict`` rebuilds.
    """
    if not attachments:
        return []

    if prompt == "":
        raise HarnessError.attachment_without_prompt()

    admitted: list[dict[str, Any]] = []

    for index, attachment in enumerate(attachments):
        to_dict: Callable[[], object] | None = getattr(attachment, "to_dict", None)

        if callable(to_dict):
            # Asked of the OBJECT first. A prism-ai-core media built from a local path
            # serializes as bytes with no path, so the serialized form alone could
            # not tell that it came from a file.
            if _asks(attachment, "is_url"):
                raise HarnessError.attachment_by_reference(index, "a URL")

            if _asks(attachment, "is_file"):
                raise HarnessError.attachment_by_reference(index, "a file path")

            serialized = to_dict()
        else:
            serialized = attachment

        if not isinstance(serialized, Mapping):
            raise HarnessError.attachment_not_media(index, f"a {type(attachment).__name__}")

        data = dict(serialized)

        if data.get("kind") not in _MEDIA_KINDS:
            raise HarnessError.attachment_not_media(
                index, "an object with no image, document, audio or video kind"
            )

        # ANY string, including the empty one. The reference's is_url() asks
        # whether a URL was set at all, and from_url('') is still media built
        # from a URL.
        if isinstance(data.get("url"), str):
            raise HarnessError.attachment_by_reference(index, "a URL")

        if isinstance(data.get("local_path"), str) or isinstance(data.get("storage_path"), str):
            raise HarnessError.attachment_by_reference(index, "a file path")

        chunks = data.get("chunks")
        carries = (
            _carries_bytes(data.get("base64"))
            or _non_empty_string(data.get("file_id"))
            or (data["kind"] == "document" and isinstance(chunks, list) and len(chunks) > 0)
        )

        if not carries:
            raise HarnessError.attachment_empty(index)

        admitted.append(data)

    return admitted
