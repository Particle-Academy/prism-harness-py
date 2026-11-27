"""Suite-wide: no test reaches the network.

Every test runs with outbound connections and name lookups refused, so a test
that would make a real network call fails instead. The package itself has no
network client; this keeps every test (and any probe added to this suite)
offline. Child processes a test starts are not covered by this in-process
block.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


class NetworkBlocked(AssertionError):
    """A test tried to use the network."""


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise NetworkBlocked("A test tried to use the network.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
    yield
