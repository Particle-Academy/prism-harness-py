"""The suite-wide network block trips (tests/conftest.py).

Each attempt below would leave the process if the block were missing; each
must fail inside the process instead.
"""

from __future__ import annotations

import socket
import urllib.request

import pytest
from conftest import NetworkBlocked


def test_a_raw_socket_connect_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as raw:
        with pytest.raises(NetworkBlocked):
            raw.connect(("127.0.0.1", 9))
        with pytest.raises(NetworkBlocked):
            raw.connect_ex(("127.0.0.1", 9))


def test_create_connection_is_refused() -> None:
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 80), timeout=1)


def test_a_name_lookup_is_refused() -> None:
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("example.com", 443)


def test_an_http_request_through_the_standard_library_is_refused() -> None:
    with pytest.raises(NetworkBlocked):
        urllib.request.urlopen("http://example.com/", timeout=1)
