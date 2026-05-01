"""Tests for auth.py — token storage, OAuth web flow, fixed-port callback server."""

from __future__ import annotations

import http.client
import os
import socket
import stat
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import httpx
import pytest

from annotator.auth import (
    AuthToken,
    ContributorInfo,
    _create_callback_server,
    delete_token,
    exchange_code,
    fetch_client_id,
    load_token,
    login,
    save_token,
)
from annotator.config import Settings
from annotator.errors import AuthError

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_httpx import HTTPXMock


def _make_token(
    expires_delta: timedelta = timedelta(days=7),
) -> AuthToken:
    return AuthToken(
        kombinat_url="http://test.local",
        access_token="test-jwt",
        expires_at=datetime.now(tz=UTC) + expires_delta,
        contributor=ContributorInfo(
            id="uuid-123",
            github_username="octocat",
            github_avatar_url="https://github.com/octocat.png",
        ),
    )


def _free_port() -> int:
    """Pick a port that's free right now. Tests must bind quickly to avoid races."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class TestTokenStorage:
    def test_save_and_load_token(self, tmp_annotator_home: Path) -> None:
        token = _make_token()
        save_token(token, tmp_annotator_home)
        loaded = load_token(tmp_annotator_home)
        assert loaded is not None
        assert loaded.access_token == token.access_token
        assert loaded.contributor.github_username == "octocat"

    def test_token_file_permissions(self, tmp_annotator_home: Path) -> None:
        save_token(_make_token(), tmp_annotator_home)
        path = tmp_annotator_home / "auth.json"
        mode = os.stat(path).st_mode
        assert mode & stat.S_IRUSR
        assert mode & stat.S_IWUSR
        assert not (mode & stat.S_IRGRP)
        assert not (mode & stat.S_IROTH)

    def test_load_token_not_found(self, tmp_annotator_home: Path) -> None:
        assert load_token(tmp_annotator_home) is None

    def test_load_token_expired(self, tmp_annotator_home: Path) -> None:
        token = _make_token(expires_delta=timedelta(minutes=-1))
        save_token(token, tmp_annotator_home)
        assert load_token(tmp_annotator_home) is None

    def test_load_token_within_expiry_buffer(self, tmp_annotator_home: Path) -> None:
        token = _make_token(expires_delta=timedelta(minutes=3))
        save_token(token, tmp_annotator_home)
        assert load_token(tmp_annotator_home) is None

    def test_load_token_valid(self, tmp_annotator_home: Path) -> None:
        token = _make_token(expires_delta=timedelta(days=7))
        save_token(token, tmp_annotator_home)
        loaded = load_token(tmp_annotator_home)
        assert loaded is not None
        assert not loaded.is_expired()

    def test_delete_token(self, tmp_annotator_home: Path) -> None:
        save_token(_make_token(), tmp_annotator_home)
        assert (tmp_annotator_home / "auth.json").exists()
        delete_token(tmp_annotator_home)
        assert not (tmp_annotator_home / "auth.json").exists()

    def test_delete_token_not_found(self, tmp_annotator_home: Path) -> None:
        delete_token(tmp_annotator_home)

    def test_load_token_corrupted(self, tmp_annotator_home: Path) -> None:
        path = tmp_annotator_home / "auth.json"
        path.write_text("not valid json{{{")
        assert load_token(tmp_annotator_home) is None


class TestFetchClientId:
    def test_fetch_client_id_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"client_id": "gh-client-xyz"}
        with patch("annotator.auth.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.get.return_value = mock_resp
            assert fetch_client_id("http://test.local") == "gh-client-xyz"

    def test_fetch_client_id_non_200_raises(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "boom"
        with patch("annotator.auth.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.get.return_value = mock_resp
            with pytest.raises(AuthError, match="500"):
                fetch_client_id("http://test.local")

    def test_fetch_client_id_empty_raises(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"client_id": ""}
        with patch("annotator.auth.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.get.return_value = mock_resp
            with pytest.raises(AuthError, match="misconfigured"):
                fetch_client_id("http://test.local")

    def test_fetch_client_id_unreachable_raises(self) -> None:
        with patch("annotator.auth.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.get.side_effect = httpx.ConnectError("refused")
            with pytest.raises(AuthError, match="could not reach kombinat"):
                fetch_client_id("http://test.local")


KOMBINAT_AUTH_URL = "http://test.local/v1/auth/github"


class TestExchangeCode:
    """The web-flow code exchange POSTs `{code, state}` to /v1/auth/github."""

    def test_returns_auth_token_on_success(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=KOMBINAT_AUTH_URL,
            method="POST",
            json={
                "access_token": "jwt-xyz",
                "expires_in": 604800,
                "contributor": {
                    "id": "uuid-1",
                    "github_username": "octocat",
                    "github_avatar_url": "https://github.com/octocat.png",
                },
            },
        )
        token = exchange_code("oauth-code", "csrf-state", "http://test.local")
        assert token.access_token == "jwt-xyz"
        assert token.contributor.github_username == "octocat"
        assert token.contributor.id == "uuid-1"
        assert token.kombinat_url == "http://test.local"

    def test_401_raises_rejected(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=KOMBINAT_AUTH_URL,
            method="POST",
            status_code=401,
        )
        with pytest.raises(AuthError, match="rejected"):
            exchange_code("bad-code", "state", "http://test.local")

    def test_500_raises_with_status(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=KOMBINAT_AUTH_URL,
            method="POST",
            status_code=500,
            text="boom",
        )
        with pytest.raises(AuthError, match="500"):
            exchange_code("code", "state", "http://test.local")


class TestCallbackServer:
    """Fixed-port callback server: Docker-friendly, configurable, fails clearly when blocked."""

    def test_binds_to_wildcard_on_configured_port(self) -> None:
        port = _free_port()
        event = threading.Event()
        server = _create_callback_server(port, event)
        try:
            assert server.server_address == ("0.0.0.0", port)
        finally:
            server.server_close()

    def test_port_in_use_raises_clear_error(self) -> None:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", 0))
            probe.listen(1)
            port = int(probe.getsockname()[1])

            event = threading.Event()
            with pytest.raises(AuthError, match="ANNOTATOR_AUTH_PORT"):
                _create_callback_server(port, event)

    def test_callback_captures_code_and_state(self) -> None:
        port = _free_port()
        event = threading.Event()
        server = _create_callback_server(port, event)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/callback?code=abc123&state=xyz789")
            resp = conn.getresponse()
            resp.read()
            conn.close()

            assert resp.status == 200
            assert event.wait(timeout=2)
            assert server.auth_code == "abc123"  # type: ignore[attr-defined]
            assert server.auth_state == "xyz789"  # type: ignore[attr-defined]
        finally:
            server.shutdown()
            thread.join(timeout=2)


def _settings_with_port(home: Path, port: int) -> Settings:
    return Settings(
        annotator_home=home,
        kombinat_url="http://test-kombinat.local",
        auth_port=port,
    )


def _fire_callback(port: int, query: str, delay: float = 0.1) -> threading.Thread:
    """Fire a fake OAuth redirect at the in-process callback server, after a brief delay."""

    def _send() -> None:
        time.sleep(delay)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", f"/callback?{query}")
        conn.getresponse().read()
        conn.close()

    thread = threading.Thread(target=_send, daemon=True)
    thread.start()
    return thread


class TestLogin:
    def test_full_flow_uses_fixed_port_and_saves_token(
        self, tmp_annotator_home: Path, httpx_mock: HTTPXMock
    ) -> None:
        port = _free_port()
        settings = _settings_with_port(tmp_annotator_home, port)
        console = MagicMock()

        httpx_mock.add_response(
            url="http://test-kombinat.local/v1/auth/config",
            method="GET",
            json={"client_id": "test-client-id"},
        )
        httpx_mock.add_response(
            url="http://test-kombinat.local/v1/auth/github",
            method="POST",
            json={
                "access_token": "jwt-login",
                "expires_in": 604800,
                "contributor": {
                    "id": "uuid-login",
                    "github_username": "testuser",
                    "github_avatar_url": None,
                },
            },
        )

        opened: dict[str, str] = {}

        def fake_open(url: str) -> bool:
            opened["url"] = url
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(url).query)
            state = qs["state"][0]
            _fire_callback(port, f"code=oauth-code&state={state}")
            return True

        with patch("annotator.auth.webbrowser.open", side_effect=fake_open):
            token = login(settings, console)

        assert token.access_token == "jwt-login"
        assert token.contributor.github_username == "testuser"
        loaded = load_token(tmp_annotator_home)
        assert loaded is not None
        assert loaded.access_token == "jwt-login"
        assert f"localhost%3A{port}" in opened["url"]

    def test_state_mismatch_raises_csrf(
        self, tmp_annotator_home: Path, httpx_mock: HTTPXMock
    ) -> None:
        port = _free_port()
        settings = _settings_with_port(tmp_annotator_home, port)
        console = MagicMock()

        httpx_mock.add_response(
            url="http://test-kombinat.local/v1/auth/config",
            method="GET",
            json={"client_id": "test-client-id"},
        )

        def fake_open(_url: str) -> bool:
            _fire_callback(port, "code=oauth-code&state=WRONG_STATE")
            return True

        with (
            patch("annotator.auth.webbrowser.open", side_effect=fake_open),
            pytest.raises(AuthError, match="state"),
        ):
            login(settings, console)

    def test_timeout_when_no_callback(
        self, tmp_annotator_home: Path, httpx_mock: HTTPXMock
    ) -> None:
        port = _free_port()
        settings = _settings_with_port(tmp_annotator_home, port)
        console = MagicMock()

        httpx_mock.add_response(
            url="http://test-kombinat.local/v1/auth/config",
            method="GET",
            json={"client_id": "test-client-id"},
        )

        with (
            patch("annotator.auth.LOGIN_TIMEOUT", 0.5),
            patch("annotator.auth.webbrowser.open"),
            pytest.raises(AuthError, match="timed out"),
        ):
            login(settings, console)

    def test_port_in_use_raises_before_browser_opens(
        self, tmp_annotator_home: Path, httpx_mock: HTTPXMock
    ) -> None:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", 0))
            probe.listen(1)
            port = int(probe.getsockname()[1])

            settings = _settings_with_port(tmp_annotator_home, port)
            console = MagicMock()

            httpx_mock.add_response(
                url="http://test-kombinat.local/v1/auth/config",
                method="GET",
                json={"client_id": "test-client-id"},
            )

            with (
                patch("annotator.auth.webbrowser.open") as mock_open,
                pytest.raises(AuthError, match="ANNOTATOR_AUTH_PORT"),
            ):
                login(settings, console)

            mock_open.assert_not_called()
