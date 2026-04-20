from __future__ import annotations

import os
import stat
import threading
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
from annotator.errors import AuthError

if TYPE_CHECKING:
    from pathlib import Path


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
        # Within 5-minute buffer, should be considered expired
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
        # Should not raise
        delete_token(tmp_annotator_home)

    def test_load_token_corrupted(self, tmp_annotator_home: Path) -> None:
        path = tmp_annotator_home / "auth.json"
        path.write_text("not valid json{{{")
        assert load_token(tmp_annotator_home) is None


class TestExchangeCode:
    def test_exchange_code_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "jwt-xyz",
            "expires_in": 604800,
            "contributor": {
                "id": "uuid-456",
                "github_username": "octocat",
                "github_avatar_url": "https://github.com/octocat.png",
            },
        }
        with patch("annotator.auth.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post.return_value = mock_resp

            token = exchange_code("code-abc", "state-xyz", "http://test.local")
            assert token.access_token == "jwt-xyz"
            assert token.contributor.github_username == "octocat"

    def test_exchange_code_401_raises(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch("annotator.auth.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post.return_value = mock_resp

            with pytest.raises(AuthError, match="rejected"):
                exchange_code("bad-code", "state-xyz", "http://test.local")

    def test_exchange_code_500_raises(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch("annotator.auth.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post.return_value = mock_resp

            with pytest.raises(AuthError, match="500"):
                exchange_code("code-abc", "state-xyz", "http://test.local")


class TestCallbackServer:
    def test_callback_captures_code_and_state(self) -> None:
        auth_event = threading.Event()
        server = _create_callback_server(auth_event)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/callback?code=test-code&state=test-state")
            assert resp.status_code == 200
            assert "Authentication successful" in resp.text
            assert auth_event.is_set()
            assert server.auth_code == "test-code"  # type: ignore[attr-defined]
            assert server.auth_state == "test-state"  # type: ignore[attr-defined]
        finally:
            server.shutdown()

    def test_callback_missing_params(self) -> None:
        auth_event = threading.Event()
        server = _create_callback_server(auth_event)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/callback")
            assert resp.status_code == 200
            assert auth_event.is_set()
            assert server.auth_code is None  # type: ignore[attr-defined]
        finally:
            server.shutdown()


def _make_settings(home: Path) -> MagicMock:
    """Create a mock Settings object for login tests."""
    settings = MagicMock()
    settings.kombinat_url = "http://test-kombinat.local"
    settings.annotator_home = home
    return settings


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


class TestLogin:
    def test_login_success(self, tmp_annotator_home: Path) -> None:
        """Simulate a full login by having webbrowser.open trigger the callback."""
        settings = _make_settings(tmp_annotator_home)
        console = MagicMock()

        mock_token_data = {
            "access_token": "jwt-login",
            "expires_in": 604800,
            "contributor": {
                "id": "uuid-login",
                "github_username": "testuser",
                "github_avatar_url": None,
            },
        }

        def fake_browser_open(url: str) -> None:
            """Simulate GitHub redirecting back to the local callback server."""
            import urllib.parse

            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            redirect_uri = urllib.parse.unquote(params["redirect_uri"][0])
            state = params["state"][0]
            # Hit the callback server like GitHub would
            httpx.get(f"{redirect_uri}?code=gh-code-123&state={state}")

        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = mock_token_data

        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = {"client_id": "test-client-id"}

        with (
            patch("annotator.auth.webbrowser.open", side_effect=fake_browser_open),
            patch("annotator.auth.httpx.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post.return_value = post_resp
            mock_client_cls.return_value.get.return_value = get_resp

            token = login(settings, console)

        assert token.access_token == "jwt-login"
        assert token.contributor.github_username == "testuser"
        # Token should be saved to disk
        loaded = load_token(tmp_annotator_home)
        assert loaded is not None
        assert loaded.access_token == "jwt-login"

    def test_login_timeout(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        console = MagicMock()

        with (
            patch("annotator.auth.fetch_client_id", return_value="test-client-id"),
            patch("annotator.auth.webbrowser.open"),  # do nothing
            patch("annotator.auth.LOGIN_TIMEOUT", 0.1),
            pytest.raises(AuthError, match="timed out"),
        ):
            login(settings, console)

    def test_login_state_mismatch(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        console = MagicMock()

        def fake_browser_open(url: str) -> None:
            import urllib.parse

            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            redirect_uri = urllib.parse.unquote(params["redirect_uri"][0])
            # Send back a WRONG state
            httpx.get(f"{redirect_uri}?code=gh-code-123&state=WRONG-STATE")

        with (
            patch("annotator.auth.fetch_client_id", return_value="test-client-id"),
            patch("annotator.auth.webbrowser.open", side_effect=fake_browser_open),
            pytest.raises(AuthError, match="state mismatch"),
        ):
            login(settings, console)
