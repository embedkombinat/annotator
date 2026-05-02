"""Tests for auth.py — token storage and the GitHub Device Flow."""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import httpx
import pytest

from annotator.auth import (
    AuthToken,
    ContributorInfo,
    delete_token,
    exchange_github_token,
    fetch_client_id,
    load_token,
    login,
    poll_for_access_token,
    request_device_code,
    save_token,
)
from annotator.config import Settings
from annotator.errors import AuthError

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_httpx import HTTPXMock


GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
KOMBINAT_DEVICE_AUTH_URL = "http://test.local/v1/auth/github-device"


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


class TestRequestDeviceCode:
    """`request_device_code` POSTs to /login/device/code and returns the parsed payload."""

    def test_returns_device_data(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=GITHUB_DEVICE_CODE_URL,
            method="POST",
            json={
                "device_code": "abc",
                "user_code": "1234-WXYZ",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 5,
            },
        )
        data = request_device_code("test-client-id")
        assert data["device_code"] == "abc"
        assert data["user_code"] == "1234-WXYZ"
        assert data["interval"] == 5

    def test_missing_fields_raises_helpful_error(self, httpx_mock: HTTPXMock) -> None:
        """Partial payload usually means the OAuth app doesn't have Device Flow enabled."""
        httpx_mock.add_response(
            url=GITHUB_DEVICE_CODE_URL,
            method="POST",
            json={"error": "device_flow_disabled"},
        )
        with pytest.raises(AuthError, match="Device Flow"):
            request_device_code("test-client-id")

    def test_non_200_raises(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=GITHUB_DEVICE_CODE_URL,
            method="POST",
            status_code=500,
            text="boom",
        )
        with pytest.raises(AuthError, match="500"):
            request_device_code("test-client-id")

    def test_unreachable_raises(self) -> None:
        with patch("annotator.auth.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = lambda s: s
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value.post.side_effect = httpx.ConnectError("refused")
            with pytest.raises(AuthError, match="could not reach GitHub"):
                request_device_code("test-client-id")


class TestPollForAccessToken:
    """Polling loop returns the token, retrying on transient errors."""

    def test_returns_token_on_first_poll(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=GITHUB_TOKEN_URL,
            method="POST",
            json={"access_token": "gho_x", "token_type": "bearer"},
        )
        with patch("annotator.auth.MIN_POLL_INTERVAL", 0.0):
            token = poll_for_access_token("client-id", "device-code", interval=0.0, expires_in=10.0)
        assert token == "gho_x"

    def test_retries_on_authorization_pending(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=GITHUB_TOKEN_URL, method="POST", json={"error": "authorization_pending"}
        )
        httpx_mock.add_response(url=GITHUB_TOKEN_URL, method="POST", json={"access_token": "gho_y"})
        with patch("annotator.auth.MIN_POLL_INTERVAL", 0.0):
            token = poll_for_access_token("c", "d", interval=0.0, expires_in=10.0)
        assert token == "gho_y"

    def test_slow_down_then_success(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=GITHUB_TOKEN_URL, method="POST", json={"error": "slow_down"})
        httpx_mock.add_response(url=GITHUB_TOKEN_URL, method="POST", json={"access_token": "gho_z"})
        with patch("annotator.auth.MIN_POLL_INTERVAL", 0.0):
            token = poll_for_access_token("c", "d", interval=0.0, expires_in=10.0)
        assert token == "gho_z"

    def test_access_denied_raises(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=GITHUB_TOKEN_URL, method="POST", json={"error": "access_denied"}
        )
        with (
            patch("annotator.auth.MIN_POLL_INTERVAL", 0.0),
            pytest.raises(AuthError, match="denied"),
        ):
            poll_for_access_token("c", "d", interval=0.0, expires_in=10.0)

    def test_expired_token_raises(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=GITHUB_TOKEN_URL, method="POST", json={"error": "expired_token"}
        )
        with (
            patch("annotator.auth.MIN_POLL_INTERVAL", 0.0),
            pytest.raises(AuthError, match="expired"),
        ):
            poll_for_access_token("c", "d", interval=0.0, expires_in=10.0)

    def test_unexpected_error_raises(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=GITHUB_TOKEN_URL, method="POST", json={"error": "something_weird"}
        )
        with (
            patch("annotator.auth.MIN_POLL_INTERVAL", 0.0),
            pytest.raises(AuthError, match="unexpected error"),
        ):
            poll_for_access_token("c", "d", interval=0.0, expires_in=10.0)

    def test_timeout_raises_when_deadline_passed(self) -> None:
        with pytest.raises(AuthError, match="timed out"):
            poll_for_access_token("c", "d", interval=5.0, expires_in=0.0)


class TestExchangeGithubToken:
    """`exchange_github_token` POSTs the GitHub access token to kombinat /v1/auth/github-device."""

    def test_returns_auth_token_on_success(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=KOMBINAT_DEVICE_AUTH_URL,
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
        token = exchange_github_token("gho_mock", "http://test.local")
        assert token.access_token == "jwt-xyz"
        assert token.contributor.github_username == "octocat"
        assert token.contributor.id == "uuid-1"
        assert token.kombinat_url == "http://test.local"

    def test_401_raises_rejected(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=KOMBINAT_DEVICE_AUTH_URL, method="POST", status_code=401)
        with pytest.raises(AuthError, match="rejected"):
            exchange_github_token("gho_bad", "http://test.local")

    def test_500_raises_with_status(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=KOMBINAT_DEVICE_AUTH_URL, method="POST", status_code=500, text="boom"
        )
        with pytest.raises(AuthError, match="500"):
            exchange_github_token("gho_x", "http://test.local")


class TestLogin:
    """End-to-end device flow: config → device code → poll → kombinat exchange → save."""

    def test_full_flow_persists_token(
        self, tmp_annotator_home: Path, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url="http://test-kombinat.local/v1/auth/config",
            method="GET",
            json={"client_id": "test-client-id"},
        )
        httpx_mock.add_response(
            url=GITHUB_DEVICE_CODE_URL,
            method="POST",
            json={
                "device_code": "abc",
                "user_code": "1234-WXYZ",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 5,
            },
        )
        httpx_mock.add_response(
            url=GITHUB_TOKEN_URL, method="POST", json={"access_token": "gho_token"}
        )
        httpx_mock.add_response(
            url="http://test-kombinat.local/v1/auth/github-device",
            method="POST",
            json={
                "access_token": "jwt-xyz",
                "expires_in": 604800,
                "contributor": {
                    "id": "uuid-1",
                    "github_username": "octocat",
                    "github_avatar_url": None,
                },
            },
        )
        settings = Settings(
            kombinat_url="http://test-kombinat.local",
            annotator_home=tmp_annotator_home,
        )
        console = MagicMock()
        with (
            patch("annotator.auth.MIN_POLL_INTERVAL", 0.0),
            patch("annotator.auth.webbrowser.open"),
        ):
            token = login(settings, console)

        assert token.contributor.github_username == "octocat"
        loaded = load_token(tmp_annotator_home)
        assert loaded is not None
        assert loaded.access_token == "jwt-xyz"

    def test_propagates_device_flow_disabled_error(
        self, tmp_annotator_home: Path, httpx_mock: HTTPXMock
    ) -> None:
        """When the OAuth app doesn't have Device Flow enabled, login surfaces a clear error."""
        httpx_mock.add_response(
            url="http://test-kombinat.local/v1/auth/config",
            method="GET",
            json={"client_id": "test-client-id"},
        )
        httpx_mock.add_response(
            url=GITHUB_DEVICE_CODE_URL,
            method="POST",
            json={"error": "device_flow_disabled"},
        )
        settings = Settings(
            kombinat_url="http://test-kombinat.local",
            annotator_home=tmp_annotator_home,
        )
        console = MagicMock()
        with pytest.raises(AuthError, match="Device Flow"):
            login(settings, console)
