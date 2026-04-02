"""GitHub OAuth web flow and token management."""

from __future__ import annotations

import json
import os
import secrets
import stat
import threading
import urllib.parse
import webbrowser
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel

from annotator.errors import AuthError

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from annotator.config import Settings

AUTH_FILE = "auth.json"
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
EXPIRY_BUFFER = timedelta(minutes=5)
LOGIN_TIMEOUT = 120.0


class ContributorInfo(BaseModel):
    id: str
    github_username: str
    github_avatar_url: str | None = None


class AuthToken(BaseModel):
    kombinat_url: str
    access_token: str
    expires_at: datetime
    contributor: ContributorInfo

    def is_expired(self) -> bool:
        return datetime.now(tz=UTC) >= (self.expires_at - EXPIRY_BUFFER)


def save_token(token: AuthToken, home: Path) -> None:
    """Save auth token to disk with restricted permissions."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / AUTH_FILE
    path.write_text(token.model_dump_json(indent=2))
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def load_token(home: Path) -> AuthToken | None:
    """Load auth token from disk. Returns None if missing or expired."""
    path = home / AUTH_FILE
    if not path.exists():
        return None
    try:
        token = AuthToken.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None
    if token.is_expired():
        return None
    return token


def delete_token(home: Path) -> None:
    """Remove auth token from disk."""
    path = home / AUTH_FILE
    if path.exists():
        path.unlink()


def exchange_code(code: str, state: str, kombinat_url: str) -> AuthToken:
    """Exchange a GitHub OAuth code+state for a kombinat JWT."""
    with httpx.Client() as client:
        resp = client.post(
            f"{kombinat_url}/v1/auth/github",
            json={"code": code, "state": state},
            timeout=30.0,
        )
    if resp.status_code == 401:
        raise AuthError("kombinat rejected the GitHub code (invalid or expired)")
    if resp.status_code != 200:
        raise AuthError(f"kombinat auth failed: {resp.status_code} {resp.text}")
    data = resp.json()
    expires_at = datetime.now(tz=UTC) + timedelta(seconds=data["expires_in"])
    contributor_data = data["contributor"]
    return AuthToken(
        kombinat_url=kombinat_url,
        access_token=data["access_token"],
        expires_at=expires_at,
        contributor=ContributorInfo(
            id=contributor_data["id"],
            github_username=contributor_data["github_username"],
            github_avatar_url=contributor_data.get("github_avatar_url"),
        ),
    )


def _make_callback_handler(auth_event: threading.Event) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP request handler that captures OAuth callback parameters."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            self.server.auth_code = params.get("code", [None])[0]  # type: ignore[attr-defined]
            self.server.auth_state = params.get("state", [None])[0]  # type: ignore[attr-defined]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            body = (
                "<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
                "<h2>Authentication successful!</h2>"
                "<p>You can close this tab and return to the terminal.</p>"
                "</body></html>"
            )
            self.wfile.write(body.encode())
            auth_event.set()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # suppress HTTP server logging

    return _Handler


def _create_callback_server(auth_event: threading.Event) -> HTTPServer:
    """Start a local HTTP server on a random port to receive the OAuth callback."""
    server = HTTPServer(("127.0.0.1", 0), _make_callback_handler(auth_event))
    server.auth_code = None  # type: ignore[attr-defined]
    server.auth_state = None  # type: ignore[attr-defined]
    return server


def login(settings: Settings, console: Console) -> AuthToken:
    """Run the full login flow: open browser -> local server captures callback -> exchange."""
    console.print("  No credentials found. Starting login...\n")

    state = secrets.token_urlsafe(16)
    auth_event = threading.Event()
    server = _create_callback_server(auth_event)
    port = server.server_address[1]

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        redirect_uri = f"http://localhost:{port}/callback"
        authorize_url = (
            f"{GITHUB_AUTHORIZE_URL}"
            f"?client_id={settings.github_client_id}"
            f"&redirect_uri={urllib.parse.quote(redirect_uri, safe='')}"
            f"&state={state}"
            f"&scope=read:user"
        )

        console.print("  [bold]-> Opening browser for GitHub authorization...[/bold]")
        console.print(f"  -> {authorize_url}\n")
        webbrowser.open(authorize_url)
        console.print("  -> Waiting for authorization...\n")

        if not auth_event.wait(timeout=LOGIN_TIMEOUT):
            raise AuthError(
                f"Login timed out — no callback received within {LOGIN_TIMEOUT:.0f} seconds"
            )

        code = server.auth_code  # type: ignore[attr-defined]
        callback_state = server.auth_state  # type: ignore[attr-defined]

        if callback_state != state:
            raise AuthError("OAuth state mismatch — possible CSRF attack")
        if not code:
            raise AuthError("No authorization code received")

        token = exchange_code(code, state, settings.kombinat_url)
        save_token(token, settings.annotator_home)

        console.print(
            f"  [#00E5B0]\u2713[/#00E5B0] Authenticated as {token.contributor.github_username}"
        )
        return token
    finally:
        server.shutdown()
