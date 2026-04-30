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

from annotator import TEAL
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


def fetch_client_id(kombinat_url: str) -> str:
    """Fetch the public GitHub OAuth client_id from kombinat."""
    try:
        with httpx.Client() as client:
            resp = client.get(f"{kombinat_url}/v1/auth/config", timeout=10.0)
    except httpx.HTTPError as exc:
        raise AuthError(f"could not reach kombinat at {kombinat_url}: {exc}") from exc
    if resp.status_code != 200:
        raise AuthError(f"kombinat auth config fetch failed: {resp.status_code} {resp.text}")
    client_id = resp.json().get("client_id")
    if not isinstance(client_id, str) or not client_id:
        raise AuthError("kombinat returned empty client_id — server is misconfigured")
    return client_id


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
            body = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embed Kombinat — Authenticated</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    background: #0a0a0a;
    color: #e0e0e0;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
  }
  .card {
    text-align: center;
    border: 1px solid #2dd4bf33;
    border-radius: 12px;
    padding: 48px 56px;
    background: #111;
    box-shadow: 0 0 40px #2dd4bf11;
  }
  .logo {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 2px;
    line-height: 1.3;
    white-space: pre;
    color: #2dd4bf;
    margin-bottom: 24px;
  }
  h2 {
    font-size: 20px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 8px;
  }
  p {
    font-size: 14px;
    color: #888;
  }
</style>
</head>
<body>
<div class="card">
  <div class="logo">EEEEE  K   K
E      K  K
EEEE   KKK
E      K  K
EEEEE  K   K</div>
  <h2>Authentication successful</h2>
  <p>You can close this tab and return to the terminal.</p>
</div>
</body>
</html>"""
            self.wfile.write(body.encode())
            auth_event.set()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # suppress HTTP server logging

    return _Handler


def _create_callback_server(port: int, auth_event: threading.Event) -> HTTPServer:
    """Bind an HTTP server on a fixed port to receive the OAuth callback.

    Binds the wildcard address (``0.0.0.0``) rather than ``127.0.0.1`` so the
    server is reachable through Docker's port forwarding. With
    ``-p 51820:51820`` the host forwards traffic to the container's bridge
    interface, not loopback — a server bound to 127.0.0.1 inside the container
    would never see those packets and login would silently time out.

    The wider bind is acceptable: CSRF is protected by a 16-byte URL-safe
    ``state`` token (see ``login()``), the server only runs for ~120s during
    the login flow, and the handler only processes a single GET to ``/callback``.
    """
    try:
        server = HTTPServer(("0.0.0.0", port), _make_callback_handler(auth_event))
    except OSError as exc:
        raise AuthError(
            f"Auth callback port {port} is already in use. "
            "Set ANNOTATOR_AUTH_PORT to a different value."
        ) from exc
    server.auth_code = None  # type: ignore[attr-defined]
    server.auth_state = None  # type: ignore[attr-defined]
    return server


def login(settings: Settings, console: Console) -> AuthToken:
    """Run the full login flow: open browser -> local server captures callback -> exchange."""
    console.print("  No credentials found. Starting login...\n")

    client_id = fetch_client_id(settings.kombinat_url)

    state = secrets.token_urlsafe(16)
    auth_event = threading.Event()
    server = _create_callback_server(settings.auth_port, auth_event)
    port = server.server_address[1]

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        redirect_uri = f"http://localhost:{port}/callback"
        authorize_url = (
            f"{GITHUB_AUTHORIZE_URL}"
            f"?client_id={client_id}"
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
            f"  [{TEAL}]\u2713[/{TEAL}] Authenticated as {token.contributor.github_username}"
        )
        return token
    finally:
        server.shutdown()
