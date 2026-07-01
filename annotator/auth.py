"""GitHub OAuth Device Flow and token management.

Why device flow: the previous web-flow design ran a localhost callback server
inside the CLI process, then handed GitHub a `redirect_uri=http://localhost:PORT/callback`
URI. That works on a single machine where browser and server are colocated
(your laptop), but breaks the moment the CLI runs on a remote host the user
SSH'd into (Runpod, Lambda, EC2, etc.) — the user's browser hits localhost
on *their machine*, not the remote one. The device flow has no callback at
all: the CLI prints a short user code, the user enters it in any browser on
any device, the CLI polls GitHub directly. Same auth, no machine-boundary
assumption.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import time
import webbrowser
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

from annotator import TEAL
from annotator.errors import AuthError

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from annotator.config import Settings

AUTH_FILE = "auth.json"
DEVICE_CODE_URL = "https://github.com/login/device/code"
TOKEN_URL = "https://github.com/login/oauth/access_token"
DEFAULT_SCOPE = "read:user"
EXPIRY_BUFFER = timedelta(minutes=5)
MIN_POLL_INTERVAL = 5.0


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
    # Create with 0600 atomically instead of write-then-chmod, which leaves a
    # window where the token is readable with default permissions.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as f:
        f.write(token.model_dump_json(indent=2))
    # Tighten pre-existing files too (os.open mode only applies at creation)
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


def request_device_code(client_id: str) -> dict[str, Any]:
    """Ask GitHub for a device code + user code. Returns the full response payload."""
    try:
        with httpx.Client() as client:
            resp = client.post(
                DEVICE_CODE_URL,
                data={"client_id": client_id, "scope": DEFAULT_SCOPE},
                headers={"Accept": "application/json"},
                timeout=10.0,
            )
    except httpx.HTTPError as exc:
        raise AuthError(f"could not reach GitHub: {exc}") from exc
    if resp.status_code != 200:
        raise AuthError(f"device code request failed: {resp.status_code} {resp.text}")
    data: dict[str, Any] = resp.json()
    required = {"device_code", "user_code", "verification_uri", "expires_in", "interval"}
    if not required.issubset(data):
        raise AuthError(
            "device code response is missing required fields. "
            "This usually means the OAuth app does not have Device Flow enabled — "
            "check 'Enable Device Flow' in the OAuth app settings on GitHub."
        )
    return data


def poll_for_access_token(
    client_id: str,
    device_code: str,
    interval: float,
    expires_in: float,
) -> str:
    """Poll GitHub's token endpoint until the user authorizes. Returns the access token."""
    deadline = time.monotonic() + expires_in
    poll_interval = max(interval, MIN_POLL_INTERVAL)

    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        try:
            with httpx.Client() as client:
                resp = client.post(
                    TOKEN_URL,
                    data={
                        "client_id": client_id,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={"Accept": "application/json"},
                    timeout=10.0,
                )
        except httpx.HTTPError as exc:
            raise AuthError(f"polling GitHub failed: {exc}") from exc

        data = resp.json()
        if "access_token" in data:
            return str(data["access_token"])

        error = data.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            poll_interval += 5.0
            continue
        if error == "expired_token":
            raise AuthError("device code expired before authorization completed")
        if error == "access_denied":
            raise AuthError("authorization was denied")
        raise AuthError(f"unexpected error during device flow: {data}")

    raise AuthError(f"device flow timed out after {expires_in:.0f}s")


def exchange_github_token(github_access_token: str, kombinat_url: str) -> AuthToken:
    """Exchange a GitHub access token for a kombinat JWT."""
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{kombinat_url}/v1/auth/github-device",
                json={"access_token": github_access_token},
                timeout=30.0,
            )
    except httpx.HTTPError as exc:
        raise AuthError(f"could not reach kombinat at {kombinat_url}: {exc}") from exc
    if resp.status_code == 401:
        raise AuthError("kombinat rejected the GitHub access token")
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


def login(settings: Settings, console: Console) -> AuthToken:
    """Run the GitHub Device Flow and exchange the access token for a kombinat JWT."""
    console.print("  No credentials found. Starting login...\n")

    client_id = fetch_client_id(settings.kombinat_url)
    device_data = request_device_code(client_id)

    user_code = device_data["user_code"]
    verification_uri = device_data["verification_uri"]
    device_code = device_data["device_code"]
    interval = float(device_data["interval"])
    expires_in = float(device_data["expires_in"])

    console.print(f"  -> Open in any browser: [bold]{verification_uri}[/bold]")
    console.print(f"  -> Enter code: [bold {TEAL}]{user_code}[/bold {TEAL}]\n")

    # Best-effort browser open as a convenience on machines that have one;
    # silently a no-op on headless hosts (Runpod, etc.) which is the whole
    # point of the device flow.
    with contextlib.suppress(Exception):
        webbrowser.open(verification_uri)

    console.print("  -> Waiting for authorization...\n")

    github_token = poll_for_access_token(client_id, device_code, interval, expires_in)
    token = exchange_github_token(github_token, settings.kombinat_url)
    save_token(token, settings.annotator_home)

    console.print(f"  [{TEAL}]✓[/{TEAL}] Authenticated as {token.contributor.github_username}")
    return token
