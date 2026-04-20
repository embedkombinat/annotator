from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _force_plain_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Rich/Typer help output to plain text during tests.

    Rich auto-detects CI environments (CI=1, GITHUB_ACTIONS=1) and starts
    emitting bold ANSI styles even when stdout is a BytesIO, which splits
    option names like `--batch-size` across escape sequences and breaks
    substring assertions in tests/test_cli.py. TERM=dumb tells Rich the
    terminal supports no styling at all, producing plain output.
    """
    monkeypatch.setenv("TERM", "dumb")


@pytest.fixture()
def tmp_annotator_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary annotator home directory and set it via env var."""
    home = tmp_path / ".annotator"
    home.mkdir()
    monkeypatch.setenv("ANNOTATOR_ANNOTATOR_HOME", str(home))
    monkeypatch.setenv("ANNOTATOR_KOMBINAT_URL", "http://test-kombinat.local")
    return home
