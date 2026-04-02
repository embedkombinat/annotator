from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def tmp_annotator_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary annotator home directory and set it via env var."""
    home = tmp_path / ".annotator"
    home.mkdir()
    monkeypatch.setenv("ANNOTATOR_ANNOTATOR_HOME", str(home))
    monkeypatch.setenv("ANNOTATOR_KOMBINAT_URL", "http://test-kombinat.local")
    monkeypatch.setenv("ANNOTATOR_GITHUB_CLIENT_ID", "test-client-id")
    return home
