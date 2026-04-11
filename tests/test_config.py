from __future__ import annotations

from typing import TYPE_CHECKING

from annotator.config import ExitCode, Settings

if TYPE_CHECKING:
    import pytest


class TestSettings:
    def test_default_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Don't let the repo-local .env shadow the code defaults under test.
        # We clear env_file on the class; other tests that rely on a real .env
        # aren't affected because monkeypatch restores it.
        monkeypatch.setitem(Settings.model_config, "env_file", None)
        s = Settings()
        assert s.kombinat_url == "https://api.embedkombinat.dev"
        assert s.batch_size == 100
        assert s.chunk_size == 50
        assert s.gpu_memory_utilization == 0.9
        assert s.max_model_len == 4096
        assert s.max_output_tokens == 256
        assert str(s.annotator_home).endswith(".annotator")

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANNOTATOR_KOMBINAT_URL", "http://localhost:8000")
        monkeypatch.setenv("ANNOTATOR_BATCH_SIZE", "200")
        s = Settings()
        assert s.kombinat_url == "http://localhost:8000"
        assert s.batch_size == 200

    def test_settings_annotator_home_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANNOTATOR_ANNOTATOR_HOME", "/tmp/test-annotator")
        s = Settings()
        assert str(s.annotator_home) == "/tmp/test-annotator"


class TestExitCode:
    def test_exit_codes(self) -> None:
        assert ExitCode.SUCCESS == 0
        assert ExitCode.AUTH_FAILURE == 1
        assert ExitCode.NO_COMPATIBLE_HARDWARE == 2
        assert ExitCode.MODEL_LOADING_FAILED == 3
        assert ExitCode.KOMBINAT_UNREACHABLE == 4
        assert ExitCode.UNRECOVERABLE == 5
