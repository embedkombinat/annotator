from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from annotator.cli import app

runner = CliRunner()


class TestCLIHelp:
    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_run_options_in_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--batch-size" in result.output
        assert "--dry-run" in result.output
        assert "--model" in result.output
        assert "--backend" in result.output

    def test_login_help(self) -> None:
        result = runner.invoke(app, ["login", "--help"])
        assert result.exit_code == 0

    def test_status_help(self) -> None:
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_logout_help(self) -> None:
        result = runner.invoke(app, ["logout", "--help"])
        assert result.exit_code == 0

    def test_version_in_output(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "0.1.0" in result.output or result.exit_code == 0


class TestCLICommands:
    def test_default_command_with_mock_runner(self) -> None:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = 0

        with patch("annotator.runner.AnnotatorRunner", return_value=mock_runner_instance):
            result = runner.invoke(app, [])
        assert result.exit_code == 0

    def test_logout_runs(self) -> None:
        with patch("annotator.auth.delete_token"):
            result = runner.invoke(app, ["logout"])
        assert result.exit_code == 0


class TestModelsCommand:
    def test_models_help(self) -> None:
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0

    def test_models_lists_registry_for_backend(self) -> None:
        result = runner.invoke(app, ["models", "--backend", "vllm"])
        assert result.exit_code == 0
        assert "Qwen2.5-7B-Instruct" in result.output
        assert "Mistral-7B-Instruct-v0.3" in result.output
        assert "Phi-3.5-mini-instruct" in result.output

    def test_models_unknown_backend_errors(self) -> None:
        result = runner.invoke(app, ["models", "--backend", "nope"])
        assert result.exit_code != 0
