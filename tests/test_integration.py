"""End-to-end integration tests with all mocks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from annotator.auth import AuthToken, ContributorInfo, save_token
from annotator.cli import app
from annotator.client import AnnotationResult, BatchResponse, ContributorProfile, PairData
from annotator.engine.base import EngineInfo, LabelingOutput

if TYPE_CHECKING:
    from pathlib import Path

cli_runner = CliRunner()


def _populate_token(home: Path) -> None:
    save_token(
        AuthToken(
            kombinat_url="http://test.local",
            access_token="test-jwt",
            expires_at=datetime.now(tz=UTC) + timedelta(days=7),
            contributor=ContributorInfo(
                id="uuid-1",
                github_username="octocat",
                github_avatar_url="https://github.com/octocat.png",
            ),
        ),
        home,
    )


def _make_batch(n: int = 4) -> BatchResponse:
    return BatchResponse(
        batch_id="batch-1",
        expires_at=datetime.now(tz=UTC) + timedelta(hours=24),
        pairs=[PairData(pair_id=f"p{i}", query_text=f"q{i}", doc_text=f"d{i}") for i in range(n)],
    )


def _make_mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.info.return_value = EngineInfo(model_id="test-model", quantization="awq", backend="vllm")
    engine.label_batch.side_effect = lambda pairs: [
        LabelingOutput(
            pair_id=p.pair_id,
            label=2,
            reasoning="ok",
            input_tokens=100,
            output_tokens=20,
            raw_response_hash="sha256:abc",
        )
        for p in pairs
    ]
    return engine


def _mock_runner_context(
    mock_engine: MagicMock, mock_client: MagicMock
) -> tuple[patch, patch, patch, patch]:  # type: ignore[type-arg]
    """Return context managers for common runner mocks."""
    return (
        patch(
            "annotator.runner.resolve",
            return_value=MagicMock(
                gpu_name="Test GPU",
                gpu_vram_gb=24.0,
                backend="vllm",
                model_spec=MagicMock(model_id="test-model"),
            ),
        ),
        patch("annotator.runner.create_engine", return_value=mock_engine),
        patch("annotator.runner.KombinatClient", return_value=mock_client),
        patch("annotator.runner.time.sleep"),
    )


class TestFullCycle:
    def test_dry_run(self, tmp_annotator_home: Path) -> None:
        _populate_token(tmp_annotator_home)

        mock_engine = _make_mock_engine()
        mock_client = MagicMock()
        mock_client.claim_batch.return_value = _make_batch(4)
        mock_client.submit_annotations.return_value = AnnotationResult(accepted=2, rejected=0)

        p1, p2, p3, p4 = _mock_runner_context(mock_engine, mock_client)
        with p1, p2, p3, p4:
            result = cli_runner.invoke(app, ["--dry-run"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_custom_batch_size(self, tmp_annotator_home: Path) -> None:
        _populate_token(tmp_annotator_home)

        mock_engine = _make_mock_engine()
        mock_client = MagicMock()
        # Return batch on first call, None on second to end the loop
        mock_client.claim_batch.side_effect = [_make_batch(4), None]
        mock_client.submit_annotations.return_value = AnnotationResult(accepted=2, rejected=0)

        p1, p2, p3, _ = _mock_runner_context(mock_engine, mock_client)
        with p1, p2, p3, patch("annotator.runner.time.sleep") as mock_sleep:
            shutdown_triggered = False

            def trigger_shutdown(*a: object, **k: object) -> None:
                nonlocal shutdown_triggered
                if not shutdown_triggered:
                    shutdown_triggered = True
                    raise SystemExit(0)

            mock_sleep.side_effect = trigger_shutdown
            cli_runner.invoke(
                app,
                ["--batch-size", "200"],
            )

        mock_client.claim_batch.assert_any_call(200)


class TestLoginCommand:
    def test_login_calls_auth_flow(self, tmp_annotator_home: Path) -> None:
        mock_token = AuthToken(
            kombinat_url="http://test.local",
            access_token="jwt",
            expires_at=datetime.now(tz=UTC) + timedelta(days=7),
            contributor=ContributorInfo(
                id="1",
                github_username="test",
                github_avatar_url="https://github.com/test.png",
            ),
        )

        with patch("annotator.auth.login", return_value=mock_token):
            result = cli_runner.invoke(app, ["login"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "Authenticated" in result.output


class TestStatusCommand:
    def test_status_shows_profile(self, tmp_annotator_home: Path) -> None:
        _populate_token(tmp_annotator_home)

        mock_profile = ContributorProfile(
            id="uuid-1",
            github_username="octocat",
            total_annotations=500,
            reputation_score=0.95,
        )
        mock_client_instance = MagicMock()
        mock_client_instance.get_profile.return_value = mock_profile

        with patch("annotator.client.KombinatClient", return_value=mock_client_instance):
            result = cli_runner.invoke(app, ["status"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "octocat" in result.output
        assert "500" in result.output

    def test_status_no_token(self, tmp_annotator_home: Path) -> None:
        result = cli_runner.invoke(app, ["status"], catch_exceptions=False)
        assert result.exit_code == 1
        assert "Not logged in" in result.output


class TestLogoutCommand:
    def test_logout_removes_token(self, tmp_annotator_home: Path) -> None:
        _populate_token(tmp_annotator_home)
        assert (tmp_annotator_home / "auth.json").exists()

        result = cli_runner.invoke(app, ["logout"], catch_exceptions=False)

        assert result.exit_code == 0
        assert not (tmp_annotator_home / "auth.json").exists()
        assert "Logged out" in result.output
