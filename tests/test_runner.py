from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from rich.console import Console

from annotator.auth import AuthToken, ContributorInfo, save_token
from annotator.client import AnnotationResult, BatchResponse, PairData
from annotator.config import ExitCode, Settings
from annotator.engine.base import EngineInfo, LabelingInput, LabelingOutput
from annotator.runner import AnnotatorRunner

if TYPE_CHECKING:
    from pathlib import Path


def _make_settings(home: Path) -> Settings:
    return Settings(
        annotator_home=home,
        kombinat_url="http://test.local",
        chunk_size=2,
    )


def _make_token(home: Path) -> AuthToken:
    token = AuthToken(
        kombinat_url="http://test.local",
        access_token="test-jwt",
        expires_at=datetime.now(tz=UTC) + timedelta(days=7),
        contributor=ContributorInfo(
            id="uuid-1",
            github_username="octocat",
            github_avatar_url="https://github.com/octocat.png",
        ),
    )
    save_token(token, home)
    return token


def _make_batch(n_pairs: int = 4) -> BatchResponse:
    return BatchResponse(
        batch_id="batch-1",
        expires_at=datetime.now(tz=UTC) + timedelta(hours=24),
        pairs=[
            PairData(pair_id=f"p{i}", query_text=f"q{i}", doc_text=f"d{i}") for i in range(n_pairs)
        ],
    )


def _make_output(pair_id: str) -> LabelingOutput:
    return LabelingOutput(
        pair_id=pair_id,
        label=2,
        reasoning="relevant",
        input_tokens=100,
        output_tokens=20,
        raw_response_hash="sha256:abc",
    )


def _make_mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.info.return_value = EngineInfo(
        model_id="test-model",
        quantization="awq",
        backend="vllm",
    )
    engine.label_batch.side_effect = lambda pairs: [_make_output(p.pair_id) for p in pairs]
    return engine


class TestRunnerBasicLoop:
    def test_claims_and_processes_batch(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        _make_token(tmp_annotator_home)
        console = Console(quiet=True)
        runner = AnnotatorRunner(settings, console)

        mock_engine = _make_mock_engine()
        mock_client = MagicMock()
        mock_client.claim_batch.side_effect = [_make_batch(4), None]
        mock_client.submit_annotations.return_value = AnnotationResult(accepted=2, rejected=0)

        with (
            patch("annotator.runner.resolve") as mock_resolve,
            patch("annotator.runner.create_engine", return_value=mock_engine),
            patch("annotator.runner.KombinatClient", return_value=mock_client),
            patch("annotator.runner.time.sleep", side_effect=_trigger_shutdown(runner)),
        ):
            mock_resolve.return_value = MagicMock(
                gpu_name="Test GPU",
                gpu_vram_gb=24.0,
                backend="vllm",
                model_spec=MagicMock(model_id="test-model"),
            )
            code = runner.run()

        assert code == ExitCode.SUCCESS
        assert mock_client.submit_annotations.call_count == 1  # single batch submission

    def test_chunks_batch(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        _make_token(tmp_annotator_home)
        console = Console(quiet=True)
        runner = AnnotatorRunner(settings, console)

        mock_engine = _make_mock_engine()
        mock_client = MagicMock()
        # 6 pairs with chunk_size=2 = 3 chunks
        mock_client.claim_batch.side_effect = [_make_batch(6), None]
        mock_client.submit_annotations.return_value = AnnotationResult(accepted=2, rejected=0)

        with (
            patch("annotator.runner.resolve") as mock_resolve,
            patch("annotator.runner.create_engine", return_value=mock_engine),
            patch("annotator.runner.KombinatClient", return_value=mock_client),
            patch("annotator.runner.time.sleep", side_effect=_trigger_shutdown(runner)),
        ):
            mock_resolve.return_value = MagicMock(
                gpu_name="Test GPU",
                gpu_vram_gb=24.0,
                backend="vllm",
                model_spec=MagicMock(model_id="test-model"),
            )
            code = runner.run()

        assert code == ExitCode.SUCCESS
        assert mock_client.submit_annotations.call_count == 1  # single batch submission

    def test_skips_failed_pairs(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        _make_token(tmp_annotator_home)
        console = Console(quiet=True)
        runner = AnnotatorRunner(settings, console)

        mock_engine = _make_mock_engine()
        # Return only 1 of 2 pairs (one "failed")
        mock_engine.label_batch.side_effect = lambda pairs: (
            [_make_output(pairs[0].pair_id)] if len(pairs) >= 1 else []
        )

        mock_client = MagicMock()
        mock_client.claim_batch.side_effect = [_make_batch(2), None]
        mock_client.submit_annotations.return_value = AnnotationResult(accepted=1, rejected=0)

        with (
            patch("annotator.runner.resolve") as mock_resolve,
            patch("annotator.runner.create_engine", return_value=mock_engine),
            patch("annotator.runner.KombinatClient", return_value=mock_client),
            patch("annotator.runner.time.sleep", side_effect=_trigger_shutdown(runner)),
        ):
            mock_resolve.return_value = MagicMock(
                gpu_name="Test GPU",
                gpu_vram_gb=24.0,
                backend="vllm",
                model_spec=MagicMock(model_id="test-model"),
            )
            code = runner.run()

        assert code == ExitCode.SUCCESS
        # Only 1 annotation submitted per chunk
        submission = mock_client.submit_annotations.call_args[0][0]
        assert len(submission.annotations) == 1


class TestRunnerAuth:
    def test_auto_login_when_no_credentials(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        # No token saved
        console = Console(quiet=True)
        runner = AnnotatorRunner(settings, console)

        mock_token = AuthToken(
            kombinat_url="http://test.local",
            access_token="test-jwt",
            expires_at=datetime.now(tz=UTC) + timedelta(days=7),
            contributor=ContributorInfo(
                id="uuid-1",
                github_username="octocat",
                github_avatar_url="https://github.com/octocat.png",
            ),
        )

        mock_engine = _make_mock_engine()
        mock_client = MagicMock()
        mock_client.claim_batch.return_value = None

        with (
            patch("annotator.runner.auth.login", return_value=mock_token) as mock_login,
            patch("annotator.runner.resolve") as mock_resolve,
            patch("annotator.runner.create_engine", return_value=mock_engine),
            patch("annotator.runner.KombinatClient", return_value=mock_client),
            patch("annotator.runner.time.sleep", side_effect=_trigger_shutdown(runner)),
        ):
            mock_resolve.return_value = MagicMock(
                gpu_name="Test GPU",
                gpu_vram_gb=24.0,
                backend="vllm",
                model_spec=MagicMock(model_id="test-model"),
            )
            runner.run()

        assert mock_login.call_count == 1

    def test_auth_failure_exits_1(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        console = Console(quiet=True)
        runner = AnnotatorRunner(settings, console)

        from annotator.errors import AuthError

        with patch("annotator.runner.auth.login", side_effect=AuthError("fail")):
            code = runner.run()

        assert code == ExitCode.AUTH_FAILURE


class TestRunnerDryRun:
    def test_dry_run_processes_without_full_batch(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        _make_token(tmp_annotator_home)
        console = Console(quiet=True)
        runner = AnnotatorRunner(settings, console)

        mock_engine = _make_mock_engine()
        mock_client = MagicMock()
        mock_client.claim_batch.return_value = _make_batch(4)
        mock_client.submit_annotations.return_value = AnnotationResult(accepted=2, rejected=0)

        with (
            patch("annotator.runner.resolve") as mock_resolve,
            patch("annotator.runner.create_engine", return_value=mock_engine),
            patch("annotator.runner.KombinatClient", return_value=mock_client),
        ):
            mock_resolve.return_value = MagicMock(
                gpu_name="Test GPU",
                gpu_vram_gb=24.0,
                backend="vllm",
                model_spec=MagicMock(model_id="test-model"),
            )
            code = runner.run(dry_run=True)

        assert code == ExitCode.SUCCESS


class TestRunnerShutdown:
    def test_graceful_shutdown(self, tmp_annotator_home: Path) -> None:
        settings = _make_settings(tmp_annotator_home)
        _make_token(tmp_annotator_home)
        console = Console(quiet=True)
        runner = AnnotatorRunner(settings, console)

        call_count = 0

        def label_and_shutdown(pairs: list[LabelingInput]) -> list[LabelingOutput]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                runner._shutdown_requested = True
            return [_make_output(p.pair_id) for p in pairs]

        mock_engine = _make_mock_engine()
        mock_engine.label_batch.side_effect = label_and_shutdown

        mock_client = MagicMock()
        mock_client.claim_batch.return_value = _make_batch(6)
        mock_client.submit_annotations.return_value = AnnotationResult(accepted=2, rejected=0)

        with (
            patch("annotator.runner.resolve") as mock_resolve,
            patch("annotator.runner.create_engine", return_value=mock_engine),
            patch("annotator.runner.KombinatClient", return_value=mock_client),
        ):
            mock_resolve.return_value = MagicMock(
                gpu_name="Test GPU",
                gpu_vram_gb=24.0,
                backend="vllm",
                model_spec=MagicMock(model_id="test-model"),
            )
            code = runner.run()

        assert code == ExitCode.SUCCESS
        # Should have submitted the completed chunks before stopping
        assert mock_client.submit_annotations.call_count >= 1


def _trigger_shutdown(runner: AnnotatorRunner):  # type: ignore[no-untyped-def]
    """Side effect for time.sleep that triggers shutdown on first call."""

    def side_effect(*args: object, **kwargs: object) -> None:
        runner._shutdown_requested = True

    return side_effect
