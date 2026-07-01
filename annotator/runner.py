"""Main labeling loop: claim -> chunk -> submit -> repeat."""

from __future__ import annotations

import contextlib
import logging
import signal
import sys
import time
from typing import TYPE_CHECKING

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from annotator import AMBER, TEAL, auth
from annotator.client import (
    AnnotationPayload,
    AnnotationSubmission,
    KombinatClient,
    NoPairsBackoff,
)
from annotator.config import ExitCode
from annotator.engine import create_engine
from annotator.engine.base import LabelingInput
from annotator.errors import AuthError, KombinatError, ResolverError
from annotator.labeler import compute_max_user_chars, truncate_document
from annotator.resolver import resolve

if TYPE_CHECKING:
    from rich.console import Console

    from annotator.config import Settings
    from annotator.engine.base import BaseEngine, EngineInfo, LabelingOutput

logger = logging.getLogger(__name__)


class AnnotatorRunner:
    def __init__(self, settings: Settings, console: Console) -> None:
        self._settings = settings
        # Documents must be truncated to fit the engine context window; an
        # overlong prompt makes vLLM raise mid-batch and crashes the run.
        self._max_user_chars = compute_max_user_chars(
            settings.max_model_len, settings.max_output_tokens
        )
        self._console = console
        self._shutdown_requested = False
        self._last_signal_time = 0.0
        self._active_batch_id: str | None = None
        self._client: KombinatClient | None = None

    def run(
        self,
        batch_size: int = 100,
        model_override: str | None = None,
        quantization_override: str | None = None,
        backend_override: str | None = None,
        gpu_memory_utilization: float = 0.9,
        dry_run: bool = False,
    ) -> int:
        """Main entry point. Returns an exit code."""
        self._install_signal_handlers()

        token = auth.load_token(self._settings.annotator_home)
        if token is None:
            try:
                token = auth.login(self._settings, self._console)
            except AuthError as e:
                self._console.print(f"  [{AMBER}]\u2717[/{AMBER}] Login failed: {e}")
                return ExitCode.AUTH_FAILURE

        self._console.print(
            f"  [{TEAL}]\u2713[/{TEAL}] Authenticated as {token.contributor.github_username}"
        )

        try:
            runtime = resolve(
                override_model=model_override,
                override_quantization=quantization_override,
                override_backend=backend_override,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except ResolverError as e:
            self._console.print(f"  [{AMBER}]\u2717[/{AMBER}] {e}")
            return ExitCode.NO_COMPATIBLE_HARDWARE

        self._console.print(
            f"  [{TEAL}]\u2713[/{TEAL}] Detected: {runtime.gpu_name or 'CPU'}"
            + (f" ({runtime.gpu_vram_gb:.0f} GB)" if runtime.gpu_vram_gb else "")
        )
        self._console.print(f"  [{TEAL}]\u2713[/{TEAL}] Best fit: {runtime.model_spec.model_id}")
        self._console.print(f"  [{TEAL}]\u2713[/{TEAL}] Using: {runtime.backend} backend")

        engine = create_engine(
            runtime,
            gpu_memory_utilization,
            self._settings.max_model_len,
            self._settings.max_output_tokens,
        )
        try:
            self._console.print("  \u2193 Loading model...")
            engine.load()
        except Exception as e:
            self._console.print(f"  [{AMBER}]\u2717[/{AMBER}] Model loading failed: {e}")
            return ExitCode.MODEL_LOADING_FAILED

        self._console.print(f"  [{TEAL}]\u2713[/{TEAL}] Model loaded. Starting labeling.\n")

        self._client = KombinatClient(token.kombinat_url, token.access_token)

        try:
            return self._main_loop(engine, batch_size, dry_run)
        except AuthError:
            self._console.print(
                f"\n  [{AMBER}]\u2717[/{AMBER}] Authentication expired. Run 'annotator login'."
            )
            return ExitCode.AUTH_FAILURE
        except KombinatError as e:
            self._console.print(f"\n  [{AMBER}]\u2717[/{AMBER}] kombinat error: {e}")
            return ExitCode.KOMBINAT_UNREACHABLE
        finally:
            self._client.close()

    def _main_loop(self, engine: BaseEngine, batch_size: int, dry_run: bool) -> int:
        assert self._client is not None
        backoff = NoPairsBackoff()
        total_pairs = 0
        total_input_tokens = 0
        total_output_tokens = 0
        batch_num = 0
        engine_info = engine.info()

        while not self._shutdown_requested:
            batch = self._client.claim_batch(batch_size)
            if batch is None:
                wait = backoff.wait_duration()
                backoff.record_empty()
                if backoff.consecutive_empty >= 5:
                    self._console.print(
                        f"\n  [{AMBER}]No pairs available after 5 attempts. "
                        f"Check project status.[/{AMBER}]"
                    )
                self._console.print(f"  [dim]No pairs available. Waiting {wait:.0f}s...[/dim]")
                time.sleep(wait)
                continue

            backoff.reset()
            batch_num += 1
            self._active_batch_id = batch.batch_id
            pairs_in_batch = len(batch.pairs)

            self._console.print(f"  \u2500\u2500 Batch {batch_num} " + "\u2500" * 40)
            self._console.print(f"    Claimed {pairs_in_batch} pairs")

            # Label all pairs, accumulate, then submit once
            chunk_size = self._settings.chunk_size
            all_outputs: list[LabelingOutput] = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("({task.percentage:>5.1f}%)"),
                TimeElapsedColumn(),
                console=self._console,
                transient=True,
            ) as progress:
                task = progress.add_task("Labeling", total=pairs_in_batch)

                for chunk_start in range(0, pairs_in_batch, chunk_size):
                    if self._shutdown_requested:
                        break

                    chunk_end = min(chunk_start + chunk_size, pairs_in_batch)
                    chunk_pairs = [
                        LabelingInput(
                            pair_id=p.pair_id,
                            query_text=p.query_text,
                            doc_text=truncate_document(
                                p.query_text, p.doc_text, self._max_user_chars
                            ),
                        )
                        for p in batch.pairs[chunk_start:chunk_end]
                    ]

                    outputs: list[LabelingOutput] = engine.label_batch(chunk_pairs)
                    all_outputs.extend(outputs)
                    progress.update(task, completed=chunk_end)

                    if dry_run:
                        self._console.print(
                            f"\n  [{TEAL}]\u2713[/{TEAL}] Dry run complete. "
                            f"Processed {len(outputs)} pair(s), no submission."
                        )
                        self._release_active_batch()
                        return ExitCode.SUCCESS

            # Single submission for the whole batch
            batch_labeled = 0
            if all_outputs:
                submission = _build_submission(batch.batch_id, all_outputs, engine_info)
                result = self._client.submit_annotations(submission)
                batch_labeled = len(all_outputs)
                for out in all_outputs:
                    total_input_tokens += out.input_tokens
                    total_output_tokens += out.output_tokens
                logger.info(
                    "Batch submitted: %d accepted, %d rejected",
                    result.accepted,
                    result.rejected,
                )
                self._active_batch_id = None

            total_pairs += batch_labeled
            self._console.print(
                f"    [{TEAL}]\u2713[/{TEAL}] Batch {batch_num}: "
                f"{batch_labeled}/{pairs_in_batch} pairs submitted"
            )

        # Shutdown summary
        self._console.print(
            f"\n  [{TEAL}]\u2713[/{TEAL}] Session total: {total_pairs} pairs "
            f"\u00b7 {total_input_tokens + total_output_tokens:,} tokens contributed"
        )
        self._console.print("  Run again anytime with the same command.\n")

        self._release_active_batch()
        return ExitCode.SUCCESS

    def _release_active_batch(self) -> None:
        """Best-effort release of any in-flight batch back to the pool."""
        if self._active_batch_id and self._client:
            with contextlib.suppress(Exception):
                self._client.release_batch(self._active_batch_id)
            self._active_batch_id = None

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, frame: object) -> None:
            now = time.monotonic()
            if self._shutdown_requested and (now - self._last_signal_time) < 3.0:
                self._console.print("\n  Forced exit.")
                sys.exit(0)
            self._shutdown_requested = True
            self._last_signal_time = now
            self._console.print(f"\n  [{AMBER}]\u26a0[/{AMBER}] Finishing current chunk...")

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


def _build_submission(
    batch_id: str, outputs: list[LabelingOutput], engine_info: EngineInfo
) -> AnnotationSubmission:
    return AnnotationSubmission(
        batch_id=batch_id,
        model_id=engine_info.model_id,
        quantization=engine_info.quantization,
        annotations=[
            AnnotationPayload(
                pair_id=out.pair_id,
                label=out.label,
                input_tokens=out.input_tokens,
                output_tokens=out.output_tokens,
                raw_response_hash=out.raw_response_hash,
            )
            for out in outputs
        ],
    )
