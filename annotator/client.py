"""HTTP client for kombinat API."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime  # noqa: TC003 - needed at runtime by Pydantic

import httpx
from pydantic import BaseModel

from annotator.errors import AuthError, KombinatError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
MIN_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 30.0
NO_PAIRS_INITIAL_WAIT = 30.0
NO_PAIRS_MAX_WAIT = 600.0


class PairData(BaseModel):
    pair_id: str
    query_text: str
    doc_text: str


class BatchResponse(BaseModel):
    batch_id: str
    expires_at: datetime
    pairs: list[PairData]


class AnnotationPayload(BaseModel):
    pair_id: str
    label: int
    input_tokens: int
    output_tokens: int
    raw_response_hash: str


class AnnotationSubmission(BaseModel):
    batch_id: str
    model_id: str
    quantization: str
    annotations: list[AnnotationPayload]


class AnnotationResult(BaseModel):
    accepted: int
    rejected: int
    honeypot_accuracy: float | None = None
    pairs_verified: int = 0
    contributor_tokens: dict[str, int] = {}


class ContributorProfile(BaseModel):
    id: str
    github_username: str
    github_avatar_url: str | None = None
    total_annotations: int = 0
    reputation_score: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    created_at: datetime | None = None
    last_seen_at: datetime | None = None


class NoPairsBackoff:
    """Tracks consecutive 204s and computes wait duration."""

    def __init__(self) -> None:
        self._consecutive_empty = 0

    def wait_duration(self) -> float:
        """Get the next wait duration in seconds."""
        duration = NO_PAIRS_INITIAL_WAIT * (2**self._consecutive_empty)
        result: float = min(duration, NO_PAIRS_MAX_WAIT)
        return result

    def record_empty(self) -> None:
        self._consecutive_empty += 1

    def reset(self) -> None:
        self._consecutive_empty = 0

    @property
    def consecutive_empty(self) -> int:
        return self._consecutive_empty


class KombinatClient:
    def __init__(self, base_url: str, access_token: str) -> None:
        self.http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def claim_batch(self, size: int = 100) -> BatchResponse | None:
        """Claim a batch of pairs. Returns None if no pairs available (204)."""
        resp = self._request_with_retry("POST", "/v1/batches/claim", json={"size": size})
        if resp.status_code == 204:
            return None
        return BatchResponse.model_validate(resp.json())

    def submit_annotations(self, submission: AnnotationSubmission) -> AnnotationResult:
        """Submit a chunk of annotations."""
        resp = self._request_with_retry(
            "POST",
            "/v1/annotations",
            json=submission.model_dump(),
        )
        return AnnotationResult.model_validate(resp.json())

    def release_batch(self, batch_id: str) -> None:
        """Release an unfinished batch back to the pool."""
        self._request_with_retry("DELETE", f"/v1/batches/{batch_id}")

    def get_profile(self) -> ContributorProfile:
        """Get the contributor's profile and stats."""
        resp = self._request_with_retry("GET", "/v1/contributors/me")
        return ContributorProfile.model_validate(resp.json())

    def close(self) -> None:
        self.http.close()

    def _request_with_retry(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        """Make an HTTP request with exponential backoff retry on 5xx/network errors."""
        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self.http.request(method, url, **kwargs)  # type: ignore[arg-type]

                if resp.status_code == 401:
                    raise AuthError("Authentication failed (401). Run 'annotator login'.")
                if resp.status_code == 204:
                    return resp
                if 400 <= resp.status_code < 500:
                    raise KombinatError(f"kombinat error {resp.status_code}: {resp.text}")
                if resp.status_code >= 500:
                    last_exception = KombinatError(
                        f"kombinat server error {resp.status_code}: {resp.text}"
                    )
                    if attempt < MAX_RETRIES:
                        self._backoff_sleep(attempt)
                        continue
                    raise last_exception

                return resp

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = KombinatError(f"kombinat unreachable: {e}")
                last_exception.__cause__ = e
                if attempt < MAX_RETRIES:
                    self._backoff_sleep(attempt)
                    continue
                raise KombinatError(f"kombinat unreachable after {MAX_RETRIES} retries: {e}") from e

        msg = "Request failed after all retries"
        raise KombinatError(msg) if last_exception is None else last_exception

    def _backoff_sleep(self, attempt: int) -> None:
        delay = min(MIN_RETRY_DELAY * (2**attempt) + random.uniform(0, 1), MAX_RETRY_DELAY)
        logger.debug("Retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
        time.sleep(delay)
