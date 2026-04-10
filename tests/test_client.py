from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest

from annotator.client import (
    AnnotationPayload,
    AnnotationSubmission,
    KombinatClient,
    NoPairsBackoff,
)
from annotator.errors import AuthError, KombinatError

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

BASE_URL = "http://test-kombinat.local"
TOKEN = "test-jwt-token"

BATCH_RESPONSE = {
    "batch_id": "batch-123",
    "expires_at": "2026-04-03T14:30:00+00:00",
    "pairs": [
        {"pair_id": "p1", "query_text": "query 1", "doc_text": "doc 1"},
        {"pair_id": "p2", "query_text": "query 2", "doc_text": "doc 2"},
    ],
}

ANNOTATION_RESULT = {"accepted": 2, "rejected": 0, "honeypot_accuracy": None}

PROFILE_RESPONSE = {
    "id": "uuid-1",
    "github_username": "octocat",
    "total_annotations": 500,
    "reputation_score": 0.95,
}


def _make_submission() -> AnnotationSubmission:
    return AnnotationSubmission(
        batch_id="batch-123",
        model_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        annotations=[
            AnnotationPayload(
                pair_id="p1",
                label=2,
                input_tokens=100,
                output_tokens=20,
                raw_response_hash="sha256:abc",
            ),
        ],
    )


class TestClaimBatch:
    def test_success(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            json=BATCH_RESPONSE,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        batch = client.claim_batch(100)
        assert batch is not None
        assert batch.batch_id == "batch-123"
        assert len(batch.pairs) == 2
        client.close()

    def test_no_pairs(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            status_code=204,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        assert client.claim_batch() is None
        client.close()

    def test_unauthorized(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            status_code=401,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        with pytest.raises(AuthError):
            client.claim_batch()
        client.close()


class TestSubmitAnnotations:
    def test_success(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/annotations",
            json=ANNOTATION_RESULT,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        result = client.submit_annotations(_make_submission())
        assert result.accepted == 2
        assert result.rejected == 0
        client.close()

    def test_payload_format(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/annotations",
            json=ANNOTATION_RESULT,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        client.submit_annotations(_make_submission())
        request = httpx_mock.get_requests()[0]
        import json

        body = json.loads(request.content)
        assert body["batch_id"] == "batch-123"
        assert len(body["annotations"]) == 1
        assert body["annotations"][0]["pair_id"] == "p1"
        client.close()


class TestReleaseBatch:
    def test_success(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="DELETE",
            url=f"{BASE_URL}/v1/batches/batch-123",
            status_code=204,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        client.release_batch("batch-123")  # Should not raise
        client.close()


class TestGetProfile:
    def test_success(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/v1/contributors/me",
            json=PROFILE_RESPONSE,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        profile = client.get_profile()
        assert profile.github_username == "octocat"
        assert profile.total_annotations == 500
        client.close()


class TestRetryLogic:
    def test_retry_on_500(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            status_code=500,
        )
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            json=BATCH_RESPONSE,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        with patch("annotator.client.time.sleep"):
            batch = client.claim_batch()
        assert batch is not None
        client.close()

    def test_retry_on_503(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            status_code=503,
        )
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            json=BATCH_RESPONSE,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        with patch("annotator.client.time.sleep"):
            batch = client.claim_batch()
        assert batch is not None
        client.close()

    def test_no_retry_on_401(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            status_code=401,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        with pytest.raises(AuthError):
            client.claim_batch()
        assert len(httpx_mock.get_requests()) == 1
        client.close()

    def test_no_retry_on_400(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            status_code=400,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        with pytest.raises(KombinatError):
            client.claim_batch()
        assert len(httpx_mock.get_requests()) == 1
        client.close()

    def test_max_retries_exceeded(self, httpx_mock: HTTPXMock) -> None:
        for _ in range(4):
            httpx_mock.add_response(
                method="POST",
                url=f"{BASE_URL}/v1/batches/claim",
                status_code=500,
            )
        client = KombinatClient(BASE_URL, TOKEN)
        with patch("annotator.client.time.sleep"), pytest.raises(KombinatError):
            client.claim_batch()
        client.close()

    def test_retry_on_network_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
        )
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/batches/claim",
            json=BATCH_RESPONSE,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        with patch("annotator.client.time.sleep"):
            batch = client.claim_batch()
        assert batch is not None
        client.close()

    def test_auth_header_present(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/v1/contributors/me",
            json=PROFILE_RESPONSE,
        )
        client = KombinatClient(BASE_URL, TOKEN)
        client.get_profile()
        request = httpx_mock.get_requests()[0]
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        client.close()


class TestNoPairsBackoff:
    def test_initial_wait(self) -> None:
        b = NoPairsBackoff()
        assert b.wait_duration() == 30.0

    def test_doubling(self) -> None:
        b = NoPairsBackoff()
        b.record_empty()
        assert b.wait_duration() == 60.0
        b.record_empty()
        assert b.wait_duration() == 120.0
        b.record_empty()
        assert b.wait_duration() == 240.0

    def test_max_cap(self) -> None:
        b = NoPairsBackoff()
        for _ in range(10):
            b.record_empty()
        assert b.wait_duration() == 600.0

    def test_reset(self) -> None:
        b = NoPairsBackoff()
        b.record_empty()
        b.record_empty()
        b.reset()
        assert b.wait_duration() == 30.0
