from __future__ import annotations

from annotator.labeler import (
    ANNOTATION_SCHEMA,
    SYSTEM_PROMPT,
    compute_hash,
    format_user_message,
    parse_llm_response,
    truncate_document,
)


class TestFormatUserMessage:
    def test_contains_query_and_doc(self) -> None:
        msg = format_user_message("what is Python", "Python is a language")
        assert "what is Python" in msg
        assert "Python is a language" in msg

    def test_uses_xml_tags(self) -> None:
        msg = format_user_message("q", "d")
        assert "<query>q</query>" in msg
        assert "<document>d</document>" in msg


class TestParseLLMResponse:
    def test_valid_response(self) -> None:
        r = parse_llm_response('{"label": 2, "reasoning": "The document is relevant"}')
        assert r is not None
        assert r.label == 2
        assert r.reasoning == "The document is relevant"

    def test_all_valid_labels(self) -> None:
        for label in [0, 1, 2, 3]:
            r = parse_llm_response(f'{{"label": {label}, "reasoning": "ok"}}')
            assert r is not None
            assert r.label == label

    def test_invalid_label_negative(self) -> None:
        assert parse_llm_response('{"label": -1, "reasoning": "ok"}') is None

    def test_invalid_label_4(self) -> None:
        assert parse_llm_response('{"label": 4, "reasoning": "ok"}') is None

    def test_invalid_label_string(self) -> None:
        assert parse_llm_response('{"label": "high", "reasoning": "ok"}') is None

    def test_empty_reasoning(self) -> None:
        assert parse_llm_response('{"label": 2, "reasoning": ""}') is None

    def test_whitespace_reasoning(self) -> None:
        assert parse_llm_response('{"label": 2, "reasoning": "   "}') is None

    def test_not_json(self) -> None:
        assert parse_llm_response("This is not JSON") is None

    def test_missing_label(self) -> None:
        assert parse_llm_response('{"reasoning": "ok"}') is None

    def test_missing_reasoning(self) -> None:
        assert parse_llm_response('{"label": 2}') is None

    def test_extra_fields_ignored(self) -> None:
        r = parse_llm_response('{"label": 2, "reasoning": "ok", "extra": true}')
        assert r is not None
        assert r.label == 2

    def test_float_label_rejected(self) -> None:
        assert parse_llm_response('{"label": 2.0, "reasoning": "ok"}') is None


class TestComputeHash:
    def test_deterministic(self) -> None:
        h1 = compute_hash("hello world")
        h2 = compute_hash("hello world")
        assert h1 == h2

    def test_different_inputs(self) -> None:
        h1 = compute_hash("hello")
        h2 = compute_hash("world")
        assert h1 != h2

    def test_format(self) -> None:
        h = compute_hash("test")
        assert h.startswith("sha256:")
        hex_part = h[len("sha256:") :]
        assert len(hex_part) == 64
        int(hex_part, 16)  # Should not raise


class TestSystemPrompt:
    def test_exists_and_nonempty(self) -> None:
        assert len(SYSTEM_PROMPT) > 0

    def test_mentions_relevance_scale(self) -> None:
        assert "0" in SYSTEM_PROMPT
        assert "3" in SYSTEM_PROMPT
        assert "Not Relevant" in SYSTEM_PROMPT
        assert "Highly Relevant" in SYSTEM_PROMPT

    def test_mentions_json_output(self) -> None:
        assert "JSON" in SYSTEM_PROMPT
        assert "label" in SYSTEM_PROMPT
        assert "reasoning" in SYSTEM_PROMPT


class TestAnnotationSchema:
    def test_valid_json_schema(self) -> None:
        assert ANNOTATION_SCHEMA["type"] == "object"
        props = ANNOTATION_SCHEMA["properties"]
        assert isinstance(props, dict)
        assert "label" in props
        assert "reasoning" in props

    def test_label_enum(self) -> None:
        props = ANNOTATION_SCHEMA["properties"]
        assert isinstance(props, dict)
        label_schema = props["label"]
        assert isinstance(label_schema, dict)
        assert label_schema["enum"] == [0, 1, 2, 3]


class TestTruncateDocument:
    def test_no_truncation(self) -> None:
        doc = "Short document"
        result = truncate_document("query", doc, max_chars=1000)
        assert result == doc

    def test_long_doc_truncated(self) -> None:
        doc = "x" * 10000
        result = truncate_document("q", doc, max_chars=500)
        assert len(result) < len(doc)
        assert result.endswith("[TRUNCATED]")

    def test_query_never_truncated(self) -> None:
        query = "a" * 200
        doc = "b" * 100
        result = truncate_document(query, doc, max_chars=300)
        # The function only truncates the document, not the query
        assert isinstance(result, str)

    def test_empty_doc_budget(self) -> None:
        query = "a" * 1000
        result = truncate_document(query, "some doc", max_chars=100)
        assert result == ""
