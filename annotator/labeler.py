"""Prompt template, response parsing, and validation."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You are a relevance assessor for information retrieval. Judge how relevant a document is to a \
search query.

Rate relevance on a 0-3 scale:
- 0 (Not Relevant): The document does not address the query. Topic, entities, or intent are absent.
- 1 (Marginally Relevant): The document touches on the general topic but does not directly answer \
the query or provide the specific information sought.
- 2 (Relevant): The document addresses the query with partially or indirectly useful information. \
It may lack specificity or depth.
- 3 (Highly Relevant): The document directly and thoroughly addresses the query with the specific \
information sought.

Respond with a JSON object and nothing else:
{"label": <integer 0-3>, "reasoning": "<1-2 sentence explanation>"}

Focus on topical relevance, not document quality."""

ANNOTATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "label": {"type": "integer", "enum": [0, 1, 2, 3]},
        "reasoning": {"type": "string"},
    },
    "required": ["label", "reasoning"],
}

ANNOTATION_SCHEMA_JSON = json.dumps(ANNOTATION_SCHEMA)


class LLMResponse(BaseModel):
    """Validated LLM output."""

    label: int
    reasoning: str


def format_user_message(query: str, doc: str) -> str:
    """Build the user message with query and document."""
    return f"<query>{query}</query>\n<document>{doc}</document>"


def parse_llm_response(raw_text: str) -> LLMResponse | None:
    """Parse and validate LLM JSON output. Returns None on failure."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    label = data.get("label")
    reasoning = data.get("reasoning")

    if not isinstance(label, int) or label not in {0, 1, 2, 3}:
        return None
    if not isinstance(reasoning, str) or len(reasoning.strip()) == 0:
        return None

    return LLMResponse(label=label, reasoning=reasoning)


def compute_hash(raw_text: str) -> str:
    """Compute sha256 hash of raw response text."""
    return f"sha256:{hashlib.sha256(raw_text.encode()).hexdigest()}"


def truncate_document(query: str, doc: str, max_chars: int) -> str:
    """Truncate document from the end if the combined prompt exceeds max_chars.

    Query is never truncated. Returns the (possibly truncated) document.
    """
    # Account for the XML tags and system prompt overhead
    overhead = len("<query></query>\n<document></document>") + len(query)
    available = max_chars - overhead
    if available <= 0:
        return ""
    if len(doc) <= available:
        return doc
    return doc[:available] + "\n[TRUNCATED]"
