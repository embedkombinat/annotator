"""Base engine interface and data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LabelingInput:
    """A single (query, document) pair to label."""

    pair_id: str
    query_text: str
    doc_text: str


@dataclass
class LabelingOutput:
    """LLM response + engine metadata. Ready for submission to kombinat."""

    pair_id: str
    label: int
    reasoning: str
    input_tokens: int
    output_tokens: int
    raw_response_hash: str


@dataclass
class EngineInfo:
    """Model metadata — submitted to kombinat with every annotation."""

    model_id: str
    quantization: str
    backend: str


class BaseEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        """Download model (if not cached) and load into memory."""
        ...

    @abstractmethod
    def label_batch(self, pairs: list[LabelingInput]) -> list[LabelingOutput]:
        """Run inference on a batch. Returns results for successfully labeled pairs only.

        Pairs that fail parsing/validation after retry are silently dropped.
        """
        ...

    @abstractmethod
    def info(self) -> EngineInfo:
        """Return model metadata for submission to kombinat."""
        ...
