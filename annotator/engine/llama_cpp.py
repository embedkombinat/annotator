"""llama.cpp engine stub (Phase 3 — not yet implemented)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from annotator.engine.base import BaseEngine, EngineInfo, LabelingInput, LabelingOutput

if TYPE_CHECKING:
    from annotator.resolver import ModelSpec


class LlamaCppEngine(BaseEngine):
    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec

    def load(self) -> None:
        raise NotImplementedError("llama.cpp backend not yet implemented")

    def label_batch(self, pairs: list[LabelingInput]) -> list[LabelingOutput]:
        raise NotImplementedError("llama.cpp backend not yet implemented")

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_id=self.model_spec.model_id,
            quantization=self.model_spec.quantization or "Q4_K_M",
            backend="llama_cpp",
        )
