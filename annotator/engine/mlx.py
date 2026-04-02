"""MLX engine implementation for Apple Silicon."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from annotator.engine.base import BaseEngine, EngineInfo, LabelingInput, LabelingOutput
from annotator.labeler import (
    SYSTEM_PROMPT,
    compute_hash,
    format_user_message,
    parse_llm_response,
)

if TYPE_CHECKING:
    from annotator.resolver import ModelSpec


class MLXEngine(BaseEngine):
    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        from mlx_lm import load  # type: ignore[import-not-found]

        self._model, self._tokenizer = load(self.model_spec.model_id)

    def label_batch(self, pairs: list[LabelingInput]) -> list[LabelingOutput]:
        from mlx_lm import generate

        results: list[LabelingOutput] = []

        for pair in pairs:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_message(pair.query_text, pair.doc_text)},
            ]

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_tokens = len(self._tokenizer.encode(prompt))

            raw_text: str = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=256,
            )
            llm_response = parse_llm_response(raw_text)

            if llm_response is None:
                # Retry once
                raw_text = generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=256,
                )
                llm_response = parse_llm_response(raw_text)

            if llm_response is None:
                continue

            output_tokens = len(self._tokenizer.encode(raw_text))
            results.append(
                LabelingOutput(
                    pair_id=pair.pair_id,
                    label=llm_response.label,
                    reasoning=llm_response.reasoning,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    raw_response_hash=compute_hash(raw_text),
                )
            )

        return results

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_id=self.model_spec.model_id,
            quantization=self.model_spec.quantization or "4bit",
            backend="mlx",
        )
