"""vLLM engine implementation for NVIDIA GPUs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from annotator.engine.base import BaseEngine, EngineInfo, LabelingInput, LabelingOutput
from annotator.labeler import (
    ANNOTATION_SCHEMA_JSON,
    SYSTEM_PROMPT,
    compute_hash,
    format_user_message,
    parse_llm_response,
)

if TYPE_CHECKING:
    from annotator.resolver import ModelSpec


class VLLMEngine(BaseEngine):
    def __init__(self, model_spec: ModelSpec, gpu_memory_utilization: float = 0.9) -> None:
        self.model_spec = model_spec
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm: Any = None

    def load(self) -> None:
        from vllm import LLM  # type: ignore[import-not-found]

        self.llm = LLM(
            model=self.model_spec.model_id,
            quantization=self.model_spec.quantization,
            revision=self.model_spec.revision,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=4096,
            seed=42,
        )

    def label_batch(self, pairs: list[LabelingInput]) -> list[LabelingOutput]:
        from vllm import SamplingParams
        from vllm.sampling_params import (  # type: ignore[import-not-found]
            GuidedDecodingParams,
        )

        guided_params = GuidedDecodingParams(json=ANNOTATION_SCHEMA_JSON)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            guided_decoding=guided_params,
        )

        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_message(pair.query_text, pair.doc_text)},
            ]
            for pair in pairs
        ]

        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        results: list[LabelingOutput] = []
        for pair, output in zip(pairs, outputs, strict=True):
            raw_text: str = output.outputs[0].text
            llm_response = parse_llm_response(raw_text)

            if llm_response is None:
                # Retry once for this pair
                retry_output = self.llm.chat(
                    messages=[conversations[pairs.index(pair)]],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                raw_text = retry_output[0].outputs[0].text
                llm_response = parse_llm_response(raw_text)

            if llm_response is None:
                continue

            results.append(
                LabelingOutput(
                    pair_id=pair.pair_id,
                    label=llm_response.label,
                    reasoning=llm_response.reasoning,
                    input_tokens=len(output.prompt_token_ids),
                    output_tokens=len(output.outputs[0].token_ids),
                    raw_response_hash=compute_hash(raw_text),
                )
            )

        return results

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_id=self.model_spec.model_id,
            quantization=self.model_spec.quantization or "fp16",
            backend="vllm",
        )
