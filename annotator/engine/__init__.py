"""Inference engine backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from annotator.engine.base import BaseEngine
    from annotator.resolver import ResolvedRuntime


def create_engine(
    runtime: ResolvedRuntime,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    max_output_tokens: int = 256,
) -> BaseEngine:
    """Create the appropriate engine for the resolved runtime."""
    if runtime.backend == "vllm":
        from annotator.engine.vllm import VLLMEngine

        return VLLMEngine(
            runtime.model_spec, gpu_memory_utilization, max_model_len, max_output_tokens
        )
    elif runtime.backend == "mlx":
        from annotator.engine.mlx import MLXEngine

        return MLXEngine(runtime.model_spec)
    elif runtime.backend == "llama_cpp":
        from annotator.engine.llama_cpp import LlamaCppEngine

        return LlamaCppEngine(runtime.model_spec)
    else:
        raise ValueError(f"Unknown backend: {runtime.backend}")
