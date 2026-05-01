from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from annotator.engine.base import LabelingInput
from annotator.resolver import ModelSpec, ResolvedRuntime


def _make_runtime(backend: str = "vllm") -> ResolvedRuntime:
    return ResolvedRuntime(
        model_spec=ModelSpec(
            model_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
            quantization="awq",
            min_vram_gb=8.0,
            download_gb=4.5,
            backend=backend,
            revision="main",
        ),
        gpu_name="RTX 3090",
        gpu_vram_gb=24.0,
        backend=backend,
    )


def _make_pairs(n: int = 2) -> list[LabelingInput]:
    return [
        LabelingInput(
            pair_id=f"pair-{i}",
            query_text=f"query {i}",
            doc_text=f"document {i}",
        )
        for i in range(n)
    ]


def _make_vllm_output(text: str = '{"label": 2, "reasoning": "relevant"}') -> MagicMock:
    """Create a mock vLLM output object."""
    output = MagicMock()
    output.outputs = [MagicMock()]
    output.outputs[0].text = text
    output.outputs[0].token_ids = list(range(20))
    output.prompt_token_ids = list(range(100))
    return output


@pytest.fixture()
def mock_vllm() -> MagicMock:
    """Provide a mocked vllm module in sys.modules."""
    mock_mod = MagicMock()
    mock_sampling = MagicMock()
    mock_mod.SamplingParams = MagicMock()
    mock_mod.LLM = MagicMock()
    mock_sampling.StructuredOutputsParams = MagicMock()

    with patch.dict(
        sys.modules,
        {"vllm": mock_mod, "vllm.sampling_params": mock_sampling},
    ):
        yield mock_mod


@pytest.fixture()
def mock_mlx_lm() -> MagicMock:
    """Provide a mocked mlx_lm module in sys.modules."""
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"mlx_lm": mock_mod}):
        yield mock_mod


class TestCreateEngine:
    def test_create_vllm_engine(self, mock_vllm: MagicMock) -> None:
        from annotator.engine import create_engine

        engine = create_engine(_make_runtime("vllm"))
        assert type(engine).__name__ == "VLLMEngine"

    def test_create_mlx_engine(self, mock_mlx_lm: MagicMock) -> None:
        from annotator.engine import create_engine

        engine = create_engine(_make_runtime("mlx"))
        assert type(engine).__name__ == "MLXEngine"

    def test_create_llama_cpp_engine(self) -> None:
        from annotator.engine import create_engine

        engine = create_engine(_make_runtime("llama_cpp"))
        assert type(engine).__name__ == "LlamaCppEngine"

    def test_unknown_backend_raises(self) -> None:
        from annotator.engine import create_engine

        with pytest.raises(ValueError, match="Unknown backend"):
            create_engine(_make_runtime("unknown"))


class TestVLLMEngine:
    def _make_engine(self, mock_vllm: MagicMock) -> tuple[MagicMock, object]:
        """Create a VLLMEngine with a mocked LLM."""
        from annotator.engine.vllm import VLLMEngine

        spec = _make_runtime("vllm").model_spec
        engine = VLLMEngine(spec, gpu_memory_utilization=0.9)
        mock_llm = MagicMock()
        engine.llm = mock_llm
        return mock_llm, engine

    def test_label_batch_calls_chat(self, mock_vllm: MagicMock) -> None:
        mock_llm, engine = self._make_engine(mock_vllm)
        pairs = _make_pairs(2)
        mock_llm.chat.return_value = [_make_vllm_output(), _make_vllm_output()]

        results = engine.label_batch(pairs)  # type: ignore[union-attr]
        assert mock_llm.chat.call_count == 1
        assert len(results) == 2

    def test_label_batch_parses_responses(self, mock_vllm: MagicMock) -> None:
        mock_llm, engine = self._make_engine(mock_vllm)
        pairs = _make_pairs(2)
        mock_llm.chat.return_value = [
            _make_vllm_output('{"label": 0, "reasoning": "not relevant"}'),
            _make_vllm_output('{"label": 3, "reasoning": "highly relevant"}'),
        ]

        results = engine.label_batch(pairs)  # type: ignore[union-attr]
        assert results[0].label == 0
        assert results[0].reasoning == "not relevant"
        assert results[1].label == 3
        assert results[1].reasoning == "highly relevant"

    def test_label_batch_token_counts(self, mock_vllm: MagicMock) -> None:
        mock_llm, engine = self._make_engine(mock_vllm)
        pairs = _make_pairs(1)
        mock_llm.chat.return_value = [_make_vllm_output()]

        results = engine.label_batch(pairs)  # type: ignore[union-attr]
        assert results[0].input_tokens == 100
        assert results[0].output_tokens == 20

    def test_label_batch_response_hash(self, mock_vllm: MagicMock) -> None:
        mock_llm, engine = self._make_engine(mock_vllm)
        pairs = _make_pairs(1)
        mock_llm.chat.return_value = [_make_vllm_output()]

        results = engine.label_batch(pairs)  # type: ignore[union-attr]
        assert results[0].raw_response_hash.startswith("sha256:")

    def test_retry_on_parse_failure(self, mock_vllm: MagicMock) -> None:
        mock_llm, engine = self._make_engine(mock_vllm)
        pairs = _make_pairs(2)

        mock_llm.chat.side_effect = [
            [
                _make_vllm_output("not json"),
                _make_vllm_output('{"label": 2, "reasoning": "ok"}'),
            ],
            [_make_vllm_output('{"label": 1, "reasoning": "retry ok"}')],
        ]

        results = engine.label_batch(pairs)  # type: ignore[union-attr]
        assert len(results) == 2
        assert mock_llm.chat.call_count == 2

    def test_drop_on_double_failure(self, mock_vllm: MagicMock) -> None:
        mock_llm, engine = self._make_engine(mock_vllm)
        pairs = _make_pairs(2)

        mock_llm.chat.side_effect = [
            [
                _make_vllm_output("not json"),
                _make_vllm_output('{"label": 2, "reasoning": "ok"}'),
            ],
            [_make_vllm_output("still not json")],
        ]

        results = engine.label_batch(pairs)  # type: ignore[union-attr]
        assert len(results) == 1
        assert results[0].pair_id == "pair-1"

    def test_info(self, mock_vllm: MagicMock) -> None:
        from annotator.engine.vllm import VLLMEngine

        spec = _make_runtime("vllm").model_spec
        engine = VLLMEngine(spec)
        info = engine.info()
        assert info.model_id == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert info.quantization == "awq"
        assert info.backend == "vllm"


class TestMLXEngine:
    def _make_engine(self) -> tuple[MagicMock, MagicMock, object]:
        """Create an MLXEngine with mocked model and tokenizer."""
        from annotator.engine.mlx import MLXEngine

        spec = ModelSpec(
            model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
            quantization="4bit",
            min_vram_gb=4.0,
            download_gb=2.0,
            backend="mlx",
            revision="main",
        )
        engine = MLXEngine(spec)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = list(range(50))
        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        return mock_model, mock_tokenizer, engine

    def test_label_batch_sequential(self, mock_mlx_lm: MagicMock) -> None:
        _, _, engine = self._make_engine()
        pairs = _make_pairs(2)

        mock_mlx_lm.generate.return_value = '{"label": 2, "reasoning": "ok"}'
        results = engine.label_batch(pairs)  # type: ignore[union-attr]

        assert len(results) == 2
        assert mock_mlx_lm.generate.call_count == 2

    def test_label_batch_retry(self, mock_mlx_lm: MagicMock) -> None:
        _, _, engine = self._make_engine()
        pairs = _make_pairs(1)

        mock_mlx_lm.generate.side_effect = [
            "not json",
            '{"label": 1, "reasoning": "retry ok"}',
        ]
        results = engine.label_batch(pairs)  # type: ignore[union-attr]

        assert len(results) == 1
        assert results[0].label == 1
        assert mock_mlx_lm.generate.call_count == 2

    def test_label_batch_drop_on_double_failure(self, mock_mlx_lm: MagicMock) -> None:
        _, _, engine = self._make_engine()
        pairs = _make_pairs(1)

        mock_mlx_lm.generate.side_effect = ["not json", "still not json"]
        results = engine.label_batch(pairs)  # type: ignore[union-attr]

        assert len(results) == 0

    def test_info(self, mock_mlx_lm: MagicMock) -> None:
        from annotator.engine.mlx import MLXEngine

        spec = ModelSpec(
            model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
            quantization="4bit",
            min_vram_gb=4.0,
            download_gb=2.0,
            backend="mlx",
            revision="main",
        )
        engine = MLXEngine(spec)
        info = engine.info()
        assert info.model_id == "mlx-community/Qwen2.5-3B-Instruct-4bit"
        assert info.quantization == "4bit"
        assert info.backend == "mlx"


class TestLlamaCppStub:
    def test_load_raises(self) -> None:
        from annotator.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine(_make_runtime("llama_cpp").model_spec)
        with pytest.raises(NotImplementedError, match="llama.cpp"):
            engine.load()

    def test_label_batch_raises(self) -> None:
        from annotator.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine(_make_runtime("llama_cpp").model_spec)
        with pytest.raises(NotImplementedError, match="llama.cpp"):
            engine.label_batch([])
