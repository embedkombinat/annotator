from __future__ import annotations

from unittest.mock import patch

import pytest

from annotator.errors import ResolverError
from annotator.resolver import resolve

_NO_NVIDIA = patch("annotator.resolver._detect_nvidia", return_value=None)
_NO_APPLE = patch("annotator.resolver._detect_apple_silicon", return_value=None)


class TestResolveNvidia:
    def test_24gb_selects_fp16(self) -> None:
        with (
            patch("annotator.resolver._detect_nvidia", return_value=("RTX 3090", 24.0)),
            _NO_APPLE,
        ):
            rt = resolve()
        assert rt.backend == "vllm"
        assert rt.model_spec.model_id == "Qwen/Qwen2.5-7B-Instruct"
        assert rt.model_spec.quantization is None
        assert rt.gpu_name == "RTX 3090"
        assert rt.gpu_vram_gb == 24.0

    def test_10gb_selects_7b_awq(self) -> None:
        with (
            patch("annotator.resolver._detect_nvidia", return_value=("RTX 3080", 10.0)),
            _NO_APPLE,
        ):
            rt = resolve()
        assert rt.model_spec.model_id == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert rt.model_spec.quantization == "awq"

    def test_5gb_selects_3b_awq(self) -> None:
        with (
            patch("annotator.resolver._detect_nvidia", return_value=("RTX 3060", 5.0)),
            _NO_APPLE,
        ):
            rt = resolve()
        assert rt.model_spec.model_id == "Qwen/Qwen2.5-3B-Instruct-AWQ"

    def test_2gb_raises(self) -> None:
        with (
            patch("annotator.resolver._detect_nvidia", return_value=("GTX 1050", 2.0)),
            _NO_APPLE,
            pytest.raises(ResolverError, match="No vllm model fits"),
        ):
            resolve()


class TestResolveAppleSilicon:
    def test_apple_silicon_backend(self) -> None:
        with (
            _NO_NVIDIA,
            patch(
                "annotator.resolver._detect_apple_silicon",
                return_value=("Apple M2 Pro", 16.0),
            ),
        ):
            rt = resolve()
        assert rt.backend == "mlx"
        assert rt.gpu_name == "Apple M2 Pro"

    def test_16gb_selects_7b_4bit(self) -> None:
        with (
            _NO_NVIDIA,
            patch(
                "annotator.resolver._detect_apple_silicon",
                return_value=("Apple M2 Pro", 16.0),
            ),
        ):
            rt = resolve()
        # 16 * 0.9 = 14.4 >= 6.0 -> 7B-4bit
        assert rt.model_spec.model_id == "mlx-community/Qwen2.5-7B-Instruct-4bit"

    def test_8gb_selects_7b_4bit(self) -> None:
        with (
            _NO_NVIDIA,
            patch(
                "annotator.resolver._detect_apple_silicon",
                return_value=("Apple M1", 8.0),
            ),
        ):
            rt = resolve()
        # 8 * 0.9 = 7.2 >= 6.0 -> 7B-4bit still fits
        assert rt.model_spec.model_id == "mlx-community/Qwen2.5-7B-Instruct-4bit"

    def test_4gb_selects_1_5b_4bit(self) -> None:
        with (
            _NO_NVIDIA,
            patch(
                "annotator.resolver._detect_apple_silicon",
                return_value=("Apple M1", 4.0),
            ),
        ):
            rt = resolve()
        # 4 * 0.9 = 3.6 < 4.0 (3B) but >= 2.0 (1.5B)
        assert rt.model_spec.model_id == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


class TestResolveCPU:
    def test_cpu_only(self) -> None:
        with _NO_NVIDIA, _NO_APPLE:
            rt = resolve()
        assert rt.backend == "llama_cpp"
        assert rt.gpu_name is None
        assert rt.model_spec.model_id == "Qwen/Qwen2.5-3B-Instruct-GGUF"


class TestResolveOverrides:
    def test_override_model(self) -> None:
        with (
            patch("annotator.resolver._detect_nvidia", return_value=("RTX 3090", 24.0)),
            _NO_APPLE,
        ):
            rt = resolve(override_model="my-org/custom-model")
        assert rt.model_spec.model_id == "my-org/custom-model"
        assert rt.backend == "vllm"

    def test_override_backend(self) -> None:
        with (
            patch("annotator.resolver._detect_nvidia", return_value=("RTX 3090", 24.0)),
            _NO_APPLE,
        ):
            rt = resolve(override_backend="mlx")
        assert rt.backend == "mlx"

    def test_override_model_and_quantization(self) -> None:
        with _NO_NVIDIA, _NO_APPLE:
            rt = resolve(
                override_model="custom/model",
                override_quantization="gptq",
                override_backend="vllm",
            )
        assert rt.model_spec.model_id == "custom/model"
        assert rt.model_spec.quantization == "gptq"
        assert rt.backend == "vllm"

    def test_gpu_memory_utilization_affects_selection(self) -> None:
        with (
            patch("annotator.resolver._detect_nvidia", return_value=("RTX 3090", 24.0)),
            _NO_APPLE,
        ):
            rt = resolve(gpu_memory_utilization=0.3)
        # 24 * 0.3 = 7.2 < 8.0 (7B AWQ) but >= 4.0 (3B AWQ)
        assert rt.model_spec.model_id == "Qwen/Qwen2.5-3B-Instruct-AWQ"


class TestBackendAlias:
    def test_cpu_alias_maps_to_llama_cpp(self) -> None:
        """The CLI/docs advertise --backend cpu; the registry key is llama_cpp."""
        from unittest.mock import patch

        from annotator.resolver import resolve

        with (
            patch("annotator.resolver._detect_nvidia", return_value=None),
            patch("annotator.resolver._detect_apple_silicon", return_value=None),
        ):
            runtime = resolve(override_backend="cpu")
        assert runtime.backend == "llama_cpp"
        assert runtime.model_spec.backend == "llama_cpp"
