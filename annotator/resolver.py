"""Hardware detection and model selection."""

from __future__ import annotations

import os
import platform
import subprocess

from pydantic import BaseModel, ConfigDict

from annotator.errors import ResolverError


class ModelSpec(BaseModel):
    # protected_namespaces=() suppresses pydantic's "model_" field-name warning —
    # model_id is domain-intended here (a HuggingFace model identifier), not a
    # collision with pydantic's own `model_*` methods.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_id: str
    quantization: str | None
    min_vram_gb: float
    download_gb: float
    backend: str
    revision: str


class ResolvedRuntime(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_spec: ModelSpec
    gpu_name: str | None
    gpu_vram_gb: float | None
    backend: str


# Order matters: _select_model picks the FIRST spec whose min_vram_gb fits,
# so keep each backend's list sorted largest/most-capable first.
REGISTRY: dict[str, list[ModelSpec]] = {
    "vllm": [
        ModelSpec(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            quantization=None,
            min_vram_gb=18.0,
            download_gb=14.0,
            backend="vllm",
            revision="main",
        ),
        ModelSpec(
            model_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
            quantization="awq",
            min_vram_gb=8.0,
            download_gb=4.5,
            backend="vllm",
            revision="main",
        ),
        ModelSpec(
            model_id="Qwen/Qwen2.5-3B-Instruct-AWQ",
            quantization="awq",
            min_vram_gb=4.0,
            download_gb=2.0,
            backend="vllm",
            revision="main",
        ),
    ],
    "mlx": [
        ModelSpec(
            model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            quantization="4bit",
            min_vram_gb=6.0,
            download_gb=4.0,
            backend="mlx",
            revision="main",
        ),
        ModelSpec(
            model_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
            quantization="4bit",
            min_vram_gb=4.0,
            download_gb=2.0,
            backend="mlx",
            revision="main",
        ),
        ModelSpec(
            model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            quantization="4bit",
            min_vram_gb=2.0,
            download_gb=1.0,
            backend="mlx",
            revision="main",
        ),
    ],
    "llama_cpp": [
        ModelSpec(
            model_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            quantization="Q4_K_M",
            min_vram_gb=0,
            download_gb=2.0,
            backend="llama_cpp",
            revision="main",
        ),
        ModelSpec(
            model_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            quantization="Q4_K_M",
            min_vram_gb=0,
            download_gb=1.0,
            backend="llama_cpp",
            revision="main",
        ),
    ],
}


BACKEND_ALIASES = {"cpu": "llama_cpp"}


def _detect_nvidia() -> tuple[str, float] | None:
    """Detect NVIDIA GPU via pynvml. Returns (name, vram_gb) or None."""
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)
        pynvml.nvmlShutdown()
        return (name, vram_gb)
    except Exception:
        return None


def _detect_apple_silicon() -> tuple[str, float] | None:
    """Detect Apple Silicon. Returns (chip_name, memory_gb) or None."""
    if platform.system() != "Darwin" or platform.processor() != "arm":
        return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        memory_gb = (page_size * page_count) / (1024**3)
    except (ValueError, OSError):
        return None

    # Try to get the chip name from sysctl
    chip_name = "Apple Silicon"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            chip_name = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return (chip_name, memory_gb)


def _select_model(backend: str, available_vram_gb: float) -> ModelSpec:
    """Select the best model that fits available memory."""
    models = REGISTRY.get(backend, [])
    for model in models:
        if model.min_vram_gb <= available_vram_gb:
            return model
    raise ResolverError(
        f"No {backend} model fits in {available_vram_gb:.1f} GB. "
        f"Minimum required: {models[-1].min_vram_gb:.1f} GB."
        if models
        else f"No models registered for backend '{backend}'."
    )


def resolve(
    override_model: str | None = None,
    override_quantization: str | None = None,
    override_backend: str | None = None,
    gpu_memory_utilization: float = 0.9,
) -> ResolvedRuntime:
    """Detect hardware, select backend and model.

    Detection order:
      1. NVIDIA GPU -> vLLM backend
      2. Apple Silicon -> MLX backend
      3. No GPU -> llama.cpp backend

    Override flags bypass auto-detection.
    """
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    backend: str

    # Detect hardware
    nvidia = _detect_nvidia()
    apple = _detect_apple_silicon()

    if override_backend:
        # The CLI and docs advertise "cpu"; the registry/engine key is llama_cpp.
        backend = BACKEND_ALIASES.get(override_backend, override_backend)
        if nvidia:
            gpu_name, gpu_vram_gb = nvidia
        elif apple:
            gpu_name, gpu_vram_gb = apple
    elif nvidia:
        backend = "vllm"
        gpu_name, gpu_vram_gb = nvidia
    elif apple:
        backend = "mlx"
        gpu_name, gpu_vram_gb = apple
    else:
        backend = "llama_cpp"

    # Select model
    if override_model:
        model_spec = ModelSpec(
            model_id=override_model,
            quantization=override_quantization,
            min_vram_gb=0,
            download_gb=0,
            backend=backend,
            revision="main",
        )
    else:
        effective_vram = (gpu_vram_gb or 0) * gpu_memory_utilization
        model_spec = _select_model(backend, effective_vram)

    return ResolvedRuntime(
        model_spec=model_spec,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        backend=backend,
    )
