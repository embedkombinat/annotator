"""Hardware detection and model selection."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

from annotator.errors import ResolverError


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    quantization: str | None
    min_vram_gb: float
    download_gb: float
    backend: str
    revision: str


@dataclass
class ResolvedRuntime:
    model_spec: ModelSpec
    gpu_name: str | None
    gpu_vram_gb: float | None
    backend: str


REGISTRY: dict[str, list[ModelSpec]] = {
    "vllm": [
        ModelSpec("Qwen/Qwen2.5-7B-Instruct", None, 18.0, 14.0, "vllm", "main"),
        ModelSpec("Qwen/Qwen2.5-7B-Instruct-AWQ", "awq", 8.0, 4.5, "vllm", "main"),
        ModelSpec("Qwen/Qwen2.5-3B-Instruct-AWQ", "awq", 4.0, 2.0, "vllm", "main"),
    ],
    "mlx": [
        ModelSpec("mlx-community/Qwen2.5-7B-Instruct-4bit", "4bit", 6.0, 4.0, "mlx", "main"),
        ModelSpec("mlx-community/Qwen2.5-3B-Instruct-4bit", "4bit", 4.0, 2.0, "mlx", "main"),
        ModelSpec("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "4bit", 2.0, 1.0, "mlx", "main"),
    ],
    "llama_cpp": [
        ModelSpec("Qwen/Qwen2.5-3B-Instruct-GGUF", "Q4_K_M", 0, 2.0, "llama_cpp", "main"),
        ModelSpec("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "Q4_K_M", 0, 1.0, "llama_cpp", "main"),
    ],
}


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
        backend = override_backend
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
