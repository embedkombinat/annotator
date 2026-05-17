"""Annotator configuration via environment variables."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExitCode(IntEnum):
    """CLI exit codes."""

    SUCCESS = 0
    AUTH_FAILURE = 1
    NO_COMPATIBLE_HARDWARE = 2
    MODEL_LOADING_FAILED = 3
    KOMBINAT_UNREACHABLE = 4
    UNRECOVERABLE = 5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANNOTATOR_",
        env_file=Path.cwd() / ".env",
        extra="ignore",
    )

    # Override with ANNOTATOR_KOMBINAT_URL for local dev against a non-production hub.
    kombinat_url: str = "https://kombinat-production.up.railway.app"

    batch_size: int = 100
    chunk_size: int = 50
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    max_output_tokens: int = 256

    annotator_home: Path = Path.home() / ".annotator"
