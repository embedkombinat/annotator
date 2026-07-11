# CLAUDE.md — annotator

This file provides guidance to Claude Code (claude.ai/code) when working in the **annotator** sub-repo (headless CLI labeling worker, Python 3.12+, Typer, vLLM/MLX/llama.cpp).

## Build and dev commands

```bash
pip install -e ".[dev]"          # or: uv sync --all-extras

# Lint, format, type check
uv run ruff check .
uv run ruff format --check .
uv run mypy annotator/

# Tests
uv run pytest -v

# Single test
uv run pytest tests/test_runner.py::TestRunnerBasicLoop -v
```

## Linting policy

CI runs `ruff check .` and `ruff format --check .` as hard gates. Both must pass on every push.

**Always fix lint errors — never ignore them.** When you finish editing Python files, run:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .              # confirm zero errors
uv run ruff format --check .     # confirm formatting clean
```

Fix every error surfaced, including in files you did not touch directly. Do not add `# noqa` suppressions or extend `tool.ruff.lint.ignore` to silence pre-existing errors — collapse nested `with` into a single statement, rename unused variables to `_`, shorten long lines, etc. The only acceptable reason to touch ignore lists is when a rule is genuinely wrong for this codebase, and that change must be called out explicitly.

Same rule for `uv run mypy annotator/` — fix the types, don't add `# type: ignore`.

## Architecture

Typer CLI with polymorphic inference backends. Heavy modules (vllm, mlx_lm, llama_cpp) are lazy-imported to avoid startup delay.

- `annotator/cli.py` — Commands: default (run loop), login, status, logout
- `annotator/runner.py` — Orchestration: auth → resolve → load engine → claim/label/submit loop. SIGINT/SIGTERM: first = graceful (finish chunk), second within 3s = forced exit.
- `annotator/resolver.py` — Hardware detection (NVIDIA via pynvml → Apple Silicon via sysctl → CPU fallback), model selection from registry by available VRAM
- `annotator/engine/` — `BaseEngine` ABC with `load()` and `label_batch()`. Implementations: `VLLMEngine` (guided JSON decoding, batch), `MLXEngine` (sequential), `LlamaCppEngine` (stub)
- `annotator/labeler.py` — Prompt template (system + user with XML-wrapped query/doc), JSON response parsing, SHA256 hashing
- `annotator/client.py` — HTTPX client with exponential backoff (3 retries for 5xx, increasing wait on 204 no-pairs)
- `annotator/auth.py` — GitHub OAuth Device Flow (CLI prints a user code, polls GitHub for the token; no callback server, works on headless/SSH hosts), kombinat JWT persisted to `~/.annotator/auth.json` with `0600` permissions

### Key conventions

- **Pydantic everywhere** (BaseSettings, API schemas, internal DTOs). No dataclasses.
- **Config via env vars**, prefixed with `ANNOTATOR_` (e.g. `ANNOTATOR_KOMBINAT_URL`).
- **Tests do not use pytest-asyncio.** All tests are synchronous.
