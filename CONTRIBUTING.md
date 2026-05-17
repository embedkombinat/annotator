# Contributing to annotator

Thanks for your interest. This document covers what you need to know to get a development environment running and submit changes.

## Development setup

Prerequisites: Python 3.12 or newer, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/embedkombinat/annotator.git
cd annotator
uv sync --all-extras
```

The `--all-extras` flag pulls in the dev dependencies plus all three inference backends (vLLM, MLX, llama.cpp). On platforms that can't install one of them (e.g. MLX on Linux), drop the relevant extra: `uv sync --extra dev --extra vllm`.

## Workflow

- Branch off `main`. Open a pull request when ready; no direct pushes to `main`.
- Keep commits focused. Squash exploratory work before opening the PR.
- The CI gate runs `ruff check`, `ruff format --check`, `mypy`, and `pytest`. Run them locally before pushing.

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .               # confirm zero errors
uv run ruff format --check .      # confirm formatting clean
uv run mypy annotator/
uv run pytest -v
```

We do not accept `# noqa` or `# type: ignore` to silence existing errors. Fix the root cause, including in files you didn't touch. If a lint rule is genuinely wrong for this codebase, call that out explicitly in the PR.

## Adding an inference engine

Engines live under `annotator/engine/` and subclass `BaseEngine` (`annotator/engine/base.py`). Implement `load()` and `label_batch()`. Register the new engine in `annotator/resolver.py` so it can be auto-selected based on detected hardware. Add tests under `tests/`; existing engine tests are a good template.

## Reporting bugs and proposing features

File an issue at https://github.com/embedkombinat/annotator/issues. Include the output of `annotator status --debug` (when implemented) or at least your OS, Python version, and the command that failed.

## Security disclosures

For anything that looks like a real vulnerability, please don't file it as a public issue. Use [GitHub Private Vulnerability Reporting](https://github.com/embedkombinat/annotator/security/advisories/new) or email security@embedkombinat.org.
