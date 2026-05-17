<div align="center">

<h1>embedkombinat / annotator</h1>

**Distributed annotation worker for [embedkombinat](https://embedkombinat.github.io)**

Run local LLM inference on your hardware to label query-document pairs for open embedding model training.

[![CI](https://github.com/embedkombinat/annotator/actions/workflows/ci.yml/badge.svg)](https://github.com/embedkombinat/annotator/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/embedkombinat-annotator?color=blue)](https://pypi.org/project/embedkombinat-annotator/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)

---

[Website](https://embedkombinat.github.io) | [Getting Started](#getting-started) | [Models](#supported-models) | [Live Leaderboard](https://embedkombinat.github.io#leaderboard) | [Contributing](#contributing)

</div>

## What is this?

The **annotator** is a headless labeling worker that runs on contributor hardware. It claims batches of unlabeled (query, document) pairs from the [kombinat](https://github.com/embedkombinat/kombinat) server, scores relevance using a local LLM, and submits annotations back — all without sending your data to any third-party API.

```
┌─────────────┐     claim batch     ┌─────────────┐
│  kombinat   │ ◄────────────────── │  annotator  │
│   server    │ ──────────────────► │  (your hw)  │
│             │   (query, doc) pairs│             │
│             │                     │  ┌────────┐ │
│             │   submit labels     │  │ Local  │ │
│             │ ◄────────────────── │  │  LLM   │ │
└─────────────┘                     │  └────────┘ │
                                    └─────────────┘
```

### How it works

1. **Authenticate** via GitHub Device Flow — the CLI prints a short code, you enter it at `https://github.com/login/device` on any browser (laptop, phone, anywhere). Works on remote SSH'd hosts (Runpod, Lambda, EC2) and inside Docker without any port forwarding.
2. **Detect hardware** — NVIDIA GPU, Apple Silicon, or CPU-only
3. **Download & load** the best-fit LLM for your hardware from HuggingFace
4. **Claim → Label → Submit** in streaming micro-batches (lose at most one chunk on interrupt)

Each pair gets a relevance score from **0** (not relevant) to **3** (highly relevant) with a short reasoning.

## Getting Started

### Install

```bash
# NVIDIA GPU
pip install embedkombinat-annotator[vllm]

# Apple Silicon (M1/M2/M3/M4)
pip install embedkombinat-annotator[mlx]

# CPU-only
pip install embedkombinat-annotator[cpu]
```

Or run without installing:

```bash
uvx --from "embedkombinat-annotator[mlx]" annotator run
```

### Authenticate

```bash
annotator login
```

### Run

```bash
# Starts labeling (will prompt login if not authenticated)
annotator run
```

### Docker (NVIDIA)

```bash
docker compose up
```

## Supported Models

The annotator auto-selects the best model for your hardware. You can override with `--model` and `--backend`.

### NVIDIA GPU (vLLM)

| Model | Quantization | VRAM | Download |
|-------|:---:|:---:|:---:|
| `Qwen/Qwen2.5-7B-Instruct` | — | 18 GB | 14 GB |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | AWQ | 8 GB | 4.5 GB |
| `Qwen/Qwen2.5-3B-Instruct-AWQ` | AWQ | 4 GB | 2 GB |

### Apple Silicon (MLX)

| Model | Quantization | Memory | Download |
|-------|:---:|:---:|:---:|
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | 4-bit | 6 GB | 4 GB |
| `mlx-community/Qwen2.5-3B-Instruct-4bit` | 4-bit | 4 GB | 2 GB |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | 4-bit | 2 GB | 1 GB |

### CPU (llama.cpp)

| Model | Quantization | Download |
|-------|:---:|:---:|
| `Qwen/Qwen2.5-3B-Instruct-GGUF` | Q4_K_M | 2 GB |
| `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | Q4_K_M | 1 GB |

## CLI Reference

```
Usage: annotator [COMMAND] [OPTIONS]

Commands:
  run      Start the labeling loop (default)
  login    Authenticate via GitHub
  status   Show contributor profile and stats
  logout   Remove stored credentials

Options (run):
  --batch-size INT             Pairs per batch (default: 100, max: 500)
  --model TEXT                 Override model ID
  --quantization TEXT          Override quantization
  --backend [vllm|mlx|cpu]    Override backend
  --gpu-memory-utilization FLOAT  GPU fraction (default: 0.9)
  --dry-run                    Resolve hardware & model, then exit
```

## Leaderboard

See the [live leaderboard at embedkombinat.github.io](https://embedkombinat.github.io#leaderboard).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and workflow. The short version:

```bash
git clone https://github.com/embedkombinat/annotator.git
cd annotator
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy annotator/
uv run pytest -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with care by the [embedkombinat](https://embedkombinat.github.io) community.

</div>
