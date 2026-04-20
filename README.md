<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://embedkombinat.github.io/embed-kombinat.github.io/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://embedkombinat.github.io/embed-kombinat.github.io/logo-light.svg">
  <img alt="EmbedKombinat" src="https://embedkombinat.github.io/embed-kombinat.github.io/logo-dark.svg" width="420">
</picture>

<br/>
<br/>

**Distributed annotation worker for [EmbedKombinat](https://embedkombinat.github.io/embed-kombinat.github.io/index.html)**

Run local LLM inference on your hardware to label query-document pairs for open embedding model training.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/test-ann?color=blue)](https://pypi.org/project/test-ann/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](http://mypy-lang.org/)

---

[Website](https://embedkombinat.github.io/embed-kombinat.github.io/index.html) | [Getting Started](#getting-started) | [Models](#supported-models) | [Leaderboard](#-annotator-leaderboard) | [Contributing](#contributing)

</div>

## What is this?

The **annotator** is a headless labeling worker that runs on contributor hardware. It claims batches of unlabeled (query, document) pairs from the [kombinat](https://embedkombinat.github.io/embed-kombinat.github.io/index.html) server, scores relevance using a local LLM, and submits annotations back — all without sending your data to any third-party API.

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

1. **Authenticate** via GitHub OAuth device flow (works headless, over SSH, in Docker)
2. **Detect hardware** — NVIDIA GPU, Apple Silicon, or CPU-only
3. **Download & load** the best-fit LLM for your hardware from HuggingFace
4. **Claim → Label → Submit** in streaming micro-batches (lose at most one chunk on interrupt)

Each pair gets a relevance score from **0** (not relevant) to **3** (highly relevant) with a short reasoning.

## Getting Started

### Install

```bash
# NVIDIA GPU
pip install test-ann[vllm]

# Apple Silicon (M1/M2/M3/M4)
pip install test-ann[mlx]

# CPU-only
pip install test-ann[cpu]
```

Or run without installing:

```bash
uvx --from "test-ann[mlx]" annotator run
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

## Annotator Leaderboard

Top contributors by total annotations submitted. Updated in real-time by the kombinat server.

| Rank | Contributor | Annotations | Hardware | Avg Score | Streak |
|:---:|-------------|:---:|----------|:---:|:---:|
| :trophy: | **@embedmaster3000** | 284,192 | A100 80GB | 0.97 | 42 days |
| :2nd_place_medal: | **@silicon_sarah** | 201,847 | M4 Max 128GB | 0.95 | 38 days |
| :3rd_place_medal: | **@gpu_goes_brrr** | 156,330 | RTX 4090 | 0.94 | 29 days |
| 4 | @label_ninja | 98,412 | RTX 3090 | 0.93 | 15 days |
| 5 | @the_annotator | 87,201 | M3 Pro 36GB | 0.92 | 21 days |
| 6 | @qwen_whisperer | 64,553 | RTX 4080 | 0.91 | 12 days |
| 7 | @cpu_chad | 42,100 | Ryzen 9 7950X | 0.89 | 33 days |
| 8 | @macbook_warrior | 38,771 | M2 Ultra 192GB | 0.93 | 8 days |
| 9 | @batch_queen | 31,204 | 2x RTX 3080 | 0.90 | 17 days |
| 10 | @open_source_larry | 24,889 | M1 Pro 16GB | 0.88 | 45 days |

> Want to see your name here? `pip install test-ann[mlx] && annotator run`

## Contributing

```bash
# Clone and install dev dependencies
git clone https://github.com/embedkombinat/annotator.git
cd annotator
pip install -e ".[dev,mlx]"  # or .[dev,vllm] for NVIDIA

# Run checks
ruff check .
mypy annotator/
pytest
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with care by the [EmbedKombinat](https://embedkombinat.github.io/embed-kombinat.github.io/index.html) community.

</div>
