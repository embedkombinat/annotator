# annotator — Product Requirements Document

**Version**: 0.3.0
**Date**: 2026-03-28
**Status**: Draft
**Authors**: EmbedCollective

---

## 0. Context

This document specifies the `annotator` — the contributor-facing component of EmbedCollective's distributed annotation infrastructure. It is the Docker container and CLI that contributors run on their own hardware to label (query, document) pairs for embedding model training.

The annotator is one of three repositories in the `embedcollective` GitHub organization:

- **kombinat** — the coordination server (assigns work, validates submissions, tracks reputation)
- **annotator** (this repo) — the Docker container + CLI that contributors run locally
- **embed-collective.github.io** — the landing page with live progress stats

The annotator talks exclusively to kombinat. It claims batches of unlabeled pairs, runs local LLM inference to judge relevance, and submits the results. All inference happens on the contributor's hardware — only pair IDs and labels are uploaded.

For the full specification of kombinat's API, data model, and architecture, see `kombinat-prd.md`.

---

## 1. What annotator is

The annotator is a headless labeling worker. It runs a local LLM to judge whether a document is relevant to a query, producing a 0–3 relevance score for each pair. It is designed to run unattended — a contributor starts it, and it labels pairs until stopped.

### 1.1 Core responsibilities

- Authenticate the contributor with kombinat via GitHub OAuth (device authorization flow)
- Detect available hardware (NVIDIA GPU, Apple Silicon, CPU-only) and select the best-fitting model and inference backend
- Download and cache the selected model from HuggingFace Hub
- Claim batches of unlabeled pairs from kombinat
- Run batch LLM inference with structured JSON output
- Validate model responses against the output schema
- Submit annotations to kombinat in streaming micro-batches (every chunk, not at batch end)
- Handle graceful shutdown — submit completed work, lose at most one chunk (~50 pairs)
- Report progress to the contributor via terminal output

### 1.2 What annotator is NOT

- Not an interactive annotation tool — there is no UI, no human-in-the-loop
- Not a model serving system — inference runs in offline batch mode, not as an API server
- Not responsible for quality control — that's kombinat's validator
- Not a training pipeline — it produces labels, not models

---

## 2. Architecture

### 2.1 The main loop

```
┌──────────────────────────────────────────────────────────┐
│  annotator (single process)                              │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │  cli.py    │  │resolver.py │  │  engine/           │ │
│  │            │  │            │  │                    │ │
│  │  run       │  │  detect hw │  │  base.py (ABC)    │ │
│  │  login     │  │  select    │  │  vllm.py          │ │
│  │  status    │  │   backend  │  │  mlx.py           │ │
│  │  logout    │  │  select    │  │  llama_cpp.py     │ │
│  │            │  │   model    │  │                    │ │
│  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘ │
│        │               │                    │            │
│  ┌─────┴──────┐        │         ┌─────────┴──────────┐ │
│  │ auth.py    │        │         │  labeler.py        │ │
│  │            │        │         │                    │ │
│  │ device flow│        │         │  prompt template   │ │
│  │ JWT store  │        │         │  response parsing  │ │
│  └─────┬──────┘        │         └────────────────────┘ │
│        │               │                                │
│  ┌─────┴──────────────────────────────────────────────┐  │
│  │  client.py — HTTP client to kombinat               │  │
│  │  POST /v1/auth/device  │  POST /v1/batches/claim   │  │
│  │  POST /v1/annotations  │  GET /v1/contributors/me  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                           │
                    HTTPS to kombinat
```

### 2.2 Data flow for one batch cycle

```
1. POST /v1/batches/claim {"size": 100}
   ← {batch_id, expires_at, pairs: [{pair_id, query_text, doc_text, ...}]}

2. For each chunk of 50 pairs:
   a. Build chat prompt for each pair: system + user message with query and document
   b. Feed all 50 to inference engine as a batch
   c. Engine returns raw text outputs
   d. Parse each output as JSON → validate LLMResponse (label in 0-3, reasoning non-empty)
   e. On validation failure: retry that pair once. If still fails, skip it.
   f. Wrap valid responses with metadata (pair_id, token counts, response hash)

   g. POST /v1/annotations — submit this chunk immediately
      ← {accepted, rejected, honeypot_accuracy, ...}

3. Repeat step 2 for next chunk. If shutdown requested, stop after current chunk.

4. When all chunks submitted (or shutdown), claim next batch. Repeat from step 1.
```

### 2.3 Key architectural decisions

- **Offline batch inference, not a server**: No HTTP between the annotator and the inference engine. The engine loads the model in-process and runs batch inference directly. No separate process to manage.
- **Streaming submission**: Annotations are submitted to kombinat after every chunk (~50 pairs), not after the entire batch. Progress is never more than one chunk behind, and landing page stats update in near real-time.
- **Pluggable engine backends**: The engine is an abstract interface (`BaseEngine`) with three implementations (vLLM, MLX, llama.cpp), selected automatically based on detected hardware. All other code is backend-agnostic.
- **Single resolver**: Hardware detection and model selection are one operation in `resolver.py`. Input: nothing (or CLI overrides). Output: a `ModelSpec` with the exact model to run and which backend to use. No intermediate `GPUInfo` struct that exists without a consumer.
- **Device authorization flow for auth**: No localhost server, no port mapping. Works inside Docker containers, over SSH, on headless machines.
- **Single directory for all persistent state**: Auth token, model weights, and logs all live under `~/.annotator/`, mapped into Docker via one volume.
- **Auto-login on first run**: `annotator run` detects missing credentials and triggers the login flow inline.

---

## 3. Contributor experience

### 3.1 Installation paths

There are two ways to install the annotator. Docker is recommended for NVIDIA GPU users because vLLM's CUDA dependencies are notoriously difficult to install correctly. pip is the universal path that works on any hardware.

**Why Docker exists**: vLLM compiles custom CUDA kernels at install time. It requires a specific CUDA toolkit version, a matching PyTorch build, a C++ compiler, and sometimes specific NVIDIA driver versions. If any are mismatched, `pip install vllm` fails with cryptic compilation errors. Docker eliminates this — the `vllm/vllm-openai` base image ships with everything pre-compiled. For Mac and CPU-only users, `pip install` works reliably because MLX and llama-cpp-python don't have this dependency complexity.

**Docker (NVIDIA GPU — recommended)**:

```bash
curl -fsSL https://embedcollective.dev/docker-compose.yml > docker-compose.yml
docker compose run annotator
```

The `docker-compose.yml` bakes in GPU access, volume mounts, and shared memory:

```yaml
services:
  annotator:
    image: ghcr.io/embedcollective/annotator:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ~/.annotator:/root/.annotator
    shm_size: '4gb'
    stdin_open: true
    tty: true
```

For background runs: `docker compose up -d`, `docker compose logs -f` to watch, `docker compose down` to stop.

**pip — Mac with Apple Silicon**:

```bash
pip install "annotator[mlx]"
annotator
```

The `[mlx]` extra installs `mlx-lm`. The resolver detects Apple Silicon and uses the MLX backend automatically. No Docker needed — MLX installs cleanly via pip.

**pip — CPU-only (any platform)**:

```bash
pip install "annotator[cpu]"
annotator
```

The `[cpu]` extra installs `llama-cpp-python`. The resolver detects no GPU and falls back to llama.cpp. Slower (~3 pairs/min) but works on any machine.

**pip — NVIDIA GPU (for users who prefer pip over Docker)**:

```bash
pip install "annotator[vllm]"
annotator
```

Requires a working CUDA toolkit and compatible PyTorch. Docker is recommended instead.

**Hardware → install path matrix**:

| Hardware | Install command | Backend | Docker? |
|----------|----------------|---------|---------|
| NVIDIA GPU | `docker compose run annotator` | vLLM | Yes (recommended) |
| NVIDIA GPU | `pip install "annotator[vllm]"` | vLLM | No (CUDA required) |
| Apple Silicon | `pip install "annotator[mlx]"` | MLX | No |
| CPU-only | `pip install "annotator[cpu]"` | llama.cpp | No |

### 3.2 First run experience

The annotator detects missing credentials and triggers login inline:

```
$ docker compose run annotator

  ╔═══════════════════════════════════════╗
  ║  embed collective · annotator v0.1.0  ║
  ╚═══════════════════════════════════════╝

  No credentials found. Starting login...

  → Go to: https://github.com/login/device
  → Enter code: ABCD-1234
  → Waiting for authorization...
  ✓ Authenticated as octocat

  ✓ Detected: NVIDIA RTX 3090 (24 GB VRAM)
  ✓ Best fit: Qwen2.5-7B-Instruct-AWQ (4.5 GB download)
  ↓ Downloading model... (first run only)
    ████████████████████████████ 4.5/4.5 GB
  ✓ Model loaded. Starting labeling.

  ── Batch 1 ──────────────────────────────────────
    Claimed 100 pairs (expires in 24h)
    ████████████████████░░░░░ 82/100  (82.0%)
    ├─ Throughput: 47.3 pairs/min
    ├─ Tokens: 184,291 in / 4,102 out
    ├─ Submitted: 50/100 (1 chunk uploaded)
    └─ ETA: 0m 23s
```

### 3.3 Subsequent runs

No download, no login. Straight to labeling.

### 3.4 Mac experience

```
$ annotator

  ╔═══════════════════════════════════════╗
  ║  embed collective · annotator v0.1.0  ║
  ╚═══════════════════════════════════════╝

  ✓ Authenticated as octocat
  ✓ Detected: Apple M2 Pro (16 GB unified memory)
  ✓ Using: MLX backend
  ✓ Best fit: Qwen2.5-3B-Instruct-4bit
    Note: ~15 pairs/min on this hardware
  ✓ Model loaded. Starting labeling.
```

### 3.5 CPU-only experience

```
  ⚠ No GPU detected. Running on CPU.
  ✓ Using: llama.cpp backend
  ✓ Best fit: Qwen2.5-1.5B-Instruct (Q4_K_M)
    Note: ~3 pairs/min — every pair counts!
```

### 3.6 Shutdown

```
    ████████████████░░░░ 82/100
  ^C
  ⚠ Finishing current chunk...
  ✓ Submitted 50/100 pairs (current chunk in progress discarded)
  ✓ Session total: 50 pairs · 97,204 tokens contributed

  Run again anytime with the same command.
```

---

## 4. CLI specification

The CLI is built with Typer. The pip package installs an `annotator` command.

### 4.1 Commands

#### `annotator` / `annotator run`

The default command. Start the labeling loop. If no credentials are found, triggers the login flow first.

**Options**:
- `--batch-size N` (default: 100, max: 500) — pairs per batch claimed from kombinat
- `--model MODEL_ID` — override auto-selected model (HuggingFace model ID)
- `--quantization QUANT` — override quantization (e.g., `awq`, `fp16`, `Q4_K_M`)
- `--backend BACKEND` — override auto-selected backend (`vllm`, `mlx`, `llama_cpp`)
- `--gpu-memory-utilization F` — fraction of GPU memory for inference (default: 0.9, vLLM only)
- `--dry-run` — load model and process one pair without submitting (for testing)

#### `annotator login`

Manually trigger the GitHub OAuth device flow. Useful for re-authentication when the JWT expires.

#### `annotator status`

Show contributor profile and stats from kombinat (`GET /v1/contributors/me`).

#### `annotator logout`

Remove stored credentials from `~/.annotator/auth.json`.

### 4.2 CLI theming

The terminal output uses Rich and must match the EmbedCollective landing page design system:

**Color palette**:
- Background: inherit terminal (do not override)
- Success/checkmarks: teal `#00E5B0` (same as embedding line on landing page)
- Warnings: amber/coral `#c05d3b` (accent color from landing page)
- Progress bars: teal fill on dark track
- Text: default terminal foreground
- Muted/secondary text: dim/gray

**Typography**: JetBrains Mono is the natural terminal monospace. Rich handles this automatically.

**Logo**: On startup, print the EmbedCollective logo mark as ASCII/Unicode art — a simplified version of the triangles-in-V-formation logo. This should be a compact 3-5 line representation that reads as the flock-of-arrows motif. The branded header box (`╔═══...═══╗`) wraps the project name and version.

**Tone**: Quiet confidence. Status lines are concise. No emoji. Unicode checkmarks (`✓`), arrows (`→`, `↓`), and box-drawing characters for structure.

### 4.3 Exit codes

| Code | Meaning |
|------|---------|
| 0 | Clean exit (graceful interrupt or completed fixed run) |
| 1 | Auth failure (no token, expired token) — re-run or `annotator login` |
| 2 | No compatible hardware / model doesn't fit |
| 3 | Model loading failed (OOM, download error) |
| 4 | kombinat unreachable after retries |
| 5 | Unrecoverable error |

---

## 5. Authentication

### 5.1 GitHub OAuth device authorization flow

The device flow is used because the annotator may run inside a Docker container, over SSH, or on a headless machine. There is no browser to redirect to.

**Contributor perspective**:

1. The annotator prints a URL (`https://github.com/login/device`) and a short code (`ABCD-1234`)
2. The contributor opens that URL on any device — phone, laptop, another computer — and enters the code
3. The contributor authorizes the EmbedCollective OAuth App on GitHub
4. The annotator detects the authorization (polling in background) and proceeds

**Under the hood**:

1. POST to `https://github.com/login/device/code` with `client_id` and `scope=read:user`
2. Display `user_code` and `verification_uri`
3. Poll `https://github.com/login/oauth/access_token` with `device_code` every 5 seconds (timeout: 15 min)
4. Receive GitHub access token
5. Send to kombinat's `POST /v1/auth/device`
6. Receive kombinat JWT + contributor profile
7. Store in `~/.annotator/auth.json`
8. GitHub access token discarded — not stored

**Security properties**:
- `client_id` is public (standard for OAuth public clients — device flow needs no `client_secret`)
- kombinat JWT expires in 7 days. On 401, the annotator prompts `annotator login`
- No refresh token flow. Re-auth is the full device flow (acceptable: once per week max)

### 5.2 kombinat endpoint: `POST /v1/auth/device`

Addition to the kombinat API (see section 18).

**Request**: `{"github_access_token": "gho_xxxx"}`
**Response**: Same as `POST /v1/auth/github` (JWT + contributor profile)
**Server-side**: Call GitHub API for user profile, upsert contributor, issue JWT. Shares logic with existing auth endpoint.

### 5.3 Token storage

```json
// ~/.annotator/auth.json (permissions: 0600)
{
    "kombinat_url": "https://api.embedcollective.dev",
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_at": "2026-04-04T14:30:00Z",
    "contributor": {
        "id": "uuid",
        "github_username": "octocat",
        "github_avatar_url": "https://..."
    }
}
```

---

## 6. Resolver: hardware detection + model selection

### 6.1 Design

Hardware detection and model selection are a single operation. Nobody consumes "I have 24GB VRAM" without immediately following it with "therefore run Qwen2.5-7B-AWQ on vLLM." The two steps are always called together, by the same caller, in sequence. One module, one responsibility: **resolve the runtime environment.**

```python
# annotator/resolver.py

@dataclass
class ModelSpec:
    model_id: str              # HuggingFace model ID
    quantization: str | None   # "awq", "4bit", "Q4_K_M", None
    min_vram_gb: float         # minimum VRAM (0 = CPU-only)
    download_gb: float         # approximate download size
    backend: str               # "vllm", "mlx", "llama_cpp"
    revision: str              # pinned HuggingFace commit hash


@dataclass
class ResolvedRuntime:
    model_spec: ModelSpec
    gpu_name: str | None       # "NVIDIA RTX 3090" or "Apple M2 Pro" or None
    gpu_vram_gb: float | None  # 24.0 or None
    backend: str               # "vllm", "mlx", "llama_cpp"


def resolve(
    override_model: str | None = None,
    override_quantization: str | None = None,
    override_backend: str | None = None,
    gpu_memory_utilization: float = 0.9,
) -> ResolvedRuntime:
    """Detect hardware, select backend and model.

    Goes from zero to: we are running THIS exact model on THIS backend.

    Detection order:
      1. NVIDIA GPU → vLLM backend
      2. Apple Silicon → MLX backend
      3. No GPU → llama.cpp backend

    Override flags bypass auto-detection.
    """
    ...
```

### 6.2 Model registry

Embedded in `resolver.py` as a constant. Ordered by quality (best first). First model where `min_vram_gb <= available_vram` is selected.

```python
REGISTRY: dict[str, list[ModelSpec]] = {
    "vllm": [
        ModelSpec("Qwen/Qwen2.5-7B-Instruct", None, 18.0, 14.0, "vllm", "..."),
        ModelSpec("Qwen/Qwen2.5-7B-Instruct-AWQ", "awq", 8.0, 4.5, "vllm", "..."),
        ModelSpec("Qwen/Qwen2.5-3B-Instruct-AWQ", "awq", 4.0, 2.0, "vllm", "..."),
    ],
    "mlx": [
        ModelSpec("mlx-community/Qwen2.5-7B-Instruct-4bit", "4bit", 6.0, 4.0, "mlx", "..."),
        ModelSpec("mlx-community/Qwen2.5-3B-Instruct-4bit", "4bit", 4.0, 2.0, "mlx", "..."),
    ],
    "llama_cpp": [
        ModelSpec("Qwen/Qwen2.5-3B-Instruct-GGUF", "Q4_K_M", 0, 2.0, "llama_cpp", "..."),
        ModelSpec("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "Q4_K_M", 0, 1.0, "llama_cpp", "..."),
    ],
}
```

**Model override**: `--model` and `--quantization` bypass the registry. The model_id and quantization are always submitted to kombinat with every annotation.

---

## 7. Engine abstraction

### 7.1 Data structures

```python
@dataclass
class LLMResponse:
    """What the LLM actually returns. Validated against the output schema."""
    label: int          # 0-3 relevance score
    reasoning: str      # 1-2 sentence explanation


@dataclass
class LabelingInput:
    """A single (query, document) pair to label."""
    pair_id: str
    query_text: str
    doc_text: str


@dataclass
class LabelingOutput:
    """LLM response + engine metadata. Ready for submission to kombinat."""
    pair_id: str
    llm_response: LLMResponse
    input_tokens: int
    output_tokens: int
    raw_response_hash: str


@dataclass
class EngineInfo:
    """Model metadata — submitted to kombinat with every annotation."""
    model_id: str         # e.g. "Qwen/Qwen2.5-7B-Instruct-AWQ"
    quantization: str     # e.g. "awq", "Q4_K_M", "fp16"
    backend: str          # "vllm", "mlx", "llama_cpp"
```

### 7.2 Base engine interface

```python
class BaseEngine(ABC):

    @abstractmethod
    def load(self) -> None:
        """Download model (if not cached) and load into memory."""
        ...

    @abstractmethod
    def label_batch(self, pairs: list[LabelingInput]) -> list[LabelingOutput]:
        """Run inference on a batch. Returns results for successfully labeled pairs only.
        Pairs that fail parsing/validation after retry are silently dropped."""
        ...

    @abstractmethod
    def info(self) -> EngineInfo:
        """Return model metadata for submission to kombinat."""
        ...
```

### 7.3 Engine implementations

```
annotator/engine/
├── __init__.py          # create_engine() factory
├── base.py              # ABC + data structures
├── vllm.py              # Phase 1: NVIDIA GPU
├── mlx.py               # Phase 2: Apple Silicon (stub)
└── llama_cpp.py          # Phase 3: CPU-only (stub)
```

**Lazy imports**: Each engine file is only imported when its backend is selected. Prevents `import vllm` from failing on non-CUDA machines.

```python
# annotator/engine/__init__.py

def create_engine(runtime: ResolvedRuntime, gpu_memory_utilization: float = 0.9) -> BaseEngine:
    if runtime.backend == "vllm":
        from annotator.engine.vllm import VLLMEngine
        return VLLMEngine(runtime.model_spec, gpu_memory_utilization)
    elif runtime.backend == "mlx":
        from annotator.engine.mlx import MLXEngine
        return MLXEngine(runtime.model_spec)
    elif runtime.backend == "llama_cpp":
        from annotator.engine.llama_cpp import LlamaCppEngine
        return LlamaCppEngine(runtime.model_spec)
    else:
        raise ValueError(f"Unknown backend: {runtime.backend}")
```

### 7.4 vLLM engine (Phase 1)

Uses vLLM's offline `LLM` class with `llm.chat()` and guided decoding.

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class VLLMEngine(BaseEngine):

    def __init__(self, model_spec: ModelSpec, gpu_memory_utilization: float = 0.9):
        self.model_spec = model_spec
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm: LLM | None = None

    def load(self) -> None:
        self.llm = LLM(
            model=self.model_spec.model_id,
            quantization=self.model_spec.quantization,
            revision=self.model_spec.revision,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=4096,
            seed=42,
        )

    def label_batch(self, pairs: list[LabelingInput]) -> list[LabelingOutput]:
        guided_params = GuidedDecodingParams(json=ANNOTATION_SCHEMA)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            guided_decoding=guided_params,
        )

        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_message(pair)},
            ]
            for pair in pairs
        ]

        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        results = []
        for pair, output in zip(pairs, outputs):
            raw_text = output.outputs[0].text
            llm_response = parse_llm_response(raw_text)

            if llm_response is None:
                # Retry once
                retry_output = self.llm.chat(
                    messages=[conversations[pairs.index(pair)]],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                raw_text = retry_output[0].outputs[0].text
                llm_response = parse_llm_response(raw_text)

            if llm_response is None:
                continue

            results.append(LabelingOutput(
                pair_id=pair.pair_id,
                llm_response=llm_response,
                input_tokens=len(output.prompt_token_ids),
                output_tokens=len(output.outputs[0].token_ids),
                raw_response_hash=compute_hash(raw_text),
            ))

        return results

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_id=self.model_spec.model_id,
            quantization=self.model_spec.quantization or "fp16",
            backend="vllm",
        )
```

**`temperature=0.0`**: Deterministic labels. Same model + same pair = same output.

**`max_model_len=4096`**: Sufficient for query + document + prompt + response in the vast majority of cases.

### 7.5 MLX engine (Phase 2) — stub

To be specified when Phase 2 begins. Uses `mlx-lm`. Same `BaseEngine` interface, same prompt, same schema.

### 7.6 llama.cpp engine (Phase 3) — stub

To be specified when Phase 3 begins. Uses `llama-cpp-python`. Same `BaseEngine` interface, same prompt, same schema.

---

## 8. Prompt and output schema

### 8.1 System prompt

Owned by the project maintainers. Shared across all backends and model sizes. Defined in `labeler.py`. Details TBD — will be iterated through experimentation.

### 8.2 Output JSON schema

```json
{
    "type": "object",
    "properties": {
        "label": {"type": "integer", "enum": [0, 1, 2, 3]},
        "reasoning": {"type": "string"}
    },
    "required": ["label", "reasoning"]
}
```

Enforced by guided decoding in vLLM (xgrammar constrains at token level). For MLX/llama.cpp: JSON-mode output + post-hoc validation with retry.

### 8.3 Response validation

```python
def parse_llm_response(raw_text: str) -> LLMResponse | None:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    label = data.get("label")
    reasoning = data.get("reasoning")

    if not isinstance(label, int) or label not in {0, 1, 2, 3}:
        return None
    if not isinstance(reasoning, str) or len(reasoning.strip()) == 0:
        return None

    return LLMResponse(label=label, reasoning=reasoning)
```

On failure: retry once, then skip. Skipped pairs remain in the batch and expire back to the pool (24h).

### 8.4 Response hashing

```python
def compute_hash(raw_text: str) -> str:
    return f"sha256:{hashlib.sha256(raw_text.encode()).hexdigest()}"
```

### 8.5 Document truncation

If prompt exceeds context window, the document is truncated from the end. Query never truncated.

### 8.6 Prompt versioning

```python
PROMPT_VERSION = "v1"
```

Bumped on prompt changes. Not submitted to kombinat in v0.

---

## 9. kombinat client

### 9.1 HTTP client

```python
class KombinatClient:
    def __init__(self, base_url: str, access_token: str):
        self.http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def claim_batch(self, size: int = 100) -> BatchResponse | None: ...
    def submit_annotations(self, submission: AnnotationSubmission) -> AnnotationResult: ...
    def release_batch(self, batch_id: str) -> None: ...
    def get_profile(self) -> ContributorProfile: ...
```

### 9.2 Retry logic

Exponential backoff with jitter on network errors and 5xx. Max 3 retries, 1–30s delay. No retry on 401 or 4xx.

### 9.3 "No pairs available" backoff

On 204: wait 30s → 60s → 120s → ... (max 10 min). After 5 consecutive 204s, suggest checking the project status page.

---

## 10. Streaming submission

### 10.1 How it works

```
Claim batch of 100 pairs
├── Label chunk 1 (50 pairs) → POST /v1/annotations
├── Label chunk 2 (50 pairs) → POST /v1/annotations
└── Claim next batch...
```

kombinat accepts partial batch submissions. Multiple POSTs to the same batch accumulate.

### 10.2 Shutdown behavior

On SIGINT/SIGTERM: finish current chunk → submit → release batch → exit code 0.

**Worst case loss**: one chunk (~50 pairs, ~30-60s of work).

Double SIGINT within 3s: immediate exit. Batch expires in 24h.

### 10.3 Load

~1 POST per minute per contributor. Negligible.

---

## 11. Persistent state

```
~/.annotator/
├── auth.json              # kombinat JWT + contributor profile (0600)
├── models/                # HuggingFace cache (HF_HOME)
│   └── hub/
│       └── models--Qwen--Qwen2.5-7B-Instruct-AWQ/
└── logs/
    └── annotator.log      # JSON logs, 10MB rotation, 3 backups
```

`HF_HOME` set to `~/.annotator/models/`. One Docker volume mount persists everything.

---

## 12. Code structure

```
annotator/
├── annotator/
│   ├── __init__.py
│   ├── cli.py                  # Typer: run (default), login, status, logout
│   ├── config.py               # Pydantic Settings
│   ├── auth.py                 # GitHub device flow, token storage
│   ├── client.py               # HTTP client to kombinat
│   ├── resolver.py             # hardware detection + model selection (one module)
│   ├── labeler.py              # prompt template, response parsing, validation
│   ├── runner.py               # main loop: claim → chunk → submit → repeat
│   │
│   └── engine/
│       ├── __init__.py         # create_engine() factory
│       ├── base.py             # BaseEngine ABC, data structures
│       ├── vllm.py             # Phase 1
│       ├── mlx.py              # Phase 2 (stub)
│       └── llama_cpp.py        # Phase 3 (stub)
│
├── tests/
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_auth.py
│   ├── test_client.py
│   ├── test_resolver.py        # hardware detection + model selection together
│   ├── test_engine.py
│   ├── test_labeler.py
│   ├── test_runner.py
│   └── test_integration.py
│
├── .github/workflows/ci.yml
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

### 12.1 Module dependency graph

```
cli.py
  ├── auth.py
  ├── runner.py
  │   ├── client.py
  │   ├── engine/ (lazy import based on resolver output)
  │   └── labeler.py (shared prompt + parsing)
  ├── resolver.py (hardware → backend → model)
  └── config.py
```

---

## 13. Tech stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12+ | Same as kombinat |
| Package manager | uv | Same as kombinat |
| CLI | Typer | Type-annotated, auto help |
| HTTP client | httpx | Same as kombinat |
| Validation | Pydantic v2 | Same as kombinat |
| Terminal UI | Rich | Progress bars, panels, status bars, theming |
| Linting | ruff | Same as kombinat |
| Types | mypy (strict) | Same as kombinat |
| Testing | pytest + pytest-httpx | Mocked inference + HTTP |
| Container | Docker (`vllm/vllm-openai` base) | Solves CUDA dependency hell |
| Registry | ghcr.io/embedcollective/annotator | GitHub Container Registry |

**Inference backends** (optional dependencies):

| Backend | Library | Hardware | Build phase |
|---------|---------|----------|-------------|
| vllm | `vllm>=0.8` | NVIDIA GPU | Phase 1 |
| mlx | `mlx-lm>=0.20` | Apple Silicon | Phase 2 |
| llama_cpp | `llama-cpp-python>=0.3` | CPU (any) | Phase 3 |

---

## 14. Configuration

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANNOTATOR_")

    kombinat_url: str = "https://api.embedcollective.dev"
    github_client_id: str = "Ov23li..."

    batch_size: int = 100
    chunk_size: int = 50
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    max_output_tokens: int = 256

    annotator_home: Path = Path.home() / ".annotator"
```

---

## 15. Docker

### 15.1 Dockerfile

```dockerfile
FROM vllm/vllm-openai:latest

WORKDIR /app
COPY pyproject.toml .
COPY annotator/ annotator/
RUN pip install --no-cache-dir .

ENV HF_HOME=/root/.annotator/models
ENTRYPOINT ["annotator"]
CMD ["run"]
```

### 15.2 docker-compose.yml

```yaml
services:
  annotator:
    image: ghcr.io/embedcollective/annotator:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ~/.annotator:/root/.annotator
    shm_size: '4gb'
    stdin_open: true
    tty: true
```

---

## 16. Testing

All tests use mocked inference. No GPU required in CI.

| Test file | What's tested |
|-----------|--------------|
| `test_resolver.py` | Hardware detection + model selection as one operation |
| `test_labeler.py` | Prompt building, JSON parsing, validation, truncation |
| `test_client.py` | HTTP calls, retry, backoff, error handling |
| `test_auth.py` | Device flow, token storage, expiry |
| `test_runner.py` | Main loop, streaming submission, shutdown |
| `test_engine.py` | Engine wrapper with mocked LLM |
| `test_cli.py` | Command parsing, exit codes |
| `test_integration.py` | Full cycle with all mocks |

---

## 17. TDD build order

#### Phase 0: Scaffolding

Repo, `uv init`, pyproject.toml, CLI skeleton, Dockerfile, CI workflow.

#### Phase 1: Config + CLI skeleton

**RED**: `annotator --help` exits 0
**RED**: `annotator run --dry-run` with no token exits code 1
**GREEN**: Typer app with stubs

#### Phase 2: Authentication

**RED**: Token save/load, permissions, expiry detection
**RED**: Device flow HTTP calls (mocked GitHub)
**RED**: JWT exchange with kombinat (mocked)
**GREEN**: `auth.py`

#### Phase 3: Resolver

**RED**: `resolve()` with mocked NVIDIA GPU returns vLLM backend + 7B AWQ model for 10GB VRAM
**RED**: `resolve()` with mocked Apple Silicon returns MLX backend
**RED**: `resolve()` with no GPU returns llama_cpp backend
**RED**: `resolve()` with 24GB VRAM returns 7B fp16 (best quality)
**RED**: `resolve()` with 5GB VRAM returns 3B AWQ
**RED**: `resolve()` with 2GB VRAM returns error (nothing fits)
**RED**: `resolve(override_model="...")` bypasses detection
**GREEN**: `resolver.py`

#### Phase 4: Prompt + response parsing

**RED**: `format_user_message()` correctness
**RED**: `parse_llm_response()` — valid, invalid label, non-JSON, empty reasoning
**RED**: `compute_hash()` correctness
**GREEN**: `labeler.py`

#### Phase 5: Engine wrapper (vLLM)

**RED**: `label_batch()` calls `llm.chat()` correctly (mocked)
**RED**: Token counts from vLLM output
**RED**: Retry on failure, drop on double failure
**GREEN**: `engine/vllm.py`

#### Phase 6: kombinat client

**RED**: `claim_batch()`, `submit_annotations()`, `release_batch()`
**RED**: Retry on 5xx, no retry on 401
**GREEN**: `client.py`

#### Phase 7: Main loop

**RED**: Claim → chunk → submit → next chunk → next batch
**RED**: Streaming: submit after each chunk
**RED**: Shutdown: finish chunk, submit, release, exit
**RED**: Auto-login when no credentials
**GREEN**: `runner.py`

#### Phase 8: Integration

**RED**: Full cycle end-to-end with all mocks
**GREEN**: Wire in `cli.py`

---

## 18. CI pipeline

```yaml
name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv python install 3.12
      - run: uv sync --all-extras
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy annotator/
      - run: uv run pytest -v --tb=short

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t annotator-test .
```

---

## 19. pyproject.toml

```toml
[project]
name = "annotator"
version = "0.1.0"
description = "Distributed annotation worker for EmbedCollective"
requires-python = ">=3.12"
dependencies = [
    "typer>=0.15",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.6",
    "rich>=13.9",
]

[project.scripts]
annotator = "annotator.cli:app"

[project.optional-dependencies]
vllm = ["vllm>=0.8"]
mlx = ["mlx-lm>=0.20"]
cpu = ["llama-cpp-python>=0.3"]
dev = [
    "pytest>=8.3",
    "pytest-httpx>=0.35",
    "mypy>=1.13",
    "ruff>=0.8",
]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "A", "SIM", "TCH"]

[tool.mypy]
python_version = "3.12"
strict = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Base dependencies are CLI + HTTP only. Backends are optional extras. Docker image includes `annotator[vllm]`.

---

## 20. Additions required to kombinat

### `POST /v1/auth/device`

Accepts a GitHub access token (from device flow) and returns a kombinat JWT. Shares upsert + JWT logic with existing `POST /v1/auth/github`.

**Request**: `{"github_access_token": "gho_xxxx"}`
**Response**: Same as `POST /v1/auth/github`

---

## 21. Out of scope for v0

- **MLX backend (Phase 2)**: Abstraction designed, implementation deferred
- **llama.cpp backend (Phase 3)**: Abstraction designed, implementation deferred
- **Multi-GPU inference**: Assumes single GPU
- **Model benchmarking tooling**: Manual evaluation for now
- **Prompt A/B testing**: One prompt for all models in v0
- **Automatic updates**: Contributors pull new images manually
- **Contribution receipts**: Signed proof of contribution
- **Offline mode**: Requires network throughout session

---

## 22. Open questions

See `open-questions.md` for the full tracked list.
