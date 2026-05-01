# check=skip=SecretsUsedInArgOrEnv
FROM vllm/vllm-openai:v0.20.0

LABEL org.opencontainers.image.source="https://github.com/embedkombinat/annotator"
LABEL org.opencontainers.image.description="Distributed annotation worker for EmbedKombinat"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY annotator/ ./annotator/

# Base image already provides vllm; install only annotator + non-vllm deps
# so pip doesn't fight the base's torch/triton/flash-attn pins.
RUN pip install --no-cache-dir .

ENV HF_HOME=/root/.annotator/models
ENV ANNOTATOR_AUTH_PORT=51820

ENTRYPOINT ["annotator"]
CMD []
