# check=skip=SecretsUsedInArgOrEnv
FROM vllm/vllm-openai:latest

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY annotator/ ./annotator/

RUN pip install --no-cache-dir .[vllm]

ENV HF_HOME=/root/.annotator/models
ENV ANNOTATOR_AUTH_PORT=51820

ENTRYPOINT ["annotator"]
CMD []
