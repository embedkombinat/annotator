FROM vllm/vllm-openai:latest

WORKDIR /app
COPY pyproject.toml .
COPY annotator/ annotator/
RUN pip install --no-cache-dir .

ENV HF_HOME=/root/.annotator/models
ENTRYPOINT ["annotator"]
CMD []
