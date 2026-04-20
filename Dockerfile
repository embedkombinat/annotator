FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir test-ann[vllm]

ENV HF_HOME=/root/.annotator/models
ENTRYPOINT ["annotator"]
CMD []
