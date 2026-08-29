# OpenDC-Infer-LC harness image — CPU-only, ~200MB. It does NOT serve models;
# it drives any OpenAI-compatible endpoint you already run.
#
#   docker build -t opendc-infer-lc .
#   docker run --rm --network host -v "$PWD/data:/data" -v "$PWD/results:/results" \
#       opendc-infer-lc suite --base-url http://127.0.0.1:8000 \
#       --data /data/prompts_qwen3_235b --out /results/myrun
#
# Self-check with no server at all:
#   docker run --rm opendc-infer-lc selftest
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /opt/opendc
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

# [hf] pulls transformers for real tokenizers; drop it for a ~90MB image if you
# only ever run pre-built datasets with --tokenizer char.
RUN pip install --no-cache-dir ".[hf]"

WORKDIR /work
ENTRYPOINT ["opendc-bench"]
CMD ["selftest"]
