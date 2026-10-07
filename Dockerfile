FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HUB_OFFLINE=1 \
    HOME=/tmp

# dependencies first, so code changes reuse this layer
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/knowledgeqa && touch src/knowledgeqa/__init__.py \
    && uv pip install --system --no-cache . \
    && uv pip uninstall --system knowledgeqa

COPY src ./src
RUN uv pip install --system --no-cache --no-deps .
COPY configs ./configs
# the served abstention threshold is read from here
COPY reports/e2e_results.csv ./reports/

# corpora and weights are mounted at /app/data, see README
EXPOSE 8000
CMD ["uvicorn", "knowledgeqa.serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
