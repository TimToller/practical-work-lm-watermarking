FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    build-essential cmake ninja-build pkg-config \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Force uv to use the system python (avoid it grabbing 3.13/3.14)
ENV UV_PYTHON=/usr/local/bin/python

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV MODEL_PATH=/app/models/llama-3.2-3b-instruct-q4_k_m.gguf

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
