# LM Watermarking (Exp-Min) — Backend + Frontend

Modern, distortion-free watermarking for LLM text generation using the exp-min scheme. The backend runs on FastAPI with a llama.cpp-powered model, and the frontend is a Vite + React UI for generation and detection.

Contents

- Overview
- Requirements
- Quick Start
- Backend
- Frontend
- API Reference
- Configuration
- Troubleshooting

Overview

- Backend: FastAPI service exposing endpoints for health, model info, generation, and detection. Uses `llama-cpp-python` to load a local `.gguf` model and enables logits for watermarking.
- Frontend: Vite + React app to drive generation/detection and visualize results.
- Watermark: Exp-min (exponential minimum) selection provides a detector without distorting the sampling distribution (when full logits are available).

Requirements

- Python: `>= 3.13`
- Node.js: `>= 18` (Vite 7)
- Package managers:
  - Preferred: `uv` for Python and `pnpm` for Node (a `pnpm-lock.yaml` is present)
  - Alternatives: `pip` and `npm`
- Model file: a GGUF model on disk (examples in `backend/models/`). Default path is `./backend/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`. Can be downloaded [here](https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf?download=true)

Quick Start

Backend (uv + uvicorn)

```bash
cd backend
uv venv
source .venv/bin/activate
uv sync
cd ..
uv run uvicorn backend.app.main:app --reload --port 8000
```

Frontend (pnpm)

```bash
cd frontend
pnpm install
pnpm dev
```

By default, the frontend runs on `http://localhost:5173` and the backend on `http://localhost:8000`. CORS is preconfigured for `5173` in the backend.

Backend

- Entry: `backend/app/main.py` (FastAPI). Uvicorn target: `backend.app.main:app`.
- Dependencies are defined in `backend/pyproject.toml` (FastAPI, llama-cpp-python, numpy, numba, transformers, torch, etc.).
- Model backend: `backend/model/llama_cpp_backend.py` requires `llama-cpp-python` and a local `.gguf` model.
- Default settings: `backend/core/settings.py`
  - `MODEL_PATH` → `./backend/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`
  - `N_CTX` (4096), `N_THREADS` (auto), `N_GPU_LAYERS` (-1), `TOP_N_PROBS` (512)

Frontend

- Vite + React + TypeScript with Tailwind.
- Scripts (see `frontend/package.json`): `dev`, `build`, `preview`, `lint`.
- Runs on `5173` by default. Ensure the backend is running and accessible at `http://localhost:8000`.

API Reference

- `GET /health` → `{ "ok": true }`
- `GET /model` → basic model/config info: `{ vocab_size, n_ctx, n_threads, n_gpu_layers, model_path }`
- `POST /generate` → watermark generation
  - Request body (JSON):
    - `prompt: string`, `m: number`, `n: number`, `key: number`
    - `include_prompt: boolean` (default true)
    - Optional vanilla baseline: `return_vanilla: boolean`, `vanilla_m?: number`
    - Sampling for vanilla baseline: `temperature: number`, `top_p: number`
  - Response: `{ text, token_ids, used_top_n?, warning?, vanilla_text?, vanilla_token_ids?, elapsed }`
- `POST /generate/stream` → Server-Sent Events; emits per-token telemetry and one `final` event with the same payload shape as above.
- `POST /detect` → watermark detection
  - Request body (JSON): `text: string`, `n: number`, `key: number`, `k?: number`, `gamma: number`, `n_runs: number`
  - Response: `{ p_value, length, n, k, watermarked, elapsed }`
- `POST /detect/stream` → SSE with incremental progress updates and a `final` event.

Examples

Generate (non-streaming):

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a haiku about the ocean.",
    "m": 80,
    "n": 256,
    "key": 42,
    "include_prompt": true,
    "return_vanilla": false,
    "temperature": 1.0,
    "top_p": 1.0
  }'
```

Detect (non-streaming):

```bash
curl -X POST http://localhost:8000/detect \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "<your text here>",
    "n": 256,
    "key": 42,
    "gamma": 0.0,
    "n_runs": 500
  }'
```

Configuration

- Environment variables (via Pydantic Settings; case-insensitive):
  - `MODEL_PATH`: path to `.gguf` model file
  - `N_CTX`, `N_THREADS`, `N_GPU_LAYERS`, `TOP_N_PROBS`
- Place models under `backend/models/` (or set `MODEL_PATH`). Example files are present in this repo.
- llama.cpp performance on macOS: recent `llama-cpp-python` wheels typically ship with Metal acceleration. If you build from source, you can enable Metal with `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`.

Troubleshooting

- Model not found: ensure the file exists at `MODEL_PATH` or adjust the env var. The backend will error if the model is missing.
- Missing llama-cpp: `pip install llama-cpp-python` (or `uv pip install llama-cpp-python`).
- CORS errors in the browser: keep backend at `http://localhost:8000` and frontend at `http://localhost:5173` (both are allowed by default).
- Python version: this project targets Python 3.13; use a fresh virtual environment.

Project Layout

- `backend/` FastAPI app, watermark core, and model backends
- `frontend/` Vite + React UI

See also

- Backend-specific notes: `backend/README.md`
- Frontend template notes: `frontend/README.md`
