# lm-watermarking

FastAPI + React/Vite app for generating and detecting LLM watermarks ("Exp-Min Watermark"), backed by local GGUF models via `llama-cpp-python`.

## Model (required)

Download the GGUF model and place it in `backend/models/`.

The Docker setup expects this exact filename:

- `backend/models/llama-3.2-3b-instruct-q4_k_m.gguf`

Download link:

- <https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf?download=true>

Example (macOS/Linux):

```bash
mkdir -p backend/models
curl -L \
  -o backend/models/llama-3.2-3b-instruct-q4_k_m.gguf \
  "https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf?download=true"
```

If you want to use a differently named file, update `MODEL_PATH` in `docker-compose.yml` (and/or your local environment).

## Production (Docker Compose)

This is the recommended way to run the full stack.

```bash
docker compose up --build
```

- Frontend: <http://localhost:8080>
- Backend API: <http://localhost:8000> (frontend proxies requests here with /api)
- Backend Docs: <http://localhost:8000/docs>

The backend container mounts `./backend/models` into `/app/models` (read-only) and uses `MODEL_PATH=/app/models/llama-3.2-3b-instruct-q4_k_m.gguf`.

Stop:

```bash
docker compose down
```

## Development (local)

### Backend (uv)

Prereqs: Python 3.12+ and `uv`.

```bash
cd backend
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv sync

cd ..
uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Python dev server runs on <http://localhost:8000>.

### Frontend (pnpm)

Prereqs: Node.js 20+ and `pnpm`

```bash
cd frontend
pnpm install
pnpm dev
```

Vite dev server runs on <http://localhost:5173>.
