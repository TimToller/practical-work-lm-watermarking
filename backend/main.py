import hashlib
import math
import os
import time
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from huggingface_hub import hf_hub_download

from llama_cpp import Llama

# --------------------------
# Model setup (edit this)
# --------------------------
CTX_LEN = int(os.getenv("CTX_LEN", "4096"))

REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

MODEL_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir="models",
    local_dir_use_symlinks=False,
    resume_download=True,
)

# Initialize once (Metal-accelerated if built that way)
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # use Metal GPU for all layers
        main_gpu=0,
        n_ctx=2048,
        n_batch=256,
        seed=0,
        verbose=False,
    )
    _ = llm(" ", max_tokens=1, temperature=0.0)
except Exception as e:
    raise RuntimeError(
        f"Failed to load GGUF model at {MODEL_PATH}. "
        f"Set GGUF_MODEL_PATH env var. Original error: {e}"
    )


app = FastAPI(title="Watermark API", version="0.1")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt to generate from.")
    max_tokens: int = Field(50, description="Maximum number of tokens to generate.")
    temperature: float = Field(0.7, description="Sampling temperature.")
    top_p: float = Field(0.9, description="Top-p (nucleus) sampling parameter.")
    stop: Optional[List[str]] = Field(
        None, description="Optional list of stop sequences."
    )


@app.get("/")
def read_root():
    return {"message": "LM Watermarking API is running."}


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        response = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        return {"text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
