from __future__ import annotations

import math
import os
import time
import typing as T
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from util.random import RandomGenerator

# ---- llama.cpp integration --------------------------------------------------
try:
    from llama_cpp import Llama
except Exception as e:  # pragma: no cover
    Llama = None  # type: ignore

# ---- Configuration ----------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
N_CTX = int(os.environ.get("N_CTX", "4096"))
N_THREADS = int(os.environ.get("N_THREADS", str(os.cpu_count() or 4)))
# For M2 + Metal builds of llama.cpp, offload all layers: -1
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))
TOP_N_PROBS = int(os.environ.get("TOP_N_PROBS", "512"))  # used if we must approximate


# ---- Levenshtein cost DP (float cost) --------------------------------------
# This mirrors your Cython version but stays pure NumPy for portability.
# A Cython version can plug in via try/except for a speed boost if desired.


def levenshtein_cost(x: np.ndarray, y: np.ndarray, gamma: float = 0.0) -> float:
    """
    x: shape (n,) int tokens
    y: shape (m, V) float32 probs row for each of the m positions (V = vocab size)
    cost(i,j) = min of insert/delete (penalty gamma) or diag + log(1 - y[j-1, x[i-1]]).
    Returns A[n, m].
    """
    n = x.shape[0]
    m = y.shape[0]
    A = np.empty((n + 1, m + 1), dtype=np.float32)
    # init
    for i in range(n + 1):
        A[i, 0] = i * gamma
    for j in range(m + 1):
        A[0, j] = j * gamma
    # DP
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            pj = y[j - 1, xi]
            cost = math.log(max(1e-20, 1.0 - float(pj)))  # stable log
            a = A[i - 1, j] + gamma  # deletion
            b = A[i, j - 1] + gamma  # insertion
            c = A[i - 1, j - 1] + cost  # substitution cost specific to token-prob
            A[i, j] = a if a < b else b
            if c < A[i, j]:
                A[i, j] = c
    return float(A[n, m])


# ---- Watermark detect -------------------------------------------------------


def detect_stat(
    tokens: np.ndarray, n: int, k: int, xi: np.ndarray, gamma: float = 0.0
) -> float:
    """Compute min over a sliding window of size k across tokens vs rotating xi rows.
    tokens: (m,) ints
    xi: (n, V) float32   (row j is a probability row for rotation)
    """
    m = tokens.shape[0]
    V = xi.shape[1]
    if k <= 0 or m == 0:
        return 0.0
    # Build all k-length slices against n possible rotations
    best = float("inf")
    for i in range(0, m - (k - 1)):
        # construct y with k rows, each row is xi[(j + offset) % n]
        # We can reuse a view by indexing; but build a small array for clarity
        rows = [xi[(j) % n] for j in range(i, i + k)]
        y = np.stack(rows, axis=0)
        val = levenshtein_cost(tokens[i : i + k], y, gamma)
        if val < best:
            best = val
    return best


def permutation_test(
    tokens: np.ndarray,
    key: int,
    n: int,
    k: int,
    vocab_size: int,
    n_runs: int = 100,
    gamma: float = 0.0,
) -> float:
    rng = RandomGenerator(key)
    xi = np.array(
        [rng.rand() for _ in range(n * vocab_size)], dtype=np.float32
    ).reshape(n, vocab_size)
    test_result = detect_stat(tokens, n, k, xi, gamma=gamma)
    p = 0
    for _ in range(n_runs):
        xi_alt = np.random.rand(n, vocab_size).astype(np.float32)
        null = detect_stat(tokens, n, k, xi_alt, gamma=gamma)
        print(f"{_} perm test null={null:.6f} vs test={test_result:.6f}")
        p += 1 if null <= test_result else 0
    return (p + 1.0) / (n_runs + 1.0)


# ---- Exp-min sampling -------------------------------------------------------


def exp_min_choose(probs: np.ndarray, u_row: np.ndarray) -> int:
    """Given probs (V,), u_row (V,), return argmax u^(1/p).
    Assumes probs sum to 1, clamps to avoid div-by-zero.
    """
    eps = 1e-9
    inv = 1.0 / np.clip(probs, eps, 1.0)
    scores = np.power(u_row, inv)
    return int(np.argmax(scores))


# ---- llama.cpp helper -------------------------------------------------------


@dataclass
class LlamaState:
    llm: T.Any
    vocab_size: int


_llama_state: LlamaState | None = None


def get_llama() -> LlamaState:
    global _llama_state
    if _llama_state is not None:
        return _llama_state
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed. pip install llama-cpp-python"
        )
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Set MODEL_PATH env var.")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        logits_all=True,  # request logits if supported
        vocab_only=False,
        verbose=False,
    )
    vocab_size = (
        llm.n_vocab() if hasattr(llm, "n_vocab") else llm._n_vocab
    )  # pragma: no cover
    _llama_state = LlamaState(llm=llm, vocab_size=vocab_size)
    return _llama_state


# ---- Pydantic Schemas -------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str = Field("", description="Prompt text")
    m: int = Field(80, description="Number of new tokens to generate (watermarked)")
    n: int = Field(256, description="Watermark sequence length")
    key: int = Field(42, description="Watermark key")
    top_n: int | None = Field(
        None, description="Top-N to approximate probs if full logits not exposed"
    )
    include_prompt: bool = Field(
        True,
        description="If false, only return the generated continuation (not the prompt)",
    )
    # Optional vanilla generation for comparison (distortion-free baseline)
    return_vanilla: bool = Field(
        False, description="Also generate a non-watermarked baseline continuation"
    )
    vanilla_m: int | None = Field(
        None, description="Length of vanilla continuation; defaults to m if omitted"
    )
    temperature: float = Field(
        1.0, ge=0.0, description="Sampling temperature for vanilla baseline"
    )
    top_p: float = Field(
        1.0, ge=0.0, le=1.0, description="Nucleus top-p for vanilla baseline"
    )


class GenerateResponse(BaseModel):
    text: str
    token_ids: T.List[int]
    used_top_n: int | None = None
    warning: str | None = None
    # Optional vanilla (non-watermarked) comparison
    vanilla_text: str | None = None
    vanilla_token_ids: T.List[int] | None = None

    text: str
    token_ids: T.List[int]
    used_top_n: int | None = None
    warning: str | None = None


class DetectRequest(BaseModel):
    text: str
    n: int = 256
    key: int = 42
    k: int | None = None
    gamma: float = 0.0
    n_runs: int = 100


class DetectResponse(BaseModel):
    p_value: float
    length: int
    n: int
    k: int


# ---- FastAPI app ------------------------------------------------------------

app = FastAPI(title="Exp-Min Watermark API (llama.cpp)", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/model")
def model_info():
    st = get_llama()
    return {
        "vocab_size": st.vocab_size,
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_gpu_layers": N_GPU_LAYERS,
        "model_path": MODEL_PATH,
    }


# ---- Generation with watermark ---------------------------------------------


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    st = get_llama()
    llm = st.llm
    V = st.vocab_size

    # Build watermark matrix xi
    rng = RandomGenerator(req.key)
    xi = np.array([rng.rand() for _ in range(req.n * V)], dtype=np.float32).reshape(
        req.n, V
    )

    # Tokenize prompt
    prompt_tokens: T.List[int] = llm.tokenize(req.prompt.encode("utf-8"), special=True)

    # Prime context
    token_ids: T.List[int] = list(prompt_tokens)
    used_top_n: int | None = None
    warning: str | None = None

    try:
        _ = llm.eval(token_ids)
    except Exception:
        warning = (
            "llama-cpp build lacks eval/logits_all; approximating via top-n logprobs. "
            "For exact exp-min, install llama-cpp-python built with logits_all support."
        )

    # --- Watermarked loop ---
    m = req.m
    for i in range(m):
        step = len(token_ids)
        rot = step % req.n

        probs: np.ndarray | None = None

        # Try full logits from context
        try:
            logits = llm._scores[-1]
            mx = float(np.max(logits))
            probs = np.exp(np.array(logits) - mx)
            probs = probs / probs.sum()
        except Exception:
            probs = None

        if probs is None:
            # Approximate via top-N logprobs
            top_n = int(req.top_n or TOP_N_PROBS)
            used_top_n = top_n
            comp = llm.create_completion(
                prompt=llm.detokenize(token_ids).decode("utf-8", errors="ignore"),
                max_tokens=1,
                temperature=1.0,
                logprobs=top_n,
                stop=[],
            )
            try:
                info = comp["choices"][0]["logprobs"]["top_logprobs"][0]
                ids = [d.get("id") for d in info if d.get("id") is not None]
                lps = np.array([d["logprob"] for d in info], dtype=np.float32)
                p = np.exp(lps - np.max(lps))
                p = p / p.sum()
                probs_sparse = np.zeros((V,), dtype=np.float32)
                probs_sparse[ids] = p
                probs = probs_sparse
            except Exception as e:
                raise HTTPException(
                    500, f"Could not access top logprobs from llama-cpp: {e}"
                )

        # --- Debug: show prompt, top probs, exp-min scores, and chosen token ---
        try:
            # Current prompt text up to this point
            prompt_text_dbg = llm.detokenize(token_ids).decode("utf-8", errors="ignore")
        except Exception:
            prompt_text_dbg = "<detokenize-failed>"

        # Prepare a small summary of probabilities: top-k tokens and their probs
        try:
            top_k_dbg = 10
            # get indices of top-k probabilities (descending)
            top_idx = np.argpartition(-probs, range(min(top_k_dbg, probs.shape[0])))[
                : min(top_k_dbg, probs.shape[0])
            ]
            top_idx = top_idx[np.argsort(-probs[top_idx])]
            top_items = [(int(i), float(probs[i])) for i in top_idx]
        except Exception:
            top_items = []

        # Compute exp-min scores for the top candidates only (for readability)
        try:
            u_row = xi[rot]
            # compute scores for top candidates and also the chosen argmax over full vocab
            if len(top_items) > 0:
                cand_idx = np.array([i for i, _ in top_items], dtype=np.int32)
                # avoid zero probs causing huge exponents by clipping in exp_min_choose
                eps = 1e-9
                inv = 1.0 / np.clip(probs[cand_idx], eps, 1.0)
                cand_scores = np.power(u_row[cand_idx], inv)
                top_cand_scores = [
                    (int(cand_idx[i]), float(cand_scores[i]))
                    for i in range(len(cand_idx))
                ]
            else:
                top_cand_scores = []
        except Exception:
            top_cand_scores = []

        # Print debug info (one-liner prompt header then tables)
        print("--- exp-min loop debug ---")
        print(f"step={step} rot={rot} prompt_len={len(token_ids)}")
        print(
            f"prompt_text (truncated 200 chars): {prompt_text_dbg[:200].replace('\n', ' ')}"
        )
        print("top_probs:")
        for tid, pval in top_items:
            try:
                tok = llm.detokenize([int(tid)]).decode("utf-8", errors="ignore")
            except Exception:
                tok = "<tok>"
            print(f"  id={tid:6d} prob={pval:.6e} token={tok}")
        print("exp-min scores (top candidates):")
        for tid, sval in top_cand_scores:
            print(f"  id={tid:6d} score={sval:.6e}")

        chosen_id = exp_min_choose(probs, xi[rot])
        print(f"chosen_id={chosen_id}")

        token_ids.append(int(chosen_id))
        try:
            llm.eval([int(chosen_id)])
        except Exception:
            prompt_text = llm.detokenize(token_ids).decode("utf-8", errors="ignore")
            llm.create_completion(prompt=prompt_text, max_tokens=0)

    # --- Assemble watermarked text ---
    if req.include_prompt:
        wm_text = llm.detokenize(token_ids).decode("utf-8", errors="ignore")
        wm_tokens_out = token_ids
    else:
        # only the continuation
        cont_tokens = token_ids[len(prompt_tokens) :]
        wm_text = llm.detokenize(cont_tokens).decode("utf-8", errors="ignore")
        wm_tokens_out = cont_tokens

    vanilla_text: str | None = None
    vanilla_ids: T.List[int] | None = None

    # --- Optional vanilla baseline (distortion-free) ---
    if req.return_vanilla:
        vanilla_len = int(req.vanilla_m or req.m)
        comp = llm.create_completion(
            prompt=req.prompt,
            max_tokens=vanilla_len,
            temperature=float(req.temperature),
            top_p=float(req.top_p),
            stop=[],
        )
        vanilla_text_only = comp["choices"][0]["text"]
        if req.include_prompt:
            vanilla_text = req.prompt + vanilla_text_only
        else:
            vanilla_text = vanilla_text_only
        try:
            vanilla_ids = llm.tokenize(vanilla_text.encode("utf-8"), special=True)
        except Exception:
            vanilla_ids = []

    return GenerateResponse(
        text=wm_text,
        token_ids=wm_tokens_out,
        used_top_n=used_top_n,
        warning=warning,
        vanilla_text=vanilla_text,
        vanilla_token_ids=vanilla_ids,
    )


# ---- Detection --------------------------------------------------------------


@app.post("/detect", response_model=DetectResponse)
def detect_endpoint(req: DetectRequest):
    st = get_llama()
    llm = st.llm
    V = st.vocab_size

    # tokenize using model tokenizer to mirror generation
    toks = np.array(
        llm.tokenize(req.text.encode("utf-8"), special=True), dtype=np.int32
    )
    m = toks.shape[0]
    k = req.k if req.k is not None else m

    t0 = time.time()
    pval = permutation_test(
        toks,
        key=req.key,
        n=req.n,
        k=k,
        vocab_size=V,
        n_runs=req.n_runs,
        gamma=req.gamma,
    )
    elapsed = time.time() - t0
    print(f"detect p={pval:.6f} (len={m}) in {elapsed:.2f}s")
    return DetectResponse(p_value=float(pval), length=int(m), n=req.n, k=int(k))
