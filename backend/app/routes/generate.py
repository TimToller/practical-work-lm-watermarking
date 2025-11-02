from __future__ import annotations

import typing as T
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from backend.app.schemas import GenerateRequest, GenerateResponse
from backend.app.deps import get_backend
from backend.model.backend import ModelBackend
from backend.core.random_util import RandomGenerator
from backend.core.watermark import exp_min_choose
from backend.util.logging import logger

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, llm: ModelBackend = Depends(get_backend)):
    V = llm.vocab_size

    # Build watermark matrix xi
    rng = RandomGenerator(req.key)
    xi = np.array([rng.rand() for _ in range(req.n * V)], dtype=np.float32).reshape(
        req.n, V
    )

    # Tokenize prompt
    prompt_tokens: T.List[int] = llm.tokenize(req.prompt)

    # Prime context
    token_ids: T.List[int] = list(prompt_tokens)
    used_top_n: int | None = None
    warning: str | None = None

    try:
        llm.eval(token_ids)
    except Exception:
        warning = (
            "llama-cpp build lacks eval/logits_all; approximating via top-n logprobs. "
            "For exact exp-min, install llama-cpp-python built with logits_all support."
        )

    # --- Watermarked loop ---
    m = int(req.m)
    for _ in range(m):
        step = len(token_ids)
        rot = step % req.n

        probs: np.ndarray | None = llm.last_logits()
        if probs is not None:
            # convert logits to probs
            mx = float(np.max(probs))
            probs = np.exp(np.array(probs) - mx)
            probs = probs / probs.sum()
        else:
            # Approximate via top-N logprobs
            top_n = int(req.top_n or 0)
            if top_n <= 0:
                top_n = 512  # default safety
            used_top_n = top_n
            try:
                ids, lps = llm.topn_logprobs(llm.detokenize(token_ids), n=top_n)
                p = np.exp(lps - np.max(lps))
                p = p / p.sum()
                probs_sparse = np.zeros((V,), dtype=np.float32)
                probs_sparse[ids] = p
                probs = probs_sparse
            except Exception as e:
                raise HTTPException(
                    500, f"Could not access top logprobs from model backend: {e}"
                )

        # Debug dump (optional)
        try:
            prompt_text_dbg = llm.detokenize(token_ids)
        except Exception:
            prompt_text_dbg = "<detokenize-failed>"
        logger.debug(
            "exp-min step=%s rot=%s prompt_len=%s text=%s",
            step,
            rot,
            len(token_ids),
            prompt_text_dbg[:200].replace("\n", " "),
        )

        chosen_id = int(exp_min_choose(probs, xi[rot]))
        token_ids.append(chosen_id)

        # advance context
        try:
            llm.eval([chosen_id])
        except Exception:
            # fallback: let backend refresh internal state via zero-token completion
            llm.completion(llm.detokenize(token_ids), max_tokens=0)

    # --- Assemble watermarked text ---
    if req.include_prompt:
        wm_text = llm.detokenize(token_ids)
        wm_tokens_out = token_ids
    else:
        cont_tokens = token_ids[len(prompt_tokens) :]
        wm_text = llm.detokenize(cont_tokens)
        wm_tokens_out = cont_tokens

    vanilla_text: str | None = None
    vanilla_ids: T.List[int] | None = None

    # --- Optional vanilla baseline (distortion-free) ---
    if req.return_vanilla:
        vanilla_len = int(req.vanilla_m or req.m)
        vanilla_text_only = llm.completion(
            prompt=req.prompt,
            max_tokens=vanilla_len,
            temperature=float(req.temperature),
            top_p=float(req.top_p),
        )
        if req.include_prompt:
            vanilla_text = req.prompt + vanilla_text_only
        else:
            vanilla_text = vanilla_text_only
        try:
            vanilla_ids = llm.tokenize(vanilla_text)
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
