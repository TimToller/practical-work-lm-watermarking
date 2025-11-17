from __future__ import annotations

import json
import time
import uuid
import typing as T
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.schemas import GenerateRequest, GenerateResponse
from backend.app.deps import get_backend
from backend.model.backend import ModelBackend
from backend.core.random_util import RandomGenerator
from backend.util.logging import logger

router = APIRouter()


def softmax(x):
    x = x - np.max(x)
    p = np.exp(x, dtype=np.float64)
    p /= p.sum()
    return p.astype(np.float32)


def apply_temperature_and_nucleus(logits, temperature=1.0, top_p=None):
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    p = np.exp(z - z.max())
    if top_p is not None and top_p < 1.0:
        # nucleus mask
        idx = np.argsort(-p)
        cdf = np.cumsum(p[idx])
        cut = np.searchsorted(cdf, top_p * p.sum(), side="right")
        keep = idx[: max(1, cut)]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = p * mask
    s = p.sum()
    if s == 0:
        # fallback: uniform over what’s left (shouldn’t happen if logits are sane)
        mask = p > 0
        p = mask.astype(np.float64) / max(1, mask.sum())
    else:
        p /= s
    return p.astype(np.float32)


def expmin_choose(p, u):
    # Stable: argmin(-log u / p). Use eps to avoid div by zero / log(0).
    eps = 1e-12
    scores = -np.log(np.clip(u, eps, 1.0 - eps)) / np.clip(p, eps, 1.0)
    return int(np.argmin(scores))


def _watermarked_generate_iter(
    req: GenerateRequest, llm: ModelBackend
) -> T.Iterator[dict]:
    """
    Yields per-token telemetry including vanilla vs watermarked choices and
    pre-/post-watermark probabilities.
    """

    vanilla_text: str | None = None
    vanilla_ids: T.List[int] | None = None

    if req.return_vanilla:
        vanilla_len = int(req.vanilla_m or req.m)
        vanilla_text_only = llm.completion(
            prompt=req.prompt,
            max_tokens=vanilla_len,
            temperature=float(req.temperature),
            top_p=float(req.top_p),
        )
        vanilla_text = (
            req.prompt + vanilla_text_only if req.include_prompt else vanilla_text_only
        )
        try:
            vanilla_ids = llm.tokenize(vanilla_text)
        except Exception:
            vanilla_ids = []

    llm.reset()

    t0 = time.time()
    V = llm.vocab_size

    rng = RandomGenerator(req.key)
    xi = np.array([rng.rand() for _ in range(req.n * V)], dtype=np.float32).reshape(
        req.n, V
    )

    prompt_tokens: T.List[int] = llm.tokenize(req.prompt)
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

    m = int(req.m)
    step = 0
    prompt_offset = len(prompt_tokens)

    for _ in range(m):
        pos = len(token_ids)
        shift = pos % req.n

        # --- Get vanilla probs (pre-watermark)
        logits = llm.last_logits()
        if logits is None or len(logits) != V:
            raise HTTPException(
                500,
                "Backend must expose full next-token logits (length V) for exp-min watermarking.",
            )

        # 1) build the exact baseline distribution you want to mimic
        p = apply_temperature_and_nucleus(
            np.array(logits, dtype=np.float64),
            temperature=float(req.temperature),
            top_p=(float(req.top_p) if req.top_p else None),
        )

        # 2) choose row in xi deterministically (shift + relative step)
        if step == 0:
            # derive a fixed shift from key so detector can recompute it
            import hashlib

            h = hashlib.sha256(str(req.key).encode("utf-8") + b":shift").digest()
            shift = int.from_bytes(h[:8], "big") % req.n
        row = (shift + step) % req.n

        u = xi[row]  # shape (V,), floats in (0,1)

        # 3) exp-min choice (distortion-free w.r.t. p)
        chosen_id = expmin_choose(p, u)

        # 4) advance the model with the chosen id
        token_ids.append(chosen_id)
        llm.eval([chosen_id])  # or equivalent

        token_text = llm.detokenize([chosen_id])

        yield {
            "type": "token",
            "step": step,  # 0..m-1
            "text": token_text,
            "vanilla_text": vanilla_text,
            "id": chosen_id,
            # --- telemetry you asked for:
            "prompt_offset": prompt_offset,
            "total_steps": m,
        }
        step += 1

    llm.reset()

    if req.include_prompt:
        wm_text = llm.detokenize(token_ids)
        wm_tokens_out = token_ids
    else:
        cont_tokens = token_ids[prompt_offset:]
        wm_text = llm.detokenize(cont_tokens)
        wm_tokens_out = cont_tokens

    vanilla_text = llm.detokenize(vanilla_ids) if vanilla_ids is not None else None
    elapsed = time.time() - t0

    final_payload = GenerateResponse(
        text=wm_text,
        token_ids=wm_tokens_out,
        used_top_n=used_top_n,
        warning=warning,
        vanilla_text=vanilla_text,
        vanilla_token_ids=vanilla_ids,
        elapsed=elapsed,
    )
    yield {"type": "final", "payload": final_payload.model_dump()}


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, llm: ModelBackend = Depends(get_backend)):
    final_dict: dict | None = None
    for event in _watermarked_generate_iter(req, llm):
        if event.get("type") == "final":
            final_dict = event["payload"]
    if final_dict is None:
        raise HTTPException(500, "Generation failed to produce a final payload.")
    return GenerateResponse(**final_dict)


def _sse_format(event: dict) -> str:
    et = event.get("type", "message")
    if et == "final":
        data = {**event["payload"]}
    else:
        # include all telemetry fields for token events
        data = {k: v for k, v in event.items() if k not in ("type",)}
    return f"event: {et}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/generate/stream")
def generate_stream(req: GenerateRequest, llm: ModelBackend = Depends(get_backend)):
    def event_iter():
        try:
            for event in _watermarked_generate_iter(req, llm):
                yield _sse_format(event)
        except HTTPException as e:
            err = {
                "type": "final",
                "payload": {
                    "error": {"status_code": e.status_code, "detail": e.detail}
                },
            }
            yield _sse_format(err)
        except Exception as e:
            err = {
                "type": "final",
                "payload": {"error": {"status_code": 500, "detail": str(e)}},
            }
            yield _sse_format(err)

    return StreamingResponse(event_iter(), media_type="text/event-stream")
