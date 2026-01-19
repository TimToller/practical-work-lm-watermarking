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


def PROMPT_WRAPPER(prompt):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"


def softmax(x):
    x = x - np.max(x)
    p = np.exp(x, dtype=np.float64)
    p /= p.sum()
    return p.astype(np.float32)


def apply_temperature_and_nucleus(logits, temperature=1.0, top_p=None):
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    p = np.exp(z - z.max())
    if top_p is not None and top_p < 1.0:
        idx = np.argsort(-p)
        cdf = np.cumsum(p[idx])
        cut = np.searchsorted(cdf, top_p * p.sum(), side="right")
        keep = idx[: max(1, cut)]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = p * mask
    s = p.sum()
    if s == 0:
        mask = p > 0
        p = mask.astype(np.float64) / max(1, mask.sum())
    else:
        p /= s
    return p.astype(np.float32)


def expmin_choose(p, u):
    eps = 1e-12
    scores = -np.log(np.clip(u, eps, 1.0 - eps)) / np.clip(p, eps, 1.0)
    return int(np.argmin(scores))


def _watermarked_generate_iter(
    req: GenerateRequest, llm: ModelBackend
) -> T.Iterator[dict]:
    vanilla_text: str | None = None
    vanilla_ids: T.List[int] | None = None

    if req.return_vanilla:
        vanilla_len = int(req.vanilla_m or req.m)
        vanilla_text_only = llm.completion(
            prompt=PROMPT_WRAPPER(req.prompt),
            max_tokens=vanilla_len,
            temperature=float(req.temperature),
            top_p=float(req.top_p),
        )
        vanilla_text = vanilla_text_only
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

    import hashlib

    h = hashlib.sha256(str(req.key).encode("utf-8") + b":shift").digest()
    shift0 = int.from_bytes(h[:8], "big") % req.n

    for step in range(m):
        pos = len(token_ids)

        logits = llm.last_logits()
        if logits is None or len(logits) != V:
            raise HTTPException(
                500,
                "Backend must expose full next-token logits (length V) for exp-min watermarking.",
            )

        p = apply_temperature_and_nucleus(
            np.array(logits, dtype=np.float64),
            temperature=float(req.temperature),
            top_p=(float(req.top_p) if req.top_p else None),
        )

        row = (shift0 + step) % req.n
        u = xi[row]
        chosen_id = expmin_choose(p, u)

        token_ids.append(chosen_id)
        llm.eval([chosen_id])

        token_text = llm.detokenize([chosen_id])

        yield {
            "type": "token",
            "step": step,
            "text": token_text,
            "vanilla_text": vanilla_text,
            "id": chosen_id,
            "prompt_offset": prompt_offset,
            "total_steps": m,
        }
        step += 1

    llm.reset()

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
