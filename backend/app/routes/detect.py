from fastapi import APIRouter, Depends
import time
import numpy as np
import json
from typing import Iterator, Dict, Any
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.schemas import DetectRequest, DetectResponse
from backend.app.deps import get_backend
from backend.model.backend import ModelBackend
from backend.core.random_util import RandomGenerator

from backend.core.xi import xi_u_matrix
from backend.core.detect import log1m_clip, detect_stat_prepared

router = APIRouter()


def _detect_iter(req, llm) -> Iterator[Dict[str, Any]]:
    t0 = time.time()

    tokens = np.array(llm.tokenize(req.text), dtype=np.int32)
    m = int(tokens.shape[0])

    if req.k is None:
        k = m  # default: use all tokens
    else:
        k = min(int(req.k), m)

    vocab_size = int(llm.vocab_size)
    n = int(req.n)
    n_runs = int(req.n_runs)
    gamma = float(req.gamma)
    key = req.key

    yield {
        "type": "init",
        "length": m,
        "k": k,
        "n": n,
        "n_runs": n_runs,
        "vocab_size": vocab_size,
        "gamma": gamma,
        "max_offsets": 16,
    }

    block = tokens[:k]
    uniq, inv = np.unique(block, return_inverse=True)
    inv = inv.astype(np.int32)
    U = int(uniq.shape[0])

    rows = np.arange(n, dtype=np.int32)
    xi_small = xi_u_matrix(key, rows, uniq)
    log_w = log1m_clip(xi_small)

    MAX_OFFSETS = 16
    test_result = detect_stat_prepared(
        inv, log_w, n=n, gamma=gamma, max_offsets=MAX_OFFSETS
    )

    yield {
        "type": "progress",
        "stage": "observed",
        "test_result": float(test_result),
        "runs_completed": 0,
        "U": U,
    }

    null_rng = np.random.default_rng()

    p = 0
    every = 5

    TH_VERY = 0.01
    TH_LIKELY = 0.05
    early_stop_reason = None

    for i in range(1, n_runs + 1):
        xi_alt_small = null_rng.random((n, U), dtype=np.float32)
        log_w_alt = log1m_clip(xi_alt_small)

        null = detect_stat_prepared(
            inv, log_w_alt, n=n, gamma=gamma, max_offsets=MAX_OFFSETS
        )
        if null <= test_result:
            p += 1

        remaining = n_runs - i
        pval_min = (p + 1.0) / (n_runs + 1.0)  # if remaining never beat
        pval_max = (p + remaining + 1.0) / (n_runs + 1.0)  # if remaining always beat

        if pval_max < TH_VERY:
            early_stop_reason = "definitely_very_likely"
            i_final = i
            break
        if pval_min > TH_LIKELY:
            early_stop_reason = "definitely_unlikely"
            i_final = i
            break

        if (i % every == 0) or (i == n_runs):
            p_hat = (p + 1.0) / (i + 1.0)
            yield {
                "type": "progress",
                "stage": "comparing",
                "runs_completed": i,
                "p_hat": float(p_hat),
                "elapsed": time.time() - t0,
            }
    else:
        i_final = n_runs

    # Final p-value (smoothed). If early-stopped, we still report the full-run bound.
    pval = (p + 1.0) / (n_runs + 1.0)

    watermarked_status = "unlikely"
    if pval < TH_VERY:
        watermarked_status = "very likely"
    elif pval < TH_LIKELY:
        watermarked_status = "likely"

    elapsed = time.time() - t0
    final_payload = {
        "p_value": float(pval),
        "length": m,
        "n": n,
        "k": k,
        "watermarked": watermarked_status,
        "elapsed": elapsed,
        # extra demo telemetry:
        "runs_completed": int(i_final),
        "early_stop_reason": early_stop_reason,
        "U": U,
        "max_offsets": MAX_OFFSETS,
    }
    yield {"type": "final", "payload": final_payload}


@router.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest, llm: ModelBackend = Depends(get_backend)):
    final_dict = None
    for event in _detect_iter(req, llm):
        if event.get("type") == "final":
            final_dict = event["payload"]
    if final_dict is None:
        raise HTTPException(500, "Detection failed to produce a final payload.")
    return DetectResponse(**final_dict)


def _sse_format(event: Dict[str, Any]) -> str:
    et = event.get("type", "message")
    if et == "final":
        data = {**event["payload"]}
    else:
        data = {k: v for k, v in event.items() if k != "type"}
    return f"event: {et}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/detect/stream")
def detect_stream(req: DetectRequest, llm: ModelBackend = Depends(get_backend)):
    def event_iter():
        try:
            for event in _detect_iter(req, llm):
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

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
