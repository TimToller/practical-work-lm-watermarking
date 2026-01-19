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
from backend.core.detect import detect_stat

router = APIRouter()


def _detect_iter(req: DetectRequest, llm: ModelBackend) -> Iterator[Dict[str, Any]]:
    t0 = time.time()

    tokens = np.array(llm.tokenize(req.text), dtype=np.int32)
    m = int(tokens.shape[0])
    k = int(req.k if req.k is not None else m)
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
    }

    rng = RandomGenerator(key)
    xi = np.array(
        [rng.rand() for _ in range(n * vocab_size)], dtype=np.float32
    ).reshape(n, vocab_size)
    test_result = detect_stat(tokens, n, k, xi, gamma=gamma)
    yield {
        "type": "progress",
        "stage": "observed",
        "test_result": float(test_result),
        "runs_completed": 0,
    }

    p = 0
    every = 5
    for i in range(1, n_runs + 1):
        xi_alt = np.random.rand(n, vocab_size).astype(np.float32)
        null = detect_stat(tokens, n, k, xi_alt, gamma=gamma)
        if null <= test_result:
            p += 1

        if (i % every == 0) or (i == n_runs):
            p_hat = (p + 1.0) / (i + 1.0)  # streaming estimate
            yield {
                "type": "progress",
                "stage": "comparing",
                "runs_completed": i,
                "p_hat": float(p_hat),
                "elapsed": time.time() - t0,
            }

    pval = (p + 1.0) / (n_runs + 1.0)

    watermarked_status = "unlikely"
    if pval < 0.01:
        watermarked_status = "very likely"
    elif pval < 0.05:
        watermarked_status = "likely"

    elapsed = time.time() - t0
    final_payload = DetectResponse(
        p_value=float(pval),
        length=m,
        n=n,
        k=k,
        watermarked=watermarked_status,
        elapsed=elapsed,
    )
    yield {"type": "final", "payload": final_payload.model_dump()}


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
