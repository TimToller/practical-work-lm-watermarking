import time
import numpy as np
from fastapi import APIRouter, Depends

from backend.app.schemas import DetectRequest, DetectResponse
from backend.app.deps import get_backend
from backend.model.backend import ModelBackend
from backend.core.detect import permutation_test

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
def detect_endpoint(req: DetectRequest, llm: ModelBackend = Depends(get_backend)):
    # tokenize using model tokenizer to mirror generation
    toks = np.array(llm.tokenize(req.text), dtype=np.int32)
    m = int(toks.shape[0])
    k = int(req.k if req.k is not None else m)

    t0 = time.time()
    pval = permutation_test(
        toks,
        key=int(req.key),
        n=int(req.n),
        k=k,
        vocab_size=int(llm.vocab_size),
        n_runs=int(req.n_runs),
        gamma=float(req.gamma),
    )
    elapsed = time.time() - t0
    return DetectResponse(
        p_value=float(pval),
        length=m,
        n=int(req.n),
        k=k,
        watermarked=(pval < 0.01),
        elapsed=elapsed,
    )
