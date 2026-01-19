from __future__ import annotations

import numpy as np


def log1m_clip(x: np.ndarray) -> np.ndarray:
    """
    Stable log(1 - x) for x in [0,1). Avoid log(0).
    """
    return np.log1p(-np.clip(x, 0.0, 1.0 - 1e-7)).astype(np.float32)


def candidate_offsets_by_diag(
    inv: np.ndarray, log_w: np.ndarray, n: int, max_offsets: int | None
) -> np.ndarray:
    """
    Cheap pre-screen: score each offset by diagonal-only alignment cost.
    Pick the best `max_offsets` offsets to run full DP on.

    inv: (k,) int indices into uniq-token dimension (U)
    log_w: (n, U) float32 with log(1 - xi)
    """
    if max_offsets is None or max_offsets >= n:
        return np.arange(n, dtype=np.int32)

    k = inv.shape[0]
    offsets = np.arange(n, dtype=np.int32)
    j = np.arange(k, dtype=np.int32)[None, :]  # (1,k)

    rows = (offsets[:, None] + j) % n  # (n,k)
    # gather log_w[row, inv[pos]]
    scores = log_w[rows, inv[None, :]].sum(axis=1)  # (n,)

    # We want *smallest* (most negative) scores
    idx = np.argpartition(scores, max_offsets)[:max_offsets]
    idx = idx[np.argsort(scores[idx])]
    return offsets[idx]


def exp_levenshtein_cost_inv(
    inv: np.ndarray, log_w: np.ndarray, offset: int, n: int, gamma: float
) -> float:
    """
    Levenshtein-style DP between token sequence (length k) and watermark rows (length k),
    where row at position j is (offset + j) % n.
    inv are indices into uniq-token set; costs are log_w[row, inv[i]].

    Rolling 2-row DP for speed.
    """
    k = inv.shape[0]
    prev = np.arange(k + 1, dtype=np.float32) * gamma
    curr = np.empty_like(prev)

    # row index for each watermark position j (0..k-1)
    rows = (offset + np.arange(k, dtype=np.int32)) % n

    for i in range(1, k + 1):
        curr[0] = i * gamma
        ti = inv[i - 1]

        for j in range(1, k + 1):
            cost = log_w[rows[j - 1], ti]

            a = prev[j] + gamma  # deletion
            b = curr[j - 1] + gamma  # insertion
            c = prev[j - 1] + cost  # diagonal

            m1 = a if a < b else b
            curr[j] = c if c < m1 else m1

        prev, curr = curr, prev

    return float(prev[k])


def detect_stat_prepared(
    inv: np.ndarray,
    log_w: np.ndarray,
    n: int,
    gamma: float,
    max_offsets: int | None = 16,
) -> float:
    """
    Compute the detection statistic as min over offsets of exp_levenshtein_cost.
    Uses a diagonal-only pre-screen to keep it fast.
    """
    offsets = candidate_offsets_by_diag(inv, log_w, n, max_offsets=max_offsets)
    best = float("inf")
    for off in offsets:
        val = exp_levenshtein_cost_inv(inv, log_w, int(off), n, gamma)
        if val < best:
            best = val
    return best
