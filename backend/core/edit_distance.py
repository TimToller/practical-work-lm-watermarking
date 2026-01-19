from __future__ import annotations
import math
import numpy as np

try:  # optional Cython acceleration drop-in
    from .edit_distance_cy import levenshtein_cost as cy_levenshtein_cost  # type: ignore
except Exception:  # pragma: no cover
    cy_levenshtein_cost = None


def levenshtein_cost(x: np.ndarray, y: np.ndarray, gamma: float = 0.0) -> float:
    """
    x: shape (n,) int tokens
    y: shape (m, V) float32 probs row for each of the m positions (V = vocab size)
    cost(i,j) = min of insert/delete (penalty gamma) or diag + log(1 - y[j-1, x[i-1]]).
    Returns A[n, m].
    """
    if cy_levenshtein_cost is not None:
        return float(cy_levenshtein_cost(x, y, gamma))

    n = x.shape[0]
    m = y.shape[0]
    A = np.empty((n + 1, m + 1), dtype=np.float32)
    for i in range(n + 1):
        A[i, 0] = i * gamma
    for j in range(m + 1):
        A[0, j] = j * gamma
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            pj = y[j - 1, xi]
            cost = math.log(max(1e-20, 1.0 - float(pj)))
            a = A[i - 1, j] + gamma
            b = A[i, j - 1] + gamma
            c = A[i - 1, j - 1] + cost
            A[i, j] = a if a < b else b
            if c < A[i, j]:
                A[i, j] = c
    return float(A[n, m])
