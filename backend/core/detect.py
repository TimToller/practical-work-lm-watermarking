from .random_util import RandomGenerator
import numpy as np
import math


def levenshtein_cost(x: np.ndarray, y: np.ndarray, gamma: float = 0.0) -> float:
    n = x.shape[0]
    m = y.shape[0]
    A = np.empty((n + 1, m + 1), dtype=np.float32)
    for i in range(n + 1):
        A[i, 0] = i * gamma
    for j in range(m + 1):
        A[0, j] = j * gamma
    # DP
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


def detect_stat(
    tokens: np.ndarray, n: int, k: int, xi: np.ndarray, gamma: float = 0.0
) -> float:
    m = tokens.shape[0]
    k = min(k, m)
    if k <= 0:
        return 0.0

    best = float("inf")

    for start in range(0, m - k + 1):
        x = tokens[start : start + k]

        for offset in range(n):
            y = xi[(offset + np.arange(k)) % n]
            val = levenshtein_cost(x, y, gamma)
            if val < best:
                best = val

    return best
