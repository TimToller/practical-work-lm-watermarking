from .random_util import RandomGenerator
import numpy as np
import math


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


def detect_stat(
    tokens: np.ndarray, n: int, k: int, xi: np.ndarray, gamma: float = 0.0
) -> float:
    """Compute min over a sliding window of size k across tokens vs rotating xi rows."""
    m = tokens.shape[0]
    if k <= 0 or m == 0:
        return 0.0
    best = float("inf")
    for i in range(0, m - (k - 1)):
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
        p += 1 if null <= test_result else 0
    return (p + 1.0) / (n_runs + 1.0)
