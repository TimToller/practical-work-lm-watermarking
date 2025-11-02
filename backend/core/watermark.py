from __future__ import annotations
import numpy as np
from .edit_distance import levenshtein_cost
from .random_util import RandomGenerator


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


def exp_min_choose(probs: np.ndarray, u_row: np.ndarray) -> int:
    """Given probs (V,), u_row (V,), return argmax u^(1/p)."""
    eps = 1e-9
    inv = 1.0 / np.clip(probs, eps, 1.0)
    scores = np.power(u_row, inv)
    return int(np.argmax(scores))
