from __future__ import annotations
import numpy as np


def exp_min_choose(probs: np.ndarray, u_row: np.ndarray) -> int:
    """Given probs (V,), u_row (V,), return argmax u^(1/p)."""
    eps = 1e-9
    inv = 1.0 / np.clip(probs, eps, 1.0)
    scores = np.power(u_row, inv)
    return int(np.argmax(scores))
