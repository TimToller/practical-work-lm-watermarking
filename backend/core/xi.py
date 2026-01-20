from __future__ import annotations

import hashlib
import numpy as np

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

_SM64_C1 = np.uint64(0x9E3779B97F4A7C15)
_SM64_C2 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_C3 = np.uint64(0x94D049BB133111EB)

_ROW_MIX = np.uint64(0xD1342543DE82EF95)
_TOK_MIX = np.uint64(0xA24BAED4963EE407)


def seed64(key: str, salt: str) -> np.uint64:
    h = hashlib.sha256(f"{key}:{salt}".encode("utf-8")).digest()
    return np.uint64(int.from_bytes(h[:8], "big", signed=False))


def _splitmix64(x: np.ndarray) -> np.ndarray:
    z = (x + _SM64_C1) & _MASK64
    z = (z ^ (z >> np.uint64(30))) * _SM64_C2 & _MASK64
    z = (z ^ (z >> np.uint64(27))) * _SM64_C3 & _MASK64
    z = (z ^ (z >> np.uint64(31))) & _MASK64
    return z


def _u01_from_uint64(z: np.ndarray) -> np.ndarray:
    return ((z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))).astype(
        np.float32
    )


def xi_u_matrix(key: str, rows: np.ndarray, token_ids: np.ndarray) -> np.ndarray:

    base = seed64(key, "xi")
    r = rows.astype(np.uint64)[:, None]
    t = token_ids.astype(np.uint64)[None, :]
    x = base ^ (r * _ROW_MIX) ^ (t * _TOK_MIX)
    return _u01_from_uint64(_splitmix64(x))


def xi_u_vector(key: str, row: int, V: int) -> np.ndarray:
    """
    Deterministic U(0,1) vector for a specific row across the whole vocab.
    This is what generation needs for exp-min.
    """
    base = seed64(key, "xi")
    t = np.arange(V, dtype=np.uint64)
    x = base ^ (np.uint64(row) * _ROW_MIX) ^ (t * _TOK_MIX)
    return _u01_from_uint64(_splitmix64(x))
