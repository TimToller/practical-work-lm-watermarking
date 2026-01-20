from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from fastapi import APIRouter, HTTPException

from backend.app.schemas import CorruptRequest, CorruptResponse

router = APIRouter()

_WORD_OR_OTHER = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[^A-Za-zÀ-ÖØ-öø-ÿ0-9]+")


def _similarity(original: str, corrupted: str) -> float:
    if original == corrupted:
        return 1.0
    if not original and not corrupted:
        return 1.0
    if not original or not corrupted:
        return 0.0
    return max(0.0, min(1.0, float(SequenceMatcher(a=original, b=corrupted).ratio())))


def _is_word(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", tok))


def _mutate_word(rng: random.Random, w: str) -> str:
    if len(w) <= 1:
        return w

    op = rng.choice(("swap", "drop", "dup", "replace"))

    if op == "swap" and len(w) >= 2:
        i = rng.randrange(0, len(w) - 1)
        if w[i] == w[i + 1]:
            return w
        return w[:i] + w[i + 1] + w[i] + w[i + 2 :]

    if op == "drop":
        i = rng.randrange(0, len(w))
        return w[:i] + w[i + 1 :]

    if op == "dup":
        i = rng.randrange(0, len(w))
        return w[:i] + w[i] + w[i:]

    # replace
    i = rng.randrange(0, len(w))
    ch = w[i]

    if ch.isdigit():
        new = str((int(ch) + rng.choice((-1, 1))) % 10)
        return w[:i] + new + w[i + 1 :]

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    ALPHABET = alphabet.upper()
    new = rng.choice(ALPHABET if ch.isupper() else alphabet)
    if new == ch:
        return w
    return w[:i] + new + w[i + 1 :]


@dataclass(frozen=True)
class _CorruptionResult:
    corrupted_text: str
    similarity: float


def _corrupt_to_similarity_goal(
    text: str, goal: float, seed: Optional[int]
) -> _CorruptionResult:
    """
    Fast corruption: progressively mutate more word tokens until similarity <= goal.
    Returns the variant with similarity closest to goal (from above or below).
    """
    rng = random.Random(seed)

    tokens = _WORD_OR_OTHER.findall(text)
    if not tokens:
        return _CorruptionResult(text, 1.0)

    word_indices = [i for i, t in enumerate(tokens) if _is_word(t)]
    if not word_indices:
        return _CorruptionResult(text, 1.0)

    goal = max(0.0, min(1.0, float(goal)))

    # If goal is 1.0, no corruption needed.
    if goal >= 1.0:
        return _CorruptionResult(text, 1.0)

    # Shuffle deterministically so seed fully controls which words get hit first.
    rng.shuffle(word_indices)

    mutated = tokens[:]

    # Best = closest similarity to the goal (min absolute error)
    best_text = text
    best_sim = 1.0
    best_err = abs(best_sim - goal)

    # Heuristic starting point: similarity ~ 1 - fraction_changed, so fraction ~ (1-goal)
    # (SequenceMatcher isn't linear, but this is a good cheap initial guess.)
    start_k = max(1, int(round((1.0 - goal) * len(word_indices))))
    k = min(start_k, len(word_indices))

    def evaluate_current() -> None:
        nonlocal best_text, best_sim, best_err
        current_text = "".join(mutated)
        sim = _similarity(text, current_text)
        err = abs(sim - goal)
        if err < best_err:
            best_text, best_sim, best_err = current_text, sim, err

    # Apply initial batch
    for idx in word_indices[:k]:
        mutated[idx] = _mutate_word(rng, mutated[idx])
    evaluate_current()

    # If still too similar (> goal), keep mutating more words until we cross <= goal or run out
    i = k
    current_sim = best_sim
    while current_sim > goal and i < len(word_indices):
        idx = word_indices[i]
        mutated[idx] = _mutate_word(rng, mutated[idx])
        i += 1

        current_text = "".join(mutated)
        current_sim = _similarity(text, current_text)

        err = abs(current_sim - goal)
        if err < best_err:
            best_text, best_sim, best_err = current_text, current_sim, err

        # Early exit if we hit extremely close
        if best_err <= 0.002:
            break

    return _CorruptionResult(best_text, best_sim)


@router.post("/corrupt", response_model=CorruptResponse)
def corrupt(req: CorruptRequest):
    start = time.perf_counter()

    input_text = req.text
    if not input_text:
        raise HTTPException(status_code=400, detail="'text' must be non-empty")

    similarity_goal = float(req.similarity_goal)
    if not (0.0 <= similarity_goal <= 1.0):
        raise HTTPException(
            status_code=400, detail="'similarity_goal' must be between 0.0 and 1.0"
        )

    if similarity_goal == 1.0:
        elapsed = time.perf_counter() - start
        return CorruptResponse(
            corrupted_text=input_text,
            similarity=1.0,
            similarity_offset=0.0,
            elapsed=elapsed,
        )

    result = _corrupt_to_similarity_goal(
        text=input_text,
        goal=similarity_goal,
        seed=getattr(req, "seed", None),
    )

    elapsed = time.perf_counter() - start
    return CorruptResponse(
        corrupted_text=result.corrupted_text,
        similarity=result.similarity,
        similarity_offset=abs(result.similarity - similarity_goal),
        elapsed=elapsed,
    )
