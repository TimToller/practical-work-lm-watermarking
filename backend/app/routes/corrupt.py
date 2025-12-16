from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random
import re
import threading
import time
from difflib import SequenceMatcher
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.schemas import CorruptRequest, CorruptResponse

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

router = APIRouter()


_MODEL_LOCK = threading.Lock()
_TRANSLATORS: dict[tuple[str, str], tuple[Any, Any]] = {}

_INPUT_MAX_TOKENS = 512
_MAX_NEW_TOKENS = 512


def _device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model_id(src: str, tgt: str) -> str:
    # MarianMT / OPUS models
    mapping = {
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        ("fr", "ru"): "Helsinki-NLP/opus-mt-fr-ru",
        ("ru", "fr"): "Helsinki-NLP/opus-mt-ru-fr",
        ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    }
    try:
        return mapping[(src, tgt)]
    except KeyError as e:
        raise ValueError(f"Unsupported translation direction: {src}->{tgt}") from e


def _get_translator(src: str, tgt: str) -> tuple[Any, Any]:
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise RuntimeError(
            "Translation dependencies are not available. Ensure 'transformers' and 'torch' are installed."
        )

    key = (src, tgt)
    with _MODEL_LOCK:
        existing = _TRANSLATORS.get(key)
        if existing is not None:
            return existing

        model_id = _model_id(src, tgt)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(_device())
        model.eval()
        _TRANSLATORS[key] = (tokenizer, model)
        return tokenizer, model


def _translate(text: str, src: str, tgt: str) -> str:
    tokenizer, model = _get_translator(src, tgt)

    # Conservative cap: Marian models typically have small context windows.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=_INPUT_MAX_TOKENS,
    )
    if torch is not None:
        inputs = {k: v.to(_device()) for k, v in inputs.items()}

    ctx = torch.inference_mode() if torch is not None else nullcontext()
    with ctx:
        out = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=_MAX_NEW_TOKENS,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _corruption_rate(original: str, corrupted: str) -> float:
    # 0 => identical, 1 => completely different (approx. via SequenceMatcher similarity)
    if original == corrupted:
        return 0.0
    if not original and not corrupted:
        return 0.0
    if not original or not corrupted:
        return 1.0
    ratio = SequenceMatcher(a=original, b=corrupted).ratio()
    rate = 1.0 - float(ratio)
    if rate < 0.0:
        return 0.0
    if rate > 1.0:
        return 1.0
    return rate


def _split_sentences(text: str) -> list[str]:
    # Simple, punctuation-based sentence splitter that keeps trailing whitespace.
    # Examples: "Hi. Bye!" -> ["Hi. ", "Bye!"]
    chunks = re.findall(r"[^.!?]+[.!?]*\s*", text, flags=re.MULTILINE)
    # Fall back if regex fails oddly
    if not chunks:
        return [text]
    return chunks


def _back_translate_sentence(sentence: str, variant: int) -> str:
    trimmed = sentence.strip()
    if not trimmed:
        return sentence

    # Alternate paths to vary corruption slightly.
    # 0: EN -> FR -> EN
    # 1: EN -> FR -> RU -> EN
    # 2: EN -> FR -> RU -> FR -> EN
    paths: list[list[str]] = [
        ["en", "fr", "en"],
        ["en", "fr", "ru", "en"],
        ["en", "fr", "ru", "fr", "en"],
    ]
    path = paths[variant % len(paths)]

    out = trimmed
    for src, tgt in zip(path, path[1:]):
        out = _translate(out, src, tgt)

    trailing_ws = sentence[len(sentence.rstrip()) :]
    return out + trailing_ws


@dataclass(frozen=True)
class _CorruptionResult:
    corrupted_text: str
    corruption_rate: float


def _corrupt_to_goal(text: str, goal: float, seed: int | None) -> _CorruptionResult:
    rng = random.Random(seed)
    chunks = _split_sentences(text)
    if not chunks:
        return _CorruptionResult(corrupted_text=text, corruption_rate=0.0)

    n_sentences = len(chunks)
    indices = list(range(n_sentences))
    rng.shuffle(indices)

    # Translate an initial proportion of sentences, then increase until goal is met.
    # Start with k ≈ goal * n (at least 1).
    initial_k = max(1, int(round(goal * n_sentences)))
    mutated = chunks[:]

    def apply_one(sentence_index: int, variant: int) -> None:
        mutated[sentence_index] = _back_translate_sentence(
            mutated[sentence_index], variant
        )

    for variant, idx in enumerate(indices[:initial_k]):
        apply_one(idx, variant)

    corrupted_text = "".join(mutated)
    achieved = _corruption_rate(text, corrupted_text)

    variant = initial_k
    for idx in indices[initial_k:]:
        if achieved >= goal:
            break
        apply_one(idx, variant)
        variant += 1
        corrupted_text = "".join(mutated)
        achieved = _corruption_rate(text, corrupted_text)

    return _CorruptionResult(corrupted_text=corrupted_text, corruption_rate=achieved)


@router.post("/corrupt", response_model=CorruptResponse)
def corrupt(req: CorruptRequest):
    start = time.perf_counter()

    input_text = req.text
    corruption_goal = req.corruption_goal

    if not input_text:
        raise HTTPException(status_code=400, detail="'text' must be non-empty")

    if corruption_goal == 0.0:
        elapsed = time.perf_counter() - start
        return CorruptResponse(
            corrupted_text=input_text,
            corruption_rate=0.0,
            corruption_offset=0.0,
            elapsed=elapsed,
        )

    try:
        result = _corrupt_to_goal(
            text=input_text,
            goal=corruption_goal,
            seed=req.seed,
        )
        corrupted_text = result.corrupted_text
        corruption_rate = result.corruption_rate
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}") from e

    elapsed = time.perf_counter() - start
    return CorruptResponse(
        corrupted_text=corrupted_text,
        corruption_rate=corruption_rate,
        corruption_offset=abs(corruption_rate - corruption_goal),
        elapsed=elapsed,
    )
