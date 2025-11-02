from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np

try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover
    Llama = None  # type: ignore

from backend.core.settings import Settings
from backend.model.backend import ModelBackend


class LlamaCppBackend(ModelBackend):
    def __init__(self, settings: Settings):
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. pip install llama-cpp-python"
            )
        if not os.path.exists(settings.model_path):
            raise RuntimeError(
                f"Model not found at {os.path.abspath(settings.model_path)}. Set MODEL_PATH env var."
            )
        self._llm = Llama(
            model_path=settings.model_path,
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            n_gpu_layers=settings.n_gpu_layers,
            logits_all=True,
            vocab_only=False,
            verbose=False,
        )
        self._vocab_size = (
            self._llm.n_vocab() if hasattr(self._llm, "n_vocab") else self._llm._n_vocab
        )

    @property
    def vocab_size(self) -> int:
        return int(self._vocab_size)

    def tokenize(self, text: str) -> List[int]:
        return list(self._llm.tokenize(text.encode("utf-8"), special=True))

    def detokenize(self, ids: List[int]) -> str:
        return self._llm.detokenize(ids).decode("utf-8", errors="ignore")

    def eval(self, ids: List[int]) -> None:
        self._llm.eval(ids)

    def last_logits(self) -> "np.ndarray | None":
        try:
            logits = self._llm._scores[-1]
            return np.array(logits)
        except Exception:
            return None

    def topn_logprobs(self, prompt: str, n: int) -> Tuple[List[int], np.ndarray]:
        comp = self._llm.create_completion(
            prompt=prompt,
            max_tokens=1,
            temperature=1.0,
            logprobs=int(n),
            stop=[],
        )
        info = comp["choices"][0]["logprobs"]["top_logprobs"][0]
        ids = [d.get("id") for d in info if d.get("id") is not None]
        lps = np.array([d["logprob"] for d in info], dtype=np.float32)
        return ids, lps

    def completion(self, prompt: str, max_tokens: int, **kwargs) -> str:
        comp = self._llm.create_completion(
            prompt=prompt, max_tokens=int(max_tokens), **kwargs
        )
        return comp["choices"][0]["text"]

    def reset(self) -> None:
        self._llm.reset()
