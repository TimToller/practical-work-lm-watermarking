# deps.py
from functools import lru_cache
import os

from fastapi import Depends
from backend.core.settings import Settings
from backend.model.llama_cpp_backend import LlamaCppBackend
from backend.model.backend import ModelBackend


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    if s.n_threads <= 0:
        s.n_threads = os.cpu_count() or 4
    return s


@lru_cache(maxsize=1)
def get_backend() -> ModelBackend:
    settings = get_settings()
    return LlamaCppBackend(settings)
