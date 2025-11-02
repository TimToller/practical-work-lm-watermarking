from fastapi import APIRouter, Depends
from backend.app.deps import get_backend, get_settings
from backend.model.backend import ModelBackend
from backend.core.settings import Settings


router = APIRouter()


@router.get("/model")
def model_info(
    llm: ModelBackend = Depends(get_backend),
    settings: Settings = Depends(get_settings),
):
    return {
        "vocab_size": llm.vocab_size,
        "n_ctx": settings.n_ctx,
        "n_threads": settings.n_threads,
        "n_gpu_layers": settings.n_gpu_layers,
        "model_path": settings.model_path,
    }
