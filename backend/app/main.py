from fastapi import FastAPI
from backend.app.routes import health, model, generate, detect, corrupt
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def create_app() -> FastAPI:
    app = FastAPI(title="Exp-Min Watermark API (llama.cpp)", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # or ["*"] for all origins (dev only)
        allow_credentials=True,
        allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
        allow_headers=["*"],  # allow all request headers
    )

    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(generate.router)
    app.include_router(detect.router)
    app.include_router(corrupt.router)
    return app


app = create_app()
