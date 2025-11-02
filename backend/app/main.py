from fastapi import FastAPI
from backend.app.routes import health, model, generate, detect


def create_app() -> FastAPI:
    app = FastAPI(title="Exp-Min Watermark API (llama.cpp)", version="1.0.0")
    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(generate.router)
    app.include_router(detect.router)
    return app


app = create_app()
