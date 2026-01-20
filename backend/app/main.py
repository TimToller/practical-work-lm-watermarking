from fastapi import FastAPI
from backend.app.routes import health, model, generate, detect, corrupt
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def create_app() -> FastAPI:
    app = FastAPI(title="Exp-Min Watermark API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(generate.router)
    app.include_router(detect.router)
    app.include_router(corrupt.router)
    return app


app = create_app()
