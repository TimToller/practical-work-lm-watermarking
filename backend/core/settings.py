from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: str = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    n_ctx: int = 4096
    n_threads: int = 0  # 0 → auto cpu count
    n_gpu_layers: int = -1
    top_n_probs: int = 512

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
