from __future__ import annotations
from pydantic import BaseModel, Field
import typing as T


class GenerateRequest(BaseModel):
    prompt: str = Field("", description="Prompt text")
    m: int = Field(80, description="Number of new tokens to generate (watermarked)")
    n: int = Field(256, description="Watermark sequence length")
    key: int = Field(42, description="Watermark key")
    top_n: int | None = Field(
        None, description="Top-N to approximate probs if full logits not exposed"
    )
    include_system_prompt: bool = Field(
        True,
        description="If True, include the model's system prompt in the generation context",
    )
    return_vanilla: bool = Field(
        False, description="Also generate a non-watermarked baseline continuation"
    )
    vanilla_m: int | None = Field(
        None, description="Length of vanilla continuation; defaults to m if omitted"
    )
    temperature: float = Field(
        1.0, ge=0.0, description="Sampling temperature for vanilla baseline"
    )
    top_p: float = Field(
        1.0, ge=0.0, le=1.0, description="Nucleus top-p for vanilla baseline"
    )


class GenerateResponse(BaseModel):
    text: str
    token_ids: T.List[int]
    used_top_n: int | None = None
    warning: str | None = None
    vanilla_text: str | None = None
    vanilla_token_ids: T.List[int] | None = None
    elapsed: float


class DetectRequest(BaseModel):
    text: str
    n: int = 256
    key: int = 42
    k: int | None = None
    gamma: float = 0.0
    n_runs: int = 500


class DetectResponse(BaseModel):
    p_value: float
    length: int
    n: int
    k: int
    watermarked: str = Field(
        "unlikely", description="Watermarked status based on p-value"
    )
    elapsed: float


class CorruptRequest(BaseModel):
    text: str
    corruption_goal: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Target corruption rate in [0,1]. The endpoint will corrupt the text by back-translation "
            "of a subset of sentences until the measured corruption rate roughly reaches this goal."
        ),
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")


class CorruptResponse(BaseModel):
    corrupted_text: str
    corruption_rate: float
    corruption_offset: float
    elapsed: float
