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
    text: str = Field(..., description="Generated watermarked continuation")
    token_ids: T.List[int] = Field(
        ..., description="Token IDs for the generated watermarked continuation"
    )
    used_top_n: int | None = Field(
        None,
        description="Top-N actually used for probability approximation (if applicable)",
    )
    warning: str | None = Field(
        None,
        description="Optional warning message (e.g., when falling back to an approximation)",
    )
    vanilla_text: str | None = Field(
        None,
        description="Non-watermarked baseline continuation (only if requested)",
    )
    vanilla_token_ids: T.List[int] | None = Field(
        None,
        description="Token IDs for the non-watermarked baseline continuation (only if requested)",
    )
    elapsed: float = Field(..., description="Elapsed wall-clock time in seconds")


class DetectRequest(BaseModel):
    text: str = Field(..., description="Text to test for the presence of the watermark")
    n: int = Field(256, description="Watermark sequence length used during generation")
    key: int = Field(42, description="Watermark key used during generation")
    k: int | None = Field(
        None,
        description="Optional k parameter for the detector; if omitted, the server will choose a default",
    )
    gamma: float = Field(
        0.0,
        description="Optional detector parameter (gamma) controlling the test statistic",
    )
    n_runs: int = Field(
        500,
        description="Number of Monte Carlo / permutation runs used to estimate a p-value",
    )


class DetectResponse(BaseModel):
    p_value: float = Field(..., description="Estimated p-value of the watermark test")
    length: int = Field(..., description="Length of the analyzed token sequence")
    n: int = Field(..., description="Watermark sequence length used for detection")
    k: int = Field(..., description="k parameter used for detection")
    watermarked: str = Field(
        "unlikely",
        description="Watermarked status based on p-value. Either unlikely (p>0.05), likely (p<=0.05) or very likely (p<=0.01)",
    )
    elapsed: float = Field(..., description="Elapsed time")


class CorruptRequest(BaseModel):
    text: str = Field(..., description="Text to corrupt")
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
    corrupted_text: str = Field(..., description="Corrupted version of the input text")
    corruption_rate: float = Field(
        ..., description="Measured corruption rate achieved by the corruption procedure"
    )
    corruption_offset: float = Field(
        ...,
        description="Difference between achieved corruption_rate and requested corruption_goal",
    )
    elapsed: float = Field(..., description="Elapsed wall-clock time in seconds")
