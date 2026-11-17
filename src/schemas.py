"""Pydantic data contracts shared between the LLM backend and other components.

These models are the integration interface for the rest of the team:

* ``LectureChunk``  - a piece of lecture text (produced by PDF processing).
* ``LLMRequest``    - what a component asks the LLM to do.
* ``LLMResponse``   - validated result returned by ``LLMService.generate``.
* ``TokenUsage``    - token accounting (useful for benchmarking/cost tracking).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LectureChunk(BaseModel):
    """A unit of lecture material used to ground the model's answer."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1, max_length=100, description="Stable ID used for citations.")
    text: str = Field(min_length=1, description="Raw lecture text.")
    source: str | None = Field(default=None, description="e.g. file name of the lecture PDF.")
    page: int | None = Field(default=None, ge=1)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text must not be blank")
        return value


class LLMRequest(BaseModel):
    """A single request to the LLM.

    ``temperature`` and ``max_output_tokens`` fall back to configured defaults
    when left as ``None``. ``response_format="json"`` asks the provider for a
    JSON object (used by ``LLMService.generate_structured``).
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = Field(min_length=1, max_length=50_000)
    user_prompt: str = Field(min_length=1, max_length=200_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=32_000)
    response_format: Literal["text", "json"] = "text"

    @field_validator("system_prompt", "user_prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class TokenUsage(BaseModel):
    """Token counts reported by the provider (0 when not reported)."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    """Validated LLM output plus metadata for logging and benchmarking."""

    content: str = Field(min_length=1)
    model: str
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(ge=0, description="Latency of the successful attempt.")
    attempts: int = Field(ge=1, description="Total API attempts, including retries.")
