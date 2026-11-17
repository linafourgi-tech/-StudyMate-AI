"""Pydantic data contracts shared between the LLM backend and other components.

These models are the integration interface for the rest of the team:

* ``LectureChunk``  - a piece of lecture text (produced by PDF processing).
* ``LLMRequest``    - what a component asks the LLM to do.
* ``LLMResponse``   - validated result returned by ``LLMService.generate``.
* ``TokenUsage``    - token accounting (useful for benchmarking/cost tracking).
* ``LectureContext`` - which chunks were actually placed in a prompt.
* ``GroundedRequest`` - an ``LLMRequest`` together with its ``LectureContext``.
* ``CitationReport`` - citations found in an answer, checked against the context.
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
    topic: str | None = Field(
        default=None,
        description="Topic this chunk belongs to (e.g. 'Trees'); used for topic-based "
                    "quiz planning. Matched case-insensitively, ignoring extra whitespace.",
    )
    lecture_id: str | None = Field(default=None, description="ID of the lecture this chunk is from.")

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text must not be blank")
        return value

    @field_validator("topic", "lecture_id")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank when provided (omit it or use None)")
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


class LectureContext(BaseModel):
    """The lecture block placed in a prompt, and which chunks it really contains.

    Chunks that did not fit the character budget are listed in ``omitted``
    (they were not sent). ``prompt_ids`` maps each citation ID as written in
    the prompt (unsafe characters replaced with ``_``) to the original
    ``chunk_id``, so model citations can be traced back to PDF chunks.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    included: tuple[LectureChunk, ...]
    omitted: tuple[LectureChunk, ...] = ()
    truncated: bool = Field(
        default=False, description="True if the first chunk alone was too long and was cut."
    )
    prompt_ids: dict[str, str] = Field(default_factory=dict)

    @property
    def included_ids(self) -> tuple[str, ...]:
        """Original ``chunk_id`` of every chunk that was sent, in prompt order."""
        return tuple(chunk.chunk_id for chunk in self.included)

    @property
    def omitted_ids(self) -> tuple[str, ...]:
        """Original ``chunk_id`` of every chunk that was not sent."""
        return tuple(chunk.chunk_id for chunk in self.omitted)

    @property
    def is_complete(self) -> bool:
        """True when every chunk was sent in full."""
        return not self.omitted and not self.truncated


class GroundedRequest(BaseModel):
    """Result of ``prepare_grounded_request``: the request plus its context."""

    model_config = ConfigDict(frozen=True)

    request: LLMRequest
    context: LectureContext


class CitationReport(BaseModel):
    """Citations found in a model answer, checked against a ``LectureContext``."""

    model_config = ConfigDict(frozen=True)

    cited: tuple[str, ...] = Field(description="IDs as written by the model, first-seen order.")
    valid: tuple[str, ...] = Field(description="Original chunk_ids of citations that were sent.")
    unknown: tuple[str, ...] = Field(description="Cited IDs that were not in the prompt.")

    @property
    def has_citations(self) -> bool:
        return bool(self.cited)

    @property
    def all_valid(self) -> bool:
        """True when the answer cites at least one chunk and every citation was sent."""
        return bool(self.cited) and not self.unknown
