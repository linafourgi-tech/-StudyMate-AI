"""LLM integration service for StudyMate AI.

``LLMService`` is the single entry point other components use to talk to the
LLM. It supports two providers, selected with ``LLM_PROVIDER``:

* ``openai`` (default): OpenAI SDK, implemented in ``_OpenAIBackend`` below.
* ``gemini``: Google Gen AI SDK, implemented in ``src/gemini_backend.py``.

Provider-independent behaviour lives in ``LLMService``:

* a per-attempt timeout and a total time budget per call (including retries),
* retries with capped exponential backoff and jitter for transient failures
  (timeouts, connection errors, rate limits, 5xx), honouring the provider's
  requested retry delay,
* mapping of provider exceptions to the project's own exception hierarchy,
  without exposing provider authentication details,
* validation of responses (non-empty content, finish reason, refusals,
  JSON/Pydantic validation for structured output).

The provider client, ``sleep`` and the clock are injectable so the service can
be tested fully offline.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .config import ConfigurationError, Settings, load_settings
from .grounding import is_not_in_material
from .prompts import NOT_IN_MATERIAL, NOT_IN_MATERIAL_JSON, NOT_IN_MATERIAL_KEY
from .schemas import LLMRequest, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "ConfigurationError",
    "LLMAuthenticationError",
    "LLMConnectionError",
    "LLMNotInMaterialError",
    "LLMRateLimitError",
    "LLMRequestError",
    "LLMResponseError",
    "LLMService",
    "LLMServiceError",
    "LLMTimeoutError",
    "LLMTransientError",
    "LLMUnavailableError",
    "get_llm_service",
]


# --------------------------------------------------------------------------- #
# Exceptions (callers catch these instead of provider-specific errors)
# --------------------------------------------------------------------------- #
class LLMServiceError(Exception):
    """Base class for all LLM service errors.

    Attributes:
        status_code: HTTP status from the provider, if any.
        request_id: Provider request ID (``x-request-id``), useful for support.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class LLMAuthenticationError(LLMServiceError):
    """Invalid/missing API key or insufficient permissions. Not retried."""


class LLMRequestError(LLMServiceError):
    """The request was rejected (e.g. bad parameters, quota exhausted). Not retried."""


class LLMResponseError(LLMServiceError):
    """The model returned an empty, truncated, filtered, refused or invalid response."""


class LLMNotInMaterialError(LLMResponseError):
    """Structured output: the model reported that the lecture material does not cover
    the request (the not-covered fallback), so there is no output to validate.

    A subclass of ``LLMResponseError``, so existing handlers keep working. Not
    re-asked: asking again would use another request for the same answer.
    """


class LLMTransientError(LLMServiceError):
    """A temporary failure; raised only after retries or the time budget are exhausted."""


class LLMTimeoutError(LLMTransientError):
    """The request exceeded the per-attempt timeout or the total time budget."""


class LLMConnectionError(LLMTransientError):
    """The provider could not be reached."""


class LLMRateLimitError(LLMTransientError):
    """The provider rate-limited the request."""


class LLMUnavailableError(LLMTransientError):
    """The provider returned a server-side (5xx) error."""


# --------------------------------------------------------------------------- #
# Provider backend contract (internal)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Completion:
    """A provider response normalised to one shape.

    ``finish_reason`` uses OpenAI-style values for both providers:
    ``"stop"``, ``"length"``, ``"content_filter"``, or another lower-case string.
    """

    has_choice: bool
    content: str | None
    finish_reason: str | None
    refusal: bool
    model: str | None
    usage: TokenUsage


@dataclass(frozen=True)
class _MappedError:
    """A provider exception translated into a StudyMate exception."""

    error: LLMServiceError
    retry_after: float | None = None  # provider-requested delay in seconds
    retryable: bool = False


class _Backend(Protocol):
    client: Any
    error_types: tuple[type[BaseException], ...]

    def complete(self, request: LLMRequest, timeout: float) -> _Completion: ...

    def map_error(self, exc: BaseException) -> _MappedError: ...


# --------------------------------------------------------------------------- #
# OpenAI backend
# --------------------------------------------------------------------------- #
_RETRYABLE_STATUS = {408, 409, 429}


def _retry_after_seconds(exc: Exception) -> float | None:
    """Return a numeric ``Retry-After`` header in seconds; ``None`` if absent or invalid."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):  # e.g. an HTTP-date; fall back to normal backoff
        return None


def _classify(exc: Exception) -> tuple[type[LLMServiceError], str]:
    """Choose the StudyMate exception type and a safe message for an OpenAI error.

    Messages are written by us and never include provider response text.
    """
    # Order matters: APITimeoutError is a subclass of APIConnectionError.
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError, "The LLM request timed out."
    if isinstance(exc, openai.APIConnectionError):
        return LLMConnectionError, "Could not connect to the LLM provider."
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return (
            LLMAuthenticationError,
            "The LLM provider rejected the API key or permissions. Check OPENAI_API_KEY.",
        )
    if isinstance(exc, openai.RateLimitError):
        if getattr(exc, "code", None) == "insufficient_quota":
            return LLMRequestError, "The LLM account has no remaining quota."
        return LLMRateLimitError, "The LLM provider rate-limited the request."
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status >= 500 or status in _RETRYABLE_STATUS:
            return LLMUnavailableError, f"The LLM provider returned HTTP {status}."
        return LLMRequestError, f"The LLM provider rejected the request (HTTP {status})."
    return LLMServiceError, f"Unexpected LLM provider error: {type(exc).__name__}"


def _map_provider_error(exc: Exception) -> LLMServiceError:
    """Translate an OpenAI SDK exception into a StudyMate exception."""
    error_type, message = _classify(exc)
    return error_type(
        message,
        status_code=getattr(exc, "status_code", None),
        request_id=getattr(exc, "request_id", None),
    )


class _OpenAIBackend:
    error_types: tuple[type[BaseException], ...] = (openai.OpenAIError,)

    def __init__(self, settings: Settings, client: Any | None) -> None:
        self.settings = settings
        if client is None:
            client = openai.OpenAI(
                api_key=settings.api_key.get_secret_value(),
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                # Retries are handled by LLMService so they are explicit and testable.
                max_retries=0,
            )
        self.client = client

    def build_params(self, request: LLMRequest, timeout: float) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": (
                request.temperature if request.temperature is not None
                else self.settings.temperature
            ),
            "max_tokens": request.max_output_tokens or self.settings.max_output_tokens,
            "timeout": timeout,
        }
        if request.response_format == "json":
            params["response_format"] = {"type": "json_object"}
        return params

    def complete(self, request: LLMRequest, timeout: float) -> _Completion:
        completion = self.client.chat.completions.create(**self.build_params(request, timeout))
        choices = getattr(completion, "choices", None)
        usage = getattr(completion, "usage", None)
        tokens = TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
        model = getattr(completion, "model", None)
        if not choices:
            return _Completion(False, None, None, False, model, tokens)
        choice = choices[0]
        message = getattr(choice, "message", None)
        return _Completion(
            has_choice=True,
            content=getattr(message, "content", None),
            finish_reason=getattr(choice, "finish_reason", None),
            refusal=bool(getattr(message, "refusal", None)),
            model=model,
            usage=tokens,
        )

    def map_error(self, exc: BaseException) -> _MappedError:
        error = _map_provider_error(exc)  # type: ignore[arg-type]
        return _MappedError(
            error=error,
            retry_after=_retry_after_seconds(exc),  # type: ignore[arg-type]
            retryable=isinstance(error, LLMTransientError),
        )


def _make_backend(settings: Settings, client: Any | None) -> _Backend:
    if settings.provider == "gemini":
        # Imported lazily: the OpenAI path works without google-genai installed.
        from .gemini_backend import GeminiBackend

        return GeminiBackend(settings, client)
    return _OpenAIBackend(settings, client)


_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence that some models add around JSON."""
    match = _CODE_FENCE.match(text.strip())
    return match.group(1) if match else text.strip()


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class LLMService:
    """High-level, validated access to the LLM.

    Args:
        settings: Configuration. Loaded from the environment when omitted.
        client: The provider SDK client (``openai.OpenAI`` or ``google.genai.Client``,
            matching ``settings.provider``). Created from ``settings`` when
            omitted; tests inject a mock.
        sleep: Function used for backoff waits (injected in tests).
        clock: Monotonic clock used for the time budget (injected in tests).
        rng: Random source for jitter (injected in tests for determinism).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_settings()
        self._backend = _make_backend(self.settings, client)
        self._client = self._backend.client
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Send ``request`` to the LLM and return a validated ``LLMResponse``.

        The whole call, including retries and backoff waits, is bounded by
        ``settings.total_timeout_seconds`` (approximately: each attempt's HTTP
        timeout is set from the remaining budget, but httpx timeouts apply per
        network operation, not to the whole request).

        Raises:
            LLMAuthenticationError, LLMRequestError: non-retryable failures.
            LLMTransientError (subclasses): retries or time budget exhausted.
            LLMResponseError: the response was empty, filtered, refused or truncated JSON.
        """
        return self._generate(request, self._clock() + self.settings.total_timeout_seconds)

    def generate_text(self, system_prompt: str, user_prompt: str, **options: Any) -> str:
        """Convenience wrapper returning only the response text."""
        request = LLMRequest(system_prompt=system_prompt, user_prompt=user_prompt, **options)
        return self.generate(request).content

    def generate_structured(self, request: LLMRequest, output_model: type[T]) -> T:
        """Request JSON output and validate it against a Pydantic model.

        The JSON Schema of ``output_model`` is appended to the system prompt.
        If the model returns invalid JSON or JSON that fails validation, the
        request is repeated up to ``settings.validation_retries`` extra times.
        All attempts share one ``total_timeout_seconds`` budget.

        Not-covered fallback (grounded requests): if the model answers with the
        not-covered fallback (``NOT_IN_MATERIAL_JSON`` or the plain phrase),
        ``LLMNotInMaterialError`` is raised at once, without a re-ask. To get it
        as data instead, give ``output_model`` a field
        ``not_in_material: bool = False`` and defaults for its other fields; the
        fallback is then returned as an instance with ``not_in_material=True``.

        Raises:
            LLMNotInMaterialError: the material does not cover the request.
            LLMResponseError: if no valid output is produced.
            (plus every error that ``generate`` can raise)
        """
        return self._generate_structured(request, output_model)[0]

    def generate_structured_with_response(
        self, request: LLMRequest, model_class: type[T]
    ) -> tuple[T, LLMResponse]:
        """Like ``generate_structured``, but also return the ``LLMResponse``.

        Same prompt, validation, re-asks, retries, time budget and exceptions
        as ``generate_structured``. The returned ``LLMResponse`` belongs to the
        generation whose output was accepted (validated, or returned as the
        opt-in not-covered fallback); its ``latency_ms``, ``attempts`` and
        ``usage`` describe that call only, not earlier rejected re-asks.
        Metadata is passed through as ``generate`` reports it: token counts the
        provider did not report are ``0`` (the ``TokenUsage`` convention), and a
        missing model name falls back to the configured ``LLM_MODEL``.

        Returns:
            ``(validated_instance, response)``.
        """
        return self._generate_structured(request, model_class)

    def _generate_structured(
        self, request: LLMRequest, output_model: type[T]
    ) -> tuple[T, LLMResponse]:
        schema = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
        system_prompt = (
            f"{request.system_prompt}\n\nRespond with a single JSON object only, "
            f"no extra text, that conforms to this JSON Schema:\n{schema}"
        )
        if NOT_IN_MATERIAL in request.system_prompt:  # grounded: allow the JSON fallback
            system_prompt += (
                "\nException: if the lecture material does not cover the request, respond "
                f"with exactly this JSON object instead: {NOT_IN_MATERIAL_JSON}"
            )
        structured_request = request.model_copy(
            update={"response_format": "json", "system_prompt": system_prompt}
        )

        deadline = self._clock() + self.settings.total_timeout_seconds
        last_error: ValidationError | None = None
        for attempt in range(1, self.settings.validation_retries + 2):
            response = self._generate(structured_request, deadline)
            content = _strip_code_fence(response.content)
            if is_not_in_material(content):
                return self._not_in_material_result(content, output_model), response
            try:
                return output_model.model_validate_json(content), response
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Structured output failed validation for %s (attempt %d): %d error(s)",
                    output_model.__name__, attempt, exc.error_count(),
                )

        error_count = last_error.error_count() if last_error else 0
        raise LLMResponseError(
            f"The model did not return valid {output_model.__name__} output "
            f"({error_count} validation error(s))."
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _not_in_material_result(content: str, output_model: type[T]) -> T:
        """Return the fallback as data if the schema opts in, else raise."""
        if NOT_IN_MATERIAL_KEY in output_model.model_fields:
            try:
                payload = json.loads(content)
            except ValueError:  # plain-text phrase
                payload = None
            if not isinstance(payload, dict):
                payload = {NOT_IN_MATERIAL_KEY: True, "message": NOT_IN_MATERIAL}
            try:
                return output_model.model_validate(payload)
            except ValidationError:
                pass  # schema has required fields the fallback cannot fill
        logger.info("Structured output: lecture material does not cover the request")
        raise LLMNotInMaterialError("The lecture material does not cover this request.")

    def _generate(self, request: LLMRequest, deadline: float) -> LLMResponse:
        max_attempts = self.settings.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise LLMTimeoutError("The LLM call exceeded its total time budget.")
            # Never let a single attempt run past the overall deadline.
            timeout = min(self.settings.timeout_seconds, remaining)

            start = time.perf_counter()
            try:
                completion = self._backend.complete(request, timeout)
            except self._backend.error_types as exc:
                mapped = self._backend.map_error(exc)
                error = mapped.error
                # Authentication errors are raised without the provider exception
                # attached: the provider's message can include a (masked) API key.
                cause = None if isinstance(error, LLMAuthenticationError) else exc
            else:
                latency_ms = (time.perf_counter() - start) * 1000
                response = self._validate_completion(completion, request, latency_ms, attempt)
                logger.info(
                    "LLM call ok: provider=%s model=%s attempts=%d latency=%.0fms tokens=%d",
                    self.settings.provider, response.model, attempt, latency_ms,
                    response.usage.total_tokens,
                )
                return response

            # Raised outside the ``except`` block so no implicit __context__ is attached.
            if not mapped.retryable or attempt == max_attempts:
                logger.error(
                    "LLM request failed: %s status=%s (attempt %d/%d)",
                    type(error).__name__, error.status_code, attempt, max_attempts,
                )
                self._raise(error, cause)

            delay = self._backoff_delay(attempt, mapped.retry_after)
            if self._clock() + delay >= deadline:
                logger.error(
                    "LLM request failed: %s; time budget exhausted after %d attempt(s)",
                    type(error).__name__, attempt,
                )
                self._raise(error, cause)

            logger.warning(
                "Transient LLM error %s (attempt %d/%d); retrying in %.2fs",
                type(error).__name__, attempt, max_attempts, delay,
            )
            self._sleep(delay)

        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _raise(error: LLMServiceError, cause: BaseException | None) -> None:
        if cause is None:
            raise error
        raise error from cause

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        """Capped exponential backoff with jitter; a provider-requested delay takes priority."""
        cap = self.settings.retry_max_delay
        if retry_after is not None:
            return min(retry_after, cap)
        delay = min(cap, self.settings.retry_base_delay * (2 ** (attempt - 1)))
        return delay * (0.5 + 0.5 * self._rng.random())  # jitter in [50%, 100%]

    def _validate_completion(
        self, completion: _Completion, request: LLMRequest, latency_ms: float, attempts: int
    ) -> LLMResponse:
        finish_reason = completion.finish_reason
        if finish_reason == "content_filter":
            raise LLMResponseError("The LLM response was blocked by the content filter.")
        if not completion.has_choice:
            raise LLMResponseError("The LLM response contained no choices.")

        content = completion.content
        if not isinstance(content, str) or not content.strip():
            if completion.refusal:
                raise LLMResponseError("The model refused to answer the request.")
            if finish_reason == "length":
                raise LLMResponseError(
                    "The LLM returned an empty response: the output token limit was reached "
                    "(for thinking models, raise LLM_MAX_OUTPUT_TOKENS or lower the thinking level)."
                )
            raise LLMResponseError("The LLM returned an empty response.")

        if finish_reason == "length" and request.response_format == "json":
            raise LLMResponseError(
                "The JSON response was truncated (max_output_tokens reached)."
            )
        if finish_reason == "length":
            logger.warning("LLM text response was truncated at max_output_tokens.")

        return LLMResponse(
            content=content.strip(),
            model=completion.model or self.settings.model,
            finish_reason=finish_reason,
            usage=completion.usage,
            latency_ms=latency_ms,
            attempts=attempts,
        )


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Return a shared ``LLMService`` configured from the environment.

    Other components should call this instead of constructing their own
    client, so configuration and connection pooling are shared. A
    ``ConfigurationError`` is not cached: fixing the configuration and calling
    again works. Tests can reset the instance with ``get_llm_service.cache_clear()``.
    """
    return LLMService()
