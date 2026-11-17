"""Google Gemini backend for ``LLMService`` (official Google Gen AI SDK).

Selected with ``LLM_PROVIDER=gemini``. ``LLMService`` keeps owning retries,
the time budget, response validation and structured output; this module only:

* builds the ``generate_content`` call (system instruction, temperature,
  output token limit, JSON mode, optional thinking level, per-attempt timeout),
* normalises the response to the same shape as the OpenAI backend
  (finish reasons ``stop`` / ``length`` / ``content_filter``; token usage),
* maps SDK and transport exceptions to StudyMate exceptions.

Verified against google-genai 3.3.0 (installed source):
* with ``HttpOptions.retry_options=None`` the synchronous client makes exactly
  one attempt, so there is no hidden second retry layer;
* ``HttpOptions.timeout`` is in **milliseconds**;
* the API key is sent in the ``x-goog-api-key`` header, not in the URL;
* timeouts/connection failures surface as raw ``httpx2`` exceptions, and a
  non-JSON 200 body as a raw ``json.JSONDecodeError``;
* ``str(APIError)`` contains the full provider response body, so our error
  messages never reuse it.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx2

from .config import ConfigurationError, Settings
from .llm_service import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    LLMUnavailableError,
    _Completion,
    _MappedError,
)
from .schemas import LLMRequest, TokenUsage

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - exercised via a test that hides the SDK
    raise ConfigurationError(
        "LLM_PROVIDER=gemini requires the Google Gen AI SDK: pip install google-genai"
    ) from exc

try:  # The SDK uses httpx2; also accept httpx if a different transport is installed.
    import httpx

    _TIMEOUT_EXC: tuple[type[BaseException], ...] = (httpx2.TimeoutException, httpx.TimeoutException)
    _TRANSPORT_EXC: tuple[type[BaseException], ...] = (httpx2.TransportError, httpx.TransportError)
except ImportError:
    _TIMEOUT_EXC = (httpx2.TimeoutException,)
    _TRANSPORT_EXC = (httpx2.TransportError,)

# Gemini finish reasons that mean the output was blocked by a safety/policy filter.
_BLOCKED_FINISH = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "MODEL_ARMOR",
}
_RETRY_DELAY = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")
_RETRYABLE_STATUS = {408, 409}


def _normalise_finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", str(value)).upper()
    if name == "STOP":
        return "stop"
    if name == "MAX_TOKENS":
        return "length"
    if name in _BLOCKED_FINISH:
        return "content_filter"
    return name.lower()


def _error_details(exc: Any) -> list[dict[str, Any]]:
    """Return the ``error.details`` list of a Gemini error body (empty if absent)."""
    body = getattr(exc, "details", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    return [d for d in details if isinstance(d, dict)] if isinstance(details, list) else []


def _retry_delay_seconds(details: list[dict[str, Any]]) -> float | None:
    """Parse ``google.rpc.RetryInfo.retryDelay`` (e.g. ``"33s"``)."""
    for item in details:
        if str(item.get("@type", "")).endswith("RetryInfo"):
            match = _RETRY_DELAY.match(str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


def _is_daily_quota(details: list[dict[str, Any]]) -> bool:
    """True when a ``QuotaFailure`` names a per-day quota (retrying cannot succeed today)."""
    for item in details:
        if str(item.get("@type", "")).endswith("QuotaFailure"):
            for violation in item.get("violations") or []:
                if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                    return True
    return False


_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _error_reason(details: list[dict[str, Any]]) -> str | None:
    """Return ``google.rpc.ErrorInfo.reason`` if it is a plain enum-style token."""
    for item in details:
        if str(item.get("@type", "")).endswith("ErrorInfo"):
            reason = str(item.get("reason", ""))
            if _REASON.match(reason):
                return reason
    return None


def _is_api_key_error(code: int, status: str, details: list[dict[str, Any]]) -> bool:
    if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return True
    # An invalid Gemini API key is reported as 400 INVALID_ARGUMENT with this reason.
    return any(
        str(item.get("@type", "")).endswith("ErrorInfo")
        and str(item.get("reason", "")).startswith("API_KEY_")
        for item in details
    )


class GeminiBackend:
    """Adapter between ``LLMService`` and ``google.genai.Client``."""

    error_types: tuple[type[BaseException], ...] = (
        genai_errors.APIError,
        genai_errors.UnknownApiResponseError,
        json.JSONDecodeError,
        *_TRANSPORT_EXC,
    )

    def __init__(self, settings: Settings, client: Any | None) -> None:
        self.settings = settings
        if client is None:
            client = genai.Client(
                api_key=settings.api_key.get_secret_value(),
                vertexai=False,  # Gemini Developer API (AI Studio key), never Vertex AI
                http_options=genai_types.HttpOptions(
                    base_url=settings.base_url,
                    timeout=_to_ms(settings.timeout_seconds),
                    retry_options=None,  # retries are handled by LLMService
                ),
            )
        self.client = client

    @property
    def api_version(self) -> str | None:
        """The API version the client sends requests to (e.g. ``v1``), if known.

        google-genai 3.x defaults the Gemini Developer API to ``v1`` (2.x used
        ``v1beta``); ``GOOGLE_GENAI_API_VERSION`` can change it. Read from the
        SDK client for diagnostics only; ``None`` for clients that do not expose it.
        """
        options = getattr(getattr(self.client, "_api_client", None), "_http_options", None)
        version = getattr(options, "api_version", None)
        return version if isinstance(version, str) and version else None

    def build_config(self, request: LLMRequest, timeout: float) -> Any:
        level = self.settings.gemini_thinking_level
        return genai_types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            temperature=(
                request.temperature if request.temperature is not None
                else self.settings.temperature
            ),
            max_output_tokens=request.max_output_tokens or self.settings.max_output_tokens,
            response_mime_type="application/json" if request.response_format == "json" else None,
            thinking_config=(
                genai_types.ThinkingConfig(thinking_level=level.upper()) if level else None
            ),
            http_options=genai_types.HttpOptions(timeout=_to_ms(timeout)),
        )

    def complete(self, request: LLMRequest, timeout: float) -> _Completion:
        response = self.client.models.generate_content(
            model=self.settings.model,
            contents=request.user_prompt,
            config=self.build_config(request, timeout),
        )
        usage = getattr(response, "usage_metadata", None)
        prompt = getattr(usage, "prompt_token_count", None) or 0
        output = getattr(usage, "candidates_token_count", None) or 0
        thoughts = getattr(usage, "thoughts_token_count", None) or 0
        tokens = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=output + thoughts,  # thinking tokens are billed as output
            total_tokens=getattr(usage, "total_token_count", None) or prompt + output + thoughts,
        )
        model = getattr(response, "model_version", None)

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            blocked = getattr(feedback, "block_reason", None) is not None
            return _Completion(False, None, "content_filter" if blocked else None, False,
                               model, tokens)

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        # Only answer text: thought-summary parts are excluded.
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        return _Completion(
            has_choice=True,
            content=text or None,
            finish_reason=_normalise_finish_reason(getattr(candidate, "finish_reason", None)),
            refusal=False,  # Gemini reports refusals as blocked finish reasons
            model=model,
            usage=tokens,
        )

    def map_error(self, exc: BaseException) -> _MappedError:
        if isinstance(exc, _TIMEOUT_EXC):
            return _MappedError(LLMTimeoutError("The LLM request timed out."), retryable=True)
        if isinstance(exc, _TRANSPORT_EXC):
            return _MappedError(
                LLMConnectionError("Could not connect to the LLM provider."), retryable=True
            )
        if isinstance(exc, (genai_errors.UnknownApiResponseError, json.JSONDecodeError)):
            return _MappedError(LLMResponseError("The LLM provider returned an unreadable response."))
        if not isinstance(exc, genai_errors.APIError):  # pragma: no cover - defensive
            return _MappedError(LLMServiceError(f"Unexpected LLM provider error: {type(exc).__name__}"))

        code = int(getattr(exc, "code", 0) or 0)
        status = str(getattr(exc, "status", "") or "")
        details = _error_details(exc)

        def err(cls: type[LLMServiceError], message: str) -> LLMServiceError:
            return cls(message, status_code=code or None)

        if _is_api_key_error(code, status, details):
            return _MappedError(err(
                LLMAuthenticationError,
                "The LLM provider rejected the API key or permissions. Check GEMINI_API_KEY.",
            ))
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            if _is_daily_quota(details):
                return _MappedError(err(
                    LLMRequestError,
                    "The Gemini daily request quota is exhausted; it resets daily.",
                ))
            delay = _retry_delay_seconds(details)
            # If Google asks for a longer wait than we are willing to sleep, an early
            # retry would be rejected again and waste quota (Free Tier: 5 RPM, 20 RPD).
            retryable = delay is None or delay <= self.settings.retry_max_delay
            return _MappedError(
                err(LLMRateLimitError, "The LLM provider rate-limited the request."),
                retry_after=delay, retryable=retryable,
            )
        if code == 404:
            # A 404 does not by itself say *why* (unknown model ID, model not served
            # on this API version, or not offered to this key/project). Report the
            # facts we know -- requested model, API version, Google's status enum and
            # ErrorInfo reason -- but never Google's free-text message.
            facts = [f"model '{self.settings.model}'"]
            version = self.api_version
            if version:
                facts.append(f"API version {version}")
            if _REASON.match(status):
                facts.append(f"status {status}")
            reason = _error_reason(details)
            if reason:
                facts.append(f"reason {reason}")
            return _MappedError(err(
                LLMRequestError,
                f"Gemini generateContent returned HTTP 404 ({', '.join(facts)}). The model "
                "is unknown, not served on this API version, or not available to this key. "
                "Check LLM_MODEL; the provider's message is on the exception's __cause__.",
            ))
        if code >= 500 or code in _RETRYABLE_STATUS:
            return _MappedError(
                err(LLMUnavailableError, f"The LLM provider returned HTTP {code}."),
                retryable=True,
            )
        return _MappedError(
            err(LLMRequestError, f"The LLM provider rejected the request (HTTP {code}).")
        )


def _to_ms(seconds: float) -> int:
    """Convert seconds to the SDK's millisecond timeout (at least 1 ms)."""
    return max(1, int(seconds * 1000))
