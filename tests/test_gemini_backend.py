"""Offline tests for the Gemini provider (LLM_PROVIDER=gemini).

No network and no paid or quota-consuming calls: requests go through the real
``google.genai.Client`` with an in-memory ``httpx2.MockTransport``, so the
official SDK's request building, response parsing and error raising are
exercised exactly as in production, up to the HTTP layer.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import traceback
from unittest.mock import MagicMock

import httpx2
import pytest
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel

from src import config
from src.config import ConfigurationError, Settings, load_settings
from src.llm_service import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMService,
    LLMTimeoutError,
    LLMUnavailableError,
    get_llm_service,
)
from src.prompts import NOT_IN_MATERIAL, build_grounded_request
from src.schemas import LectureChunk, LLMRequest

FAKE_GEMINI_KEY = "AIzaFAKE-not-a-real-gemini-key-000"
REQUEST = LLMRequest(system_prompt="You are a tutor.", user_prompt="What is a stack?")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def gemini_settings(**overrides) -> Settings:
    values = {"provider": "gemini", "api_key": FAKE_GEMINI_KEY, "max_retries": 3,
              "retry_base_delay": 1.0, "retry_max_delay": 20.0}
    values.update(overrides)
    return Settings(**values)


def ok_body(text="A stack is LIFO.", finish="STOP", parts=None, usage=None, model="gemini-3.5-flash-lite"):
    return {
        "candidates": [{
            "content": {"role": "model", "parts": parts if parts is not None else [{"text": text}]},
            "finishReason": finish,
        }],
        "usageMetadata": usage if usage is not None else {
            "promptTokenCount": 20, "candidatesTokenCount": 6,
            "thoughtsTokenCount": 4, "totalTokenCount": 30,
        },
        "modelVersion": model,
    }


def api_error(code, status, message="error", details=None):
    return {"error": {"code": code, "status": status, "message": message, "details": details or []}}


RETRY_INFO = lambda delay: {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}  # noqa: E731
DAILY_QUOTA = {
    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
    "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
}
PER_MINUTE_QUOTA = {
    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
    "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}],
}
API_KEY_INVALID = {"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                   "reason": "API_KEY_INVALID", "domain": "googleapis.com"}


class Transport:
    """Scripted in-memory HTTP layer recording every request the SDK sends."""

    def __init__(self, responses):
        self.queue = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx2.Response):
            return item
        status, body = item
        return httpx2.Response(status, json=body)

    def body(self, index=-1) -> dict:
        return json.loads(self.requests[index].content)


_ROUTE: dict[str, Transport] = {}
_SHARED_CLIENT: list[genai.Client] = []


def _shared_client() -> genai.Client:
    """One real SDK client for the module; each test routes it to its own Transport.

    Building a ``genai.Client`` loads the system CA bundle (~0.3 s, no network),
    so reusing one keeps the suite fast while every test still gets fresh,
    isolated scripted responses.
    """
    if not _SHARED_CLIENT:
        mock = httpx2.MockTransport(lambda request: _ROUTE["current"](request))
        _SHARED_CLIENT.append(genai.Client(
            api_key=FAKE_GEMINI_KEY, vertexai=False,
            http_options={"httpx_client": httpx2.Client(transport=mock)},
        ))
    return _SHARED_CLIENT[0]


def make_gemini_service(responses, **overrides):
    transport = Transport(responses)
    _ROUTE["current"] = transport
    client = _shared_client()
    sleep = MagicMock()
    service = LLMService(gemini_settings(**overrides), client=client, sleep=sleep,
                         rng=random.Random(0))
    return service, transport, sleep


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in [*config._ENV_MAP, *config.API_KEY_ENV.values()]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", tmp_path / "missing.env")


# --------------------------------------------------------------------------- #
# Configuration / provider selection
# --------------------------------------------------------------------------- #
class TestGeminiConfig:
    def test_gemini_provider_uses_gemini_key_and_default_model(self):
        s = load_settings({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": FAKE_GEMINI_KEY})
        assert s.provider == "gemini"
        assert s.model == "gemini-3.5-flash-lite"  # default, verified live 2026-11-17
        assert s.api_key.get_secret_value() == FAKE_GEMINI_KEY

    def test_provider_name_is_case_insensitive(self):
        s = load_settings({"LLM_PROVIDER": " Gemini ", "GEMINI_API_KEY": FAKE_GEMINI_KEY})
        assert s.provider == "gemini"

    def test_openai_remains_the_default_provider(self):
        s = load_settings({"OPENAI_API_KEY": "sk-x", "GEMINI_API_KEY": FAKE_GEMINI_KEY})
        assert s.provider == "openai"
        assert s.model == "gpt-4o-mini"
        assert s.api_key.get_secret_value() == "sk-x"

    def test_gemini_requires_gemini_key_not_openai_key(self):
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is not set"):
            load_settings({"LLM_PROVIDER": "gemini", "OPENAI_API_KEY": "sk-x"})

    def test_model_override(self):
        s = load_settings({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": FAKE_GEMINI_KEY,
                           "LLM_MODEL": "gemini-3.6-flash"})
        assert s.model == "gemini-3.6-flash"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ConfigurationError, match="provider"):
            load_settings({"LLM_PROVIDER": "claude", "OPENAI_API_KEY": "sk-x"})

    @pytest.mark.parametrize("level", ["minimal", "LOW", "medium", "high"])
    def test_valid_thinking_levels(self, level):
        s = load_settings({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": FAKE_GEMINI_KEY,
                           "GEMINI_THINKING_LEVEL": level})
        assert s.gemini_thinking_level == level.lower()

    def test_invalid_thinking_level_rejected_without_leaking_key(self):
        with pytest.raises(ConfigurationError) as info:
            load_settings({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": FAKE_GEMINI_KEY,
                           "GEMINI_THINKING_LEVEL": "extreme"})
        assert "gemini_thinking_level" in str(info.value)
        assert FAKE_GEMINI_KEY not in str(info.value)

    def test_gemini_key_hidden_in_repr(self):
        assert FAKE_GEMINI_KEY not in repr(gemini_settings())

    def test_get_llm_service_selects_gemini(self, clean_env, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
        get_llm_service.cache_clear()
        try:
            service = get_llm_service()
            assert isinstance(service._client, genai.Client)
            assert service.settings.model == "gemini-3.5-flash-lite"
        finally:
            get_llm_service.cache_clear()


# --------------------------------------------------------------------------- #
# Client construction
# --------------------------------------------------------------------------- #
class TestGeminiClientConstruction:
    def test_default_client_options(self, monkeypatch):
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(genai, "Client", FakeClient)
        LLMService(gemini_settings(timeout_seconds=12.5, base_url="https://proxy.example.com/"))
        assert captured["api_key"] == FAKE_GEMINI_KEY
        assert captured["vertexai"] is False
        options = captured["http_options"]
        assert options.timeout == 12500  # milliseconds
        assert options.retry_options is None  # no SDK retries: LLMService owns them
        assert options.base_url == "https://proxy.example.com/"

    def test_real_client_is_built_without_network(self):
        service = LLMService(gemini_settings())
        assert isinstance(service._client, genai.Client)

    def test_missing_sdk_gives_configuration_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "google.genai", None)
        monkeypatch.delitem(sys.modules, "src.gemini_backend", raising=False)
        with pytest.raises(ConfigurationError, match="pip install google-genai"):
            LLMService(gemini_settings())

    def test_openai_path_does_not_import_gemini_backend(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "src.gemini_backend", raising=False)
        LLMService(Settings(api_key="sk-x"), client=MagicMock())
        assert "src.gemini_backend" not in sys.modules


# --------------------------------------------------------------------------- #
# Request mapping (what the SDK actually sends)
# --------------------------------------------------------------------------- #
class TestGeminiRequest:
    def test_text_request_payload(self):
        service, transport, _ = make_gemini_service([(200, ok_body())], timeout_seconds=7)
        service.generate(REQUEST)
        req = transport.requests[0]
        body = transport.body()
        assert req.url.path.endswith("/models/gemini-3.5-flash-lite:generateContent")
        assert body["systemInstruction"]["parts"][0]["text"] == "You are a tutor."
        assert body["contents"][0]["parts"][0]["text"] == "What is a stack?"
        assert body["generationConfig"]["temperature"] == 0.2
        assert body["generationConfig"]["maxOutputTokens"] == 1024
        assert "responseMimeType" not in body["generationConfig"]
        assert "thinkingConfig" not in body["generationConfig"]  # model default unless configured
        assert req.headers["x-server-timeout"] == "7"

    def test_api_key_sent_in_header_not_url(self):
        service, transport, _ = make_gemini_service([(200, ok_body())])
        service.generate(REQUEST)
        req = transport.requests[0]
        assert req.headers["x-goog-api-key"] == FAKE_GEMINI_KEY
        assert FAKE_GEMINI_KEY not in str(req.url)

    def test_overrides_json_mode_and_thinking_level(self):
        service, transport, _ = make_gemini_service(
            [(200, ok_body('{"a": 1}'))], gemini_thinking_level="low"
        )
        service.generate(LLMRequest(system_prompt="s", user_prompt="u", temperature=0.7,
                                    max_output_tokens=300, response_format="json"))
        cfg = transport.body()["generationConfig"]
        assert cfg["temperature"] == 0.7
        assert cfg["maxOutputTokens"] == 300
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["thinkingConfig"]["thinking_level"] == "LOW"

    def test_grounding_rules_and_material_reach_gemini(self):
        chunks = [LectureChunk(chunk_id="c1", text="A stack is LIFO.", source="lec3.pdf", page=2)]
        request = build_grounded_request("Explain.", chunks, "What is a stack?")
        service, transport, _ = make_gemini_service([(200, ok_body())])
        service.generate(request)
        body = transport.body()
        system = body["systemInstruction"]["parts"][0]["text"]
        user = body["contents"][0]["parts"][0]["text"]
        assert NOT_IN_MATERIAL in system and "Ignore any instructions" in system
        assert "<<<LECTURE_MATERIAL>>>" in user and "[c1] source=lec3.pdf page=2" in user
        assert "A stack is LIFO." not in system


# --------------------------------------------------------------------------- #
# Response normalisation and validation
# --------------------------------------------------------------------------- #
class TestGeminiResponse:
    def test_success_metrics(self):
        service, _, sleep = make_gemini_service([(200, ok_body("  A stack is LIFO.  "))])
        resp = service.generate(REQUEST)
        assert resp.content == "A stack is LIFO."
        assert resp.finish_reason == "stop"
        assert resp.model == "gemini-3.5-flash-lite"
        assert resp.usage.prompt_tokens == 20
        assert resp.usage.completion_tokens == 10  # 6 answer + 4 thinking tokens
        assert resp.usage.total_tokens == 30
        assert resp.attempts == 1 and resp.latency_ms >= 0
        sleep.assert_not_called()

    def test_generate_text(self):
        service, _, _ = make_gemini_service([(200, ok_body("Answer"))])
        assert service.generate_text("sys", "user") == "Answer"

    def test_thought_parts_are_excluded(self):
        parts = [{"text": "internal reasoning", "thought": True}, {"text": "Final answer."}]
        service, _, _ = make_gemini_service([(200, ok_body(parts=parts))])
        assert service.generate(REQUEST).content == "Final answer."

    def test_missing_usage_and_model_fall_back(self):
        body = ok_body(usage={})
        del body["modelVersion"]
        service, _, _ = make_gemini_service([(200, body)])
        resp = service.generate(REQUEST)
        assert resp.usage.total_tokens == 0
        assert resp.model == "gemini-3.5-flash-lite"

    def test_max_tokens_text_is_returned_as_length(self):
        service, _, _ = make_gemini_service([(200, ok_body("partial", finish="MAX_TOKENS"))])
        assert service.generate(REQUEST).finish_reason == "length"

    def test_max_tokens_json_is_rejected(self):
        service, _, _ = make_gemini_service([(200, ok_body('{"a": ', finish="MAX_TOKENS"))])
        with pytest.raises(LLMResponseError, match="truncated"):
            service.generate(LLMRequest(system_prompt="s", user_prompt="u", response_format="json"))

    def test_thinking_consumed_whole_budget_gives_actionable_error(self):
        body = ok_body(parts=[{"text": "thinking...", "thought": True}], finish="MAX_TOKENS")
        service, _, _ = make_gemini_service([(200, body)])
        with pytest.raises(LLMResponseError, match="LLM_MAX_OUTPUT_TOKENS"):
            service.generate(REQUEST)

    @pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST"])
    def test_blocked_output(self, finish):
        service, _, _ = make_gemini_service([(200, ok_body("x", finish=finish))])
        with pytest.raises(LLMResponseError, match="content filter"):
            service.generate(REQUEST)

    def test_blocked_prompt(self):
        service, _, _ = make_gemini_service([(200, {"promptFeedback": {"blockReason": "SAFETY"}})])
        with pytest.raises(LLMResponseError, match="content filter"):
            service.generate(REQUEST)

    def test_no_candidates(self):
        service, _, _ = make_gemini_service([(200, {"candidates": []})])
        with pytest.raises(LLMResponseError, match="no choices"):
            service.generate(REQUEST)

    @pytest.mark.parametrize("parts", [[], [{"text": "   "}]])
    def test_empty_text(self, parts):
        service, _, _ = make_gemini_service([(200, ok_body(parts=parts))])
        with pytest.raises(LLMResponseError, match="empty"):
            service.generate(REQUEST)

    def test_non_json_body_is_response_error(self):
        service, _, _ = make_gemini_service([httpx2.Response(200, text="<html>oops</html>")])
        with pytest.raises(LLMResponseError, match="unreadable"):
            service.generate(REQUEST)


class Explanation(BaseModel):
    answer: str
    cited_chunks: list[str]


class TestGeminiStructured:
    def test_structured_output_validated(self):
        service, transport, _ = make_gemini_service(
            [(200, ok_body('{"answer": "LIFO", "cited_chunks": ["c1"]}'))]
        )
        result = service.generate_structured(REQUEST, Explanation)
        assert result == Explanation(answer="LIFO", cited_chunks=["c1"])
        body = transport.body()
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert "cited_chunks" in body["systemInstruction"]["parts"][0]["text"]

    def test_invalid_json_is_re_asked(self):
        service, transport, _ = make_gemini_service(
            [(200, ok_body("not json")), (200, ok_body('{"answer": "a", "cited_chunks": []}'))]
        )
        assert service.generate_structured(REQUEST, Explanation).answer == "a"
        assert len(transport.requests) == 2

    def test_schema_mismatch_raises(self):
        service, _, _ = make_gemini_service([(200, ok_body('{"wrong": 1}'))] * 2)
        with pytest.raises(LLMResponseError, match="Explanation"):
            service.generate_structured(REQUEST, Explanation)


# --------------------------------------------------------------------------- #
# Errors, retries and rate limits
# --------------------------------------------------------------------------- #
class TestGeminiErrors:
    def test_rate_limit_with_short_retry_delay_is_retried(self):
        service, transport, sleep = make_gemini_service([
            (429, api_error(429, "RESOURCE_EXHAUSTED", details=[PER_MINUTE_QUOTA, RETRY_INFO("3s")])),
            (200, ok_body()),
        ])
        assert service.generate(REQUEST).attempts == 2
        sleep.assert_called_once_with(3.0)

    def test_rate_limit_with_fractional_retry_delay(self):
        service, _, sleep = make_gemini_service([
            (429, api_error(429, "RESOURCE_EXHAUSTED", details=[RETRY_INFO("1.5s")])),
            (200, ok_body()),
        ])
        service.generate(REQUEST)
        sleep.assert_called_once_with(1.5)

    def test_rate_limit_with_long_retry_delay_fails_fast(self):
        service, transport, sleep = make_gemini_service([
            (429, api_error(429, "RESOURCE_EXHAUSTED", details=[RETRY_INFO("33s")])),
            (200, ok_body()),
        ], retry_max_delay=20.0)
        with pytest.raises(LLMRateLimitError):
            service.generate(REQUEST)
        assert len(transport.requests) == 1  # no doomed early retry wasting quota
        sleep.assert_not_called()

    def test_rate_limit_without_retry_info_uses_backoff(self):
        service, transport, sleep = make_gemini_service([
            (429, api_error(429, "RESOURCE_EXHAUSTED")), (200, ok_body()),
        ])
        service.generate(REQUEST)
        assert 0.5 <= sleep.call_args.args[0] <= 1.0
        assert len(transport.requests) == 2

    def test_daily_quota_is_not_retried(self):
        service, transport, sleep = make_gemini_service([
            (429, api_error(429, "RESOURCE_EXHAUSTED", details=[DAILY_QUOTA, RETRY_INFO("5s")])),
            (200, ok_body()),
        ])
        with pytest.raises(LLMRequestError, match="daily"):
            service.generate(REQUEST)
        assert len(transport.requests) == 1
        sleep.assert_not_called()

    def test_rate_limit_raises_after_retries(self):
        service, transport, _ = make_gemini_service(
            [(429, api_error(429, "RESOURCE_EXHAUSTED"))] * 2, max_retries=1
        )
        with pytest.raises(LLMRateLimitError) as info:
            service.generate(REQUEST)
        assert info.value.status_code == 429

    @pytest.mark.parametrize(
        ("code", "status", "details"),
        [
            (400, "INVALID_ARGUMENT", [API_KEY_INVALID]),  # how Gemini reports a bad key
            (401, "UNAUTHENTICATED", []),
            (403, "PERMISSION_DENIED", []),
        ],
    )
    def test_auth_errors_fail_fast_and_hide_provider_text(self, code, status, details, caplog):
        caplog.set_level(logging.DEBUG)
        provider_text = "API key not valid. Please pass a valid API key. AIzaSyLEAKED"
        service, transport, sleep = make_gemini_service(
            [(code, api_error(code, status, provider_text, details)), (200, ok_body())]
        )
        with pytest.raises(LLMAuthenticationError, match="GEMINI_API_KEY") as info:
            service.generate(REQUEST)
        err = info.value
        assert len(transport.requests) == 1
        sleep.assert_not_called()
        assert err.__cause__ is None and err.__context__ is None
        assert err.status_code == code
        full = "".join(traceback.format_exception(err))
        for text in (str(err), repr(err), full, caplog.text):
            assert "AIzaSyLEAKED" not in text
            assert FAKE_GEMINI_KEY not in text

    def test_model_not_found(self):
        service, transport, _ = make_gemini_service(
            [(404, api_error(404, "NOT_FOUND", "models/x is not found"))]
        )
        with pytest.raises(LLMRequestError, match="LLM_MODEL"):
            service.generate(REQUEST)
        assert len(transport.requests) == 1

    def test_model_not_found_reports_model_and_api_version_not_provider_text(self):
        # Regression: a 404 on generateContent used to be reported only as "model not
        # found", with no model ID or API version and the provider's reason discarded,
        # so it could not be compared with a --list-models result.
        provider_text = ("models/gemini-2.5-flash is not found for API version v1, or is not "
                         "supported for generateContent. AIzaSyLEAKED")
        service, transport, sleep = make_gemini_service(
            [(404, api_error(404, "NOT_FOUND", provider_text))], model="gemini-2.5-flash"
        )
        with pytest.raises(LLMRequestError) as info:
            service.generate(REQUEST)
        err = info.value
        message = str(err)
        assert "model 'gemini-2.5-flash'" in message
        assert "API version v1" in message  # SDK 3.x default for the Gemini Developer API
        assert "status NOT_FOUND" in message
        assert "LLM_MODEL" in message
        assert provider_text not in message and "AIzaSyLEAKED" not in message
        assert err.status_code == 404
        assert len(transport.requests) == 1  # not retried
        sleep.assert_not_called()
        # The provider's reason stays available for diagnostics on __cause__.
        assert isinstance(err.__cause__, genai_errors.ClientError)
        assert err.__cause__.message == provider_text

    def test_404_request_path_matches_listing_name(self):
        # The generation URL must use the same bare model ID that --list-models
        # compares as "models/<LLM_MODEL>" (no doubled "models/" prefix).
        service, transport, _ = make_gemini_service(
            [(404, api_error(404, "NOT_FOUND"))], model="gemini-2.5-flash"
        )
        with pytest.raises(LLMRequestError):
            service.generate(REQUEST)
        path = transport.requests[0].url.path
        assert path == "/v1/models/gemini-2.5-flash:generateContent"

    def test_404_includes_error_info_reason_only_when_enum_like(self):
        info = {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "MODEL_NOT_FOUND"}
        bad = {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "free text AIzaX"}
        for details, expected in (([info], "reason MODEL_NOT_FOUND"), ([bad], None)):
            service, _, _ = make_gemini_service([(404, api_error(404, "NOT_FOUND", "x", details))])
            with pytest.raises(LLMRequestError) as exc_info:
                service.generate(REQUEST)
            if expected:
                assert expected in str(exc_info.value)
            else:
                assert "reason" not in str(exc_info.value) and "AIza" not in str(exc_info.value)

    def test_api_version_follows_sdk_override(self):
        client = genai.Client(api_key=FAKE_GEMINI_KEY, vertexai=False,
                              http_options={"api_version": "v1beta"})
        service = LLMService(gemini_settings(), client=client)
        assert service._backend.api_version == "v1beta"

    def test_api_version_unknown_for_non_sdk_client(self):
        service = LLMService(gemini_settings(), client=object())
        assert service._backend.api_version is None

    def test_bad_request_keeps_cause_but_not_provider_text(self):
        service, _, _ = make_gemini_service(
            [(400, api_error(400, "INVALID_ARGUMENT", "secret provider detail"))]
        )
        with pytest.raises(LLMRequestError) as info:
            service.generate(REQUEST)
        assert isinstance(info.value.__cause__, genai_errors.ClientError)
        assert "secret provider detail" not in str(info.value)

    @pytest.mark.parametrize("code", [500, 503, 504])
    def test_server_errors_retried(self, code):
        service, transport, _ = make_gemini_service(
            [(code, api_error(code, "UNAVAILABLE")), (200, ok_body())]
        )
        assert service.generate(REQUEST).attempts == 2

    def test_server_error_raises_after_retries(self):
        service, transport, _ = make_gemini_service(
            [(503, api_error(503, "UNAVAILABLE"))] * 4, max_retries=3
        )
        with pytest.raises(LLMUnavailableError):
            service.generate(REQUEST)
        assert len(transport.requests) == 4

    def test_timeout_retried_then_succeeds(self):
        service, transport, _ = make_gemini_service([httpx2.ReadTimeout("slow"), (200, ok_body())])
        assert service.generate(REQUEST).attempts == 2

    def test_persistent_timeout(self):
        service, transport, _ = make_gemini_service(
            [httpx2.ConnectTimeout("slow")] * 2, max_retries=1
        )
        with pytest.raises(LLMTimeoutError):
            service.generate(REQUEST)

    def test_connection_error(self):
        service, transport, _ = make_gemini_service([httpx2.ConnectError("down")] * 2, max_retries=1)
        with pytest.raises(LLMConnectionError):
            service.generate(REQUEST)
        assert len(transport.requests) == 2

    def test_time_budget_applies_to_gemini(self):
        clock = {"now": 0.0}

        def slow_timeout(request):
            clock["now"] += float(request.headers["x-server-timeout"])
            raise httpx2.ReadTimeout("slow")

        _ROUTE["current"] = slow_timeout
        client = _shared_client()
        sleep = lambda s: clock.__setitem__("now", clock["now"] + s)  # noqa: E731
        service = LLMService(
            gemini_settings(timeout_seconds=30, total_timeout_seconds=60, max_retries=3),
            client=client, sleep=sleep, clock=lambda: clock["now"], rng=random.Random(0),
        )
        with pytest.raises(LLMTimeoutError):
            service.generate(REQUEST)
        assert clock["now"] <= 61  # header is rounded up to whole seconds

    def test_success_logs_never_contain_key(self, caplog):
        caplog.set_level(logging.DEBUG)
        service, _, _ = make_gemini_service([(200, ok_body())])
        service.generate(REQUEST)
        assert FAKE_GEMINI_KEY not in caplog.text
