"""Offline tests for Step 2 (G3): ``LLMService.generate_structured_with_response``.

The method must return the validated model *and* the ``LLMResponse`` of the
generation that was accepted, reusing ``generate_structured``'s logic, while
``generate_structured`` keeps its behaviour and return type.
"""

from __future__ import annotations

import openai
import pytest
from pydantic import BaseModel

from src.llm_service import (
    LLMAuthenticationError,
    LLMNotInMaterialError,
    LLMResponseError,
    LLMTimeoutError,
)
from src.prompts import NOT_IN_MATERIAL_JSON, build_grounded_request
from src.schemas import LectureChunk, LLMRequest, LLMResponse, TokenUsage
from tests.test_gemini_backend import make_gemini_service, ok_body
from tests.test_llm_service import _REQ, make_completion, make_service, status_error

REQUEST = LLMRequest(system_prompt="You are a tutor.", user_prompt="Explain stacks.")
GROUNDED = build_grounded_request(
    "Answer.", [LectureChunk(chunk_id="c1", text="A stack is LIFO.")], "heaps",
    response_format="json")
VALID = '{"answer": "LIFO", "cited_chunks": ["c1"]}'


class Explanation(BaseModel):
    answer: str
    cited_chunks: list[str]


class OptIn(BaseModel):
    not_in_material: bool = False
    answer: str = ""


def usage(p, c):
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


class TestSuccess:
    def test_returns_validated_model_and_response(self):
        service, client, _ = make_service([make_completion(VALID, usage=usage(12, 8))])
        result, response = service.generate_structured_with_response(REQUEST, Explanation)
        assert result == Explanation(answer="LIFO", cited_chunks=["c1"])
        assert isinstance(response, LLMResponse)
        assert response.content == VALID
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.usage == TokenUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20)
        assert response.attempts == 1
        assert response.latency_ms >= 0
        assert client.chat.completions.create.call_count == 1

    def test_response_content_is_raw_model_output(self):
        fenced = f"```json\n{VALID}\n```"
        service, _, _ = make_service([make_completion(fenced)])
        result, response = service.generate_structured_with_response(REQUEST, Explanation)
        assert result.answer == "LIFO"
        assert response.content == fenced  # unmodified; parsing stripped the fence only

    def test_gemini_metadata_passed_through(self):
        service, transport, _ = make_gemini_service([(200, ok_body(VALID, usage={
            "promptTokenCount": 30, "candidatesTokenCount": 9, "thoughtsTokenCount": 3,
            "totalTokenCount": 42}, model="gemini-3.5-flash-lite-001"))])
        _, response = service.generate_structured_with_response(REQUEST, Explanation)
        assert response.model == "gemini-3.5-flash-lite-001"
        assert response.usage == TokenUsage(prompt_tokens=30, completion_tokens=12, total_tokens=42)
        assert len(transport.requests) == 1


class TestValidationRetries:
    def test_response_belongs_to_the_accepted_attempt(self):
        service, client, _ = make_service([
            make_completion('{"wrong": 1}', usage=usage(10, 5)),
            make_completion(VALID, usage=usage(20, 7)),
        ])
        result, response = service.generate_structured_with_response(REQUEST, Explanation)
        assert result.answer == "LIFO"
        assert response.content == VALID  # not the rejected first reply
        assert response.usage.total_tokens == 27  # second call only, not 15 + 27
        assert response.attempts == 1
        assert client.chat.completions.create.call_count == 2

    def test_transient_retry_counted_in_attempts(self):
        service, _, sleep = make_service([
            status_error(openai.InternalServerError, 503),
            make_completion(VALID, usage=usage(4, 4)),
        ])
        _, response = service.generate_structured_with_response(REQUEST, Explanation)
        assert response.attempts == 2  # retries of the accepted generation
        sleep.assert_called_once()

    def test_reask_then_transient_retry(self):
        service, client, _ = make_service([
            make_completion("not json"),
            openai.APITimeoutError(request=_REQ),
            make_completion(VALID),
        ])
        _, response = service.generate_structured_with_response(REQUEST, Explanation)
        assert response.attempts == 2 and response.content == VALID
        assert client.chat.completions.create.call_count == 3

    def test_opt_in_fallback_returns_its_response(self):
        service, _, _ = make_service([make_completion(NOT_IN_MATERIAL_JSON, usage=usage(9, 3))])
        result, response = service.generate_structured_with_response(GROUNDED, OptIn)
        assert result.not_in_material is True
        assert response.content == NOT_IN_MATERIAL_JSON
        assert response.usage.total_tokens == 12


class TestMissingMetadata:
    def test_unreported_usage_and_model_are_not_invented(self):
        body = ok_body(VALID, usage={})
        del body["usageMetadata"]
        del body["modelVersion"]
        service, _, _ = make_gemini_service([(200, body)])
        _, response = service.generate_structured_with_response(REQUEST, Explanation)
        # Documented conventions, identical to generate(): unreported counts are 0,
        # a missing model name is the configured LLM_MODEL.
        assert response.usage == TokenUsage()
        assert response.model == service.settings.model
        assert response.finish_reason == "stop"

    def test_same_metadata_as_generate_for_the_same_reply(self):
        body = ok_body(VALID, usage={"promptTokenCount": 5})
        del body["modelVersion"]
        service, _, _ = make_gemini_service([(200, body), (200, body)])
        plain = service.generate(REQUEST.model_copy(update={"response_format": "json"}))
        _, structured = service.generate_structured_with_response(REQUEST, Explanation)
        assert structured.usage == plain.usage
        assert structured.model == plain.model
        assert structured.finish_reason == plain.finish_reason


class TestExceptionPropagation:
    @pytest.mark.parametrize("method", ["generate_structured", "generate_structured_with_response"])
    def test_not_in_material(self, method):
        service, client, _ = make_service([make_completion(NOT_IN_MATERIAL_JSON)] * 2)
        with pytest.raises(LLMNotInMaterialError):
            getattr(service, method)(GROUNDED, Explanation)
        assert client.chat.completions.create.call_count == 1

    @pytest.mark.parametrize("method", ["generate_structured", "generate_structured_with_response"])
    def test_validation_exhausted(self, method):
        service, client, _ = make_service([make_completion('{"wrong": 1}')] * 2)
        with pytest.raises(LLMResponseError, match="valid Explanation output"):
            getattr(service, method)(REQUEST, Explanation)
        assert client.chat.completions.create.call_count == 2

    @pytest.mark.parametrize("method", ["generate_structured", "generate_structured_with_response"])
    def test_authentication_error(self, method):
        service, _, _ = make_service([status_error(openai.AuthenticationError, 401)])
        with pytest.raises(LLMAuthenticationError):
            getattr(service, method)(REQUEST, Explanation)

    @pytest.mark.parametrize("method", ["generate_structured", "generate_structured_with_response"])
    def test_timeout_after_retries(self, method):
        service, _, _ = make_service([openai.APITimeoutError(request=_REQ)] * 2, max_retries=1)
        with pytest.raises(LLMTimeoutError):
            getattr(service, method)(REQUEST, Explanation)


class TestBackwardCompatibility:
    def test_generate_structured_still_returns_only_the_model(self):
        service, _, _ = make_service([make_completion(VALID)])
        result = service.generate_structured(REQUEST, Explanation)
        assert type(result) is Explanation  # not a tuple

    def test_both_methods_send_identical_requests_and_agree(self):
        replies = [make_completion('{"wrong": 1}'), make_completion(VALID)]
        a, client_a, _ = make_service(list(replies))
        b, client_b, _ = make_service(list(replies))
        result_a = a.generate_structured(REQUEST, Explanation)
        result_b, _ = b.generate_structured_with_response(REQUEST, Explanation)
        assert result_a == result_b
        calls_a = [c.kwargs for c in client_a.chat.completions.create.call_args_list]
        calls_b = [c.kwargs for c in client_b.chat.completions.create.call_args_list]
        assert calls_a == calls_b and len(calls_a) == 2

    def test_positional_and_keyword_arguments(self):
        service, _, _ = make_service([make_completion(VALID)] * 2)
        assert service.generate_structured_with_response(REQUEST, model_class=Explanation)[0].answer
        assert service.generate_structured(REQUEST, output_model=Explanation).answer
