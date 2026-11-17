"""Offline tests for integration hardening (G1, G2, G4).

G1: JSON grounding and the not-covered fallback in structured output.
G2: which chunks were included in the prompt; citation extraction/validation.
G4: ``is_not_in_material`` detection for the frontend.

No network: services use a mocked OpenAI client, or the real Gemini SDK client
over an in-memory transport.
"""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import BaseModel, ValidationError

from src.grounding import extract_citations, is_not_in_material, validate_citations
from src.llm_service import LLMNotInMaterialError, LLMResponseError
from src.prompts import (
    GROUNDING_RULES,
    GROUNDING_RULES_JSON,
    NOT_IN_MATERIAL,
    NOT_IN_MATERIAL_JSON,
    LectureContextOverflowError,
    build_grounded_request,
    build_lecture_context,
    format_lecture_context,
    prepare_grounded_request,
)
from src.schemas import GroundedRequest, LectureChunk, LectureContext, LLMRequest
from tests.test_gemini_backend import make_gemini_service, ok_body
from tests.test_llm_service import make_completion, make_service

CHUNKS = [
    LectureChunk(chunk_id="c1", text="A stack is LIFO.", source="l1.pdf", page=1),
    LectureChunk(chunk_id="c2", text="A queue is FIFO.", source="l1.pdf", page=2),
]


class Quiz(BaseModel):
    questions: list[str]


class OptInQuiz(BaseModel):
    """Schema that opts in to receiving the fallback as data."""

    not_in_material: bool = False
    questions: list[str] = []


class OptInButRequired(BaseModel):
    not_in_material: bool = False
    questions: list[str]  # required: the fallback cannot fill it


def grounded_json(topic="stacks"):
    return build_grounded_request("Write quiz questions.", CHUNKS, topic, response_format="json")


# --------------------------------------------------------------------------- #
# G1: JSON grounding
# --------------------------------------------------------------------------- #
class TestJsonGroundingPrompt:
    def test_json_mode_uses_json_fallback_rules(self):
        req = grounded_json()
        assert GROUNDING_RULES_JSON in req.system_prompt
        assert GROUNDING_RULES not in req.system_prompt
        assert NOT_IN_MATERIAL_JSON in req.system_prompt
        assert json.loads(NOT_IN_MATERIAL_JSON) == {"not_in_material": True,
                                                    "message": NOT_IN_MATERIAL}

    def test_text_mode_is_unchanged(self):
        req = build_grounded_request("Explain.", CHUNKS, "What is a stack?")
        assert GROUNDING_RULES in req.system_prompt
        assert NOT_IN_MATERIAL_JSON not in req.system_prompt
        assert f'reply exactly: "{NOT_IN_MATERIAL}"' in req.system_prompt

    def test_structured_prompt_offers_json_fallback_only_for_grounded_requests(self):
        service, client, _ = make_service([make_completion('{"questions": ["q"]}')] * 2)
        service.generate_structured(grounded_json(), Quiz)
        system = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert system.count(NOT_IN_MATERIAL_JSON) == 2  # rule + schema exception line
        assert "conforms to this JSON Schema" in system

        service.generate_structured(LLMRequest(system_prompt="s", user_prompt="u"), Quiz)
        system = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert NOT_IN_MATERIAL_JSON not in system  # ungrounded: no fallback offered


class TestStructuredNotInMaterial:
    def test_covered_topic_returns_validated_model(self):
        service, client, _ = make_service([make_completion('{"questions": ["What is LIFO? [c1]"]}')])
        result = service.generate_structured(grounded_json(), Quiz)
        assert result == Quiz(questions=["What is LIFO? [c1]"])
        assert client.chat.completions.create.call_count == 1

    @pytest.mark.parametrize("reply", [
        NOT_IN_MATERIAL_JSON,
        f"```json\n{NOT_IN_MATERIAL_JSON}\n```",
        NOT_IN_MATERIAL,  # model ignored JSON mode and used the plain phrase
    ])
    def test_uncovered_topic_raises_without_reask(self, reply):
        # Regression: this used to fail validation, re-ask (a 2nd paid/quota request)
        # and end in a generic "did not return valid Quiz output" error.
        service, client, sleep = make_service([make_completion(reply)] * 2)
        with pytest.raises(LLMNotInMaterialError) as info:
            service.generate_structured(grounded_json("binary heaps"), Quiz)
        assert isinstance(info.value, LLMResponseError)  # existing handlers still catch it
        assert client.chat.completions.create.call_count == 1
        sleep.assert_not_called()

    def test_opt_in_schema_receives_fallback_as_data(self):
        service, client, _ = make_service([make_completion(NOT_IN_MATERIAL_JSON)])
        result = service.generate_structured(grounded_json("heaps"), OptInQuiz)
        assert result == OptInQuiz(not_in_material=True, questions=[])
        assert client.chat.completions.create.call_count == 1

    def test_opt_in_schema_receives_plain_phrase_as_data(self):
        service, _, _ = make_service([make_completion(NOT_IN_MATERIAL)])
        assert service.generate_structured(grounded_json(), OptInQuiz).not_in_material is True

    def test_opt_in_schema_covered_topic(self):
        service, _, _ = make_service([make_completion('{"questions": ["q1"]}')])
        result = service.generate_structured(grounded_json(), OptInQuiz)
        assert result == OptInQuiz(not_in_material=False, questions=["q1"])

    def test_opt_in_schema_with_required_fields_raises(self):
        service, _, _ = make_service([make_completion(NOT_IN_MATERIAL_JSON)])
        with pytest.raises(LLMNotInMaterialError):
            service.generate_structured(grounded_json(), OptInButRequired)

    def test_all_optional_schema_does_not_silently_swallow_fallback(self):
        class AllOptional(BaseModel):
            questions: list[str] = []

        service, _, _ = make_service([make_completion(NOT_IN_MATERIAL_JSON)])
        with pytest.raises(LLMNotInMaterialError):  # not an empty AllOptional()
            service.generate_structured(grounded_json(), AllOptional)

    def test_invalid_output_still_reasks_then_fails(self):
        service, client, _ = make_service([make_completion('{"wrong": 1}')] * 2)
        with pytest.raises(LLMResponseError) as info:
            service.generate_structured(grounded_json(), Quiz)
        assert not isinstance(info.value, LLMNotInMaterialError)
        assert client.chat.completions.create.call_count == 2

    def test_gemini_end_to_end_fallback(self):
        service, transport, _ = make_gemini_service([(200, ok_body(NOT_IN_MATERIAL_JSON))])
        with pytest.raises(LLMNotInMaterialError):
            service.generate_structured(grounded_json("heaps"), Quiz)
        assert len(transport.requests) == 1
        body = transport.body()
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert NOT_IN_MATERIAL_JSON in body["systemInstruction"]["parts"][0]["text"]


# --------------------------------------------------------------------------- #
# G2: included chunks, overflow, citations
# --------------------------------------------------------------------------- #
BIG = [LectureChunk(chunk_id=f"c{i}", text="x" * 5000) for i in range(1, 6)]


class TestLectureContext:
    def test_reports_included_and_omitted_chunks(self, caplog):
        caplog.set_level(logging.WARNING)
        ctx = build_lecture_context(BIG)
        assert ctx.included_ids == ("c1", "c2")
        assert ctx.omitted_ids == ("c3", "c4", "c5")
        assert not ctx.truncated and not ctx.is_complete
        assert ctx.text == format_lecture_context(BIG)  # same prompt text as before
        assert "2 of 5 chunk(s) sent" in caplog.text and "c3" in caplog.text
        assert "xxxx" not in caplog.text  # lecture text is never logged

    def test_complete_context(self, caplog):
        caplog.set_level(logging.WARNING)
        ctx = build_lecture_context(CHUNKS)
        assert ctx.included_ids == ("c1", "c2") and ctx.omitted_ids == ()
        assert ctx.is_complete
        assert caplog.text == ""  # no warning when nothing is dropped

    def test_truncated_first_chunk(self):
        ctx = build_lecture_context([LectureChunk(chunk_id="c1", text="y" * 500), CHUNKS[1]],
                                    max_chars=100)
        assert ctx.truncated and ctx.included_ids == ("c1",) and ctx.omitted_ids == ("c2",)
        assert ctx.text.count("[truncated]") == 1

    def test_on_overflow_error(self):
        with pytest.raises(LectureContextOverflowError) as info:
            build_lecture_context(BIG, on_overflow="error")
        assert info.value.omitted_ids == ("c3", "c4", "c5")
        assert isinstance(info.value, ValueError)
        assert "12000-character" in str(info.value)

    def test_on_overflow_error_for_truncation(self):
        with pytest.raises(LectureContextOverflowError) as info:
            build_lecture_context([LectureChunk(chunk_id="c1", text="y" * 500)], 100,
                                  on_overflow="error")
        assert info.value.truncated and info.value.omitted_ids == ()

    def test_format_lecture_context_now_warns_when_dropping(self, caplog):
        caplog.set_level(logging.WARNING)
        format_lecture_context(BIG)
        assert "omitted IDs" in caplog.text

    def test_prompt_ids_map_back_to_original_ids(self):
        ctx = build_lecture_context([LectureChunk(chunk_id="Lecture 3/p4#1", text="t")])
        assert ctx.prompt_ids == {"Lecture_3_p4_1": "Lecture 3/p4#1"}
        assert "[Lecture_3_p4_1]" in ctx.text

    def test_ambiguous_ids_rejected_in_new_api(self):
        dup = [LectureChunk(chunk_id="a b", text="t"), LectureChunk(chunk_id="a_b", text="u")]
        with pytest.raises(ValueError, match="unique"):
            build_lecture_context(dup)
        format_lecture_context(dup)  # existing function keeps its lenient behaviour

    def test_context_is_immutable(self):
        ctx = build_lecture_context(CHUNKS)
        with pytest.raises(ValidationError):
            ctx.text = "changed"


class TestPrepareGroundedRequest:
    def test_request_identical_to_build_grounded_request(self):
        for fmt in ("text", "json"):
            prepared = prepare_grounded_request("Task.", BIG, "q", response_format=fmt,
                                                temperature=0.1, max_output_tokens=99)
            assert isinstance(prepared, GroundedRequest)
            assert prepared.request == build_grounded_request(
                "Task.", BIG, "q", response_format=fmt, temperature=0.1, max_output_tokens=99)
            assert prepared.context.included_ids == ("c1", "c2")

    def test_on_overflow_error(self):
        with pytest.raises(LectureContextOverflowError):
            prepare_grounded_request("Task.", BIG, "q", on_overflow="error")
        prepare_grounded_request("Task.", BIG, "q", max_context_chars=30_000, on_overflow="error")

    def test_input_validation_unchanged(self):
        with pytest.raises(ValueError):
            prepare_grounded_request(" ", CHUNKS, "q")
        with pytest.raises(ValueError):
            prepare_grounded_request("t", CHUNKS, " ")
        with pytest.raises(ValueError):
            prepare_grounded_request("t", [], "q")


class TestCitations:
    @pytest.mark.parametrize("text, expected", [
        ("A stack is LIFO [c1].", ["c1"]),
        ("See [c1] and [c2], also [c1] again.", ["c1", "c2"]),
        ("Both [c1, c2] and [c3; c4].", ["c1", "c2", "c3", "c4"]),
        ("IDs [lec3-p4-c1] and [lec3:p4.c2]", ["lec3-p4-c1", "lec3:p4.c2"]),
        ("A [link](https://x.y) is not a citation.", []),
        ("No citations here.", []),
        ("", []),
        (None, []),
    ])
    def test_extract(self, text, expected):
        assert extract_citations(text) == expected

    def test_validate_against_included_chunks(self):
        prepared = prepare_grounded_request("Task.", BIG, "q")  # c1, c2 sent; c3-c5 omitted
        report = validate_citations("Uses [c1], [c2] and [c4] and [c9].", prepared.context)
        assert report.cited == ("c1", "c2", "c4", "c9")
        assert report.valid == ("c1", "c2")
        assert report.unknown == ("c4", "c9")  # c4 was omitted from the prompt
        assert report.has_citations and not report.all_valid

    def test_all_valid(self):
        ctx = build_lecture_context(CHUNKS)
        assert validate_citations("LIFO [c1]; FIFO [c2].", ctx).all_valid

    def test_no_citations_is_not_all_valid(self):
        report = validate_citations("An answer without sources.", build_lecture_context(CHUNKS))
        assert not report.has_citations and not report.all_valid

    def test_valid_ids_are_original_chunk_ids(self):
        ctx = build_lecture_context([LectureChunk(chunk_id="Lecture 3/p4#1", text="t")])
        assert validate_citations("Fact [Lecture_3_p4_1].", ctx).valid == ("Lecture 3/p4#1",)


# --------------------------------------------------------------------------- #
# G4: is_not_in_material
# --------------------------------------------------------------------------- #
class TestIsNotInMaterial:
    @pytest.mark.parametrize("text", [
        NOT_IN_MATERIAL,
        f"  {NOT_IN_MATERIAL}\n",
        NOT_IN_MATERIAL.upper(),
        NOT_IN_MATERIAL.rstrip("."),
        f'"{NOT_IN_MATERIAL}"',
        f"“{NOT_IN_MATERIAL}”",
        f"**{NOT_IN_MATERIAL}**",
        f"{NOT_IN_MATERIAL} [c1]",
        "This  is not covered in the\nprovided lecture material.",
        NOT_IN_MATERIAL_JSON,
        f"```json\n{NOT_IN_MATERIAL_JSON}\n```",
        '{"not_in_material": true}',
        '{"not_in_material": true, "questions": []}',
    ])
    def test_detects_fallback(self, text):
        assert is_not_in_material(text) is True

    @pytest.mark.parametrize("text", [
        None,
        "",
        "   ",
        "A stack is LIFO [c1].",
        f"A stack is LIFO [c1]. Heaps: {NOT_IN_MATERIAL}",  # partial answer: keep it
        f"{NOT_IN_MATERIAL} However, a stack is LIFO [c1].",
        "This is not covered.",
        '{"not_in_material": false, "questions": ["q"]}',
        '{"not_in_material": "true"}',  # must be a real boolean
        '{"questions": []}',
        "{not json",
        '["not_in_material"]',
    ])
    def test_rejects_other_text(self, text):
        assert is_not_in_material(text) is False

    def test_live_smoke_test_answer_is_detected(self):
        # Exact reply recorded from the live Gemini smoke test (2026-11-17).
        assert is_not_in_material("This is not covered in the provided lecture material.")

    def test_text_mode_generate_returns_phrase_for_frontend(self):
        service, _, _ = make_service([make_completion(NOT_IN_MATERIAL)])
        content = service.generate(build_grounded_request("Explain.", CHUNKS, "heaps")).content
        assert is_not_in_material(content)
