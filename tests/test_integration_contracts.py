"""Offline integration-contract tests across roles (Step 3).

Exercises the real public APIs of Role 1 (LLM backend) and Role 5
(personalization/analytics) end to end, with **mocked LLM replies**:

    topic-tagged chunks (Role 2 output format)
      -> lecture_topics -> plan_next_quiz (Role 5)
      -> chunks_for_topics -> prepare_grounded_request (Role 1)
      -> generate_structured_with_response with a TEST-LOCAL quiz schema
      -> SIMULATED grading -> Role 5 result dicts -> record_attempt
      -> ProfileStore round trip -> dashboard_data (Role 5)

Nothing here is a production quiz generator or grader: the quiz schema
(``make_quiz_schema``) and the grading (``simulate_grading``) are stand-ins
that pin the data contracts Roles 3 and 4 must satisfy (see
``docs/INTEGRATION.md``). A second part checks that one shared
``LLMService`` handles concurrent requests without mixing them up.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Literal

import httpx2
import openai
import pytest
from pydantic import BaseModel, Field, create_model

from src.analytics import dashboard_data
from src.grounding import validate_citations
from src.llm_service import LLMNotInMaterialError, LLMResponseError, LLMService
from src.personalization import PersonalizationEngine, ProfileStore, StudentProfile
from src.prompts import (
    NOT_IN_MATERIAL_JSON,
    chunks_for_topics,
    lecture_topics,
    prepare_grounded_request,
)
from src.schemas import LectureChunk, LLMRequest
from tests.test_gemini_backend import _shared_client, _ROUTE, gemini_settings
from tests.test_llm_service import _REQ, make_completion, make_service, make_settings

LECTURE_ID = "lec3"

# Role 2 output format: topic-tagged chunks of one lecture (plus one untagged intro).
CHUNKS = [
    LectureChunk(chunk_id="lec3-p1-c1", text="Course overview and reading list.", page=1,
                 source="lecture3.pdf", lecture_id=LECTURE_ID),
    LectureChunk(chunk_id="lec3-p2-c1", text="A stack is LIFO: push adds to the top, pop "
                 "removes the top element.", page=2, source="lecture3.pdf",
                 topic="Stacks", lecture_id=LECTURE_ID),
    LectureChunk(chunk_id="lec3-p3-c1", text="A queue is FIFO: enqueue at the back, dequeue "
                 "from the front.", page=3, source="lecture3.pdf",
                 topic="Queues", lecture_id=LECTURE_ID),
    LectureChunk(chunk_id="lec3-p4-c1", text="A binary tree node has at most two children.",
                 page=4, source="lecture3.pdf", topic="Trees", lecture_id=LECTURE_ID),
    LectureChunk(chunk_id="lec3-p5-c1", text="Stacks support undo: the last action is undone "
                 "first.", page=5, source="lecture3.pdf", topic="  stacks ",
                 lecture_id=LECTURE_ID),
]
QUIZ_TASK = (
    "Write multiple-choice questions about the lecture material. Follow the quiz plan in "
    "the student input exactly: number of questions per topic and difficulty. Use the "
    "topic names exactly as given. Cite the chunk IDs each question is based on."
)
ROLE5_RESULT_KEYS = {"question_id", "topic", "difficulty", "score", "question_type"}


# --------------------------------------------------------------------------- #
# Test-local stand-ins for Role 3 / Role 4 (NOT production implementations)
# --------------------------------------------------------------------------- #
def make_quiz_schema(allowed_topics: list[str], difficulty: str) -> type[BaseModel]:
    """Test-local quiz schema: topic restricted to the plan's topics."""
    question = create_model(
        "QuizQuestion",
        question_id=(str, Field(min_length=1)),
        topic=(Literal[tuple(allowed_topics)], ...),
        difficulty=(Literal[difficulty], ...),
        question_type=(Literal["mcq"], "mcq"),
        question=(str, Field(min_length=1)),
        options=(list[str], Field(min_length=2)),
        correct_answer=(str, Field(min_length=1)),
        cited_chunks=(list[str], Field(min_length=1)),
    )
    return create_model("Quiz", questions=(list[question], Field(min_length=1)))


def mocked_quiz_reply(plan, overrides: dict | None = None) -> str:
    """A MOCKED model reply that follows the plan (what a compliant model returns)."""
    sources = {"Stacks": "lec3-p2-c1", "Queues": "lec3-p3-c1", "Trees": "lec3-p4-c1"}
    questions, n = [], 0
    for topic, count in plan.topic_allocation.items():
        for _ in range(count):
            n += 1
            questions.append({
                "question_id": f"q{n}", "topic": topic, "difficulty": plan.difficulty.value,
                "question_type": "mcq",
                "question": f"Question {n} about {topic} [{sources[topic]}]",
                "options": ["A", "B", "C", "D"], "correct_answer": "A",
                "cited_chunks": [sources[topic]],
            })
    data = {"questions": questions}
    if overrides:
        data["questions"][0].update(overrides)
    return json.dumps(data)


def simulate_grading(quiz, wrong_topics: set[str]) -> list[dict]:
    """SIMULATED grading: a fake student answers every question on ``wrong_topics``
    wrongly and all others correctly. Output uses Role 5's documented format."""
    results = []
    for q in quiz.questions:
        chosen = "B" if q.topic in wrong_topics else q.correct_answer
        results.append({
            "question_id": q.question_id, "topic": q.topic, "difficulty": q.difficulty,
            "score": 1.0 if chosen == q.correct_answer else 0.0,
            "question_type": q.question_type,
        })
    return results


def plan_and_request(profile, engine, num_questions=4):
    topics = lecture_topics(CHUNKS)
    plan = engine.plan_next_quiz(profile, available_topics=topics, num_questions=num_questions)
    quiz_chunks = chunks_for_topics(CHUNKS, plan.topic_allocation)
    prepared = prepare_grounded_request(
        QUIZ_TASK, quiz_chunks, json.dumps(plan.to_dict()), response_format="json",
        max_output_tokens=3000, on_overflow="error")
    return topics, plan, quiz_chunks, prepared


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
class TestQuizPipeline:
    def test_full_loop_two_quizzes(self, tmp_path):
        engine, store = PersonalizationEngine(), ProfileStore(tmp_path / "profiles")
        profile = store.load("student-001")
        assert profile.attempts == []

        # --- Quiz 1 -----------------------------------------------------------
        topics, plan, quiz_chunks, prepared = plan_and_request(profile, engine)
        assert topics == ["Stacks", "Queues", "Trees"]  # untagged intro chunk skipped
        assert set(plan.topic_allocation) == set(topics)
        assert sum(plan.topic_allocation.values()) == plan.num_questions == 4
        assert [c.chunk_id for c in quiz_chunks] == [
            "lec3-p2-c1", "lec3-p3-c1", "lec3-p4-c1", "lec3-p5-c1"]
        assert prepared.context.is_complete
        assert "topic=Stacks" in prepared.request.user_prompt
        assert prepared.request.response_format == "json"
        assert prepared.request.max_output_tokens == 3000

        schema = make_quiz_schema(list(plan.topic_allocation), plan.difficulty.value)
        service, client, _ = make_service([make_completion(mocked_quiz_reply(plan))])
        quiz, response = service.generate_structured_with_response(prepared.request, schema)
        assert client.chat.completions.create.call_count == 1
        assert response.attempts == 1 and response.usage.total_tokens > 0

        # Contracts Role 3 must meet: allocation, difficulty, citations.
        assert Counter(q.topic for q in quiz.questions) == Counter(plan.topic_allocation)
        assert {q.difficulty for q in quiz.questions} == {plan.difficulty.value}
        for q in quiz.questions:
            report = validate_citations(q.question, prepared.context)
            assert report.all_valid
            assert set(q.cited_chunks) <= set(prepared.context.prompt_ids)
            assert chunks_for_topics(CHUNKS, q.topic)  # every question topic has material

        # Role 4 -> Role 5 (SIMULATED grading): Stacks answered wrongly.
        results = simulate_grading(quiz, wrong_topics={"Stacks"})
        assert all(set(r) == ROLE5_RESULT_KEYS for r in results)
        attempt = engine.record_attempt(profile, results, plan.difficulty, lecture_id=LECTURE_ID)
        assert attempt.score == 0.5 and attempt.lecture_id == LECTURE_ID
        store.save(profile)

        # Persistence round trip, then dashboard (Role 6 input).
        profile = store.load("student-001")
        data = dashboard_data(profile, engine, available_topics=topics)
        assert data["total_attempts"] == 1
        assert data["overall_accuracy"] == 0.5
        assert data["weak_topics"] == ["Stacks"]
        assert {row["topic"] for row in data["topics"]} == set(topics)
        assert data["next_quiz"]["focus_topics"][0] == "Stacks"
        json.dumps(data)  # dashboard data is JSON-serialisable

        # --- Quiz 2: adapts to the weak topic ---------------------------------
        _, plan2, quiz_chunks2, prepared2 = plan_and_request(profile, engine)
        assert plan2.focus_topics == ["Stacks"]
        assert plan2.topic_allocation["Stacks"] == max(plan2.topic_allocation.values())
        stacks_chunks = [c.chunk_id for c in chunks_for_topics(CHUNKS, "Stacks")]
        assert set(stacks_chunks) <= {c.chunk_id for c in quiz_chunks2}
        assert prepared2.context.is_complete

    def test_case_variant_topic_chunks_are_selected_together(self):
        # "Stacks" and "  stacks " are one topic for Role 1 and Role 5 alike.
        assert [c.chunk_id for c in chunks_for_topics(CHUNKS, "STACKS")] == [
            "lec3-p2-c1", "lec3-p5-c1"]


class TestInvalidTopics:
    def setup_method(self):
        self.engine, self.profile = PersonalizationEngine(), StudentProfile("s-invalid")
        _, self.plan, _, self.prepared = plan_and_request(self.profile, self.engine)
        self.schema = make_quiz_schema(list(self.plan.topic_allocation),
                                       self.plan.difficulty.value)

    @pytest.mark.parametrize("bad_topic", ["Binary Trees", "stacks", "Graphs"])
    def test_topic_outside_plan_is_reasked_then_accepted(self, bad_topic):
        service, client, _ = make_service([
            make_completion(mocked_quiz_reply(self.plan, {"topic": bad_topic})),
            make_completion(mocked_quiz_reply(self.plan)),
        ])
        quiz = service.generate_structured(self.prepared.request, self.schema)
        assert client.chat.completions.create.call_count == 2
        assert {q.topic for q in quiz.questions} <= set(self.plan.topic_allocation)

    def test_topic_outside_plan_twice_fails(self):
        bad = make_completion(mocked_quiz_reply(self.plan, {"topic": "Graphs"}))
        service, _, _ = make_service([bad, bad])
        with pytest.raises(LLMResponseError, match="valid Quiz output"):
            service.generate_structured(self.prepared.request, self.schema)

    def test_role5_does_not_reject_unknown_topics_itself(self):
        # Why the quiz schema must restrict topics: Role 5 silently starts a new topic.
        self.engine.record_attempt(self.profile, [
            {"question_id": "q1", "topic": "Graphs", "difficulty": "medium", "score": 1.0}],
            "medium")
        assert "Graphs" in self.engine.topic_stats(self.profile)
        assert chunks_for_topics(CHUNKS, "Graphs") == []  # ... with no lecture material

    def test_wrong_difficulty_is_rejected(self):
        wrong = "hard" if self.plan.difficulty.value != "hard" else "easy"
        bad = make_completion(mocked_quiz_reply(self.plan, {"difficulty": wrong}))
        service, _, _ = make_service([bad, bad])
        with pytest.raises(LLMResponseError):
            service.generate_structured(self.prepared.request, self.schema)


class TestMissingLectureTopics:
    UNTAGGED = [c.model_copy(update={"topic": None}) for c in CHUNKS]

    def test_no_topics_means_no_allocation_and_no_chunk_selection(self):
        assert lecture_topics(self.UNTAGGED) == []
        plan = PersonalizationEngine().plan_next_quiz(
            StudentProfile("s"), available_topics=lecture_topics(self.UNTAGGED), num_questions=4)
        assert plan.topic_allocation == {} and plan.focus_topics == []
        assert chunks_for_topics(self.UNTAGGED, plan.topic_allocation) == []
        with pytest.raises(ValueError, match="At least one lecture chunk"):
            prepare_grounded_request(QUIZ_TASK, chunks_for_topics(self.UNTAGGED, {}), "plan",
                                     response_format="json")

    def test_role5_rejects_results_without_topic_with_keyerror(self):
        # Sharp edge: a missing "topic" raises KeyError, not ValueError.
        with pytest.raises(KeyError):
            PersonalizationEngine().record_attempt(
                StudentProfile("s"), [{"question_id": "q1", "score": 1.0}], "medium")

    def test_role5_rejects_blank_topic_with_valueerror(self):
        with pytest.raises(ValueError):
            PersonalizationEngine().record_attempt(
                StudentProfile("s"), [{"question_id": "q1", "topic": " ", "score": 1.0}], "medium")

    def test_partially_tagged_lecture_uses_only_tagged_chunks(self):
        topics = lecture_topics(CHUNKS)
        selected = chunks_for_topics(CHUNKS, topics)
        assert "lec3-p1-c1" not in [c.chunk_id for c in selected]  # untagged intro excluded


class TestNotInMaterial:
    def test_uncovered_quiz_raises_and_records_nothing(self):
        engine, profile = PersonalizationEngine(), StudentProfile("s-nim")
        _, plan, _, prepared = plan_and_request(profile, engine)
        schema = make_quiz_schema(list(plan.topic_allocation), plan.difficulty.value)
        service, client, _ = make_service([make_completion(NOT_IN_MATERIAL_JSON)] * 2)
        with pytest.raises(LLMNotInMaterialError):
            service.generate_structured_with_response(prepared.request, schema)
        assert client.chat.completions.create.call_count == 1  # no re-ask
        assert profile.attempts == []  # nothing to grade, nothing recorded

    def test_grounded_quiz_request_offers_json_fallback(self):
        _, _, _, prepared = plan_and_request(StudentProfile("s"), PersonalizationEngine())
        assert NOT_IN_MATERIAL_JSON in prepared.request.system_prompt


# --------------------------------------------------------------------------- #
# Concurrency: one shared LLMService, many threads
# --------------------------------------------------------------------------- #
N_THREADS = 16
_KEY = re.compile(r"REQ-(\d+)")


class ThreadSafeEchoClient:
    """Mocked OpenAI client. Replies depend only on the request's own content and
    settings, so any cross-talk between threads shows up as a mismatch. The
    first call of every request waits at a barrier so all requests are in
    flight at the same time; requests in ``fail_first`` time out once first."""

    def __init__(self, n: int, fail_first: set[int] = frozenset()) -> None:
        self.barrier = threading.Barrier(n, timeout=10)
        self.lock = threading.Lock()
        self.fail_first = set(fail_first)
        self.seen: Counter[int] = Counter()
        self.mismatches: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **params):
        key = int(_KEY.search(params["messages"][1]["content"]).group(1))
        with self.lock:
            self.seen[key] += 1
            first = self.seen[key] == 1
        if first:
            self.barrier.wait()
        if params["max_tokens"] != 100 + key or params["temperature"] != (key % 3) * 0.5:
            with self.lock:
                self.mismatches.append(f"REQ-{key}: {params['max_tokens']}, {params['temperature']}")
        time.sleep(random.uniform(0, 0.01))
        if first and key in self.fail_first:
            raise openai.APITimeoutError(request=_REQ)
        return make_completion(
            json.dumps({"key": key, "answer": f"ANSWER-{key}"}),
            usage={"prompt_tokens": key, "completion_tokens": 2 * key, "total_tokens": 3 * key})


def concurrent_request(key: int, json_mode: bool = False) -> LLMRequest:
    return LLMRequest(system_prompt="You are a tutor.", user_prompt=f"REQ-{key}: explain.",
                      max_output_tokens=100 + key, temperature=(key % 3) * 0.5,
                      response_format="json" if json_mode else "text")


class Echo(BaseModel):
    key: int
    answer: str


class TestConcurrency:
    def _service(self, client):
        return LLMService(make_settings(max_retries=2), client=client,
                          sleep=lambda s: None, rng=random.Random(0))

    def test_generate_does_not_mix_responses(self):
        client = ThreadSafeEchoClient(N_THREADS)
        service = self._service(client)
        with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
            responses = list(pool.map(lambda k: service.generate(concurrent_request(k)),
                                      range(N_THREADS)))
        for key, response in enumerate(responses):
            assert json.loads(response.content) == {"key": key, "answer": f"ANSWER-{key}"}
            assert response.usage.prompt_tokens == key
            assert response.usage.total_tokens == 3 * key
            assert response.attempts == 1
        assert client.mismatches == []
        assert all(client.seen[k] == 1 for k in range(N_THREADS))

    def test_retries_are_counted_per_request(self):
        failing = {1, 4, 9, 15}
        client = ThreadSafeEchoClient(N_THREADS, fail_first=failing)
        service = self._service(client)
        with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
            responses = list(pool.map(lambda k: service.generate(concurrent_request(k)),
                                      range(N_THREADS)))
        for key, response in enumerate(responses):
            assert response.attempts == (2 if key in failing else 1)
            assert json.loads(response.content)["key"] == key
        assert client.mismatches == []

    def test_structured_with_response_does_not_mix_results(self):
        client = ThreadSafeEchoClient(N_THREADS)
        service = self._service(client)

        def call(key):
            return service.generate_structured_with_response(concurrent_request(key, True), Echo)

        with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
            pairs = list(pool.map(call, range(N_THREADS)))
        for key, (result, response) in enumerate(pairs):
            assert result == Echo(key=key, answer=f"ANSWER-{key}")
            assert response.usage.prompt_tokens == key
        assert client.mismatches == []

    def test_gemini_real_sdk_client_shared_across_threads(self):
        barrier = threading.Barrier(N_THREADS, timeout=10)

        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            key = int(_KEY.search(body["contents"][0]["parts"][0]["text"]).group(1))
            assert body["generationConfig"]["maxOutputTokens"] == 100 + key
            barrier.wait()
            time.sleep(random.uniform(0, 0.01))
            return httpx2.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": f"ANSWER-{key}"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": key, "totalTokenCount": 3 * key},
                "modelVersion": f"model-for-{key}"})

        _ROUTE["current"] = handler
        service = LLMService(gemini_settings(), client=_shared_client(),
                             sleep=lambda s: None, rng=random.Random(0))
        with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
            responses = list(pool.map(lambda k: service.generate(concurrent_request(k)),
                                      range(N_THREADS)))
        for key, response in enumerate(responses):
            assert response.content == f"ANSWER-{key}"
            assert response.model == f"model-for-{key}"
            assert response.usage.total_tokens == 3 * key
