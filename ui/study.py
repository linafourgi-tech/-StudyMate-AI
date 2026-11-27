"""Study workflow. Streamlit owns one Workspace per browser session."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.answer_evaluator import evaluate_quiz
from src.config import PROJECT_ROOT
from src.feedback_generator import generate_feedback
from src.grounding import validate_citations
from src.llm_service import LLMResponseError, LLMService, get_llm_service
from src.personalization import PersonalizationEngine, ProfileStore, QuizPlan, StudentProfile
from src.prompts import lecture_topics, prepare_grounded_request
from src.quiz_generator import QuizGenerationResult, generate_quiz
from src.schemas import LectureChunk
from ui.lecture import Lecture, select_passages

PROFILE_DIRECTORY = PROJECT_ROOT / "data" / "profiles"
_profile_write = Lock()


class SummaryPoint(BaseModel):
    heading: str = Field(min_length=1, max_length=120)
    detail: str = Field(min_length=1)


class SummaryContent(BaseModel):
    points: list[SummaryPoint] = Field(min_length=1, max_length=6)


@dataclass(frozen=True)
class Summary:
    points: list[SummaryPoint]
    sources: list[list[LectureChunk]]
    passages_used: int


def summarize(lecture: Lecture, *, service: LLMService | None = None) -> Summary:
    passages = select_passages(lecture.chunks)
    prepared = prepare_grounded_request(
        "Write a concise study summary organized by concept. Give each point a descriptive "
        "heading and a short explanatory paragraph. Put one or more supporting [chunk-id] "
        "citations in every paragraph. Use plain text and only the supplied material.",
        passages, "Summarize the selected lecture passages.", response_format="json",
        max_output_tokens=1800, on_overflow="error",
    )
    content = (service or get_llm_service()).generate_structured(prepared.request, SummaryContent)
    by_id = {c.chunk_id: c for c in prepared.context.included}
    sources = []
    for point in content.points:
        report = validate_citations(point.detail, prepared.context)
        if not report.all_valid:
            raise LLMResponseError("A summary point is missing valid lecture evidence.")
        sources.append([by_id[cid] for cid in report.valid])
    return Summary(content.points, sources, len(passages))


@dataclass
class Grades:
    results: list[dict[str, Any]]
    ungraded: list[str]

    @property
    def score(self) -> float | None:
        return sum(row["score"] for row in self.results) / len(self.results) if self.results else None


@dataclass
class Practice:
    plan: QuizPlan
    generated: QuizGenerationResult
    quiz_id: str = field(default_factory=lambda: uuid4().hex)
    position: int = 0
    answers: dict[str, str] = field(default_factory=dict)
    grades: Grades | None = None
    saved: bool = False
    explanations: dict[str, dict[str, Any]] | None = None

    @property
    def finished(self) -> bool:
        return self.grades is not None and (self.saved or not self.grades.results)

    def grade(self, *, service: LLMService | None = None) -> None:
        if self.grades is None:
            outcome = evaluate_quiz(
                self.generated.questions, self.answers, self.generated.context.included,
                service=service or get_llm_service(),
            )
            self.grades = Grades(outcome["results"], outcome["ungraded"])

    def save(self, store: ProfileStore, student_id: str, lecture_id: str) -> None:
        if self.finished or self.grades is None:
            return
        # Multiple tabs with one study link can submit different quizzes. Reload
        # inside the lock and key each attempt by the original quiz instance.
        with _profile_write:
            profile = store.load(student_id)
            if not any(attempt.attempt_id == self.quiz_id for attempt in profile.attempts):
                attempt = PersonalizationEngine().record_attempt(
                    profile, self.grades.results, self.plan.difficulty, lecture_id=lecture_id,
                )
                attempt.attempt_id = self.quiz_id
                store.save(profile)
            self.saved = True

    def explain(self, *, service: LLMService | None = None) -> None:
        if self.grades is None or self.explanations is not None:
            return
        # The short-answer grader already returns feedback and validated sources.
        rows = {row["question_id"]: row for row in self.grades.results}
        pending = [q for q in self.generated.questions if not rows.get(q.question_id, {}).get("feedback")]
        if pending:
            additions = generate_feedback(
                pending, self.answers, [rows[q.question_id] for q in pending if q.question_id in rows],
                self.generated.context.included, service=service or get_llm_service(),
                ungraded=self.grades.ungraded,
            )
            rows.update((row["question_id"], row) for row in additions)
        self.explanations = rows


def new_practice(lecture: Lecture, profile: StudentProfile, *, service: LLMService | None = None) -> Practice:
    plan = PersonalizationEngine().plan_next_quiz(profile, available_topics=lecture_topics(lecture.chunks))
    passages = select_passages(lecture.chunks, list(plan.topic_allocation))
    generated = generate_quiz(plan, passages, service=service or get_llm_service())
    return Practice(plan, generated)


@dataclass
class Workspace:
    lecture: Lecture | None = None
    summary: Summary | None = None
    practice: Practice | None = None

    def open_lecture(self, lecture: Lecture) -> None:
        if self.lecture is None or self.lecture.lecture_id != lecture.lecture_id:
            self.lecture = lecture
            self.summary = None
            self.practice = None
