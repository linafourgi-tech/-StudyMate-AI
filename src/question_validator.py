"""Validation for generated quizzes at the Role 3 integration boundary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .grounding import validate_citations
from .schemas import LectureContext


@dataclass(frozen=True)
class QuizValidationIssue:
    code: str
    message: str
    question_id: str | None = None


class QuizValidationError(ValueError):
    """Raised when a generated quiz fails deterministic quality checks."""

    def __init__(self, issues: Sequence[QuizValidationIssue]) -> None:
        self.issues = tuple(issues)
        details = "; ".join(issue.message for issue in self.issues)
        super().__init__(f"Generated quiz failed validation: {details}")


def validate_quiz(
    quiz: Any,
    plan: Any,
    context: LectureContext,
    *,
    expected_short_answer_count: int | None = None,
) -> list[QuizValidationIssue]:
    """Check plan counts, question structure, and citations against supplied chunks."""
    issues: list[QuizValidationIssue] = []
    questions = list(getattr(quiz, "questions", ()))
    allocation = dict(plan.topic_allocation)
    difficulty = getattr(plan.difficulty, "value", plan.difficulty)

    if not questions:
        issues.append(QuizValidationIssue("empty_quiz", "Quiz must contain at least one question."))
        return issues
    if len(questions) != plan.num_questions:
        issues.append(QuizValidationIssue(
            "question_count",
            f"Expected {plan.num_questions} questions from the plan; got {len(questions)}.",
        ))

    counts = Counter(str(q.topic) for q in questions)
    if counts != Counter(allocation):
        issues.append(QuizValidationIssue(
            "topic_allocation",
            f"Question counts by topic {dict(counts)} do not match the plan {allocation}.",
        ))

    short_count = sum(1 for q in questions if q.question_type == "short")
    if expected_short_answer_count is not None and short_count != expected_short_answer_count:
        issues.append(QuizValidationIssue(
            "short_answer_count",
            f"Expected {expected_short_answer_count} short-answer questions; got {short_count}.",
        ))

    seen_ids: set[str] = set()
    valid_prompt_ids = set(context.prompt_ids)
    for q in questions:
        question_id = str(getattr(q, "question_id", ""))
        if not question_id.strip():
            issues.append(QuizValidationIssue("missing_question_id", "Every question needs an ID."))
        elif question_id in seen_ids:
            issues.append(QuizValidationIssue(
                "duplicate_question_id", f"Question ID {question_id!r} is duplicated.", question_id,
            ))
        seen_ids.add(question_id)

        if q.topic not in allocation:
            issues.append(QuizValidationIssue(
                "unknown_topic", f"Topic {q.topic!r} is not in the quiz plan.", question_id,
            ))
        if q.difficulty != difficulty:
            issues.append(QuizValidationIssue(
                "wrong_difficulty", f"Question difficulty must be {difficulty!r}.", question_id,
            ))
        if q.question_type not in {"mcq", "short"}:
            issues.append(QuizValidationIssue(
                "unsupported_question_type", "Question type must be 'mcq' or 'short'.", question_id,
            ))

        options = [str(option).strip() for option in q.options]
        if q.question_type == "mcq":
            if len(options) != 4 or any(not option for option in options):
                issues.append(QuizValidationIssue(
                    "invalid_options", "Each MCQ needs exactly four non-empty options.", question_id,
                ))
            if len({option.casefold() for option in options}) != len(options):
                issues.append(QuizValidationIssue(
                    "duplicate_options", "MCQ options must be distinct.", question_id,
                ))
            if str(q.correct_answer).strip().casefold() not in {o.casefold() for o in options}:
                issues.append(QuizValidationIssue(
                    "answer_not_option", "Correct answer must exactly match one option.", question_id,
                ))
        elif q.question_type == "short" and options:
            issues.append(QuizValidationIssue(
                "short_answer_has_options", "Short-answer questions must not include options.", question_id,
            ))

        cited_chunks = list(q.cited_chunks)
        if not cited_chunks:
            issues.append(QuizValidationIssue(
                "missing_citations", "Each question must cite at least one source chunk.", question_id,
            ))
        unknown = sorted(set(cited_chunks) - valid_prompt_ids)
        if unknown:
            issues.append(QuizValidationIssue(
                "unknown_citations", f"Citations were not included in the prompt: {unknown}.", question_id,
            ))
        report = validate_citations(str(q.question), context)
        if not report.all_valid:
            issues.append(QuizValidationIssue(
                "ungrounded_question", "Question text must cite at least one valid supplied chunk.", question_id,
            ))
        elif set(report.cited) != set(cited_chunks):
            issues.append(QuizValidationIssue(
                "citation_mismatch", "Question text citations must match cited_chunks.", question_id,
            ))
    return issues
