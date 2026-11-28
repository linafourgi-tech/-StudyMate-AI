"""Validation for generated quizzes at the Role 3 integration boundary."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .grounding import validate_citations
from .schemas import LectureContext


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in",
    "into", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "which", "who", "why", "with", "does", "do", "did", "this", "that",
    "these", "those", "according", "lecture", "material", "provided",
}
def _normalise_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _meaningful_tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", value).casefold()))
    tokens.update(
        token[:-1] for token in tuple(tokens)
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss")
    )
    return tokens - _STOPWORDS


def _evidence_is_cited_quote(evidence: str, cited_ids: set[str], context: LectureContext) -> bool:
    """Require evidence words, in order, to occur in a cited chunk.

    PDF extraction and model quoting can differ in punctuation, line wrapping,
    or hyphenation. Comparing the ordered words keeps provenance strict while
    avoiding failures caused by those formatting differences.
    """
    quote_words = re.findall(r"[a-z0-9]+", _normalise_text(evidence))
    if len(quote_words) < 2:
        return False
    included_by_id = {chunk.chunk_id: chunk for chunk in context.included}
    original_ids = {context.prompt_ids[cid] for cid in cited_ids if cid in context.prompt_ids}

    def contains_ordered_quote(text: str) -> bool:
        source_words = re.findall(r"[a-z0-9]+", _normalise_text(text))
        width = len(quote_words)
        return any(source_words[index:index + width] == quote_words
                   for index in range(len(source_words) - width + 1))

    return any(chunk_id in included_by_id
               and contains_ordered_quote(included_by_id[chunk_id].text)
               for chunk_id in original_ids)


def _answer_has_evidence(answer: str, evidence: str) -> bool:
    """Require meaningful answer terms to overlap with the quoted source evidence."""
    answer_tokens = set(re.findall(r"[a-z0-9]+", _normalise_text(answer)))
    evidence_words = re.findall(r"[a-z0-9]+", _normalise_text(evidence))
    evidence_tokens = set(evidence_words)
    if "lifo" in answer_tokens:
        return "lifo" in evidence_tokens or "last in first out" in " ".join(
            re.findall(r"[a-z0-9]+", _normalise_text(evidence))
        )
    if "fifo" in answer_tokens:
        return "fifo" in evidence_tokens or "first in first out" in " ".join(
            re.findall(r"[a-z0-9]+", _normalise_text(evidence))
        )
    answer_words = [
        token for token in re.findall(r"[a-z0-9]+", _normalise_text(answer))
        if token not in _STOPWORDS
    ]
    if not answer_words:
        return False
    matched = 0
    position = 0
    for token in answer_words:
        try:
            position = evidence_words.index(token, position) + 1
        except ValueError:
            continue
        matched += 1
    return matched / len(answer_words) >= 0.5


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]


def fill_missing_evidence(quiz: Any, context: LectureContext) -> None:
    """Fill omitted evidence fields only from cited source text.

    This keeps older providers and integrations compatible while preserving
    source-based checks: question excerpts must overlap the question wording,
    and answer excerpts must support the expected answer.
    """
    included = {chunk.chunk_id: chunk for chunk in context.included}
    for question in getattr(quiz, "questions", ()):
        original_ids = [
            context.prompt_ids[cid] for cid in getattr(question, "cited_chunks", ())
            if cid in context.prompt_ids
        ]
        sources = [included[cid] for cid in original_ids if cid in included]
        if not getattr(question, "question_evidence", "").strip():
            prompt = re.sub(r"\[[^\]]+\]", " ", str(getattr(question, "question", "")))
            prompt_tokens = _meaningful_tokens(prompt)
            candidates = [
                sentence
                for source in sources
                for sentence in _sentences(source.text)
                if prompt_tokens & _meaningful_tokens(sentence)
            ]
            if candidates:
                setattr(question, "question_evidence", max(
                    candidates,
                    key=lambda sentence: len(prompt_tokens & _meaningful_tokens(sentence)),
                ))
        if not getattr(question, "answer_evidence", "").strip():
            candidates = [
                sentence
                for source in sources
                for sentence in _sentences(source.text)
                if _answer_has_evidence(str(getattr(question, "correct_answer", "")), sentence)
            ]
            if candidates:
                setattr(question, "answer_evidence", candidates[0])


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
    expected_type_distribution: Mapping[str, Mapping[str, int]] | None = None,
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

    if expected_type_distribution is not None:
        for topic, expected in expected_type_distribution.items():
            actual = Counter(
                q.question_type for q in questions if q.topic == topic
            )
            expected_counts = {kind: int(expected.get(kind, 0)) for kind in ("mcq", "short")}
            actual_counts = {kind: actual.get(kind, 0) for kind in ("mcq", "short")}
            if actual_counts != expected_counts:
                issues.append(QuizValidationIssue(
                    "topic_type_distribution",
                    f"Topic {topic!r} expected {expected_counts}; got {actual_counts}.",
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

        answer_evidence = str(getattr(q, "answer_evidence", "")).strip()
        question_evidence = str(getattr(q, "question_evidence", "")).strip()
        if not question_evidence:
            issues.append(QuizValidationIssue(
                "missing_question_evidence", "Question needs a supporting lecture excerpt.", question_id,
            ))
        elif not _evidence_is_cited_quote(question_evidence, set(cited_chunks), context):
            issues.append(QuizValidationIssue(
                "question_evidence_not_cited",
                "Question evidence must quote one of the cited lecture chunks.", question_id,
            ))
        elif not (_meaningful_tokens(question_evidence) & _meaningful_tokens(
            re.sub(r"\[[^\]]+\]", " ", str(q.question))
        )) and not _answer_has_evidence(str(q.correct_answer), answer_evidence):
            issues.append(QuizValidationIssue(
                "question_evidence_mismatch",
                "Question wording and correct answer are not supported by their cited evidence.", question_id,
            ))

        answer_evidence = str(getattr(q, "answer_evidence", "")).strip()
        if not answer_evidence:
            issues.append(QuizValidationIssue(
                "missing_answer_evidence", "Answer needs a supporting lecture excerpt.", question_id,
            ))
        elif not _evidence_is_cited_quote(answer_evidence, set(cited_chunks), context):
            issues.append(QuizValidationIssue(
                "answer_evidence_not_cited",
                "Answer evidence must quote one of the cited lecture chunks.", question_id,
            ))
        elif not _answer_has_evidence(str(q.correct_answer), answer_evidence):
            issues.append(QuizValidationIssue(
                "answer_unsupported",
                "Correct answer has too little meaningful overlap with its cited evidence.", question_id,
            ))
    return issues
