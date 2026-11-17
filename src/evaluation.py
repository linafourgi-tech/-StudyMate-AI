"""StudyMate AI - Role 7: evaluation, benchmarking and quality assurance.

Python 3.12; standard library only. This file does not contact an API.
Record actual runs separately; never present synthetic fixtures as measured results.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable, Mapping
import math

LETTERS = "ABCD"
RUBRIC_CATEGORIES = ("grounding", "correctness", "clarity", "distractors")
FEEDBACK_CATEGORIES = ("grading", "evidence", "usefulness")


def _score(value: int, label: str) -> int:
    if type(value) is not int or value not in (0, 1, 2):
        raise ValueError(f"{label} must be an integer 0, 1, or 2")
    return value


@dataclass(frozen=True)
class QuestionRubric:
    grounding: int
    correctness: int
    clarity: int
    distractors: int

    def __post_init__(self) -> None:
        for name in RUBRIC_CATEGORIES:
            _score(getattr(self, name), name)

    @property
    def score(self) -> int:
        return sum(getattr(self, name) for name in RUBRIC_CATEGORIES)

    @property
    def percentage(self) -> float:
        return 100 * self.score / 8


@dataclass(frozen=True)
class FeedbackRubric:
    grading: int
    evidence: int
    usefulness: int

    def __post_init__(self) -> None:
        for name in FEEDBACK_CATEGORIES:
            _score(getattr(self, name), name)

    @property
    def percentage(self) -> float:
        return 100 * sum(getattr(self, name) for name in FEEDBACK_CATEGORIES) / 6


def validate_case(case: Mapping[str, Any]) -> None:
    """Validate one annotated/synthetic lecture test case."""
    for key in ("case_id", "topic", "lecture_text", "fact_subject", "fact_relation", "correct_answer"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"{key} must be non-empty text")
    distractors = case.get("distractors")
    if not isinstance(distractors, list) or len(distractors) != 3:
        raise ValueError("distractors must have exactly 3 items")
    values = [case["correct_answer"], *distractors]
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError("answer and distractors must be non-empty text")
    if len({x.strip().casefold() for x in values}) != 4:
        raise ValueError("answer and distractors must be distinct")
    if type(case.get("covered")) is not bool:
        raise ValueError("covered must be true or false")


def template_based_question(case: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic, hand-annotated template baseline (not an autonomous AI).

    Requires a manually supplied fact and distractors; preparation effort must
    be considered separately in any time-saving comparison.
    """
    validate_case(case)
    if not case["covered"]:
        return {"question": None, "options": [], "correct_option": None,
                "status": "not_covered", "source_case_id": case["case_id"]}

    idx = int(sha256(case["case_id"].encode("utf-8")).hexdigest()[:8], 16) % 4
    options = list(case["distractors"])
    options.insert(idx, case["correct_answer"])
    return {
        "question": f"{case['fact_subject']} - {case['fact_relation']}?",
        "options": {letter: value for letter, value in zip(LETTERS, options)},
        "correct_option": LETTERS[idx],
        "status": "ok",
        "source_case_id": case["case_id"],
    }


def is_valid_mcq(question: Mapping[str, Any]) -> bool:
    """Structural check only; factual correctness needs human review."""
    opts = question.get("options")
    if not isinstance(question.get("question"), str) or not question["question"].strip():
        return False
    if isinstance(opts, dict):
        if set(opts) != set(LETTERS):
            return False
        values = [opts[key] for key in LETTERS]
        answer = question.get("correct_option")
        return answer in LETTERS and _valid_options(values)
    if isinstance(opts, list):
        values = opts
        answer = question.get("correct_answer")
        return _valid_options(values) and answer in values
    return False


def _valid_options(values: list[Any]) -> bool:
    return (len(values) == 4 and all(isinstance(x, str) and x.strip() for x in values)
            and len({x.strip().casefold() for x in values}) == 4)


def measured_call(function: Callable[[], Any]) -> tuple[Any, float]:
    """Wall time in milliseconds, including this function's total call time."""
    start = perf_counter()
    result = function()
    return result, (perf_counter() - start) * 1000.0


def estimated_cost_usd(prompt_tokens: int | None, completion_tokens: int | None,
                       input_usd_per_million: float | None,
                       output_usd_per_million: float | None) -> float | None:
    """Returns None when usage or the user-supplied provider prices are unknown."""
    values = [prompt_tokens, completion_tokens, input_usd_per_million, output_usd_per_million]
    if any(v is None for v in values):
        return None
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("token counts and prices must be non-negative finite numbers")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise ValueError("token counts must be integers")
    return (prompt_tokens * input_usd_per_million + completion_tokens * output_usd_per_million) / 1_000_000


def response_metrics(response: Any, case_id: str, method: str,
                     input_usd_per_million: float | None = None,
                     output_usd_per_million: float | None = None) -> dict[str, Any]:
    """Adapter for the real src.schemas.LLMResponse from Role 1.

    TokenUsage zeros mean 'not reported' per the existing schema, so usage is
    marked as unknown unless any count is positive. No API keys are logged.
    """
    usage = getattr(response, "usage", None)
    p = getattr(usage, "prompt_tokens", 0)
    c = getattr(usage, "completion_tokens", 0)
    total = getattr(usage, "total_tokens", 0)
    if not any((p, c, total)):
        p = c = None
    cost = estimated_cost_usd(p, c, input_usd_per_million, output_usd_per_million)
    latency = float(response.latency_ms)
    if not math.isfinite(latency) or latency < 0:
        raise ValueError("invalid latency")
    return {
        "case_id": case_id,
        "method": method,
        "model": str(response.model),
        "latency_ms": latency,
        "prompt_tokens": p,
        "completion_tokens": c,
        "estimated_cost_usd": cost,
        "attempts": int(response.attempts),
        "output": response.content,
        "question_quality_pct": None,  # requires human rubric scores
        "feedback_quality_pct": None,  # requires human rubric scores
        "answer_correct": None,         # only set when a ground-truth answer exists
    }


def time_saving(manual_minutes: float, ai_minutes: float) -> dict[str, float]:
    if any(not math.isfinite(x) or x < 0 for x in (manual_minutes, ai_minutes)) or manual_minutes == 0:
        raise ValueError("manual_minutes must be positive; ai_minutes cannot be negative")
    saved = manual_minutes - ai_minutes
    return {"minutes_saved": saved, "percentage_saved": saved / manual_minutes * 100}


def summarize_runs(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Means are based only on known values; no fabricated zeros."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        method = r.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("every record needs a method")
        grouped.setdefault(method, []).append(r)
    summaries: list[dict[str, Any]] = []
    for method, group in sorted(grouped.items()):
        def avg(key: str) -> float | None:
            vals = [float(x[key]) for x in group if x.get(key) is not None]
            return mean(vals) if vals else None
        times = [float(x["latency_ms"]) for x in group if x.get("latency_ms") is not None]
        graded = [bool(x["answer_correct"]) for x in group if x.get("answer_correct") is not None]
        valid_mcqs = [bool(x["structurally_valid"]) for x in group
                      if x.get("structurally_valid") is not None]
        refusals = [bool(x["correct_refusal"]) for x in group
                    if x.get("correct_refusal") is not None]
        summaries.append({
            "method": method,
            "cases": len(group),
            "latency_mean_ms": mean(times) if times else None,
            "latency_median_ms": median(times) if times else None,
            "accuracy_pct": 100 * sum(graded) / len(graded) if graded else None,
            "accuracy_evaluated_cases": len(graded),
            "mcq_format_valid_pct": 100 * sum(valid_mcqs) / len(valid_mcqs) if valid_mcqs else None,
            "mcq_cases_checked": len(valid_mcqs),
            "unsupported_refusal_accuracy_pct": 100 * sum(refusals) / len(refusals) if refusals else None,
            "refusal_cases_checked": len(refusals),
            "question_quality_pct": avg("question_quality_pct"),
            "feedback_quality_pct": avg("feedback_quality_pct"),
            "estimated_cost_mean_usd": avg("estimated_cost_usd"),
            "priced_cases": sum(r.get("estimated_cost_usd") is not None for r in group),
        })
    return summaries


def grade_mcq_letter(response: str, correct_letter: str) -> bool | None:
    """Recognizes one unambiguous choice letter; None for non-parseable output."""
    if correct_letter not in LETTERS:
        raise ValueError("correct_letter must be A, B, C, or D")
    value = response.strip().upper().strip(". )")
    if not value or value not in LETTERS:
        return None
    return value == correct_letter


def audit_grades(graded_records: list[Mapping[str, Any]], tolerance: float = 1e-8) -> dict[str, Any]:
    """Compare Role 4's numeric scores with verified reference scores.

    Required record keys: case_id, actual_score, reference_score (each 0..1).
    Does not judge whether the accompanying natural-language explanation is good.
    """
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    if not graded_records:
        return {"count": 0, "grade_accuracy_pct": None, "incorrect_case_ids": []}
    wrong = []
    for item in graded_records:
        for name in ("actual_score", "reference_score"):
            value = item.get(name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not (0 <= value <= 1)):
                raise ValueError(f"{name} must be a finite score from 0 to 1")
        if abs(item["actual_score"] - item["reference_score"]) > tolerance:
            wrong.append(str(item["case_id"]))
    return {"count": len(graded_records),
            "grade_accuracy_pct": 100 * (len(graded_records) - len(wrong)) / len(graded_records),
            "incorrect_case_ids": wrong}
