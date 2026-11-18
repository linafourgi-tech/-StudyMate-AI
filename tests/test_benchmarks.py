"""Tests for Role 7 algorithms, not claims of real AI performance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import (FeedbackRubric, QuestionRubric, estimated_cost_usd,
                            grade_mcq_letter, is_valid_mcq, measured_call,
                            response_metrics, summarize_runs, template_based_question,
                            time_saving, validate_case)
from evaluation.run_benchmarks import baseline

CASES = json.loads((Path(__file__).parent.parent / "evaluation" / "benchmark_cases.json").read_text(encoding="utf-8"))


def test_dataset_has_synthetic_supported_and_unsupported_cases():
    assert len(CASES) == 10
    assert sum(c["covered"] for c in CASES) == 8
    assert len({c["case_id"] for c in CASES}) == 10
    for c in CASES:
        validate_case(c)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_template_outputs(case):
    output = template_based_question(case)
    if case["covered"]:
        assert is_valid_mcq(output)
        assert output["options"][output["correct_option"]] == case["correct_answer"]
    else:
        assert output["status"] == "not_covered"
        assert not is_valid_mcq(output)


def test_template_deterministic():
    assert template_based_question(CASES[0]) == template_based_question(CASES[0])


def test_case_rejects_duplicate_options():
    case = dict(CASES[0], distractors=[CASES[0]["correct_answer"], "X", "Y"])
    with pytest.raises(ValueError, match="distinct"):
        validate_case(case)


def test_case_rejects_missing_covered_flag():
    case = dict(CASES[0]); case.pop("covered")
    with pytest.raises(ValueError, match="covered"):
        validate_case(case)


@pytest.mark.parametrize("score", [-1, 3, 1.5, True, None])
def test_question_rubric_rejects_bad_scores(score):
    with pytest.raises(ValueError):
        QuestionRubric(score, 2, 2, 2)


def test_question_rubric_perfect_and_low():
    assert QuestionRubric(2, 2, 2, 2).percentage == 100
    assert QuestionRubric(0, 0, 0, 0).score == 0
    assert QuestionRubric(1, 2, 1, 2).percentage == 75


def test_feedback_rubric():
    assert FeedbackRubric(2, 2, 2).percentage == 100
    assert FeedbackRubric(0, 0, 0).percentage == 0
    with pytest.raises(ValueError):
        FeedbackRubric(2, 4, 1)


def test_structural_validation_does_not_accept_duplicate_distractors():
    data = template_based_question(CASES[0])
    data["options"]["B"] = data["options"]["A"]
    assert not is_valid_mcq(data)


def test_llm_style_array_mcq():
    assert is_valid_mcq({"question":"Which?", "options":["a","b","c","d"], "correct_answer":"c"})
    assert not is_valid_mcq({"question":"Which?", "options":["a","b","c","d"], "correct_answer":"x"})


@pytest.mark.parametrize("value,expected", [("A",True), ("a.",True), ("D",False), ("answer: A",None), ("",None)])
def test_grade_letter(value, expected):
    assert grade_mcq_letter(value,"A") is expected


def test_measured_call_captures_return_value_and_nonnegative_time():
    data, ms = measured_call(lambda: 42)
    assert data == 42 and ms >= 0


def test_cost_formula_uses_million_tokens():
    assert estimated_cost_usd(1000, 500, 1.0, 2.0) == pytest.approx(0.002)
    assert estimated_cost_usd(None, 0, 1.0, 2.0) is None
    assert estimated_cost_usd(10, 10, None, None) is None
    with pytest.raises(ValueError):
        estimated_cost_usd(10, 10, -1.0, 1.0)


def test_cost_disallows_fractional_tokens():
    with pytest.raises(ValueError):
        estimated_cost_usd(0.5, 1, 1, 1)


def test_response_adapter_from_existing_role1_schema_shape():
    llmresponse = SimpleNamespace(content="hello", model="test-model", latency_ms=125.1,
         attempts=2, usage=SimpleNamespace(prompt_tokens=200, completion_tokens=100, total_tokens=300))
    result = response_metrics(llmresponse, "bio_01", "llm", 1.0, 2.0)
    assert result["estimated_cost_usd"] == pytest.approx(0.0004)
    assert result["latency_ms"] == 125.1
    assert result["question_quality_pct"] is None


def test_zero_token_usage_is_marked_unknown():
    llmresponse = SimpleNamespace(content="hello", model="test-model", latency_ms=4.0,
         attempts=1, usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0))
    row = response_metrics(llmresponse, "bio_01", "llm", 1.0, 2.0)
    assert row["prompt_tokens"] is None
    assert row["estimated_cost_usd"] is None


def test_summary_never_invents_missing_quality_metrics():
    records = [{"method":"a", "latency_ms":2, "answer_correct":True, "estimated_cost_usd":None},
               {"method":"a", "latency_ms":4, "answer_correct":False, "estimated_cost_usd":None}]
    result = summarize_runs(records)[0]
    assert result["latency_mean_ms"] == 3
    assert result["accuracy_pct"] == 50
    assert result["question_quality_pct"] is None
    assert result["priced_cases"] == 0


def test_time_saving_positive_and_negative():
    assert time_saving(20, 5)["percentage_saved"] == pytest.approx(75)
    assert time_saving(5, 10)["minutes_saved"] == -5
    with pytest.raises(ValueError):
        time_saving(0, 1)


def test_offline_runner_produces_rows_without_api():
    rows = baseline(CASES)
    assert len(rows) == len(CASES)
    assert all(r["method"] == "annotated_template_baseline" for r in rows)
    assert all(r["estimated_cost_usd"] is None for r in rows)
    assert all(r["question_quality_pct"] is None for r in rows)


def test_grade_audit_checks_actual_vs_reference():
    from src.evaluation import audit_grades
    result = audit_grades([
        {"case_id":"q1", "actual_score":1, "reference_score":1},
        {"case_id":"q2", "actual_score":1, "reference_score":0},
        {"case_id":"q3", "actual_score":0, "reference_score":0},
    ])
    assert result["grade_accuracy_pct"] == pytest.approx(200/3)
    assert result["incorrect_case_ids"] == ["q2"]


def test_grade_audit_empty_is_unknown_not_zero():
    from src.evaluation import audit_grades
    assert audit_grades([])["grade_accuracy_pct"] is None


@pytest.mark.parametrize("invalid", [-1, 2, float('nan'), True])
def test_grade_audit_rejects_invalid_grade(invalid):
    from src.evaluation import audit_grades
    with pytest.raises(ValueError):
        audit_grades([{"case_id":"q1", "actual_score":invalid, "reference_score":1}])


def test_baseline_summary_computes_format_and_missing_info_refusal():
    from src.evaluation import summarize_runs
    result = summarize_runs(baseline(CASES))[0]
    assert result["mcq_cases_checked"] == 8
    assert result["mcq_format_valid_pct"] == 100
    assert result["refusal_cases_checked"] == 2
    assert result["unsupported_refusal_accuracy_pct"] == 100
    assert result["question_quality_pct"] is None
