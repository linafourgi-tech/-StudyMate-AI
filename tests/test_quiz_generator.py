"""Role 3 quiz generation, validation, grounding, and shared-service tests."""

from __future__ import annotations

import json
from collections import Counter

from src.personalization import Difficulty, QuizPlan
from src.prompts import prepare_grounded_request
from src.quiz_generator import generate_quiz, make_quiz_schema, question_sources
from src.question_validator import validate_quiz
from src.schemas import LectureChunk
from tests.test_llm_service import make_completion, make_service


CHUNKS = [
    LectureChunk(
        chunk_id="stack-p1",
        text=("A stack follows LIFO: the last item added is the first item removed. "
              "LIFO means last in, first out."),
        source="lecture.pdf",
        page=1,
        topic="Stacks",
    ),
    LectureChunk(
        chunk_id="queue-p2",
        text=("A queue follows FIFO: the first item added is the first item removed. "
              "FIFO means first in, first out."),
        source="lecture.pdf",
        page=2,
        topic="Queues",
    ),
]
PLAN = QuizPlan(
    difficulty=Difficulty.MEDIUM,
    focus_topics=["Stacks", "Queues"],
    num_questions=4,
    reason="Test plan",
    topic_allocation={"Stacks": 2, "Queues": 2},
)
TYPE_DISTRIBUTION = {
    "Stacks": {"mcq": 1, "short": 1},
    "Queues": {"mcq": 1, "short": 1},
}


def _question(question_id: str, topic: str, question_type: str) -> dict:
    if topic == "Stacks":
        chunk_id = "stack-p1"
        concept = "LIFO"
        prompt_evidence = "A stack follows LIFO"
        answer_evidence = "LIFO means last in, first out."
    else:
        chunk_id = "queue-p2"
        concept = "FIFO"
        prompt_evidence = "A queue follows FIFO"
        answer_evidence = "FIFO means first in, first out."

    if question_type == "mcq":
        question = f"Which access rule does a {topic[:-1].lower()} follow? [{chunk_id}]"
        options = ["LIFO", "FIFO", "Random order", "No order"]
        correct_answer = concept
    else:
        question = f"What does {concept} mean for a {topic[:-1].lower()}? [{chunk_id}]"
        options = []
        correct_answer = "last in, first out" if topic == "Stacks" else "first in, first out"
    return {
        "question_id": question_id,
        "topic": topic,
        "difficulty": "medium",
        "question_type": question_type,
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "cited_chunks": [chunk_id],
        "question_evidence": prompt_evidence,
        "answer_evidence": answer_evidence,
    }


def _quiz(plan=PLAN, questions=None):
    questions = questions or [
        _question("q1", "Stacks", "mcq"),
        _question("q2", "Stacks", "short"),
        _question("q3", "Queues", "mcq"),
        _question("q4", "Queues", "short"),
    ]
    return make_quiz_schema(plan)(questions=questions)


def _context(chunks=CHUNKS):
    return prepare_grounded_request(
        "Generate a quiz.", chunks, json.dumps(PLAN.to_dict()),
        response_format="json", max_output_tokens=3000, on_overflow="error",
    ).context


def test_valid_mixed_quiz_passes_type_distribution_and_evidence_checks():
    quiz = _quiz()

    issues = validate_quiz(
        quiz,
        PLAN,
        _context(),
        expected_short_answer_count=2,
        expected_type_distribution=TYPE_DISTRIBUTION,
    )

    assert issues == []
    assert Counter((q.topic, q.question_type) for q in quiz.questions) == Counter({
        ("Stacks", "mcq"): 1,
        ("Stacks", "short"): 1,
        ("Queues", "mcq"): 1,
        ("Queues", "short"): 1,
    })


def test_accepts_supported_paraphrased_question_with_no_shared_content_words():
    questions = [_question("q1", "Stacks", "mcq"), _question("q2", "Stacks", "short"),
                 _question("q3", "Queues", "mcq"), _question("q4", "Queues", "short")]
    questions[0]["question"] = "What happens to the newest element? [stack-p1]"
    quiz = _quiz(questions=questions)

    issues = validate_quiz(quiz, PLAN, _context(), expected_type_distribution=TYPE_DISTRIBUTION)

    assert not any(issue.code in {"question_evidence_mismatch", "answer_unsupported"}
                   and issue.question_id == "q1" for issue in issues)


def test_accepts_evidence_quote_with_pdf_punctuation_and_line_break_variation():
    questions = [_question("q1", "Stacks", "mcq"), _question("q2", "Stacks", "short"),
                 _question("q3", "Queues", "mcq"), _question("q4", "Queues", "short")]
    questions[0]["question_evidence"] = (
        "A stack follows LIFO - the last item added is the first item removed"
    )
    quiz = _quiz(questions=questions)

    issues = validate_quiz(quiz, PLAN, _context(), expected_type_distribution=TYPE_DISTRIBUTION)

    assert not any(issue.code == "question_evidence_not_cited" and issue.question_id == "q1"
                   for issue in issues)


def test_rejects_wrong_mcq_short_distribution_with_totals_unchanged():
    questions = [
        _question("q1", "Stacks", "mcq"),
        _question("q2", "Stacks", "mcq"),
        _question("q3", "Queues", "short"),
        _question("q4", "Queues", "short"),
    ]
    quiz = _quiz(questions=questions)

    issues = validate_quiz(
        quiz,
        PLAN,
        _context(),
        expected_short_answer_count=2,
        expected_type_distribution=TYPE_DISTRIBUTION,
    )

    mismatches = [issue for issue in issues if issue.code == "topic_type_distribution"]
    assert {issue.message.split(" expected", 1)[0] for issue in mismatches} == {
        "Topic 'Stacks'", "Topic 'Queues'",
    }


def test_rejects_correct_answer_not_supported_by_its_cited_evidence():
    questions = [_question("q1", "Stacks", "mcq"), _question("q2", "Stacks", "short"),
                 _question("q3", "Queues", "mcq"), _question("q4", "Queues", "short")]
    questions[0]["correct_answer"] = "FIFO"
    quiz = _quiz(questions=questions)

    issues = validate_quiz(quiz, PLAN, _context(), expected_type_distribution=TYPE_DISTRIBUTION)

    assert any(issue.code == "answer_unsupported" and issue.question_id == "q1" for issue in issues)


def test_rejects_evidence_quote_missing_from_the_cited_chunk():
    questions = [_question("q1", "Stacks", "mcq"), _question("q2", "Stacks", "short"),
                 _question("q3", "Queues", "mcq"), _question("q4", "Queues", "short")]
    questions[0]["answer_evidence"] = "A binary tree has at most two children."
    quiz = _quiz(questions=questions)

    issues = validate_quiz(quiz, PLAN, _context(), expected_type_distribution=TYPE_DISTRIBUTION)

    assert any(issue.code == "answer_evidence_not_cited" and issue.question_id == "q1" for issue in issues)


def test_rejects_question_unrelated_to_its_cited_evidence():
    questions = [_question("q1", "Stacks", "mcq"), _question("q2", "Stacks", "short"),
                 _question("q3", "Queues", "mcq"), _question("q4", "Queues", "short")]
    questions[0]["question"] = "Which sorting algorithm uses a pivot? [stack-p1]"
    questions[0]["correct_answer"] = "pivot"
    questions[0]["options"] = ["pivot", "FIFO", "Random order", "No order"]
    quiz = _quiz(questions=questions)

    issues = validate_quiz(quiz, PLAN, _context(), expected_type_distribution=TYPE_DISTRIBUTION)

    assert any(issue.code == "question_evidence_mismatch" and issue.question_id == "q1" for issue in issues)


def test_rejects_unknown_citation_even_if_evidence_is_valid_elsewhere():
    questions = [_question("q1", "Stacks", "mcq"), _question("q2", "Stacks", "short"),
                 _question("q3", "Queues", "mcq"), _question("q4", "Queues", "short")]
    questions[0]["cited_chunks"] = ["unknown"]
    questions[0]["question"] = questions[0]["question"].replace("[stack-p1]", "[unknown]")
    quiz = _quiz(questions=questions)

    issues = validate_quiz(quiz, PLAN, _context(), expected_type_distribution=TYPE_DISTRIBUTION)

    assert any(issue.code == "unknown_citations" and issue.question_id == "q1" for issue in issues)
    assert any(issue.code == "question_evidence_not_cited" and issue.question_id == "q1" for issue in issues)


def test_generator_uses_shared_llm_service_and_resolves_source_chunks():
    response_questions = [
        _question("q1", "Stacks", "mcq"),
        _question("q2", "Stacks", "short"),
        _question("q3", "Queues", "mcq"),
        _question("q4", "Queues", "short"),
    ]
    # The generator must also ground older adapters that omit evidence fields.
    for question in response_questions:
        question.pop("question_evidence")
        question.pop("answer_evidence")
    service, client, _ = make_service([make_completion(json.dumps({"questions": response_questions}))])

    result = generate_quiz(PLAN, CHUNKS, service=service)

    assert len(result.questions) == 4
    assert client.chat.completions.create.call_count == 1
    sent_request = client.chat.completions.create.call_args.kwargs
    assert sent_request["response_format"] == {"type": "json_object"}
    assert {q.question_type for q in result.questions} == {"mcq", "short"}
    assert all(q.question_evidence and q.answer_evidence for q in result.questions)
    assert question_sources(result.questions[0], CHUNKS, result.context) == [CHUNKS[0]]


def test_rejects_invalid_mcq_options():
    questions = [_question("q1", "Stacks", "mcq"), _question("q2", "Stacks", "short"),
                 _question("q3", "Queues", "mcq"), _question("q4", "Queues", "short")]
    questions[0]["correct_answer"] = "not an option"
    quiz = _quiz(questions=questions)

    issues = validate_quiz(quiz, PLAN, _context(), expected_type_distribution=TYPE_DISTRIBUTION)

    assert any(issue.code == "answer_not_option" and issue.question_id == "q1" for issue in issues)
