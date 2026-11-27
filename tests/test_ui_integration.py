"""Real PDF bytes, role modules and Streamlit controls; only provider replies are mocked."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from openai.types.chat import ChatCompletion
from streamlit.testing.v1 import AppTest

from src.analytics import dashboard_data
from src.config import PROJECT_ROOT, ConfigurationError
from src.llm_service import LLMResponseError, LLMTimeoutError
from src.personalization import ProfileStore
from src.prompts import NOT_IN_MATERIAL_JSON, build_lecture_context, lecture_topics
from src.question_validator import QuizValidationError
from src.quiz_generator import question_sources
from tests.test_llm_service import make_completion, make_service
from ui import study
from ui.lecture import LectureInputError, read_lecture, select_passages


def pdf_bytes(pages: list[tuple[str, str]], **save_options: Any) -> bytes:
    with pymupdf.open() as doc:
        for title, body in pages:
            page = doc.new_page()
            page.insert_text((72, 65), title, fontsize=20)
            page.insert_text((72, 105), body)
        return doc.tobytes(**save_options)


@pytest.fixture
def lecture_pdf() -> bytes:
    return pdf_bytes([
        ("Stacks", "A stack removes the most recently added item first."),
        ("Queues", "A queue removes the earliest added item first."),
    ])


PASSAGES = {"Stacks": ("p1-c1", "Most recently added"), "Queues": ("p2-c1", "Earliest added")}


class ProviderReplies:
    """Reply at the SDK transport boundary, exercising the real grounded prompts and schema parsing."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.bad_citation = False
        self.uncovered = False
        self.feedback_timeout = False

    def __call__(self, *, messages: list[dict[str, str]], **_: Any) -> ChatCompletion:
        system, prompt = (message["content"] for message in messages)
        student = prompt.split("Student input:\n")[-1]
        if "SummaryContent" in system:
            self.calls["summary"] += 1
            citation = "p88-c1" if self.bad_citation else "p1-c1"
            payload = {"points": [{"heading": "Removal order", "detail": f"Stacks remove the newest item [{citation}]."}]}
        elif "StudyMateQuiz" in system:
            self.calls["quiz"] += 1
            plan = json.loads(student)
            allocation, _ = json.JSONDecoder().raw_decode(system.split("per-topic type allocation: ")[1])
            questions = []
            for topic, types in allocation.items():
                citation, answer = PASSAGES[topic]
                assert f"[{citation}]" in prompt
                if self.bad_citation:
                    citation = "p88-c1"
                for kind, count in types.items():
                    for _index in range(count):
                        questions.append({
                            "question_id": f"q{len(questions) + 1}", "question_type": kind,
                            "topic": topic, "difficulty": plan["difficulty"],
                            "question": f"Which item is removed first in {topic.lower()}? [{citation}]",
                            "options": ["Most recently added", "Earliest added", "Largest", "Smallest"] if kind == "mcq" else [],
                            "correct_answer": answer, "cited_chunks": [citation],
                        })
            payload = {"questions": questions}
        elif "ShortAnswerGrades" in system:
            self.calls["grade"] += 1
            if self.uncovered:
                return make_completion(NOT_IN_MATERIAL_JSON)
            grades = []
            for item in json.loads(student)["answers"]:
                cid, expected = PASSAGES[item["topic"]]
                correct = item["student_answer"] == expected
                grades.append({"question_id": item["question_id"], "score": 1 if correct else 0.5,
                               "feedback": f"The removal order is {expected.lower()} [{cid}].", "cited_chunks": [cid]})
            payload = {"grades": grades}
        elif "FeedbackBatch" in system:
            self.calls["feedback"] += 1
            if self.feedback_timeout:
                raise LLMTimeoutError("simulated transport timeout")
            feedback = []
            for item in json.loads(student)["items"]:
                cid, expected = PASSAGES[item["topic"]]
                feedback.append({"question_id": item["question_id"],
                                 "feedback": f"The lecture specifies {expected.lower()} [{cid}].", "cited_chunks": [cid]})
            payload = {"feedback": feedback}
        else:
            pytest.fail("Unexpected model request")
        return make_completion(json.dumps(payload))


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProviderReplies:
    replies = ProviderReplies()
    service, _, _ = make_service(replies, max_retries=0)
    monkeypatch.setattr(study, "get_llm_service", lambda: service)
    monkeypatch.setattr(study, "PROFILE_DIRECTORY", tmp_path)
    return replies


def test_pdf_through_saved_adaptive_quiz(lecture_pdf: bytes, provider: ProviderReplies, tmp_path: Path) -> None:
    pages = []
    lecture = read_lecture("week-1.pdf", lecture_pdf, lambda page, total: pages.append((page, total)))
    assert pages == [(1, 2), (2, 2)]
    assert lecture_topics(lecture.chunks) == ["Stacks", "Queues"]
    assert study.summarize(lecture).sources == [[lecture.chunks[0]]]

    store = ProfileStore(tmp_path)
    practice = study.new_practice(lecture, store.load("learner"))
    for question in practice.generated.questions:
        source, = question_sources(question, lecture.chunks, practice.generated.context)
        assert source.topic == question.topic and source.source == lecture.filename
        practice.answers[question.question_id] = "Largest" if question.question_type == "mcq" else "Unsure"
    practice.grade()
    practice.explain()
    assert practice.grades.score == pytest.approx(0.2)
    assert all(row["cited_chunks"] for row in practice.explanations.values())
    practice.save(store, "learner", lecture.lecture_id)
    practice.saved = False  # A retry even if acknowledgment of the successful write was lost.
    practice.save(store, "learner", lecture.lecture_id)
    profile = store.load("learner")
    dashboard = dashboard_data(profile, available_topics=lecture_topics(lecture.chunks))
    assert dashboard["total_attempts"] == 1
    assert dashboard["score_history"][0]["score"] == pytest.approx(0.2)
    assert set(dashboard["weak_topics"]) == {"Stacks", "Queues"}
    assert dashboard["next_quiz"]["difficulty"] == "easy"
    following = study.new_practice(lecture, profile)
    assert all(q.difficulty == "easy" for q in following.generated.questions)
    assert following.answers == {} and following.quiz_id != practice.quiz_id
    assert provider.calls == {"summary": 1, "quiz": 2, "grade": 1, "feedback": 1}


def test_retrieval_spans_long_lecture_and_respects_budget() -> None:
    body = "A stack grows when an item is pushed. " * 4
    lecture = read_lecture("long.pdf", pdf_bytes([("Stacks" if i % 2 else "Queues", body) for i in range(100)]))
    summary_passages = select_passages(lecture.chunks)
    assert len(summary_passages) < len(lecture.chunks)
    assert (summary_passages[0].page, summary_passages[-1].page) == (1, 100)
    for topics in (["Queues"], ["Stacks", "Queues"]):
        passages = select_passages(lecture.chunks, topics)
        assert set(lecture_topics(passages)) == set(topics)
        assert build_lecture_context(passages, on_overflow="error").is_complete


def test_multiple_bookmarks_on_one_page_keep_their_own_evidence() -> None:
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 70), "Stacks\nThe latest item is removed first.")
        page.insert_text((72, 170), "Queues\nThe earliest item is removed first.")
        doc.new_page().insert_text((72, 70), "The front item leaves the queue.")
        doc.set_toc([[1, "Stacks", 1], [1, "Queues", 1]])
        data = doc.tobytes()
    lecture = read_lecture("lecture_notes.pdf", data)
    assert lecture.title == "lecture notes"
    assert [(c.topic, c.chunk_id) for c in lecture.chunks] == [
        ("Stacks", "p1-c1"), ("Queues", "p1-c2"), ("Queues", "p2-c1"),
    ]
    assert "earliest" not in lecture.chunks[0].text


def test_retrying_a_failed_save_reuses_grades(lecture_pdf: bytes, provider: ProviderReplies,
                                              tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ProfileStore(tmp_path)
    lecture = read_lecture("week-1.pdf", lecture_pdf)
    practice = study.new_practice(lecture, store.load("learner"))
    practice.answers = {q.question_id: q.correct_answer for q in practice.generated.questions}
    practice.grade()

    def fail_save(_profile: object) -> None:
        raise OSError("simulated disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(store, "save", fail_save)
        with pytest.raises(OSError):
            practice.save(store, "learner", lecture.lecture_id)
    assert practice.grades.score == 1 and not practice.finished
    assert store.load("learner").attempts == []
    practice.grade()
    practice.save(store, "learner", lecture.lecture_id)
    assert practice.finished and len(store.load("learner").attempts) == 1
    assert provider.calls["grade"] == 1


@pytest.mark.parametrize("case,message", [
    ("empty", "empty"), ("corrupt", "could not be read"), ("scan", "OCR"), ("locked", "unlocked"),
])
def test_unusable_pdfs_have_specific_recovery(case: str, message: str) -> None:
    if case == "scan":
        with pymupdf.open() as doc:
            page = doc.new_page()
            image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), 0)
            image.clear_with(200)
            page.insert_image(page.rect, pixmap=image)
            data = doc.tobytes()
    elif case == "locked":
        data = pdf_bytes([("Stacks", "Lecture")], encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="reader")
    else:
        data = b"" if case == "empty" else b"This is not a PDF."
    with pytest.raises(LectureInputError, match=message):
        read_lecture("lecture.pdf", data)


def test_unverifiable_outputs_and_ungraded_answers(lecture_pdf: bytes, provider: ProviderReplies, tmp_path: Path) -> None:
    lecture = read_lecture("lecture.pdf", lecture_pdf)
    store = ProfileStore(tmp_path)
    provider.bad_citation = True
    with pytest.raises(LLMResponseError):
        study.summarize(lecture)
    with pytest.raises(QuizValidationError):
        study.new_practice(lecture, store.load("reader"))
    provider.bad_citation = False
    practice = study.new_practice(lecture, store.load("reader"))
    practice.answers = {q.question_id: q.correct_answer for q in practice.generated.questions}
    provider.uncovered = True
    practice.grade()
    practice.explain()
    practice.save(store, "reader", lecture.lecture_id)
    assert len(practice.grades.ungraded) == 2
    assert all(practice.explanations[qid]["ungraded"] for qid in practice.grades.ungraded)
    attempt, = store.load("reader").attempts
    assert attempt.num_questions == 3 and attempt.score == 1.0


def click(app: AppTest, label: str) -> AppTest:
    button, = [button for button in app.main.button if button.label == label]
    assert not button.disabled
    return button.click().run()


def open_pdf(app: AppTest, data: bytes, name: str = "week-1.pdf") -> None:
    app.file_uploader[0].set_value((name, data, "application/pdf")).run()
    click(app, "Open lecture")


def test_streamlit_upload_quiz_retry_and_history(lecture_pdf: bytes, provider: ProviderReplies) -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20).run()
    open_pdf(app, lecture_pdf)
    assert {m.label: m.value for m in app.metric} == {"Pages": "2", "Passages": "2", "Topics": "2"}
    click(app, "Study summary")
    click(app, "Generate summary")
    assert any("p. 1" in exp.label for exp in app.expander)
    click(app, "Practice this lecture")
    click(app, "Generate quiz")
    practice = app.session_state["workspace"].practice
    for index, question in enumerate(practice.generated.questions):
        if question.question_type == "mcq":
            app.radio[0].set_value(question.correct_answer).run()
        else:
            app.text_area[0].set_value(question.correct_answer).run()
        if index == 0:
            app.sidebar.button(key="nav_summary").click().run()
            app.sidebar.button(key="nav_practice_quiz").click().run()
            assert app.radio[0].value == question.correct_answer
        if index < len(practice.generated.questions) - 1:
            click(app, "Next question")
    provider.feedback_timeout = True
    click(app, "Grade quiz")
    assert not app.exception
    assert any("timed out" in banner.value for banner in app.error)
    assert any(m.label == "Quiz score" and m.value == "100%" for m in app.metric)
    provider.feedback_timeout = False
    click(app, "Get lecture explanations")
    assert not app.error and len(app.expander) == len(practice.generated.questions)
    click(app, "View progress")
    assert {m.label: m.value for m in app.metric} == {
        "Latest score": "100%", "Overall accuracy": "100%", "Quizzes graded": "1",
    }
    click(app, "Generate Next Adaptive Quiz")
    assert app.session_state["workspace"].practice.plan.difficulty.value == "hard"
    assert app.radio[0].value is None and not app.exception
    assert provider.calls == {"summary": 1, "quiz": 2, "grade": 1, "feedback": 2}
    # A fresh browser session with the bookmarked link sees the same saved history.
    reopened = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20)
    reopened.query_params["study"] = app.session_state["student_id"]
    reopened.run().sidebar.button(key="nav_progress").click().run()
    assert any(m.label == "Quizzes graded" and m.value == "1" for m in reopened.metric)


def test_streamlit_failure_keeps_previous_lecture(lecture_pdf: bytes, provider: ProviderReplies,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20).run()
    open_pdf(app, lecture_pdf)
    lecture = app.session_state["workspace"].lecture
    open_pdf(app, b"invalid", "broken.pdf")
    assert app.session_state["workspace"].lecture.lecture_id == lecture.lecture_id
    assert any("Export a fresh PDF" in banner.value for banner in app.error)
    app.sidebar.button(key="nav_summary").click().run()
    def missing_key() -> None:
        raise ConfigurationError("missing key")
    with monkeypatch.context() as patch:
        patch.setattr(study, "get_llm_service", missing_key)
        click(app, "Generate summary")
    assert any(".env" in banner.value for banner in app.error)
    click(app, "Generate summary")
    assert app.session_state["workspace"].summary and not app.error and not app.exception
