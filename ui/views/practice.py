from __future__ import annotations

from html import escape
import re

import streamlit as st

from src.personalization import PersonalizationEngine, ProfileStore, StudentProfile
from src.prompts import lecture_topics
from src.quiz_generator import question_sources
from ui.components import badge, copy, evidence, go, heading, needs_lecture, run_step, title
from ui.lecture import Lecture
from ui.study import Practice, Workspace, new_practice


def start_button(space: Workspace, lecture: Lecture, profile: StudentProfile, label: str) -> None:
    if st.button(label, type="primary", icon=":material/quiz:"):
        def generate() -> None:
            space.practice = new_practice(lecture, profile)
        if run_step("Planning the quiz, generating questions, and checking sources", generate):
            go("Practice quiz")
            st.rerun()


def _save_answer(practice: Practice, qid: str, widget: str) -> None:
    if practice.grades is None:
        practice.answers[qid] = st.session_state[widget] or ""


def _option_label(option: str) -> str:
    # Native radio labels accept Markdown. Show literal lecture/model text.
    return re.sub(r"([\\\x60*_\[\]()<>~$|])", r"\\\1", option)


def _take_quiz(practice: Practice, lecture: Lecture, profile: StudentProfile, store: ProfileStore) -> None:
    questions = practice.generated.questions
    index = practice.position
    question = questions[index]
    qid = question.question_id
    count = len(questions)
    answered = sum(bool(value.strip()) for value in practice.answers.values())
    st.progress(answered / count, text=f"Question {index + 1} of {count} · {answered} answered")
    with st.container(border=True):
        kind = "Multiple choice" if question.question_type == "mcq" else "Short answer"
        st.html(f'{badge(question.difficulty.title(), question.difficulty)} '
                f'<span class="sm-small">{kind} · {escape(question.topic)}</span>')
        st.html(f'<h2 class="sm-question">{escape(question.question)}</h2>')
        widget = f"answer_{practice.quiz_id}_{index}"
        if widget not in st.session_state:
            empty = None if question.question_type == "mcq" else ""
            st.session_state[widget] = practice.answers.get(qid, empty)
        args = (practice, qid, widget)
        if question.question_type == "mcq":
            st.radio("Choose one answer", question.options, key=widget, index=None,
                     format_func=_option_label, on_change=_save_answer, args=args)
        else:
            st.text_area("Your answer", key=widget, placeholder="Explain in your own words…",
                         on_change=_save_answer, args=args)
        evidence(question_sources(question, lecture.chunks, practice.generated.context))

    with st.container(horizontal=True):
        if st.button("Previous", disabled=index == 0, icon=":material/arrow_back:"):
            practice.position -= 1
            st.rerun()
        if index < count - 1:
            if st.button("Next question", type="primary", icon=":material/arrow_forward:"):
                if not practice.answers.get(qid, "").strip():
                    st.info("Enter an answer before moving to the next question.")
                    return
                practice.position += 1
                st.rerun()
        elif st.button("Grade quiz", type="primary", icon=":material/check:"):
            if answered < count:
                st.info("Answer each question before submitting the quiz.")
                return
            def submit() -> None:
                practice.grade()
                practice.save(store, profile.student_id, lecture.lecture_id)
            if run_step("Grading responses against the lecture and saving the attempt", submit):
                st.session_state.explain_once = practice.quiz_id
                st.rerun()


def _review(practice: Practice, lecture: Lecture, profile: StudentProfile, store: ProfileStore) -> None:
    grades = practice.grades
    assert grades is not None
    if not practice.finished:
        st.warning("Your quiz was graded. Save the attempt to include it in your study history.")
        if st.button("Retry saving attempt"):
            if run_step("Saving the graded attempt",
                        lambda: practice.save(store, profile.student_id, lecture.lecture_id)):
                st.session_state.explain_once = practice.quiz_id
                st.rerun()
    score, graded = st.columns(2)
    score.metric("Quiz score", f"{grades.score:.0%}" if grades.score is not None else "Ungraded")
    graded.metric("Questions graded", f"{len(grades.results)} / {len(practice.generated.questions)}")
    if grades.ungraded:
        st.info(f"{len(grades.ungraded)} question(s) had insufficient lecture evidence and were excluded from the score.")
    auto_explain = st.session_state.pop("explain_once", None) == practice.quiz_id
    if practice.explanations is None:
        retry = st.button("Get lecture explanations", icon=":material/menu_book:")
        if (auto_explain or retry) and run_step("Writing explanations and checking evidence", practice.explain):
            st.rerun()
    st.button("View progress", icon=":material/monitoring:", on_click=go, args=("Progress",), type="primary")
    rows = practice.explanations or {r["question_id"]: r for r in grades.results}
    by_id = {c.chunk_id: c for c in lecture.chunks}
    for index, question in enumerate(practice.generated.questions, 1):
        row = rows.get(question.question_id, {})
        value = row.get("score")
        if value is None:
            label, tone = "Ungraded", ""
        elif value == 1:
            label, tone = "Correct", "correct"
        elif value == 0:
            label, tone = "Incorrect", "incorrect"
        else:
            label, tone = f"Partial credit · {value:.0%}", "partial"
        with st.container(border=True):
            st.html(f'{badge(label, tone)} <span class="sm-small">Question {index} · {escape(question.topic)}</span>')
            st.html(f'<h3 class="sm-question">{escape(question.question)}</h3>')
            copy(f"Your answer · {practice.answers.get(question.question_id) or 'Unanswered'}")
            if value is not None and value < 1:
                copy(f"Expected answer · {question.correct_answer}")
            if row.get("feedback"):
                copy(row["feedback"])
            sources = [by_id[cid] for cid in row.get("cited_chunks", []) if cid in by_id]
            if not sources:
                sources = question_sources(question, lecture.chunks, practice.generated.context)
            evidence(sources)


def practice_view(space: Workspace, profile: StudentProfile, store: ProfileStore) -> None:
    lecture, practice = space.lecture, space.practice
    title("Quiz review" if practice and practice.grades else "Practice quiz",
          lecture.filename if lecture else "Practice with questions drawn from your lecture.")
    if lecture is None:
        needs_lecture()
    elif practice is None:
        plan = PersonalizationEngine().plan_next_quiz(profile, lecture_topics(lecture.chunks))
        with st.container(border=True):
            heading("Your quiz plan")
            st.html(f'{badge(plan.difficulty.value.title(), plan.difficulty.value)} '
                    f'<span class="sm-small">{plan.num_questions} questions · Multiple choice and short answer</span>')
            copy("Topics in this quiz · " + ", ".join(plan.topic_allocation))
            start_button(space, lecture, profile, "Generate quiz")
    elif practice.grades is None:
        _take_quiz(practice, lecture, profile, store)
    else:
        _review(practice, lecture, profile, store)
