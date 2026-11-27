from __future__ import annotations

from typing import Any

import streamlit as st

from src.analytics import dashboard_data
from src.personalization import PersonalizationEngine, StudentProfile
from src.prompts import lecture_topics
from ui.components import badge, copy, go, heading, needs_lecture, title
from ui.study import Workspace
from ui.views.practice import start_button


def _history(rows: list[dict[str, Any]]) -> None:
    heading("Score progression")
    st.vega_lite_chart(spec={
        "data": {"values": rows},
        "height": 195,
        "mark": {"type": "line", "color": "#22634d", "strokeWidth": 3,
                 "point": {"filled": True, "size": 90, "color": "#22634d"}},
        "config": {"axis": {"labelColor": "#42574c", "labelFontSize": 12, "titleFontSize": 13,
                            "titleColor": "#344b3f", "gridColor": "#dce4dc"},
                   "view": {"stroke": None}, "background": "transparent"},
        "encoding": {
            "x": {"field": "attempt", "type": "ordinal", "sort": "ascending", "title": "Quiz",
                  "axis": {"labelAngle": 0}},
            "y": {"field": "score", "type": "quantitative", "scale": {"domain": [0, 1]},
                  "axis": {"format": ".0%", "tickCount": 5}, "title": None},
            "tooltip": [
                {"field": "attempt", "title": "Quiz"},
                {"field": "score", "type": "quantitative", "format": ".0%", "title": "Score"},
                {"field": "difficulty", "title": "Difficulty"},
                {"field": "num_questions", "title": "Questions graded"},
            ],
        },
    }, width="stretch", theme=None, alt="Quiz score by attempt, in chronological order.")


def progress_view(space: Workspace, profile: StudentProfile) -> None:
    lecture = space.lecture
    data = dashboard_data(profile, PersonalizationEngine(), lecture_topics(lecture.chunks) if lecture else [])
    history = data["score_history"]
    plan = data["next_quiz"]
    title("Study progress", "Your graded attempts, topic accuracy, and next practice plan.")
    if history:
        latest, accuracy, total = st.columns(3)
        latest.metric("Latest score", f"{history[-1]['score']:.0%}")
        accuracy.metric("Overall accuracy", f"{data['overall_accuracy']:.0%}")
        total.metric("Quizzes graded", data["total_attempts"])
        trend_column, plan_column = st.columns([1.15, 1], gap="medium")
        with trend_column.container(border=True):
            _history(history)
    else:
        st.info("Complete a quiz to see your first score and topic breakdown.", icon=":material/monitoring:")
        plan_column = st.container()

    with plan_column.container(border=True):
        heading("Next adaptive quiz")
        st.html(badge(plan["difficulty"].title(), plan["difficulty"]))
        decision = plan["reason"].partition(";")[0]
        copy(decision[:1].upper() + decision[1:] + ".")
        if lecture:
            allocation = ", ".join(f"{topic} ({count})" for topic, count in plan["topic_allocation"].items())
            copy("Questions by topic · " + allocation)
        if lecture is None:
            needs_lecture()
        elif space.practice and not space.practice.finished:
            st.button("Resume current quiz", on_click=go, args=("Practice quiz",), type="primary")
        else:
            start_button(space, lecture, profile, "Generate Next Adaptive Quiz")

    if history:
        with st.container(border=True):
            heading("Topic mastery")
            if data["weak_topics"]:
                copy("Weak topics · " + ", ".join(data["weak_topics"]))
            else:
                copy("No topics currently meet the weak-topic threshold.")
            for row in data["topics"]:
                count = row["answered"]
                status = {"ok": "Developing", "insufficient data": "Gathering evidence"}.get(
                    row["status"], row["status"].title())
                st.progress(row["accuracy"], text=f"{row['topic']} · {row['accuracy']:.0%}")
                st.caption(f"{status} · {count} graded {'answer' if count == 1 else 'answers'}")
