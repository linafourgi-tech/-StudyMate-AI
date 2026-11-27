"""Run the StudyMate AI student workspace with streamlit run app.py."""

from __future__ import annotations

from html import escape
import re
from uuid import uuid4

import streamlit as st

from src.personalization import PersonalizationEngine, ProfileStore, StudentProfile
from ui import study
from ui.components import badge, diagnostic, go
from ui.study import Workspace
from ui.styles import apply_theme
from ui.views.lecture import lecture_view, summary_view
from ui.views.practice import practice_view
from ui.views.progress import progress_view

NAVIGATION = {
    "Lecture": ":material/menu_book:",
    "Summary": ":material/notes:",
    "Practice quiz": ":material/quiz:",
    "Progress": ":material/monitoring:",
}


def _workspace() -> tuple[str, Workspace]:
    student_id = st.query_params.get("study", "")
    if not re.fullmatch(r"[0-9a-f]{32}", student_id):
        student_id = uuid4().hex
        st.query_params["study"] = student_id
    if st.session_state.get("student_id") != student_id:
        st.session_state.student_id = student_id
        st.session_state.workspace = Workspace()
        st.session_state.view = "Lecture"
    return student_id, st.session_state.workspace


def _sidebar(space: Workspace, profile: StudentProfile) -> None:
    with st.sidebar:
        st.html('<div class="sm-brand"><span class="sm-mark" aria-hidden="true">S</span>'
                '<div><b>StudyMate</b> <small>AI</small></div></div>'
                '<p class="sm-label">Workspace</p>')
        for view, icon in NAVIGATION.items():
            st.button(view, icon=icon, width="stretch", key="nav_" + view.lower().replace(" ", "_"),
                      type="primary" if view == st.session_state.view else "tertiary",
                      on_click=go, args=(view,))
        lecture = space.lecture
        level = PersonalizationEngine().next_difficulty(profile).value
        st.html('<div class="sm-side"><div class="sm-label">Active lecture</div>'
                f'<div class="sm-side-name">{escape(lecture.filename) if lecture else "No PDF open"}</div>'
                f'<div class="sm-label">Next quiz difficulty</div>{badge(level.title(), level)}</div>')
        st.caption("Bookmark this private study link to return to your scores.")


def main() -> None:
    st.set_page_config(page_title="StudyMate AI", page_icon="📖", layout="wide",
                       initial_sidebar_state="auto")
    apply_theme()
    student_id, space = _workspace()
    store = ProfileStore(study.PROFILE_DIRECTORY)
    try:
        profile = store.load(student_id)
    except (OSError, ValueError, TypeError, KeyError):
        st.error("Your saved study history could not be opened. Check the profile file in data/profiles, then retry.")
        st.button("Retry loading history")
        return
    _sidebar(space, profile)
    view = st.session_state.view
    if view == "Lecture":
        lecture_view(space)
    elif view == "Summary":
        summary_view(space)
    elif view == "Practice quiz":
        practice_view(space, profile, store)
    else:
        progress_view(space, profile)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error(diagnostic(exc), icon=":material/error:")
        st.button("Return to lecture", on_click=go, args=("Lecture",))
