from __future__ import annotations

from html import escape

import streamlit as st

from src.prompts import lecture_topics
from ui.components import copy, evidence, go, heading, needs_lecture, run_step, title
from ui.lecture import read_lecture
from ui.study import Workspace, summarize
from ui.styles import lecture_header


def _upload(space: Workspace) -> None:
    file = st.file_uploader("Lecture PDF", type=["pdf"], key="lecture_file",
                            label_visibility="collapsed")
    if st.button("Open lecture", icon=":material/upload_file:", type="primary", disabled=file is None):
        def ingest() -> None:
            meter = st.progress(0, text="Opening the PDF")

            def update(page: int, total: int) -> None:
                meter.progress(page / total, text=f"Reading page {page} of {total}")

            lecture = read_lecture(file.name, file.getvalue(), on_page=update)
            space.open_lecture(lecture)

        if run_step("Extracting text and indexing passages", ingest):
            st.rerun()


def lecture_view(space: Workspace) -> None:
    lecture_header()
    lecture = space.lecture
    if lecture is None:
        upload, guide = st.columns([1.7, 1], gap="large")
        with upload, st.container(border=True):
            heading("Open a lecture")
            copy("Drop your PDF below or browse your files.")
            _upload(space)
        with guide:
            heading("Preparing your file")
            copy("Use slides or notes with selectable text. For a scan, run text recognition before uploading.")
            st.caption("You can inspect the extracted passages before sending any material to your model provider.")
        return

    heading(lecture.title)
    st.html(f'<p class="sm-small">{escape(lecture.filename)} · Ready to study</p>')
    pages, passages, topics = st.columns(3)
    pages.metric("Pages", lecture.pages)
    passages.metric("Passages", len(lecture.chunks))
    topics.metric("Topics", len(lecture_topics(lecture.chunks)))
    if lecture.text_pages < lecture.pages:
        st.info(f"Selectable text was found on {lecture.text_pages} of {lecture.pages} pages. "
                "Run OCR on the remaining pages to include them.")
    with st.container(horizontal=True):
        st.button("Study summary", icon=":material/notes:", on_click=go, args=("Summary",), type="primary")
        st.button("Practice quiz", icon=":material/quiz:", on_click=go, args=("Practice quiz",))
    with st.expander("Extracted lecture preview", expanded=True, icon=":material/find_in_page:"):
        page = st.selectbox("Preview page", sorted({c.page for c in lecture.chunks}),
                            key=f"preview_{lecture.lecture_id}")
        for chunk in lecture.chunks:
            if chunk.page == page:
                st.html(f'<p class="sm-small">{escape(chunk.topic or "")} · {escape(chunk.chunk_id)}</p>'
                        f'<p class="sm-passage">{escape(chunk.text)}</p>')
    with st.expander("Change lecture", icon=":material/swap_horiz:"):
        if space.practice and not space.practice.finished:
            st.caption("Opening a different lecture closes the current quiz.")
        _upload(space)


def summary_view(space: Workspace) -> None:
    lecture = space.lecture
    title("Study summary", lecture.filename if lecture else "A reading guide grounded in your lecture.")
    if lecture is None:
        needs_lecture()
        return
    if space.summary is None:
        with st.container(border=True):
            heading("Summarize the lecture")
            copy("Generate a concept-by-concept guide. Each explanation opens back to its supporting passage.")
            if st.button("Generate summary", type="primary", icon=":material/notes:"):
                def generate() -> None:
                    space.summary = summarize(lecture)
                if run_step("Reading selected passages and checking citations", generate):
                    st.rerun()
        return
    summary = space.summary
    st.caption(f"Summary based on {summary.passages_used} of {len(lecture.chunks)} lecture passages.")
    for point, sources in zip(summary.points, summary.sources):
        with st.container(border=True):
            heading(point.heading)
            copy(point.detail)
            evidence(sources, "Read the evidence")
    st.button("Practice this lecture", icon=":material/quiz:", on_click=go, args=("Practice quiz",), type="primary")
