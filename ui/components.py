"""Shared interface elements and recoverable failure messages."""

from collections.abc import Callable, Sequence
from html import escape
import logging

import streamlit as st

from src.config import ConfigurationError
from src.llm_service import (
    LLMAuthenticationError, LLMNotInMaterialError, LLMRateLimitError, LLMRequestError,
    LLMResponseError, LLMTimeoutError, LLMTransientError,
)
from src.prompts import LectureContextOverflowError
from src.question_validator import QuizValidationError
from src.schemas import LectureChunk
from ui.lecture import LectureInputError


def go(view: str) -> None:
    st.session_state.view = view


def title(heading: str, description: str) -> None:
    st.html(f'<header class="sm-title"><h1>{escape(heading)}</h1><p>{escape(description)}</p></header>')


def heading(value: str) -> None:
    st.html(f'<h2 class="sm-h2">{escape(value)}</h2>')


def copy(value: str) -> None:
    st.html(f'<p class="sm-copy">{escape(value)}</p>')


def badge(label: str, tone: str = "") -> str:
    colors = {"easy", "medium", "hard", "correct", "partial", "incorrect"}
    return f'<span class="sm-badge sm-{tone if tone in colors else "neutral"}">{escape(label)}</span>'


def evidence(chunks: Sequence[LectureChunk], label: str = "Source passage") -> None:
    if not chunks:
        return
    pages = ", ".join(str(page) for page in dict.fromkeys(c.page for c in chunks))
    with st.expander(f"{label} · p. {pages}", icon=":material/menu_book:"):
        for chunk in chunks:
            st.html(
                f'<p class="sm-small">{escape(chunk.source or "Lecture")} · Page {chunk.page} '
                f'· {escape(chunk.chunk_id)}</p><p class="sm-passage">{escape(chunk.text)}</p>'
            )


def needs_lecture() -> None:
    st.info("Open a lecture PDF to start this step.", icon=":material/menu_book:")
    st.button("Open a lecture", on_click=go, args=("Lecture",), type="primary")


def diagnostic(exc: Exception) -> str:
    if isinstance(exc, LectureInputError):
        return str(exc)
    if isinstance(exc, ConfigurationError):
        return ("Check the provider settings in the project's .env file. Set LLM_PROVIDER and its "
                "matching OPENAI_API_KEY or GEMINI_API_KEY, then retry.")
    if isinstance(exc, LLMAuthenticationError):
        return "The provider rejected the API credentials. Update the key, restart the app, and retry."
    if isinstance(exc, LLMRateLimitError):
        return "The provider is rate-limiting requests. Wait a moment, then retry this step."
    if isinstance(exc, LLMTimeoutError):
        return "The provider timed out. Retry this step when the connection is ready."
    if isinstance(exc, LLMTransientError):
        return "The provider is temporarily unavailable. Check your connection and retry."
    if isinstance(exc, LLMNotInMaterialError):
        return "The selected passages do not cover this request. Try a lecture with more detail."
    if isinstance(exc, LectureContextOverflowError):
        return "These passages exceed the model's context budget. Try a shorter lecture section."
    if isinstance(exc, LLMRequestError):
        return "The provider rejected this request. Check the model name and available quota, then retry."
    if isinstance(exc, (QuizValidationError, LLMResponseError, ValueError)):
        return "The response could not be verified against the lecture. Retry this step."
    if isinstance(exc, OSError):
        return "Study history could not be read or saved. Check access to data/profiles and retry."
    logging.getLogger(__name__).error("Study workspace operation failed (%s)", type(exc).__name__)
    return "This step was interrupted. Retry it or reopen the lecture workspace."


def run_step(label: str, action: Callable[[], None]) -> bool:
    with st.status(label, expanded=True) as status:
        try:
            action()
        except Exception as exc:
            status.update(label="This step needs attention", state="error", expanded=True)
            st.error(diagnostic(exc), icon=":material/error:")
            return False
        status.update(label="Complete", state="complete", expanded=False)
    return True
