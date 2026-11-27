"""Convert an uploaded PDF to the team's LectureChunk contract."""

from __future__ import annotations

import hashlib
import re
import textwrap
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import pymupdf

from src.prompts import DEFAULT_MAX_CONTEXT_CHARS, build_lecture_context, chunks_for_topics
from src.schemas import LectureChunk

_pdf_access = Lock()
_page_number = re.compile(r"(?:page\s+)?\d+(?:\s*/\s*\d+)?", re.I)


class LectureInputError(ValueError):
    """A PDF problem with a student-facing recovery instruction."""


@dataclass(frozen=True)
class Lecture:
    lecture_id: str
    filename: str
    title: str
    pages: int
    text_pages: int
    chunks: tuple[LectureChunk, ...]


def _clean_line(line: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", line).replace("\xad", "").split())


def _page_sections(
    lines: list[str], bookmarks: list[str], section: str, fallback: str,
) -> tuple[list[tuple[str, str]], str]:
    labels = {label.casefold(): label for label in bookmarks}
    if bookmarks and not any(line.casefold() in labels for line in lines):
        section = bookmarks[0]  # Some PDFs use outline labels absent from the page text.
    first = lines[0] if lines else ""
    heading = first if len(first) <= 80 and not first.endswith((".", "?", "!")) else ""
    topic = section or heading or fallback
    parts: list[tuple[str, str]] = []
    paragraph: list[str] = []
    for line in lines:
        matched = labels.get(line.casefold())
        if matched:
            if paragraph:
                parts.append((topic, " ".join(paragraph)))
            section = topic = matched
            paragraph = []
        paragraph.append(line)
    if paragraph:
        parts.append((topic, " ".join(paragraph)))
    return parts, section


def read_lecture(
    filename: str, data: bytes, on_page: Callable[[int, int], None] | None = None,
) -> Lecture:
    if not data:
        raise LectureInputError("This file is empty. Choose a PDF that contains lecture text.")
    filename = Path(filename.replace("\\", "/")).name or "lecture.pdf"
    lecture_id = hashlib.sha256(data).hexdigest()
    chunks: list[LectureChunk] = []
    text_pages = 0
    try:
        # MuPDF document operations must be serialized across Streamlit sessions.
        with _pdf_access, pymupdf.open(stream=data, filetype="pdf") as document:
            if not document.is_pdf:
                raise LectureInputError("This file could not be read as a PDF. Export a fresh PDF and try again.")
            if document.needs_pass:
                raise LectureInputError("This PDF is locked. Upload an unlocked copy.")
            pages = len(document)
            title = _clean_line((document.metadata or {}).get("title") or Path(filename).stem.replace("_", " "))
            bookmarks: dict[int, list[str]] = defaultdict(list)
            for _, label, page_number in document.get_toc():
                if page_number > 0 and label.strip():
                    bookmarks[page_number].append(_clean_line(label))
            section = ""
            for number, page in enumerate(document, start=1):
                raw = page.get_text("text", sort=True, flags=pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE)
                lines = [_clean_line(line) for line in raw.splitlines()]
                lines = [line for line in lines if line and not _page_number.fullmatch(line)]
                parts, section = _page_sections(lines, bookmarks[number], section, title or f"Page {number}")
                index = 0
                for topic, body in parts:
                    if not any(char.isalnum() for char in body):
                        continue
                    for passage in textwrap.wrap(body, width=1100, break_on_hyphens=False):
                        index += 1
                        chunks.append(LectureChunk(
                            chunk_id=f"p{number}-c{index}", text=passage, page=number,
                            topic=topic[:90], source=filename, lecture_id=lecture_id,
                        ))
                if index:
                    text_pages += 1
                if on_page:
                    on_page(number, pages)
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise LectureInputError("This PDF could not be read. Export a fresh PDF and upload it again.") from exc
    if not chunks:
        raise LectureInputError(
            "No selectable lecture text was found. For scanned slides, run text recognition (OCR) "
            "and upload the searchable PDF."
        )
    return Lecture(lecture_id, filename, title, pages, text_pages, tuple(chunks))


def _spread(items: Sequence[LectureChunk], count: int) -> list[LectureChunk]:
    if count == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def select_passages(
    chunks: Sequence[LectureChunk], topics: Sequence[str] = (),
) -> list[LectureChunk]:
    """Fit the existing model budget while covering the document or every planned topic."""
    eligible = chunks_for_topics(chunks, topics) if topics else list(chunks)
    if not eligible:
        raise ValueError("No passages match the requested topics.")
    # A single-chunk context includes its real header and delimiters. Summing
    # these sizes is conservative, including if the grader reorders the chunks.
    sizes = [len(build_lecture_context([chunk]).text) for chunk in eligible]
    if sum(sizes) <= DEFAULT_MAX_CONTEXT_CHARS:
        return eligible
    capacity = DEFAULT_MAX_CONTEXT_CHARS // max(sizes)
    groups = [chunks_for_topics(eligible, topic) for topic in topics] if topics else [eligible]
    if capacity < len(groups) or any(not group for group in groups):
        raise ValueError("The planned topics could not all be represented in one prompt.")
    # Give each topic a slot before allocating additional passages in rounds.
    counts = [1] * len(groups)
    remaining = capacity - len(groups)
    while remaining:
        for i, group in enumerate(groups):
            if remaining and counts[i] < len(group):
                counts[i] += 1
                remaining -= 1
    chosen = {c.chunk_id for group, count in zip(groups, counts) for c in _spread(group, count)}
    return [chunk for chunk in eligible if chunk.chunk_id in chosen]
