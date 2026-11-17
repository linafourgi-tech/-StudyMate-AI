"""Helpers for interpreting grounded model answers (no API calls).

* ``is_not_in_material(text)``  - did the model use the not-covered fallback?
* ``extract_citations(text)``   - chunk IDs cited as ``[id]`` in an answer.
* ``validate_citations(text, context)`` - check those IDs against the chunks
  that were actually sent (``LectureContext`` from ``prepare_grounded_request``).
"""

from __future__ import annotations

import json
import re
import unicodedata

from .prompts import NOT_IN_MATERIAL, NOT_IN_MATERIAL_KEY
from .schemas import CitationReport, LectureContext

_ID = r"[A-Za-z0-9_.:\-]+"
# "[c1]", "[c1, c2]", "[c1; c2]"; not Markdown links "[text](url)".
_CITATION = re.compile(rf"\[\s*({_ID}(?:\s*[,;]\s*{_ID})*)\s*\](?!\()")
_TRAILING_CITATIONS = re.compile(rf"(?:\s*\[\s*{_ID}(?:\s*[,;]\s*{_ID})*\s*\])+\s*$")
_CODE_FENCE = re.compile(r"^```(?:json|text)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_WRAPPERS = "\"'`*_ \t\r\n“”‘’«»"
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return " ".join(text.split()).casefold()


_PHRASE = _normalise(NOT_IN_MATERIAL).rstrip(".")


def is_not_in_material(text: str | None) -> bool:
    """True when an answer is the not-covered fallback (text or JSON form).

    Accepts the exact phrase ``NOT_IN_MATERIAL`` with harmless formatting
    differences (case, whitespace, curly quotes, surrounding quotes or Markdown
    emphasis, a code fence, a missing final full stop, trailing citations such
    as ``[c1]``) and the JSON fallback object ``{"not_in_material": true, ...}``.

    Returns False for answers that merely *contain* the phrase alongside other
    content (for example a partial answer), so real answers are never hidden.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    fence = _CODE_FENCE.match(stripped)
    if fence:
        stripped = fence.group(1).strip()

    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except ValueError:
            return False
        return isinstance(data, dict) and data.get(NOT_IN_MATERIAL_KEY) is True

    candidate = _TRAILING_CITATIONS.sub("", stripped).strip(_WRAPPERS)
    return _normalise(candidate).rstrip(".").strip(_WRAPPERS) == _PHRASE


def extract_citations(text: str | None) -> list[str]:
    """Return the chunk IDs cited as ``[id]`` (also ``[a, b]``), unique, in order."""
    found: list[str] = []
    for group in _CITATION.findall(text or ""):
        for cid in re.split(r"\s*[,;]\s*", group.strip()):
            if cid and cid not in found:
                found.append(cid)
    return found


def validate_citations(text: str | None, context: LectureContext) -> CitationReport:
    """Check an answer's citations against the chunks that were actually sent.

    ``valid`` holds the original ``chunk_id`` of each cited chunk that was in
    the prompt; ``unknown`` holds cited IDs that were not (omitted chunks,
    typos or invented IDs). Use ``report.all_valid`` for a single yes/no.
    """
    cited = extract_citations(text)
    valid = tuple(context.prompt_ids[c] for c in cited if c in context.prompt_ids)
    unknown = tuple(c for c in cited if c not in context.prompt_ids)
    return CitationReport(cited=tuple(cited), valid=valid, unknown=unknown)
