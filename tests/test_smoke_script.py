"""Offline tests for the diagnostics in ``scripts/llm_smoke_test.py``.

No network: the SDK client is routed to an in-memory transport and the
service is injected, so no real API key or request is used.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx2
import pytest
from google import genai

from src.llm_service import LLMRequestError, LLMService

from tests.test_gemini_backend import (
    FAKE_GEMINI_KEY,
    REQUEST,
    api_error,
    gemini_settings,
    make_gemini_service,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "llm_smoke_test.py"
_spec = importlib.util.spec_from_file_location("llm_smoke_test", _SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _failed_404(text):
    service, _, _ = make_gemini_service([(404, api_error(404, "NOT_FOUND", text))])
    with pytest.raises(LLMRequestError) as info:
        service.generate(REQUEST)
    return info.value


class TestProviderDiagnostics:
    def test_shows_provider_reason_for_404(self):
        err = _failed_404("models/gemini-2.5-flash is not found for API version v1")
        out = smoke.provider_diagnostics(err, (FAKE_GEMINI_KEY,))
        assert out == ("Provider said: HTTP 404 NOT_FOUND: "
                       "models/gemini-2.5-flash is not found for API version v1")

    def test_redacts_configured_key_and_key_shaped_text(self):
        err = _failed_404(f"bad {FAKE_GEMINI_KEY} and AIzaSyA1234567890abcdefghijklmnopqrstu")
        out = smoke.provider_diagnostics(err, (FAKE_GEMINI_KEY,))
        assert FAKE_GEMINI_KEY not in out
        assert "AIzaSy" not in out
        assert out.count("[REDACTED]") == 2

    def test_truncates_long_provider_text(self):
        out = smoke.provider_diagnostics(_failed_404("x" * 2000))
        assert out.endswith("...") and len(out) < 600

    def test_none_without_provider_cause(self):
        assert smoke.provider_diagnostics(LLMRequestError("no cause")) is None


class _Model:
    def __init__(self, name, actions):
        self.name, self.supported_actions = name, actions


class _Models:
    def list(self):
        return [_Model("models/gemini-2.5-flash", ["generateContent"]),
                _Model("models/embedding-001", ["embedContent"])]


class _Client:
    models = _Models()


class TestListModelsWording:
    def test_listing_does_not_claim_generation_works(self, monkeypatch, capsys):
        # Regression: the listing used to print "AVAILABLE to this key", which reads
        # as a guarantee that generateContent will succeed. It is only catalog data.
        settings = gemini_settings(model="gemini-2.5-flash")
        monkeypatch.setattr(smoke, "load_settings", lambda: settings)
        monkeypatch.setattr(smoke, "LLMService", lambda s: LLMService(s, client=_Client()))
        assert smoke.list_models() == 0
        out = capsys.readouterr().out
        assert "AVAILABLE" not in out
        assert "LISTED with generateContent" in out
        assert "does not prove generation will succeed" in out
        assert "(API version unknown)" in out
        assert "embedding-001" not in out


def _catalog_client(pages):
    """Real SDK client whose models.list() is served from scripted pages (no network)."""
    def handler(request):
        assert request.method == "GET" and request.url.path.endswith("/models")
        return httpx2.Response(200, json=pages[request.url.params.get("pageToken")])

    return genai.Client(api_key=FAKE_GEMINI_KEY, vertexai=False, http_options={
        "httpx_client": httpx2.Client(transport=httpx2.MockTransport(handler))})


def _model(name, *actions):
    return {"name": name, "supportedGenerationMethods": list(actions)}


class TestListModelsOutput:
    PAGES = {
        None: {"models": [_model("models/gemini-2.5-flash", "generateContent"),
                          _model("models/embedding-001", "embedContent")],
               "nextPageToken": "page-2"},
        "page-2": {"models": [_model("models/gemini-3.5-flash-lite", "generateContent", "countTokens"),
                              _model("models/gemini-3-flash-preview", "generateContent")]},
    }

    def _run(self, monkeypatch, capsys, model):
        settings = gemini_settings(model=model)
        client = _catalog_client(self.PAGES)
        monkeypatch.setattr(smoke, "load_settings", lambda: settings)
        monkeypatch.setattr(smoke, "LLMService", lambda s: LLMService(s, client=client))
        code = smoke.list_models()
        return code, capsys.readouterr().out.splitlines()

    def test_every_generate_model_is_listed_first_with_count(self, monkeypatch, capsys):
        # Regression: the model IDs were printed without a heading or count, above the
        # summary and among HTTP log lines, so they were easy to miss entirely.
        code, lines = self._run(monkeypatch, capsys, "gemini-2.5-flash")
        assert code == 0
        assert lines[0] == "Models supporting generateContent for this key: 3 (API version v1)"
        assert lines[2:5] == ["* gemini-2.5-flash", "  gemini-3-flash-preview",
                              "  gemini-3.5-flash-lite"]
        assert not any("embedding-001" in line for line in lines)
        assert "Configured LLM_MODEL=gemini-2.5-flash: LISTED with generateContent" in lines

    def test_ids_are_bare_llm_model_values_from_all_pages(self, monkeypatch, capsys):
        _, lines = self._run(monkeypatch, capsys, "gemini-2.5-flash")
        ids = [line[2:] for line in lines[2:5]]
        assert ids == sorted(ids)
        assert set(ids) == {"gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite"}
        assert not any(i.startswith("models/") for i in ids)

    def test_unlisted_configured_model_still_shows_full_list(self, monkeypatch, capsys):
        code, lines = self._run(monkeypatch, capsys, "gemini-9-unknown")
        assert code == 1
        assert lines[0].startswith("Models supporting generateContent for this key: 3")
        assert not any(line.startswith("* ") for line in lines)
        assert "Configured LLM_MODEL=gemini-9-unknown: NOT in the list for this key" in lines

    def test_list_mode_suppresses_http_info_logs(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(smoke.logging, "basicConfig", lambda **kw: captured.update(kw))
        monkeypatch.setattr(smoke, "list_models", lambda: 0)
        monkeypatch.setattr(smoke.sys, "argv", ["llm_smoke_test.py", "--list-models"])
        assert smoke.main() == 0
        assert captured["level"] == smoke.logging.WARNING
