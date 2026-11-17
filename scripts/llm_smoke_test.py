"""Manual end-to-end check against the REAL LLM API (OpenAI or Gemini).

NOT part of the automated test suite. Uses the provider configured in .env
(LLM_PROVIDER=openai|gemini).

    .venv\\Scripts\\python.exe scripts\\llm_smoke_test.py
        Makes 2 generation requests. With OpenAI this costs a small amount;
        with the Gemini Free Tier it uses 2 of your daily requests (RPD).

    .venv\\Scripts\\python.exe scripts\\llm_smoke_test.py --list-models
        Gemini only. Lists the models the API catalog shows to your key with
        generateContent in their supported actions, and whether the configured
        LLM_MODEL is among them. Does not call generateContent, so it cannot
        prove that generation will succeed -- only the smoke test can.

If a request fails, the provider's own status and message are printed (with
anything that looks like an API key redacted), because our exception messages
deliberately omit provider text.

It prints what actually happened, so the output can be used as evidence of a
working integration (or of the exact error if it fails).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigurationError, load_settings  # noqa: E402
from src.llm_service import LLMService, LLMServiceError  # noqa: E402
from src.prompts import build_tutor_question_request  # noqa: E402
from src.schemas import LectureChunk  # noqa: E402


_GOOGLE_KEY = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_MAX_PROVIDER_TEXT = 500


def redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Remove the configured key(s) and anything shaped like a Google API key."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return _GOOGLE_KEY.sub("[REDACTED]", text)


def provider_diagnostics(exc: BaseException, secrets: tuple[str, ...] = ()) -> str | None:
    """Describe the provider error attached as ``exc.__cause__`` (redacted), if any."""
    cause = exc.__cause__
    if cause is None:
        return None
    code = getattr(cause, "code", None)
    status = getattr(cause, "status", None)
    message = getattr(cause, "message", None)
    if code is None and status is None and message is None:
        return None
    text = redact(str(message or ""), secrets).strip()
    if len(text) > _MAX_PROVIDER_TEXT:
        text = text[:_MAX_PROVIDER_TEXT] + "..."
    return f"Provider said: HTTP {code} {status or ''}".rstrip() + (f": {text}" if text else "")


def _api_version(service: LLMService) -> str:
    return getattr(service._backend, "api_version", None) or "unknown"


def list_models() -> int:
    settings = load_settings()
    if settings.provider != "gemini":
        print("--list-models is only implemented for LLM_PROVIDER=gemini.")
        return 2
    service = LLMService(settings)
    wanted = f"models/{settings.model}"
    names = []
    try:
        for model in service._client.models.list():
            if "generateContent" in (model.supported_actions or []):
                names.append(model.name)
    except Exception as exc:  # report the type only; provider text may echo details
        print(f"Listing models failed: {type(exc).__name__}")
        return 1
    print(f"Models supporting generateContent for this key: {len(names)} "
          f"(API version {_api_version(service)})")
    print("Each ID below is the exact value to use as LLM_MODEL "
          "(the API's name is 'models/<ID>'); * = currently configured.")
    for name in sorted(names):
        model_id = name.removeprefix("models/")
        print(("* " if name == wanted else "  ") + model_id)
    print(f"\nConfigured LLM_MODEL={settings.model}: "
          f"{'LISTED with generateContent' if wanted in names else 'NOT in the list for this key'}")
    print("Note: the listing is catalog metadata. It does not call generateContent, so it does "
          "not prove generation will succeed or show your rate limits; run the smoke test.")
    return 0 if wanted in names else 1


def smoke_test() -> int:
    chunks = [
        LectureChunk(
            chunk_id="c1",
            text="A stack is a linear data structure that follows Last-In-First-Out (LIFO) order. "
                 "push adds an element to the top and pop removes the top element.",
            source="sample_lecture.pdf",
            page=1,
        )
    ]
    service = LLMService()
    version = f"  API version: {_api_version(service)}" if service.settings.provider == "gemini" else ""
    print(f"Provider: {service.settings.provider}  Model: {service.settings.model}{version}")
    for question in ("What does pop do on a stack?", "What is a binary heap?"):
        try:
            response = service.generate(build_tutor_question_request(question, chunks))
        except LLMServiceError as exc:
            details = provider_diagnostics(exc, (service.settings.api_key.get_secret_value(),))
            if details:
                print(details)
            raise
        print(f"\nQ: {question}\nA: {response.content}")
        print(f"   model={response.model} latency={response.latency_ms:.0f}ms "
              f"attempts={response.attempts} tokens={response.usage.total_tokens} "
              f"finish={response.finish_reason}")
    print("\nSmoke test completed. Check that the second answer says the topic is not covered.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list-models", action="store_true",
                        help="Gemini: list models available to your key (no generation).")
    args = parser.parse_args()
    # In --list-models mode, per-request INFO logs would bury the model list.
    level = logging.WARNING if args.list_models else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return list_models() if args.list_models else smoke_test()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 2
    except LLMServiceError as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
