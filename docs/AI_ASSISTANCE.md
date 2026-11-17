# AI-Assistance Disclosure: Role 1 (LLM Integration & Backend)

**Tool used:** Claude Code (Anthropic), model Claude Opus 5.5, used as a coding assistant.
**Date:** 2026-11-16 to 2026-11-17
**Branch:** `feature/llm-backend`

## What the AI assistant did
1. **Initial implementation.** Inspected the repository (which had empty
   placeholder files), proposed the architecture and interfaces, and drafted
   `src/config.py`, `src/schemas.py`, `src/prompts.py`, `src/llm_service.py`,
   `tests/test_llm_service.py`, `scripts/llm_smoke_test.py`, `requirements.txt`,
   `pyproject.toml`, `.env.example`, the `.gitignore` additions, and this
   documentation. Ran the offline tests (51 passed).
2. **Read-only audit, at the student's request.** Re-checked every file, re-ran
   the tests, and exercised the code through the real OpenAI SDK with an
   in-memory HTTP transport. Reported three defects:
   - B1: lecture text could spoof the prompt delimiters;
   - B2: `.env` discovery depended on the working directory or debugger;
   - B3: the provider's auth error (with a masked key fragment) stayed attached
     to our exception's traceback.
   It also reported risks: model-parameter compatibility, and an unbounded
   worst-case duration (about 180 s).
3. **Fixes, at the student's request.**
   - B1: untrusted text is sanitised and metadata is constrained.
   - B2: `.env` is read from an explicit project-root path, without modifying `os.environ`.
   - B3: auth errors are raised without the provider exception; `status_code` and `request_id` are kept.
   - Added validation of the retry settings, a total time budget per call, and refusal detection.
   - Added 45 tests: regression tests for B1–B3 and tests for the gaps the audit found.
   - Confirmed by mutation testing that each fix is guarded by a failing test.
   - Updated the docs. The shared files (`.gitignore`, `requirements.txt`,
     `pyproject.toml`, `README.md`) were not changed in this step.

4. **Gemini provider, at the student's request.**
   - Installed `google-genai` 3.3.0 into the local, git-ignored `.venv`, and
     checked the SDK's source for its retry, timeout, key-header and error behaviour.
   - Checked Google's model, deprecation, pricing, rate-limit and thinking docs
     (2026-11-16). It reported that `gemini-2.5-flash` is deprecated (routed to
     `gemini-3.5-flash-lite` from 2026-12-01), and that Free Tier availability of
     `gemini-3.5-flash-lite` could not be verified for the student's project.
   - The student chose the defaults: first `gemini-3.5-flash-lite`, then, after
     asking for verification, the `gemini-2.5-flash` fallback they had confirmed
     in their own Free Tier limits.
   - Refactored the OpenAI-specific code into a backend adapter, keeping the
     public interfaces and all 96 earlier tests unchanged and passing. Added
     `src/gemini_backend.py`, provider selection in `src/config.py`, 60 Gemini
     tests (real SDK client over an in-memory transport, no network), and a
     `--list-models` option in the smoke script.
   - Mutation-tested the key Gemini behaviours. Updated `.env.example` and the docs.
   - With the student's approval (2026-11-17), added the single line
     `google-genai>=3.3,<4` to `requirements.txt` after checking that the file
     contained no teammates' work and that the version constraints were compatible.

5. **Live-test support and Gemini 404 investigation, at the student's request (2026-11-17).**
   - Filled the student's local, git-ignored `.env` from `.env.example`
     (`LLM_PROVIDER=gemini`, `LLM_MAX_RETRIES=1`). The file was empty, and the
     AI left `GEMINI_API_KEY` blank for the student to enter. The AI never read
     or displayed a key.
   - After the student's live run of `gemini-2.5-flash` returned HTTP 404 even
     though `--list-models` showed it, the AI investigated offline. With a fake
     key and the network call stubbed, it captured the exact SDK requests and
     found the model ID, endpoint and body correct, with the same client and key
     used for listing and generation. It also found that google-genai 3.x
     defaults to API version `v1`.
   - Made the 404 error report the model, API version and Google's status code,
     still without Google's free text. Made the smoke script print Google's
     original message with keys redacted, and corrected the `--list-models`
     wording, which had overstated what a listing proves.
   - Read Google's public deprecations page and forum reports (no API calls).
     The likely cause is Google's documented restriction of the 2.5 models to
     past users. This was not confirmed from Google's error text.
   - Made the `--list-models` output show a heading with the model count and
     bare `LLM_MODEL`-ready IDs, and turned off HTTP log lines in that mode.
     14 regression tests were added across these changes.
   - After the student's successful live test, changed the Gemini default model
     from `gemini-2.5-flash` to `gemini-3.5-flash-lite` in `src/config.py`,
     `.env.example`, the affected tests and the docs. OpenAI's default and
     default provider (`gpt-4o-mini`, `openai`) were not changed.

## What the student did
- Defined the requirements, scope and constraints (role boundaries, security, Git rules).
- Requested the audit and decided which findings to fix.
- Makes every commit and push personally. The AI is not a commit author or co-author.
- Entered their own Gemini API key in `.env` and ran every live API call
  personally: `--list-models`, the failing `gemini-2.5-flash` runs (API
  versions `v1` and `v1beta`), and the successful `gemini-3.5-flash-lite` smoke test.
- Chose `gemini-3.5-flash-lite` and asked for it to become the Gemini default.
- _To complete by the student:_ confirm you have reviewed and can explain the
  code, and note any changes made afterwards.

## Verification
- Automated tests: **170 passed** (offline: mocks plus the real SDKs over
  in-memory transports), re-run on 2026-11-17 with outbound socket connections
  blocked. Python 3.12.10, pytest 9.2.0, openai 3.51.0, google-genai 3.3.0.
  `pip check`: no broken requirements.
- Live Gemini API: **passed** on 2026-11-17. The student ran it and reported
  the results, which the AI recorded here; the AI did not run it.
  `scripts/llm_smoke_test.py` with provider `gemini`, model
  `gemini-3.5-flash-lite`, API version `v1`, on the student's Free Tier project:

  | Request | HTTP | Latency | Tokens | Attempts | Result |
  |---|---|---|---|---|---|
  | "What does pop do on a stack?" | 200 OK | 2553 ms | 410 | 1 | Answered from the lecture, citing `[c1]` |
  | "What is a binary heap?" (not in the material) | 200 OK | 1015 ms | 281 | 1 | "This is not covered in the provided lecture material." |

  Covered by this run: text generation only. Not yet verified live: JSON mode
  (`generate_structured`), `GEMINI_THINKING_LEVEL`, and live error responses.
- Earlier live attempt: `gemini-2.5-flash` returned HTTP 404 NOT_FOUND on API
  versions `v1` and `v1beta` although `models.list()` showed it; see
  `docs/LLM_BACKEND.md`.
- Live OpenAI API: **not yet performed**.
- The AI made no LLM API calls of any kind (paid or Free Tier). Its only network
  use was reading public Google documentation and forum pages.
