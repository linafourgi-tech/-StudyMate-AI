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

6. **Shared configuration review and edits, at the student's request (2026-11-17).**
   - Reviewed `.gitignore`, `requirements.txt` and `pyproject.toml` read-only,
     then, as the student asked, applied the proposed changes on a separate
     branch for team review:
     - declared `httpx2>=2.12,<3`, which `src/gemini_backend.py` imports directly;
     - labelled the shared test dependencies and marked where other roles add theirs;
     - set pytest to `--import-mode=importlib`;
     - added credential-file and OS-file ignore patterns.
   - Excluded `gemini_models.txt` locally through `.git/info/exclude`.
   - Teammates' components and `README.md` were not changed.

7. **Integration review and hardening (G1, G2, G4), at the student's request (2026-11-17).**
   - Did a read-only integration-readiness review, wrote an integration guide,
     and listed gaps G1–G5. Each gap was reproduced offline before being reported.
   - **G1:** JSON-mode grounding rules whose not-covered fallback is a JSON
     object. `generate_structured` raises the new `LLMNotInMaterialError` (a
     subclass of `LLMResponseError`) without re-asking, or returns the fallback
     as data when the schema has a `not_in_material` field.
   - **G2:** `prepare_grounded_request` and `build_lecture_context` report which
     chunks were included, omitted or truncated. They can raise on overflow,
     and reject chunk IDs that become identical after sanitising. Dropping is
     now logged. `extract_citations` and `validate_citations` were added in the
     new `src/grounding.py`.
   - **G4:** `is_not_in_material(text)`.
   - Existing functions keep their signatures. Text-mode prompts were checked
     to be byte-for-byte identical to `main`; JSON-mode prompts differ only by
     the new rule block.
   - Added 65 tests in `tests/test_grounding.py`. Mutation-checked that the key
     tests fail when the G1 detection or JSON rules are disabled, or when
     detection becomes a naive substring check.
   - G3 (metrics from `generate_structured`) and G5 (conversation history)
     were not implemented.

8. **Topic and lecture integration (Step 1), at the student's request (2026-11-17).**
   - Did a read-only integration review of `main` after Role 5 was merged.
     It found that Role 5 needs topics and that no role produced them.
   - Added optional `topic` and `lecture_id` to `LectureChunk` (blank values
     rejected), plus `lecture_topics` and `chunks_for_topics` in
     `src/prompts.py`. They use the same topic-matching rule as Role 5.
   - The prompt header shows `topic=` only when a chunk has a topic. Prompts
     for chunks without topics were checked to be byte-for-byte identical to
     `main`.
   - Added 38 tests in `tests/test_topics.py`. They include a contract check
     through Role 5's public API; Role 5 code was not modified.

9. **Structured output metrics (Step 2, gap G3), at the student's request (2026-11-17).**
   - Added `LLMService.generate_structured_with_response(request, model_class)`,
     which returns `(validated instance, LLMResponse)`. The response is the one
     for the accepted generation, not an earlier rejected re-ask.
   - `generate_structured` and the new method now share one internal loop, so
     prompt, validation, re-asks, retries and exceptions are identical;
     `generate_structured` still returns only the model. Metadata is passed
     through unchanged, using the existing conventions (unreported token
     counts are 0; a missing model name falls back to `LLM_MODEL`). No new
     values are estimated.
   - Added 20 tests in `tests/test_structured_metrics.py`.

10. **Integration contracts, documentation and offline tests (Step 3), at the student's request (2026-11-17).**
    - Wrote `docs/INTEGRATION.md`. It covers the interfaces between Roles 1–6
      and marks each as **implemented** (Role 1 and Role 5 code) or
      **proposed** (Role 2 chunk producer, Role 3 quiz module and schema,
      Role 4 grading, Role 6 app), which teammates still have to build.
    - Added 18 tests in `tests/test_integration_contracts.py`:
      - The full quiz loop over two quizzes, through Role 1's and Role 5's
        public APIs.
      - Invalid topics, lectures without topics, and the not-covered case.
      - Concurrency: 16 simultaneous requests on one `LLMService`, with a
        thread-safe mocked OpenAI client and with the real Gemini SDK client
        over an in-memory transport.
    - The LLM replies are **mocked**, the quiz schema is **test-local**, and
      grading is **simulated**. No production quiz generator or grader was
      written, and Role 5 code was not modified.
    - Before writing the tests, Role 5's actual behaviour was checked. That
      surfaced two contract details now documented: a result without `topic`
      raises `KeyError`, and unknown topics are accepted silently.

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
- Automated tests: **392 passed** on the full project (316 on `main` after the Role 5 merge,
  plus 38 for Step 1 topics, 20 for Step 2 metrics and 18 for Step 3 contracts). Earlier: 170, then 235 with G1/G2/G4 (offline: mocks plus the real SDKs over
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

  Covered by this run: text generation only.
- Live Gemini structured output (G1): **passed** on 2026-11-17. At the
  student's request, **the AI ran this check itself**, using the student's
  local `.env` (the key was never printed). It used a temporary, untracked
  script outside the repository, with retries and re-asks turned off for that
  run only, and made **exactly 2 requests** to `gemini-3.5-flash-lite` (API `v1`):

  | Case | Reply | Result |
  |---|---|---|
  | Covered: "What does pop do on a stack?" | Valid JSON `{"answer": "Pop removes the top element from a stack [c1].", "cited_chunks": ["c1"]}` (454 tokens) | Validated `Answer` model; citation `[c1]` valid |
  | Uncovered: "What is a binary heap?" | Exactly `NOT_IN_MATERIAL_JSON` (437 tokens) | `LLMNotInMaterialError` raised, no re-ask |

  Still not verified live: the opt-in schema path (fallback returned as data),
  `GEMINI_THINKING_LEVEL`, and live error responses.
- Earlier live attempt: `gemini-2.5-flash` returned HTTP 404 NOT_FOUND on API
  versions `v1` and `v1beta` although `models.list()` showed it; see
  `docs/LLM_BACKEND.md`.
- Live OpenAI API: **not yet performed**.
- LLM API calls by the AI: only the **2** Gemini Free Tier requests of the G1
  structured-output check above, made at the student's request. All other live
  calls were made by the student. Apart from that check, the AI's only network
  use was reading public Google documentation and forum pages.
