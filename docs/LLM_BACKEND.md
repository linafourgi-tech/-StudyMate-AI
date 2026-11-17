# LLM Integration & Backend (Role 1)

This component is the single gateway between StudyMate AI and the LLM provider.
Other components never call a provider SDK directly. They build an `LLMRequest`
and pass it to `LLMService`.

Two providers are supported, selected with `LLM_PROVIDER` in `.env`:

| `LLM_PROVIDER` | SDK | Key variable | Default model |
|---|---|---|---|
| `openai` (default) | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `gemini` | `google-genai` (official Google Gen AI SDK) | `GEMINI_API_KEY` | `gemini-3.5-flash-lite` (verified live) |

Teammates' code is identical for both providers: same functions, same
`LLMResponse`, same exceptions, same finish reasons.

## Architecture

```
quiz / evaluation / personalization / frontend
        │  build_grounded_request(...) -> LLMRequest
        │  prepare_grounded_request(...) -> GroundedRequest (request + LectureContext)
        ▼
src/prompts.py      grounding rules (text + JSON), lecture-context building, input sanitising (no API calls)
src/grounding.py    answer helpers: is_not_in_material, extract_citations, validate_citations (no API calls)
src/schemas.py      Pydantic contracts: LectureChunk, LLMRequest, LLMResponse, TokenUsage,
                    LectureContext, GroundedRequest, CitationReport
src/llm_service.py  LLMService: time budget, retries, validation, structured output
                    + _OpenAIBackend (OpenAI call, error mapping, response parsing)
src/gemini_backend.py GeminiBackend (Gemini call, error mapping, response parsing)
src/config.py       Settings from environment / <project root>/.env (API key as SecretStr)
        ▼
OpenAI SDK  or  Google Gen AI SDK  (injected client, mocked in tests)
```

Each backend implements two operations: `complete(request, timeout)`, which
returns a normalised result, and `map_error(exc)`, which returns a StudyMate
exception plus whether to retry. Everything else (retries, time budget,
validation, structured output, logging) is shared, so both providers behave
the same.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # includes google-genai>=3.3,<4 for Gemini
copy .env.example .env    # in the project root; then set the provider and its key
```

The OpenAI path does not import `google-genai`. Selecting Gemini without it
installed raises a clear `ConfigurationError`.

`.env` is git-ignored. Never commit it. It is always read from the project
root (next to `src/`), whatever the current working directory.

## Running

```powershell
# Automated tests (offline, no API key needed, no paid calls)
.venv\Scripts\python.exe -m pytest -v

# Manual end-to-end check against the real API of the configured provider
#   OpenAI: small paid calls. Gemini Free Tier: uses 2 of your daily requests.
.venv\Scripts\python.exe scripts\llm_smoke_test.py

# Gemini: list every model ID the catalog shows with generateContent (no generation).
# A listed model is NOT proof that your key can generate with it; only the smoke test is.
.venv\Scripts\python.exe scripts\llm_smoke_test.py --list-models
```

## Integration interface for teammates

```python
from src.llm_service import get_llm_service, LLMServiceError, LLMTransientError
from src.prompts import build_grounded_request
from src.schemas import LectureChunk

chunks = [LectureChunk(chunk_id="lec3-p4-c1", text="...", source="lecture3.pdf", page=4)]

# Free-text answer
request = build_grounded_request(
    task_instructions="Explain the concept to the student.",
    chunks=chunks,
    user_input="What is a stack?",
)
response = get_llm_service().generate(request)   # -> LLMResponse
print(response.content, response.usage.total_tokens, response.latency_ms)

# Structured (validated JSON) output: define your own Pydantic model
from pydantic import BaseModel

class MyOutput(BaseModel):
    answer: str
    cited_chunks: list[str]

request = build_grounded_request("...", chunks, "...", response_format="json")
result = get_llm_service().generate_structured(request, MyOutput)   # -> MyOutput
```

### Not-covered answers, included chunks and citations

```python
from src.grounding import is_not_in_material, validate_citations
from src.llm_service import get_llm_service, LLMNotInMaterialError
from src.prompts import prepare_grounded_request

# Text mode: detect the fallback in the frontend (don't compare strings yourself)
prepared = prepare_grounded_request("Explain the concept.", chunks, "What is a heap?")
answer = get_llm_service().generate(prepared.request).content
if is_not_in_material(answer):
    ...  # show "not in your lecture" UI
else:
    report = validate_citations(answer, prepared.context)
    report.valid      # original chunk_ids that were cited and actually sent
    report.unknown    # cited IDs that were NOT sent (omitted, typo or invented)
    report.all_valid  # at least one citation, and all of them were sent

prepared.context.included_ids   # chunks really in the prompt, in order
prepared.context.omitted_ids    # chunks dropped by the character budget
prepared.context.is_complete    # False if anything was dropped or truncated

# Structured mode: the not-covered fallback is a JSON object, not plain text
try:
    quiz = get_llm_service().generate_structured(
        prepare_grounded_request("Write 5 questions.", chunks, "heaps",
                                 response_format="json").request, Quiz)
except LLMNotInMaterialError:
    ...  # topic not in the lecture: no second request is made
```

- **JSON mode** (`response_format="json"`) uses `GROUNDING_RULES_JSON`. Their
  fallback is the JSON object `NOT_IN_MATERIAL_JSON`
  (`{"not_in_material": true, "message": "This is not covered in the provided lecture material."}`),
  so a not-covered reply is still valid JSON. Text mode is unchanged.
- `generate_structured` raises **`LLMNotInMaterialError`** (a subclass of
  `LLMResponseError`) when the model returns the fallback, either as JSON or
  as the plain phrase. It does not re-ask. To receive the fallback **as data**
  instead, add `not_in_material: bool = False` to your schema and give the
  other fields defaults; you then get an instance with `not_in_material=True`.
  A schema with other required fields still gets the exception.
- **`is_not_in_material(text)`** accepts the phrase with formatting
  differences (case, whitespace, quotes, Markdown, a trailing `[c1]`, a code
  fence) and the JSON object. It returns **False** when the phrase appears
  inside a longer answer, so partial answers are never hidden.
- **Context budget.** Chunks that don't fit `max_context_chars` (12,000 by
  default) are still dropped, but now with a logged warning (IDs only, never
  lecture text). They're also listed in `context.omitted_ids`. Pass
  `on_overflow="error"` to get `LectureContextOverflowError` (a `ValueError`)
  instead.
- **Chunk IDs** must stay unique after sanitising (`"a b"` and `"a_b"` would
  both become `[a_b]`). `prepare_grounded_request` and `build_lecture_context`
  raise `ValueError` for such duplicates. The older `build_grounded_request`
  and `format_lecture_context` keep their previous lenient behaviour.

| Interface | Purpose |
|---|---|
| `LectureChunk(chunk_id, text, source?, page?)` | Input format expected from PDF processing. Use IDs made of `A-Z a-z 0-9 _ . : -`; any other character is replaced with `_` in the prompt (and therefore in citations). |
| `LLMRequest(system_prompt, user_prompt, temperature?, max_output_tokens?, response_format)` | One LLM call |
| `LLMResponse(content, model, finish_reason, usage, latency_ms, attempts)` | Validated result plus metrics (for benchmarking) |
| `get_llm_service()` | Shared, environment-configured service instance (configuration errors are not cached) |
| `LLMService.generate(request)` | Returns `LLMResponse` |
| `LLMService.generate_text(system, user, **opts)` | Returns `str` |
| `LLMService.generate_structured(request, Model)` | Returns validated `Model` instance |
| `build_grounded_request(task, chunks, user_input, ...)` | Lecture-grounded `LLMRequest` |
| `prepare_grounded_request(task, chunks, user_input, ..., on_overflow="drop"\|"error")` | Same request, plus the `LectureContext` showing which chunks were sent (`GroundedRequest`) |
| `format_lecture_context(chunks, max_chars)` | Delimited, sanitised lecture block |
| `build_lecture_context(chunks, max_chars, on_overflow=...)` | The same block as a `LectureContext` (`text`, `included_ids`, `omitted_ids`, `truncated`, `is_complete`, `prompt_ids`) |
| `neutralize_delimiters(text)` | Sanitiser for any other untrusted text a teammate puts into a prompt |
| `build_tutor_question_request(question, chunks)` | Ready-made tutor Q&A request |
| `NOT_IN_MATERIAL` | Exact phrase the model uses when the answer isn't in the lecture |
| `NOT_IN_MATERIAL_JSON` | The JSON-mode equivalent: `{"not_in_material": true, "message": ...}` |
| `src.grounding.is_not_in_material(text)` | True if an answer is the not-covered fallback (text or JSON) |
| `src.grounding.extract_citations(text)` | `[id]` citations in an answer, unique, in order |
| `src.grounding.validate_citations(text, context)` | `CitationReport(cited, valid, unknown, has_citations, all_valid)` |

Task-specific prompts (quiz generation, answer evaluation, and so on) belong to
the components that own them. They pass their own `task_instructions` and reuse
the shared grounding rules. `task_instructions` is treated as **trusted**
(written by developers). Never put student or lecture text in it; pass that as
`user_input` or `chunks`, which are sanitised.

### Errors

```
LLMServiceError                      .status_code, .request_id (safe diagnostics)
├── LLMAuthenticationError   bad/missing key, 401/403          (not retried)
├── LLMRequestError          400/404/422, quota exhausted      (not retried)
├── LLMResponseError         empty, refused, filtered, truncated JSON, schema mismatch
│   └── LLMNotInMaterialError   structured output: material doesn't cover the request (not re-asked)
└── LLMTransientError        raised after retries or the time budget are used up
    ├── LLMTimeoutError
    ├── LLMConnectionError
    ├── LLMRateLimitError    429
    └── LLMUnavailableError  5xx / 408 / 409
ConfigurationError (src.config)  missing/invalid environment settings
ValueError                       invalid input when *building* a request: no chunks, blank text,
                                 duplicate chunk IDs, LectureContextOverflowError (on_overflow="error"),
                                 pydantic ValidationError (e.g. page=0). Not an LLMServiceError.
```

The frontend can catch `LLMTransientError` and show "please try again", and
catch `LLMServiceError` for everything else. Error messages are written by
this component and are safe to show to users; they never include provider
response text.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` (case-insensitive) |
| `OPENAI_API_KEY` | (required for openai) | OpenAI key |
| `GEMINI_API_KEY` | (required for gemini) | Google AI Studio key |
| `LLM_MODEL` | `gpt-4o-mini` / `gemini-3.5-flash-lite` | Model name (check it is available on your account) |
| `GEMINI_THINKING_LEVEL` | unset (model default) | Gemini only: `minimal`, `low`, `medium`, `high` |
| `LLM_BASE_URL` | unset | OpenAI-compatible endpoint |
| `LLM_TIMEOUT_SECONDS` | 30 | Timeout for one HTTP attempt |
| `LLM_TOTAL_TIMEOUT_SECONDS` | 60 | Budget for one call including all retries and waits |
| `LLM_MAX_RETRIES` | 3 | Retries for transient errors (0–10) |
| `LLM_RETRY_BASE_DELAY` | 1.0 | Backoff base (seconds) |
| `LLM_RETRY_MAX_DELAY` | 20.0 | Backoff cap (seconds); must be ≥ the base delay |
| `LLM_TEMPERATURE` | 0.2 | Low temperature favours factual answers |
| `LLM_MAX_OUTPUT_TOKENS` | 1024 | Output length limit |
| `LLM_VALIDATION_RETRIES` | 1 | Re-asks when structured output is invalid |

Priority: real environment variable > `.env` file > default. Loading `.env`
does not modify `os.environ`. Invalid values raise `ConfigurationError` naming
the field, never echoing the value.

## Technical decisions

1. **Retries are implemented in the service, and the SDK's own retries are off
   (`max_retries=0`).** This avoids retrying twice, makes the policy explicit and
   unit-testable, and gives one place to decide which errors are retryable.
2. **Capped exponential backoff with jitter** (`base·2^(n-1)`, capped, then scaled
   by a random factor in [0.5, 1]). A numeric `Retry-After` header takes priority
   (capped, negatives clamped to 0). Non-numeric values such as HTTP dates fall
   back to normal backoff.
3. **Only transient failures are retried.** Authentication errors, bad requests
   and exhausted quota fail immediately, because retrying cannot fix them.
4. **Total time budget.** Every call, including all retries, waits and (for
   `generate_structured`) validation re-asks, shares one
   `LLM_TOTAL_TIMEOUT_SECONDS` budget. Each attempt's timeout is shortened to
   the time remaining, and the service never sleeps if the wait would pass the
   deadline. With the defaults, a failing call normally gives up within about
   **60 s** (see the limitation on HTTP timeouts below). Without the budget,
   the worst case would be 4 × 30 s plus up to 3 × 20 s of waiting, about 180 s.
5. **Provider-independent exceptions.** Teammates depend on our error types, not
   on OpenAI's, so the provider could be swapped without changing their code.
6. **Response validation.** Content must be non-empty. Refusals and
   content-filtered responses are rejected with distinct messages. Truncated
   JSON is rejected. Structured output is validated with Pydantic and re-asked
   up to `LLM_VALIDATION_RETRIES` times.
7. **Lecture grounding.** The system prompt tells the model to answer only from
   the delimited lecture material, to cite chunk IDs, and to use a fixed phrase
   when the answer isn't covered. In JSON mode the fallback is a JSON object,
   so it stays valid structured output. A character budget bounds prompt size
   by dropping whole chunks. Dropping is logged and reported in
   `LectureContext`, and can be made an error.
8. **Dependency injection** of the client, `sleep`, clock and RNG makes all
   tests offline, deterministic and fast (no real waiting, no network).

## Gemini provider details

**Model choice.** The default is **`gemini-3.5-flash-lite`**, the model that
was verified against the live API (see *Live verification* below). It is the
model Google names as the replacement for the deprecated `gemini-2.5-flash`,
and Google lists it as not deprecated.

`gemini-2.5-flash`, the earlier default, was **not** usable on this project.
Google's deprecations page says it is deprecated, "no longer available to
select in Google AI Studio", that "access to the 2.5 models is limited to users
who have actively used them in the past", and that requests to it will route to
`gemini-3.5-flash-lite` from 1 Dec 2026 (earliest possible date). On the
student's Free Tier project it **was listed** by `models.list()` with
`generateContent`, yet generation returned **HTTP 404 NOT_FOUND** on both API
versions `v1` and `v1beta`. Listing and generation used the same SDK client and
key, and the request (model ID, endpoint, body) was checked to be well-formed.
The exact cause was not confirmed from Google's error text. The likely
explanation is the documented access restriction above.

Lessons: **`models.list()` is catalog metadata and does not prove generation
access**, and model availability is per account or project. To change models,
run `--list-models` to find a candidate ID, set `LLM_MODEL`, then run the smoke
test once.

**Live verification (2026-11-17, run by the student, Free Tier project).**
`scripts/llm_smoke_test.py` with `LLM_PROVIDER=gemini`,
`LLM_MODEL=gemini-3.5-flash-lite`, API version `v1`, `LLM_MAX_RETRIES=1`:

| Request | HTTP | Latency | Tokens | Attempts | Result |
|---|---|---|---|---|---|
| "What does pop do on a stack?" | 200 | 2553 ms | 410 | 1 | Answered from the lecture, citing `[c1]` |
| "What is a binary heap?" (not in the material) | 200 | 1015 ms | 281 | 1 | "This is not covered in the provided lecture material." |

This confirms the real SDK call path, authentication, the request mapping, the
response parsing, and grounding behaviour (citation and the not-covered phrase)
for this model.

**Live structured-output check (G1, 2026-11-17, 2 requests, retries and
re-asks off).** `generate_structured` with `gemini-3.5-flash-lite`, API `v1`:

| Case | Model reply | Result |
|---|---|---|
| Covered ("What does pop do on a stack?") | Valid JSON `{"answer": "Pop removes the top element from a stack [c1].", "cited_chunks": ["c1"]}` | Validated Pydantic model; `validate_citations`: `[c1]` valid |
| Uncovered ("What is a binary heap?") | Exactly `{"not_in_material": true, "message": "This is not covered in the provided lecture material."}` | `LLMNotInMaterialError`, no re-ask |

Still tested offline only: the opt-in schema path (fallback returned as
data), `GEMINI_THINKING_LEVEL`, and live error responses (429, 5xx). One
sample per case shows the mechanism works, not how often the model follows it.

**API version.** google-genai 3.x defaults the Gemini Developer API to `v1`
(2.x used `v1beta`). The `GOOGLE_GENAI_API_VERSION` environment variable
overrides it (for example `beta` → `v1beta`). The smoke script prints the
version in use.

**SDK behaviour (verified in google-genai 3.3.0 source):**
- Client: `genai.Client(api_key=..., vertexai=False, http_options=HttpOptions(timeout=<ms>, retry_options=None))`.
  `vertexai=False` is set explicitly, so an environment variable can't switch the client to Vertex AI.
- With `retry_options=None` the synchronous client makes exactly **one** attempt, so `LLMService` remains the only retry layer.
- `HttpOptions.timeout` is in **milliseconds**. The per-attempt timeout (from the time budget) is passed with every call.
- The API key is sent in the `x-goog-api-key` header, never in the URL.
- `str(APIError)` contains the full provider response body. Our messages never reuse it.

**Request mapping:** `system_prompt` → `system_instruction`; `user_prompt` →
`contents`; `temperature` and `max_output_tokens` as usual; JSON mode →
`response_mime_type="application/json"`; `GEMINI_THINKING_LEVEL` →
`ThinkingConfig(thinking_level=...)`, which is only sent when set.

**Response mapping (same shape as OpenAI):**

| Gemini | `LLMResponse` / error |
|---|---|
| `STOP` | `finish_reason="stop"` |
| `MAX_TOKENS` | `finish_reason="length"`; for JSON or empty text: `LLMResponseError` |
| `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, image-safety, `MODEL_ARMOR` | `LLMResponseError` (content filter) |
| prompt blocked (`promptFeedback.blockReason`) | `LLMResponseError` (content filter) |
| thought parts (`thought=True`) | excluded from `content` |
| `usageMetadata` | `prompt_tokens`; `completion_tokens` = output + thinking tokens; `total_tokens` |
| `modelVersion` | `model` |

**Error mapping:**

| Gemini error | StudyMate exception | Retried? |
|---|---|---|
| 400 with `ErrorInfo.reason=API_KEY_INVALID` (how Gemini reports a bad key), 401, 403 | `LLMAuthenticationError` (no provider exception attached) | No |
| 429 with a per-day `QuotaFailure` (e.g. Free Tier RPD) | `LLMRequestError` ("daily quota exhausted") | No |
| 429 with `RetryInfo.retryDelay` ≤ `LLM_RETRY_MAX_DELAY` | `LLMRateLimitError` | Yes, after exactly that delay |
| 429 with `retryDelay` > `LLM_RETRY_MAX_DELAY` | `LLMRateLimitError` | **No** (an early retry would be rejected again and waste quota) |
| 429 without `RetryInfo` | `LLMRateLimitError` | Yes, normal backoff |
| 404 | `LLMRequestError`: names the model, API version, Google's status and `ErrorInfo.reason` (enum values only, never Google's free text); Google's message stays on `__cause__` and the smoke script prints it with keys redacted | No |
| 5xx, 408, 409 | `LLMUnavailableError` | Yes |
| other 4xx | `LLMRequestError` | No |
| `httpx2` timeout / transport error | `LLMTimeoutError` / `LLMConnectionError` | Yes |
| unreadable body (non-JSON 200, `UnknownApiResponseError`) | `LLMResponseError` | No |

**Free Tier notes (5 RPM, 20 RPD).** Every retry is a real request and counts
against the daily limit. That's why daily-quota errors and long Google-requested
waits are not retried. Consider `LLM_MAX_RETRIES=1` while testing.

**Thinking tokens.** Google's docs state that `max_output_tokens` covers
thinking tokens *and* answer tokens. For a model that thinks by default, a long
internal reasoning step can truncate or empty the answer with
`LLM_MAX_OUTPUT_TOKENS=1024`. (Both live smoke-test answers with
`gemini-3.5-flash-lite` finished normally within 1024 tokens.) The service then raises an actionable `LLMResponseError`.
Fixes: raise `LLM_MAX_OUTPUT_TOKENS`, or set `GEMINI_THINKING_LEVEL=low`.
Neither has been tested against the live API.

**Structured output.** It uses the same mechanism as OpenAI: JSON mode, plus
the Pydantic JSON Schema in the system instruction, plus Pydantic validation
with one re-ask. Gemini's native `response_json_schema` enforcement is not
used, which keeps `LLMRequest` unchanged; the Pydantic validation is what
guarantees the result.

## Security

### API key protection: what is and is not covered
- The key is read only from the environment or the git-ignored `.env`, and is
  held as a `SecretStr`, so `repr`, `str` and `model_dump` show `**********`.
- The service never logs prompts, responses or the key. Logs contain the
  error type, HTTP status, attempt count, latency and token counts.
- `ConfigurationError` messages list field names and reasons, never values.
- **Authentication failures (401/403)**: the provider's error text can contain
  a masked fragment of the key (for example `sk-ab***wxyz`). For these errors
  `LLMAuthenticationError` is raised **without** the provider exception attached
  (`__cause__` and `__context__` are both `None`), so neither the message nor a
  printed traceback contains provider text. The HTTP status and provider
  `request_id` are kept as attributes for debugging.
- **Other errors** keep the provider exception as `__cause__` for debugging. Our
  own message never repeats provider text, but a full traceback of those errors
  does include the provider's message, so don't show raw tracebacks to end users.
- The OpenAI SDK redacts the `Authorization` header in its own debug logs
  (checked in the installed SDK source). The Gemini SDK sends the key in the
  `x-goog-api-key` header, not the URL.
- The same rule applies to Gemini authentication failures, including Gemini's
  "400 API_KEY_INVALID": raised with no provider exception attached.
- **Not covered:** anyone who can read the process environment, the `.env`
  file, or the machine's memory can obtain the key. That's normal for a local
  key; use a project-scoped key with a spending limit.

### Prompt injection: a mitigation, not a guarantee
Lecture text, student input and PDF metadata are untrusted. The component:
- puts them only in the **user** message, never in the system message;
- wraps the lecture material in `<<<LECTURE_MATERIAL>>>` … `<<<END_LECTURE_MATERIAL>>>`
  and **neutralises delimiter spoofing**: every run of three or more `<` in
  untrusted text is collapsed to `<<`. Both delimiters start with `<<<`, so
  untrusted text cannot reproduce them. Ordinary code such as `a << b` and
  Python `>>>` prompts is preserved;
- limits chunk IDs to `[A-Za-z0-9_.:-]` and flattens `source` to one bracket-free
  line (≤ 200 chars), so metadata can't add fake lines or fake chunk headers;
- instructs the model to treat the material as data and ignore instructions in it.

These measures keep the prompt structure intact. They **cannot fully prevent**
prompt injection: the model still reads the text, and a persuasive instruction
inside it (or a fake `[id]`-style citation written in the lecture text itself)
may still influence the answer. Downstream components should treat model
output as untrusted too (for example, escape it before rendering it as HTML).

## Limitations

- Automated tests use mocks and an in-memory HTTP transport. They prove the
  logic and the SDK integration path (for Gemini, through the real
  `google.genai.Client`), not the live API. Real integration is only
  confirmed for a provider once `scripts/llm_smoke_test.py` has succeeded with
  a valid key for that provider. **Gemini (`gemini-3.5-flash-lite`) was
  verified live on 2026-11-17** (text generation only; see *Live
  verification*). **OpenAI has not been verified live yet.**
- Gemini error-body formats (`RetryInfo`, `QuotaFailure`, `ErrorInfo`) in the
  tests follow Google's documented error model. They were not captured from
  the live API.
- Gemini model access differs per account and project, and Google changes it
  over time (for example, `gemini-2.5-flash` was listed but returned 404 here).
  Re-run the smoke test after changing `LLM_MODEL` or using a different key.
- The service sends `max_tokens` and `temperature`. These are supported by the
  intended default model (`gpt-4o-mini`), but some newer reasoning models
  require `max_completion_tokens` and reject a custom temperature. Re-check with
  the smoke test if `LLM_MODEL` is changed.
- Grounding is enforced by prompt instructions, which reduces hallucination but
  cannot guarantee it. Measuring it is the benchmarking role's job.
- The calls are synchronous only (no async or streaming).
- The time budget is **approximate, not a hard wall-clock limit.** It is
  checked between attempts, and each attempt's timeout is set from the
  remaining budget. However, the SDK/httpx timeout applies to each network
  operation (connect, each read), not to the whole request, so a server that
  keeps sending data slowly can make one attempt last longer than its timeout.
  A hard limit would need a separate thread or async cancellation, which was
  left out to keep the component simple.
- Context is limited by a character budget, not an exact token count.
- There is no response caching and no cost tracking beyond reported token usage.
