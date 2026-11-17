# StudyMate AI: integration contracts

How the roles' components connect: the exact functions, data formats and
errors at each boundary. Every interface is marked with its status:

| Mark | Meaning |
|---|---|
| ✅ **Implemented** | Exists in code and is covered by offline tests |
| 🟡 **Proposed** | Contract only. The owning role still has to build it; nothing in the code implements it yet |

Role 1 functions added in Steps 1–2 (`topic`/`lecture_id`, `lecture_topics`,
`chunks_for_topics`, `generate_structured_with_response`) are implemented on
the branch `feature/backend-integration` and are pending review. They are not
on `main` until that branch is merged.

The contracts below are checked end to end by
`tests/test_integration_contracts.py`. That file uses **mocked LLM replies**, a
**test-local** quiz schema and **simulated** grading: it pins the contracts,
but it is not a quiz generator or grader.

```
Role 2  PDF ──► list[LectureChunk]  (topic, lecture_id)
                    │ lecture_topics()                         ┌──────────────┐
                    ├──────────────► Role 5 plan_next_quiz() ──► QuizPlan     │
                    │ chunks_for_topics(plan.topic_allocation) └──────┬───────┘
                    ▼                                                 │
Role 3  prepare_grounded_request(..., "json") ─► generate_structured* ─► quiz questions
                                                                      │
Role 4  grade answers ─► result dicts ─► Role 5 record_attempt() ─► ProfileStore.save()
                                                                      │
Role 6  dashboard_data() ◄────────────────────────────────────────────┘
        tutor Q&A: prepare_grounded_request ─► generate ─► is_not_in_material / validate_citations
```

---

## 1. Role 2 (PDF processing) → Role 1

**✅ Data format: `LectureChunk`** (`src/schemas.py`, Pydantic, frozen)

| Field | Type | Rule |
|---|---|---|
| `chunk_id` | `str` | 1–100 characters. Use only `A–Z a–z 0–9 _ . : -` (e.g. `lec3-p4-c1`); other characters become `_` in the prompt and in citations. Must be unique within a lecture, and must not collide after that replacement (`"a b"` and `"a_b"` do). |
| `text` | `str` | Not blank |
| `source` | `str \| None` | e.g. the PDF file name |
| `page` | `int \| None` | ≥ 1 |
| `topic` | `str \| None` | Not blank if given. Matched case- and whitespace-insensitively (`"Trees"` = `" trees "`); `"Tree"` ≠ `"Trees"`. **Needed for personalization.** |
| `lecture_id` | `str \| None` | Not blank if given. Not shown in prompts. |

Invalid values raise a Pydantic `ValidationError` (a `ValueError`).

**🟡 Producer (Role 2 to build).** A function such as
`chunk_lecture(pdf_path, lecture_id) -> list[LectureChunk]`. The name and
signature are Role 2's choice; the output format above is the contract.
Requirements:
- Tag **every content chunk with a topic** (e.g. from slide or section
  headings), and keep topic names consistent within a lecture.
- Return chunks in lecture order. Chunks without a topic (title page, reading
  list) are allowed but are never selected for quizzes.

## 2. Role 1: LLM backend ✅

All model calls go through Role 1. No other role imports `openai` or
`google.genai`.

| Function (file) | Input → output |
|---|---|
| `lecture_topics(chunks)` (`src/prompts.py`) | → `list[str]`: unique topics, first-seen spelling, in lecture order; untagged chunks skipped |
| `chunks_for_topics(chunks, topics)` (`src/prompts.py`) | `topics` = one name, a list, or `plan.topic_allocation` → matching chunks, in their original order |
| `prepare_grounded_request(task, chunks, user_input, *, response_format="text"\|"json", temperature, max_output_tokens, max_context_chars=12000, on_overflow="drop"\|"error")` (`src/prompts.py`) | → `GroundedRequest(request: LLMRequest, context: LectureContext)` |
| `build_grounded_request(...)` (`src/prompts.py`) | the same arguments except `on_overflow` → `LLMRequest` |
| `get_llm_service()` (`src/llm_service.py`) | → the shared `LLMService` |
| `LLMService.generate(request)` | → `LLMResponse(content, model, finish_reason, usage, latency_ms, attempts)` |
| `LLMService.generate_structured(request, Model)` | → a validated instance of your Pydantic `Model` |
| `LLMService.generate_structured_with_response(request, model_class)` | → `(instance, LLMResponse)`; the response is the accepted generation's (for benchmarking) |
| `is_not_in_material(text)` (`src/grounding.py`) | → `bool`: is this the not-covered fallback (text or JSON form)? |
| `validate_citations(text, context)` (`src/grounding.py`) | → `CitationReport(cited, valid, unknown, has_citations, all_valid)` |

**Errors** (`src/llm_service.py`; all are `LLMServiceError`, unless noted):

| Error | When | Retried? |
|---|---|---|
| `ConfigurationError` (`src/config.py`; not an `LLMServiceError`) | Key missing or setting invalid | — |
| `ValueError` (not an `LLMServiceError`) | Building a request: no chunks, blank task or input, duplicate chunk IDs, `LectureContextOverflowError` | — |
| `LLMNotInMaterialError` (subclass of `LLMResponseError`) | Structured call: the material doesn't cover the request | No |
| `LLMResponseError` | Empty, filtered or refused output, or still invalid after the re-ask | 1 re-ask (invalid output only) |
| `LLMAuthenticationError`, `LLMRequestError` | Bad key or model, quota exhausted, bad request | No |
| `LLMTransientError` (`LLMTimeoutError`, `LLMConnectionError`, `LLMRateLimitError`, `LLMUnavailableError`) | After retries or the time budget (about 60 s) are used up | Yes, before raising |

**Thread safety.** One shared `LLMService` handled 16 simultaneous requests
without mixing up responses, token usage, per-request settings or retry
counts. That was checked offline with a mocked OpenAI client, and with the
real Gemini SDK client over an in-memory transport. It hasn't been checked
against the live API.

## 3. Role 5 (personalization and analytics) ✅

Implemented by Role 5 (`src/personalization.py`, `src/analytics.py`). The
full description is in `ROLE5_INTERFACE.md`. What the other roles rely on:

| Function | Input → output |
|---|---|
| `PersonalizationEngine().plan_next_quiz(profile, available_topics, num_questions=None)` | → `QuizPlan`: `difficulty` (`Difficulty` enum; `.value` is `"easy"`, `"medium"` or `"hard"`), `focus_topics`, `num_questions`, `topic_allocation: dict[str, int]`, `reason`; `.to_dict()` |
| `PersonalizationEngine().record_attempt(profile, results, difficulty, lecture_id=None)` | `results`: a list of dicts (see §5) → `QuizAttempt` |
| `ProfileStore(directory).load(student_id)` / `.save(profile)` | → `StudentProfile` / file path (JSON, `data/profiles/` is git-ignored) |
| `dashboard_data(profile, engine, available_topics)` (`src/analytics.py`) | → a JSON-serialisable dict: `student_id`, `total_attempts`, `overall_accuracy`, `score_history`, `difficulty_history`, `topics`, `weak_topics`, `strong_topics`, `improvement`, `weak_topic_gain`, `next_quiz` |

Behaviour the tests confirm, which other roles must allow for:
- `available_topics=lecture_topics(chunks)`. If it's empty, `topic_allocation`
  is `{}`, and there's nothing to select chunks with (see blocker B1).
- **Role 5 accepts any topic name.** An unknown topic (e.g. `"Graphs"`) silently
  starts a new topic in the statistics, so Role 3 must restrict question
  topics to the plan.
- A result **without a `topic` key raises `KeyError`**, not `ValueError`; a
  blank topic raises `ValueError`.

## 4. Role 3 (quiz generation) 🟡

**Proposed contract (Role 3 to build).** Role 1 provides the calls; Role 3
owns the schema, the prompt wording and the module.

1. **Input:** `plan = plan_next_quiz(...)` and the lecture's chunks.
2. Select the material with `chunks_for_topics(chunks, plan.topic_allocation)`.
3. Build **one** request per quiz:
   `prepare_grounded_request(task, quiz_chunks, json.dumps(plan.to_dict()), response_format="json", max_output_tokens=3000, on_overflow="error")`.
   The default limit of 1024 output tokens is too small for a full quiz.
4. Call `generate_structured(request, QuizModel)`, or
   `generate_structured_with_response` if you need metrics.
5. Handle `LLMNotInMaterialError` (e.g. tell the student, or drop the topic and
   re-plan) and `LectureContextOverflowError` (pass fewer chunks).

**Proposed question schema** (Pydantic; field names matter to Roles 4 and 5):

| Field | Type | Rule |
|---|---|---|
| `question_id` | `str` | Unique within the quiz |
| `topic` | `Literal[...]` of the plan's topics | **Exactly** a key of `plan.topic_allocation`; anything else must fail validation, which triggers the automatic re-ask |
| `difficulty` | `"easy" \| "medium" \| "hard"` | = `plan.difficulty.value` |
| `question_type` | `"mcq"` (later perhaps `"short"`) | |
| `question`, `options`, `correct_answer` | `str`, `list[str]`, `str` | |
| `cited_chunks` | `list[str]` | Must be a subset of `prepared.context.prompt_ids`; check with `validate_citations` |

A working example of this schema, built per plan with `pydantic.create_model`,
is `make_quiz_schema` in `tests/test_integration_contracts.py`. It's test code,
not a production generator. After validation, Role 3 should also check that
the number of questions per topic equals `plan.topic_allocation`. Pydantic
can't express that rule; the contract test shows the check.

## 5. Role 4 (answer evaluation) 🟡 → Role 5 ✅

**Proposed contract (Role 4 to build).**
- **Multiple choice:** grade in code (compare with `correct_answer`), with no
  model call.
- **Short answers** (if used): **one** `generate_structured` call per quiz
  covering all answers, with `score: float = Field(ge=0, le=1)` in the schema.
  Grading per answer would use up Free Tier quota quickly.
- **Output to Role 5 (✅ format defined by Role 5):** one dict per question:

```python
{"question_id": "q1", "topic": "Stacks", "difficulty": "medium",
 "score": 1.0, "question_type": "mcq"}
# score 0..1 (partial credit OK) — or "correct": True/False instead of score
```

  `topic` and `difficulty` are copied from the question. Then call
  `engine.record_attempt(profile, results, plan.difficulty, lecture_id=...)`
  and `store.save(profile)`. If you use Pydantic models, convert them with
  `.model_dump()`.
- If quiz generation raised `LLMNotInMaterialError`, there's nothing to grade
  and nothing should be recorded.

## 6. Role 6 (frontend) 🟡 app, using ✅ APIs

The app (e.g. Streamlit) is still to be built by Role 6. Everything it calls
exists.
- **Startup:** call `get_llm_service()` once. A `ConfigurationError` means a
  setup problem (no key).
- **Tutor Q&A:** `prepared = prepare_grounded_request(task, chunks, question)`
  → `generate(prepared.request).content`. If `is_not_in_material(content)`,
  show a "not in your lecture" message. Otherwise
  `validate_citations(content, prepared.context)` gives links to chunk and
  page.
- **Quiz:** Role 3's function, then Role 4's, then Role 5's
  `record_attempt` and `save`.
- **Dashboard:** `dashboard_data(profile, engine, available_topics=lecture_topics(chunks))`.
- **Errors:** for `LLMTransientError` show "please try again"; for
  `LLMNotInMaterialError`, "not covered by this lecture". For any other
  `LLMServiceError`, show its message, which is safe to display. A
  `ValueError` means a bug in input data. Escape model output before
  rendering it as HTML.
- Calls are synchronous: typically 1–3 s, giving up after about 60 s. Show a
  spinner. Conversation history (multi-turn chat) is **not supported**
  (gap G5).

---

## Verified by `tests/test_integration_contracts.py`

| Contract | Test |
|---|---|
| Chunks → topics → plan → chunk selection → grounded JSON request → validated quiz → simulated grading → `record_attempt` → saved profile → `dashboard_data`, over two quizzes, with the second adapting to the weak topic | `TestQuizPipeline` |
| Question topics outside the plan (other names, wrong case, unknown) are re-asked or rejected; the wrong difficulty is rejected; Role 5 itself accepts unknown topics | `TestInvalidTopics` |
| A lecture without topics: empty allocation, no chunks, `ValueError`; Role 5's `KeyError` / `ValueError` for missing or blank topics | `TestMissingLectureTopics` |
| Not-covered quiz: `LLMNotInMaterialError`, one request, nothing recorded | `TestNotInMaterial` |
| 16 concurrent requests on one `LLMService` (OpenAI mock and the real Gemini SDK client): no mixed-up content, usage, settings or retry counts | `TestConcurrency` |

## Open blockers and decisions

| # | Item | Owner |
|---|---|---|
| B1 | Topics must be produced: Role 2 tags chunks with `topic`. Without topics, personalization has nothing to work with. | Role 2 |
| B2 | Quiz question schema and quiz module (§4) | Role 3 |
| B3 | Grading module and its output in Role 5's format (§5) | Role 4 |
| B4 | Selecting chunks for **tutor questions** (retrieval/ranking) still has no owner; `chunks_for_topics` only selects by topic. | Team decision |
| B5 | Not verified live: the full quiz loop, concurrency, `generate_structured_with_response` | Role 1, after B2 |
| G5 | Conversation history, if Role 6 wants chat | Role 1, on request |
