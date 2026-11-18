# StudyMate AI — Role 7 evaluation / benchmarking

This folder is intended to be **added to the existing repo**, not used to replace any teammate's files. Python **3.12**. No new dependencies beyond the project's pytest. Run commands from repo root.

## Files
- `src/evaluation.py`: deterministic template baseline, human rubrics, `audit_grades()` for Role 4 score correctness, latency/cost math, summary functions, and adapter for Role 1's `LLMResponse`.
- `evaluation/feedback_cases.json`: **6 reference grading and feedback cases** for Role 4; these are not observations of Role 4 yet.
- `evaluation/benchmark_cases.json`: **10 synthetic, hand-written test fixtures** (8 covered, 2 missing-information cases). These are NOT evaluation results or actual uploaded lecture PDFs. Replace/add real, permissible lecture samples and record source/consent if needed.
- `evaluation/run_benchmarks.py`: runnable offline baseline and optional live LLM experiment.
- `evaluation/quality_rubric.md`: clearly defined human question and feedback scoring.
- `evaluation/REVIEW_FORM.md`, `evaluation/REPORT_DRAFT_TEMPLATE.md`: templates for logging real human judgments and report data.
- `tests/test_benchmarks.py`: repeatable pytest checks.

## Run (offline, no Gemini key and no charges)

```sh
python -m pytest tests/test_benchmarks.py -v
python -m evaluation.run_benchmarks --mode baseline --output evaluation/baseline_results.json
```

Offline baseline result file contains **measured local execution time** and **no LLM tokens/cost**. The method creates questions from facts and distractors hand-entered in the dataset. It is NOT evidence that the system autonomously extracted facts or generated better questions.

## Optional real Gemini/OpenAI run (may incur cost)

Configure backend secrets **locally only** using existing `.env.example` / backend setup, do **not commit `.env`**. Test manually before triggering many paid calls.

```sh
python -m evaluation.run_benchmarks --mode live --output evaluation/live_results.json
```

This calls Role 1's existing `get_llm_service` using `LLMRequest` and collects actual `LLMResponse.latency_ms`, model and reported tokens. It prompts the LLM to create MCQs from the same lecture excerpts. This measures a **backend-level component** until Role 3's real quiz-generation pipeline is integrated. Responses still need a human review: structural validity is NOT proof of fact grounding.

Prices are intentionally unspecified. To estimate costs, supply verified provider/model prices (USD per **million** input/output tokens) from the date of the run:

Example syntax: add `--input-price <verified_rate> --output-price <verified_rate>` using the provider's current documented model-specific rates. **Do not guess prices.** Do not give prices at all if provider prices are unknown. Zero token counts in the backend's `TokenUsage` may mean not reported; the summary will keep unknowns as `null`.

## Human evaluation / comparison

1. Open `baseline_results.json` and `live_results.json` and inspect a matching set of cases. Use a real lecture-based dataset for final results; keep the same cases and document prompts/settings.
2. Score output MCQs 0, 1, or 2 per criterion in `quality_rubric.md`, writing scores, justifications and reviewer name in your own review sheet. For missing-information cases, evaluate whether the model correctly refused to answer: count as correct *only* if it explicitly returns not-covered status.
3. Use `summarize_runs` to aggregate only filled-in, measured values. Do not claim ungraded question quality is zero or 100%.
4. For Role 4 feedback, create a small set of student wrong/right answers and reference feedback, apply the feedback rubric, and log error cases.
5. Ask the team to measure preparation of the *same* quiz manually and using StudyMate, including manual checking/edits; enter recorded minutes into `time_saving()` and report conditions.
6. Keep and explain one real failure (e.g., incomplete PDF extraction, unsupported question, misleading distractor, API timeout). Record its actual cause and mitigation.
7. Role 5's learner simulation runs via `python -m src.learner_simulation`. Its metrics are **simulated**, not student-test results; label them accordingly in report/slide.

### Fair-comparison warning
The baseline uses manually curated fact triples and distractors, whereas the LLM generates them from source text. This baseline is deliberately simple and is **not a controlled replacement for a full-context LLM vs retrieval-based LLM study**. For project report's three alternatives, discuss template-based, full-context LLM, and retrieval-based LLM. When Role 2/3 are connected, benchmark full-context vs retrieval using identical documents, prompts/temperature and human rating, then add those real results separately.

## Integrating with the team

- Role 1: `src.schemas.LLMResponse` contains `model`, `latency_ms`, `usage`, `attempts`; `response_metrics()` is compatible.
- Role 5: `ROLE5_INTERFACE.md` defines score/weak-topic semantics and has simulation info. Do not overwrite its code.
- Role 3: replace the direct backend prompt in the optional runner with the real quiz-generator call once its API is finalized.
- Role 4: send graded answers and feedback to this evaluation layer for quality review.
- Role 6: may display summary reports, but evaluation itself can run without Streamlit.

Results files are intended as local artifacts and should be reviewed before committing. Never commit API credentials, private students' uploaded materials, or `.env`.

## Report text outline
- **Method:** 10 initial synthetic unit fixtures, followed by real representative lecture-PDF cases. Include supported and unsupported questions; compare against a fixed template baseline, then relevant LLM alternatives.
- **Metrics:** question rubric (0-8), feedback rubric (0-6), MCQ validity/accuracy where a gold answer exists, latency (ms), provider token usage, estimated USD only with verified rates, preparation time (minutes).
- **Results:** only values from experiment JSON and signed review sheet. Keep unknown values `N/A`.
- **Limitations:** limited case mix, human ratings subjective, API variability, provider rates change, template authoring is manual, backend-only does not equal fully-integrated pipeline.
