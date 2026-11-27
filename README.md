# -StudyMate-AI
SWE015 – Personalized AI Tutor

StudyMate turns a lecture PDF into a cited summary and a mixed practice quiz. Students can open the source passage for a question, see how their answer was graded, and generate a new quiz from their performance.

For the Role 6 contribution, backend interfaces, and verification, see the [Frontend & Integration handoff](ROLE6_INTERFACE.md).

## Run locally

Use Python 3.12 or later. From the repository root in PowerShell, create and activate an environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the template with `cp -n .env.example .env`.

In the local `.env` file, set `LLM_PROVIDER=openai` and supply `OPENAI_API_KEY`, or set `LLM_PROVIDER=gemini` and supply `GEMINI_API_KEY`. Replace the example value for the provider you choose. The file is git-ignored. The template also lists optional model, timeout, and retry settings. Restart the app after changing an active provider or key.

Launch the workspace with one command.

```sh
streamlit run app.py
```

Open a lecture and inspect its extracted text. **Study summary** generates a reading guide. **Practice quiz** offers five questions by default, including two short answers. PDF extraction works before a key is configured. Generation, short-answer grading, and explanations send selected passages and, where needed, answers to your configured provider.

## Run the tests

```sh
pytest
```

To run just the frontend integration suite, use `pytest tests/test_ui_integration.py`. The tests create real PDFs in memory, run the team's production modules, and drive Streamlit's file uploader, navigation, answer widgets, and dashboard. Provider replies are mocked and profile data goes to a temporary directory.

## How the pieces connect

| Stage | Implementation |
| --- | --- |
| Ingest and retrieve | `ui/lecture.py` uses PyMuPDF to extract selectable text into the shared `LectureChunk` schema. Every passage carries its original page and filename. Topics come from PDF bookmarks, page headings, or the document title. Long lectures are sampled across the document for summaries and across the planned topics for quizzes. The summary shows its passage coverage. |
| Summarize | `ui/study.py` calls `prepare_grounded_request` and `LLMService.generate_structured`. Every summary point must cite a passage in the actual prompt before the app displays it. |
| Generate a quiz | `PersonalizationEngine.plan_next_quiz` supplies the topic allocation and difficulty. Role 3's `generate_quiz` validates the questions using `validate_quiz`. `question_sources` resolves their citations. |
| Grade and explain | `evaluate_quiz` checks MCQs locally and grades short answers in one grounded call. `generate_feedback` adds missing explanations. Scores survive a feedback timeout, and a retry reuses the graded attempt. Questions with insufficient evidence are excluded from the score. |
| Adapt and track | `record_attempt` and `ProfileStore` save graded results. `dashboard_data` supplies the latest score, score history, topic accuracy, weak topics, and the next quiz plan. With the default rules, a score of at least 80% moves up one level and a score below 50% moves down one level, within the easy-to-hard range. |

The Role 6 branch builds on the team's `feature/role3-quiz-generation` commit `a3ff6e2`. The PDF ingestion adapter lives in `ui/lecture.py` because the repository had no Role 2 implementation at integration time.

**Study history.** Bookmark the URL to reopen your scores on the same server. Anyone with that private link can view its progress. Graded results are stored under the git-ignored `data/profiles/` directory. The Streamlit server keeps the uploaded lecture, generated materials, and in-progress answers in memory for that browser session. Re-upload the PDF after a fresh session.

## 60-second live demo

| Time | Show and explain |
| --- | --- |
| 0–15 s | Open a text-based lecture. Show its page count and extracted preview. Each passage retains its PDF page number. |
| 15–25 s | Open the study summary and expand a citation. Read the supporting sentence directly from the lecture. |
| 25–40 s | Answer an MCQ and a short question. For a timed talk, prepare a quiz on the last question beforehand. Submit it and open the evidence behind a correction. |
| 40–50 s | Open Progress. Point to the actual score and the reason for the next difficulty. Use **Generate Next Adaptive Quiz** to apply that plan. |
| 50–60 s | Under **Lecture → Change lecture**, open a scanned PDF without a text layer. Show the OCR recovery message and the previous lecture still available. |

The local header photograph in `ui/assets/study-desk.webp` was generated with the built-in ImageGen tool, using the prompt “Wide editorial photograph of an open cream notebook and a brass pencil on a deep pine-green study desk. Soft side light, tactile paper, subjects grouped on the right, dark empty left half for a lecture header. No writing, people, logos, or watermark.”
