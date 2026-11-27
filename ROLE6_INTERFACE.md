# Role 6 — Frontend & Integration

**Contributor** · Moha-Bishly

Role 6 connects the team's backend to four student-facing screens. Students can upload a lecture, inspect a cited summary, complete a quiz, and see how their results change the next quiz.

## Integration points

| Stage | What the frontend passes and receives |
| --- | --- |
| Lecture upload | `ui.lecture.read_lecture(filename, data)` accepts PDF bytes and returns a `Lecture`. Its `LectureChunk` entries carry the filename, page, topic, and passage ID. The PDF adapter supplies these inputs until a Role 2 implementation is available. |
| Summary | `ui.study.summarize(lecture)` uses Role 1's `prepare_grounded_request` and `generate_structured`. Each paragraph links to its source passages after `validate_citations` checks the IDs against the actual request. |
| Quiz generation | `ui.study.new_practice(lecture, profile)` passes Role 5's topic allocation and difficulty to Role 3's `generate_quiz`. The practice screen uses `question_sources` to display the original PDF passages. The generator and validator come from [Role 3's PR #8](https://github.com/linafourgi-tech/-StudyMate-AI/pull/8). |
| Grading and feedback | `Practice.grade()` submits answers keyed by question ID to Role 4's `evaluate_quiz`. `Practice.save()` records graded results once per quiz. `Practice.explain()` reuses short-answer feedback and requests missing explanations with `generate_feedback`. A student can retry a failed explanation after seeing their score. |
| Progress | The dashboard reads Role 5's `dashboard_data` for score history, topic accuracy, and weak topics. The next-quiz button applies the planned difficulty and topic allocation to a new set of questions. |

## Verification and demo

- Run `pytest tests/test_ui_integration.py` for the 11 Role 6 cases, or `pytest` for the full suite. The tests use real PDF bytes, production backend functions, temporary profiles, and mocked model replies.
- The Streamlit tests cover upload, navigation, answer retention, grading, retrying an explanation, reopening saved history, and generating the next quiz. A Chrome walkthrough also verified the quiz flow and rendered dashboard using mocked replies.
- To demonstrate a failure, open an image-only PDF under **Lecture → Change lecture**. The app explains how to recover with OCR and keeps the previous lecture available.
- Live generation uses the provider and API key configured in the local `.env` file. See the [setup instructions and timed walkthrough](README.md).

## AI assistance

OpenAI Codex assisted with the frontend code, integration tests, documentation, and browser verification. OpenAI ImageGen created the lecture header image.
