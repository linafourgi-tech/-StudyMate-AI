# Role 4 — Answer Evaluation & Feedback

Files: `src/answer_evaluator.py`, `src/feedback_generator.py`.
Dependencies: existing Pydantic and shared Role 1 LLM service; no new package.

## Grade a complete quiz

```python
from src.answer_evaluator import evaluate_quiz

outcome = evaluate_quiz(quiz.questions, student_answers, lecture_chunks)
results = outcome["results"]  # only these are sent to Role 5
ungraded_ids = outcome["ungraded"]  # display separately; do not record as wrong
```

`student_answers` maps question IDs to the student's selected option or text.
`evaluate_quiz` grades MCQs locally first, then grades answered short questions
with one structured LLM request. It returns a dictionary with `results` and
`ungraded` question IDs. Pass only `results` to Role 5; do not record ungraded
questions as wrong. If `results` is empty, skip `record_attempt` for that quiz. Results are in quiz order and use Role 5's fields
(`question_id`, `topic`, `difficulty`, `score`, `question_type`) with feedback
fields added.

## Multiple-choice questions

```python
from src.answer_evaluator import grade_mcq_quiz

results = grade_mcq_quiz(quiz.questions, {"q1": "A", "q2": "choice text"})
```

The answer and answer key can use an option letter (`A`, `a.`, `(a)`) or the
full option text. Exact full-text options are matched before interpreting a
leading letter as a label, preserving names such as “E. coli”. Matching ignores
letter case and surrounding whitespace.
Missing answers score 0. Duplicate question IDs are rejected. Questions need
`question_id`, `topic`, `difficulty`, `options`, and `correct_answer`.

## Short-answer questions

```python
from src.answer_evaluator import evaluate_short_answers

outcome = evaluate_short_answers(short_questions, student_answers, lecture_chunks)
results = outcome["results"]
ungraded_ids = outcome["ungraded"]
```

Blank answers score 0 locally and are omitted from the model request. If every
answer is blank, no API call is made. Question `cited_chunks` are preferred first.
Otherwise matching topic chunks are used; if none match (including untagged chunks or topic spelling
differences), all supplied chunks are used as a fallback. The model receives
this scoring guide: 1.0 correct and complete, 0.5 partially correct, 0.0
wrong or unsupported. The output uses
Role 5's required fields and includes feedback and original chunk IDs in
`cited_chunks`. IDs that cannot be grounded are returned separately in
`ungraded`; do not send them to Role 5. For partial or wrong scores, valid
evidence in either the feedback text or `cited_chunks` is accepted. Unknown
citations are rejected. If Role 1 raises `LLMNotInMaterialError`,
answered short questions are placed in `ungraded`, not counted as wrong.

Pass an injected `LLMService` as `service=` for offline tests or custom
integration. The output token budget grows with the number of answered
questions. Context overflow raises the shared `LectureContextOverflowError`.

## Standalone feedback

```python
from src.feedback_generator import generate_feedback

outcome = evaluate_quiz(quiz.questions, student_answers, lecture_chunks)
results_with_feedback = generate_feedback(
    quiz.questions, student_answers, outcome["results"], lecture_chunks,
    ungraded=outcome["ungraded"],
)
```

Feedback generation includes display-only feedback entries for ungraded IDs;
those entries have no score and must not be sent to Role 5. Feedback generation prefers question-cited chunks, then topic-matched chunks,
and falls back to all supplied chunks when no topics match. It increases the
output token budget with quiz size, validates IDs from feedback text and the
structured citation list, and returns original chunk IDs. A valid citation in
either location is accepted; unknown IDs are rejected. Blank answers receive
`feedback="No answer was provided."` and `cited_chunks=[]`, so UI rows have a
consistent shape. Existing not-covered results are returned without another
model call.

## Verification and known failure cases

The offline Role 4 tests mock the model. They cover correct/wrong/blank MCQ
answers, partial short-answer scores, invalid IDs or citations, prompt-injection
text, and uncovered material. The prompt treats student answers as untrusted;
this reduces injection risk but does not guarantee that a model will always
follow the instruction. An unsupported question, invalid response structure,
missing evidence, or oversized lecture context can still prevent grading.
