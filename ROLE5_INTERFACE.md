# Role 5 — Personalization & Analytics: interface

Files: `src/personalization.py`, `src/analytics.py`, `src/learner_simulation.py`, `tests/test_personalization.py`
No new dependencies (standard library only).

## Input from Role 4 (Answer Evaluation)
A list of graded results, one per question:

```python
{"question_id": "q1", "topic": "Sorting", "difficulty": "medium", "score": 1.0}
# score: 0..1 (partial credit OK). "correct": True/False is also accepted instead of score.
```
**Every question needs a `topic`** (Role 3 must tag each question with one).

## Output to Role 3 (Quiz Generation)
```python
plan = engine.plan_next_quiz(profile, available_topics=lecture_topics)
plan.to_dict()
# {"difficulty": "hard", "focus_topics": ["Trees"], "num_questions": 5,
#  "topic_allocation": {"Trees": 3, "Sorting": 1, "Stacks": 1},
#  "reason": "scored 80% on medium, moving up to hard; focus on weak topics: Trees"}
```

## Output to Role 6 (Frontend)
```python
from src.analytics import dashboard_data
data = dashboard_data(profile, engine, available_topics=lecture_topics)
# keys: score_history, difficulty_history, topics, weak_topics, strong_topics,
#       overall_accuracy, improvement, weak_topic_gain, next_quiz
```

## Typical flow (in app.py)
```python
from src.personalization import PersonalizationEngine, ProfileStore
engine, store = PersonalizationEngine(), ProfileStore("data/profiles")
profile = store.load(student_id)
plan = engine.plan_next_quiz(profile, available_topics=lecture_topics)
# ... Role 3 generates quiz from plan, Role 4 grades it ...
engine.record_attempt(profile, graded_results, plan.difficulty, lecture_id=lecture_id)
store.save(profile)
```

## Adaptation rules (defaults, all in `PersonalizationConfig`)
| Rule | Value |
|---|---|
| Last score ≥ 80% | one level harder |
| Last score < 50% | one level easier |
| Topic accuracy < 60% (last 3 quizzes, ≥ 2 questions) | weak |
| Topic accuracy ≥ 85% | strong |
| Next quiz | 60% of questions on up to 3 weak topics, rest review |
| No weak topics | introduce unseen topics, else mixed review |

## Gain measurement for Role 7 / report section 6
`python -m src.learner_simulation` compares adaptive vs random quizzes on 200
**simulated** learners (6 quizzes × 5 questions). Label these as simulated in the report.
Real-user gain: `analytics.weak_topic_gain(profile)` on actual team runs.

Profiles are saved to `data/profiles/` (git-ignored). Filenames include a hash of the exact student ID, so similar IDs never collide.
