"""Simulated-learner experiment: adaptive quizzes vs. random quizzes.

Purpose: a reproducible, offline measurement for the Gain & Value section.
No LLM calls. Results are from SIMULATED students and must be labelled as
such in the report; real-user results should come from the team's own runs.

Model (deliberately simple, stated in the report as an assumption):
* each simulated student has a hidden mastery 0..1 per topic;
* P(correct) = mastery + difficulty offset (easy +0.15, medium 0, hard -0.15);
* each question practised on a topic raises its mastery by
  ``learning_rate * (1 - mastery)``.

Run:  python -m src.learner_simulation
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any

try:
    from .personalization import Difficulty, PersonalizationEngine, QuestionResult, StudentProfile
except ImportError:  # pragma: no cover
    from personalization import Difficulty, PersonalizationEngine, QuestionResult, StudentProfile

DIFFICULTY_OFFSET = {Difficulty.EASY: 0.15, Difficulty.MEDIUM: 0.0, Difficulty.HARD: -0.15}
DEFAULT_TOPICS = ["Arrays", "Linked Lists", "Stacks", "Queues", "Trees", "Sorting"]


@dataclass
class SimulatedLearner:
    mastery: dict[str, float]
    learning_rate: float = 0.08

    def answer(self, topic: str, difficulty: Difficulty, rng: random.Random) -> float:
        p = min(0.98, max(0.02, self.mastery[topic] + DIFFICULTY_OFFSET[difficulty]))
        correct = rng.random() < p
        self.mastery[topic] += self.learning_rate * (1 - self.mastery[topic])
        return 1.0 if correct else 0.0


def make_learner(topics: list[str], rng: random.Random, n_weak: int = 2) -> SimulatedLearner:
    weak = set(rng.sample(topics, k=min(n_weak, len(topics))))
    mastery = {t: rng.uniform(0.2, 0.4) if t in weak else rng.uniform(0.6, 0.85) for t in topics}
    return SimulatedLearner(mastery)


def _run_quiz(learner, profile, engine, topics, strategy, rng, n_questions, qid_start):
    if strategy == "adaptive":
        plan = engine.plan_next_quiz(profile, available_topics=topics, num_questions=n_questions)
        difficulty = plan.difficulty
        order = [t for t, c in plan.topic_allocation.items() for _ in range(c)]
    elif strategy == "random":
        difficulty = Difficulty.MEDIUM
        order = [rng.choice(topics) for _ in range(n_questions)]
    else:
        raise ValueError("strategy must be 'adaptive' or 'random'")
    results = [
        QuestionResult(f"q{qid_start + i}", t, difficulty, learner.answer(t, difficulty, rng))
        for i, t in enumerate(order)
    ]
    engine.record_attempt(profile, results, difficulty)


def simulate(
    strategy: str,
    seed: int,
    topics: list[str] | None = None,
    num_quizzes: int = 6,
    num_questions: int = 5,
) -> dict[str, float]:
    """Simulate one learner. Both strategies see the same learner for a given seed."""
    topics = topics or DEFAULT_TOPICS
    learner = make_learner(topics, random.Random(seed))
    initially_weak = sorted(topics, key=lambda t: learner.mastery[t])[:2]
    start_weak = statistics.mean(learner.mastery[t] for t in initially_weak)

    rng = random.Random(seed * 7919 + 1)
    engine = PersonalizationEngine()
    profile = StudentProfile(f"sim-{strategy}-{seed}")
    for q in range(num_quizzes):
        _run_quiz(learner, profile, engine, topics, strategy, rng, num_questions, q * num_questions)

    return {
        "weak_mastery_start": start_weak,
        "weak_mastery_end": statistics.mean(learner.mastery[t] for t in initially_weak),
        "overall_mastery_end": statistics.mean(learner.mastery.values()),
        "weak_topic_questions": sum(
            1 for a in profile.attempts for r in a.results if r.topic in initially_weak
        ),
    }


def compare_strategies(n_learners: int = 200, **kwargs: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for strategy in ("random", "adaptive"):
        runs = [simulate(strategy, seed, **kwargs) for seed in range(n_learners)]
        out[strategy] = {k: round(statistics.mean(r[k] for r in runs), 4) for k in runs[0]}
        out[strategy]["weak_mastery_gain"] = round(
            out[strategy]["weak_mastery_end"] - out[strategy]["weak_mastery_start"], 4
        )
    return out


if __name__ == "__main__":
    res = compare_strategies()
    print("SIMULATED learners (n=200, 6 quizzes x 5 questions)\n")
    print(f"{'metric':<24}{'random':>10}{'adaptive':>10}")
    for key in res["random"]:
        print(f"{key:<24}{res['random'][key]:>10}{res['adaptive'][key]:>10}")
