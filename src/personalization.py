"""Personalization engine for StudyMate AI (Role 5 - Personalization & Analytics).

Responsibilities
----------------
1. Record graded quiz attempts for each student.
2. Compute per-topic accuracy and detect weak topics.
3. Decide the difficulty and focus topics of the next quiz.
4. Persist student profiles as JSON (no database, no API keys).

Inputs come from the answer evaluator (Role 4) and the output (``QuizPlan``)
is consumed by the quiz generator (Role 3). See ``ROLE5_INTERFACE.md``.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Difficulty
# --------------------------------------------------------------------------- #
class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @classmethod
    def parse(cls, value: "Difficulty | str") -> "Difficulty":
        if isinstance(value, Difficulty):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            valid = ", ".join(d.value for d in cls)
            raise ValueError(f"Invalid difficulty {value!r}; expected one of: {valid}") from exc

    @property
    def level(self) -> int:
        return _ORDER.index(self)

    def harder(self) -> "Difficulty":
        return _ORDER[min(self.level + 1, len(_ORDER) - 1)]

    def easier(self) -> "Difficulty":
        return _ORDER[max(self.level - 1, 0)]


_ORDER = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_topic(topic: str) -> str:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    return " ".join(topic.split())


def _topic_key(topic: str) -> str:
    """Case/whitespace-insensitive key so 'Sorting' and ' sorting ' merge."""
    return _clean_topic(topic).casefold()


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
@dataclass
class QuestionResult:
    """One graded question. ``score`` is 0..1 (MCQ: 0 or 1; short answer: partial)."""

    question_id: str
    topic: str
    difficulty: Difficulty
    score: float
    question_type: str = "mcq"

    def __post_init__(self) -> None:
        self.question_id = str(self.question_id)
        self.topic = _clean_topic(self.topic)
        self.difficulty = Difficulty.parse(self.difficulty)
        if isinstance(self.score, bool):
            self.score = 1.0 if self.score else 0.0
        try:
            self.score = float(self.score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score must be a number, got {self.score!r}") from exc
        if math.isnan(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0 and 1, got {self.score}")

    @property
    def correct(self) -> bool:
        return self.score >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "topic": self.topic,
            "difficulty": self.difficulty.value,
            "score": self.score,
            "question_type": self.question_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_difficulty: Difficulty | str = Difficulty.MEDIUM) -> "QuestionResult":
        """Build from the evaluator's output.

        Accepts either ``score`` (0..1) or ``correct`` / ``is_correct`` (bool),
        so it tolerates small differences in Role 4's format.
        """
        if "score" in data and data["score"] is not None:
            score = data["score"]
        elif "correct" in data:
            score = bool(data["correct"])
        elif "is_correct" in data:
            score = bool(data["is_correct"])
        else:
            raise ValueError(f"Result for question {data.get('question_id')!r} has no score/correct field")
        return cls(
            question_id=data.get("question_id", data.get("id", "")),
            topic=data["topic"],
            difficulty=data.get("difficulty", default_difficulty),
            score=score,
            question_type=data.get("question_type", data.get("type", "mcq")),
        )


@dataclass
class QuizAttempt:
    results: list[QuestionResult]
    difficulty: Difficulty
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=_now_iso)
    lecture_id: str | None = None

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("A quiz attempt must contain at least one result")
        self.difficulty = Difficulty.parse(self.difficulty)

    @property
    def score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def num_questions(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "timestamp": self.timestamp,
            "lecture_id": self.lecture_id,
            "difficulty": self.difficulty.value,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuizAttempt":
        return cls(
            results=[QuestionResult.from_dict(r) for r in data["results"]],
            difficulty=data["difficulty"],
            attempt_id=data.get("attempt_id") or uuid.uuid4().hex[:12],
            timestamp=data.get("timestamp") or _now_iso(),
            lecture_id=data.get("lecture_id"),
        )


@dataclass
class StudentProfile:
    student_id: str
    attempts: list[QuizAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"student_id": self.student_id, "attempts": [a.to_dict() for a in self.attempts]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StudentProfile":
        return cls(
            student_id=data["student_id"],
            attempts=[QuizAttempt.from_dict(a) for a in data.get("attempts", [])],
        )


@dataclass
class TopicStats:
    topic: str
    answered: int = 0
    total_score: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.total_score / self.answered if self.answered else 0.0


@dataclass
class QuizPlan:
    """What the quiz generator (Role 3) should produce next."""

    difficulty: Difficulty
    focus_topics: list[str]
    num_questions: int
    reason: str
    topic_allocation: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty.value,
            "focus_topics": list(self.focus_topics),
            "num_questions": self.num_questions,
            "topic_allocation": dict(self.topic_allocation),
            "reason": self.reason,
        }


@dataclass
class PersonalizationConfig:
    raise_threshold: float = 0.8       # last score >= this -> harder
    lower_threshold: float = 0.5       # last score <  this -> easier
    weak_threshold: float = 0.6        # topic accuracy < this -> weak
    strong_threshold: float = 0.85     # topic accuracy >= this -> strong
    min_questions_per_topic: int = 2   # need this much evidence to label a topic
    recent_attempts: int = 3           # window for weak-topic detection
    max_focus_topics: int = 3
    focus_share: float = 0.6           # share of next quiz spent on focus topics
    num_questions: int = 5
    default_difficulty: Difficulty = Difficulty.MEDIUM

    def __post_init__(self) -> None:
        self.default_difficulty = Difficulty.parse(self.default_difficulty)
        for name in ("raise_threshold", "lower_threshold", "weak_threshold", "strong_threshold", "focus_share"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.lower_threshold >= self.raise_threshold:
            raise ValueError("lower_threshold must be smaller than raise_threshold")
        if self.weak_threshold >= self.strong_threshold:
            raise ValueError("weak_threshold must be smaller than strong_threshold")
        if self.min_questions_per_topic < 1 or self.recent_attempts < 1:
            raise ValueError("min_questions_per_topic and recent_attempts must be >= 1")
        if self.num_questions < 1 or self.max_focus_topics < 0:
            raise ValueError("num_questions must be >= 1 and max_focus_topics >= 0")


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class PersonalizationEngine:
    def __init__(self, config: PersonalizationConfig | None = None) -> None:
        self.config = config or PersonalizationConfig()

    # -- recording ---------------------------------------------------------- #
    def record_attempt(
        self,
        profile: StudentProfile,
        results: Iterable[QuestionResult | dict[str, Any]],
        difficulty: Difficulty | str,
        lecture_id: str | None = None,
    ) -> QuizAttempt:
        difficulty = Difficulty.parse(difficulty)
        parsed = [r if isinstance(r, QuestionResult) else QuestionResult.from_dict(r, difficulty) for r in results]
        attempt = QuizAttempt(results=parsed, difficulty=difficulty, lecture_id=lecture_id)
        profile.attempts.append(attempt)
        return attempt

    # -- statistics --------------------------------------------------------- #
    def topic_stats(self, profile: StudentProfile, recent: int | None = None) -> dict[str, TopicStats]:
        """Per-topic stats over all attempts, or only the last ``recent`` attempts."""
        attempts = profile.attempts[-recent:] if recent else profile.attempts
        stats: dict[str, TopicStats] = {}
        for attempt in attempts:
            for r in attempt.results:
                key = _topic_key(r.topic)
                entry = stats.setdefault(key, TopicStats(topic=r.topic))
                entry.answered += 1
                entry.total_score += r.score
        return {s.topic: s for s in stats.values()}

    def _labelled(self, profile: StudentProfile) -> list[TopicStats]:
        stats = self.topic_stats(profile, recent=self.config.recent_attempts)
        return [s for s in stats.values() if s.answered >= self.config.min_questions_per_topic]

    def weak_topics(self, profile: StudentProfile) -> list[str]:
        """Weak topics in the recent window, weakest first."""
        weak = [s for s in self._labelled(profile) if s.accuracy < self.config.weak_threshold]
        weak.sort(key=lambda s: (s.accuracy, -s.answered, s.topic.casefold()))
        return [s.topic for s in weak]

    def strong_topics(self, profile: StudentProfile) -> list[str]:
        strong = [s for s in self._labelled(profile) if s.accuracy >= self.config.strong_threshold]
        strong.sort(key=lambda s: (-s.accuracy, s.topic.casefold()))
        return [s.topic for s in strong]

    def topic_status(self, profile: StudentProfile, topic: str) -> str:
        stats = self.topic_stats(profile, recent=self.config.recent_attempts)
        key = _topic_key(topic)
        match = next((s for s in stats.values() if _topic_key(s.topic) == key), None)
        if match is None or match.answered < self.config.min_questions_per_topic:
            return "insufficient data"
        if match.accuracy < self.config.weak_threshold:
            return "weak"
        if match.accuracy >= self.config.strong_threshold:
            return "strong"
        return "ok"

    # -- adaptation --------------------------------------------------------- #
    def next_difficulty(self, profile: StudentProfile) -> Difficulty:
        if not profile.attempts:
            return self.config.default_difficulty
        last = profile.attempts[-1]
        if last.score >= self.config.raise_threshold:
            return last.difficulty.harder()
        if last.score < self.config.lower_threshold:
            return last.difficulty.easier()
        return last.difficulty

    def plan_next_quiz(
        self,
        profile: StudentProfile,
        available_topics: Iterable[str] | None = None,
        num_questions: int | None = None,
    ) -> QuizPlan:
        n = self.config.num_questions if num_questions is None else num_questions
        if n < 1:
            raise ValueError("num_questions must be >= 1")
        available = _dedupe([_clean_topic(t) for t in available_topics]) if available_topics else []
        available_keys = {_topic_key(t) for t in available}

        difficulty = self.next_difficulty(profile)
        weak = self.weak_topics(profile)
        if available:  # only focus on topics that exist in the current lecture
            weak = [t for t in weak if _topic_key(t) in available_keys]

        if weak:
            focus = weak[: self.config.max_focus_topics]
            focus_reason = f"focus on weak topics: {', '.join(focus)}"
        else:
            seen = {_topic_key(t) for t in self.topic_stats(profile)}
            unseen = [t for t in available if _topic_key(t) not in seen]
            focus = unseen[: self.config.max_focus_topics]
            focus_reason = (
                f"no weak topics; introduce unseen topics: {', '.join(focus)}" if focus
                else "no weak topics; mixed review"
            )

        if not profile.attempts:
            diff_reason = f"first quiz, starting at {difficulty.value}"
        else:
            last = profile.attempts[-1]
            if difficulty.level > last.difficulty.level:
                move = "moving up to"
            elif difficulty.level < last.difficulty.level:
                move = "moving down to"
            else:
                move = "staying at"
            diff_reason = f"scored {last.score:.0%} on {last.difficulty.value}, {move} {difficulty.value}"

        allocation = self.allocate_questions(focus, available, n)
        return QuizPlan(
            difficulty=difficulty,
            focus_topics=focus,
            num_questions=n,
            reason=f"{diff_reason}; {focus_reason}",
            topic_allocation=allocation,
        )

    def allocate_questions(self, focus: list[str], available: list[str], n: int) -> dict[str, int]:
        """Split ``n`` questions: ``focus_share`` to focus topics, the rest to others.

        Returns an empty dict if no topics are known (generator picks freely).
        """
        focus_keys = {_topic_key(t) for t in focus}
        others = [t for t in available if _topic_key(t) not in focus_keys]
        if not focus and not others:
            return {}
        if not others:
            focus_n = n
        elif not focus:
            focus_n = 0
        else:
            focus_n = min(n, max(len(focus), math.ceil(n * self.config.focus_share)))
            focus_n = min(focus_n, n - 1) if n > 1 else n  # leave room for review
        allocation: dict[str, int] = {}
        for i in range(focus_n):
            topic = focus[i % len(focus)]
            allocation[topic] = allocation.get(topic, 0) + 1
        for i in range(n - focus_n):
            topic = others[i % len(others)]
            allocation[topic] = allocation.get(topic, 0) + 1
        return allocation


def _dedupe(topics: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in topics:
        key = _topic_key(t)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class ProfileStore:
    """Saves one JSON file per student under ``directory`` (default ``data/profiles``)."""

    def __init__(self, directory: str | Path = "data/profiles") -> None:
        self.directory = Path(directory)

    def _path(self, student_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(student_id)).strip("._")
        if not safe:
            raise ValueError("student_id must contain at least one letter or digit")
        return self.directory / f"{safe}.json"

    def load(self, student_id: str) -> StudentProfile:
        path = self._path(student_id)
        if not path.exists():
            return StudentProfile(student_id=student_id)
        with path.open(encoding="utf-8") as f:
            return StudentProfile.from_dict(json.load(f))

    def save(self, profile: StudentProfile) -> Path:
        path = self._path(profile.student_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
        tmp.replace(path)  # atomic write
        return path
