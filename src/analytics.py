"""Analytics for StudyMate AI (Role 5 - Personalization & Analytics).

Turns a ``StudentProfile`` into numbers and tables for the Streamlit
dashboard (Role 6) and for the Gain & Value section of the report (R7).
Every function returns plain JSON-serialisable data.
"""

from __future__ import annotations

from typing import Any

try:  # works both as ``src.analytics`` and with ``src`` on the path
    from .personalization import PersonalizationEngine, StudentProfile, _topic_key
except ImportError:  # pragma: no cover
    from personalization import PersonalizationEngine, StudentProfile, _topic_key


def score_history(profile: StudentProfile) -> list[dict[str, Any]]:
    """One row per attempt - use for a line chart of score over time."""
    return [
        {
            "attempt": i,
            "attempt_id": a.attempt_id,
            "timestamp": a.timestamp,
            "lecture_id": a.lecture_id,
            "difficulty": a.difficulty.value,
            "num_questions": a.num_questions,
            "score": round(a.score, 4),
        }
        for i, a in enumerate(profile.attempts, start=1)
    ]


def difficulty_history(profile: StudentProfile) -> list[str]:
    return [a.difficulty.value for a in profile.attempts]


def overall_accuracy(profile: StudentProfile) -> float | None:
    results = [r for a in profile.attempts for r in a.results]
    if not results:
        return None
    return round(sum(r.score for r in results) / len(results), 4)


def topic_summary(profile: StudentProfile, engine: PersonalizationEngine | None = None) -> list[dict[str, Any]]:
    """All-time per-topic table plus the current (recent-window) status. Weakest first."""
    engine = engine or PersonalizationEngine()
    rows = [
        {
            "topic": s.topic,
            "answered": s.answered,
            "accuracy": round(s.accuracy, 4),
            "status": engine.topic_status(profile, s.topic),
        }
        for s in engine.topic_stats(profile).values()
    ]
    rows.sort(key=lambda r: (r["accuracy"], r["topic"].casefold()))
    return rows


def improvement(profile: StudentProfile) -> dict[str, Any] | None:
    """First vs latest quiz score. ``None`` until there are two attempts."""
    if len(profile.attempts) < 2:
        return None
    first, last = profile.attempts[0].score, profile.attempts[-1].score
    return {
        "first_score": round(first, 4),
        "latest_score": round(last, 4),
        "change": round(last - first, 4),
        "attempts": len(profile.attempts),
    }


def weak_topic_gain(
    profile: StudentProfile,
    engine: PersonalizationEngine | None = None,
    baseline_attempts: int = 1,
) -> dict[str, Any] | None:
    """Did the topics that were weak at the start improve afterwards?

    Weak topics are identified from the first ``baseline_attempts`` attempts,
    then their accuracy there is compared with their accuracy in all later
    attempts. Returns ``None`` if there is not enough data.
    """
    engine = engine or PersonalizationEngine()
    if baseline_attempts < 1 or len(profile.attempts) <= baseline_attempts:
        return None

    baseline = StudentProfile(profile.student_id, profile.attempts[:baseline_attempts])
    after = StudentProfile(profile.student_id, profile.attempts[baseline_attempts:])
    base_stats = engine.topic_stats(baseline)
    weak = [
        s for s in base_stats.values()
        if s.accuracy < engine.config.weak_threshold
    ]
    if not weak:
        return None

    after_stats = {_topic_key(s.topic): s for s in engine.topic_stats(after).values()}
    per_topic = []
    for s in weak:
        later = after_stats.get(_topic_key(s.topic))
        per_topic.append({
            "topic": s.topic,
            "before": round(s.accuracy, 4),
            "after": round(later.accuracy, 4) if later else None,
            "questions_after": later.answered if later else 0,
        })

    measured = [t for t in per_topic if t["after"] is not None]
    if not measured:
        return {"topics": per_topic, "mean_before": None, "mean_after": None, "mean_gain": None}
    mean_before = sum(t["before"] for t in measured) / len(measured)
    mean_after = sum(t["after"] for t in measured) / len(measured)
    return {
        "topics": per_topic,
        "mean_before": round(mean_before, 4),
        "mean_after": round(mean_after, 4),
        "mean_gain": round(mean_after - mean_before, 4),
    }


def dashboard_data(
    profile: StudentProfile,
    engine: PersonalizationEngine | None = None,
    available_topics: list[str] | None = None,
) -> dict[str, Any]:
    """Everything the dashboard needs in one call."""
    engine = engine or PersonalizationEngine()
    return {
        "student_id": profile.student_id,
        "total_attempts": len(profile.attempts),
        "overall_accuracy": overall_accuracy(profile),
        "score_history": score_history(profile),
        "difficulty_history": difficulty_history(profile),
        "topics": topic_summary(profile, engine),
        "weak_topics": engine.weak_topics(profile),
        "strong_topics": engine.strong_topics(profile),
        "improvement": improvement(profile),
        "weak_topic_gain": weak_topic_gain(profile, engine),
        "next_quiz": engine.plan_next_quiz(profile, available_topics).to_dict(),
    }
