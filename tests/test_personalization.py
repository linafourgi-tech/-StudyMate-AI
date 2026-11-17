"""Tests for Role 5 - personalization, analytics and simulation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from src.analytics import (  # noqa: E402
    dashboard_data,
    difficulty_history,
    improvement,
    overall_accuracy,
    score_history,
    topic_summary,
    weak_topic_gain,
)
from src.learner_simulation import compare_strategies, simulate  # noqa: E402
from src.personalization import (  # noqa: E402
    Difficulty,
    PersonalizationConfig,
    PersonalizationEngine,
    ProfileStore,
    QuestionResult,
    QuizAttempt,
    StudentProfile,
)


# ----------------------------------------------------------------- helpers --
def r(topic, score, qid="q", diff="medium"):
    return QuestionResult(qid, topic, diff, score)


def add(engine, profile, spec, diff="medium"):
    """spec: list of (topic, score)."""
    return engine.record_attempt(profile, [r(t, s, f"q{i}", diff) for i, (t, s) in enumerate(spec)], diff)


@pytest.fixture
def engine():
    return PersonalizationEngine()


@pytest.fixture
def profile():
    return StudentProfile("s1")


# -------------------------------------------------------------- Difficulty --
class TestDifficulty:
    def test_parse(self):
        assert Difficulty.parse(" HARD ") is Difficulty.HARD
        assert Difficulty.parse(Difficulty.EASY) is Difficulty.EASY

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            Difficulty.parse("extreme")

    def test_harder_easier_clamped(self):
        assert Difficulty.EASY.harder() is Difficulty.MEDIUM
        assert Difficulty.HARD.harder() is Difficulty.HARD
        assert Difficulty.EASY.easier() is Difficulty.EASY


# ---------------------------------------------------------- QuestionResult --
class TestQuestionResult:
    def test_bool_score(self):
        assert r("A", True).score == 1.0
        assert r("A", False).score == 0.0

    @pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), "abc"])
    def test_invalid_score(self, bad):
        with pytest.raises(ValueError):
            r("A", bad)

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_invalid_topic(self, bad):
        with pytest.raises(ValueError):
            r(bad, 1)

    def test_topic_whitespace_cleaned(self):
        assert r("  Linked   Lists ", 1).topic == "Linked Lists"

    def test_partial_credit_correct_flag(self):
        assert r("A", 0.5).correct and not r("A", 0.49).correct

    def test_from_dict_variants(self):
        assert QuestionResult.from_dict({"question_id": 1, "topic": "A", "score": 0.7}).score == 0.7
        assert QuestionResult.from_dict({"id": "x", "topic": "A", "correct": True}).score == 1.0
        assert QuestionResult.from_dict({"topic": "A", "is_correct": False}).score == 0.0

    def test_from_dict_default_difficulty(self):
        res = QuestionResult.from_dict({"topic": "A", "score": 1}, default_difficulty="hard")
        assert res.difficulty is Difficulty.HARD

    def test_from_dict_missing_score(self):
        with pytest.raises(ValueError):
            QuestionResult.from_dict({"question_id": "q", "topic": "A"})

    def test_roundtrip(self):
        x = QuestionResult("q1", "Trees", "hard", 0.25, "short")
        assert QuestionResult.from_dict(x.to_dict()) == x


# ------------------------------------------------------------- QuizAttempt --
class TestQuizAttempt:
    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            QuizAttempt(results=[], difficulty="easy")

    def test_score_is_mean(self):
        a = QuizAttempt([r("A", 1), r("B", 0), r("C", 0.5)], "medium")
        assert a.score == pytest.approx(0.5)
        assert a.num_questions == 3


# ------------------------------------------------------------------ Config --
class TestConfig:
    @pytest.mark.parametrize("kwargs", [
        {"lower_threshold": 0.9, "raise_threshold": 0.8},
        {"weak_threshold": 0.9, "strong_threshold": 0.8},
        {"focus_share": 1.5},
        {"min_questions_per_topic": 0},
        {"num_questions": 0},
    ])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            PersonalizationConfig(**kwargs)


# ------------------------------------------------------- record & stats --
class TestRecordingAndStats:
    def test_record_dicts_from_evaluator(self, engine, profile):
        engine.record_attempt(profile, [{"question_id": "1", "topic": "A", "correct": True}], "easy", "lec1")
        a = profile.attempts[0]
        assert a.lecture_id == "lec1"
        assert a.results[0].difficulty is Difficulty.EASY

    def test_topic_stats_merge_case(self, engine, profile):
        add(engine, profile, [("Sorting", 1), ("sorting ", 0)])
        stats = engine.topic_stats(profile)
        assert len(stats) == 1
        assert next(iter(stats.values())).accuracy == 0.5

    def test_recent_window(self, engine, profile):
        add(engine, profile, [("A", 0), ("A", 0)])
        add(engine, profile, [("A", 1), ("A", 1)])
        assert engine.topic_stats(profile)["A"].accuracy == 0.5
        assert engine.topic_stats(profile, recent=1)["A"].accuracy == 1.0


# -------------------------------------------------------------- weak/strong --
class TestTopicLabels:
    def test_weak_sorted_weakest_first(self, engine, profile):
        add(engine, profile, [("A", 0.5), ("A", 0.5), ("B", 0), ("B", 0), ("C", 1), ("C", 1)])
        assert engine.weak_topics(profile) == ["B", "A"]
        assert engine.strong_topics(profile) == ["C"]

    def test_min_evidence(self, engine, profile):
        add(engine, profile, [("A", 0), ("B", 1)])
        assert engine.weak_topics(profile) == []
        assert engine.topic_status(profile, "A") == "insufficient data"

    def test_weak_topic_recovers(self, engine, profile):
        add(engine, profile, [("A", 0), ("A", 0)])
        for _ in range(3):
            add(engine, profile, [("A", 1), ("A", 1)])
        assert engine.weak_topics(profile) == []
        assert engine.topic_status(profile, "a") == "strong"

    def test_status_ok_and_unknown(self, engine, profile):
        add(engine, profile, [("A", 1), ("A", 0.5)])
        assert engine.topic_status(profile, "A") == "ok"
        assert engine.topic_status(profile, "Z") == "insufficient data"


# -------------------------------------------------------------- difficulty --
class TestNextDifficulty:
    def test_default_when_new(self, engine, profile):
        assert engine.next_difficulty(profile) is Difficulty.MEDIUM

    @pytest.mark.parametrize("score,start,expected", [
        (1.0, "medium", "hard"),
        (0.8, "easy", "medium"),      # boundary: >= raise
        (0.79, "medium", "medium"),
        (0.5, "medium", "medium"),    # boundary: not < lower
        (0.4, "medium", "easy"),
        (0.0, "easy", "easy"),        # clamp
        (1.0, "hard", "hard"),        # clamp
    ])
    def test_rules(self, engine, profile, score, start, expected):
        engine.record_attempt(profile, [r("A", score, diff=start)], start)
        assert engine.next_difficulty(profile).value == expected

    def test_custom_default(self, profile):
        eng = PersonalizationEngine(PersonalizationConfig(default_difficulty="easy"))
        assert eng.next_difficulty(profile) is Difficulty.EASY


# --------------------------------------------------------------- planning --
class TestPlan:
    def test_first_quiz(self, engine, profile):
        plan = engine.plan_next_quiz(profile, ["A", "B", "C", "D"])
        assert plan.difficulty is Difficulty.MEDIUM
        assert plan.focus_topics == ["A", "B", "C"]  # unseen topics, capped at 3
        assert sum(plan.topic_allocation.values()) == 5
        assert "first quiz" in plan.reason

    def test_focus_on_weak(self, engine, profile):
        add(engine, profile, [("A", 0), ("A", 0), ("B", 1), ("B", 1), ("C", 1)])
        plan = engine.plan_next_quiz(profile, ["A", "B", "C"])
        assert plan.focus_topics == ["A"]
        assert plan.topic_allocation["A"] == 3          # ceil(5 * 0.6)
        assert sum(plan.topic_allocation.values()) == 5
        assert "weak topics: A" in plan.reason
        assert "staying at medium" in plan.reason

    def test_weak_not_in_lecture_ignored(self, engine, profile):
        add(engine, profile, [("Old", 0), ("Old", 0)])
        plan = engine.plan_next_quiz(profile, ["New1", "New2"])
        assert "Old" not in plan.focus_topics
        assert plan.focus_topics == ["New1", "New2"]

    def test_no_topics_known(self, engine, profile):
        plan = engine.plan_next_quiz(profile)
        assert plan.topic_allocation == {}
        assert "mixed review" in plan.reason

    def test_max_focus_topics(self, engine, profile):
        add(engine, profile, [(t, 0) for t in "ABCDE" for _ in range(2)])
        plan = engine.plan_next_quiz(profile, list("ABCDE"))
        assert len(plan.focus_topics) == 3
        assert "moving down to easy" in plan.reason

    def test_num_questions_override_and_invalid(self, engine, profile):
        assert engine.plan_next_quiz(profile, ["A"], num_questions=10).num_questions == 10
        with pytest.raises(ValueError):
            engine.plan_next_quiz(profile, ["A"], num_questions=0)

    def test_duplicate_available_topics(self, engine, profile):
        plan = engine.plan_next_quiz(profile, ["A", "a", " A "])
        assert plan.topic_allocation == {"A": 5}

    def test_to_dict_json(self, engine, profile):
        json.dumps(engine.plan_next_quiz(profile, ["A"]).to_dict())


class TestAllocation:
    @pytest.mark.parametrize("focus,available,n", [
        (["A"], ["A", "B", "C"], 5),
        (["A", "B", "C"], ["A", "B", "C", "D"], 2),
        ([], ["A", "B"], 3),
        (["A"], ["A"], 4),
        (["A"], ["A", "B"], 1),
    ])
    def test_always_sums_to_n(self, engine, focus, available, n):
        assert sum(engine.allocate_questions(focus, available, n).values()) == n

    def test_leaves_room_for_review(self, engine):
        alloc = engine.allocate_questions(["A", "B", "C"], ["A", "B", "C", "D"], 3)
        assert alloc.get("D", 0) >= 1


# ------------------------------------------------------------- persistence --
class TestStore:
    def test_roundtrip(self, tmp_path, engine, profile):
        add(engine, profile, [("A", 1), ("B", 0.5)], "hard")
        store = ProfileStore(tmp_path)
        store.save(profile)
        loaded = store.load("s1")
        assert loaded.to_dict() == profile.to_dict()

    def test_missing_returns_empty(self, tmp_path):
        assert ProfileStore(tmp_path).load("nobody").attempts == []

    def test_unsafe_id_sanitised(self, tmp_path):
        path = ProfileStore(tmp_path).save(StudentProfile("../../evil id"))
        assert path.parent == tmp_path

    def test_empty_id_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            ProfileStore(tmp_path).save(StudentProfile("///"))


# --------------------------------------------------------------- analytics --
class TestAnalytics:
    def test_empty_profile(self, profile):
        data = dashboard_data(profile)
        assert data["total_attempts"] == 0
        assert data["overall_accuracy"] is None
        assert data["improvement"] is None
        assert data["weak_topic_gain"] is None
        json.dumps(data)

    def test_history_and_improvement(self, engine, profile):
        add(engine, profile, [("A", 0), ("B", 1)])          # 0.5
        add(engine, profile, [("A", 1), ("B", 1)], "hard")  # 1.0
        hist = score_history(profile)
        assert [h["score"] for h in hist] == [0.5, 1.0]
        assert difficulty_history(profile) == ["medium", "hard"]
        assert improvement(profile)["change"] == 0.5
        assert overall_accuracy(profile) == 0.75

    def test_topic_summary_sorted(self, engine, profile):
        add(engine, profile, [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
        rows = topic_summary(profile)
        assert [x["topic"] for x in rows] == ["B", "A"]
        assert rows[0]["status"] == "weak" and rows[1]["status"] == "strong"

    def test_weak_topic_gain(self, engine, profile):
        add(engine, profile, [("A", 0), ("A", 0), ("B", 1)])
        add(engine, profile, [("A", 1), ("A", 0)])
        add(engine, profile, [("A", 1), ("A", 1)])
        gain = weak_topic_gain(profile)
        assert gain["mean_before"] == 0.0
        assert gain["mean_after"] == 0.75
        assert gain["mean_gain"] == 0.75

    def test_weak_topic_gain_not_practised(self, engine, profile):
        add(engine, profile, [("A", 0)])
        add(engine, profile, [("B", 1)])
        gain = weak_topic_gain(profile)
        assert gain["mean_gain"] is None
        assert gain["topics"][0]["questions_after"] == 0

    def test_weak_topic_gain_none_cases(self, engine, profile):
        add(engine, profile, [("A", 1)])
        add(engine, profile, [("A", 1)])
        assert weak_topic_gain(profile) is None              # nothing weak
        assert weak_topic_gain(profile, baseline_attempts=0) is None

    def test_dashboard_full(self, engine, profile):
        add(engine, profile, [("A", 0), ("A", 0), ("B", 1), ("B", 1)])
        data = dashboard_data(profile, available_topics=["A", "B"])
        assert data["weak_topics"] == ["A"]
        assert data["next_quiz"]["focus_topics"] == ["A"]
        json.dumps(data)


# -------------------------------------------------------------- simulation --
class TestSimulation:
    def test_deterministic(self):
        assert simulate("adaptive", seed=3) == simulate("adaptive", seed=3)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            simulate("magic", seed=0)

    def test_adaptive_practises_weak_topics_more(self):
        res = compare_strategies(n_learners=40)
        assert res["adaptive"]["weak_topic_questions"] > res["random"]["weak_topic_questions"]
        assert res["adaptive"]["weak_mastery_gain"] > res["random"]["weak_mastery_gain"]
