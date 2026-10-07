"""Cross-checked against sklearn where the assertion is worth the dependency.

The AUC implementation is the one piece here that must match a reference, so a
skipped-by-default test compares it against ``sklearn.metrics.roc_auc_score`` on
random draws including ties. ``uv sync --extra bench`` (or any sklearn) enables it.
"""

import random

import pytest

from textscore.metrics import (
    accuracy_at,
    brier,
    by_group,
    class_means,
    confusion_at,
    evaluate,
    project_doc_seconds,
    roc_auc,
)


class TestRocAuc:
    def test_perfect_separation(self):
        assert roc_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0

    def test_inverted_ranking(self):
        assert roc_auc([0.1, 0.2, 0.9], [1, 1, 0]) == 0.0

    def test_mixed_ranking_is_quarter(self):
        scores, labels = [0.1, 0.9, 0.2, 0.8], [1, 0, 0, 1]
        # pos {0.1, 0.8} vs neg {0.9, 0.2}: only 0.8 > 0.2 wins -> 1 of 4 pairs
        assert roc_auc(scores, labels) == 0.25

    def test_all_ties_are_half(self):
        # every positive equals every negative -> exactly 0.5, not 0 or 1
        assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5

    def test_partial_ties(self):
        # pos {0.5, 0.5} vs neg {0.5, 0.1}: two ties + two wins = 3/4
        assert roc_auc([0.5, 0.5, 0.5, 0.1], [1, 1, 0, 0]) == pytest.approx(0.75)

    def test_single_class_is_undefined_not_zero(self):
        assert roc_auc([0.1, 0.2], [1, 1]) is None
        assert roc_auc([0.1, 0.2], [0, 0]) is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            roc_auc([0.1], [1, 0])

    def test_matches_sklearn_when_available(self):
        pytest.importorskip("sklearn.metrics")
        from sklearn.metrics import roc_auc_score

        rng = random.Random(7)
        for _ in range(30):
            scores = [round(rng.random(), 2) for _ in range(120)]  # coarse -> lots of ties
            labels = [rng.randint(0, 1) for _ in scores]
            if len(set(labels)) < 2:
                continue
            assert roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-9)


class TestThresholdMetrics:
    def test_accuracy_and_confusion(self):
        scores = [0.9, 0.4, 0.6, 0.1]
        labels = [1, 0, 0, 0]
        assert accuracy_at(scores, labels) == 0.75
        assert confusion_at(scores, labels) == {"tp": 1, "fp": 1, "tn": 2, "fn": 0}

    def test_threshold_moves_the_call(self):
        scores, labels = [0.6, 0.4], [1, 0]
        assert accuracy_at(scores, labels, 0.5) == 1.0
        assert accuracy_at(scores, labels, 0.7) == 0.5

    def test_brier_perfect_and_worst(self):
        assert brier([1.0, 0.0], [1, 0]) == 0.0
        assert brier([0.0, 1.0], [1, 0]) == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            accuracy_at([], [])
        with pytest.raises(ValueError):
            brier([], [])


class TestCalibration:
    def test_separation_direction(self):
        m = class_means([0.8, 0.9, 0.2, 0.1], [1, 1, 0, 0])
        assert m.pos_mean > m.neg_mean and not m.inverted

    def test_flipped_label_axis_is_detected(self):
        # the bug this exists to catch: scoring the wrong logits column
        m = class_means([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0])
        assert m.inverted

    def test_missing_class_gives_none_not_crash(self):
        m = class_means([0.5], [1])
        assert m.neg_mean is None and m.separation is None and not m.inverted


class TestAggregateReports:
    def test_evaluate_shape(self):
        out = evaluate([0.9, 0.2, 0.7, 0.1], [1, 1, 0, 0])
        assert out["n"] == 4 and out["auc"] == 0.75 and out["tp"] == 1 and out["fp"] == 1
        assert not out["inverted"]

    def test_evaluate_with_single_class_reports_no_auc(self):
        assert evaluate([0.9, 0.8], [1, 1])["auc"] is None

    def test_by_group_splits_the_signal(self):
        scores = [0.9, 0.8, 0.2, 0.1, 0.6]
        labels = [1, 1, 0, 0, 1]
        groups = ["gpt", "gpt", "human", "human", "other"]
        out = by_group(scores, labels, groups)
        assert out["gpt"] == {"n": 2, "mean_score": 0.85, "accuracy": 1.0, "label": "pos"}
        assert out["other"]["label"] == "pos" and out["human"]["accuracy"] == 1.0

    def test_by_group_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            by_group([0.1], [1], ["a", "b"])

    def test_latency_projection(self):
        assert project_doc_seconds(0.0035, 250) == 0.875
