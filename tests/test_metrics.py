import numpy as np
import pytest

from src.evaluation import (
    compute_auc,
    compute_metrics,
    compute_roc_curve,
    confusion_counts,
    select_threshold,
    threshold_sweep,
)



class TestConfusionCounts:
    def test_all_correct(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        cm = confusion_counts(y_true, y_pred)
        assert cm == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}

    def test_all_wrong(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        cm = confusion_counts(y_true, y_pred)
        assert cm == {"tp": 0, "fp": 2, "tn": 0, "fn": 2}

    def test_mixed(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        cm = confusion_counts(y_true, y_pred)
        assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}



class TestComputeMetrics:
    def test_perfect_separation(self):
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        m = compute_metrics(y_true, scores, threshold=0.5)
        assert m["tp"] == 2
        assert m["tn"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["balanced_accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_threshold_boundary(self):
        y_true = np.array([1, 0])
        scores = np.array([0.5, 0.5])
        m = compute_metrics(y_true, scores, threshold=0.5)
        assert m["tp"] == 1
        assert m["fp"] == 1
        assert m["tn"] == 0
        assert m["fn"] == 0

    def test_all_negative_prediction(self):
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.3, 0.3, 0.3, 0.3])
        m = compute_metrics(y_true, scores, threshold=0.5)
        assert m["tp"] == 0
        assert m["fn"] == 2
        assert m["tn"] == 2
        assert m["fp"] == 0
        assert m["tpr"] == pytest.approx(0.0)
        assert m["fpr"] == pytest.approx(0.0)

    def test_balanced_accuracy_equals_average_sensitivity_specificity(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        scores = np.array([0.8, 0.2, 0.6, 0.4, 0.7, 0.3])
        m = compute_metrics(y_true, scores, threshold=0.5)
        expected_ba = (m["tpr"] + m["tnr"]) / 2.0
        assert m["balanced_accuracy"] == pytest.approx(expected_ba, abs=1e-6)


class TestThresholdSweep:
    def test_returns_one_dict_per_threshold(self):
        y_true = np.array([1, 0, 1, 0])
        scores = np.array([0.8, 0.3, 0.7, 0.2])
        thresholds = np.array([0.1, 0.5, 0.9])
        results = threshold_sweep(y_true, scores, thresholds)
        assert len(results) == 3

    def test_sorted_ascending(self):
        y_true = np.array([1, 0])
        scores = np.array([0.7, 0.3])
        thresholds = np.array([0.9, 0.1, 0.5])
        results = threshold_sweep(y_true, scores, thresholds)
        thr_values = [r["threshold"] for r in results]
        assert thr_values == sorted(thr_values)

    def test_high_threshold_predicts_all_negative(self):
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        results = threshold_sweep(y_true, scores, np.array([1.0]))
        assert results[0]["tp"] == 0 



class TestSelectThreshold:
    def _make_sweep(self, thrs, bal_accs, f1s=None, fnrs=None):
        results = []
        for i, t in enumerate(thrs):
            r = {
                "threshold": t,
                "balanced_accuracy": bal_accs[i],
                "f1": (f1s[i] if f1s else 0.0),
                "fnr": (fnrs[i] if fnrs else 0.0),
            }
            results.append(r)
        return results

    def test_max_balanced_accuracy(self):
        sweep = self._make_sweep([0.3, 0.5, 0.7], [0.70, 0.85, 0.75])
        assert select_threshold(sweep, rule="max_balanced_accuracy") == pytest.approx(0.5)

    def test_max_f1(self):
        sweep = self._make_sweep([0.3, 0.5, 0.7], [0.0, 0.0, 0.0], f1s=[0.6, 0.9, 0.7])
        assert select_threshold(sweep, rule="max_f1") == pytest.approx(0.5)

    def test_min_fnr(self):
        sweep = self._make_sweep([0.3, 0.5, 0.7], [0.0, 0.0, 0.0], fnrs=[0.4, 0.2, 0.5])
        assert select_threshold(sweep, rule="min_fnr") == pytest.approx(0.5)

    def test_unknown_rule_raises(self):
        sweep = self._make_sweep([0.5], [0.8])
        with pytest.raises(ValueError, match="Unknown threshold selection rule"):
            select_threshold(sweep, rule="best_vibes")


class TestRocAndAuc:
    def test_roc_shape(self):
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.8, 0.6, 0.4, 0.2])
        fpr, tpr, thrs = compute_roc_curve(y_true, scores, n_points=11)
        assert len(fpr) == 11
        assert len(tpr) == 11
        assert len(thrs) == 11

    def test_perfect_classifier_auc(self):
        y_true = np.array([1] * 50 + [0] * 50)
        scores = np.array([0.9] * 50 + [0.1] * 50)
        fpr, tpr, _ = compute_roc_curve(y_true, scores, n_points=201)
        auc = compute_auc(fpr, tpr)
        assert auc > 0.95

    def test_random_classifier_auc(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=1000)
        scores = rng.random(1000)
        fpr, tpr, _ = compute_roc_curve(y_true, scores, n_points=201)
        auc = compute_auc(fpr, tpr)
        assert 0.4 < auc < 0.6
