import numpy as np
from pneumreliab.evaluate import (abstention_curve, basic_metrics, ece, fit_temperature,
                                  softmax, threshold_at_sensitivity)


def test_softmax_rows_sum_to_one():
    assert np.allclose(softmax(np.random.randn(5, 2)).sum(1), 1)


def test_ece_zero_for_perfect_confident_predictions():
    y = np.array([0, 1, 0, 1]); pr = np.eye(2)[y]
    assert ece(pr, y) == 0


def test_temperature_softens_overconfident_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    logits = np.stack([-np.where(y == 1, 1, -1), np.where(y == 1, 1, -1)], 1) * 8.0
    flip = rng.random(500) < 0.25            # 25% confidently wrong
    logits[flip] *= -1
    assert fit_temperature(logits.astype(np.float32), y) > 1.5


def test_abstention_improves_accuracy_when_confidence_is_informative():
    y = np.array([1, 1, 0, 0]); probs = np.array([[.01, .99], [.01, .99], [.45, .55], [.45, .55]])
    y = np.array([1, 1, 0, 0]); probs[2] = [.45, .55]  # wrong but unsure
    out = abstention_curve(probs, y, covs=(1.0, 0.5))
    assert out["0.5"]["accuracy"] > out["1.0"]["accuracy"]


def test_basic_metrics_perfect():
    y = np.array([0, 0, 1, 1]); m = basic_metrics(y, np.array([.1, .2, .8, .9]))
    assert m["auroc"] == 1 and m["sensitivity"] == 1 and m["specificity"] == 1


def test_operating_threshold_reaches_target_sensitivity():
    y = np.array([1] * 20 + [0] * 20); p = np.r_[np.linspace(.3, .95, 20), np.linspace(.0, .6, 20)]
    thr = threshold_at_sensitivity(y, p, 0.95)
    assert (p[y == 1] >= thr).mean() >= 0.95
