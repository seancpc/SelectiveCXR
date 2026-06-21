"""src/uncertainty/disagreement.py 單元測試。"""

import numpy as np
import pytest

from config import NUM_LABELS
from src.uncertainty.disagreement import (
    disagreement_scores,
    per_label_std,
    per_label_vote_disagreement,
    per_label_js_divergence,
)


def test_identical_predictions_zero_disagreement(make_prediction):
    """三個完全相同的預測 → 所有度量分歧為 0。"""
    preds = [make_prediction(np.full(NUM_LABELS, 0.7), f"m{i}") for i in range(3)]
    assert np.allclose(per_label_std(preds), 0.0)
    assert np.allclose(per_label_vote_disagreement(preds), 0.0)
    assert np.allclose(per_label_js_divergence(preds), 0.0)


def test_vote_disagreement_minority_ratio(make_prediction):
    """3 模型 2 陽 1 陰 → 少數派比例 = 1/3。"""
    probs = [np.full(NUM_LABELS, 0.9), np.full(NUM_LABELS, 0.9), np.full(NUM_LABELS, 0.1)]
    preds = [make_prediction(pr, f"m{i}") for i, pr in enumerate(probs)]
    assert np.allclose(per_label_vote_disagreement(preds), 1.0 / 3.0)


def test_vote_disagreement_full_split(make_prediction):
    """4 模型完全對半分裂 → 少數派比例 = 0.5。"""
    probs = [
        np.full(NUM_LABELS, 0.9), np.full(NUM_LABELS, 0.9),
        np.full(NUM_LABELS, 0.1), np.full(NUM_LABELS, 0.1),
    ]
    preds = [make_prediction(pr, f"m{i}") for i, pr in enumerate(probs)]
    assert np.allclose(per_label_vote_disagreement(preds), 0.5)


def test_std_known_value(make_prediction):
    """std 度量:[0, 1] 的母體標準差 = 0.5。"""
    probs = [np.zeros(NUM_LABELS), np.ones(NUM_LABELS)]
    preds = [make_prediction(pr, f"m{i}") for i, pr in enumerate(probs)]
    assert np.allclose(per_label_std(preds), 0.5)


def test_js_divergence_full_split(make_prediction):
    """JS divergence:完全分裂(0 vs 1)→ 接近 1,且值域 [0, 1]。"""
    probs = [np.zeros(NUM_LABELS), np.ones(NUM_LABELS)]
    preds = [make_prediction(pr, f"m{i}") for i, pr in enumerate(probs)]
    js = per_label_js_divergence(preds)
    assert np.all(js >= 0.0) and np.all(js <= 1.0)
    assert np.allclose(js, 1.0, atol=1e-6)


def test_disagreement_scores_structure(make_prediction):
    preds = [make_prediction(np.full(NUM_LABELS, 0.5), f"m{i}") for i in range(3)]
    out = disagreement_scores(preds, method="vote")
    assert out["method"] == "vote"
    assert out["per_label"].shape == (NUM_LABELS,)
    assert len(out["per_label_named"]) == NUM_LABELS
    assert isinstance(out["per_study"], float)


def test_requires_at_least_two_models(make_prediction):
    with pytest.raises(ValueError):
        per_label_std([make_prediction(np.full(NUM_LABELS, 0.5))])


def test_unknown_method_raises(make_prediction):
    preds = [make_prediction(np.full(NUM_LABELS, 0.5), f"m{i}") for i in range(2)]
    with pytest.raises(ValueError):
        disagreement_scores(preds, method="bogus")
