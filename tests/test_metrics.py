"""eval/metrics.py 單元測試。"""

import numpy as np

from eval.metrics import (
    coverage,
    selective_accuracy,
    risk_coverage_curve,
    aurc,
    expected_calibration_error,
    conformal_empirical_coverage,
    abstention_precision,
)


def test_coverage():
    mask = np.array([False, False, True, True])      # 2 棄答 / 4
    assert coverage(mask) == 0.5


def test_selective_accuracy():
    is_correct = np.array([1, 0, 1, 1])
    abstain = np.array([False, False, True, True])   # 不棄答 = index 0,1
    assert selective_accuracy(is_correct, abstain) == 0.5


def test_risk_coverage_curve_known_values():
    is_correct = np.array([1, 1, 0, 1])
    uncertainty = np.array([0.1, 0.2, 0.3, 0.4])
    cov, risk = risk_coverage_curve(is_correct, uncertainty)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0.0, 0.0, 1.0 / 3.0, 0.25])


def test_aurc_perfect_is_zero():
    """全對 → risk 恆 0 → AURC = 0。"""
    assert aurc(np.ones(10), np.linspace(0.0, 1.0, 10)) == 0.0


def test_ece_perfectly_calibrated():
    """信心 = 實際正確率 → ECE = 0。"""
    confidence = np.array([1.0, 1.0, 0.0, 0.0])
    is_correct = np.array([1, 1, 0, 0])
    assert expected_calibration_error(confidence, is_correct, n_bins=10) == 0.0


def test_ece_fully_miscalibrated():
    """信心 1.0 卻全錯 → ECE = 1。"""
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([0, 0])) == 1.0


def test_conformal_empirical_coverage():
    is_error = np.array([0, 1, 0, 0])
    abstain = np.array([False, False, True, True])   # 不棄答 = index 0,1
    assert conformal_empirical_coverage(is_error, abstain) == 0.5


def test_abstention_precision():
    is_error = np.array([0, 0, 1, 1])
    abstain = np.array([False, False, True, True])   # 棄答 = index 2,3
    assert abstention_precision(is_error, abstain) == 1.0
