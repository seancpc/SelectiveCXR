"""src/conformal/calibrator.py 單元測試。"""

import numpy as np
import pytest

from config import NUM_LABELS
from src.conformal.calibrator import (
    LabelCalibrator,
    ConformalCalibrator,
    _hoeffding_upper_bound,
)


def test_hoeffding_bound_basics():
    assert _hoeffding_upper_bound(0.0, 0) == 1.0                  # n=0 → 最保守
    ub_small = _hoeffding_upper_bound(0.1, 10)
    ub_large = _hoeffding_upper_bound(0.1, 10000)
    assert ub_large < ub_small                                   # n 越大 margin 越小
    assert ub_large >= 0.1                                       # 上界 >= 觀察值
    assert _hoeffding_upper_bound(0.9, 100) <= 1.0               # 不超過 1


def test_loose_alpha_accepts_all():
    """alpha=1.0 → 門檻 = 最大不確定性,全部自動判讀。"""
    rng = np.random.default_rng(0)
    unc = rng.random(500)
    err = (rng.random(500) < 0.3).astype(float)
    cal = LabelCalibrator("Pneumonia", alpha=1.0).calibrate(unc, err)
    assert cal.threshold_ == pytest.approx(unc.max())
    assert not cal.should_abstain(unc.max())


def test_separable_case_threshold():
    """低不確定性全對、高的全錯 → 門檻落在分界,低值不棄答、高值棄答。"""
    n = 1000
    unc = np.concatenate([np.full(n, 0.1), np.full(n, 0.9)])
    err = np.concatenate([np.zeros(n), np.ones(n)])
    cal = LabelCalibrator("Edema", alpha=0.05, delta=0.1).calibrate(unc, err)
    assert cal.threshold_ == pytest.approx(0.1)
    assert cal.should_abstain(0.9)
    assert not cal.should_abstain(0.1)


def test_impossible_guarantee_abstains_all():
    """全部樣本都錯 → 無門檻能達標 → threshold = -inf,全棄答。"""
    unc = np.linspace(0.0, 1.0, 200)
    err = np.ones(200)
    cal = LabelCalibrator("Fracture", alpha=0.05).calibrate(unc, err)
    assert cal.threshold_ == float("-inf")
    assert cal.should_abstain(0.0)


def test_uncalibrated_label_raises():
    with pytest.raises(RuntimeError):
        LabelCalibrator("Pneumonia").should_abstain(0.5)


def test_conformal_calibrator_shape_check():
    cc = ConformalCalibrator(alpha=0.1)
    bad = np.zeros((10, NUM_LABELS - 1))
    with pytest.raises(ValueError):
        cc.calibrate(bad, bad)


def test_conformal_calibrator_end_to_end():
    rng = np.random.default_rng(1)
    unc = rng.random((300, NUM_LABELS))
    err = (rng.random((300, NUM_LABELS)) < 0.1).astype(float)
    cc = ConformalCalibrator(alpha=0.2).calibrate(unc, err)
    assert cc.calibrated_
    assert len(cc.thresholds()) == NUM_LABELS
    mask = cc.abstain_mask(unc[0])
    assert mask.shape == (NUM_LABELS,)
    assert mask.dtype == bool


def test_conformal_abstain_mask_uncalibrated_raises():
    with pytest.raises(RuntimeError):
        ConformalCalibrator().abstain_mask(np.zeros(NUM_LABELS))
