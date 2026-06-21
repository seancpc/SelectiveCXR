"""src/decision/triage.py 單元測試。"""

import numpy as np
import pytest

from config import NUM_LABELS, CHEXPERT_LABELS
from src.conformal.calibrator import ConformalCalibrator
from src.decision.triage import triage, TriageDecision


@pytest.fixture
def calibrator():
    """用可分離資料校準:每個 label 門檻 = 0.1(低不確定性自動,高的棄答)。"""
    n = 500
    unc = np.concatenate([
        np.full((n, NUM_LABELS), 0.1),
        np.full((n, NUM_LABELS), 0.9),
    ])
    err = np.concatenate([
        np.zeros((n, NUM_LABELS)),
        np.ones((n, NUM_LABELS)),
    ])
    return ConformalCalibrator(alpha=0.05, delta=0.1).calibrate(unc, err)


def test_all_certain_is_auto(calibrator):
    """全部 label 不確定性低 → AUTO。"""
    res = triage(np.full(NUM_LABELS, 0.05), calibrator)
    assert res.decision == TriageDecision.AUTO
    assert res.uncertain_labels == []


def test_few_uncertain_is_flag(calibrator):
    """少數 label 不確定(<= max)→ FLAG。"""
    fused = np.full(NUM_LABELS, 0.05)
    fused[3] = 0.95
    res = triage(fused, calibrator, max_uncertain=3)
    assert res.decision == TriageDecision.FLAG
    assert res.uncertain_labels == [CHEXPERT_LABELS[3]]


def test_many_uncertain_is_refer(calibrator):
    """不確定 label 過多 → REFER。"""
    res = triage(np.full(NUM_LABELS, 0.95), calibrator, max_uncertain=3)
    assert res.decision == TriageDecision.REFER
    assert len(res.uncertain_labels) == NUM_LABELS


def test_triage_result_fields(calibrator):
    res = triage(np.full(NUM_LABELS, 0.05), calibrator)
    assert res.abstain_mask.shape == (NUM_LABELS,)
    assert len(res.per_label_uncertainty) == NUM_LABELS
    assert isinstance(res.boundary_labels, list)
    assert isinstance(res.note, str) and res.note


def test_boundary_labels_excluded_from_decision():
    """門檻 -inf 的能力邊界 label 不計入 selective 決策,也不觸發 REFER。"""
    n = 500
    unc = np.full((n, NUM_LABELS), 0.5)
    err = np.zeros((n, NUM_LABELS))
    err[:, 0] = 1.0          # label 0 全錯 → 任何門檻 risk=1.0 > 0.05 → -inf(能力邊界)
    cc = ConformalCalibrator(alpha=0.05, delta=0.1).calibrate(unc, err)

    # fused 全 0.4:label 0 是能力邊界;其餘門檻 0.5,0.4<0.5 → 自動判讀
    res = triage(np.full(NUM_LABELS, 0.4), cc, max_uncertain=3)
    assert res.boundary_labels == [CHEXPERT_LABELS[0]]
    assert res.uncertain_labels == []
    assert res.decision == TriageDecision.AUTO   # 能力邊界 label 不污染決策
