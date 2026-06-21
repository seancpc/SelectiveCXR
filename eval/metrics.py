"""
eval/metrics.py — 評測指標

選擇性判讀系統的評測:不只看準確率,重點在「棄答品質」。
  - coverage                   :自動判讀(不棄答)比例
  - selective_accuracy         :不棄答子集的準確率
  - risk_coverage_curve / aurc :風險-覆蓋權衡
  - expected_calibration_error :信心校準誤差(ECE)
  - conformal_empirical_coverage:自動判讀子集實際錯誤率(驗證 conformal 保證)
  - abstention_precision       :棄答的是否真為難 case

純數學,不需 GPU。
"""

from __future__ import annotations

import numpy as np


def coverage(abstain_mask) -> float:
    """自動判讀(不棄答)的比例。"""
    abstain = np.asarray(abstain_mask, dtype=bool).reshape(-1)
    return float((~abstain).mean()) if abstain.size else float("nan")


def selective_accuracy(is_correct, abstain_mask) -> float:
    """不棄答子集的準確率。is_correct / abstain_mask: (n,)。"""
    is_correct = np.asarray(is_correct, dtype=float).reshape(-1)
    abstain = np.asarray(abstain_mask, dtype=bool).reshape(-1)
    kept = ~abstain
    if kept.sum() == 0:
        return float("nan")
    return float(is_correct[kept].mean())


def risk_coverage_curve(is_correct, uncertainty):
    """Risk-Coverage 曲線。

    依不確定性由低到高逐步納入樣本,回傳 (coverage[], risk[])。
    risk = 已納入樣本的錯誤率。
    """
    is_correct = np.asarray(is_correct, dtype=float).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)
    order = np.argsort(uncertainty, kind="stable")     # 低不確定性優先納入
    err = 1.0 - is_correct[order]
    counts = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / counts
    cov = counts / len(err)
    return cov, risk


def aurc(is_correct, uncertainty) -> float:
    """Area Under Risk-Coverage curve —— 越低越好。"""
    cov, risk = risk_coverage_curve(is_correct, uncertainty)
    if len(cov) < 2:
        return float("nan")
    # 梯形積分(避免依賴 np.trapz / np.trapezoid 的版本差異)
    return float(np.sum(np.diff(cov) * (risk[:-1] + risk[1:]) / 2.0))


def expected_calibration_error(confidence, is_correct, n_bins: int = 10) -> float:
    """ECE。confidence: (n,) 模型信心;is_correct: (n,) 0/1。"""
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    is_correct = np.asarray(is_correct, dtype=float).reshape(-1)
    n = len(confidence)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence > lo) & (confidence <= hi)
        if i == 0:                                     # 第一個 bin 含下界
            mask |= confidence <= lo
        if mask.sum() == 0:
            continue
        acc = is_correct[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def conformal_empirical_coverage(is_error, abstain_mask) -> float:
    """自動判讀子集的實際錯誤率 —— 應 ≤ conformal 設定的 α,用以驗證保證成立。"""
    is_error = np.asarray(is_error, dtype=float).reshape(-1)
    abstain = np.asarray(abstain_mask, dtype=bool).reshape(-1)
    kept = ~abstain
    if kept.sum() == 0:
        return float("nan")
    return float(is_error[kept].mean())


def abstention_precision(is_error, abstain_mask) -> float:
    """棄答精確度 —— 被棄答的樣本中真的會答錯的比例。

    高 = 棄答的確實是難 case;低 = 過度保守、亂棄。
    """
    is_error = np.asarray(is_error, dtype=float).reshape(-1)
    abstain = np.asarray(abstain_mask, dtype=bool).reshape(-1)
    if abstain.sum() == 0:
        return float("nan")
    return float(is_error[abstain].mean())
