"""
src/conformal/calibrator.py — Conformal 校準層(支柱②)

把「不確定性分數」轉成「有統計保證的棄答門檻」。

採 Mondrian / per-label conformal:14 個 CheXpert label 各自校準一個門檻。
目標 —— 控制 selective risk:被「自動判讀」(不棄答)的樣本,其錯誤率 ≤ α。

方法:在 calibration set 上,對每個 label 由大到小掃描候選門檻 τ;對每個 τ
計算 selective risk 的 Hoeffding 上信賴界;取「上界 ≤ α」中能涵蓋最多樣本者
(即最大的 τ)。distribution-free、具有限樣本保證。

純統計,不需 GPU。
"""

from __future__ import annotations

import numpy as np

from config import CHEXPERT_LABELS, NUM_LABELS, CONFORMAL_ALPHA


def _hoeffding_upper_bound(error_rate: float, n: int, delta: float = 0.1) -> float:
    """selective risk 的 Hoeffding (1 - delta) 上信賴界。

    在 n 個樣本上觀察到 error_rate,回傳真實 risk 的保守上界。
    """
    if n == 0:
        return 1.0
    margin = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    return float(min(1.0, error_rate + margin))


class LabelCalibrator:
    """單一 label 的 conformal 棄答門檻校準器。"""

    def __init__(self, label: str, alpha: float = CONFORMAL_ALPHA, delta: float = 0.1,
                 use_hoeffding: bool = False):
        self.label = label
        self.alpha = alpha            # 目標錯誤率上限
        self.delta = delta            # Hoeffding 信賴界的失敗機率
        # False = empirical selective risk(少樣本/pilot 適用);True = Hoeffding 保守上界(需 ~500+ 樣本)
        self.use_hoeffding = use_hoeffding
        self.threshold_: float | None = None
        self.calibrated_ = False

    def calibrate(self, uncertainty: np.ndarray, is_error: np.ndarray) -> "LabelCalibrator":
        """以 calibration set 校準棄答門檻。

        uncertainty: (n,) 每個 calibration 樣本對本 label 的不確定性分數
        is_error   : (n,) 0/1,模型對本 label 的判定是否錯誤
        """
        uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)
        is_error = np.asarray(is_error, dtype=float).reshape(-1)
        if uncertainty.shape != is_error.shape:
            raise ValueError(f"label {self.label!r}:uncertainty 與 is_error 長度不一致")

        best_tau: float | None = None
        # 由大到小掃描 —— 門檻越大涵蓋越多,取仍滿足保證的最大門檻
        for tau in np.unique(uncertainty)[::-1]:
            mask = uncertainty <= tau                      # 被自動判讀的樣本
            n_auto = int(mask.sum())
            if n_auto == 0:
                continue
            emp_risk = float(is_error[mask].mean())
            risk = (_hoeffding_upper_bound(emp_risk, n_auto, self.delta)
                    if self.use_hoeffding else emp_risk)
            if risk <= self.alpha:
                best_tau = float(tau)
                break

        # 無門檻能滿足保證 → -inf(全部棄答,最保守)
        self.threshold_ = best_tau if best_tau is not None else float("-inf")
        self.calibrated_ = True
        return self

    def should_abstain(self, uncertainty: float) -> bool:
        """推論期:不確定性高於門檻 → 棄答。"""
        if not self.calibrated_:
            raise RuntimeError(f"label {self.label!r} 尚未校準")
        return float(uncertainty) > self.threshold_


class ConformalCalibrator:
    """全 14 label 的 conformal 校準器(Mondrian:每 label 獨立校準)。"""

    def __init__(self, alpha: float = CONFORMAL_ALPHA, delta: float = 0.1,
                 use_hoeffding: bool = False):
        self.alpha = alpha
        self.delta = delta
        self.use_hoeffding = use_hoeffding
        self.label_calibrators: dict[str, LabelCalibrator] = {
            lbl: LabelCalibrator(lbl, alpha, delta, use_hoeffding) for lbl in CHEXPERT_LABELS
        }
        self.calibrated_ = False

    def calibrate(self, uncertainty: np.ndarray, is_error: np.ndarray) -> "ConformalCalibrator":
        """uncertainty / is_error: shape (n_samples, NUM_LABELS)。"""
        uncertainty = np.asarray(uncertainty, dtype=float)
        is_error = np.asarray(is_error, dtype=float)
        if uncertainty.shape != is_error.shape or uncertainty.ndim != 2 \
                or uncertainty.shape[1] != NUM_LABELS:
            raise ValueError(f"輸入 shape 需為 (n_samples, {NUM_LABELS})")
        for i, lbl in enumerate(CHEXPERT_LABELS):
            self.label_calibrators[lbl].calibrate(uncertainty[:, i], is_error[:, i])
        self.calibrated_ = True
        return self

    def thresholds(self) -> dict[str, float]:
        """回傳各 label 校準後的棄答門檻。"""
        return {lbl: c.threshold_ for lbl, c in self.label_calibrators.items()}

    def abstain_mask(self, uncertainty: np.ndarray) -> np.ndarray:
        """單一 study 的 (NUM_LABELS,) 不確定性 → (NUM_LABELS,) bool 棄答遮罩。"""
        if not self.calibrated_:
            raise RuntimeError("ConformalCalibrator 尚未校準")
        uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)
        return np.array([
            self.label_calibrators[lbl].should_abstain(uncertainty[i])
            for i, lbl in enumerate(CHEXPERT_LABELS)
        ])
