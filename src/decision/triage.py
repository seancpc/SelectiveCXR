"""
src/decision/triage.py — 分流決策層(支柱③ 前段)

依 conformal 棄答門檻 + 融合不確定性分數,把每個 study 分流為三檔(雙分界):
  - AUTO :能力內棄答 ≤ AUTO_MAX → 高度自動(僅少數待確認)
  - FLAG :AUTO_MAX < 棄答 ≤ FLAG_MAX → 標記那些 label,其餘自動判讀
  - REFER:棄答 > FLAG_MAX → 整案轉人工
(多標籤聯合棄答下「整張零棄答」幾乎不發生,故 AUTO 採「棄答 ≤ 門檻」而非「= 0」)

純規則,不需 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config import CHEXPERT_LABELS, AUTO_MAX_UNCERTAIN, FLAG_MAX_UNCERTAIN
from src.conformal.calibrator import ConformalCalibrator


class TriageDecision(str, Enum):
    AUTO = "auto"        # 自動判讀
    FLAG = "flag"        # 標記可疑 finding,其餘自動
    REFER = "refer"      # 整案轉人工


@dataclass
class TriageResult:
    decision: TriageDecision
    uncertain_labels: list[str]                  # 能力內 label 但這張棄答(真正 selective 棄答)
    boundary_labels: list[str]                   # 能力邊界 label(門檻 -inf,系統無把握,恆交人工)
    abstain_mask: np.ndarray                     # (NUM_LABELS,) bool
    per_label_uncertainty: dict[str, float]      # label -> 融合不確定性分數
    note: str = ""


def triage(
    fused_uncertainty: np.ndarray,
    calibrator: ConformalCalibrator,
    auto_max: int = AUTO_MAX_UNCERTAIN,
    flag_max: int = FLAG_MAX_UNCERTAIN,
) -> TriageResult:
    """對單一 study 做三檔分流。

    區分兩類 label:
      - 能力邊界(門檻 = -inf):conformal 在校準集上對此 label 找不到任何門檻能滿足
        錯誤率保證 → 系統承認對它無自動判讀能力,一律交人工。
      - 能力內(門檻有效):對這些做真正的 selective —— 這張片子哪些自動、哪些棄答。
    AUTO / FLAG / REFER 只依「能力內 label 的棄答數」判定,不被能力邊界 label 污染。

    fused_uncertainty: (NUM_LABELS,) 融合後 per-label 不確定性分數
    calibrator       : 已校準的 ConformalCalibrator
    auto_max         : 能力內棄答數 ≤ 此值 → AUTO(高度自動)
    flag_max         : 能力內棄答數 ≤ 此值(且 > auto_max) → FLAG;> 此值 → REFER
    """
    fused = np.asarray(fused_uncertainty, dtype=float).reshape(-1)
    mask = calibrator.abstain_mask(fused)                       # True = 棄答
    thresholds = calibrator.thresholds()
    per_label = {lbl: float(fused[i]) for i, lbl in enumerate(CHEXPERT_LABELS)}

    boundary_labels: list[str] = []     # 門檻 -inf:系統能力邊界,恆交人工
    uncertain_labels: list[str] = []    # 能力內但這張棄答:真正 selective 棄答
    for i, lbl in enumerate(CHEXPERT_LABELS):
        if thresholds[lbl] == float("-inf"):
            boundary_labels.append(lbl)
        elif mask[i]:
            uncertain_labels.append(lbl)

    n = len(uncertain_labels)
    if n <= auto_max:
        decision = TriageDecision.AUTO
        note = f"{n} 個能力內 label 待確認(≤{auto_max}),高度自動判讀放行。"
    elif n <= flag_max:
        decision = TriageDecision.FLAG
        note = f"{n} 個能力內 label 待確認,標記後其餘自動判讀。"
    else:
        decision = TriageDecision.REFER
        note = f"{n} 個能力內 label 待確認(超過上限 {flag_max}),整案轉人工。"

    if boundary_labels:
        note += f" 另有 {len(boundary_labels)} 個 label 屬系統能力邊界,一律交人工。"

    return TriageResult(
        decision=decision,
        uncertain_labels=uncertain_labels,
        boundary_labels=boundary_labels,
        abstain_mask=mask,
        per_label_uncertainty=per_label,
        note=note,
    )
