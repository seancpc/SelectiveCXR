"""
src/anatomy/anatomy_map.py — CheXpert finding → 解剖先驗區對照(定義 A)

把 14 個 CheXpert finding 對應到「該病灶解剖上合理出現的區域」,用 PSPNet 的
14 解剖類表達。此表由放射學常識自建,**未搬運任何 gated 資料**(如 Chest ImaGenome);
僅供 grounding 一致性檢查,非臨床標準,可後續校驗調整。

特殊區域(PSPNet 無具名類別,由肺 mask 幾何推導):
  - 肺尖 apex          :肺 mask 最上緣帶狀區(Pneumothorax 好發)
  - 肋膈角 costophrenic:肺 mask 最下緣帶狀區(Pleural Effusion 好發)

值為 None = 該 finding 不適用解剖 grounding
(No Finding 無病灶位置;Support Devices 管線位置不定)。

純數學,不需 GPU。
"""

from __future__ import annotations

import numpy as np

LUNG = ["Left Lung", "Right Lung"]

# finding → 解剖區(PSPNet 類名);None = 不適用 grounding
FINDING_TO_ANATOMY: dict[str, list[str] | None] = {
    "No Finding":                 None,
    "Enlarged Cardiomediastinum": ["Mediastinum", "Heart", "Aorta"],
    "Cardiomegaly":               ["Heart"],
    "Lung Opacity":               LUNG,
    "Lung Lesion":                LUNG,
    "Edema":                      LUNG,
    "Consolidation":              LUNG,
    "Pneumonia":                  LUNG,
    "Atelectasis":                LUNG,
    "Pneumothorax":               LUNG,                                  # + 肺尖(幾何)
    "Pleural Effusion":           ["Facies Diaphragmatica"] + LUNG,      # + 肋膈角(幾何)
    "Pleural Other":              LUNG,
    "Fracture":                   ["Left Clavicle", "Right Clavicle",
                                   "Left Scapula", "Right Scapula", "Spine"],
    "Support Devices":            None,
}

# 需要幾何推導特殊區域的 finding
APEX_FINDINGS = {"Pneumothorax"}              # 肺尖
COSTOPHRENIC_FINDINGS = {"Pleural Effusion"}  # 肋膈角


def apex_region(lung_mask: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """肺尖:肺 mask 垂直範圍內,最上緣 frac 高度的帶狀區。"""
    ys = np.where(lung_mask.any(axis=1))[0]
    if len(ys) == 0:
        return np.zeros_like(lung_mask)
    top, bot = int(ys.min()), int(ys.max())
    cut = int(top + (bot - top) * frac)
    region = np.zeros_like(lung_mask)
    region[top:cut + 1] = lung_mask[top:cut + 1]
    return region


def costophrenic_region(lung_mask: np.ndarray, frac: float = 0.25) -> np.ndarray:
    """肋膈角:肺 mask 垂直範圍內,最下緣 frac 高度的帶狀區。"""
    ys = np.where(lung_mask.any(axis=1))[0]
    if len(ys) == 0:
        return np.zeros_like(lung_mask)
    top, bot = int(ys.min()), int(ys.max())
    cut = int(bot - (bot - top) * frac)
    region = np.zeros_like(lung_mask)
    region[cut:bot + 1] = lung_mask[cut:bot + 1]
    return region
