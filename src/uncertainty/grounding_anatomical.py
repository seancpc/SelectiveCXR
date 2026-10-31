"""
src/uncertainty/grounding_anatomical.py — 定義 A:VLM 判讀 vs 解剖位置一致性

對 VLM 報陽性的 finding,檢查其 bbox 是否落在「該病解剖上合理的區域」
(用 PSPNet 可靠的解剖 mask)。相對定義 B(兩個都不準的框互比)的優勢:
  - 一端(解剖區)可靠 → 訊號有臨床語義
  - 超大框(跨多區、定位不精確)直接懲罰
  - bbox 落在錯誤解剖區 → 高不確定

grounding_A = 1 - (bbox 落在合理解剖區的比例);高 = 指錯地方 / 框太大 = 不確定。
無模型給框、或 finding 不適用 grounding(No Finding / Support Devices)→ nan(退回 disagreement)。

純數學(解剖 mask 由 src/anatomy/segmenter 預先算好)。
"""

from __future__ import annotations

import numpy as np

from config import CHEXPERT_LABELS
from src.anatomy.anatomy_map import (
    FINDING_TO_ANATOMY, APEX_FINDINGS, COSTOPHRENIC_FINDINGS,
    apex_region, costophrenic_region,
)

OVERSIZE_AREA = 0.4   # bbox normalized 面積 > 此值 → 視為定位不可信(超大框)


def _bbox_to_pixel(box, R):
    """normalized [x1,y1,x2,y2] → (y1,y2,x1,x2) 像素 slice 範圍(R×R)。"""
    x1, y1, x2, y2 = box
    return (int(np.clip(y1, 0, 1) * R), int(np.clip(y2, 0, 1) * R),
            int(np.clip(x1, 0, 1) * R), int(np.clip(x2, 0, 1) * R))


def region_for_finding(finding, anatomy_masks, segmenter):
    """取 finding 的解剖先驗區 mask (R,R) bool;不適用 grounding → None。"""
    names = FINDING_TO_ANATOMY.get(finding)
    if names is None:
        return None
    region = segmenter.region_mask(anatomy_masks, names)
    if finding in APEX_FINDINGS or finding in COSTOPHRENIC_FINDINGS:
        lung = segmenter.lung_mask(anatomy_masks)
        if finding in APEX_FINDINGS:
            region = region | apex_region(lung)
        if finding in COSTOPHRENIC_FINDINGS:
            region = region | costophrenic_region(lung)
    return region


def grounding_anatomical_one(model_boxes, ensemble_prob, anatomy_masks, segmenter):
    """單張:VLM bbox + 解剖 mask → (14,) 定義 A 不一致性(含 nan)。

    model_boxes  : (M, 14, 4) normalized bbox,nan = 無框
    ensemble_prob: (14,) 模型平均機率(只檢查 >0.5 的報陽性 finding)
    anatomy_masks: (14, R, R) PSPNet 解剖機率 mask
    segmenter    : AnatomySegmenter(提供 region_mask / lung_mask)
    """
    R = anatomy_masks.shape[-1]
    out = np.full(len(CHEXPERT_LABELS), np.nan, dtype=float)
    for f_i, finding in enumerate(CHEXPERT_LABELS):
        if ensemble_prob[f_i] <= 0.5:
            continue                                       # 只檢查報陽性的
        region = region_for_finding(finding, anatomy_masks, segmenter)
        if region is None or not region.any():
            continue                                       # 不適用 / 該區未分割出
        overlaps = []
        for m in range(model_boxes.shape[0]):
            box = model_boxes[m, f_i]
            if np.isnan(box).any():
                continue
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if area > OVERSIZE_AREA:
                overlaps.append(0.0)                       # 超大框 = 定位不可信
                continue
            y1, y2, x1, x2 = _bbox_to_pixel(box, R)
            if y2 <= y1 or x2 <= x1:
                overlaps.append(0.0)                       # 退化框
                continue
            overlaps.append(float(region[y1:y2, x1:x2].mean()))  # bbox 內落在解剖區比例
        if overlaps:
            out[f_i] = 1.0 - float(np.mean(overlaps))      # 高 = 指錯 / 框爛 = 不確定
    return out
