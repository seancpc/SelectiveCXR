"""
src/uncertainty/grounding_consistency.py — 支柱①(b):跨模型 bbox 一致性訊號

對每個 finding,計算「有定位的模型之間 bbox 的兩兩 IoU 平均」:
  - 高 IoU = 三模型把病灶框在同位置(定位可信)
  - 低 IoU = 定位分歧(不確定)
grounding 不一致性 = 1 - 平均 IoU(高 = 不確定),作為第二種不確定性訊號,
與 disagreement(label 機率分歧)互補 —— 一個看「判定」,一個看「定位」。

注:這是跨模型 bbox IoU 版(定義 B,不需解剖模型)。單模型文字↔bbox 的
解剖合理性(定義 A)需 CXR 解剖偵測模型,屬後續。

純數學,不需 GPU。
"""

from __future__ import annotations

import numpy as np

from config import NUM_LABELS


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """兩個 normalized box [x_min,y_min,x_max,y_max] 的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def grounding_inconsistency(model_boxes: np.ndarray) -> np.ndarray:
    """單張影像:M 個模型的 bbox → per-finding grounding 不一致性。

    model_boxes: (M, NUM_LABELS, 4),nan 代表該模型對該 finding 無 box。
    回傳: (NUM_LABELS,),值 = 1 - 兩兩 IoU 平均;有 box 的模型 < 2 → nan(無法算)。
    """
    M = model_boxes.shape[0]
    incon = np.full(NUM_LABELS, np.nan, dtype=float)
    for f in range(NUM_LABELS):
        boxes = [model_boxes[m, f] for m in range(M)
                 if not np.isnan(model_boxes[m, f]).any()]
        if len(boxes) < 2:
            continue
        ious = [_iou(boxes[i], boxes[j])
                for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
        incon[f] = 1.0 - float(np.mean(ious))
    return incon


def grounding_inconsistency_batch(model_boxes_batch: np.ndarray) -> np.ndarray:
    """批量:(N, M, NUM_LABELS, 4) → (N, NUM_LABELS)。"""
    return np.array([grounding_inconsistency(mb) for mb in model_boxes_batch])
