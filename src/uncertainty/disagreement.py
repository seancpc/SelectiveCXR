"""
src/uncertainty/disagreement.py — 訊號1:跨模型 disagreement

三個 ensemble 成員(Qwen3-VL / MedGemma / CheXagent)對同一張 CXR 各自給出
14 維機率向量。本模組量化它們之間的「分歧」—— 分歧大代表不確定性高。

提供三種分歧度量,供「訊號有效性實驗」比較:
  - std  :逐 label 機率標準差
  - vote :逐 label 二元判定的少數派比例
  - js   :逐 label 的 Jensen-Shannon divergence

純數學,不需 GPU。
"""

from __future__ import annotations

import numpy as np

from config import CHEXPERT_LABELS, POSITIVE_THRESHOLD
from src.models.base import CXRPrediction


def _stack_probs(predictions: list[CXRPrediction]) -> np.ndarray:
    """把多個 CXRPrediction 的 label_probs 疊成 (M, NUM_LABELS)。"""
    if len(predictions) < 2:
        raise ValueError("disagreement 需至少 2 個模型輸出")
    return np.stack([p.label_probs for p in predictions], axis=0)


def per_label_std(predictions: list[CXRPrediction]) -> np.ndarray:
    """逐 label 的機率標準差,shape (NUM_LABELS,)。"""
    return _stack_probs(predictions).std(axis=0)


def per_label_vote_disagreement(
    predictions: list[CXRPrediction],
    threshold: float = POSITIVE_THRESHOLD,
) -> np.ndarray:
    """逐 label 的二元判定不一致比例。

    每個模型對該 label 做陽性/陰性判定,回傳少數派比例:
    0 = 全體一致,接近 0.5 = 完全分裂。
    """
    probs = _stack_probs(predictions)
    votes = (probs >= threshold).astype(float)   # (M, L)
    m = votes.shape[0]
    pos = votes.sum(axis=0)                       # 每個 label 的陽性票數
    minority = np.minimum(pos, m - pos)           # 少數派票數
    return minority / m


def per_label_js_divergence(predictions: list[CXRPrediction]) -> np.ndarray:
    """逐 label 的 Jensen-Shannon divergence(以 log2,值域 0-1)。

    把每個模型對某 label 的輸出視為 Bernoulli 分布 [p, 1-p],
    JS = H(平均分布) - 平均(H(各分布))。
    """
    probs = _stack_probs(predictions)             # (M, L)
    eps = 1e-12
    p = np.clip(probs, eps, 1.0 - eps)

    def _binary_entropy(x: np.ndarray) -> np.ndarray:
        return -(x * np.log2(x) + (1.0 - x) * np.log2(1.0 - x))

    h_mean = _binary_entropy(p.mean(axis=0))
    h_each = _binary_entropy(p).mean(axis=0)
    return np.clip(h_mean - h_each, 0.0, 1.0)


def disagreement_scores(
    predictions: list[CXRPrediction],
    method: str = "vote",
) -> dict:
    """主入口。回傳 per-label 與 per-study 的 disagreement 分數。

    method: "std" | "vote" | "js"
    """
    dispatch = {
        "std": per_label_std,
        "vote": per_label_vote_disagreement,
        "js": per_label_js_divergence,
    }
    if method not in dispatch:
        raise ValueError(f"未知 method:{method!r},可選 {list(dispatch)}")
    per_label = dispatch[method](predictions)
    return {
        "method": method,
        "per_label": per_label,                                    # (NUM_LABELS,)
        "per_label_named": dict(zip(CHEXPERT_LABELS, per_label)),
        "per_study": float(per_label.mean()),
    }
