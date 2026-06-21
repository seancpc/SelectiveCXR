"""
src/models/base.py

三模型 ensemble 的統一輸出 schema 與抽象介面。

所有 ensemble 成員(Qwen3-VL / MedGemma / CheXagent)的原生輸出,都必須由
各自的 adapter map 成 CXRPrediction —— 下游的不確定性引擎、conformal 校準、
分流決策才能用一致的方式處理。此檔為所有下游模組的依賴基礎。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import CHEXPERT_LABELS, NUM_LABELS, LABEL_TO_INDEX, POSITIVE_THRESHOLD


@dataclass
class BoundingBox:
    """病灶定位框。

    座標一律 normalize 至 [0, 1](相對影像寬 / 高),統一座標系 ——
    各模型原生 bbox 格式不同(像素 / 0-1000 grid 等),由 adapter 換算。
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float = 1.0  # 模型對此框的信心(若模型未提供則為 1.0)

    def to_list(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def iou(self, other: "BoundingBox") -> float:
        """與另一個框的 Intersection-over-Union(grounding 一致性訊號會用到)。"""
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)
        inter = max(0.0, ix_max - ix_min) * max(0.0, iy_max - iy_min)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


@dataclass
class CXRPrediction:
    """單一模型對單張 CXR 的判讀結果(統一 schema)。"""

    model_name: str
    # shape (NUM_LABELS,),值域 [0, 1],順序固定同 config.CHEXPERT_LABELS
    label_probs: np.ndarray
    # finding 名稱 -> 該 finding 的 bbox 清單(僅含有定位的陽性病灶)
    boxes: dict[str, list[BoundingBox]] = field(default_factory=dict)
    # 模型原始輸出,保留供 audit log 重播
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label_probs = np.asarray(self.label_probs, dtype=float).reshape(-1)
        if self.label_probs.shape != (NUM_LABELS,):
            raise ValueError(
                f"label_probs 維度需為 ({NUM_LABELS},),"
                f"實際為 {self.label_probs.shape}(model={self.model_name})"
            )
        for name in self.boxes:
            if name not in LABEL_TO_INDEX:
                raise ValueError(f"boxes 含未知 finding 名稱:{name!r}")

    def prob(self, finding: str) -> float:
        """取得指定 finding 的陽性機率。"""
        return float(self.label_probs[LABEL_TO_INDEX[finding]])

    def positive_findings(self, threshold: float = POSITIVE_THRESHOLD) -> list[str]:
        """回傳機率 >= threshold 的 finding 名稱清單。"""
        return [
            CHEXPERT_LABELS[i]
            for i, p in enumerate(self.label_probs)
            if p >= threshold
        ]


class CXRModel(ABC):
    """ensemble 成員的抽象介面。

    各模型 adapter(qwen3vl.py / medgemma.py / chexagent.py)繼承本類別,
    負責把模型原生輸出轉成 CXRPrediction。adapter 的實作依賴 GPU,
    於 4090 桌機開發;本介面本身不依賴 GPU。
    """

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """載入模型權重(通常到 GPU)。"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Any) -> CXRPrediction:
        """對單張 CXR 影像推論,回傳統一 schema 的 CXRPrediction。"""
        raise NotImplementedError

    def unload(self) -> None:
        """釋放模型資源(預設不動作,adapter 需要時覆寫)。"""
        return None
