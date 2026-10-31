"""pytest 共用設定:確保專案根目錄在 sys.path,並提供共用 fixture。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from config import CHEXPERT_LABELS, NUM_LABELS
from src.models.base import CXRPrediction


@pytest.fixture
def make_prediction():
    """factory fixture:用機率向量(或 {label: prob} dict)建 CXRPrediction。"""

    def _make(probs, model_name: str = "m", boxes=None) -> CXRPrediction:
        if isinstance(probs, dict):
            arr = np.zeros(NUM_LABELS, dtype=float)
            for label, value in probs.items():
                arr[CHEXPERT_LABELS.index(label)] = value
            probs = arr
        return CXRPrediction(
            model_name=model_name,
            label_probs=np.asarray(probs, dtype=float),
            boxes=boxes or {},
        )

    return _make
