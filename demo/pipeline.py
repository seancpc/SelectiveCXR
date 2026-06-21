"""
demo/pipeline.py — demo 推論管線(階段 A:用預跑結果)

把「一張影像的三模型輸出」轉成完整選擇性判讀決策:
  model_probs → 跨模型 disagreement → conformal 棄答 → triage 三檔。
全程重用 src/(conformal / decision),不重算推論。

階段 A:從 inference_results.npz 取預跑結果。
階段 B(後續):analyze() 的輸入改接即時推論結果,介面不變。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.conformal.calibrator import ConformalCalibrator
from src.decision.triage import triage, TriageResult


@dataclass
class StudyResult:
    dicom_id: str
    ensemble_prob: np.ndarray        # (14,) 三模型平均機率
    model_probs: np.ndarray          # (M, 14) 各模型各自的機率(展示跨 backbone 分歧)
    uncertainty: np.ndarray          # (14,) 跨模型 disagreement(分流依據)
    ground_truth: np.ndarray | None  # (14,) 0/1 or None
    triage: TriageResult


class DemoResults:
    """載入 inference_results.npz + 預校準 calibrator + manifest,提供逐張分析。"""

    def __init__(self, npz_path, calibrator_path, data_dir):
        self.data_dir = Path(data_dir)
        d = np.load(npz_path, allow_pickle=True)
        self.models = [str(m) for m in d["models"]]
        self.dicom_ids = d["dicom_ids"]
        self.subsets = d["subsets"]
        self.model_probs = d["model_probs"]   # (N, M, 14)
        self.ensemble = d["ensemble_prob"]    # (N, 14)
        self.gt = d["ground_truth"]           # (N, 14)
        with open(calibrator_path, "rb") as f:
            self.cc: ConformalCalibrator = pickle.load(f)
        self._man = (pd.read_csv(self.data_dir / "subset_manifest.csv")
                     .drop_duplicates("dicom_id").set_index("dicom_id"))

    def test_indices(self) -> np.ndarray:
        return np.where(self.subsets == "test")[0]

    def image_path(self, index: int) -> Path:
        did = str(self.dicom_ids[index])
        return self.data_dir / self._man.loc[did, "rel_path"]

    def analyze(self, index: int) -> StudyResult:
        probs = self.model_probs[index]       # (M, 14)
        uncertainty = probs.std(axis=0)       # (14,) 跨模型 disagreement
        tr = triage(uncertainty, self.cc)
        return StudyResult(
            dicom_id=str(self.dicom_ids[index]),
            ensemble_prob=self.ensemble[index],
            model_probs=probs,
            uncertainty=uncertainty,
            ground_truth=self.gt[index] if self.gt is not None else None,
            triage=tr,
        )
