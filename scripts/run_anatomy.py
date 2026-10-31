"""
scripts/run_anatomy.py — 定義 A 批量計算 + 對比定義 B(關鍵驗證)

對 test 影像跑 PSPNet 解剖分割,算定義 A grounding(VLM bbox vs 解剖區一致性),
存檔並與定義 B(跨模型 IoU)正面對比:哪個跟「實際錯誤」更相關
= 更好的不確定性訊號。這是定義 A 值不值得的判準。

桌機跑(mars env,需 GPU + torchxrayvision;inference_results.npz 須已存在):
  python scripts/run_anatomy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CXR_DATA_DIR

import numpy as np
import pandas as pd

from src.anatomy.segmenter import AnatomySegmenter
from src.uncertainty.grounding_anatomical import grounding_anatomical_one
from src.uncertainty.grounding_consistency import grounding_inconsistency

DATA = CXR_DATA_DIR


def auc(scores, labels):
    """rank-based AUC:scores 預測 labels==1 的能力;單一類別 → nan。"""
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    alls = np.concatenate([pos, neg])
    ranks = alls.argsort().argsort() + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    d = np.load(DATA / "inference_results.npz", allow_pickle=True)
    test = d["subsets"] == "test"
    model_boxes = d["model_boxes"][test]      # (Nt, M, 14, 4)
    ensemble = d["ensemble_prob"][test]       # (Nt, 14)
    gt = d["ground_truth"][test]              # (Nt, 14)
    dids = d["dicom_ids"][test]
    Nt = int(test.sum())

    man = (pd.read_csv(DATA / "subset_manifest.csv")
           .drop_duplicates("dicom_id").set_index("dicom_id"))
    seg = AnatomySegmenter()

    print(f"對 {Nt} 張 test 影像跑 PSPNet 解剖分割...")
    gA = np.full((Nt, 14), np.nan)
    for i in range(Nt):
        rel = man.loc[str(dids[i]), "rel_path"]
        masks = seg.segment(DATA / rel)
        gA[i] = grounding_anatomical_one(model_boxes[i], ensemble[i], masks, seg)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{Nt}")

    gB = np.array([grounding_inconsistency(model_boxes[i]) for i in range(Nt)])
    is_error = ((ensemble > 0.5).astype(float) != gt).astype(float)

    np.savez(DATA / "grounding_anatomical.npz",
             grounding_A=gA, grounding_B=gB, is_error=is_error, dicom_ids=dids)

    print(f"\n{'='*60}\n定義 A vs 定義 B —— 在各自有訊號的點上,預測 error 的能力\n{'='*60}")
    for name, g in [("定義B 跨模型IoU ", gB), ("定義A 解剖一致性", gA)]:
        m = ~np.isnan(g)
        gv, ev = g[m], is_error[m]
        err_mean = gv[ev == 1].mean() if (ev == 1).any() else float("nan")
        cor_mean = gv[ev == 0].mean() if (ev == 0).any() else float("nan")
        print(f"{name}: 訊號點 {int(m.sum()):5d} | error組均值 {err_mean:.3f} "
              f"| correct組均值 {cor_mean:.3f} | 差 {err_mean - cor_mean:+.3f} "
              f"| AUC {auc(gv, ev):.3f}")
    print("AUC 越高 / 兩組均值差越大 → 該訊號越能區辨錯誤 = 越好的不確定性訊號")


if __name__ == "__main__":
    main()
