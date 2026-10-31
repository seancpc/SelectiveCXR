"""
scripts/merge_vllm.py — 合併各模型 vllm_{model}.npz → inference_results.npz

把 run_inference_vllm.py 各模型的單獨結果合併成 run_conformal 需要的格式:
  - 堆 model_probs (N, M, 14)
  - disagreement = 模型間 std(連續)
  - ensemble_prob = 模型間平均
  - ground_truth 從 subset_manifest.csv(CheXpert 1.0=陽性)

用法(任一 env,純 numpy/pandas):
  python scripts/merge_vllm.py --models qwen3vl,medgemma
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import CHEXPERT_LABELS, NUM_LABELS, LABEL_TO_INDEX, CXR_DATA_DIR

DATA_DIR = CXR_DATA_DIR


def ground_truth_vector(row) -> np.ndarray:
    gt = np.zeros(NUM_LABELS, dtype=float)
    for lbl in CHEXPERT_LABELS:
        if row.get(lbl) == 1.0:
            gt[LABEL_TO_INDEX[lbl]] = 1.0
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3vl,medgemma")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    models = [m.strip() for m in args.models.split(",")]

    arrs = {m: np.load(data_dir / f"vllm_{m}.npz", allow_pickle=True) for m in models}
    ref = arrs[models[0]]
    dicom_ids = ref["dicom_ids"]
    subsets = ref["subsets"]

    # 確認各模型的影像順序一致(都來自同一 manifest 切法)
    for m in models[1:]:
        assert np.array_equal(arrs[m]["dicom_ids"], dicom_ids), \
            f"{m} 的 dicom_ids 順序與 {models[0]} 不一致 —— 各模型須用相同 --n-cal/--n-test 跑"

    model_probs = np.stack([arrs[m]["probs"] for m in models], axis=1)  # (N, M, 14)
    model_boxes = np.stack([arrs[m]["boxes"] for m in models], axis=1)  # (N, M, 14, 4)
    disagreement = model_probs.std(axis=1)                              # (N, 14)
    ensemble = model_probs.mean(axis=1)                                 # (N, 14)

    man = pd.read_csv(data_dir / "subset_manifest.csv").drop_duplicates("dicom_id").set_index("dicom_id")
    gt = np.array([ground_truth_vector(man.loc[did]) for did in dicom_ids])

    out = data_dir / "inference_results.npz"
    np.savez(
        out,
        models=np.array(models),
        dicom_ids=dicom_ids,
        subsets=subsets,
        model_probs=model_probs,
        model_boxes=model_boxes,
        disagreement=disagreement,
        ground_truth=gt,
        ensemble_prob=ensemble,
    )
    print(f"合併 {len(models)} 模型 × {len(dicom_ids)} 張 → {out}")
    print(f"  models={models}  shape={model_probs.shape}")


if __name__ == "__main__":
    main()
