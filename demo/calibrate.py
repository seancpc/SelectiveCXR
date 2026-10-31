"""
demo/calibrate.py — 預校準 conformal 棄答門檻,供 demo 載入即用

讀 inference_results.npz 的 calibration 子集,用純跨模型 disagreement
不確定性校準 per-label 棄答門檻,
把校準好的 ConformalCalibrator pickle 存成 demo/calibrator.pkl。
demo app 啟動時載入即用,免每次重校。

用法(桌機,任一 env,純 numpy/pandas;inference_results.npz 須已存在):
  python demo/calibrate.py
  python demo/calibrate.py --alpha 0.05 --hoeffding
"""

import sys
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CHEXPERT_LABELS, CONFORMAL_ALPHA, CXR_DATA_DIR
from src.conformal.calibrator import ConformalCalibrator

DEFAULT_NPZ = CXR_DATA_DIR / "inference_results.npz"
DEFAULT_OUT = Path(__file__).resolve().parent / "calibrator.pkl"


def fuse_uncertainty(model_probs: np.ndarray):
    """純 disagreement(跨模型機率 std)。

    grounding(定義 A 解剖 / 定義 B 跨模型 IoU)經 AUC 驗證均無區辨力,已移除融合 ——
    系統最終的不確定性訊號就是跨 backbone disagreement。回傳 uncertainty (N, 14)。
    """
    return model_probs.std(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_NPZ))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--alpha", type=float, default=CONFORMAL_ALPHA)
    ap.add_argument("--hoeffding", action="store_true",
                    help="用 Hoeffding 上界(統計保證,需大 cal);預設 empirical")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    subsets = d["subsets"]
    model_probs = d["model_probs"]       # (N, M, 14)
    gt = d["ground_truth"]               # (N, 14)
    ens = d["ensemble_prob"]             # (N, 14)

    uncertainty = fuse_uncertainty(model_probs)
    is_error = ((ens > 0.5).astype(float) != gt).astype(float)

    cal = subsets == "calibration"
    print(f"calibration {int(cal.sum())} 張 | 訊號=純 disagreement "
          f"| alpha={args.alpha} | 模式={'Hoeffding' if args.hoeffding else 'empirical'}")

    cc = ConformalCalibrator(alpha=args.alpha, use_hoeffding=args.hoeffding)
    cc.calibrate(uncertainty[cal], is_error[cal])

    with open(args.out, "wb") as f:
        pickle.dump(cc, f)
    print(f"存檔 → {args.out}\n")

    thr = cc.thresholds()
    print(f"{'finding':28s} {'門檻':>8s}")
    print("-" * 38)
    for lbl in CHEXPERT_LABELS:
        t = thr[lbl]
        print(f"{lbl:28s} {'-inf' if t == float('-inf') else f'{t:.3f}':>8s}")


if __name__ == "__main__":
    main()
