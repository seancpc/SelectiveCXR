"""
scripts/run_conformal.py — 讀 inference_results.npz → conformal 校準 → triage → 評測

整條「選擇性判讀」pipeline 的最後一段:
  - 用 calibration 子集校準 conformal 棄答門檻(empirical selective risk,適配 pilot 規模)
  - 對 test 子集逐 label 決定棄答,算選擇性判讀評測

不確定性訊號:用三/兩模型機率的「標準差」(連續,粒度比 vote 細)。
ensemble 預測:模型平均機率 > 0.5。is_error = 預測 != ground truth。

用法(於專案根目錄,inference_results.npz 已存在):
  python scripts/run_conformal.py
  python scripts/run_conformal.py --alpha 0.10
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CHEXPERT_LABELS, CONFORMAL_ALPHA, CXR_DATA_DIR
from src.conformal.calibrator import ConformalCalibrator
from src.uncertainty.grounding_consistency import grounding_inconsistency_batch
from eval.metrics import (
    coverage,
    selective_accuracy,
    conformal_empirical_coverage,
    abstention_precision,
)

DEFAULT_NPZ = CXR_DATA_DIR / "inference_results.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_NPZ))
    ap.add_argument("--alpha", type=float, default=CONFORMAL_ALPHA)
    ap.add_argument("--hoeffding", action="store_true",
                    help="用 Hoeffding 上界(distribution-free 統計保證,需大 cal);預設 empirical")
    ap.add_argument("--grounding", action="store_true",
                    help="融合 grounding 一致性訊號(跨模型 bbox IoU)到不確定性")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    subsets = d["subsets"]
    model_probs = d["model_probs"]       # (N, M, 14)
    gt = d["ground_truth"]               # (N, 14)
    ens = d["ensemble_prob"]             # (N, 14)

    # 不確定性 = 模型間機率標準差(disagreement,連續)
    disagreement = model_probs.std(axis=1)   # (N, 14)
    if args.grounding:
        # 融合 grounding 不一致性(跨模型 bbox IoU);無足夠 box 的 finding 維持 disagreement
        grounding = grounding_inconsistency_batch(d["model_boxes"])   # (N, 14),含 nan
        n_grounded = int((~np.isnan(grounding)).sum())
        uncertainty = np.where(np.isnan(grounding), disagreement,
                               (disagreement + grounding) / 2.0)
        print(f"grounding 融合:{n_grounded} 個 (影像×finding) 有 bbox 一致性訊號")
    else:
        uncertainty = disagreement
    # ensemble 預測與錯誤
    pred = (ens > 0.5).astype(float)
    is_error = (pred != gt).astype(float)

    cal = subsets == "calibration"
    test = subsets == "test"
    print(f"calibration {int(cal.sum())} + test {int(test.sum())} 張 | "
          f"模型數={model_probs.shape[1]} | alpha={args.alpha}")

    # 校準(empirical,適配 pilot)
    cc = ConformalCalibrator(alpha=args.alpha, use_hoeffding=args.hoeffding)
    cc.calibrate(uncertainty[cal], is_error[cal])
    print(f"校準模式: {'Hoeffding 統計保證' if args.hoeffding else 'empirical'}")

    # test 棄答遮罩
    u_test = uncertainty[test]
    err_test = is_error[test]
    abstain = np.array([cc.abstain_mask(u_test[i]) for i in range(u_test.shape[0])])  # (n, 14)

    ab = abstain.flatten()
    err = err_test.flatten()
    correct = 1.0 - err

    print(f"\n{'='*60}\n選擇性判讀評測(test 子集,per-label 決策)\n{'='*60}")
    print(f"  coverage(自動判讀比例)         : {coverage(ab):.3f}")
    print(f"  selective accuracy(不棄答準確) : {selective_accuracy(correct, ab):.3f}")
    print(f"  conformal 實際錯誤率(目標 ≤{args.alpha}) : {conformal_empirical_coverage(err, ab):.3f}")
    print(f"  abstention precision(棄答真錯率): {abstention_precision(err, ab):.3f}")
    print(f"  [對照] 全部都答的準確率        : {correct.mean():.3f}")

    print(f"\n{'finding':28s} {'門檻':>8s} {'棄答率':>8s} {'陽性GT':>8s}")
    print("-" * 56)
    thr = cc.thresholds()
    for i, lbl in enumerate(CHEXPERT_LABELS):
        t = thr[lbl]
        t_str = "-inf" if t == float("-inf") else f"{t:.3f}"
        print(f"{lbl:28s} {t_str:>8s} {abstain[:, i].mean():8.2f} {gt[test][:, i].mean():8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
