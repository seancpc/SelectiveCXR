"""
scripts/triage_distribution.py — 三檔分流分布診斷(診斷用,非系統元件)

目的:看 test set 在不同 (α, max_uncertain) 組合下 AUTO/FLAG/REFER 的分布,
用證據決定怎麼調參,讓 demo 三檔都展示得到、又不美化系統能力。

邏輯與 demo 完全一致:純 disagreement(跨模型 std)→ per-label conformal 門檻 →
能力內棄答數 n → n=0 AUTO、n≤max FLAG、n>max REFER。能力邊界(-inf)不計入 n。

用法(桌機,inference_results.npz 須存在):
  python scripts/triage_distribution.py
  python scripts/triage_distribution.py --data /path/to/inference_results.npz
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CHEXPERT_LABELS, CXR_DATA_DIR
from src.conformal.calibrator import ConformalCalibrator

DEFAULT_NPZ = CXR_DATA_DIR / "inference_results.npz"
ALPHAS = [0.05, 0.10, 0.15, 0.20]
MAXES = [3, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_NPZ))
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    subsets = d["subsets"]
    model_probs = d["model_probs"]                                # (N, M, 14)
    gt = d["ground_truth"]                                        # (N, 14)
    ens = d["ensemble_prob"]                                      # (N, 14)

    uncertainty = model_probs.std(axis=1)                        # (N, 14) disagreement
    is_error = ((ens > 0.5).astype(float) != gt).astype(float)   # (N, 14)

    cal = subsets == "calibration"
    test = subsets == "test"
    unc_test = uncertainty[test]
    n_test = int(test.sum())
    print(f"cal {int(cal.sum())} 張 | test {n_test} 張 | 訊號=純 disagreement\n")

    # 欄位用英文避免中文寬度對齊跑掉;bound=能力邊界數,avg_abst=平均能力內棄答數
    print(f"{'alpha':>6} {'max':>4} {'bound':>6} {'avg_abst':>9} "
          f"{'AUTO':>13} {'FLAG':>13} {'REFER':>13}")
    print("-" * 72)

    def pct(x):
        return f"{x} ({x / n_test * 100:.0f}%)"

    for alpha in ALPHAS:
        cc = ConformalCalibrator(alpha=alpha)
        cc.calibrate(uncertainty[cal], is_error[cal])
        thr_arr = np.array([cc.thresholds()[l] for l in CHEXPERT_LABELS])   # (14,) 含 -inf
        in_cap = thr_arr != float("-inf")                                   # (14,)
        n_boundary = int((~in_cap).sum())
        abst = unc_test > thr_arr                                           # (Ntest, 14)
        n_unc = (abst & in_cap).sum(axis=1)                                 # (Ntest,) 能力內棄答數
        avg = float(n_unc.mean())
        for max_u in MAXES:
            auto = int((n_unc == 0).sum())
            flag = int(((n_unc >= 1) & (n_unc <= max_u)).sum())
            refer = int((n_unc > max_u).sum())
            print(f"{alpha:>6.2f} {max_u:>4} {n_boundary:>6} {avg:>9.1f} "
                  f"{pct(auto):>13} {pct(flag):>13} {pct(refer):>13}")
        print()

    print("=== 各 α 下「能力內棄答數」histogram(重定義三檔分界用)===\n")
    for alpha in ALPHAS:
        cc = ConformalCalibrator(alpha=alpha)
        cc.calibrate(uncertainty[cal], is_error[cal])
        thr_arr = np.array([cc.thresholds()[l] for l in CHEXPERT_LABELS])
        in_cap = thr_arr != float("-inf")
        n_unc = ((unc_test > thr_arr) & in_cap).sum(axis=1)
        counts = np.bincount(n_unc, minlength=int(n_unc.max()) + 1)
        cum = 0
        print(f"α={alpha:.2f}  (能力內 {int(in_cap.sum())} label, 平均棄答 {n_unc.mean():.1f}):")
        for k, c in enumerate(counts):
            cum += c
            bar = "█" * round(c / n_test * 40)
            print(f"  棄答={k:2d} | {c:4d} 張 ({c / n_test * 100:4.1f}%) | 累積 {cum / n_test * 100:5.1f}%  {bar}")
        print()

    print("判讀(重定義三檔):高度自動(棄答≤lo)/ 部分標記(lo<棄答≤hi)/ 整案轉人工(棄答>hi)。")
    print("用上面『累積%』挑分界:累積到 lo = AUTO 占比、累積到 hi = AUTO+FLAG 占比。")
    print("把 histogram 貼回來,一起定 (α, lo, hi),再改 triage.py + config 重校 demo。")


if __name__ == "__main__":
    main()
