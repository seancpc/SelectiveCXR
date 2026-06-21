"""
scripts/check_single.py — 加 ensemble 前必跑:驗證單一模型的輸出品質

讀 vllm_{model}.npz(run_inference_vllm 小批測的輸出),檢查是否像 phi 一樣填模板:
  - 每張平均 prob 是否雙峰(大量全 0)
  - prob 是否只有少數固定值
  - bbox 是否集中在固定框(如 phi 的 [0,0,0.5,0.5])

★ 制度化教訓:phi 就是沒先驗證單模型品質才混進 ensemble。任何新成員先過這關。

桌機跑(先 run_inference_vllm --model X --n-test 30 產生 vllm_X.npz):
  python scripts/check_single.py --model internvl3
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CXR_DATA_DIR

import numpy as np

DATA = CXR_DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    d = np.load(DATA / f"vllm_{args.model}.npz", allow_pickle=True)
    pp = d["probs"]                                  # (N, 14)
    bb = d["boxes"]                                  # (N, 14, 4)
    N = pp.shape[0]
    per = pp.mean(axis=1)
    vals = np.round(pp.flatten(), 2)
    n_unique = len(set(vals.tolist()))
    boxes = bb.reshape(-1, 4)
    valid = boxes[~np.isnan(boxes).any(axis=1)]
    bk = [tuple(np.round(b, 2).tolist()) for b in valid]

    print(f"\n{args.model}: {N} 張")
    print(f"  每張平均 prob: 近0(<0.05) {int((per < 0.05).sum())} | "
          f"高(>0.5) {int((per > 0.5).sum())} | 中位 {np.median(per):.3f}")
    print(f"  prob unique 值數: {n_unique}")
    print(f"    最常見 prob: {Counter(vals.tolist()).most_common(6)}")
    print(f"  有 box {len(valid)} | box unique {len(set(bk))}")
    print(f"    最常見 box: {Counter(bk).most_common(3)}")

    # 自動判定
    全0比例 = (per < 0.05).sum() / N
    box_集中 = (Counter(bk).most_common(1)[0][1] / max(len(valid), 1)) if valid.size else 0
    print(f"\n{'判定':=^40}")
    bad = []
    if 全0比例 > 0.4:
        bad.append(f"全0影像比例 {全0比例:.0%} 過高(像 phi)")
    if n_unique <= 6:
        bad.append(f"prob 只有 {n_unique} 種值(像填模板)")
    if box_集中 > 0.4:
        bad.append(f"box {box_集中:.0%} 集中單一框(像填模板)")
    if bad:
        print("❌ 疑似填模板,不建議納入 ensemble:")
        for b in bad:
            print(f"   - {b}")
    else:
        print("✅ 輸出健康(prob 多樣、box 分散、無大量全0),可納入 ensemble")


if __name__ == "__main__":
    main()
