"""
scripts/compare_selective.py — CNN vs VLM ensemble vs 混合 的 selective 對比(真章)

核心問題:VLM 集成 / CNN+VLM 混合,在選擇性判讀上贏過傳統專用 CNN 嗎?
在 CNN 與 VLM 都涵蓋的 11 個 label 上公平比較三種「(主判斷, 不確定性訊號)」的
risk-coverage 表現:
  - 純 CNN          : pred=CNN>0.5,       unc=1-|2·prob-1|(距 0.5 多近)
  - 純 VLM ensemble : pred=VLM平均>0.5,    unc=跨 backbone std(disagreement)
  - CNN+VLM 混合     : pred=CNN>0.5(主判),  unc=|CNN-VLM|(兩範式不一致)

AURC(risk-coverage 曲線下面積)越低越好 = 同 coverage 下錯越少。

桌機跑(任一 env,純 numpy;需 inference_results.npz + cnn_test.npz):
  python scripts/compare_selective.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import LABEL_TO_INDEX, CXR_DATA_DIR

DATA = CXR_DATA_DIR


def risk_coverage(correct, uncertainty):
    """按 uncertainty 升序(最確定先納入)→ AURC = 各 coverage 的平均 risk。"""
    order = np.argsort(uncertainty, kind="stable")
    c = correct[order].astype(float)
    ks = np.arange(1, len(c) + 1)
    risks = 1.0 - np.cumsum(c) / ks
    return float(risks.mean())


def sel_acc_at(correct, uncertainty, cov):
    """coverage=cov 時的 selective accuracy(只自動判讀最確定的 cov 比例)。"""
    order = np.argsort(uncertainty, kind="stable")
    k = max(1, int(len(correct) * cov))
    return float(correct[order][:k].mean())


def auc(scores, labels):
    """rank-based AUC:scores 排序 labels==1 的能力,不依賴 threshold;單一類別→nan。"""
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    d = np.load(DATA / "inference_results.npz", allow_pickle=True)
    c = np.load(DATA / "cnn_test.npz", allow_pickle=True)

    test = d["subsets"] == "test"
    vlm_dids = d["dicom_ids"][test]
    vlm_ens = d["ensemble_prob"][test]               # (Nt, 14)
    vlm_std = d["model_probs"][test].std(axis=1)     # (Nt, 14) disagreement
    gt = d["ground_truth"][test]                     # (Nt, 14)

    # 對齊 CNN 到 VLM 的 dicom_id 順序
    cnn_idx = {str(x): i for i, x in enumerate(c["dicom_ids"])}
    sel = [cnn_idx[str(x)] for x in vlm_dids]
    cnn = c["cnn_prob"][sel]                          # (Nt, 14),非重疊 label = nan

    cov_labels = [str(x) for x in c["covered_labels"]]
    cidx = [LABEL_TO_INDEX[l] for l in cov_labels]

    cnn_p = cnn[:, cidx]
    vlm_p = vlm_ens[:, cidx]
    vlm_d = vlm_std[:, cidx]
    g = (gt[:, cidx] > 0.5)
    cnn_t = np.asarray(c["cnn_threshs"])[cidx]       # 每 label 的校準判定門檻(非 0.5)
    cnn_pred = cnn_p > cnn_t                           # 用 op_threshs 二值化(公平)

    settings = {
        # CNN 不確定性:prob 距「該 label 校準門檻」越近 = 越不確定(負距離,升序=最確定先)
        "純 CNN          ": (cnn_pred, -np.abs(cnn_p - cnn_t)),
        "純 VLM ensemble ": ((vlm_p > 0.5), vlm_d),
        "CNN+VLM 混合     ": (cnn_pred, np.abs(cnn_p - vlm_p)),
    }

    print(f"對比樣本: {g.shape[0]} 張 × {len(cidx)} label = {g.size} 個 (影像×label)")
    print(f"重疊 label: {cov_labels}\n")
    print(f"{'設定':18s} {'AURC↓':>8s} {'全答acc':>8s} {'selAcc@70%':>11s} {'selAcc@50%':>11s}")
    print("-" * 62)
    for name, (pred, unc) in settings.items():
        correct = (pred == g).flatten().astype(float)
        u = unc.flatten()
        print(f"{name} {risk_coverage(correct, u):8.4f} {correct.mean():8.3f} "
              f"{sel_acc_at(correct, u, 0.7):11.3f} {sel_acc_at(correct, u, 0.5):11.3f}")
    print("\nAURC 越低越好;selAcc@X% = 只自動判讀最確定的 X% 時的準確率")
    print("(註:上表 accuracy 受 threshold 影響大,僅參考;下表 AUC 才是公平的判斷品質對比)")

    print(f"\n{'='*48}\n判斷品質 per-label AUC(不依賴 threshold,越高越好)\n{'='*48}")
    print(f"{'label':28s} {'CNN':>7s} {'VLM':>7s}")
    cnn_aucs, vlm_aucs = [], []
    for j, lbl in enumerate(cov_labels):
        gj = g[:, j].astype(int)
        ac, av = auc(cnn_p[:, j], gj), auc(vlm_p[:, j], gj)
        cnn_aucs.append(ac)
        vlm_aucs.append(av)
        print(f"{lbl:28s} {ac:7.3f} {av:7.3f}")
    print("-" * 44)
    print(f"{'平均 AUC':28s} {np.nanmean(cnn_aucs):7.3f} {np.nanmean(vlm_aucs):7.3f}")
    print("→ CNN 是 CXR SOTA,預期 AUC 應 ≥ VLM;若 CNN AUC 合理(>0.7),"
          "代表之前 accuracy 崩確是 threshold 假象")


if __name__ == "__main__":
    main()
