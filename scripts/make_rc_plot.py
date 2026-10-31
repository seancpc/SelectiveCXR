"""
scripts/make_rc_plot.py — 從真實資料產生 Risk–Coverage 曲線 SVG(選擇性預測招牌圖)

讀 inference_results.npz(test 子集)+ demo/calibrator.pkl,
以「能力內 label 的跨模型 disagreement」為不確定性、以 ensemble>0.5 是否等於 GT 為正確,
用 eval.metrics 算 Risk–Coverage 曲線與 AURC,輸出深色主題 SVG 到 docs/risk-coverage.svg。

RC 曲線是選擇性預測領域最一眼認得的招牌圖:x=coverage(自動判讀比例)、y=risk(該子集錯誤率);
曲線越貼左下、AURC 越低 = 棄答策略越好。標出 α=0.05 / 0.15 參考線與對應操作點。

用法(桌機,MARS_CXR_DIR 指向含 inference_results.npz 的資料夾):
  python scripts/make_rc_plot.py
  python scripts/make_rc_plot.py --data /path/to/inference_results.npz --out docs/risk-coverage.svg
"""

import sys
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CHEXPERT_LABELS, CXR_DATA_DIR
from eval.metrics import risk_coverage_curve, aurc

DEFAULT_NPZ = CXR_DATA_DIR / "inference_results.npz"
DEFAULT_PKL = Path(__file__).resolve().parent.parent / "demo" / "calibrator.pkl"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs" / "risk-coverage.svg"

# 深色主題(對齊 GitHub Pages)
BG, AXIS, GRID, CURVE, REF, TEXT, MUTED = (
    "#1e1e1e", "#888", "#333", "#00bcd4", "#ff9800", "#e0e0e0", "#9aa0a6")


def build_svg(cov, risk, aurc_val, ops):
    """cov/risk: 遞增覆蓋率與對應風險;ops: [(alpha, coverage_at_alpha), ...]。"""
    W, H = 720, 472
    x0, x1, y0, y1 = 70, 690, 68, 380            # 繪圖區(y0 上緣下移,上方留 header 帶,標題/副標不與圖重疊)
    ymax = max(0.3, float(np.ceil(risk.max() * 10) / 10))   # y 軸上限

    def X(c): return x0 + c * (x1 - x0)
    def Y(r): return y0 + (r / ymax) * (y1 - y0)

    # 降採樣到 ~240 點,SVG 才不肥
    n = len(cov)
    idx = np.linspace(0, n - 1, min(n, 240)).astype(int)
    pts = " ".join(f"{X(cov[i]):.1f},{Y(risk[i]):.1f}" for i in idx)

    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="Arial, sans-serif"><rect width="{W}" height="{H}" rx="8" fill="{BG}"/>']
    s.append(f'<text x="20" y="26" fill="{CURVE}" font-size="16" font-weight="bold">'
             f'Risk–Coverage 曲線(選擇性判讀,test 子集)</text>')
    s.append(f'<text x="20" y="44" fill="{MUTED}" font-size="12">'
             f'x=自動判讀比例 · y=該子集錯誤率 · 曲線越貼左下越好 · AURC={aurc_val:.3f}(越低越好)</text>')
    s.append(f'<line x1="20" y1="56" x2="{W-20}" y2="56" stroke="{GRID}" stroke-width="1" opacity="0.6"/>')  # header 分隔線

    # 座標軸 + 格線
    for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s.append(f'<line x1="{X(g):.1f}" y1="{y0}" x2="{X(g):.1f}" y2="{y1}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{X(g):.1f}" y="{y1+18}" fill="{MUTED}" font-size="11" text-anchor="middle">{g:.2f}</text>')
    ny = 4
    for k in range(ny + 1):
        r = ymax * k / ny
        s.append(f'<line x1="{x0}" y1="{Y(r):.1f}" x2="{x1}" y2="{Y(r):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x0-8}" y="{Y(r)+4:.1f}" fill="{MUTED}" font-size="11" text-anchor="end">{r:.2f}</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-6}" fill="{TEXT}" font-size="12" text-anchor="middle">coverage(自動判讀比例)</text>')
    s.append(f'<text x="16" y="{(y0+y1)/2:.0f}" fill="{TEXT}" font-size="12" text-anchor="middle" '
             f'transform="rotate(-90 16 {(y0+y1)/2:.0f})">risk(錯誤率)</text>')

    # α 參考線 + 操作點
    for alpha, cov_a in ops:
        s.append(f'<line x1="{x0}" y1="{Y(alpha):.1f}" x2="{x1}" y2="{Y(alpha):.1f}" '
                 f'stroke="{REF}" stroke-width="1" stroke-dasharray="5 4" opacity="0.8"/>')
        s.append(f'<text x="{x1-4}" y="{Y(alpha)-4:.1f}" fill="{REF}" font-size="11" text-anchor="end">α={alpha:.2f}</text>')
        if not np.isnan(cov_a):
            s.append(f'<circle cx="{X(cov_a):.1f}" cy="{Y(alpha):.1f}" r="4" fill="{REF}"/>')
            s.append(f'<text x="{X(cov_a):.1f}" y="{Y(alpha)+18:.1f}" fill="{REF}" font-size="10" '
                     f'text-anchor="middle">cov={cov_a:.2f}</text>')

    # RC 曲線
    s.append(f'<polyline points="{pts}" fill="none" stroke="{CURVE}" stroke-width="2.2"/>')
    s.append('</svg>')
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_NPZ))
    ap.add_argument("--calibrator", default=str(DEFAULT_PKL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    subsets = d["subsets"]
    model_probs = d["model_probs"]        # (N, M, 14)
    gt = d["ground_truth"]                # (N, 14)
    ens = d["ensemble_prob"]              # (N, 14)

    with open(args.calibrator, "rb") as f:
        cc = pickle.load(f)
    thr = cc.thresholds()
    in_cap = np.array([thr[l] != float("-inf") for l in CHEXPERT_LABELS])   # (14,)

    test = subsets == "test"
    unc = model_probs.std(axis=1)                              # (N,14) disagreement
    is_correct = ((ens > 0.5).astype(float) == gt).astype(float)
    # 攤平 test × 能力內 label
    U = unc[test][:, in_cap].reshape(-1)
    C = is_correct[test][:, in_cap].reshape(-1)

    cov, risk = risk_coverage_curve(C, U)
    a = aurc(C, U)

    # 各 α 下「風險仍 ≤ α」能達到的最大 coverage(操作點)
    ops = []
    for alpha in (0.05, 0.15):
        ok = np.where(risk <= alpha)[0]
        ops.append((alpha, float(cov[ok[-1]]) if len(ok) else float("nan")))

    svg = build_svg(cov, risk, a, ops)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(f"AURC = {a:.4f}(越低越好) | test×能力內 label = {len(C)} 個決策")
    for alpha, c in ops:
        print(f"  風險 ≤ {alpha:.2f} 時,最大 coverage = {c:.3f}" if not np.isnan(c)
              else f"  風險 ≤ {alpha:.2f}:無法達成(最低風險 {risk.min():.3f})")
    print(f"存檔 → {args.out}")


if __name__ == "__main__":
    main()
