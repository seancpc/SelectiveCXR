"""
scripts/make_demo_manifest.py — 為公開 demo 影像生成 subset_manifest.csv

掃描 <data-dir>/files/ 下所有影像,產生 demo 用的 manifest(全部標為 test、
無 ground truth)。搭配可自由散布的公開 CXR(如 NIH ChestX-ray14)使用,
讓 demo / 錄影 / 截圖可合法公開,完全不含 MIMIC 影像。

公開展示只需「三模型判讀 → 分歧 → 分流」的行為,不需對照答案,故 ground truth
留空(merge 時自動填 0,demo 也不顯示)。棄答門檻沿用 demo/calibrator.pkl
(在 MIMIC 校準),屬「展示系統行為」,非「保證此資料集上的 α"。

用法(桌機):
  # 1. 把公開影像放到 <data-dir>/files/ 底下(.png/.jpg)
  # 2. python scripts/make_demo_manifest.py --data-dir /mnt/d/mars_data/demo_public
"""

import argparse
from pathlib import Path

import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="公開 demo 資料根(其下需有 files/ 放影像)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files_dir = data_dir / "files"
    if not files_dir.is_dir():
        raise SystemExit(f"找不到 {files_dir} —— 請先把公開 CXR 影像放進 <data-dir>/files/")

    imgs = sorted(p for p in files_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"{files_dir} 底下沒有影像(.png/.jpg)")

    rows = [
        {
            "dicom_id": p.stem,                                  # 用檔名當 id(非 MIMIC dicom_id)
            "subset": "test",                                    # 全部歸 test
            "rel_path": p.relative_to(data_dir).as_posix(),      # files/xxx.png
        }
        for p in imgs
    ]
    df = pd.DataFrame(rows)
    out = data_dir / "subset_manifest.csv"
    df.to_csv(out, index=False)
    print(f"生成 {out}({len(df)} 張,全部標為 test、無 ground truth)")


if __name__ == "__main__":
    main()
