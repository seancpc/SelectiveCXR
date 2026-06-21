"""
scripts/run_cnn_baseline.py — CNN baseline 推論(CNN vs VLM 對比實驗)

用 torchxrayvision pretrained DenseNet(傳統專用 CNN)對 test 影像跑 CXR 分類,
對照到 CheXpert 14 label,存 cnn_test.npz。供 compare_selective.py 做
CNN vs VLM ensemble vs 混合 的 selective 對比。

先印出 xrv pathologies 與 CheXpert 的對照 + 第一張的輸出範圍(sanity check),
確認對照正確、輸出確實是 0-1 機率,再繼續寫對比腳本。

桌機跑(mars env,GPU):
  python scripts/run_cnn_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sys as _sys
_saved = list(_sys.path)
import torchxrayvision as xrv      # 同 segmenter:還原 sys.path 避免 config 污染
_sys.path[:] = _saved

import numpy as np
import torch
import pandas as pd
from PIL import Image

from config import CHEXPERT_LABELS, NUM_LABELS, LABEL_TO_INDEX, CXR_DATA_DIR

DATA = CXR_DATA_DIR

# xrv DenseNet pathology → CheXpert label(重疊者;名稱差異在此對齊)
XRV_TO_CHEXPERT = {
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Cardiomegaly":               "Cardiomegaly",
    "Lung Opacity":               "Lung Opacity",
    "Lung Lesion":                "Lung Lesion",
    "Edema":                      "Edema",
    "Consolidation":              "Consolidation",
    "Pneumonia":                  "Pneumonia",
    "Atelectasis":                "Atelectasis",
    "Pneumothorax":               "Pneumothorax",
    "Effusion":                   "Pleural Effusion",
    "Fracture":                   "Fracture",
}


def main():
    model = xrv.models.DenseNet(weights="densenet121-res224-all").cuda().eval()
    paths = list(model.pathologies)
    print("xrv DenseNet pathologies → CheXpert 對照:")
    for i, p in enumerate(paths):
        tgt = XRV_TO_CHEXPERT.get(p)
        print(f"  [{i:2d}] {p:30s} {'→ ' + tgt if tgt else '(無對應,略過)'}")

    xrv_to_cidx = {i: LABEL_TO_INDEX[XRV_TO_CHEXPERT[p]]
                   for i, p in enumerate(paths) if p in XRV_TO_CHEXPERT}

    man = pd.read_csv(DATA / "subset_manifest.csv")
    test = man[man["subset"] == "test"].reset_index(drop=True)
    resizer = xrv.datasets.XRayResizer(224)

    cnn = np.full((len(test), NUM_LABELS), np.nan, dtype=float)
    dids = []
    for i, row in test.iterrows():
        img = np.array(Image.open(DATA / row["rel_path"]).convert("L"), dtype=np.float32)
        img = xrv.datasets.normalize(img, 255)
        img = resizer(img[None, ...])
        x = torch.from_numpy(img)[None, ...].cuda()
        with torch.no_grad():
            out = model(x)[0].cpu().numpy()        # (n_path,) — 應為 0-1 機率
        if i == 0:
            print(f"\n[sanity] 第一張輸出範圍: min={out.min():.3f} max={out.max():.3f} "
                  f"(應落在 0-1;若超出代表是 logits,需 sigmoid)")
        for xi, ci in xrv_to_cidx.items():
            cnn[i, ci] = float(out[xi])
        dids.append(row["dicom_id"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test)}")

    # xrv 每個 pathology 有校準判定門檻(op_threshs),不是 0.5 —— 用它才公平
    op = model.op_threshs.detach().cpu().numpy().astype(float)
    cnn_threshs = np.full(NUM_LABELS, 0.5)
    print("\nxrv op_threshs(校準判定門檻,非 0.5):")
    for xi, ci in xrv_to_cidx.items():
        t = float(op[xi])
        cnn_threshs[ci] = t if not np.isnan(t) else 0.5
        print(f"  {paths[xi]:30s} op_thresh={t:.3f}")

    covered = [XRV_TO_CHEXPERT[p] for p in paths if p in XRV_TO_CHEXPERT]
    np.savez(DATA / "cnn_test.npz", cnn_prob=cnn, dicom_ids=np.array(dids),
             covered_labels=np.array(covered), cnn_threshs=cnn_threshs)
    print(f"\n存檔 → cnn_test.npz  ({len(test)} 張, 對照 {len(xrv_to_cidx)} 個 CheXpert label)")
    print(f"重疊 label: {covered}")


if __name__ == "__main__":
    main()
