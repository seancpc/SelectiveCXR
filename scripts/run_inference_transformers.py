"""
scripts/run_inference_transformers.py — transformers 版單模型推論(產 vLLM 同格式 npz)

給 vLLM 不支援的模型(如 Llama-3.2-Vision 的 Mllama 架構)用 transformers 4-bit 跑,
產出跟 run_inference_vllm 完全同格式的 vllm_{model}.npz → 可直接 merge_vllm / check_single。

慢(4-bit,每張數秒~數十秒);小批先驗證能力,健康再背景跑全量。
桌機跑(mars env,GPU):
  conda activate mars
  python scripts/run_inference_transformers.py --model llama32v --n-test 30   # 小批驗證
  python scripts/run_inference_transformers.py --model llama32v                # 全量(背景)
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from config import NUM_LABELS, LABEL_TO_INDEX, CXR_DATA_DIR
from src.models.llama32v import Llama32VAdapter

DATA = CXR_DATA_DIR
ADAPTERS = {"llama32v": Llama32VAdapter}


def pred_to_arrays(pred):
    """CXRPrediction → (probs (14,), boxes (14,4) nan-filled),同 vLLM npz 格式。"""
    probs = np.asarray(pred.label_probs, dtype=float).reshape(-1)
    boxes = np.full((NUM_LABELS, 4), np.nan, dtype=float)
    for finding, blist in pred.boxes.items():
        if blist:
            b = blist[0]
            boxes[LABEL_TO_INDEX[finding]] = [b.x_min, b.y_min, b.x_max, b.y_max]
    return probs, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--n-cal", type=int, default=None,
                    help="cal 張數:0=不跑、N=前 N、省略=全部")
    ap.add_argument("--n-test", type=int, default=None,
                    help="test 張數:0=不跑、N=前 N、省略=全部")
    args = ap.parse_args()

    man = pd.read_csv(DATA / "subset_manifest.csv")
    cal = man[man["subset"] == "calibration"]
    test = man[man["subset"] == "test"]
    if args.n_cal is not None:
        cal = cal.head(args.n_cal)       # 0 → head(0) → 空(不跑 cal)
    if args.n_test is not None:
        test = test.head(args.n_test)
    df = pd.concat([cal, test], ignore_index=True)
    exists = df["rel_path"].apply(lambda p: (DATA / p).exists())
    df = df[exists].reset_index(drop=True)
    print(f"{args.model}: {len(df)} 張")

    model = ADAPTERS[args.model]()
    print("載入模型(transformers 4-bit)...")
    model.load()

    probs_all, boxes_all = [], []
    t0 = time.time()
    for i, row in df.iterrows():
        img = Image.open(DATA / row["rel_path"]).convert("RGB")
        img.thumbnail((1024, 1024))
        p, b = pred_to_arrays(model.predict(img))
        probs_all.append(p)
        boxes_all.append(b)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(df)}  ({(time.time() - t0) / (i + 1):.1f}s/張)")

    out = DATA / f"vllm_{args.model}.npz"
    np.savez(out, probs=np.array(probs_all), boxes=np.array(boxes_all),
             dicom_ids=df["dicom_id"].values, subsets=df["subset"].values)
    print(f"\n存檔 → {out}  ({len(df)} 張, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
