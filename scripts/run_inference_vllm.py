"""
scripts/run_inference_vllm.py — vLLM 批量推論(單模型,guided decoding 強制 JSON)

一次跑一個模型(避免 vLLM 同 process 多實例釋放問題),用 continuous batching
對所有影像批量推論。存單模型結果 vllm_{model}.npz;三模型各跑一遍後由
merge 步驟合併成 inference_results.npz(供 run_conformal)。

要點(均已在 test_vllm 驗證):
  - 縮圖 1024(避免 vision token 吃光 max_model_len)
  - structured_outputs JSON schema(xgrammar 強制合法 JSON,免容錯 parse、MedGemma 不 thinking)
  - flashinfer sampler 禁用(WSL 無 nvcc)

用法(mars_vllm env):
  conda activate mars_vllm
  python scripts/run_inference_vllm.py --model qwen3vl --n-cal 10 --n-test 10   # 小批測速
  python scripts/run_inference_vllm.py --model qwen3vl                          # 全量
"""

import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import sys
import argparse
import base64
import io
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    from vllm import StructuredOutputsParams

from config import CHEXPERT_LABELS, NUM_LABELS, LABEL_TO_INDEX, CXR_DATA_DIR

DATA_DIR = CXR_DATA_DIR
MODELS = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "medgemma": "google/medgemma-1.5-4b-it",
    "llama32v": "meta-llama/Llama-3.2-11B-Vision-Instruct",   # ⚠️ vLLM 0.22 不支援 Mllama 架構
    "phi35v": "microsoft/Phi-3.5-vision-instruct",            # ❌ 棄用:對 CXR 填模板(72% 全0,ReXVQA 僅 47%)
    "internvl3": "OpenGVLab/InternVL3-9B-AWQ",                # ❌ vLLM 0.22 載入失敗(intern_vit 權重 KeyError)
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",                # 保底第三模型(vLLM 最成熟,CXR ReXVQA 65%;與 qwen3vl 同源)
    "pixtral": "mistralai/Pixtral-12B-2409",                  # ❌ vLLM image processor 與 transformers 5.x 不相容(fetch_images)
}
# 第三模型探索結論(2026-06-14):非同源候選在 vLLM 0.22+transformers 5.x 都撞版本牆
#   InternVL3-AWQ → AWQ fused-QKV KeyError;Pixtral → image processor fetch_images 不相容
#   → 退回 qwen3vl + medgemma 兩模型(乾淨有效的跨 backbone ensemble)


def build_schema() -> dict:
    finding = {
        "type": "object",
        "properties": {
            "prob": {"type": "number"},
            "box": {"type": ["array", "null"], "items": {"type": "number"}},
        },
        "required": ["prob", "box"],
    }
    return {
        "type": "object",
        "properties": {lbl: finding for lbl in CHEXPERT_LABELS},
        "required": list(CHEXPERT_LABELS),
    }


def build_prompt() -> str:
    return (
        "Assess each of these 14 chest X-ray findings for visibility: "
        + ", ".join(CHEXPERT_LABELS)
        + ". For each output prob (0.0-1.0 it is visible) and a normalized box "
        "[x_min,y_min,x_max,y_max] (each 0-1) if prob>0.5, else null."
    )


def to_messages(path: Path) -> list:
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024))   # 減少 vision token
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": build_prompt()},
        ],
    }]


def parse_vec(text: str):
    probs = np.zeros(NUM_LABELS, dtype=float)
    boxes = np.full((NUM_LABELS, 4), np.nan, dtype=float)   # 每 finding 一個 box(nan=無)
    try:
        d = json.loads(text)
    except Exception:
        return probs, boxes
    if not isinstance(d, dict):
        return probs, boxes
    for lbl, i in LABEL_TO_INDEX.items():
        e = d.get(lbl)
        if not isinstance(e, dict):
            continue
        if isinstance(e.get("prob"), (int, float)):
            probs[i] = float(np.clip(e["prob"], 0.0, 1.0))
        box = e.get("box")
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            boxes[i] = [float(np.clip(v, 0.0, 1.0)) for v in box]
    return probs, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n-cal", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=0)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    man = pd.read_csv(data_dir / "subset_manifest.csv")
    cal = man[man["subset"] == "calibration"]
    test = man[man["subset"] == "test"]
    if args.n_cal:
        cal = cal.head(args.n_cal)
    if args.n_test:
        test = test.head(args.n_test)
    df = pd.concat([cal, test], ignore_index=True)
    # 過濾未下載的影像(cal 拉大時,部分影像可能沒下到,避免 Image.open 失敗)
    exists = df["rel_path"].apply(lambda p: (data_dir / p).exists())
    n_missing = int((~exists).sum())
    df = df[exists].reset_index(drop=True)
    print(f"{args.model}: {len(df)} 張可用"
          + (f"(略過 {n_missing} 張未下載)" if n_missing else ""))

    print("準備 messages(縮圖 + b64)...")
    messages_list = [to_messages(data_dir / row["rel_path"]) for _, row in df.iterrows()]

    sp = SamplingParams(
        max_tokens=1536,
        temperature=0.0,
        structured_outputs=StructuredOutputsParams(json=build_schema()),
    )
    llm_kwargs = dict(
        model=MODELS[args.model],
        limit_mm_per_prompt={"image": 1},
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    if args.model == "pixtral":
        # Pixtral 12B:需 mistral tokenizer;fp8 動態量化壓到 ~12GB(4090 支援 fp8);縮 KV 省顯存
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["quantization"] = "fp8"
        llm_kwargs["max_model_len"] = 4096
    llm = LLM(**llm_kwargs)

    t0 = time.time()
    outputs = llm.chat(messages_list, sampling_params=sp)
    dt = time.time() - t0

    parsed = [parse_vec(o.outputs[0].text) for o in outputs]
    probs = np.array([p for p, _ in parsed])    # (N, 14)
    boxes = np.array([b for _, b in parsed])    # (N, 14, 4)

    out = data_dir / f"vllm_{args.model}.npz"
    np.savez(
        out,
        probs=probs,
        boxes=boxes,
        dicom_ids=df["dicom_id"].values,
        subsets=df["subset"].values,
    )
    print(f"\n存檔 → {out}  ({len(df)} 張, {dt:.1f}s, {dt/len(df):.2f}s/張)")


if __name__ == "__main__":
    main()
