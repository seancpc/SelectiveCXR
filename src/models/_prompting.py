"""
src/models/_prompting.py — 策略 A 共用:結構化 prompt 建構 + 容錯 parse

三個 processor 型 adapter(Qwen3-VL / MedGemma / Llama-3.2-V)共用同一套
prompt 與輸出解析,確保它們對同一張影像產生「可比較」的 14-label + bbox。
容錯解析理念移植自工作專案 MRRA 的 JSON 救援(模型偶爾格式跑掉也能救)。

不需 GPU。
"""

from __future__ import annotations

import json
import re

import numpy as np

from config import CHEXPERT_LABELS, NUM_LABELS, LABEL_TO_INDEX


def build_prompt() -> str:
    """要求 VLM 對 14 個 CheXpert finding 輸出 prob + bbox 的 JSON。"""
    labels = ", ".join(CHEXPERT_LABELS)
    example = (
        '{"Cardiomegaly": {"prob": 0.82, "box": [0.30, 0.55, 0.70, 0.88]}, '
        '"Pneumothorax": {"prob": 0.05, "box": null}, '
        '"No Finding": {"prob": 0.10, "box": null}}'
    )
    return (
        "Structured data-annotation task on a public de-identified research chest X-ray "
        "dataset (MIMIC-CXR). NOT clinical diagnosis; will not affect any patient.\n\n"
        f"For EACH of these 14 findings, assess visibility in the image: {labels}.\n\n"
        "Output for each finding: the probability (0.0-1.0) it is visible, and if "
        "probability > 0.5 a bounding box [x_min, y_min, x_max, y_max] in NORMALIZED "
        "coordinates where EVERY value is between 0.0 and 1.0 (a fraction of image "
        "width/height), otherwise null.\n\n"
        "Respond with ONLY a single JSON object and nothing else — no explanation, do NOT "
        "repeat this task description. Example of the exact format (values illustrative):\n"
        f"{example}\n\n"
        "Output the JSON object IMMEDIATELY with NO reasoning, NO thinking steps, NO "
        "explanation — output only the JSON, covering all 14 findings using the exact "
        "finding names listed above:"
    )


def _extract_json(text: str) -> dict | None:
    """抽出最後一個可解析的 JSON 物件(避開 thinking 段落內的雜散括號)。"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    # 以括號平衡掃出所有頂層 {...} 候選
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
    # 從最後一個(通常是 thinking 之後的最終輸出)往前試
    for blob in reversed(candidates):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
            except json.JSONDecodeError:
                continue
    return None


def parse_response(text: str):
    """解析 VLM 輸出 → (label_probs (NUM_LABELS,), boxes dict)。

    容錯:整段 JSON 壞掉 → 全 0、無 box;個別 label 缺失 → 該 label prob 0。
    回傳的 boxes:finding 名稱 -> [BoundingBox]。
    """
    # 延遲 import 避免循環依賴
    from src.models.base import BoundingBox

    probs = np.zeros(NUM_LABELS, dtype=float)
    boxes: dict[str, list] = {}

    data = _extract_json(text)
    if not isinstance(data, dict):
        return probs, boxes

    for label, idx in LABEL_TO_INDEX.items():
        entry = data.get(label)
        if not isinstance(entry, dict):
            continue
        p = entry.get("prob")
        if isinstance(p, (int, float)):
            probs[idx] = float(np.clip(p, 0.0, 1.0))
        box = entry.get("box")
        if (isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) for v in box)):
            x1, y1, x2, y2 = (float(np.clip(v, 0.0, 1.0)) for v in box)
            if x2 > x1 and y2 > y1:
                boxes[label] = [BoundingBox(x1, y1, x2, y2)]

    return probs, boxes
