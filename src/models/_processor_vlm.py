"""
src/models/_processor_vlm.py — processor 型 VLM 共用 adapter 基類

Qwen3-VL / MedGemma / Llama-3.2-Vision 三者都是 transformers 原生
AutoModelForImageTextToText + AutoProcessor,4-bit 載入與推論邏輯一致,
僅 model_id 不同(由 config.ENSEMBLE_MODELS 依 name 取得)。

各子類只需設定 `name`。若某模型的 chat template / image 餵入方式有差異
(例如 Llama-3.2-Vision 的 image 處理可能不同),覆寫 `_build_messages`
或 `predict` 即可。

需 GPU。
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

from config import ENSEMBLE_MODELS
from src.models.base import CXRModel, CXRPrediction
from src.models._prompting import build_prompt, parse_response

GEN_MAX_NEW_TOKENS = 1536  # 加大:容納 MedGemma thinking 模式 + JSON(qwen/llama 早結束不受影響)


class ProcessorVLMAdapter(CXRModel):
    """processor 型 VLM 共用 adapter。子類設定 name 即可。"""

    name = "base"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or ENSEMBLE_MODELS[self.name]
        self.processor = None
        self.model = None

    def load(self) -> None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _build_messages(self, image: Any, prompt: str) -> list:
        """預設 messages 格式(Qwen3-VL / MedGemma 適用)。Llama 若不同可覆寫。"""
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

    def predict(self, image: Any) -> CXRPrediction:
        if self.model is None:
            raise RuntimeError("請先呼叫 load()")
        messages = self._build_messages(image, build_prompt())
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=False,
            )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        probs, boxes = parse_response(text)
        return CXRPrediction(
            model_name=self.name,
            label_probs=probs,
            boxes=boxes,
            raw={"response": text},
        )

    def unload(self) -> None:
        del self.model, self.processor
        self.model = self.processor = None
        torch.cuda.empty_cache()
