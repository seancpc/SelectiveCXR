"""
src/models/llama32v.py — Llama-3.2-Vision adapter(Llama backbone)。

Llama-3.2-Vision(Mllama)的 image 餵入方式與 Qwen/MedGemma 不同:
image 不放在 messages 內容裡,而是 apply_chat_template 只產文字(image 為
placeholder {"type":"image"}),再由 processor(image, text) 一起處理。故覆寫 predict。
"""

from __future__ import annotations

import torch

from src.models._processor_vlm import ProcessorVLMAdapter, GEN_MAX_NEW_TOKENS
from src.models._prompting import build_prompt, parse_response
from src.models.base import CXRPrediction


class Llama32VAdapter(ProcessorVLMAdapter):
    name = "llama32v"

    def predict(self, image):
        if self.model is None:
            raise RuntimeError("請先呼叫 load()")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": build_prompt()}],
        }]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False,
            )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        probs, boxes = parse_response(text)
        return CXRPrediction(
            model_name=self.name, label_probs=probs, boxes=boxes, raw={"response": text},
        )
