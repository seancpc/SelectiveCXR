"""src/models/qwen3vl.py — Qwen3-VL adapter(已驗證策略 A 可行)。"""

from src.models._processor_vlm import ProcessorVLMAdapter


class Qwen3VLAdapter(ProcessorVLMAdapter):
    name = "qwen3vl"
