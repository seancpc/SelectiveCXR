"""src/models/medgemma.py — MedGemma adapter(Gemma backbone,醫療專精)。"""

from src.models._processor_vlm import ProcessorVLMAdapter


class MedGemmaAdapter(ProcessorVLMAdapter):
    name = "medgemma"
