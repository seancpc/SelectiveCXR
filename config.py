"""
Project MARS — 設定集中管理

集中管理所有可變參數:路徑、模型、conformal、資料切分、分流決策。

本檔不含任何 endpoint / api_key / secret —— 新方向全部使用本地模型推論。
gated 模型(MedGemma)的存取請先於終端機執行 `huggingface-cli login`,
HuggingFace 憑證由 HF 工具鏈管理,不寫入本檔。
"""

import os
from pathlib import Path

# =============================================================================
# 路徑
# =============================================================================
PROJECT_ROOT = Path(__file__).parent

# 桌機執行時的 MIMIC-CXR-JPG 資料根(影像 files/ + inference_results.npz + subset_manifest.csv)。
# 公開 repo 不含資料;預設為開發機路徑,可用環境變數 MARS_CXR_DIR 覆寫(clone 者改這裡或設環境變數即可)。
CXR_DATA_DIR = Path(os.environ.get("MARS_CXR_DIR", "/mnt/d/mars_data/mimic-cxr-jpg"))

DATA_DIR = PROJECT_ROOT / "data"
MIMIC_CXR_DIR = DATA_DIR / "mimic_cxr"            # MIMIC-CXR 影像
MSCXR_DIR = DATA_DIR / "ms_cxr"                   # MS-CXR / Chest ImaGenome bbox 標註
SPLITS_DIR = DATA_DIR / "splits"                  # prepare_splits.py 輸出
GOLDEN_DATASET_DIR = DATA_DIR / "golden_dataset"  # 評測集

LOG_DIR = PROJECT_ROOT / "logs"
AUDIT_LOG_DIR = LOG_DIR / "audit"                 # audit/logger.py 輸出
EVAL_OUTPUT_DIR = PROJECT_ROOT / "eval_results"   # 評測報告輸出

# =============================================================================
# CheXpert 14 標籤 —— label 機率向量的固定順序,全專案共用
# =============================================================================
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
NUM_LABELS = len(CHEXPERT_LABELS)
LABEL_TO_INDEX = {name: i for i, name in enumerate(CHEXPERT_LABELS)}

# =============================================================================
# Ensemble 模型 —— 跨 backbone(Qwen / Gemma / Llama),刻意選不同來源以製造
# decorrelated disagreement 訊號(架構多樣性 > 能力差距,arXiv 2603.25450)。
# Phase 0 選型結果(已實測於 4090,transformers 5.x + bnb 4bit):
#   - 三個都能在主環境乾淨載入(AutoModelForImageTextToText),無 remote code 地獄
#   - CheXagent(原第三選擇,Phi backbone)remote code 硬依賴 tensorflow 且與
#     transformers 5.x 不相容 → 棄用,改 Llama-3.2-Vision(乾淨、Llama backbone)
# =============================================================================
ENSEMBLE_MODELS = {
    "qwen3vl":   "Qwen/Qwen3-VL-8B-Instruct",                 # Qwen backbone,通用視角(~6.4GB)
    "medgemma":  "google/medgemma-1.5-4b-it",                 # Gemma backbone,醫療專精(~3.2GB)
    "llama32v":  "meta-llama/Llama-3.2-11B-Vision-Instruct",  # Llama backbone,通用視角(~7.2GB)
}
DEVICE = "cuda"
LOAD_IN_4BIT = True              # 4-bit 量化以節省 VRAM
POSITIVE_THRESHOLD = 0.5         # label 機率轉二元判定的門檻

# -----------------------------------------------------------------------------
# Inference Engine —— Phase 0 任務 0.5 雙軌驗證
# vLLM 提供 continuous batching + PagedAttention(對 transformers 約 14-24x throughput)
# 但 VRAM 較貪;三模型同跑可行性須於 4090 實測,實測前以 transformers+bnb 為穩妥起點。
# adapter(src/models/*.py)依本旗標切換 backend。
# -----------------------------------------------------------------------------
INFERENCE_ENGINE = "transformers"            # "transformers" | "vllm"

# vLLM 專屬設定(僅在 INFERENCE_ENGINE = "vllm" 時生效)
VLLM_GPU_MEMORY_UTILIZATION = 0.30           # 三模型同跑時各佔 ~1/3 VRAM,Phase 0 微調
VLLM_DTYPE = "auto"
VLLM_GUIDED_DECODING_BACKEND = "outlines"    # JSON schema 約束結構化輸出

# =============================================================================
# Conformal Prediction
# =============================================================================
CONFORMAL_ALPHA = 0.15           # 目標錯誤率上限(per-label);demo 展示操作點,α=0.05 真實分布見 README 誠實註記
CONFORMAL_MODE = "mondrian"      # mondrian = per-label 各自校準

# =============================================================================
# 資料切分 —— patient-level(prepare_splits.py 使用)
# conformal 需要獨立的 calibration set,不得與 train / test 重疊
# =============================================================================
SPLIT_RATIOS = {"train": 0.6, "calibration": 0.2, "test": 0.2}
SPLIT_SEED = 42

# =============================================================================
# 分流決策
# =============================================================================
# 三檔分流(雙分界):單一 study 能力內棄答 finding 數 n
#   n ≤ AUTO_MAX_UNCERTAIN          → AUTO  高度自動(僅少數待確認)
#   AUTO_MAX < n ≤ FLAG_MAX         → FLAG  部分標記、其餘自動
#   n > FLAG_MAX_UNCERTAIN          → REFER 整案轉人工
# 註:多標籤聯合棄答下「整張零棄答」幾乎不發生,故 AUTO 採「棄答 ≤ 門檻」而非「= 0」
AUTO_MAX_UNCERTAIN = 2
FLAG_MAX_UNCERTAIN = 3
