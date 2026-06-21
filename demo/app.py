"""
demo/app.py — SelectiveCXR demo(階段 A:預跑結果瀏覽)

Gradio 介面:從 test set 選一張 CXR,展示選擇性判讀全流程 ——
三模型 ensemble 判讀 + 跨 backbone disagreement + conformal 棄答門檻 + 三檔分流決策,
並把「三模型各自判讀」視覺化(展示 VLM ensemble 的核心優點)。

版面:分流決策(白話) → 原圖 + 14-finding 判讀 → 三模型各自判讀。
(grounding bbox 疊圖已移除:定義 A/B 驗證 AUC<0.5,定位訊號不可靠,留著只會混淆焦點)

前置(桌機):
  1. pip install gradio matplotlib
  2. python demo/calibrate.py        # 產生 demo/calibrator.pkl
用法:
  python demo/app.py                 # 開 http://localhost:7860
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CXR_DATA_DIR

import gradio as gr
from PIL import Image

from demo.pipeline import DemoResults
from demo.visualize import decision_html, merged_chart_html

DATA_DIR = CXR_DATA_DIR
NPZ = DATA_DIR / "inference_results.npz"
PKL = Path(__file__).resolve().parent / "calibrator.pkl"

results = DemoResults(NPZ, PKL, DATA_DIR)
TEST_IDX = results.test_indices()
CHOICES = [(str(results.dicom_ids[i]), int(i)) for i in TEST_IDX]

INTRO = """
# SelectiveCXR — 根據三模型判斷分歧分流的選擇性判讀系統

不同體系的三個模型集成做影像辨識,**有把握就自動判讀放行,沒把握就標記交由人工審查**。
框架領域無關 —— 換成晶圓缺陷檢測、金融風控同樣適用。
"""


def show(index):
    study = results.analyze(index)
    base = Image.open(results.image_path(index)).convert("RGB")
    base.thumbnail((768, 768))
    return (
        decision_html(study),
        base,
        merged_chart_html(study, results.models),
    )


with gr.Blocks(title="SelectiveCXR — 選擇性判讀 demo") as app:
    gr.Markdown(INTRO)
    sel = gr.Dropdown(choices=CHOICES, label="選一張 test CXR (dicom_id)",
                      value=CHOICES[0][1] if CHOICES else None)

    # 1. 分流決策(白話,最先顯示)
    decision = gr.HTML()

    # 2. 原圖 + 合併圖(三模型各自判讀 + 系統決策,同框 —— 分歧↔棄答連結可見)
    with gr.Row():
        base_img = gr.Image(label="原始 CXR", type="pil", scale=2)
        with gr.Column(scale=3):
            gr.Markdown(
                "**三個 AI 各自判讀(彩點)＋ 系統決策(背景色)**　"
                "點越散 = AI 越不同意 = 越可能標「待確認」"
            )
            merged = gr.HTML()

    outputs = [decision, base_img, merged]
    sel.change(show, inputs=sel, outputs=outputs)
    if CHOICES:
        app.load(show, inputs=sel, outputs=outputs)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
