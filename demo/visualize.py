"""
demo/visualize.py — demo 視覺化元件

把 pipeline.StudyResult 轉成可顯示元件:
  - decision_html     : triage 三檔決策卡(白話 + 病徵中文)
  - merged_chart_html : 每病徵一行的合併圖(三模型彩點 + 平均線 + 決策背景色)

全程走 HTML(瀏覽器顯示中文),無 matplotlib / PIL 依賴。
"""

from __future__ import annotations

from config import CHEXPERT_LABELS
from src.decision.triage import TriageDecision

# 三模型「各自判讀」彩點色:刻意避開決策背景的綠(自動)/紅(待確認),改用藍/橘/紫
_MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]

# CheXpert 14 病徵中文對照(決策卡白話化,給非醫療面試官)
FINDING_ZH = {
    "No Finding": "無異常", "Enlarged Cardiomediastinum": "縱膈腔增大",
    "Cardiomegaly": "心臟肥大", "Lung Opacity": "肺部陰影", "Lung Lesion": "肺部病灶",
    "Edema": "肺水腫", "Consolidation": "肺實變", "Pneumonia": "肺炎",
    "Atelectasis": "肺塌陷", "Pneumothorax": "氣胸", "Pleural Effusion": "胸腔積液",
    "Pleural Other": "其他胸膜病變", "Fracture": "骨折", "Support Devices": "醫療裝置",
}


def _zh(labels):
    """finding 英文 list → 中文頓號串;空 → (無)。"""
    return "、".join(FINDING_ZH.get(l, l) for l in labels) if labels else "(無)"


def decision_html(study):
    """triage 三檔決策卡(全白話 + 病徵中文,非醫療、非技術背景也能讀懂)。"""
    tr = study.triage
    d = tr.decision
    title, color = {
        TriageDecision.AUTO:  ("高度自動判讀", "#2ca02c"),
        TriageDecision.FLAG:  ("標記少部分項目待確認", "#ff7f0e"),
        TriageDecision.REFER: ("整案轉人工審查", "#d62728"),
    }[d]
    plain = {
        TriageDecision.AUTO:
            "整張走自動流程,系統僅附註極少數待確認提示,不需人工逐項複查。",
        TriageDecision.FLAG:
            "系統多數有把握,少數待確認,標記後其餘自動判讀。",
        TriageDecision.REFER:
            "待確認項目過多,交由人工審查。",
    }[d]
    html = (
        f'<div style="font-size:15px;line-height:1.8">'
        f'<div style="font-size:22px;font-weight:bold;color:{color}">AI 判讀結果:{title}</div>'
        f'<p style="font-size:16px">{plain}</p>'
        f'<p><b style="color:#d62728">標記待確認:</b>{_zh(tr.uncertain_labels)}</p>'
    )
    return html + "</div>"


def merged_chart_html(study, model_names):
    """合併圖(HTML):每病徵一行 = 三模型彩點(分歧)+ 三模型平均豎線 + 背景決策色。

    取代 prob_bar + model_compare 兩張 matplotlib 圖 ——
    連結同框(點散開↔背景紅)、病徵中文、未來可加 hover。色彩繼承 gradio 主題文字色 +
    rgba 半透明背景,深/淺色主題皆可讀。
    """
    tr = study.triage
    boundary = set(tr.boundary_labels)
    abstain = tr.abstain_mask
    probs = study.model_probs           # (M, 14)
    ens = study.ensemble_prob           # (14,)
    M = probs.shape[0]

    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;'
        f'background:{_MODEL_COLORS[m % len(_MODEL_COLORS)]};display:inline-block;"></span>'
        f'{model_names[m] if m < len(model_names) else f"模型{m}"}</span>'
        for m in range(M)
    )

    rows = ""
    for i, lbl in enumerate(CHEXPERT_LABELS):
        if lbl in boundary:
            continue                       # 能力邊界不展示,聚焦系統有把握的判讀項目
        zh = FINDING_ZH.get(lbl, lbl)
        if abstain[i]:
            bg, tag, tc = "rgba(214,39,40,0.16)", "待確認", "#e24b4a"
        else:
            bg, tag, tc = "rgba(44,160,44,0.16)", "自動判讀", "#3aa83a"
        mid = ('<div style="position:absolute;left:50%;top:0;bottom:0;'
               'border-left:1px dashed rgba(255,255,255,0.35);"></div>')
        ensm = (f'<div style="position:absolute;left:{ens[i]*100:.0f}%;top:0;bottom:0;width:2px;'
                f'background:#f5c518;opacity:0.95;transform:translateX(-50%);"></div>')
        dots = "".join(
            f'<div style="position:absolute;left:{probs[m, i]*100:.0f}%;top:50%;'
            f'width:9px;height:9px;border-radius:50%;'
            f'background:{_MODEL_COLORS[m % len(_MODEL_COLORS)]};transform:translate(-50%,-50%);"></div>'
            for m in range(M)
        )
        inner = mid + ensm + dots
        rows += (
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">'
            f'<div style="width:96px;text-align:right;font-size:13px;opacity:0.8;">{zh}</div>'
            f'<div style="flex:1;position:relative;height:24px;background:{bg};border-radius:6px;">{inner}</div>'
            f'<div style="width:62px;font-size:12px;color:{tc};">{tag}</div>'
            f'</div>'
        )

    return (
        f'<div style="font-size:13px;line-height:1.5;">'
        f'<div style="margin-bottom:8px;opacity:0.85;">{legend}'
        f'<span style="margin-left:4px;">｜背景:<span style="color:#3aa83a;">綠自動判讀</span> '
        f'<span style="color:#e24b4a;">紅待確認</span> ｜ '
        f'<span style="color:#f5c518;">黃實線</span>=三模型平均、灰虛線=0.5 分界</span></div>'
        f'{rows}'
        f'<div style="display:flex;padding-left:106px;opacity:0.6;font-size:11px;margin-top:2px;">'
        f'<span>0 無</span><span style="margin-left:auto;">判定有此病徵的可能性</span>'
        f'<span style="margin-left:auto;">有 1.0</span></div>'
        f'</div>'
    )
