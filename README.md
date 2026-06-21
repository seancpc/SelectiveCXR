# SelectiveCXR

**Selective Chest X-ray Interpretation with Multi-Source Uncertainty & Conformal Guarantees**
具統計保證的選擇性胸腔 X 光判讀系統

**[English](#english)** | **[繁體中文](#繁體中文)**

`Python` · `PyTorch` · `vLLM` · `transformers` · `Conformal Prediction` · `Gradio` · `RTX 4090`

---

<a name="english"></a>

## English

> **An AI that knows when to say "I'm not sure."** When confident, it reads autonomously; when unsure, it honestly flags the case and defers to a human expert.
> Validated on chest X-rays (CXR), but the framework is **domain-agnostic**: the same logic fits semiconductor defect inspection, financial risk control — any setting where errors are costly and trust must be provable.

### TL;DR

- **Not chasing diagnostic accuracy** (a red ocean) but **trustworthy automation**: the system knows when it is unreliable and defers those cases to humans.
- **Method**: three cross-backbone vision-language models (VLMs) read independently → **inter-model disagreement** quantifies uncertainty → **conformal prediction** gives a distribution-free statistical guarantee → three-way triage (auto-read / flag-for-review / refer-to-human).
- **Paradigm-agnostic**: swap the front-end models for a wafer-defect model and the uncertainty + decision layers stay untouched.
- **The most valuable part isn't an accuracy number — it's a complete "validate → honestly reject → correct" engineering trail**: a signal abandoned after validation, a garbage model caught and removed, and one overturning of my own assumption.

### 1. The Problem: the danger isn't being wrong — it's being *confidently* wrong

Today's medical AI sounds exactly as certain when it is wrong as when it is right — it gives a definitive answer for every image and never says "I'm not sure about this one." In high-cost settings (a missed diagnosis, an entire wafer batch scrapped, a financial misjudgment), a *confident error* is far more dangerous than an honest "I don't know."

**SelectiveCXR isn't a more accurate AI — it's an AI with self-awareness**, like a good resident: it reads most images independently, but **knows when to ask the attending physician**.

### 2. Core Method

```
              Chest X-ray
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
  Qwen3-VL    MedGemma    Llama-3.2-V    ← three "differently-born" VLMs (cross-backbone)
  (general)   (medical)   (general)         uncorrelated errors → disagreement is meaningful
     └────────────┼────────────┘
                  ▼
       cross-model disagreement            ← uncertainty signal
                  ▼
        Conformal Prediction calibration    ← distribution-free guarantee
                  ▼
            three-way triage
     ┌────────────┼────────────┐
     ▼            ▼            ▼
  auto-read   flag-review   refer-to-human
```

- **Cross-backbone ensemble**: deliberately three different-origin models (Qwen / Gemma / Llama). Architectural diversity → uncorrelated errors → when they "argue," it is a genuinely hard case.
- **Disagreement as the uncertainty signal**: the more the three models disagree on a finding, the more likely the system abstains.
- **Conformal Prediction**: statistically derives the abstention threshold, giving a **black-and-white guarantee** that "the error rate on auto-read cases ≤ α" — not a hand-tuned threshold.
- **Three-way triage**: auto-read / flag a few for review / refer the whole study — and honestly marks the **capability boundary** (findings the system inherently cannot judge, always deferred).

### 3. How it compares to traditional ML

**Core argument: traditional ML's problem isn't insufficient accuracy — it's not knowing when it is wrong.**

| | Traditional CNN | This system |
|---|---|---|
| On low-confidence input | answers anyway (softmax often overconfident) | **abstains, defers to human** |
| Source of confidence | a single softmax (poorly calibrated, no guarantee) | cross-backbone disagreement |
| What it can promise | "90% accurate on average" (but *which* 10% is wrong?) | **"error rate ≤ 5%" distribution-free guarantee** |
| Problem it solves | accuracy | **trustworthy automation** |

> **Not competing on the accuracy battlefield** (that is not this project's value). In fact, on raw judgment quality a dedicated CNN still slightly beats the general-VLM ensemble (see below). This system's value is **maximizing what can be safely automated and leaving the dangerous part to humans** — with a statistical guarantee.

### 4. The most valuable part: an honest engineering trail

Development wasn't a straight success — it was a series of **validate → find the problem → honestly correct**. This demonstrates research integrity and engineering judgment better than any pretty number:

- ✅ **The disagreement signal works**: after selective prediction, accuracy on the auto-read portion is markedly higher than answering everything.
- ❌ **Grounding (localization) signal: both designs validated and abandoned.** The hypothesis was that "the three models' disagreement on lesion location" is a useful signal, but under strict AUC validation, **both cross-model bbox IoU (Definition B) and anatomical-model verification (Definition A) scored AUC < 0.5**. Root cause: VLMs are reliable at *judging* but inherently weak at *localizing*. **Honestly dropped — no forcing in a dead signal.**
- 🔍 **Caught a garbage ensemble member**: the demo's three-model visualization exposed one member "template-filling" 72% of images (all-zeros or a fixed box). I built the `check_single` quality gate, corroborated it with a third-party benchmark (ReXVQA 47%), and removed the member — **shrinking the capability boundary from 5 findings to 2**.
- 🔧 **Seeing the real constraint**: the third-model candidates kept failing on inference-framework version compatibility, not on the models themselves. Realizing "the constraint is vLLM, not the model," I **ran Llama-3.2-Vision via transformers**, restoring the originally intended cross-backbone design (mixed engines, invisible to the front end).
- 📊 **CNN vs VLM comparison + overturning myself**: the first comparison showed "VLM crushes CNN," but an absurd CNN accuracy of 0.188 triggered suspicion — **it was a threshold artifact**. Redoing it with threshold-free per-label AUC produced the honest conclusion.

### 5. Results (honest numbers)

> Scale: MIMIC-CXR, 1000 calibration + 671 test images, 14 CheXpert findings. Triage uses an α = 0.15 operating point (see "multi-label joint abstention" below).

- **Core selective-prediction validation**: after abstaining on the most uncertain samples, accuracy on the auto-read portion clearly beats answering everything (~0.80 when answering all).
- **Multi-label joint abstention (a structural finding + an honest operating point)**: under a strict α=0.05, the pure disagreement signal makes per-image abstention counts highly concentrated (~6 of 12 in-capability labels on average), and "zero abstention over a whole study" almost never happens — revealing a structural challenge of multi-label selective prediction (more labels → "all pass" is harder; hence AUTO is defined as "≤2 abstentions" rather than "=0"). **The demo's triage uses an α=0.15 operating point** (auto-read error ≤15%, a defensible compromise for low-risk triage), letting all three tiers — high-auto / partial-flag / refer-to-human — appear. **The true α=0.05 distribution (high abstention, rare AUTO) is recorded here honestly — a display operating point ≠ hiding the truth.**
- **Capability boundary = 2 / 14**: the system honestly marks the two high-prevalence findings that disagreement cannot rescue (Cardiomegaly, Lung Opacity) as always-deferred, and runs selective prediction on the rest.
- **CNN vs VLM (per-label AUC, threshold-free)**: CNN **0.626** > VLM ensemble **0.585** — **the VLM ensemble's judgment is slightly below a dedicated CNN** (honest, not overclaimed); but the two **win and lose per-finding and have uncorrelated errors** (CNN stronger on density-type lesions, VLM on morphology-type), which is exactly the basis for a "CNN + VLM hybrid."
- **The VLM's real value** isn't accuracy — it's the cross-backbone disagreement signal, interpretability, and an uncorrelated second opinion to the CNN.

### 6. Transferability (the cross-domain pivot)

The framework fits scenarios that **simultaneously** satisfy three conditions: **(1) high / asymmetric error cost, (2) a human fallback exists, (3) trust must be provable**.

→ Medical imaging, **semiconductor wafer-defect inspection**, financial risk control, and insurance underwriting all qualify. Swap chest X-rays for wafer images and only the front-end model changes — **the selective-prediction + conformal decision layer stays untouched**.

### 7. Tech Stack

- **Models**: Qwen3-VL-8B, MedGemma-1.5-4b, Llama-3.2-11B-Vision (4-bit quantized, ~17 GB / 24 GB total)
- **Inference**: vLLM (continuous batching, ~28× speedup) + transformers (mixed engine)
- **Method**: Conformal Prediction (Mondrian / per-label), cross-model disagreement
- **Front end**: Gradio (plain-language demo, Chinese finding names, merged visualization)
- **Hardware**: a single RTX 4090 (WSL2)

### 8. Honest limitations

- The VLM ensemble's judgment quality is **slightly below** a dedicated CNN (this system's value is uncertainty handling, not accuracy).
- 2 high-prevalence findings fall on the capability boundary, unrecoverable by disagreement (needs a stronger signal or a more accurate model).
- The grounding (localization) signal validated as ineffective and was removed; Definition A (anatomical-model version) is future work.
- Conformal struggles to hit the α=0.05 ceiling on medical multi-label: a narrow abstention distribution and near-zero whole-study zero-abstention (multi-label joint abstention) — a genuine research finding; the demo uses an α=0.15 operating point (see §5).

### 9. Data & Compliance

- This project is for **research and technical demonstration, not a clinical diagnostic tool**, and must not be used for actual clinical decisions.
- MIMIC-CXR is governed by the PhysioNet Data Use Agreement (DUA); **image data is not publicly redistributed** — use public or de-identified CXR for demos.
- MedGemma and similar models are research-use licensed.

---

<a name="繁體中文"></a>

## 繁體中文

> **一個「會說我不確定」的 AI 判讀系統** —— 有把握就自動判讀,沒把握就誠實標記、交給人類專家。
> 以胸腔 X 光(CXR)驗證,但**框架領域無關**:同一套邏輯可用於半導體缺陷檢測、金融風控等任何「錯誤成本高 + 需要可證明信任」的場景。

### TL;DR

- **不拼判讀準確率**(那是紅海),拼 **可信賴的自動化**:系統知道自己**何時不可靠**,把不確定的交給人。
- **方法**:三個跨 backbone 的視覺語言模型(VLM)各自判讀 → 用 **模型間分歧** 量化不確定性 → **conformal prediction** 提供 distribution-free 統計保證 → 三檔分流(自動 / 標記待確認 / 整張交醫師)。
- **範式無關**:把前端 AI 換成晶圓缺陷模型,中間的不確定性與決策層完全不動。
- **這個專案最值錢的不是某個準確率數字,而是一條「驗證 → 誠實否定 → 修正」的完整工程歷程** —— 包含一個被驗證後放棄的訊號、一個被揪出的垃圾模型、與一次對自身假設的推翻。

### 1. 問題:AI 最危險的不是「看錯」,是「看錯了還很自信」

現在的醫療 AI,判斷錯誤時語氣跟判斷正確時一樣篤定 —— 它對每張片子都給斬釘截鐵的答案,從不說「這張我不確定」。在錯一次代價極高的場景(漏診、晶圓整批報廢、金融誤判),一個「很有自信的錯誤」遠比「老實承認不知道」危險。

**SelectiveCXR 要做的不是更準的 AI,而是有「自知之明」的 AI** —— 像一個好的實習醫師:大部分片子能獨立判讀,但**知道自己什麼時候該去問主治醫師**。

### 2. 核心方法

```
              胸腔 X 光
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
  Qwen3-VL    MedGemma    Llama-3.2-V      ← 三個「出身不同」的 VLM(跨 backbone)
   (通用)      (醫療)       (通用)            錯誤不相關,分歧才有意義
     └────────────┼────────────┘
                  ▼
       跨模型 disagreement(判斷分歧)        ← 不確定性訊號
                  ▼
        Conformal Prediction 校準           ← distribution-free 統計保證
                  ▼
            三檔分流決策
     ┌────────────┼────────────┐
     ▼            ▼            ▼
   自動判讀    標記待確認    整張交醫師
```

- **跨 backbone ensemble**:刻意選三個不同來源的模型(Qwen / Gemma / Llama)。架構多樣性 → 錯誤不相關 → 它們「吵架」時,才是真的難 case。
- **disagreement 當不確定性訊號**:三模型對同一病徵的判斷越分歧,越可能棄答。
- **Conformal Prediction**:用統計方法決定棄答門檻,可給「自動判讀部分錯誤率 ≤ α」的**白紙黑字保證**,而非工程師手調。
- **三檔分流**:自動判讀 / 標記少數待確認 / 整張交醫師,並誠實標出系統「天生判不準、固定交人工」的能力邊界。

### 3. 跟傳統機器學習比,好在哪

**核心論點:傳統 ML 的問題不是不夠準,是它「不知道自己什麼時候會錯」。**

| | 傳統專用 CNN | 本系統 |
|---|---|---|
| 對沒把握的輸入 | 照樣給答案(softmax 常過度自信) | **主動棄答,交給人** |
| 信心來源 | 單一 softmax(校準差、無理論保證) | 跨 backbone 分歧 |
| 能給的承諾 | 「平均 90% 準」(但不知哪 10% 錯) | **「錯誤率 ≤ 5%」的 distribution-free 保證** |
| 解決的問題 | 準確率 | **可信賴的自動化** |

> **不在準確率戰場上比**(那不是本專案的價值)。實測也顯示:在純判斷品質上,專用 CNN 仍略勝通用 VLM ensemble(見下)。本系統的價值在於**把能安全自動化的部分最大化、把危險的留給人**,並對此提供統計保證。

### 4. 這個專案最值錢的:誠實的工程歷程

開發過程不是「一路成功」,而是一連串**驗證 → 發現問題 → 誠實修正**。這比任何漂亮數字更能展現研究誠信與工程判斷力:

- ✅ **disagreement 訊號有效**:選擇性判讀後,自動判讀部分的準確率顯著高於全答。
- ❌ **grounding(定位)訊號:兩種設計都驗證後放棄**。原本假設「三模型對病灶定位的分歧」是有用訊號,但用 AUC 嚴格驗證後,**跨模型 bbox IoU(定義 B)與解剖模型驗證(定義 A)的預測 AUC 均 < 0.5**。根因:VLM「判斷」可靠但「定位」能力本質不足。**誠實放棄,不硬塞無效訊號**。
- 🔍 **揪出一個垃圾 ensemble 成員**:demo 的三模型視覺化暴露出某成員對 72% 影像「填模板」(全 0 或固定框)。建立 `check_single` 品質關卡、用第三方 benchmark(ReXVQA 47%)佐證後移除 —— **能力邊界因此從 5 個縮到 2 個**。
- 🔧 **看穿真正的約束**:第三模型的候選一再卡在推論框架的版本相容性,而非模型本身。意識到「約束是 vLLM 而非模型」後,**改用 transformers 跑 Llama-3.2-Vision**,回到最初理想的跨 backbone 設計(混合引擎,對前端展示無影響)。
- 📊 **CNN vs VLM 對比 + 推翻自己**:初版對比顯示「VLM 碾壓 CNN」,但 CNN 準確率 0.188 這個荒謬數字觸發警覺 —— **發現是 threshold 假象**,改用不依賴 threshold 的 per-label AUC 重做,得到誠實結論。

### 5. 結果(誠實數字)

> 規模:MIMIC-CXR,calibration 1000 + test 671 張,CheXpert 14 病徵。三檔分流採 α = 0.15 操作點(理由見下方「多標籤聯合棄答」)。

- **選擇性判讀核心驗證**:棄答最不確定的樣本後,自動判讀部分準確率明顯優於全答(全答 ~0.80)。
- **多標籤聯合棄答(結構性發現 + 誠實操作點)**:嚴格 α=0.05 下,純 disagreement 訊號讓每張片子棄答數高度集中(平均約 6 / 12 個能力內 label),且「整張零棄答」幾乎不發生 —— 揭示多標籤 selective prediction 的結構性挑戰(label 越多,「全部通過」越難;故三檔的 AUTO 採「棄答 ≤2 項」而非「=0」)。**demo 三檔分流採 α=0.15 操作點**(自動部分錯誤率 ≤15%,低風險初篩可辯護的折衷),使「高度自動 / 部分標記 / 整案轉人工」三檔都能呈現;**α=0.05 的真實分布(棄答率高、AUTO 罕見)如實記錄於此 —— 展示操作點 ≠ 隱藏真相**。
- **能力邊界 = 2 / 14**:系統誠實標出 disagreement 救不回的兩個高陽性率病徵(Cardiomegaly、Lung Opacity)固定交人工,其餘做選擇性判讀。
- **CNN vs VLM(per-label AUC,不依賴 threshold)**:CNN **0.626** > VLM ensemble **0.585** —— **VLM 判斷品質略遜專用 CNN**(誠實,不誇大);但兩者**逐病徵互有勝負、錯誤不相關**(CNN 強密度型病灶、VLM 強型態型),這正是「CNN + VLM 混合」的理論基礎。
- **VLM 的真正價值**不在準確率,而在:跨 backbone 分歧訊號、可解釋、與 CNN 不相關的第二意見。

### 6. 可遷移性(跨領域支點)

本框架適用於**同時滿足**三條件的場景:**(1) 錯誤成本高 / 不對稱、(2) 有人工 fallback、(3) 需要可證明的信任**。

→ 醫療影像、**半導體晶圓缺陷檢測**、金融風控、保險核保皆命中。把胸腔 X 光換成晶圓影像,只需更換前端判讀模型,**selective prediction + conformal 的決策層完全不動**。

### 7. 技術棧

- **模型**:Qwen3-VL-8B、MedGemma-1.5-4b、Llama-3.2-11B-Vision(4-bit 量化,三模型共 ~17 GB / 24 GB)
- **推論**:vLLM(continuous batching,~28× 提速)+ transformers(混合引擎)
- **方法**:Conformal Prediction(Mondrian / per-label)、跨模型 disagreement
- **前端**:Gradio(白話化 demo,病徵中文、合併視覺化)
- **硬體**:單張 RTX 4090(WSL2)

### 8. 誠實的限制

- VLM ensemble 的判斷品質**略遜**專用 CNN(本系統價值在不確定性處理,非準確率)。
- 2 個高陽性率病徵落在能力邊界,disagreement 救不回(需更強訊號或更準模型)。
- grounding(定位)訊號驗證無效已移除;定義 A(解剖模型版)為未來工作。
- conformal 在醫療多標籤上難壓進 α=0.05 天花板:棄答數分布窄、整張零棄答幾乎不發生(多標籤聯合棄答),為真實研究發現;demo 展示改採 α=0.15 操作點,真實分布見 §5。

### 9. 資料與合規

- 本專案為**研究與技術展示用途,非臨床診斷工具**,不得用於實際臨床決策。
- 使用之 MIMIC-CXR 受 PhysioNet 資料使用協議(DUA)規範,**不公開散布影像資料**;demo 展示請使用公開或去識別 CXR。
- MedGemma 等模型授權為 research-use。
