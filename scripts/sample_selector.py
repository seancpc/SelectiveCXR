"""
MIMIC-CXR 樣本篩選器
從 radiology.csv 中篩選適合 MVP 的高品質胸部 X-ray 報告
"""

import gzip
import csv
import re
from collections import defaultdict
from pathlib import Path

# 設定路徑
RADIOLOGY_CSV = Path("c:/Project_MARS/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/radiology.csv.gz")
OUTPUT_DIR = Path("c:/Project_MARS/data/sample_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 篩選條件
TARGET_SAMPLES = 500  # 目標樣本數
MIN_REPORT_LENGTH = 200  # 報告最小字數（過濾太短的）
MAX_REPORT_LENGTH = 3000  # 報告最大字數（過濾異常長的）

# 胸部 X-ray 關鍵字（EXAMINATION 欄位）
CHEST_XRAY_PATTERNS = [
    r'CHEST\s*\(PA AND LAT\)',
    r'CHEST\s*\(PORTABLE AP\)',
    r'CHEST\s*\(AP\)',
    r'CHEST\s*\(SINGLE VIEW\)',
    r'CHEST\s*X-?RAY',
    r'CHEST RADIOGRAPH',
]

# 報告品質關鍵字（需要包含這些結構化內容）
QUALITY_KEYWORDS = [
    'IMPRESSION:',
    'FINDINGS:',
    'CONCLUSION:',
]

# 有臨床發現的關鍵字（優先選擇有異常的案例）
CLINICAL_FINDINGS = [
    'pneumonia', 'consolidation', 'infiltrate',
    'effusion', 'pleural effusion',
    'cardiomegaly', 'enlarged heart',
    'edema', 'pulmonary edema',
    'nodule', 'mass', 'lesion',
    'pneumothorax', 'atelectasis',
    'fracture', 'opacity',
]


def is_chest_xray(text: str) -> bool:
    """檢查是否為胸部 X-ray 報告"""
    text_upper = text.upper()
    for pattern in CHEST_XRAY_PATTERNS:
        if re.search(pattern, text_upper):
            return True
    return False


def has_quality_structure(text: str) -> bool:
    """檢查報告是否有結構化內容"""
    text_upper = text.upper()
    return any(kw.upper() in text_upper for kw in QUALITY_KEYWORDS)


def count_clinical_findings(text: str) -> int:
    """計算臨床發現數量"""
    text_lower = text.lower()
    return sum(1 for finding in CLINICAL_FINDINGS if finding in text_lower)


def extract_impression(text: str) -> str:
    """提取 IMPRESSION 部分"""
    match = re.search(r'IMPRESSION:(.*?)(?=\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:500]
    return ""


def main():
    print("=" * 60)
    print("MIMIC-CXR 樣本篩選器")
    print("=" * 60)

    # 儲存符合條件的樣本
    candidates = []

    # 統計資訊
    stats = {
        'total_processed': 0,
        'chest_xray_count': 0,
        'quality_passed': 0,
        'length_passed': 0,
    }

    print(f"\n正在分析 {RADIOLOGY_CSV}...")
    print("這可能需要幾分鐘...\n")

    with gzip.open(RADIOLOGY_CSV, 'rt', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            stats['total_processed'] += 1

            # 進度顯示
            if stats['total_processed'] % 1000000 == 0:
                print(f"已處理: {stats['total_processed']:,} 筆, 候選樣本: {len(candidates)}")

            text = row.get('text', '')

            # 篩選條件 1: 是否為胸部 X-ray
            if not is_chest_xray(text):
                continue
            stats['chest_xray_count'] += 1

            # 篩選條件 2: 報告長度
            text_length = len(text)
            if text_length < MIN_REPORT_LENGTH or text_length > MAX_REPORT_LENGTH:
                continue
            stats['length_passed'] += 1

            # 篩選條件 3: 報告品質（有結構化內容）
            if not has_quality_structure(text):
                continue
            stats['quality_passed'] += 1

            # 計算臨床發現分數
            finding_score = count_clinical_findings(text)
            impression = extract_impression(text)

            candidates.append({
                'note_id': row['note_id'],
                'subject_id': row['subject_id'],
                'hadm_id': row.get('hadm_id', ''),
                'charttime': row.get('charttime', ''),
                'text_length': text_length,
                'finding_score': finding_score,
                'impression': impression,
                'full_text': text,
            })

            # 收集足夠候選後提早結束（加速處理）
            if len(candidates) >= TARGET_SAMPLES * 3:
                print(f"\n已收集足夠候選樣本，停止掃描...")
                break

    print("\n" + "=" * 60)
    print("統計結果")
    print("=" * 60)
    print(f"總處理筆數: {stats['total_processed']:,}")
    print(f"胸部 X-ray 報告: {stats['chest_xray_count']:,}")
    print(f"長度符合: {stats['length_passed']:,}")
    print(f"品質通過: {stats['quality_passed']:,}")
    print(f"候選樣本數: {len(candidates)}")

    # 根據臨床發現分數排序，優先選擇有異常的案例
    candidates.sort(key=lambda x: (-x['finding_score'], x['text_length']))

    # 選擇最終樣本（確保 subject_id 多樣性）
    selected = []
    seen_subjects = set()

    # 第一輪：有臨床發現的（找 300 個）
    for c in candidates:
        if c['finding_score'] > 0 and c['subject_id'] not in seen_subjects:
            selected.append(c)
            seen_subjects.add(c['subject_id'])
            if len(selected) >= 300:
                break

    # 第二輪：正常案例（補足到 500 個）
    for c in candidates:
        if c['subject_id'] not in seen_subjects:
            selected.append(c)
            seen_subjects.add(c['subject_id'])
            if len(selected) >= TARGET_SAMPLES:
                break

    print(f"\n最終選擇樣本數: {len(selected)}")
    print(f"  - 有臨床發現: {sum(1 for s in selected if s['finding_score'] > 0)}")
    print(f"  - 正常案例: {sum(1 for s in selected if s['finding_score'] == 0)}")

    # 輸出結果
    # 1. Subject ID 清單（用於下載）
    subject_ids = sorted(set(s['subject_id'] for s in selected))
    subject_file = OUTPUT_DIR / "selected_subject_ids.txt"
    with open(subject_file, 'w') as f:
        for sid in subject_ids:
            f.write(f"{sid}\n")
    print(f"\nSubject ID 清單已儲存: {subject_file}")

    # 2. 詳細樣本資訊（CSV）
    detail_file = OUTPUT_DIR / "selected_samples.csv"
    with open(detail_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject_id', 'note_id', 'hadm_id', 'charttime',
            'text_length', 'finding_score', 'impression'
        ])
        writer.writeheader()
        for s in selected:
            writer.writerow({
                'subject_id': s['subject_id'],
                'note_id': s['note_id'],
                'hadm_id': s['hadm_id'],
                'charttime': s['charttime'],
                'text_length': s['text_length'],
                'finding_score': s['finding_score'],
                'impression': s['impression'][:200] if s['impression'] else '',
            })
    print(f"樣本詳情已儲存: {detail_file}")

    # 3. 完整報告文本（用於建立 Golden Dataset）
    reports_file = OUTPUT_DIR / "selected_reports.csv"
    with open(reports_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject_id', 'note_id', 'full_text'
        ])
        writer.writeheader()
        for s in selected:
            writer.writerow({
                'subject_id': s['subject_id'],
                'note_id': s['note_id'],
                'full_text': s['full_text'],
            })
    print(f"完整報告已儲存: {reports_file}")

    # 4. 下載指令腳本
    download_script = OUTPUT_DIR / "download_images.sh"
    with open(download_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# MIMIC-CXR-JPG 影像下載腳本\n")
        f.write("# 請先設定 PhysioNet 帳號\n\n")
        f.write("PHYSIONET_USER='your_username'\n")
        f.write("OUTPUT_DIR='./data/mimic_cxr'\n\n")
        f.write("mkdir -p $OUTPUT_DIR\n\n")
        f.write("# 下載選定的 subject 影像\n")
        for sid in subject_ids:
            # subject_id 格式: ######## -> p10/p########
            prefix = f"p{str(sid)[:2]}"
            f.write(f"wget -r -N -c -np -nH --cut-dirs=5 "
                   f"--user=$PHYSIONET_USER --ask-password "
                   f"-P $OUTPUT_DIR "
                   f"https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/{prefix}/p{sid}/\n")
    print(f"下載腳本已儲存: {download_script}")

    # 5. 顯示樣本預覽
    print("\n" + "=" * 60)
    print("樣本預覽（前 5 筆）")
    print("=" * 60)
    for i, s in enumerate(selected[:5]):
        print(f"\n[{i+1}] Subject: {s['subject_id']}")
        print(f"    發現分數: {s['finding_score']}, 報告長度: {s['text_length']}")
        print(f"    IMPRESSION: {s['impression'][:150]}...")


if __name__ == "__main__":
    main()
