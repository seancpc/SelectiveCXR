"""
整理 MIMIC-CXR 資料，建立影像與報告的對應關係
產出 mimic-cxr-reports.csv 供 ingest.py 使用
"""

import csv
from pathlib import Path

# 路徑設定
PROJECT_ROOT = Path(__file__).parent.parent
CXR_DIR = PROJECT_ROOT / "data" / "mimic_cxr"
REPORTS_FILE = PROJECT_ROOT / "data" / "sample_selection" / "matched_reports.csv"
OUTPUT_CSV = CXR_DIR / "mimic-cxr-reports.csv"


def main():
    print("=" * 60)
    print("MIMIC-CXR 資料整理")
    print("=" * 60)

    # 1. 讀取報告
    print(f"\n讀取報告: {REPORTS_FILE}")
    reports = {}
    with open(REPORTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row['subject_id']
            reports[subject_id] = row['full_text']
    print(f"  載入 {len(reports)} 份報告")

    # 2. 掃描影像目錄
    print(f"\n掃描影像目錄: {CXR_DIR}")

    # 建立影像與報告的對應
    data = []

    for jpg_file in CXR_DIR.rglob("*.jpg"):
        # 路徑格式: p<subject>/s<study>/xxx.jpg
        parts = jpg_file.relative_to(CXR_DIR).parts
        if len(parts) < 3:
            continue

        subject_dir = parts[0]  # 例: p########
        study_dir = parts[1]    # 例: s########
        dicom_id = jpg_file.stem  # xxx

        subject_id = subject_dir.replace('p', '')

        # 取得對應報告
        report = reports.get(subject_id, "")

        # 相對路徑 (相對於 mimic_cxr 目錄)
        relative_path = jpg_file.relative_to(CXR_DIR)

        data.append({
            'dicom_id': dicom_id,
            'subject_id': subject_id,
            'study_id': study_dir.replace('s', ''),
            'image_path': str(relative_path),
            'report': report,
        })

    print(f"  找到 {len(data)} 張影像")

    # 3. 儲存 CSV
    print(f"\n儲存: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dicom_id', 'subject_id', 'study_id', 'image_path', 'report'])
        writer.writeheader()
        writer.writerows(data)

    # 4. 統計
    with_report = sum(1 for d in data if d['report'])
    without_report = sum(1 for d in data if not d['report'])

    print(f"\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"總影像數: {len(data)}")
    print(f"有報告: {with_report}")
    print(f"無報告: {without_report}")
    print(f"\n輸出檔案: {OUTPUT_CSV}")
    print(f"\n下一步: 執行 python -m src.ingest --reset")


if __name__ == "__main__":
    main()
