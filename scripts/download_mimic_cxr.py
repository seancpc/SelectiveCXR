"""
MIMIC-CXR-JPG 影像下載器 (Python 版本)
適用於 Windows / Linux / Mac
"""

import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# PhysioNet 認證 — 從環境變數讀取,絕不硬編碼(避免公開洩漏 credentialed 帳號)
#   執行前先設定:
#     export PHYSIONET_USER="你的帳號"
#     export PHYSIONET_PASS="你的密碼"
# ============================================================
PHYSIONET_USER = os.environ.get("PHYSIONET_USER", "")
PHYSIONET_PASS = os.environ.get("PHYSIONET_PASS", "")
if not PHYSIONET_USER or not PHYSIONET_PASS:
    sys.exit("錯誤:請先設定環境變數 PHYSIONET_USER 與 PHYSIONET_PASS")
# ============================================================

# 路徑設定
PROJECT_ROOT = Path(__file__).parent.parent
SUBJECT_LIST = PROJECT_ROOT / "data" / "sample_selection" / "selected_subject_ids.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "mimic_cxr"

# 基礎 URL
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"


def download_subject(subject_id: str, index: int, total: int) -> bool:
    """下載單一病患的影像"""

    # 計算路徑 (例: ######## -> p10/p########)
    prefix = f"p{subject_id[:2]}"
    url = f"{BASE_URL}/{prefix}/p{subject_id}/"

    # 輸出目錄
    subject_dir = OUTPUT_DIR / prefix / f"p{subject_id}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{index}/{total}] 下載 p{subject_id}...", end=" ", flush=True)

    # 使用 wget 下載
    cmd = [
        "wget",
        "-r", "-N", "-c", "-np", "-nH",
        "--cut-dirs=5",
        f"--user={PHYSIONET_USER}",
        f"--password={PHYSIONET_PASS}",
        "-P", str(OUTPUT_DIR),
        "-q",  # 靜默模式
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓")
            return True
        else:
            print(f"✗ (錯誤: {result.stderr[:100]})")
            return False
    except subprocess.TimeoutExpired:
        print("✗ (超時)")
        return False
    except FileNotFoundError:
        print("✗ (找不到 wget，請先安裝)")
        return False


def check_wget():
    """檢查 wget 是否可用"""
    try:
        subprocess.run(["wget", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def main():
    print("=" * 60)
    print("MIMIC-CXR-JPG 影像下載器")
    print("=" * 60)

    # 檢查設定
    if PHYSIONET_USER == "your_username" or PHYSIONET_PASS == "your_password":
        print("\n❌ 錯誤: 請先編輯此檔案，填入你的 PhysioNet 帳號密碼")
        print(f"   檔案位置: {__file__}")
        print("\n   找到這兩行並修改:")
        print('   PHYSIONET_USER = "your_username"  # 改成你的帳號')
        print('   PHYSIONET_PASS = "your_password"  # 改成你的密碼')
        sys.exit(1)

    # 檢查 wget
    if not check_wget():
        print("\n❌ 錯誤: 找不到 wget")
        print("\n   Windows 安裝方式:")
        print("   1. 安裝 Git for Windows (包含 wget)")
        print("      https://git-scm.com/download/win")
        print("   或")
        print("   2. 使用 chocolatey: choco install wget")
        print("   或")
        print("   3. 使用 winget: winget install GnuWin32.Wget")
        sys.exit(1)

    # 判斷是否為重試模式
    retry_mode = "--retry" in sys.argv
    failed_file = OUTPUT_DIR / "failed_subjects.txt"

    if retry_mode:
        if not failed_file.exists():
            print("\n❌ 沒有找到失敗清單，不需要重試")
            sys.exit(0)
        subjects = failed_file.read_text().strip().split("\n")
        print(f"\n🔄 重試模式: 從失敗清單載入")
    else:
        if not SUBJECT_LIST.exists():
            print(f"\n❌ 錯誤: 找不到 subject 清單: {SUBJECT_LIST}")
            sys.exit(1)
        subjects = SUBJECT_LIST.read_text().strip().split("\n")

    total = len(subjects)

    print(f"\n輸出目錄: {OUTPUT_DIR}")
    print(f"待下載: {total} 位病患")
    print(f"帳號: {PHYSIONET_USER}")
    print()

    # 確認開始
    input("按 Enter 開始下載 (Ctrl+C 取消)...")
    print()

    # 建立輸出目錄
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 開始下載
    success = 0
    failed = []

    for i, subject_id in enumerate(subjects, 1):
        subject_id = subject_id.strip()
        if not subject_id:
            continue

        if download_subject(subject_id, i, total):
            success += 1
        else:
            failed.append(subject_id)

    # 結果摘要
    print()
    print("=" * 60)
    print("下載完成！")
    print("=" * 60)
    print(f"成功: {success}/{total}")
    print(f"失敗: {len(failed)}")

    if failed:
        failed_file = OUTPUT_DIR / "failed_subjects.txt"
        failed_file.write_text("\n".join(failed))
        print(f"失敗清單已儲存: {failed_file}")

    print(f"\n影像儲存於: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
