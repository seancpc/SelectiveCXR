"""
scripts/prepare_splits.py — 資料切分(patient-level)

把 MIMIC-CXR 切成 train / calibration / test 三份。
強制 patient-level:同一病人(pXXXXXXXX)的所有影像落在同一 split,避免洩漏。
conformal 需要獨立的 calibration set。

用法(於專案根目錄):python -m scripts.prepare_splits
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# 允許從專案根目錄 import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MIMIC_CXR_DIR, SPLITS_DIR, SPLIT_RATIOS, SPLIT_SEED  # noqa: E402


def collect_patients(mimic_dir: Path) -> list[str]:
    """掃描 MIMIC-CXR 目錄,回傳所有病人目錄名(pXXXXXXXX)。

    相容兩種結構:病人目錄直接位於 mimic_dir 下,或位於群組目錄(如 p10)之下。
    """
    patients: set[str] = set()
    for entry in sorted(mimic_dir.glob("p*")):
        if not entry.is_dir():
            continue
        name = entry.name
        if len(name) > 3 and name[1:].isdigit():
            patients.add(name)                       # 病人目錄
        else:
            for sub in sorted(entry.glob("p*")):     # 群組目錄 → 往下一層
                if sub.is_dir() and len(sub.name) > 3 and sub.name[1:].isdigit():
                    patients.add(sub.name)
    return sorted(patients)


def split_patients(patients: list[str], ratios: dict, seed: int) -> dict[str, list[str]]:
    """patient-level 切分。"""
    patients = sorted(set(patients))
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(n * ratios["train"])
    n_calib = int(n * ratios["calibration"])
    return {
        "train": sorted(patients[:n_train]),
        "calibration": sorted(patients[n_train:n_train + n_calib]),
        "test": sorted(patients[n_train + n_calib:]),
    }


def main() -> None:
    if not MIMIC_CXR_DIR.exists():
        raise SystemExit(f"找不到 MIMIC-CXR 目錄:{MIMIC_CXR_DIR}")

    patients = collect_patients(MIMIC_CXR_DIR)
    if not patients:
        raise SystemExit(f"{MIMIC_CXR_DIR} 下未找到 patient 目錄(pXXXXXXXX)")

    splits = split_patients(patients, SPLIT_RATIOS, SPLIT_SEED)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, plist in splits.items():
        out = SPLITS_DIR / f"{name}.json"
        out.write_text(json.dumps(plist, indent=2), encoding="utf-8")
        print(f"  {name:12s}: {len(plist):6d} patients -> {out}")

    # 檢核:三 split 無重疊
    s = {k: set(v) for k, v in splits.items()}
    assert not (s["train"] & s["calibration"]), "train / calibration 重疊"
    assert not (s["train"] & s["test"]), "train / test 重疊"
    assert not (s["calibration"] & s["test"]), "calibration / test 重疊"
    print(f"  總計 {len(patients)} patients,三 split 無重疊 (patient-level) OK")


if __name__ == "__main__":
    main()
