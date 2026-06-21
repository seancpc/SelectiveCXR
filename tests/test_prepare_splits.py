"""scripts/prepare_splits.py 單元測試。"""

from scripts.prepare_splits import split_patients, collect_patients

_RATIOS = {"train": 0.6, "calibration": 0.2, "test": 0.2}


def test_split_ratios():
    patients = [f"p{i:08d}" for i in range(100)]
    splits = split_patients(patients, _RATIOS, seed=42)
    assert len(splits["train"]) == 60
    assert len(splits["calibration"]) == 20
    assert len(splits["test"]) == 20


def test_split_no_overlap_and_complete():
    patients = [f"p{i:08d}" for i in range(100)]
    splits = split_patients(patients, _RATIOS, seed=42)
    s_train = set(splits["train"])
    s_cal = set(splits["calibration"])
    s_test = set(splits["test"])
    assert not (s_train & s_cal)
    assert not (s_train & s_test)
    assert not (s_cal & s_test)
    assert len(s_train | s_cal | s_test) == 100


def test_split_deterministic():
    patients = [f"p{i:08d}" for i in range(50)]
    assert split_patients(patients, _RATIOS, 42) == split_patients(patients, _RATIOS, 42)


def test_collect_patients_flat(tmp_path):
    """病人目錄直接位於 mimic_dir 下。"""
    for pid in ["p10000001", "p10000002", "p10000003"]:
        (tmp_path / pid).mkdir()
    assert collect_patients(tmp_path) == ["p10000001", "p10000002", "p10000003"]


def test_collect_patients_grouped(tmp_path):
    """病人目錄位於群組目錄(p10)之下。"""
    grp = tmp_path / "p10"
    grp.mkdir()
    for pid in ["p10000004", "p10000005"]:
        (grp / pid).mkdir()
    found = collect_patients(tmp_path)
    assert "p10000004" in found and "p10000005" in found
