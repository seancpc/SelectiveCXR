"""
src/anatomy/segmenter.py — CXR 解剖分割(定義 A 的「可靠定位」來源)

用 TorchXRayVision PSPNet 把 CXR 分割成 14 個解剖結構 mask,取代 VLM 不可靠的 bbox。

PSPNet 14 類(已於 scripts/test_anatomy.py 驗證):
  Left/Right Clavicle, Left/Right Scapula, Left/Right Lung,
  Left/Right Hilus Pulmonis, Heart, Aorta, Facies Diaphragmatica(橫膈),
  Mediastinum, Weasand(食道), Spine。

需 GPU(輕量,4090 即時)。Apache-2.0 授權,可公開展示。
"""

from __future__ import annotations

import sys as _sys

import numpy as np
import torch
from PIL import Image

# torchxrayvision 的 jfhealthcare 子模組 import 時會把自身目錄插進 sys.path 最前面,
# 蓋過專案根的 config.py(害其他模組 `from config import` 解析到 torchxrayvision 的 config)。
# 保存並還原 sys.path,隔離這個污染。
_saved_syspath = list(_sys.path)
import torchxrayvision as xrv
_sys.path[:] = _saved_syspath


class AnatomySegmenter:
    """PSPNet 解剖分割 wrapper。"""

    def __init__(self, device: str = "cuda", resize: int = 512):
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
        self.targets = list(self.model.targets)            # 14 類名稱
        self.name_to_idx = {t: i for i, t in enumerate(self.targets)}
        self.device = device
        self.resize = resize
        self._resizer = xrv.datasets.XRayResizer(resize)

    @torch.no_grad()
    def segment(self, image_path) -> np.ndarray:
        """CXR → (14, resize, resize) 解剖機率 mask (0-1)。"""
        img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        img = xrv.datasets.normalize(img, 255)             # 8-bit → [-1024,1024]
        img = self._resizer(img[None, ...])                # (1, R, R)
        x = torch.from_numpy(img)[None, ...].to(self.device)   # (1,1,R,R)
        out = self.model(x)
        return torch.sigmoid(out)[0].cpu().numpy()         # (14, R, R)

    def region_mask(self, masks: np.ndarray, anatomy_names: list[str],
                    thr: float = 0.5) -> np.ndarray:
        """合併多個解剖類的二值 mask → 一個區域 mask (R, R) bool。"""
        idxs = [self.name_to_idx[n] for n in anatomy_names if n in self.name_to_idx]
        if not idxs:
            return np.zeros((self.resize, self.resize), dtype=bool)
        return (masks[idxs] > thr).any(axis=0)

    def lung_mask(self, masks: np.ndarray, thr: float = 0.5) -> np.ndarray:
        """左右肺合併 mask(供肺尖/肋膈角幾何推導)。"""
        return self.region_mask(masks, ["Left Lung", "Right Lung"], thr)
