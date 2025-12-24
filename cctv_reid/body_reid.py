from __future__ import annotations

"""Body re-identification (appearance embedding).

This project originally attempted to use `torchreid.utils.FeatureExtractor`,
but the PyPI package named `torchreid` is not the same as the widely-used
`deep-person-reid` project and does not expose that API. On macOS this
frequently leads to `ModuleNotFoundError: torchreid.utils`.

To keep the prototype robust and easy to run for a recruitment task, we use a
dependency-light fallback: a pretrained torchvision ResNet-50 backbone as a
global appearance embedding.

Notes:
  - This is NOT clothing-invariant ReID. It is a strong baseline fallback when
    faces are not visible.
  - Embeddings are L2-normalised for cosine distance matching.
"""

from typing import Optional

import numpy as np
import cv2

import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


class ResNetBodyReID:
    """Lightweight body embedding using torchvision ResNet-50.

    Produces a 2048-d embedding from a person crop.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove classifier head -> 2048-d pooled features.
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        self.model = model

        # Common ReID preprocessing: (H, W) = (256, 128)
        self.tf = T.Compose(
            [
                T.ToTensor(),
                T.Resize((256, 128)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def embed_crop(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        rgb = _to_rgb(crop_bgr)
        x = self.tf(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)

        # L2 normalise
        feat /= (np.linalg.norm(feat) + 1e-12)
        return feat
