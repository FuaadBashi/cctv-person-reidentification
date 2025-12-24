from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path

    det_model: str = "yolov8n.pt"
    det_conf: float = 0.35
    det_iou: float = 0.5

    tracker_max_age: int = 45
    tracker_n_init: int = 3

    face_enabled: bool = True
    body_enabled: bool = True
    face_every: int = 3
    body_every: int = 2
    face_dist_thresh: float = 0.35
    body_dist_thresh: float = 0.45

    max_gallery: int = 2000

    show: bool = False
    stride: int = 1
