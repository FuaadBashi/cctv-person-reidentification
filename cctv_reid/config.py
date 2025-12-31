from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path

    # Detector (YOLO)
    det_model: str = "yolov8n.pt"
    det_conf: float = 0.35
    det_iou: float = 0.5

    # Small-person rescue mode (only triggers when full-frame finds too few people)
    det_imgsz: int = 960              # normal pass inference size
    det_min_full_dets: int = 1        # if < this, run rescue pass
    det_small_conf: float = 0.15      # rescue pass confidence
    det_small_imgsz: int = 1536       # rescue pass inference size (bigger helps tiny people)

    det_tile_enabled: bool = True     # rescue tiling on/off
    det_tile_size: int = 960          # tile crop size (px)
    det_tile_overlap: float = 0.20    # 0.0-0.5 typical
    det_tile_nms_iou: float = 0.50    # merge overlaps from tiles

    # Tracker (DeepSORT)
    tracker_max_age: int = 45
    tracker_n_init: int = 3

    # Face/Body ReID
    face_enabled: bool = True
    body_enabled: bool = True
    face_every: int = 3
    body_every: int = 2
    face_dist_thresh: float = 0.35
    body_dist_thresh: float = 0.45
    max_gallery: int = 2000

    # UI / speed
    show: bool = False
    stride: int = 1
