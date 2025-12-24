from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float


class PersonDetectorYOLO:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35, iou: float = 0.5):
        self.conf = float(conf)
        self.iou = float(iou)
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError("ultralytics is required. pip install ultralytics") from e
        self.model = YOLO(model_name)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        dets: List[Detection] = []
        if res.boxes is None:
            return dets
        for b in res.boxes:
            if int(b.cls.item()) != 0:  # COCO person
                continue
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            dets.append(Detection(bbox_xyxy=(x1, y1, x2, y2), confidence=conf))
        return dets
