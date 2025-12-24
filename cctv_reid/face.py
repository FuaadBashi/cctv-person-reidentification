from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


@dataclass(frozen=True)
class FaceDet:
    bbox_xyxy: Tuple[float, float, float, float]
    embedding: np.ndarray


class InsightFaceModule:
    def __init__(self):
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise ImportError("insightface is required. pip install insightface onnxruntime") from e
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(1280, 1280))

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDet]:
        faces = self.app.get(frame_bgr)
        out: List[FaceDet] = []
        for f in faces:
            x1, y1, x2, y2 = [float(v) for v in f.bbox.tolist()]
            out.append(FaceDet(bbox_xyxy=(x1, y1, x2, y2), embedding=f.embedding.astype(np.float32)))
        return out

    @staticmethod
    def assign_faces_to_tracks(
        faces: List[FaceDet],
        tracks_xyxy: Dict[int, Tuple[float, float, float, float]],
    ) -> Dict[int, FaceDet]:
        def iou(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            return inter / (area_a + area_b - inter + 1e-9)

        assigned: Dict[int, FaceDet] = {}
        best: Dict[int, float] = {}
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox_xyxy
            cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            for tid, tb in tracks_xyxy.items():
                x1, y1, x2, y2 = tb
                if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                    continue
                s = iou(face.bbox_xyxy, tb)
                if tid not in best or s > best[tid]:
                    best[tid] = s
                    assigned[tid] = face
        return assigned
