from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from .detector import Detection


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    is_confirmed: bool
    time_since_update: int


def _xyxy_to_ltwh(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


class DeepSortTracker:
    def __init__(self, max_age: int = 45, n_init: int = 3):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except Exception as e:
            raise ImportError("deep_sort_realtime is required. pip install deep_sort_realtime") from e
        self.ds = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections: List[Detection], frame_bgr: Optional[np.ndarray] = None) -> List[Track]:
        ds_dets = [(_xyxy_to_ltwh(d.bbox_xyxy), float(d.confidence), "person") for d in detections]
        tracks = self.ds.update_tracks(ds_dets, frame=frame_bgr)
        out: List[Track] = []
        for trk in tracks:
            l, t, r, b = trk.to_ltrb()
            out.append(
                Track(
                    track_id=int(trk.track_id),
                    bbox_xyxy=(float(l), float(t), float(r), float(b)),
                    confidence=float(getattr(trk, "det_conf", 1.0) or 1.0),
                    is_confirmed=bool(trk.is_confirmed()),
                    time_since_update=int(trk.time_since_update),
                )
            )
        return out
