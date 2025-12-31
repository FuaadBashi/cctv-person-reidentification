from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from ultralytics import YOLO
import cv2
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection:
    """
    Standard detection container used by tracker.
    bbox_xyxy: (x1, y1, x2, y2)
    confidence: detection confidence
    feature: optional appearance embedding
    """
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    feature: Optional[np.ndarray] = None


@dataclass
class Det:
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float

    # backward compatibility
    @property
    def conf(self) -> float:
        return self.confidence



def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N,4], b: [M,4] -> IoU [N,M]
    """
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = (iw * ih).reshape(a.shape[0], -1)

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a.reshape(-1, 1) + area_b.reshape(1, -1) - inter + 1e-9
    return inter / union


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """
    Standard greedy NMS. Returns kept indices.
    """
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_xyxy(boxes[i : i + 1], boxes[rest]).reshape(-1)
        rest = rest[ious <= iou_thr]
        order = rest

    return keep


class PersonDetectorYOLO:
    """
    Person detector with a safe "small person rescue" mode:
    - Run a normal full-frame pass (keeps your existing behaviour)
    - If detections are too few, run a second pass:
        - higher imgsz
        - optional tiling (better for tiny subjects far from camera)
    """

    def __init__(
        self,
        model_name: str,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 960,
        min_full_dets: int = 1,
        small_conf: float = 0.15,
        small_imgsz: int = 1536,
        tile_enabled: bool = True,
        tile_size: int = 960,
        tile_overlap: float = 0.20,
        tile_nms_iou: float = 0.50,
    ):
        self.model = YOLO(model_name)
        self.conf = float(conf)
        self.iou = float(iou)

        self.imgsz = int(imgsz)
        self.min_full_dets = int(min_full_dets)

        self.small_conf = float(small_conf)
        self.small_imgsz = int(small_imgsz)

        self.tile_enabled = bool(tile_enabled)
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.tile_nms_iou = float(tile_nms_iou)

        # COCO class 0 = person
        self.person_class = [0]

    def _predict(self, frame_bgr: np.ndarray, conf: float, imgsz: int) -> List[Det]:
        res = self.model.predict(
            source=frame_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=self.iou,
            classes=self.person_class,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        dets: List[Det] = []
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            dets.append(Det((float(x1), float(y1), float(x2), float(y2)), float(c)))
        return dets

    def _tile_coords(self, W: int, H: int) -> List[Tuple[int, int, int, int]]:
        ts = self.tile_size
        ov = max(0.0, min(0.5, self.tile_overlap))
        step = max(1, int(round(ts * (1.0 - ov))))

        xs = list(range(0, max(1, W - ts + 1), step))
        ys = list(range(0, max(1, H - ts + 1), step))

        # ensure right/bottom coverage
        if len(xs) == 0:
            xs = [0]
        if len(ys) == 0:
            ys = [0]
        if xs[-1] != max(0, W - ts):
            xs.append(max(0, W - ts))
        if ys[-1] != max(0, H - ts):
            ys.append(max(0, H - ts))

        coords: List[Tuple[int, int, int, int]] = []
        for y0 in ys:
            for x0 in xs:
                x1 = x0
                y1 = y0
                x2 = min(W, x0 + ts)
                y2 = min(H, y0 + ts)
                coords.append((x1, y1, x2, y2))
        return coords

    def _predict_tiled(self, frame_bgr: np.ndarray) -> List[Det]:
        H, W = frame_bgr.shape[:2]
        coords = self._tile_coords(W, H)

        all_boxes: List[Tuple[float, float, float, float]] = []
        all_scores: List[float] = []

        for (x1, y1, x2, y2) in coords:
            tile = frame_bgr[y1:y2, x1:x2]
            dets = self._predict(tile, conf=self.small_conf, imgsz=self.small_imgsz)

            for d in dets:
                tx1, ty1, tx2, ty2 = d.bbox_xyxy
                gx1 = float(tx1 + x1)
                gy1 = float(ty1 + y1)
                gx2 = float(tx2 + x1)
                gy2 = float(ty2 + y1)
                all_boxes.append((gx1, gy1, gx2, gy2))
                all_scores.append(float(d.conf))

        if not all_boxes:
            return []

        boxes = np.array(all_boxes, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)
        keep = _nms_xyxy(boxes, scores, iou_thr=self.tile_nms_iou)

        return [Det(tuple(map(float, boxes[i])), float(scores[i])) for i in keep]

    def detect(self, frame_bgr: np.ndarray) -> List[Det]:
        # 1) Normal behaviour (keeps your other videos stable)
        dets = self._predict(frame_bgr, conf=self.conf, imgsz=self.imgsz)
        if len(dets) >= self.min_full_dets:
            return dets

        # 2) Rescue mode (only when full-frame found too few)
        if self.tile_enabled:
            tiled = self._predict_tiled(frame_bgr)
            if len(tiled) > 0:
                return tiled

        # 3) Last resort: full-frame with small-person params
        return self._predict(frame_bgr, conf=self.small_conf, imgsz=self.small_imgsz)
