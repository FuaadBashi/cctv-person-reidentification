from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .video_io import VideoReader, VideoWriter
from .detector import PersonDetectorYOLO
from .tracker import DeepSortTracker
from .identity import IdentityGallery, IdentityManager
from .visualize import to_gray_overlay, draw_box_with_label
from .face import InsightFaceModule, FaceDet
from .body_reid import ResNetBodyReID


def crop_xyxy(frame: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1i = max(0, min(w - 1, int(x1)))
    y1i = max(0, min(h - 1, int(y1)))
    x2i = max(0, min(w, int(x2)))
    y2i = max(0, min(h, int(y2)))
    if x2i <= x1i or y2i <= y1i:
        return np.zeros((0, 0, 3), dtype=frame.dtype)
    return frame[y1i:y2i, x1i:x2i]


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.f.close()


class VideoReIDPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        self.detector = PersonDetectorYOLO(cfg.det_model, conf=cfg.det_conf, iou=cfg.det_iou)
        self.tracker = DeepSortTracker(max_age=cfg.tracker_max_age, n_init=cfg.tracker_n_init)

        self.face = InsightFaceModule() if cfg.face_enabled else None
        # Body ReID fallback is a torchvision ResNet-50 appearance embedding.
        self.body = ResNetBodyReID(device="cpu") if cfg.body_enabled else None

        self.gallery = IdentityGallery(max_size=cfg.max_gallery)
        self.id_manager = IdentityManager(self.gallery, cfg.face_dist_thresh, cfg.body_dist_thresh)

        self.annotated_path = cfg.output_dir / "annotated.mp4"
        self.tracks_jsonl = JsonlWriter(cfg.output_dir / "tracks.jsonl")
        self.events_jsonl = JsonlWriter(cfg.output_dir / "events.jsonl")
        self.tracks_csv_path = cfg.output_dir / "tracks.csv"

        self._csv_file = self.tracks_csv_path.open("w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=[
            "frame", "time_s", "track_id", "global_id", "x1", "y1", "x2", "y2", "conf",
            "is_confirmed", "tsu", "has_face", "assign_reason", "assign_dist"
        ])
        self._csv.writeheader()

        self._body_cache: Dict[int, Tuple[int, np.ndarray]] = {}

    def _log_event(self, kind: str, payload: dict) -> None:
        self.events_jsonl.write({"ts": time.time(), "kind": kind, **payload})

    def run(self) -> None:
        cfg = self.cfg
        vr = VideoReader(cfg.input_path)
        vw = VideoWriter(self.annotated_path, fps=vr.fps / cfg.stride, size=(vr.width, vr.height))

        pbar = tqdm(total=vr.frame_count, desc="Processing", unit="frame")
        alive_prev: set[int] = set()

        try:
            for item in vr:
                pbar.update(1)
                if cfg.stride > 1 and (item.index % cfg.stride != 0):
                    continue

                frame = item.frame_bgr
                frame_idx = item.index
                time_s = item.time_s

                dets = self.detector.detect(frame)
                tracks = self.tracker.update(dets, frame_bgr=frame)
                tracks_conf = [t for t in tracks if t.is_confirmed and t.time_since_update == 0]

                face_map: Dict[int, FaceDet] = {}
                if self.face is not None and (frame_idx % max(1, cfg.face_every) == 0):
                    faces = self.face.detect(frame)
                    face_map = self.face.assign_faces_to_tracks(faces, {t.track_id: t.bbox_xyxy for t in tracks_conf})

                current_alive = {t.track_id for t in tracks_conf}
                dead = alive_prev - current_alive
                for tid in dead:
                    gid = self.id_manager.track_to_gid.get(tid)
                    if gid is not None:
                        self._log_event("EXIT", {"frame": frame_idx, "track_id": tid, "global_id": gid})
                alive_prev = current_alive

                annotated = to_gray_overlay(frame.copy())
                per_frame_records: List[dict] = []

                for t in tracks_conf:
                    tid = t.track_id
                    face_emb = face_map.get(tid).embedding if tid in face_map else None

                    body_emb: Optional[np.ndarray] = None
                    if self.body is not None:
                        cached = self._body_cache.get(tid)
                        if cached is None or (frame_idx - cached[0]) >= max(1, cfg.body_every):
                            crop = crop_xyxy(frame, t.bbox_xyxy)
                            emb = self.body.embed_crop(crop)
                            if emb is not None:
                                self._body_cache[tid] = (frame_idx, emb)
                                body_emb = emb
                        else:
                            body_emb = cached[1]

                    gid, reason, dist = self.id_manager.assign_gid(tid, face_emb, body_emb)
                    self.id_manager.update_embeddings(tid, face_emb, body_emb)

                    if reason in {"new", "face_match", "body_match"}:
                        self._log_event("ENTER_OR_MATCH", {
                            "frame": frame_idx,
                            "time_s": time_s,
                            "track_id": tid,
                            "global_id": gid,
                            "reason": reason,
                            "distance": dist,
                            "has_face": face_emb is not None,
                        })

                    label = f"GID {gid} | T{tid}" + (" | FACE" if face_emb is not None else "")
                    draw_box_with_label(annotated, t.bbox_xyxy, label)

                    x1, y1, x2, y2 = t.bbox_xyxy
                    rec = {
                        "frame": frame_idx,
                        "time_s": time_s,
                        "track_id": tid,
                        "global_id": gid,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "conf": t.confidence,
                        "is_confirmed": t.is_confirmed,
                        "tsu": t.time_since_update,
                        "has_face": face_emb is not None,
                        "assign_reason": reason,
                        "assign_dist": dist,
                    }
                    per_frame_records.append(rec)

                    self._csv.writerow({
                        "frame": frame_idx,
                        "time_s": time_s,
                        "track_id": tid,
                        "global_id": gid,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": t.confidence,
                        "is_confirmed": int(t.is_confirmed),
                        "tsu": t.time_since_update,
                        "has_face": int(face_emb is not None),
                        "assign_reason": reason,
                        "assign_dist": dist,
                    })

                self.tracks_jsonl.write({"frame": frame_idx, "time_s": time_s, "tracks": per_frame_records})
                vw.write(annotated)

                if cfg.show:
                    cv2.imshow("CCTV ReID (ESC to quit)", annotated)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        self._log_event("USER_ABORT", {"frame": frame_idx})
                        break

        finally:
            vr.release()
            vw.release()
            self.tracks_jsonl.close()
            self.events_jsonl.close()
            self._csv_file.close()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            pbar.close()
