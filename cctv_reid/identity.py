from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import time

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - (np.dot(a, b) / denom))


@dataclass
class IdentityRecord:
    gid: int
    created_ts: float
    last_seen_ts: float
    face_proto: Optional[np.ndarray] = None
    body_proto: Optional[np.ndarray] = None
    face_count: int = 0
    body_count: int = 0


class IdentityGallery:
    def __init__(self, max_size: int = 2000):
        self.max_size = int(max_size)
        self._gid_next = 1
        self.records: Dict[int, IdentityRecord] = {}

    def new_identity(self) -> int:
        if len(self.records) >= self.max_size:
            oldest_gid = min(self.records.values(), key=lambda r: r.last_seen_ts).gid
            del self.records[oldest_gid]
        gid = self._gid_next
        self._gid_next += 1
        now = time.time()
        self.records[gid] = IdentityRecord(gid=gid, created_ts=now, last_seen_ts=now)
        return gid

    def touch(self, gid: int) -> None:
        if gid in self.records:
            self.records[gid].last_seen_ts = time.time()

    def _update_proto(self, proto: Optional[np.ndarray], count: int, emb: np.ndarray) -> Tuple[np.ndarray, int]:
        emb = emb.astype(np.float32)
        if proto is None:
            return emb, 1
        n = count + 1
        return (proto * count + emb) / n, n

    def update_face(self, gid: int, emb: np.ndarray) -> None:
        rec = self.records[gid]
        rec.last_seen_ts = time.time()
        rec.face_proto, rec.face_count = self._update_proto(rec.face_proto, rec.face_count, emb)

    def update_body(self, gid: int, emb: np.ndarray) -> None:
        rec = self.records[gid]
        rec.last_seen_ts = time.time()
        rec.body_proto, rec.body_count = self._update_proto(rec.body_proto, rec.body_count, emb)

    def best_match_face(self, emb: np.ndarray) -> Tuple[Optional[int], float]:
        best_gid, best_d = None, 1e9
        for gid, rec in self.records.items():
            if rec.face_proto is None:
                continue
            d = cosine_distance(emb, rec.face_proto)
            if d < best_d:
                best_gid, best_d = gid, d
        return best_gid, best_d

    def best_match_body(self, emb: np.ndarray) -> Tuple[Optional[int], float]:
        best_gid, best_d = None, 1e9
        for gid, rec in self.records.items():
            if rec.body_proto is None:
                continue
            d = cosine_distance(emb, rec.body_proto)
            if d < best_d:
                best_gid, best_d = gid, d
        return best_gid, best_d


class IdentityManager:
    def __init__(self, gallery: IdentityGallery, face_dist_thresh: float = 0.35, body_dist_thresh: float = 0.45):
        self.gallery = gallery
        self.face_dist_thresh = float(face_dist_thresh)
        self.body_dist_thresh = float(body_dist_thresh)
        self.track_to_gid: Dict[int, int] = {}

    def assign_gid(
        self,
        track_id: int,
        face_emb: Optional[np.ndarray],
        body_emb: Optional[np.ndarray],
    ) -> Tuple[int, str, float]:
        if track_id in self.track_to_gid:
            gid = self.track_to_gid[track_id]
            self.gallery.touch(gid)
            return gid, "existing", 0.0

        if face_emb is not None:
            gid, d = self.gallery.best_match_face(face_emb)
            if gid is not None and d <= self.face_dist_thresh:
                self.track_to_gid[track_id] = gid
                self.gallery.update_face(gid, face_emb)
                return gid, "face_match", d

        if body_emb is not None:
            gid, d = self.gallery.best_match_body(body_emb)
            if gid is not None and d <= self.body_dist_thresh:
                self.track_to_gid[track_id] = gid
                self.gallery.update_body(gid, body_emb)
                return gid, "body_match", d

        gid = self.gallery.new_identity()
        self.track_to_gid[track_id] = gid
        if face_emb is not None:
            self.gallery.update_face(gid, face_emb)
        if body_emb is not None:
            self.gallery.update_body(gid, body_emb)
        return gid, "new", 0.0

    def update_embeddings(self, track_id: int, face_emb: Optional[np.ndarray], body_emb: Optional[np.ndarray]) -> None:
        if track_id not in self.track_to_gid:
            return
        gid = self.track_to_gid[track_id]
        if face_emb is not None:
            self.gallery.update_face(gid, face_emb)
        if body_emb is not None:
            self.gallery.update_body(gid, body_emb)
