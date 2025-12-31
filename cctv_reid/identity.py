from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = float(np.linalg.norm(x) + 1e-12)
    return x / n


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    # cosine distance = 1 - cosine similarity
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass
class MatchResult:
    gid: Optional[int]
    best_dist: float
    second_dist: float


class IdentityGallery:
    """
    Stores a rolling gallery of embeddings per global identity (GID).
    We keep separate face and body banks so we can match using either modality.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = int(max_size)
        self._faces: Dict[int, List[np.ndarray]] = {}
        self._bodies: Dict[int, List[np.ndarray]] = {}

    def add_face(self, gid: int, emb: np.ndarray) -> None:
        if emb is None or emb.size == 0:
            return
        emb = _l2_normalize(emb)
        bank = self._faces.setdefault(gid, [])
        bank.append(emb)
        if len(bank) > self.max_size:
            del bank[0 : len(bank) - self.max_size]

    def add_body(self, gid: int, emb: np.ndarray) -> None:
        if emb is None or emb.size == 0:
            return
        emb = _l2_normalize(emb)
        bank = self._bodies.setdefault(gid, [])
        bank.append(emb)
        if len(bank) > self.max_size:
            del bank[0 : len(bank) - self.max_size]

    def match_face(self, emb: np.ndarray) -> MatchResult:
        return self._match(emb, self._faces)

    def match_body(self, emb: np.ndarray) -> MatchResult:
        return self._match(emb, self._bodies)

    def _match(self, emb: np.ndarray, store: Dict[int, List[np.ndarray]]) -> MatchResult:
        if emb is None or emb.size == 0 or len(store) == 0:
            return MatchResult(None, 1e9, 1e9)

        emb = _l2_normalize(emb)

        best_gid: Optional[int] = None
        best = 1e9
        second = 1e9

        for gid, bank in store.items():
            if not bank:
                continue
            # nearest neighbour within that GID bank
            d = min(cosine_distance(emb, e) for e in bank)

            if d < best:
                second = best
                best = d
                best_gid = gid
            elif d < second:
                second = d

        return MatchResult(best_gid, best, second)


class IdentityManager:
    """
    Assigns a stable global identity (GID) to transient tracker IDs (TID).

    Matching logic:
      1) If TID already has a GID -> keep it ("existing")
      2) If face embedding available -> face match if dist <= face_thresh
      3) Else body embedding -> body match if dist <= body_thresh AND (second-best - best) >= body_margin
      4) Else create new GID
    """

    def __init__(
        self,
        gallery: IdentityGallery,
        face_thresh: Optional[float] = None,
        body_thresh: Optional[float] = None,
        body_margin: float = 0.0,
        face_latch_frames: int = 0,  # kept for config compatibility; latch is handled in pipeline UI
        # Backward-compatible aliases (if older code calls these):
        face_dist_thresh: Optional[float] = None,
        body_dist_thresh: Optional[float] = None,
    ):
        self.gallery = gallery

        # Accept either naming style
        if face_thresh is None:
            face_thresh = face_dist_thresh if face_dist_thresh is not None else 0.55
        if body_thresh is None:
            body_thresh = body_dist_thresh if body_dist_thresh is not None else 0.40

        self.face_thresh = float(face_thresh)
        self.body_thresh = float(body_thresh)
        self.body_margin = float(body_margin)

        self.track_to_gid: Dict[int, int] = {}
        self._next_gid = 1

    def assign_gid(
        self,
        track_id: int,
        face_emb: Optional[np.ndarray],
        body_emb: Optional[np.ndarray],
    ) -> Tuple[int, str, float]:
        # If already assigned, keep stable identity
        if track_id in self.track_to_gid:
            return self.track_to_gid[track_id], "existing", 0.0

        # 1) Face match (preferred)
        if face_emb is not None and face_emb.size > 0:
            m = self.gallery.match_face(face_emb)
            if m.gid is not None and m.best_dist <= self.face_thresh:
                self.track_to_gid[track_id] = m.gid
                return m.gid, "face_match", float(m.best_dist)

        # 2) Body match (fallback)
        if body_emb is not None and body_emb.size > 0:
            m = self.gallery.match_body(body_emb)
            if m.gid is not None and m.best_dist <= self.body_thresh:
                # margin test improves identity separation (reduces "sticky" wrong matches)
                if (m.second_dist - m.best_dist) >= self.body_margin:
                    self.track_to_gid[track_id] = m.gid
                    return m.gid, "body_match", float(m.best_dist)

        # 3) New identity
        gid = self._next_gid
        self._next_gid += 1
        self.track_to_gid[track_id] = gid
        return gid, "new", 0.0

    def update_embeddings(
        self,
        track_id: int,
        face_emb: Optional[np.ndarray],
        body_emb: Optional[np.ndarray],
    ) -> None:
        gid = self.track_to_gid.get(track_id)
        if gid is None:
            return
        if face_emb is not None and face_emb.size > 0:
            self.gallery.add_face(gid, face_emb)
        if body_emb is not None and body_emb.size > 0:
            self.gallery.add_body(gid, body_emb)
