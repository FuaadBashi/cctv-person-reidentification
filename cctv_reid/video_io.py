from __future__ import annotations

import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple


@dataclass(frozen=True)
class FrameItem:
    index: int
    time_s: float
    frame_bgr: "cv2.Mat"


class VideoReader:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[FrameItem]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield FrameItem(index=idx, time_s=idx / self.fps, frame_bgr=frame)
            idx += 1

    def release(self) -> None:
        self.cap.release()


class VideoWriter:
    def __init__(self, path: Path, fps: float, size: Tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer at: {path}")

    def write(self, frame_bgr: "cv2.Mat") -> None:
        self.writer.write(frame_bgr)

    def release(self) -> None:
        self.writer.release()
