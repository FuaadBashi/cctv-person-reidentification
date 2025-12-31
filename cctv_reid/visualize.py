from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def to_gray_overlay(frame_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def gid_to_color(gid: int) -> Tuple[int, int, int]:
    """
    Deterministic per-identity colour (BGR) for better separation.
    """
    h = (gid * 37) % 180
    s = 200
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def format_label(
    global_id: int,
    track_id: int,
    display_has_face: bool,
    face_latched: bool = False,
    is_reentry: bool = False,
) -> str:
    mode = "FACE" if display_has_face else "BODY"
    # show an asterisk when FACE is shown due to latch (not current detection)
    if face_latched and display_has_face:
        mode = "FACE*"
    re = " RE" if is_reentry else ""
    return f"G{global_id} T{track_id} {mode}{re}"


def draw_box_with_label(
    frame_bgr: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    label: str,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
) -> None:
    if color is None:
        color = (255, 255, 255)

    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_bgr.shape[1] - 1, x2), min(frame_bgr.shape[0] - 1, y2)

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pad = 4
    y_text = max(0, y1 - 8)
    x_text = x1

    cv2.rectangle(
        frame_bgr,
        (x_text, y_text - th - pad),
        (x_text + tw + pad, y_text + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame_bgr,
        label,
        (x_text + 2, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
