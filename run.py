from __future__ import annotations

import argparse
from pathlib import Path

from cctv_reid.config import PipelineConfig
from cctv_reid.pipeline import VideoReIDPipeline


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CCTV Person Tracking + ReID (single camera)")
    p.add_argument("--input", required=True, help="Path to input video (e.g., mp4)")
    p.add_argument("--output_dir", required=True, help="Output directory for run artifacts")

    # Detector
    p.add_argument("--det_model", default="yolov8n.pt", help="Ultralytics YOLO model (e.g., yolov8n.pt, yolov8s.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="Full-frame person detection confidence")
    p.add_argument("--iou", type=float, default=0.5, help="Detector NMS IoU threshold")
    p.add_argument("--det_imgsz", type=int, default=960, help="Full-frame inference size (bigger helps small people)")

    # Small-person rescue mode
    p.add_argument("--min_full_dets", type=int, default=1, help="If full-frame detections < this, run rescue pass")
    p.add_argument("--small_conf", type=float, default=0.15, help="Rescue confidence (lower detects small people, more FPs)")
    p.add_argument("--small_imgsz", type=int, default=1536, help="Rescue inference size")
    p.add_argument("--no_tile", action="store_true", help="Disable tiling in rescue mode")
    p.add_argument("--tile_size", type=int, default=960, help="Tile size (px) for rescue mode")
    p.add_argument("--tile_overlap", type=float, default=0.20, help="Tile overlap fraction (0.0-0.5)")
    p.add_argument("--tile_nms_iou", type=float, default=0.50, help="NMS IoU used to merge tiled detections")

    # Tracker
    p.add_argument("--max_age", type=int, default=45, help="Tracker max age (frames) before track is deleted")
    p.add_argument("--n_init", type=int, default=3, help="Tracker n_init (confirm after this many hits)")

    # ReID thresholds
    p.add_argument("--reid_face_thresh", type=float, default=0.35, help="Cosine distance threshold for face match")
    p.add_argument("--reid_body_thresh", type=float, default=0.45, help="Cosine distance threshold for body match")

    # Face/body compute frequency
    p.add_argument("--face_every", type=int, default=3, help="Run face analysis every N frames")
    p.add_argument("--body_every", type=int, default=2, help="Compute body embedding every N frames per track")

    p.add_argument("--max_gallery", type=int, default=2000, help="Max identities stored in gallery (safety cap)")
    p.add_argument("--show", action="store_true", help="Show live window while processing (ESC to quit)")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame (1 = all frames)")
    p.add_argument("--no_face", action="store_true", help="Disable face recognition module")
    p.add_argument("--no_body", action="store_true", help="Disable body ReID module")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = PipelineConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),

        det_model=args.det_model,
        det_conf=args.conf,
        det_iou=args.iou,
        det_imgsz=args.det_imgsz,

        det_min_full_dets=args.min_full_dets,
        det_small_conf=args.small_conf,
        det_small_imgsz=args.small_imgsz,
        det_tile_enabled=(not args.no_tile),
        det_tile_size=args.tile_size,
        det_tile_overlap=args.tile_overlap,
        det_tile_nms_iou=args.tile_nms_iou,

        tracker_max_age=args.max_age,
        tracker_n_init=args.n_init,

        face_enabled=not args.no_face,
        body_enabled=not args.no_body,
        face_every=args.face_every,
        body_every=args.body_every,
        face_dist_thresh=args.reid_face_thresh,
        body_dist_thresh=args.reid_body_thresh,
        max_gallery=args.max_gallery,

        show=args.show,
        stride=max(1, args.stride),
    )

    VideoReIDPipeline(cfg).run()


if __name__ == "__main__":
    main()
