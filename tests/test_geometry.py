import numpy as np
import pytest

from cctv_reid.detector import _iou_xyxy, _nms_xyxy, tile_coords


def test_iou_of_identical_boxes_is_one_and_of_disjoint_boxes_is_zero():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)

    iou = _iou_xyxy(a, b)

    assert iou.shape == (1, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(0.0)


def test_iou_of_half_overlapping_boxes_is_one_third():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)

    assert _iou_xyxy(a, b)[0, 0] == pytest.approx(50 / 150)


def test_nms_keeps_the_best_of_overlapping_boxes_and_all_separate_ones():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
        dtype=np.float32,
    )
    scores = np.array([0.6, 0.9, 0.8], dtype=np.float32)

    assert sorted(_nms_xyxy(boxes, scores, iou_thr=0.5)) == [1, 2]


def test_nms_of_no_boxes_keeps_nothing():
    assert _nms_xyxy(np.empty((0, 4), np.float32), np.empty(0, np.float32), 0.5) == []


@pytest.mark.parametrize(("w", "h"), [(1920, 1080), (1000, 700), (500, 400), (960, 960)])
def test_tiles_cover_every_pixel_of_the_frame(w, h):
    covered = np.zeros((h, w), dtype=bool)

    for x1, y1, x2, y2 in tile_coords(w, h, tile_size=960, overlap=0.2):
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        covered[y1:y2, x1:x2] = True

    assert covered.all()


def test_neighbouring_tiles_overlap_by_roughly_the_requested_fraction():
    tiles = tile_coords(3000, 960, tile_size=1000, overlap=0.2)
    xs = sorted({x1 for x1, _, _, _ in tiles})

    assert xs[1] - xs[0] == 800
