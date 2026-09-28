import numpy as np
import pytest

from cctv_reid.identity import IdentityGallery, IdentityManager, cosine_distance


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def manager(**kwargs):
    return IdentityManager(IdentityGallery(max_size=5), face_thresh=0.3, body_thresh=0.3, **kwargs)


def test_cosine_distance_is_zero_for_the_same_direction_and_two_for_opposite():
    assert cosine_distance(unit(1, 0), unit(3, 0)) == pytest.approx(0.0, abs=1e-6)
    assert cosine_distance(unit(1, 0), unit(-1, 0)) == pytest.approx(2.0)


def test_a_new_person_gets_a_new_global_id():
    m = manager()

    assert m.assign_gid(1, None, unit(1, 0, 0))[:2] == (1, "new")
    assert m.assign_gid(2, None, unit(0, 1, 0))[:2] == (2, "new")


def test_a_returning_person_is_matched_by_appearance_under_a_new_track_id():
    m = manager()
    m.assign_gid(1, None, unit(1, 0, 0))
    m.update_embeddings(1, None, unit(1, 0, 0))

    gid, reason, dist = m.assign_gid(7, None, unit(0.98, 0.05, 0))

    assert (gid, reason) == (1, "body_match")
    assert dist < 0.3


def test_a_face_match_is_preferred_over_the_body():
    m = manager()
    m.assign_gid(1, unit(1, 0), unit(1, 0, 0))
    m.update_embeddings(1, unit(1, 0), unit(1, 0, 0))

    gid, reason, _ = m.assign_gid(9, unit(1, 0.01), unit(0, 0, 1))

    assert (gid, reason) == (1, "face_match")


def test_an_ambiguous_body_match_creates_a_new_identity_instead_of_guessing():
    m = manager(body_margin=0.1)
    for tid, emb in ((1, unit(1, 1, 0)), (2, unit(1, -1, 0))):  # two distinct people
        assert m.assign_gid(tid, None, emb)[1] == "new"
        m.update_embeddings(tid, None, emb)

    # Within the threshold of both (distance 0.29) but equally close: the margin test refuses
    # to pick one.
    gid, reason, _ = m.assign_gid(3, None, unit(1, 0, 0))

    assert (gid, reason) == (3, "new")


def test_a_track_keeps_its_identity_once_assigned():
    m = manager()
    first = m.assign_gid(1, None, unit(1, 0, 0))[0]

    assert m.assign_gid(1, None, unit(0, 1, 0))[:2] == (first, "existing")


def test_each_identity_bank_keeps_only_the_most_recent_embeddings():
    gallery = IdentityGallery(max_size=3)
    for i in range(10):
        gallery.add_body(1, unit(1, i, 0))

    assert len(gallery._bodies[1]) == 3
