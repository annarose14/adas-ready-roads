"""OpenCV 5 photo checks: quality gate, resizing and own bonnet detection."""
import numpy as np

from adas_pipeline import WIDTH, _find_ego_top, quality_check, resize


def textured(h=720, w=1280, mean=120):
    rng = np.random.default_rng(0)
    return rng.integers(mean - 60, mean + 60, (h, w, 3)).astype(np.uint8)


def test_dark_photo_is_rejected():
    assert quality_check(np.full((720, 1280, 3), 20, np.uint8)) == "too_dark"


def test_panorama_is_rejected():
    assert quality_check(textured(500, 1280)) == "panorama"


def test_flat_photo_is_rejected_as_blurry():
    assert quality_check(np.full((720, 1280, 3), 120, np.uint8)) == "too_blurry"


def test_normal_photo_passes():
    assert quality_check(textured()) == "ok"


def test_resize_keeps_aspect_ratio():
    out = resize(np.zeros((1000, 2000, 3), np.uint8))
    assert out.shape[:2] == (640, WIDTH)


def test_bonnet_rows_are_found():
    vehicle = np.zeros((720, 1280), np.uint8)
    vehicle[600:, :] = 255
    assert _find_ego_top(vehicle, 720, 1280) == 600


def test_no_bonnet_keeps_whole_photo():
    assert _find_ego_top(np.zeros((720, 1280), np.uint8), 720, 1280) == 720
