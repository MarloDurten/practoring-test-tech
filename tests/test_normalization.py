"""Тесты нормализации: любой входной размер → фиксированный профиль."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import QUALITY_PROFILES
from app.normalization import FrameNormalizer, map_point_from_working_to_source


def _bgr(h: int, w: int, value: int = 120) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


@pytest.mark.parametrize("profile", list(QUALITY_PROFILES))
@pytest.mark.parametrize(
    "src_h,src_w",
    [
        (480, 640),
        (720, 1280),
        (1080, 1920),
        (240, 320),
        (1080, 1080),
        (720, 960),
        (2160, 3840),
        (160, 1920),
    ],
)
def test_output_matches_profile(profile: str, src_h: int, src_w: int) -> None:
    spec = QUALITY_PROFILES[profile]
    tw, th = spec["width"], spec["height"]
    norm = FrameNormalizer.from_profile(profile, mode="letterbox")
    out, info = norm.process(_bgr(src_h, src_w))
    assert out.shape[0] == th
    assert out.shape[1] == tw
    assert info.source_width == src_w
    assert info.source_height == src_h
    assert info.working_width == tw
    assert info.working_height == th
    assert info.mode == "letterbox"


def test_letterbox_preserves_content_aspect() -> None:
    frame = _bgr(200, 800, 200)
    frame[90:110, 390:410] = (0, 0, 255)
    norm = FrameNormalizer(640, 360, mode="letterbox")
    out, info = norm.process(frame)
    assert out.shape == (360, 640, 3)
    assert info.pad_y > 0 or info.pad_x >= 0
    cy = info.pad_y + int(round(100 * info.scale))
    cx = info.pad_x + int(round(400 * info.scale))
    assert 0 <= cy < 360 and 0 <= cx < 640
    assert out[cy, cx, 2] >= 150


def test_center_crop_fills_target() -> None:
    frame = _bgr(1080, 1920, 90)
    norm = FrameNormalizer(640, 480, mode="center_crop")
    out, info = norm.process(frame)
    assert out.shape == (480, 640, 3)
    assert info.mode == "center_crop"
    assert info.crop_x >= 0 and info.crop_y >= 0


def test_4k_triggers_prescale() -> None:
    frame = _bgr(2160, 3840, 40)
    norm = FrameNormalizer(1280, 720, mode="letterbox", max_input_pixels=1920 * 1080)
    out, info = norm.process(frame)
    assert out.shape == (720, 1280, 3)
    assert info.prescaled is True
    assert info.source_width == 3840
    assert info.source_height == 2160


def test_empty_frame_raises() -> None:
    norm = FrameNormalizer(640, 360)
    with pytest.raises(ValueError):
        norm.process(np.array([]))


def test_inverse_mapping_letterbox() -> None:
    info_holder = {}
    frame = _bgr(480, 640)
    norm = FrameNormalizer(640, 360, mode="letterbox")
    out, info = norm.process(frame)
    info_holder["i"] = info
    sx, sy = map_point_from_working_to_source(info.pad_x + 10, info.pad_y + 10, info)
    assert sx >= 0
    assert sy >= 0
    assert out.shape == (360, 640, 3)


def test_unknown_profile() -> None:
    with pytest.raises(ValueError):
        FrameNormalizer.from_profile("ultra")
