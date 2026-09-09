"""Нормализация кадров к внутреннему стандарту системы.

Камера и видеофайлы отдают произвольные разрешения. Мы никогда не считаем,
что cap.set(WIDTH/HEIGHT) сработал: фактический размер берётся из самого кадра.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from app.config import MAX_INPUT_PIXELS, QUALITY_PROFILES, ResizeMode

logger = logging.getLogger(__name__)


@dataclass
class NormalizationInfo:
    """Метаданные одного акта нормализации — логируем source и working отдельно."""

    source_width: int
    source_height: int
    working_width: int
    working_height: int
    scale: float
    pad_x: int
    pad_y: int
    crop_x: int
    crop_y: int
    mode: ResizeMode
    prescaled: bool
    interpolation: str

    def as_dict(self) -> dict:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "working_width": self.working_width,
            "working_height": self.working_height,
            "scale": self.scale,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "crop_x": self.crop_x,
            "crop_y": self.crop_y,
            "mode": self.mode,
            "prescaled": self.prescaled,
            "interpolation": self.interpolation,
        }


class FrameNormalizer:
    """Приводит произвольный кадр к фиксированному профилю с сохранением AR."""

    def __init__(
        self,
        target_width: int,
        target_height: int,
        mode: ResizeMode = "letterbox",
        max_input_pixels: int = MAX_INPUT_PIXELS,
        pad_value: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target_width и target_height должны быть > 0")
        if mode not in ("letterbox", "center_crop"):
            raise ValueError(f"Неизвестный resize mode: {mode}")
        self.target_width = target_width
        self.target_height = target_height
        self.mode: ResizeMode = mode
        self.max_input_pixels = max_input_pixels
        self.pad_value = pad_value
        self._logged_once = False

    @classmethod
    def from_profile(
        cls,
        profile: str,
        mode: ResizeMode = "letterbox",
        max_input_pixels: int = MAX_INPUT_PIXELS,
    ) -> "FrameNormalizer":
        if profile not in QUALITY_PROFILES:
            raise ValueError(
                f"Неизвестный профиль '{profile}'. Доступны: {list(QUALITY_PROFILES)}"
            )
        spec = QUALITY_PROFILES[profile]
        return cls(int(spec["width"]), int(spec["height"]), mode=mode, max_input_pixels=max_input_pixels)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, NormalizationInfo]:
        """Вернуть кадр строго размера (target_h, target_w, C) и метаданные."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Пустой кадр: нечего нормализовать")
        if frame.ndim not in (2, 3):
            raise ValueError(f"Ожидался 2D/3D кадр, получено ndim={frame.ndim}")

        source_h, source_w = frame.shape[:2]
        work = frame
        prescaled = False

        # Слишком большой вход (4K и выше): дешёвый downscale до ~1080p,
        # чтобы INTER_AREA на финальном шаге не работал с гигантской матрицей.
        pixels = source_w * source_h
        if pixels > self.max_input_pixels:
            pre_scale = (self.max_input_pixels / float(pixels)) ** 0.5
            pre_w = max(1, int(source_w * pre_scale))
            pre_h = max(1, int(source_h * pre_scale))
            work = cv2.resize(work, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
            prescaled = True
            logger.debug(
                "Pre-downscale %dx%d -> %dx%d (pixels %d > %d)",
                source_w,
                source_h,
                pre_w,
                pre_h,
                pixels,
                self.max_input_pixels,
            )

        if self.mode == "letterbox":
            out, info = self._letterbox(work, source_w, source_h, prescaled)
        else:
            out, info = self._center_crop(work, source_w, source_h, prescaled)

        if not self._logged_once:
            logger.info(
                "Нормализация: source=%dx%d  working=%dx%d  mode=%s  prescaled=%s",
                info.source_width,
                info.source_height,
                info.working_width,
                info.working_height,
                info.mode,
                info.prescaled,
            )
            self._logged_once = True
        return out, info

    def _choose_interpolation(self, scale: float) -> Tuple[int, str]:
        if scale < 1.0:
            return cv2.INTER_AREA, "INTER_AREA"
        return cv2.INTER_LINEAR, "INTER_LINEAR"

    def _letterbox(
        self,
        frame: np.ndarray,
        source_w: int,
        source_h: int,
        prescaled: bool,
    ) -> Tuple[np.ndarray, NormalizationInfo]:
        h, w = frame.shape[:2]
        scale = min(self.target_width / w, self.target_height / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp, interp_name = self._choose_interpolation(scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        channels = 1 if resized.ndim == 2 else resized.shape[2]
        canvas_shape: Tuple[int, ...]
        if channels == 1 and resized.ndim == 2:
            canvas = np.full(
                (self.target_height, self.target_width),
                self.pad_value[0],
                dtype=resized.dtype,
            )
        else:
            canvas = np.zeros(
                (self.target_height, self.target_width, channels),
                dtype=resized.dtype,
            )
            canvas[:] = np.array(self.pad_value[:channels], dtype=resized.dtype)

        pad_x = (self.target_width - new_w) // 2
        pad_y = (self.target_height - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        info = NormalizationInfo(
            source_width=source_w,
            source_height=source_h,
            working_width=self.target_width,
            working_height=self.target_height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            crop_x=0,
            crop_y=0,
            mode="letterbox",
            prescaled=prescaled,
            interpolation=interp_name,
        )
        return canvas, info

    def _center_crop(
        self,
        frame: np.ndarray,
        source_w: int,
        source_h: int,
        prescaled: bool,
    ) -> Tuple[np.ndarray, NormalizationInfo]:
        h, w = frame.shape[:2]
        # Масштаб, при котором кадр покрывает весь target, лишнее обрезается.
        scale = max(self.target_width / w, self.target_height / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp, interp_name = self._choose_interpolation(scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        crop_x = max(0, (new_w - self.target_width) // 2)
        crop_y = max(0, (new_h - self.target_height) // 2)
        cropped = resized[
            crop_y : crop_y + self.target_height,
            crop_x : crop_x + self.target_width,
        ]

        # Защита от ошибки округления (1px).
        if cropped.shape[0] != self.target_height or cropped.shape[1] != self.target_width:
            cropped = cv2.resize(
                cropped,
                (self.target_width, self.target_height),
                interpolation=interp,
            )

        info = NormalizationInfo(
            source_width=source_w,
            source_height=source_h,
            working_width=self.target_width,
            working_height=self.target_height,
            scale=scale,
            pad_x=0,
            pad_y=0,
            crop_x=crop_x,
            crop_y=crop_y,
            mode="center_crop",
            prescaled=prescaled,
            interpolation=interp_name,
        )
        return cropped, info


def map_point_from_working_to_source(
    x: float,
    y: float,
    info: NormalizationInfo,
) -> Tuple[float, float]:
    """Обратное отображение точки из рабочего кадра в исходный (для отладки)."""
    if info.mode == "letterbox":
        src_x = (x - info.pad_x) / info.scale if info.scale else x
        src_y = (y - info.pad_y) / info.scale if info.scale else y
        return src_x, src_y
    src_x = (x + info.crop_x) / info.scale if info.scale else x
    src_y = (y + info.crop_y) / info.scale if info.scale else y
    return src_x, src_y
