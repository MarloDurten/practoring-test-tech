"""Предобработка кадра перед CV-инференсом.

Нормализация размера выполняется отдельно (app.normalization). Здесь —
цветовое пространство, контраст и подготовка ROI.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessedFrame:
    """Кадр, готовый к инференсу MediaPipe / эвристикам."""

    bgr: np.ndarray
    rgb: np.ndarray
    gray: np.ndarray


class FramePreprocessor:
    """Дешёвая CPU-предобработка без копирования лишних каналов, где возможно."""

    def __init__(self, clahe: bool = False) -> None:
        self.clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
        )

    def process(self, bgr: np.ndarray) -> PreprocessedFrame:
        if bgr is None or bgr.size == 0:
            raise ValueError("Пустой кадр на предобработке")

        work = bgr
        if work.ndim == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        elif work.shape[2] == 4:
            work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)

        if self.clahe is not None:
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            work = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        return PreprocessedFrame(bgr=work, rgb=rgb, gray=gray)
