"""Захват кадров с веб-камеры или из видеофайла.

Важно: OpenCV `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)` — только подсказка драйверу.
Многие веб-камеры её игнорируют или отдают ближайший режим. Фактический размер
всегда читается из полученного numpy-кадра, а не из get()/set().
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

from app.config import DEFAULT_MAX_DURATION_SEC

logger = logging.getLogger(__name__)

SourceType = Union[int, str]


@dataclass
class CapturedFrame:
    """Один кадр источника до нормализации."""

    image: np.ndarray
    source_index: int
    source_width: int
    source_height: int
    timestamp_sec: float
    wall_time_sec: float
    from_camera: bool


@dataclass
class SourceInfo:
    requested_source: str
    opened: bool
    from_camera: bool
    reported_width: int
    reported_height: int
    reported_fps: float
    actual_width: int
    actual_height: int
    frame_count_hint: int
    backend: str

    def as_dict(self) -> dict:
        return {
            "requested_source": self.requested_source,
            "opened": self.opened,
            "from_camera": self.from_camera,
            "reported_width": self.reported_width,
            "reported_height": self.reported_height,
            "reported_fps": self.reported_fps,
            "actual_width": self.actual_width,
            "actual_height": self.actual_height,
            "frame_count_hint": self.frame_count_hint,
            "backend": self.backend,
        }


def parse_source(value: str | int) -> SourceType:
    """'0' → камера 0, путь к файлу остаётся строкой."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


class VideoSource:
    """Итератор кадров с ограничением по длительности (до 10 минут)."""

    def __init__(
        self,
        source: str | int = 0,
        max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
        max_frames: Optional[int] = None,
        hint_width: Optional[int] = None,
        hint_height: Optional[int] = None,
    ) -> None:
        self.raw_source = source
        self.source = parse_source(source)
        self.max_duration_sec = float(max_duration_sec)
        self.max_frames = max_frames
        self.hint_width = hint_width
        self.hint_height = hint_height
        self.cap: Optional[cv2.VideoCapture] = None
        self._pending: Optional[np.ndarray] = None
        self.info = SourceInfo(
            requested_source=str(source),
            opened=False,
            from_camera=isinstance(self.source, int),
            reported_width=0,
            reported_height=0,
            reported_fps=0.0,
            actual_width=0,
            actual_height=0,
            frame_count_hint=0,
            backend="",
        )
        self._opened_at = 0.0

    def open(self) -> SourceInfo:
        if isinstance(self.source, str):
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Видеофайл не найден: {path.resolve()}")
            if not path.is_file():
                raise ValueError(f"Ожидался файл, получено: {path}")
            self.cap = cv2.VideoCapture(str(path))
            self.info.from_camera = False
            self.info.backend = "file"
        else:
            # На Windows DirectShow стабильнее, чем MSMF, для веб-камер.
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(self.source), backend)
            self.info.from_camera = True
            self.info.backend = "dshow" if sys.platform == "win32" else "default"

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть источник '{self.source}'. "
                "Проверьте индекс камеры или путь к файлу, а также кодеки OpenCV."
            )

        # Подсказка драйверу. Результат НЕ используем как истину.
        if self.hint_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.hint_width))
        if self.hint_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.hint_height))

        self.info.reported_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.info.reported_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.info.reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.info.frame_count_hint = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.info.opened = True

        # Пробуем один кадр, чтобы узнать РЕАЛЬНОЕ разрешение.
        ok, probe = self.cap.read()
        if not ok or probe is None:
            self.release()
            raise RuntimeError(
                "Источник открыт, но первый кадр прочитать не удалось "
                "(камера занята, битый файл или нет прав доступа)."
            )
        self.info.actual_width = int(probe.shape[1])
        self.info.actual_height = int(probe.shape[0])

        if (
            self.info.reported_width
            and self.info.actual_width
            and (
                self.info.reported_width != self.info.actual_width
                or self.info.reported_height != self.info.actual_height
            )
        ):
            logger.warning(
                "Камера/файл сообщили %dx%d, фактически кадр %dx%d. "
                "Используем фактический размер (cap.set ненадёжен).",
                self.info.reported_width,
                self.info.reported_height,
                self.info.actual_width,
                self.info.actual_height,
            )

        # Возвращаем пробный кадр в поток через буфер — иначе потеряем первый кадр.
        self._pending: Optional[np.ndarray] = probe
        self._opened_at = time.perf_counter()

        logger.info(
            "Источник открыт: %s  reported=%dx%d@%.2ffps  actual=%dx%d  frames_hint=%d",
            self.source,
            self.info.reported_width,
            self.info.reported_height,
            self.info.reported_fps,
            self.info.actual_width,
            self.info.actual_height,
            self.info.frame_count_hint,
        )
        return self.info

    def frames(self) -> Iterator[CapturedFrame]:
        if self.cap is None or not self.info.opened:
            self.open()

        idx = 0
        fps = self.info.reported_fps if self.info.reported_fps > 1e-3 else 30.0
        start_wall = time.perf_counter()

        while True:
            if self.max_frames is not None and idx >= self.max_frames:
                logger.info("Остановка: достигнут лимит max_frames=%d", self.max_frames)
                break

            wall = time.perf_counter() - start_wall
            if wall > self.max_duration_sec:
                logger.info(
                    "Остановка: достигнут лимит длительности %.1f сек",
                    self.max_duration_sec,
                )
                break

            if self._pending is not None:
                image = self._pending
                self._pending = None
                ok = True
            else:
                assert self.cap is not None
                ok, image = self.cap.read()

            if not ok or image is None:
                if self.info.from_camera:
                    logger.warning("Камера перестала отдавать кадры на index=%d", idx)
                break

            h, w = image.shape[:2]
            # Для файла таймстемп считаем по FPS; для камеры — wall-clock.
            if self.info.from_camera:
                ts = wall
            else:
                ts = idx / fps

            if (not self.info.from_camera) and ts > self.max_duration_sec:
                logger.info(
                    "Остановка: видео-время %.1f сек превысило лимит %.1f",
                    ts,
                    self.max_duration_sec,
                )
                break

            yield CapturedFrame(
                image=image,
                source_index=idx,
                source_width=w,
                source_height=h,
                timestamp_sec=ts,
                wall_time_sec=wall,
                from_camera=self.info.from_camera,
            )
            idx += 1

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:  # pragma: no cover - защитный путь
                logger.warning("Ошибка при закрытии VideoCapture: %s", exc)
            self.cap = None
            self.info.opened = False

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


def is_camera_source(source: str | int) -> bool:
    parsed = parse_source(source)
    return isinstance(parsed, int)
