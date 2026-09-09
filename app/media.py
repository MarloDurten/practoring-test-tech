"""Поддерживаемые видеоконтейнеры и пресеты сжатия marked-ролика."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".m4v", ".mpeg", ".mpg")


@dataclass(frozen=True)
class CompressionPreset:
    id: str
    label: str
    fourcc: str
    extension: str
    note: str


COMPRESSION_PRESETS: tuple[CompressionPreset, ...] = (
    CompressionPreset(
        "mp4v",
        "MP4 / MPEG-4",
        "mp4v",
        ".mp4",
        "Самый совместимый вариант на Windows (OpenCV).",
    ),
    CompressionPreset(
        "avc1",
        "MP4 / H.264",
        "avc1",
        ".mp4",
        "Сильнее сжатие. Если запись не откроется — будет откат на MPEG-4.",
    ),
    CompressionPreset(
        "xvid",
        "AVI / XVID",
        "XVID",
        ".avi",
        "Классический AVI, среднее сжатие.",
    ),
    CompressionPreset(
        "mjpg",
        "AVI / MJPEG",
        "MJPG",
        ".avi",
        "Слабое сжатие, большой файл, легко открывается.",
    ),
)


def preset_by_id(preset_id: str) -> CompressionPreset:
    for item in COMPRESSION_PRESETS:
        if item.id == preset_id:
            return item
    raise KeyError(f"Неизвестный пресет сжатия: {preset_id}")


def list_videos(folder: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(folder).resolve()
    if not root.is_dir():
        return []
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files, key=lambda p: p.name.lower())


def open_video_writer(
    path: Path,
    fps: float,
    frame_size: tuple[int, int],
    fourcc_str: str,
) -> tuple[cv2.VideoWriter, str, Path]:
    """Открыть VideoWriter. Если кодек не принят — откат на mp4v/.mp4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    attempts = [(fourcc_str, path)]
    if fourcc_str.lower() != "mp4v":
        attempts.append(("mp4v", path.with_suffix(".mp4")))

    last_error = ""
    for code, candidate in attempts:
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(str(candidate), fourcc, float(fps), frame_size)
        if writer.isOpened():
            return writer, code, candidate
        writer.release()
        last_error = code
    raise RuntimeError(
        f"Не удалось открыть запись marked-видео ({path}). Последний кодек: {last_error}"
    )
