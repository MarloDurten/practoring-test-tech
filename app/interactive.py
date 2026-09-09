"""Интерактивный запуск: выбор видео и формата сжатия стрелками в терминале."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import QUALITY_PROFILES, AppConfig
from app.media import COMPRESSION_PRESETS, VIDEO_EXTENSIONS, list_videos
from app.terminal_menu import select_option


def _label_video(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{path.name}   ({size_mb:.1f} МБ, {path.suffix.lower()})"


def _label_profile(name: str) -> str:
    spec = QUALITY_PROFILES[name]
    return (
        f"{name}  —  {spec['width']}×{spec['height']}, "
        f"каждый {spec['frame_stride']}-й кадр  ({spec['description']})"
    )


def _label_codec(preset) -> str:
    return f"{preset.label}  ({preset.fourcc}{preset.extension})  — {preset.note}"


def collect_interactive_config(scan_dir: Path | None = None) -> AppConfig:
    folder = (scan_dir or Path.cwd()).resolve()
    print()
    print("=" * 66)
    print("  ПРОВЕРКА ВИДЕО")
    print("=" * 66)
    print(f"  Папка: {folder}")
    print(f"  Форматы: {', '.join(VIDEO_EXTENSIONS)}")
    print()

    videos = list_videos(folder)
    if not videos:
        raise FileNotFoundError(
            f"В папке нет подходящих видео:\n  {folder}\n"
            f"Положите файл {', '.join(VIDEO_EXTENSIONS)} и запустите снова."
        )

    video_idx = select_option("Выберите видео", [_label_video(p) for p in videos])
    video = videos[video_idx]
    print(f"  Выбрано видео: {video.name}\n")

    profile_names = list(QUALITY_PROFILES)
    ordered = ["balanced"] + [p for p in profile_names if p != "balanced"]
    profile_idx = select_option("Выберите профиль качества / нагрузки", [_label_profile(p) for p in ordered])
    profile = ordered[profile_idx]
    print(f"  Профиль: {profile}\n")

    codec_idx = select_option(
        "Выберите формат сжатия marked-видео",
        [_label_codec(p) for p in COMPRESSION_PRESETS],
    )
    preset = COMPRESSION_PRESETS[codec_idx]
    print(f"  Сжатие: {preset.label} ({preset.fourcc})\n")
    print("  Идёт проверка. После завершения здесь же появится отчёт...")
    print()

    return AppConfig(
        profile=profile,  # type: ignore[arg-type]
        source=str(video),
        output_dir=str(Path(__file__).resolve().parent.parent / "reports"),
        preview=False,
        save_marked=True,
        compression_id=preset.id,
    )


def run_interactive(scan_dir: Path | None = None) -> int:
    from app.main import run_session, setup_logging

    setup_logging("WARNING")
    logging.getLogger("app.main").setLevel(logging.WARNING)
    try:
        cfg = collect_interactive_config(scan_dir)
    except KeyboardInterrupt:
        print("\nОтменено.")
        return 0
    except FileNotFoundError as exc:
        print(exc)
        return 2
    return run_session(cfg, quiet_logs=True)
