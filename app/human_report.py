"""Понятный текстовый отчёт для PowerShell: длительность, время проверки, ресурсы."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psutil

from app.config import QUALITY_PROFILES


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "н/д"
    total = float(seconds)
    if total < 60:
        return f"{total:.1f} с"
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours} ч {minutes} мин {sec:.0f} с"
    return f"{int(minutes)} мин {sec:.1f} с"


def estimate_source_duration_sec(summary: dict[str, Any]) -> float:
    info = summary.get("source_info") or {}
    fps = float(info.get("reported_fps") or 0.0)
    hinted = int(info.get("frame_count_hint") or 0)
    frames = int(summary.get("frames_read") or 0)
    if fps > 1.0 and frames > 0:
        analyzed = frames / fps
        if hinted > 0 and frames < hinted * 0.95:
            return analyzed
        if hinted > 0:
            return hinted / fps
        return analyzed
    explicit = float(summary.get("source_duration_sec") or 0.0)
    return explicit


def format_human_report(summary: dict[str, Any]) -> str:
    source = str(summary.get("source") or "")
    name = Path(source).name if source and not str(source).isdigit() else source
    duration = estimate_source_duration_sec(summary)
    check_sec = float(summary.get("wall_time_sec") or 0.0)
    profile = str(summary.get("profile") or "")
    spec = QUALITY_PROFILES.get(profile, {})
    src_res = summary.get("source_resolution") or []
    wrk_res = summary.get("working_resolution") or []
    info = summary.get("source_info") or {}
    fps = float(info.get("reported_fps") or 0.0)
    cpu = summary.get("cpu_process_percent") or {}
    ram = summary.get("ram_mb") or {}
    lat = summary.get("latency_ms") or {}
    events = summary.get("events") or {}
    gpu = summary.get("gpu")
    cores = int(summary.get("cpu_logical_count") or psutil.cpu_count() or 0)
    cpu_avg = float(cpu.get("avg") or 0.0)
    cores_used = cpu_avg / 100.0 if cpu_avg else 0.0
    codec = summary.get("compression") or {}

    src_txt = "×".join(str(x) for x in src_res) if src_res else "н/д"
    wrk_txt = "×".join(str(x) for x in wrk_res) if wrk_res else "н/д"
    fps_txt = f"{fps:.0f} fps" if fps > 1 else "fps н/д"

    lines = [
        "",
        "=" * 66,
        "  ИТОГИ ПРОВЕРКИ",
        "=" * 66,
        f"  Видео:                 {name}",
        f"  Длительность видео:    {format_duration(duration)}"
        f"  ({summary.get('frames_read', 0)} кадров, {src_txt}, {fps_txt})",
        f"  Проверка заняла:       {format_duration(check_sec)}",
        "",
        f"  Профиль качества:      {profile}"
        f"  ({spec.get('width')}×{spec.get('height')}, каждый {spec.get('frame_stride')}-й кадр)",
        f"  Рабочее разрешение:    {wrk_txt}",
        f"  Сжатие marked-видео:   {codec.get('label', 'не сохранялось')}",
        "",
        "  РЕСУРСЫ",
        f"  CPU процесса:          среднее {cpu.get('avg', 0):.1f}%   пик {cpu.get('max', 0):.1f}%",
        f"  Задействовано ядер:    ≈ {cores_used:.2f} из {cores} логических"
        "  (100% ≈ одно ядро)",
        f"  RAM процесса:          среднее {ram.get('avg', 0):.0f} МБ   пик {ram.get('peak', 0):.0f} МБ",
    ]
    if gpu:
        lines.append(
            f"  GPU:                   загрузка {gpu.get('util_percent_avg')}%"
            f" (пик {gpu.get('util_percent_max')}%), "
            f"VRAM {gpu.get('vram_mb_avg')} МБ (пик {gpu.get('vram_mb_peak')} МБ)"
        )
    else:
        lines.append("  GPU:                   не использовался (режим CPU-only)")

    face_ratio = float(events.get("face_detected_ratio") or 0.0) * 100
    dist_ratio = float(events.get("distraction_ratio") or 0.0) * 100
    ai_ratio = float(events.get("ai_suspicious_ratio") or 0.0) * 100
    lines.extend(
        [
            "",
            "  СКОРОСТЬ",
            f"  Latency кадра:         mean {lat.get('mean')} мс · "
            f"median {lat.get('median')} мс · p95 {lat.get('p95')} мс",
            f"  Скорость анализа:      {summary.get('avg_fps_processed')} обработанных кадр/с",
            f"  Скорость чтения:       {summary.get('avg_fps_read')} кадр/с",
            "",
            "  СОБЫТИЯ",
            f"  Лицо на кадре:         {events.get('face_detected_frames')} "
            f"({face_ratio:.1f}%)",
            f"  Отвлечение:            {events.get('distraction_frames')} "
            f"({dist_ratio:.1f}%)",
            f"  AI-подозрительные:     {events.get('ai_suspicious_frames')} "
            f"({ai_ratio:.1f}%)",
            "",
            "  ФАЙЛЫ",
            f"  Папка отчёта:          {summary.get('report_dir', '—')}",
            f"  CSV:                   {summary.get('csv_path', '—')}",
            f"  JSON:                  {summary.get('json_path', '—')}",
            f"  Marked:                {summary.get('marked_video_path', 'не создан')}",
            "=" * 66,
            "",
        ]
    )
    return "\n".join(lines)
