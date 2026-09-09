"""Сбор и агрегация метрик нагрузки: latency, FPS, CPU, RAM, опционально GPU."""

from __future__ import annotations

import csv
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)

try:
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except Exception:  # noqa: BLE001
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False


def percentile(values: list[float], p: float) -> float:
    """Перцентиль без numpy (p в диапазоне 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if p <= 0:
        return float(ordered[0])
    if p >= 100:
        return float(ordered[-1])
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


class GpuMonitor:
    """Опциональный монитор NVIDIA через NVML. Если GPU нет — молча отключается."""

    def __init__(self) -> None:
        self.enabled = False
        self.handle = None
        if not _NVML_AVAILABLE:
            logger.info("GPU-мониторинг недоступен: пакет nvidia-ml-py не установлен")
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                logger.info("NVML инициализирован, но GPU не найдено")
                return
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
            name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            logger.info("GPU-мониторинг включён: %s", name)
        except Exception as exc:  # noqa: BLE001
            logger.info("GPU-мониторинг отключён: %s", exc)
            self.enabled = False

    def sample(self) -> tuple[Optional[float], Optional[float]]:
        """Вернуть (util_percent, used_vram_mb) либо (None, None)."""
        if not self.enabled or self.handle is None:
            return None, None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return float(util.gpu), float(mem.used) / (1024.0 * 1024.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Не удалось снять GPU-метрики: %s", exc)
            return None, None

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        self.enabled = False


@dataclass
class FrameRecord:
    """Одна строка CSV: метрики и результаты анализа конкретного кадра."""

    source_frame_index: int
    processed_index: int
    skipped: bool
    timestamp_sec: float
    source_width: int
    source_height: int
    working_width: int
    working_height: int
    process_ms: float
    capture_ms: float
    normalize_ms: float
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float
    cpu_process_percent: float
    cpu_system_percent: float
    ram_mb: float
    gpu_util_percent: Optional[float]
    vram_mb: Optional[float]
    face_detected: bool
    face_confidence: float
    head_yaw: Optional[float]
    head_pitch: Optional[float]
    head_roll: Optional[float]
    gaze_yaw: Optional[float]
    gaze_pitch: Optional[float]
    ear: Optional[float]
    blink: bool
    distraction: bool
    distraction_reasons: str
    ai_suspicion_score: float
    ai_suspicious: bool
    ai_signals: str

    def to_row(self) -> dict[str, Any]:
        return {
            "source_frame_index": self.source_frame_index,
            "processed_index": self.processed_index,
            "skipped": int(self.skipped),
            "timestamp_sec": f"{self.timestamp_sec:.4f}",
            "source_width": self.source_width,
            "source_height": self.source_height,
            "working_width": self.working_width,
            "working_height": self.working_height,
            "process_ms": f"{self.process_ms:.3f}",
            "capture_ms": f"{self.capture_ms:.3f}",
            "normalize_ms": f"{self.normalize_ms:.3f}",
            "preprocess_ms": f"{self.preprocess_ms:.3f}",
            "infer_ms": f"{self.infer_ms:.3f}",
            "postprocess_ms": f"{self.postprocess_ms:.3f}",
            "cpu_process_percent": f"{self.cpu_process_percent:.2f}",
            "cpu_system_percent": f"{self.cpu_system_percent:.2f}",
            "ram_mb": f"{self.ram_mb:.2f}",
            "gpu_util_percent": "" if self.gpu_util_percent is None else f"{self.gpu_util_percent:.2f}",
            "vram_mb": "" if self.vram_mb is None else f"{self.vram_mb:.2f}",
            "face_detected": int(self.face_detected),
            "face_confidence": f"{self.face_confidence:.4f}",
            "head_yaw": "" if self.head_yaw is None else f"{self.head_yaw:.3f}",
            "head_pitch": "" if self.head_pitch is None else f"{self.head_pitch:.3f}",
            "head_roll": "" if self.head_roll is None else f"{self.head_roll:.3f}",
            "gaze_yaw": "" if self.gaze_yaw is None else f"{self.gaze_yaw:.3f}",
            "gaze_pitch": "" if self.gaze_pitch is None else f"{self.gaze_pitch:.3f}",
            "ear": "" if self.ear is None else f"{self.ear:.4f}",
            "blink": int(self.blink),
            "distraction": int(self.distraction),
            "distraction_reasons": self.distraction_reasons,
            "ai_suspicion_score": f"{self.ai_suspicion_score:.4f}",
            "ai_suspicious": int(self.ai_suspicious),
            "ai_signals": self.ai_signals,
        }


CSV_FIELDS = list(FrameRecord(0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, False, 0, None, None, None, None, None, None, False, False, "", 0, False, "").to_row().keys())


class MetricsCollector:
    """Сбор per-frame записей + сэмплы ресурсов + итоговая агрегация."""

    def __init__(self) -> None:
        self.records: list[FrameRecord] = []
        self.process = psutil.Process()
        # Первый вызов cpu_percent возвращает 0 — «прогреваем» счётчик.
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self.gpu = GpuMonitor()
        self.wall_start = 0.0
        self.wall_end = 0.0
        self.warmup_skipped = 0

    def start(self) -> None:
        self.wall_start = time.perf_counter()

    def stop(self) -> None:
        self.wall_end = time.perf_counter()

    def sample_resources(self) -> dict[str, Any]:
        cpu_proc = float(self.process.cpu_percent(interval=None))
        cpu_sys = float(psutil.cpu_percent(interval=None))
        ram_mb = float(self.process.memory_info().rss) / (1024.0 * 1024.0)
        gpu_util, vram_mb = self.gpu.sample()
        return {
            "cpu_process_percent": cpu_proc,
            "cpu_system_percent": cpu_sys,
            "ram_mb": ram_mb,
            "gpu_util_percent": gpu_util,
            "vram_mb": vram_mb,
        }

    def add(self, record: FrameRecord) -> None:
        self.records.append(record)

    def processed_latencies_ms(self, include_warmup: bool = False) -> list[float]:
        rows = self.records
        if not include_warmup:
            rows = [r for r in rows if not r.skipped]
        else:
            rows = [r for r in rows if not r.skipped]
        return [r.process_ms for r in rows]

    def summarize(self, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        processed = [r for r in self.records if not r.skipped]
        skipped = [r for r in self.records if r.skipped]
        lat = [r.process_ms for r in processed]
        infer = [r.infer_ms for r in processed]
        wall = max(0.0, self.wall_end - self.wall_start) if self.wall_end else 0.0
        avg_fps = (len(processed) / wall) if wall > 0 else 0.0
        read_fps = (len(self.records) / wall) if wall > 0 else 0.0

        cpu_proc = [r.cpu_process_percent for r in processed]
        cpu_sys = [r.cpu_system_percent for r in processed]
        ram = [r.ram_mb for r in processed]
        gpu_u = [r.gpu_util_percent for r in processed if r.gpu_util_percent is not None]
        vram = [r.vram_mb for r in processed if r.vram_mb is not None]

        distraction_n = sum(1 for r in processed if r.distraction)
        ai_n = sum(1 for r in processed if r.ai_suspicious)
        face_n = sum(1 for r in processed if r.face_detected)

        source_res = None
        working_res = None
        if self.records:
            source_res = [self.records[0].source_width, self.records[0].source_height]
            working_res = [self.records[0].working_width, self.records[0].working_height]

        summary: dict[str, Any] = {
            "frames_read": len(self.records),
            "frames_processed": len(processed),
            "frames_skipped": len(skipped),
            "source_resolution": source_res,
            "working_resolution": working_res,
            "wall_time_sec": round(wall, 4),
            "latency_ms": {
                "mean": round(statistics.fmean(lat), 4) if lat else 0.0,
                "median": round(statistics.median(lat), 4) if lat else 0.0,
                "p95": round(percentile(lat, 95), 4) if lat else 0.0,
                "min": round(min(lat), 4) if lat else 0.0,
                "max": round(max(lat), 4) if lat else 0.0,
            },
            "infer_ms": {
                "mean": round(statistics.fmean(infer), 4) if infer else 0.0,
                "median": round(statistics.median(infer), 4) if infer else 0.0,
                "p95": round(percentile(infer, 95), 4) if infer else 0.0,
            },
            "avg_fps_processed": round(avg_fps, 4),
            "avg_fps_read": round(read_fps, 4),
            "cpu_process_percent": {
                "avg": round(statistics.fmean(cpu_proc), 3) if cpu_proc else 0.0,
                "max": round(max(cpu_proc), 3) if cpu_proc else 0.0,
            },
            "cpu_system_percent": {
                "avg": round(statistics.fmean(cpu_sys), 3) if cpu_sys else 0.0,
                "max": round(max(cpu_sys), 3) if cpu_sys else 0.0,
            },
            "ram_mb": {
                "avg": round(statistics.fmean(ram), 3) if ram else 0.0,
                "peak": round(max(ram), 3) if ram else 0.0,
            },
            "cpu_logical_count": int(psutil.cpu_count() or 0),
            "source_duration_sec": 0.0,
            "gpu": None,
            "events": {
                "face_detected_frames": face_n,
                "face_detected_ratio": round(face_n / len(processed), 4) if processed else 0.0,
                "distraction_frames": distraction_n,
                "distraction_ratio": round(distraction_n / len(processed), 4) if processed else 0.0,
                "ai_suspicious_frames": ai_n,
                "ai_suspicious_ratio": round(ai_n / len(processed), 4) if processed else 0.0,
            },
        }
        if gpu_u or vram:
            summary["gpu"] = {
                "util_percent_avg": round(statistics.fmean(gpu_u), 3) if gpu_u else None,
                "util_percent_max": round(max(gpu_u), 3) if gpu_u else None,
                "vram_mb_avg": round(statistics.fmean(vram), 3) if vram else None,
                "vram_mb_peak": round(max(vram), 3) if vram else None,
            }
        if extra:
            summary.update(extra)
        return summary

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec.to_row())
        logger.info("CSV метрик: %s (%d строк)", path, len(self.records))

    def write_json(self, path: Path, summary: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        logger.info("JSON итог: %s", path)

    def close(self) -> None:
        self.gpu.close()


def format_console_summary(summary: dict[str, Any]) -> str:
    """Человекочитаемый summary в консоль."""
    lat = summary.get("latency_ms", {})
    cpu = summary.get("cpu_process_percent", {})
    ram = summary.get("ram_mb", {})
    events = summary.get("events", {})
    gpu = summary.get("gpu")
    src = summary.get("source_resolution")
    wrk = summary.get("working_resolution")
    lines = [
        "=" * 64,
        "BENCHMARK SUMMARY",
        "=" * 64,
        f"Профиль:              {summary.get('profile')}",
        f"Источник:             {summary.get('source')}",
        f"Source resolution:    {src}",
        f"Working resolution:   {wrk}",
        f"Resize mode:          {summary.get('resize_mode')}",
        f"Кадров прочитано:     {summary.get('frames_read')}",
        f"Кадров обработано:    {summary.get('frames_processed')}",
        f"Кадров пропущено:     {summary.get('frames_skipped')}",
        f"Суммарное время, с:   {summary.get('wall_time_sec')}",
        f"Latency mean/med/p95: {lat.get('mean')} / {lat.get('median')} / {lat.get('p95')} мс",
        f"Средний FPS (proc):   {summary.get('avg_fps_processed')}",
        f"Средний FPS (read):   {summary.get('avg_fps_read')}",
        f"CPU process avg/max: {cpu.get('avg')} / {cpu.get('max')} %",
        f"RAM avg/peak, МБ:     {ram.get('avg')} / {ram.get('peak')}",
    ]
    if gpu:
        lines.append(
            f"GPU util avg/max:     {gpu.get('util_percent_avg')} / {gpu.get('util_percent_max')} %"
        )
        lines.append(
            f"VRAM avg/peak, МБ:    {gpu.get('vram_mb_avg')} / {gpu.get('vram_mb_peak')}"
        )
    else:
        lines.append("GPU:                  недоступен / не измерялся (CPU-only)")
    lines.extend(
        [
            f"Лицо найдено:         {events.get('face_detected_frames')} "
            f"({events.get('face_detected_ratio')})",
            f"Отвлечение:           {events.get('distraction_frames')} "
            f"({events.get('distraction_ratio')})",
            f"AI-подозрительные:     {events.get('ai_suspicious_frames')} "
            f"({events.get('ai_suspicious_ratio')})",
            "=" * 64,
        ]
    )
    return "\n".join(lines)
