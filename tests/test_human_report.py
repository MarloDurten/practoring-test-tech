"""Тесты человекочитаемого отчёта и списка видео."""

from __future__ import annotations

from pathlib import Path

from app.human_report import estimate_source_duration_sec, format_duration, format_human_report
from app.media import list_videos, preset_by_id


def test_format_duration() -> None:
    assert format_duration(12.3).endswith("с")
    assert "мин" in format_duration(67.8)
    assert format_duration(None) == "н/д"


def test_estimate_duration_from_hint() -> None:
    summary = {
        "source_info": {"reported_fps": 60.0, "frame_count_hint": 4065},
        "frames_read": 4064,
    }
    dur = estimate_source_duration_sec(summary)
    assert 67.0 < dur < 68.5


def test_human_report_contains_key_blocks() -> None:
    text = format_human_report(
        {
            "source": r"C:\tmp\exam.mkv",
            "source_info": {"reported_fps": 30.0, "frame_count_hint": 300},
            "frames_read": 300,
            "wall_time_sec": 12.5,
            "profile": "balanced",
            "source_resolution": [1920, 1080],
            "working_resolution": [640, 480],
            "cpu_process_percent": {"avg": 80.0, "max": 120.0},
            "ram_mb": {"avg": 200.0, "peak": 220.0},
            "cpu_logical_count": 8,
            "latency_ms": {"mean": 20, "median": 19, "p95": 30},
            "avg_fps_processed": 10.0,
            "avg_fps_read": 30.0,
            "events": {
                "face_detected_frames": 100,
                "face_detected_ratio": 1.0,
                "distraction_frames": 2,
                "distraction_ratio": 0.02,
                "ai_suspicious_frames": 0,
                "ai_suspicious_ratio": 0.0,
            },
            "compression": {"label": "MP4 / MPEG-4"},
            "report_dir": "reports/exam/1",
        }
    )
    assert "ИТОГИ ПРОВЕРКИ" in text
    assert "Длительность видео" in text
    assert "Проверка заняла" in text
    assert "CPU процесса" in text
    assert "RAM процесса" in text
    assert "exam.mkv" in text


def test_list_videos(tmp_path: Path) -> None:
    (tmp_path / "a.mkv").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("no")
    (tmp_path / "c.mp4").write_bytes(b"x")
    names = [p.name for p in list_videos(tmp_path)]
    assert names == ["a.mkv", "c.mp4"]


def test_preset_by_id() -> None:
    p = preset_by_id("mp4v")
    assert p.extension == ".mp4"
    assert p.fourcc == "mp4v"
