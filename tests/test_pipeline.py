"""Интеграционные тесты пайплайна и захвата на синтетическом видео."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.analyzers.face_tracking import FaceTracker
from app.analyzers.gaze_estimation import GazeEstimator
from app.analyzers.pipeline import AnalysisPipeline
from app.config import DEFAULT_FALLBACK_FPS, AppConfig, ModuleToggles
from app.video_capture import (
    CapturedFrame,
    VideoSource,
    effective_source_fps,
    marked_output_fps,
    parse_source,
)


def _write_video(path: Path, n: int = 12, w: int = 320, h: int = 240, fps: int = 10) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "OpenCV не смог создать тестовый видеофайл"
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = ((i * 17) % 200, 40, 80)
        cv2.circle(frame, (w // 2, h // 2), 40, (220, 220, 220), -1)
        writer.write(frame)
    writer.release()


def test_parse_source() -> None:
    assert parse_source("0") == 0
    assert parse_source("2") == 2
    assert parse_source("clip.mp4") == "clip.mp4"


def test_effective_fps_fallback_matches_marked_output() -> None:
    assert effective_source_fps(0.0) == DEFAULT_FALLBACK_FPS
    assert effective_source_fps(None) == DEFAULT_FALLBACK_FPS
    assert effective_source_fps(25.0) == 25.0
    assert marked_output_fps(0.0, 1) == DEFAULT_FALLBACK_FPS
    assert marked_output_fps(0.0, 3) == pytest.approx(DEFAULT_FALLBACK_FPS / 3)
    assert marked_output_fps(30.0, 3) == pytest.approx(10.0)


def test_video_landmarker_timestamps_are_monotonic() -> None:
    tracker = FaceTracker(enabled=False)
    assert tracker._next_video_timestamp_ms(0) == 0
    assert tracker._next_video_timestamp_ms(0) == 1
    assert tracker._next_video_timestamp_ms(40) == 40
    assert tracker._next_video_timestamp_ms(None) == 73
    tracker.close()
    assert tracker._next_video_timestamp_ms(0) == 0


def test_video_source_reads_actual_resolution(tmp_path: Path) -> None:
    path = tmp_path / "dummy.mp4"
    _write_video(path, n=8, w=320, h=240)
    with VideoSource(str(path), max_duration_sec=5, max_frames=5) as src:
        info = src.info
        assert info.opened
        assert info.actual_width == 320
        assert info.actual_height == 240
        frames = list(src.frames())
    assert 1 <= len(frames) <= 5
    assert frames[0].source_width == 320
    assert frames[0].source_height == 240


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        VideoSource("definitely_missing_file_xyz.mp4").open()


def test_pipeline_cpu_without_face(tmp_path: Path) -> None:
    path = tmp_path / "run.mp4"
    _write_video(path, n=10, w=480, h=360)
    cfg = AppConfig(
        profile="fast",
        source=str(path),
        max_duration_sec=5,
        max_frames=10,
        warmup_frames=0,
        output_dir=str(tmp_path / "out"),
        modules=ModuleToggles(
            face_tracking=False,
            gaze_estimation=False,
            distraction_heuristics=True,
            ai_generated_frame_heuristics=True,
        ),
    )
    pipeline = AnalysisPipeline(cfg)
    summary = pipeline.run()
    assert summary["frames_read"] >= 1
    assert summary["frames_processed"] >= 1
    assert summary["working_resolution"] == [640, 360]
    assert summary["source_resolution"] == [320, 240] or summary["source_resolution"] == [480, 360]
    assert summary["latency_ms"]["mean"] >= 0.0
    assert "cpu_process_percent" in summary
    assert "ram_mb" in summary


def test_handle_frame_stride() -> None:
    cfg = AppConfig(
        profile="balanced",
        source=0,
        modules=ModuleToggles(
            face_tracking=False,
            gaze_estimation=False,
            distraction_heuristics=False,
            ai_generated_frame_heuristics=False,
        ),
    )
    pipeline = AnalysisPipeline(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    captured0 = CapturedFrame(frame, 0, 640, 480, 0.0, 0.0, False)
    captured1 = CapturedFrame(frame, 1, 640, 480, 0.1, 0.1, False)
    rec0, analysis0, _ = pipeline._handle_frame(captured0)
    rec1, analysis1, _ = pipeline._handle_frame(captured1)
    pipeline.close()
    assert rec0.skipped is False
    assert rec1.skipped is True
    assert analysis0 is not None
    assert analysis1 is None


def test_gaze_disabled_and_no_face() -> None:
    ge = GazeEstimator(enabled=False)
    from app.analyzers.face_tracking import FaceResult

    r = ge.infer(FaceResult(detected=True))
    assert r.method == "disabled"
    ge2 = GazeEstimator(enabled=True)
    r2 = ge2.infer(FaceResult(detected=False))
    assert r2.method == "no_face"
