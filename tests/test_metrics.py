"""Тесты агрегации latency / percentile / summary."""

from __future__ import annotations

from app.metrics import FrameRecord, MetricsCollector, percentile


def _rec(ms: float, skipped: bool = False, ram: float = 100.0) -> FrameRecord:
    return FrameRecord(
        source_frame_index=0,
        processed_index=0,
        skipped=skipped,
        timestamp_sec=0.0,
        source_width=1920,
        source_height=1080,
        working_width=640,
        working_height=480,
        process_ms=ms,
        capture_ms=0.1,
        normalize_ms=0.2,
        preprocess_ms=0.3,
        infer_ms=ms * 0.8,
        postprocess_ms=0.1,
        cpu_process_percent=20.0,
        cpu_system_percent=30.0,
        ram_mb=ram,
        gpu_util_percent=None,
        vram_mb=None,
        face_detected=True,
        face_confidence=0.9,
        head_yaw=0.0,
        head_pitch=0.0,
        head_roll=0.0,
        gaze_yaw=0.0,
        gaze_pitch=0.0,
        ear=0.3,
        blink=False,
        distraction=False,
        distraction_reasons="",
        ai_suspicion_score=0.1,
        ai_suspicious=False,
        ai_signals="",
    )


def test_percentile_edges() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p50 = percentile(vals, 50)
    assert 5.0 <= p50 <= 6.0
    assert percentile(vals, 100) == 10.0


def test_summary_ignores_skipped_for_latency() -> None:
    coll = MetricsCollector()
    coll.start()
    coll.add(_rec(10.0, skipped=True))
    coll.add(_rec(20.0))
    coll.add(_rec(30.0))
    coll.add(_rec(40.0))
    coll.stop()
    summary = coll.summarize()
    assert summary["frames_read"] == 4
    assert summary["frames_processed"] == 3
    assert summary["frames_skipped"] == 1
    assert summary["latency_ms"]["mean"] == 30.0
    assert summary["source_resolution"] == [1920, 1080]
    assert summary["working_resolution"] == [640, 480]
    assert summary["gpu"] is None
    coll.close()
