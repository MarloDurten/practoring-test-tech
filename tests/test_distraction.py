"""Тесты эвристик отвлечения на синтетических FaceResult."""

from __future__ import annotations

from app.analyzers.distraction import DistractionHeuristics
from app.analyzers.face_tracking import FaceResult
from app.analyzers.gaze_estimation import GazeResult
from app.config import DistractionThresholds


def test_disabled_module_returns_empty() -> None:
    h = DistractionHeuristics(enabled=False)
    r = h.infer(FaceResult(), GazeResult(), 1.0)
    assert r.distracted is False
    assert r.reasons == []


def test_missing_face_streak() -> None:
    t = DistractionThresholds(missing_face_frames=3)
    h = DistractionHeuristics(thresholds=t)
    last = None
    for i in range(3):
        last = h.infer(FaceResult(detected=False), GazeResult(), i * 0.1)
    assert last is not None
    assert last.distracted
    assert any(x.startswith("face_missing") for x in last.reasons)


def test_head_turn() -> None:
    h = DistractionHeuristics()
    face = FaceResult(detected=True, confidence=0.9, head_yaw=55.0, head_pitch=0.0)
    r = h.infer(face, GazeResult(available=True, yaw_deg=0.0, pitch_deg=0.0), 1.0)
    assert r.distracted
    assert any("head_yaw" in x for x in r.reasons)


def test_gaze_away() -> None:
    h = DistractionHeuristics()
    face = FaceResult(detected=True, confidence=0.9, head_yaw=0.0, head_pitch=0.0)
    gaze = GazeResult(available=True, yaw_deg=40.0, pitch_deg=0.0)
    r = h.infer(face, gaze, 1.0)
    assert r.distracted
    assert any("gaze_yaw" in x for x in r.reasons)


def test_blink_and_eyes_closed() -> None:
    t = DistractionThresholds(ear_closed=0.21, eyes_closed_frames=3)
    h = DistractionHeuristics(thresholds=t)
    face_open = FaceResult(detected=True, confidence=0.9, mean_ear=0.30)
    h.infer(face_open, GazeResult(), 0.0)
    last = None
    for i in range(3):
        closed = FaceResult(detected=True, confidence=0.9, mean_ear=0.10)
        last = h.infer(closed, GazeResult(), 0.1 * (i + 1))
    assert last is not None
    assert last.blink or last.eyes_closed_streak >= 3
    assert last.distracted
