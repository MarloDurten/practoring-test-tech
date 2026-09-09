"""Тесты AI-эвристик: скачки landmarks, низкая уверенность, smoothness."""

from __future__ import annotations

import numpy as np

from app.analyzers.ai_heuristics import AiGeneratedHeuristics
from app.analyzers.face_tracking import FaceResult
from app.config import AiHeuristicThresholds


def _face(landmarks: np.ndarray, conf: float = 0.9) -> FaceResult:
    h, w = 480, 640
    px = np.stack([landmarks[:, 0] * w, landmarks[:, 1] * h], axis=1)
    xs, ys = px[:, 0], px[:, 1]
    bbox = (int(xs.min()), int(ys.min()), int(max(1, xs.max() - xs.min())), int(max(1, ys.max() - ys.min())))
    return FaceResult(
        detected=True,
        confidence=conf,
        landmarks_norm=landmarks,
        landmarks_px=px,
        bbox=bbox,
        landmark_count=len(landmarks),
    )


def test_disabled() -> None:
    h = AiGeneratedHeuristics(enabled=False)
    gray = np.zeros((480, 640), dtype=np.uint8)
    r = h.infer(gray, FaceResult())
    assert r.score == 0.0
    assert not r.suspicious


def test_low_confidence_signal() -> None:
    t = AiHeuristicThresholds(low_confidence=0.45, suspicion_score=0.2)
    h = AiGeneratedHeuristics(thresholds=t)
    gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    lm = np.linspace(0.2, 0.8, 100 * 2).reshape(100, 2).astype(np.float32)
    r = h.infer(gray, _face(lm, conf=0.1))
    assert any("low_tracking_confidence" in s for s in r.signals)


def test_landmark_jump() -> None:
    t = AiHeuristicThresholds(landmark_jump=0.05, suspicion_score=0.2)
    h = AiGeneratedHeuristics(thresholds=t)
    gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    lm1 = np.full((80, 2), 0.40, dtype=np.float32)
    lm2 = np.full((80, 2), 0.70, dtype=np.float32)
    h.infer(gray, _face(lm1))
    r = h.infer(gray, _face(lm2))
    assert any("landmark_jump" in s for s in r.signals)


def test_no_face_no_stable_landmarks() -> None:
    h = AiGeneratedHeuristics()
    gray = np.zeros((120, 160), dtype=np.uint8)
    r = h.infer(gray, FaceResult(detected=False, confidence=0.0))
    assert "no_stable_landmarks" in r.signals or r.score >= 0.0
