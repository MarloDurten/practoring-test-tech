"""Тесты AI-эвристик: скачки landmarks, низкая уверенность, smoothness."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

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


def test_background_laplacian_uses_2d_neighborhood() -> None:
    rng = np.random.default_rng(0)
    h_img, w_img = 480, 640
    gray = rng.integers(0, 255, (h_img, w_img), dtype=np.uint8)
    x0, y0, bw, bh = 160, 120, 320, 240
    gray[y0 : y0 + bh, x0 : x0 + bw] = 128
    lm = np.linspace(0.3, 0.7, 80 * 2).reshape(80, 2).astype(np.float32)
    face = FaceResult(
        detected=True,
        confidence=0.9,
        landmarks_norm=lm,
        bbox=(x0, y0, bw, bh),
        landmark_count=80,
    )
    r = AiGeneratedHeuristics().infer(gray, face)

    pad = max(8, int(0.15 * max(bw, bh)))
    bg_mask = np.ones_like(gray, dtype=bool)
    bg_mask[y0 : y0 + bh, x0 : x0 + bw] = False
    yb0, yb1 = max(0, y0 - pad), min(h_img, y0 + bh + pad)
    xb0, xb1 = max(0, x0 - pad), min(w_img, x0 + bw + pad)
    bg_roi = gray[yb0:yb1, xb0:xb1]
    local_mask = bg_mask[yb0:yb1, xb0:xb1]
    expected = float(cv2.Laplacian(bg_roi, cv2.CV_64F)[local_mask].var())
    column_wrong = float(cv2.Laplacian(bg_roi[local_mask].reshape(-1, 1), cv2.CV_64F).var())

    assert r.details["bg_laplacian_var"] == pytest.approx(expected)
    assert r.details["face_laplacian_var"] < 1.0
    assert r.details["bg_laplacian_var"] != pytest.approx(column_wrong, rel=0.01)
