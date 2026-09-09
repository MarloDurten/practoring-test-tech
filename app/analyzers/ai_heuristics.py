"""Сигналы «подозрительный кадр» для возможной проверки на AI-generated content.

Это НЕ финальный детектор дипфейка. Набор дешёвых эвристик, которые на сервере
можно использовать как pre-filter перед тяжёлой моделью (CNN / forensic net):
- низкая уверенность трекинга;
- резкие скачки координат landmarks;
- отсутствие стабильных landmarks (jitter);
- странные артефакты на лице (слишком гладко / слишком шумно относительно фона).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.analyzers.face_tracking import FaceResult
from app.config import AiHeuristicThresholds


@dataclass
class AiHeuristicResult:
    score: float = 0.0
    suspicious: bool = False
    signals: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class AiGeneratedHeuristics:
    def __init__(
        self,
        thresholds: Optional[AiHeuristicThresholds] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.t = thresholds or AiHeuristicThresholds()
        self._prev_landmarks: Optional[np.ndarray] = None
        self._jitter_window: deque[float] = deque(maxlen=self.t.jitter_window)

    def reset(self) -> None:
        self._prev_landmarks = None
        self._jitter_window.clear()

    def infer(self, gray: np.ndarray, face: FaceResult) -> AiHeuristicResult:
        if not self.enabled:
            return AiHeuristicResult()

        signals: list[str] = []
        weights: list[float] = []
        details: dict = {}

        if not face.detected or face.landmarks_norm is None:
            # Нет стабильного лица — слабый сигнал (может быть просто пустой кадр).
            self._prev_landmarks = None
            self._jitter_window.append(0.0)
            if face.confidence < self.t.low_confidence:
                signals.append("no_stable_landmarks")
                weights.append(0.25)
            return self._pack(signals, weights, details)

        if face.confidence < self.t.low_confidence:
            signals.append(f"low_tracking_confidence:{face.confidence:.3f}")
            weights.append(0.35)
        details["confidence"] = face.confidence

        lm = face.landmarks_norm[:, :2]
        jump = 0.0
        if self._prev_landmarks is not None and self._prev_landmarks.shape == lm.shape:
            delta = np.linalg.norm(lm - self._prev_landmarks, axis=1)
            jump = float(np.percentile(delta, 90))
            details["landmark_jump_p90"] = jump
            if jump > self.t.landmark_jump:
                signals.append(f"landmark_jump:{jump:.3f}")
                weights.append(min(1.0, jump / max(self.t.landmark_jump, 1e-6)) * 0.45)
            self._jitter_window.append(jump)
        else:
            self._jitter_window.append(0.0)
        self._prev_landmarks = lm.copy()

        if len(self._jitter_window) >= max(3, self.t.jitter_window // 2):
            jitter_std = float(np.std(self._jitter_window))
            details["landmark_jitter_std"] = jitter_std
            if jitter_std > self.t.landmark_jitter_std:
                signals.append(f"unstable_landmarks:{jitter_std:.3f}")
                weights.append(0.30)

        # Лицо должно быть «живым»: естественная текстура кожи, не пластик и не сетка артефактов.
        artifact_score = self._face_artifact_signals(gray, face, signals, weights, details)
        details["artifact_score"] = artifact_score

        return self._pack(signals, weights, details)

    def _face_artifact_signals(
        self,
        gray: np.ndarray,
        face: FaceResult,
        signals: list[str],
        weights: list[float],
        details: dict,
    ) -> float:
        if face.bbox is None:
            return 0.0
        x, y, w, h = face.bbox
        h_img, w_img = gray.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w_img, x + w)
        y1 = min(h_img, y + h)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return 0.0

        face_roi = gray[y0:y1, x0:x1]
        face_lap = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())
        details["face_laplacian_var"] = face_lap

        # Фон: рамка вокруг лица, если влезает.
        pad = max(8, int(0.15 * max(w, h)))
        bg_mask = np.ones_like(gray, dtype=bool)
        bg_mask[y0:y1, x0:x1] = False
        # Берём только окрестность лица, чтобы не сравнивать с чёрным letterbox.
        yb0, yb1 = max(0, y0 - pad), min(h_img, y1 + pad)
        xb0, xb1 = max(0, x0 - pad), min(w_img, x1 + pad)
        bg_roi = gray[yb0:yb1, xb0:xb1]
        local_mask = bg_mask[yb0:yb1, xb0:xb1]
        if bg_roi.size and local_mask.any():
            bg_pixels = bg_roi[local_mask]
            if bg_pixels.size > 64:
                bg_lap = float(cv2.Laplacian(bg_pixels.reshape(-1, 1), cv2.CV_64F).var())
            else:
                bg_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        else:
            bg_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        details["bg_laplacian_var"] = bg_lap

        score = 0.0
        if face_lap < self.t.face_too_smooth:
            signals.append(f"face_too_smooth:{face_lap:.1f}")
            weights.append(0.40)
            score += 0.4

        if bg_lap > 1e-3:
            ratio = face_lap / bg_lap
            details["face_bg_lap_ratio"] = ratio
            if ratio > self.t.face_artifact_ratio:
                signals.append(f"face_artifacts:{ratio:.2f}")
                weights.append(0.35)
                score += 0.35

        # Высокочастотный шум (шахматный паттерн части GAN-upscalers).
        fshift = np.fft.fftshift(np.fft.fft2(face_roi.astype(np.float32)))
        mag = np.abs(fshift)
        h2, w2 = mag.shape
        cy, cx = h2 // 2, w2 // 2
        ch, cw = max(1, h2 // 8), max(1, w2 // 8)
        center_energy = float(mag[cy - ch : cy + ch, cx - cw : cx + cw].sum())
        total_energy = float(mag.sum()) + 1e-6
        hf_share = (total_energy - center_energy) / total_energy
        details["hf_energy_share"] = hf_share
        if hf_share > 0.92:
            signals.append(f"high_frequency_artifacts:{hf_share:.3f}")
            weights.append(0.20)
            score += 0.2
        return score

    def _pack(self, signals: list[str], weights: list[float], details: dict) -> AiHeuristicResult:
        raw = float(min(1.0, sum(weights))) if weights else 0.0
        # Несколько слабых сигналов вместе важнее одного.
        if len(signals) >= 3:
            raw = min(1.0, raw + 0.1)
        return AiHeuristicResult(
            score=raw,
            suspicious=raw >= self.t.suspicion_score,
            signals=signals,
            details=details,
        )
