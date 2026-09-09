"""Оценка взгляда по iris-landmarks MediaPipe + позе головы.

Это не калиброванный eye-tracker. Для нагрузочного теста достаточно относительного
yaw/pitch: «смотрит примерно в камеру» vs «отвернулся».

Py-Feat / LibreFace дают калиброванный gaze и AU, но тянут PyTorch и большие
веса — поэтому здесь landmark-эвристика, а тяжёлые библиотеки описаны в README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.analyzers.face_tracking import (
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    FaceResult,
)


@dataclass
class GazeResult:
    available: bool = False
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    iris_offset_x: Optional[float] = None
    iris_offset_y: Optional[float] = None
    method: str = "disabled"


class GazeEstimator:
    """Комбинирует смещение радужки в глазнице и ориентацию головы."""

    def __init__(
        self,
        enabled: bool = True,
        iris_yaw_gain: float = 35.0,
        iris_pitch_gain: float = 30.0,
        head_blend: float = 0.55,
    ) -> None:
        self.enabled = enabled
        self.iris_yaw_gain = iris_yaw_gain
        self.iris_pitch_gain = iris_pitch_gain
        self.head_blend = head_blend

    def infer(self, face: FaceResult) -> GazeResult:
        if not self.enabled:
            return GazeResult(method="disabled")
        if not face.detected or face.landmarks_norm is None:
            return GazeResult(method="no_face")

        offset_x, offset_y = self._iris_offsets(face)
        iris_yaw = iris_pitch = None
        if offset_x is not None and offset_y is not None:
            iris_yaw = float(offset_x * self.iris_yaw_gain)
            iris_pitch = float(offset_y * self.iris_pitch_gain)

        head_yaw = face.head_yaw
        head_pitch = face.head_pitch

        if iris_yaw is not None and head_yaw is not None:
            yaw = self.head_blend * head_yaw + (1.0 - self.head_blend) * iris_yaw
            pitch = self.head_blend * (head_pitch or 0.0) + (1.0 - self.head_blend) * (iris_pitch or 0.0)
            method = "iris+head"
        elif iris_yaw is not None:
            yaw, pitch, method = iris_yaw, iris_pitch, "iris"
        elif head_yaw is not None:
            yaw, pitch, method = head_yaw, head_pitch, "head"
        else:
            return GazeResult(method="insufficient_landmarks")

        return GazeResult(
            available=True,
            yaw_deg=float(yaw) if yaw is not None else None,
            pitch_deg=float(pitch) if pitch is not None else None,
            iris_offset_x=offset_x,
            iris_offset_y=offset_y,
            method=method,
        )

    def _iris_offsets(self, face: FaceResult) -> tuple[Optional[float], Optional[float]]:
        lm = face.landmarks_norm
        if lm is None or lm.shape[0] < 478:
            return None, None
        left = self._one_eye_offset(lm, LEFT_EYE_OUTER, LEFT_EYE_INNER, 159, 145, face.left_iris_norm)
        right = self._one_eye_offset(lm, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, 386, 374, face.right_iris_norm)
        offsets = [o for o in (left, right) if o is not None]
        if not offsets:
            return None, None
        xs = [o[0] for o in offsets]
        ys = [o[1] for o in offsets]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    @staticmethod
    def _one_eye_offset(
        lm: np.ndarray,
        left_idx: int,
        right_idx: int,
        top_idx: int,
        bottom_idx: int,
        iris: Optional[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        if iris is None:
            return None
        left, right = lm[left_idx], lm[right_idx]
        top, bottom = lm[top_idx], lm[bottom_idx]
        width = float(right[0] - left[0])
        height = float(bottom[1] - top[1])
        if abs(width) < 1e-5 or abs(height) < 1e-5:
            return None
        ox = (iris[0] - float(left[0])) / width - 0.5
        oy = (iris[1] - float(top[1])) / height - 0.5
        return float(np.clip(ox, -1.0, 1.0)), float(np.clip(oy, -1.0, 1.0))
